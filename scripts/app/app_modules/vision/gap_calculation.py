# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: f1_strat_manager
#     language: python
#     name: python3
# ---

# # Computer Vision: Gap Calculation using YOLO net and OpenCV

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import Counter
import time as pytime
import pandas as pd
import sys
import os
from datetime import datetime

# --- RUTAS ABSOLUTAS ---


def get_abs_path(relative_path):
    """Devuelve la ruta absoluta a partir de una ruta relativa al proyecto."""
    base_dir = os.getcwd()
    return os.path.abspath(os.path.join(base_dir, relative_path))


# Use GPU if available
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load the Anti-Alpine optimized model
MODEL_PATH = get_abs_path("f1-strategy/weights/model_anti_alpine.pt")
model = YOLO(MODEL_PATH)
model.to(DEVICE)

# ---

# ## 1. Gap Calculation and Yolo video processing

# ### 1.1 Dedining constants and variables

# Size and scale configuration
FRAME_WIDTH = 1280  # Higher resolution for better detail
CAR_LENGTH_METERS = 5.63  # Real car length in meters

# FOR GAPS: use a lower global threshold to maximize detections
GAP_DETECTION_THRESHOLD = 0.25  # Low threshold to detect all possible cars

# Specific colors for each F1 team (BGR format for OpenCV)
class_colors = {
    'Ferrari': (0, 0, 255),         # Red (BGR)
    'Mercedes': (200, 200, 200),    # Silver (BGR)
    'Red Bull': (139, 0, 0),        # Dark Blue (BGR)
    'McLaren': (0, 165, 255),       # Orange (BGR)
    'Aston Martin': (0, 128, 0),    # Green (BGR)
    'Alpine': (128, 0, 0),          # Blue (BGR)
    'Williams': (205, 0, 0),        # Light Blue (BGR)
    'Haas': (255, 255, 255),        # White (BGR)
    'Kick Sauber': (255, 255, 0),   # Cyan (BGR)
    'Racing Bulls': (0, 0, 255),    # Red (BGR)
    'background': (128, 128, 128),  # Gray (BGR)
    # Yellow for cars without secure classification
    'unknown': (0, 255, 255)
}

# Thresholds to show classification (not for filtering detections)
class_thresholds = {
    'Williams': 0.90,     # Very high for Williams
    'Alpine': 0.90,       # Extremely high for Alpine
    'McLaren': 0.30,      # Low for McLaren
    'Red Bull': 0.85,     # High for Red Bull
    'Ferrari': 0.40,      # Normal
    'Mercedes': 0.50,     # Medium-high
    'Haas': 0.40,         # Normal
    'Kick Sauber': 0.40,  # Normal
    'Racing Bulls': 0.40,  # Normal
    'Aston Martin': 0.40,  # Normal
    'background': 0.50    # High for background
}

# Detection history for stabilization
last_detections = {}
track_history = {}
id_counter = 0
class_history = {}


# ### 1.2 Calculating the gap

def calculate_gap(box1, box2, class1, class2):
    """Calculates the distance between centers using car width for scale"""
    # Box centers
    cx1, cy1 = (box1[0] + box1[2])/2, (box1[1] + box1[3])/2
    cx2, cy2 = (box2[0] + box2[2])/2, (box2[1] + box2[3])/2

    # Distance in pixels
    pixel_distance = np.hypot(cx2 - cx1, cy2 - cy1)

    # Scale based on average width of detected cars
    avg_width = ((box1[2] - box1[0]) + (box2[2] - box2[0])) / 2
    scale = CAR_LENGTH_METERS / avg_width if avg_width != 0 else 0

    # Calculate gap time at 300km/h (83.33 m/s)
    speed_mps = 83.33  # Meters per second at 300km/h
    gap_time = (pixel_distance * scale) / speed_mps

    return {
        'distance': pixel_distance * scale,  # Distance in meters
        'time': gap_time,                   # Time in seconds at 300km/h
        'car1': class1,                     # Team of first car
        'car2': class2                      # Team of second car
    }


# ### 1.3 Processing the video with YOLO

def process_video_with_yolo(video_path, output_path=None):
    global last_detections, track_history, id_counter, class_history, GAP_DETECTION_THRESHOLD

    video_path = get_abs_path(video_path)
    if output_path is not None:
        output_path = get_abs_path(output_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Get original video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # Calculate new height maintaining aspect ratio
    target_height = int(FRAME_WIDTH * original_height / original_width)

    out = None
    if output_path:
        # Change codec from 'mp4v' to 'XVID' which is more reliable
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (FRAME_WIDTH, target_height))
        if not out.isOpened():
            print(
                f"Error: Could not create output video file at {output_path}")
            print("Continuing without saving output...")
            output_path = None

    # Variables for calculating real FPS
    frame_count = 0
    start_time = pytime.time()
    current_fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_frame += 1

        # Resize maintaining aspect ratio
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, target_height))
        original_frame = frame_resized.copy()

        # Run YOLOv8 detection with low threshold to maximize detections
        results = model.predict(
            source=frame_resized,
            conf=GAP_DETECTION_THRESHOLD,  # Low threshold to detect all possible cars
            iou=0.45,   # Standard IoU
            max_det=20,  # Maximum detections
            verbose=False
        )[0]

        # Current detections
        current_detections = {}

        # Process detection results
        if results.boxes and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            # Create detection list with all information
            detections = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = float(confs[i])
                class_id = int(class_ids[i])
                cls_name = model.names[class_id]

                # Determine whether to show team classification based on threshold
                # Note: we still detect the car but might not show its team
                classified = conf >= class_thresholds.get(cls_name, 0.40)

                # KEY: For gaps, we detect all cars even if we're not sure of the team
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)

                # Assign unique ID or retrieve existing ID
                object_id = None
                for old_id, old_info in last_detections.items():
                    old_cx, old_cy = old_info['center']
                    old_cls = old_info['class']

                    # Distance between centers
                    dist = np.sqrt((center_x - old_cx)**2 +
                                   (center_y - old_cy)**2)

                    # If it's close, it could be the same object
                    if dist < 100:
                        object_id = old_id

                        # If previous class was valid and new one is uncertain, keep the previous one
                        if old_info['classified'] and not classified:
                            cls_name = old_cls
                            classified = True

                        # Stabilize classification with history for problematic classes
                        if classified and old_cls != cls_name and (cls_name == 'Williams' or cls_name == 'Alpine'):
                            if old_info['classified']:
                                cls_name = old_cls
                        break

                # If no match found, assign new ID
                if object_id is None:
                    object_id = id_counter
                    id_counter += 1
                    track_history[object_id] = []
                    class_history[object_id] = []

                # Update history
                if object_id in class_history:
                    # Only add to history if we're sure of the class
                    if classified:
                        class_history[object_id].append(cls_name)
                        # Limit history to 5 classes
                        if len(class_history[object_id]) > 5:
                            class_history[object_id].pop(0)

                    # Use the most common class from history for stability
                    if len(class_history[object_id]) >= 3:
                        counts = Counter(class_history[object_id])
                        if counts:  # Make sure it's not empty
                            most_common = counts.most_common(1)[0][0]
                            cls_name = most_common
                            classified = True

                # Save current detection
                current_detections[object_id] = {
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'class': cls_name,
                    'classified': classified,
                    'center': (center_x, center_y),
                    'area': area,
                    'y_bottom': y2  # For sorting by vertical position
                }

                # Add to detection list for gap calculation
                detections.append({
                    'id': object_id,
                    'box': (x1, y1, x2, y2),
                    'class': cls_name,
                    'classified': classified,
                    'conf': conf,
                    'y_bottom': y2
                })

            # Sort by vertical position (cars more below first - closer)
            detections = sorted(
                detections, key=lambda x: x['y_bottom'], reverse=True)

            # Draw boxes and gaps
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['box']
                cls_name = det['class']
                conf = det['conf']
                classified = det['classified']

                # Get specific color for the team
                if classified:
                    color = class_colors.get(cls_name, (0, 255, 0))
                else:
                    # Yellow for cars without secure classification
                    color = class_colors['unknown']

                # Draw box with team color
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)

                # Label with class and confidence
                if classified:
                    label = f"{cls_name}: {conf:.2f}"
                else:
                    label = f"F1 Car: {conf:.2f}"

                t_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    frame_resized, (x1, y1-t_size[1]-3), (x1+t_size[0], y1), color, -1)
                cv2.putText(frame_resized, label, (x1, y1-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Only if there's a next car
                if i < len(detections)-1:
                    next_det = detections[i+1]
                    gap_info = calculate_gap(
                        det['box'], next_det['box'],
                        det['class'] if det['classified'] else "F1 Car",
                        next_det['class'] if next_det['classified'] else "F1 Car"
                    )

                    # Connection points
                    cx1, cy1 = int((x1+x2)/2), int(y1)  # Use top of the car
                    nx1, ny1, nx2, ny2 = next_det['box']
                    # Use bottom of the next car
                    cx2, cy2 = int((nx1+nx2)/2), int(ny2)

                    # Diagonal line between cars
                    cv2.line(frame_resized, (cx1, cy1),
                             (cx2, cy2), (0, 255, 0), 2)

                    # Text at midpoint with more information
                    mid_x, mid_y = (cx1+cx2)//2, (cy1+cy2)//2

                    # Distance and gap time
                    dist_text = f"{gap_info['distance']:.1f}m"
                    time_text = f"{gap_info['time']:.2f}s"

                    # Background for text
                    dist_size = cv2.getTextSize(
                        dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    time_size = cv2.getTextSize(
                        time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                    # Draw semi-transparent background
                    overlay = frame_resized.copy()
                    cv2.rectangle(overlay,
                                  (mid_x - 5, mid_y - 50),
                                  (mid_x +
                                   max(dist_size[0], time_size[0]) + 10, mid_y + 10),
                                  (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame_resized,
                                    0.4, 0, frame_resized)

                    # Draw texts
                    cv2.putText(frame_resized, dist_text, (mid_x, mid_y - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame_resized, time_text, (mid_x, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Update last_detections for next iteration
        last_detections = current_detections

        # Calculate FPS
        if frame_count % 10 == 0:
            current_time = pytime.time()
            current_fps = 10.0 / (current_time - start_time)
            start_time = current_time

        # Show FPS and model information
        cv2.putText(frame_resized, f"FPS: {current_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame_resized, "F1 Gap Detection", (FRAME_WIDTH - 300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show detection mode and progress
        detection_mode = f"Detection Threshold: {GAP_DETECTION_THRESHOLD:.2f}"
        cv2.putText(frame_resized, detection_mode, (20, target_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show video progress
        progress_text = f"Frame: {current_frame}/{total_frames} ({current_frame/total_frames*100:.1f}%)"
        cv2.putText(frame_resized, progress_text, (FRAME_WIDTH - 400, target_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save processed frame if requested
        if output_path:
            out.write(frame_resized)

        # Show frame
        cv2.imshow("F1 Gap Detection", frame_resized)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('+'):  # Increase threshold
            GAP_DETECTION_THRESHOLD = min(GAP_DETECTION_THRESHOLD + 0.05, 0.95)
            print(
                f"Detection threshold increased to {GAP_DETECTION_THRESHOLD:.2f}")
        elif key == ord('-'):  # Decrease threshold
            GAP_DETECTION_THRESHOLD = max(GAP_DETECTION_THRESHOLD - 0.05, 0.05)
            print(
                f"Detection threshold decreased to {GAP_DETECTION_THRESHOLD:.2f}")
        elif key == ord('d'):  # Skip forward 10 seconds
            # 10 seconds * fps = number of frames to skip
            skip_frames = int(fps * 10)
            new_frame_pos = min(current_frame + skip_frames, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_pos)
            current_frame = new_frame_pos - 1  # Will be incremented in the next cycle
            # Temporarily reset tracking
            last_detections = {}
            print(f"Skipped forward 10 seconds to frame {new_frame_pos}")

    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


# ### 1.4 Running a demo

# Run with your video
def main():
    # video_path = "../f1-strategy/data/videos/best_overtakes_2023.mp4.f399.mp4"
    # video_path = "../f1-strategy/data/videos/spain_2023_race.mp4.f399.mp4"

    video_path = "f1-strategy/data/videos/belgium_gp.f399.mp4"
    output_path = "f1-strategy/data/videos/gap_detection_output.mp4"
    process_video_with_yolo(video_path, output_path)


if __name__ == "__main__":
    main()


# Controls
#
# 'q': out
# '+': more detection threshold
# '-': less detection threshold
# 'd': 10 seconds ahead

# ---

# ## 2. Gap Extraction

def extract_gaps_from_video(
    video_path,
    sample_interval_seconds=10,
    output_csv=None,
    show_video=True,
    streamlit_callback=None,
    start_time=0  # Nuevo parámetro para salto en segundos
):
    """
    Process a video and extract gap data at regular intervals with visualization

    Args:
        video_path: Path to the F1 video
        sample_interval_seconds: How often to sample gap data (in seconds)
        output_csv: Path to save CSV data (if None, will generate a default path)
        show_video: Whether to display the video during processing
        streamlit_callback: función para actualizar progreso/log/frame en Streamlit
        start_time: segundos desde los que empezar el procesamiento

    Returns:
        DataFrame with extracted gap data
    """
    global last_detections, track_history, id_counter, class_history, GAP_DETECTION_THRESHOLD

    all_gaps = []

    video_path = get_abs_path(video_path)
    if output_csv is not None:
        output_csv = get_abs_path(output_csv)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"Error opening video: {video_path}"
        if streamlit_callback:
            streamlit_callback(0, msg, None)
        else:
            print(msg)
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * sample_interval_seconds)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_height = int(FRAME_WIDTH * original_height / original_width)

    # Salto inicial si se especifica
    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

    frame_count = 0
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    last_sample_frame = current_frame - frame_interval
    start_time_clock = pytime.time()
    current_fps = 0

    last_detections = {}
    track_history = {}
    id_counter = 0
    class_history = {}

    if streamlit_callback:
        streamlit_callback(
            0, f"Starting gap extraction from {os.path.basename(video_path)}...", None)
    else:
        print(f"Starting gap extraction from {video_path}...")
        print(
            f"Will sample approximately every {sample_interval_seconds} seconds")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_frame += 1
        timestamp = current_frame / fps
        is_sample_frame = (current_frame - last_sample_frame >= frame_interval)

        # Progreso y logs para Streamlit SIEMPRE (aunque no show_video)
        if streamlit_callback:
            progress = min(current_frame / total_frames, 1.0)
            log = f"Progress: Frame {current_frame}/{total_frames} ({current_frame/total_frames*100:.1f}%)"
            frame_to_show = None
            # Solo muestra el frame procesado si show_video y es sample frame
            if show_video and is_sample_frame:
                frame_to_show = frame.copy()
            streamlit_callback(progress, log, frame_to_show,
                               partial_gaps=all_gaps)
        else:
            if current_frame % 100 == 0:
                print(
                    f"Progress: Frame {current_frame}/{total_frames} ({current_frame/total_frames*100:.1f}%)")

        if not show_video and not is_sample_frame:
            continue

        if is_sample_frame:
            last_sample_frame = current_frame
            msg = f"Taking sample at frame {current_frame} (timestamp: {timestamp:.2f}s)"
            if streamlit_callback:
                streamlit_callback(progress, msg, frame_to_show)
            else:
                print(msg)

        frame_resized = cv2.resize(frame, (FRAME_WIDTH, target_height))
        original_frame = frame_resized.copy()

        results = model.predict(
            source=frame_resized,
            conf=GAP_DETECTION_THRESHOLD,
            iou=0.45,
            max_det=20,
            verbose=False
        )[0]

        current_detections = {}

        if results.boxes and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            detections = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = float(confs[i])
                class_id = int(class_ids[i])
                cls_name = model.names[class_id]

                classified = conf >= class_thresholds.get(cls_name, 0.40)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)

                object_id = None
                for old_id, old_info in last_detections.items():
                    old_cx, old_cy = old_info['center']
                    old_cls = old_info['class']

                    dist = np.sqrt((center_x - old_cx)**2 +
                                   (center_y - old_cy)**2)

                    if dist < 100:
                        object_id = old_id

                        if old_info['classified'] and not classified:
                            cls_name = old_cls
                            classified = True

                        if classified and old_cls != cls_name and (cls_name == 'Williams' or cls_name == 'Alpine'):
                            if old_info['classified']:
                                cls_name = old_cls
                        break

                if object_id is None:
                    object_id = id_counter
                    id_counter += 1
                    track_history[object_id] = []
                    class_history[object_id] = []

                if object_id in class_history:
                    if classified:
                        class_history[object_id].append(cls_name)
                        if len(class_history[object_id]) > 5:
                            class_history[object_id].pop(0)

                    if len(class_history[object_id]) >= 3:
                        counts = Counter(class_history[object_id])
                        if counts:
                            most_common = counts.most_common(1)[0][0]
                            cls_name = most_common
                            classified = True

                current_detections[object_id] = {
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'class': cls_name,
                    'classified': classified,
                    'center': (center_x, center_y),
                    'area': area,
                    'y_bottom': y2
                }

                detections.append({
                    'id': object_id,
                    'box': (x1, y1, x2, y2),
                    'class': cls_name,
                    'classified': classified,
                    'conf': conf,
                    'y_bottom': y2
                })

            detections = sorted(
                detections, key=lambda x: x['y_bottom'], reverse=True)

            frame_gaps = []

            if show_video:
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det['box']
                    cls_name = det['class']
                    conf = det['conf']
                    classified = det['classified']

                    if classified:
                        color = class_colors.get(cls_name, (0, 255, 0))
                    else:
                        color = class_colors['unknown']

                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)

                    if classified:
                        label = f"{cls_name}: {conf:.2f}"
                    else:
                        label = f"F1 Car: {conf:.2f}"

                    t_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(
                        frame_resized, (x1, y1-t_size[1]-3), (x1+t_size[0], y1), color, -1)
                    cv2.putText(frame_resized, label, (x1, y1-3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if i < len(detections)-1:
                        next_det = detections[i+1]
                        gap_info = calculate_gap(
                            det['box'], next_det['box'],
                            det['class'] if det['classified'] else "F1 Car",
                            next_det['class'] if next_det['classified'] else "F1 Car"
                        )

                        if is_sample_frame:
                            frame_gaps.append({
                                'frame': current_frame,
                                'timestamp': timestamp,
                                'car1_id': det['id'],
                                'car2_id': next_det['id'],
                                'car1_team': gap_info['car1'],
                                'car2_team': gap_info['car2'],
                                'distance_meters': gap_info['distance'],
                                'gap_seconds': gap_info['time']
                            })

                        cx1, cy1 = int((x1+x2)/2), int(y1)
                        nx1, ny1, nx2, ny2 = next_det['box']
                        cx2, cy2 = int((nx1+nx2)/2), int(ny2)

                        line_color = (
                            0, 0, 255) if is_sample_frame else (0, 255, 0)
                        line_thickness = 3 if is_sample_frame else 2
                        cv2.line(frame_resized, (cx1, cy1),
                                 (cx2, cy2), line_color, line_thickness)

                        mid_x, mid_y = (cx1+cx2)//2, (cy1+cy2)//2

                        dist_text = f"{gap_info['distance']:.1f}m"
                        time_text = f"{gap_info['time']:.2f}s"

                        dist_size = cv2.getTextSize(
                            dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        time_size = cv2.getTextSize(
                            time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                        overlay = frame_resized.copy()
                        bg_color = (
                            0, 0, 180) if is_sample_frame else (0, 0, 0)
                        cv2.rectangle(overlay,
                                      (mid_x - 5, mid_y - 50),
                                      (mid_x +
                                          max(dist_size[0], time_size[0]) + 10, mid_y + 10),
                                      bg_color, -1)
                        cv2.addWeighted(
                            overlay, 0.6, frame_resized, 0.4, 0, frame_resized)

                        if is_sample_frame:
                            saved_text = "SAVED"
                            cv2.putText(frame_resized, saved_text, (mid_x, mid_y - 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        cv2.putText(frame_resized, dist_text, (mid_x, mid_y - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame_resized, time_text, (mid_x, mid_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            elif is_sample_frame:
                for i, det in enumerate(detections):
                    if i < len(detections)-1:
                        next_det = detections[i+1]
                        gap_info = calculate_gap(
                            det['box'], next_det['box'],
                            det['class'] if det['classified'] else "F1 Car",
                            next_det['class'] if next_det['classified'] else "F1 Car"
                        )

                        frame_gaps.append({
                            'frame': current_frame,
                            'timestamp': round(timestamp, 2),
                            'car1_id': det['id'],
                            'car2_id': next_det['id'],
                            'car1_team': gap_info['car1'],
                            'car2_team': gap_info['car2'],
                            'distance_meters': round(gap_info['distance'], 2),
                            'gap_seconds': round(gap_info['time'], 2)
                        })

            if is_sample_frame:
                all_gaps.extend(frame_gaps)
                if not streamlit_callback:
                    print(
                        f"Found {len(frame_gaps)} car gaps at timestamp {timestamp:.2f}s")

        last_detections = current_detections

        if show_video and frame_count % 10 == 0:
            current_time = pytime.time()
            current_fps = 10.0 / (current_time - start_time_clock)
            start_time_clock = current_time

        if show_video and streamlit_callback:
            # Enviar el frame procesado a Streamlit en cada iteración
            streamlit_callback(progress, log, frame_resized)

    cap.release()
    cv2.destroyAllWindows()

    if all_gaps:
        gaps_df = pd.DataFrame(all_gaps)

        if output_csv is None:
            video_name = os.path.basename(video_path).split('.')[0]
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = get_abs_path(
                f"f1-strategy/data/gaps/gap_data_{video_name}_{timestamp_str}.csv")

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        gaps_df.to_csv(output_csv, index=False, float_format='%.3f')
        msg = f"Gap data saved to {output_csv}\nTotal of {len(gaps_df)} gap measurements collected"
        if streamlit_callback:
            streamlit_callback(1.0, msg, None)
        else:
            print(f"Gap data saved to {output_csv}")
            print(f"Total of {len(gaps_df)} gap measurements collected")

        return gaps_df
    else:
        msg = "No gap data could be collected!"
        if streamlit_callback:
            streamlit_callback(1.0, msg, None)
        else:
            print("No gap data could be collected!")
        return None
