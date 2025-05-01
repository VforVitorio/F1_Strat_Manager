import streamlit as st
import tempfile
import os
from app_modules.vision import gap_calculation


def render_vision_view():
    st.header("Gap Analysis with Computer Vision (YOLO)")

    uploaded_file = st.file_uploader(
        "Upload an F1 video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Layout: sliders on the left, processed frame on the right
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Extraction Parameters")
            interval = st.slider("Sampling interval (s)",
                                 1, 60, 10, key="interval_slider")
            threshold = st.slider(
                "Detection threshold", 0.05, 0.95, 0.25, 0.01, key="threshold_slider")
        with col2:
            frame_placeholder = st.empty()
            frame_placeholder.info("Video processing will appear here...")

        # Buttons in one row
        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            if "skip_seconds" not in st.session_state:
                st.session_state["skip_seconds"] = 0
            if st.button("Skip 30 seconds"):
                st.session_state["skip_seconds"] += 30
        with col_btn2:
            process_clicked = st.button(
                "Process and extract gaps", use_container_width=True)

        # Checkbox below the buttons
        show_video = st.checkbox(
            "Show video processing", value=True, key="show_video_checkbox")

        # Placeholders for feedback
        progress_bar = st.progress(0)
        log_placeholder = st.empty()
        table_placeholder = st.empty()

        if process_clicked:
            def streamlit_callback(progress, log, frame=None, partial_gaps=None):
                progress_bar.progress(progress)
                log_placeholder.text(log)
                if frame is not None:
                    import cv2
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        frame_rgb, caption="Processed Frame", use_container_width=True)
                # Display partial table if received
                if partial_gaps is not None and len(partial_gaps) > 0:
                    import pandas as pd
                    df_partial = pd.DataFrame(partial_gaps)
                    table_placeholder.dataframe(
                        df_partial.tail(20))

            with st.spinner("Processing video..."):
                gap_calculation.GAP_DETECTION_THRESHOLD = threshold
                df = gap_calculation.extract_gaps_from_video(
                    video_path=video_path,
                    sample_interval_seconds=interval,
                    output_csv=None,
                    show_video=show_video,
                    streamlit_callback=streamlit_callback,
                    start_time=st.session_state["skip_seconds"]
                )
            if df is not None:
                st.success("Extraction completed!")
                st.dataframe(df.head())
                csv_path = None
                if hasattr(df, "to_csv"):
                    csv_temp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".csv")
                    df.to_csv(csv_temp.name, index=False)
                    csv_path = csv_temp.name
                if csv_path and os.path.exists(csv_path):
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            label="Download gaps CSV",
                            data=f,
                            file_name="gaps_data.csv",
                            mime="text/csv"
                        )
            else:
                st.error("No gaps were extracted from the video.")
