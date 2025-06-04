"""

Data augmentation file for the train training set of the image dataset.

"""


import os
import random
import numpy as np
from PIL import Image
import albumentations as A

# Class and parameter configuration
CLASS_NAMES = [
    "Kick Sauber", "Racing Bulls", "Alpine", "Aston Martin",
    "Ferarri", "Haas", "Mclaren", "Mercedes", "Red Bull", "Williams"
]
TARGET_PER_CLASS = 250
BASE_DIR = r"C:\Users\victo\Desktop\Documents\Tercer año\Segundo Cuatrimestre\Finales\f1-strategy\f1-dataset"

# Transformaciones con Albumentations
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=15,
            p=0.5
        ),
        A.GaussianBlur(p=0.2),
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    )
)


def get_dominant_class_id(label_path):
    """Obtiene la clase dominante de las etiquetas YOLO"""
    if not os.path.exists(label_path):
        return None

    class_ids = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(CLASS_NAMES):
                        class_ids.append(class_id)
                except ValueError:
                    continue

    return max(set(class_ids), key=class_ids.count) if class_ids else None


def read_yolo_labels(label_path):
    """Lee archivos de etiquetas YOLO"""
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    bboxes.append(list(map(float, parts[1:5])))
    return bboxes


def write_yolo_labels(label_path, bboxes, class_id):
    """Escribe etiquetas en formato YOLO"""
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            f.write(f"{class_id} {' '.join(map(str, bbox))}\n")


def augment_image(img_path, label_path, output_img_path, output_label_path, class_id):
    """Realiza el aumento de datos usando Pillow"""
    try:
        # Cargar imagen con Pillow
        with Image.open(img_path) as img:
            image_np = np.array(img.convert('RGB'))
            height, width = image_np.shape[:2]

            # Leer bounding boxes
            bboxes = read_yolo_labels(label_path)
            class_labels = [class_id] * len(bboxes)

            # Aplicar transformaciones
            transformed = transform(
                image=image_np,
                bboxes=bboxes,
                class_labels=class_labels
            )

            # Guardar imagen aumentada
            aug_img = Image.fromarray(transformed['image'])
            aug_img.save(output_img_path, quality=95)

            # Guardar etiquetas
            if transformed['bboxes']:
                write_yolo_labels(output_label_path,
                                  transformed['bboxes'], class_id)
            else:
                open(output_label_path, 'w').close()

    except Exception as e:
        print(f"Error procesando {img_path}: {str(e)}")


def main():    # Configure directories
    train_images_dir = os.path.join(BASE_DIR, "train", "images")
    train_labels_dir = os.path.join(BASE_DIR, "train", "labels")

    # Verify directories
    if not os.path.exists(train_images_dir):
        raise FileNotFoundError(
            f"Directory not found: {train_images_dir}")
    if not os.path.exists(train_labels_dir):
        raise FileNotFoundError(
            f"Directory not found: {train_labels_dir}")

    # Organize images by class
    class_files = {i: [] for i in range(len(CLASS_NAMES))}

    for filename in os.listdir(train_images_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(train_images_dir, filename)
        label_path = os.path.join(
            train_labels_dir, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        class_id = get_dominant_class_id(label_path)
        if class_id is not None and class_id in class_files:
            class_files[class_id].append((img_path, label_path))

    # Balancear clases
    for class_id, files in class_files.items():
        class_name = CLASS_NAMES[class_id]
        current_count = len(files)
        if current_count >= TARGET_PER_CLASS:
            print(f"{class_name}: Sufficient images ({current_count})")
            continue

        print(
            f"{class_name}: Generating {TARGET_PER_CLASS - current_count} augmentations...")
        for i in range(TARGET_PER_CLASS - current_count):
            src_img, src_label = random.choice(files)
            base_name = os.path.splitext(os.path.basename(src_img))[0]

            output_img = os.path.join(
                train_images_dir, f"{base_name}_aug_{i}.jpg")
            output_label = os.path.join(
                train_labels_dir, f"{base_name}_aug_{i}.txt")

            augment_image(src_img, src_label, output_img,
                          output_label, class_id)


if __name__ == "__main__":
    main()
