import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "model_out" / "best_age_model.keras"
IMAGE_SIZE = 200
TARGET_MIN_AGE = 0.0
TARGET_MAX_AGE = 60.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_to_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)


def collect_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    return sorted(
        child
        for child in path.rglob("*")
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )


def predict_images(model, image_paths: list[Path], batch_size: int) -> list[tuple[Path, float]]:
    results = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = np.stack([image_to_array(path) for path in batch_paths])
        predictions = model.predict(batch, verbose=0).reshape(-1)
        for path, prediction in zip(batch_paths, predictions):
            age = min(TARGET_MAX_AGE, max(TARGET_MIN_AGE, float(prediction)))
            results.append((path, age))
    return results


def write_csv(path: Path, results: list[tuple[Path, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image", "predicted_age"])
        for image_path, age in results:
            writer.writerow([str(image_path), round(age, 2)])


def parse_args():
    parser = argparse.ArgumentParser(description="Predict age for images with the trained Keras model.")
    parser.add_argument("image_path", type=Path, help="Image file or folder of images to predict.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")

    return args


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    image_paths = collect_image_paths(args.image_path)
    if not image_paths:
        raise ValueError(f"No supported images found at: {args.image_path}")

    model = tf.keras.models.load_model(args.model)
    results = predict_images(model, image_paths, args.batch_size)

    for image_path, age in results:
        print(f"{image_path}: {age:.1f} years")

    if args.csv is not None:
        write_csv(args.csv, results)
        print(f"Wrote predictions to {args.csv}")


if __name__ == "__main__":
    main()
