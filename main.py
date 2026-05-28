import argparse
import csv
import io
import json
import math
import platform
import random
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "age" / "data"
OUTPUT_DIR = PROJECT_DIR / "model_out"
DEFAULT_DATASET = "typorch/age"
IMAGE_SIZE = 200
AUTOTUNE = None
TARGET_MIN_AGE = 0.0
TARGET_MAX_AGE = 60.0


def import_runtime():
    try:
        import numpy as np
        import tensorflow as tf
        from datasets import Image as DatasetImage
        from datasets import load_dataset
        from PIL import Image, ImageFile, UnidentifiedImageError
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependencies.\n\n"
            "This project needs TensorFlow/Keras, Hugging Face datasets, Pillow, "
            "and NumPy:\n"
            "  py -3.11 -m pip install tensorflow datasets pillow numpy\n\n"
            "For NVIDIA GPU training on Windows, run this from WSL2/Linux with a "
            "GPU-enabled TensorFlow install. Native Windows TensorFlow 2.11+ does "
            "not use CUDA GPUs."
        ) from exc

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    global AUTOTUNE
    AUTOTUNE = tf.data.AUTOTUNE
    return np, tf, keras, load_dataset, DatasetImage, Image, UnidentifiedImageError


def load_age_dataset(load_dataset, dataset_name: str, data_dir: Path | None, split: str):
    if data_dir is not None:
        files = sorted(data_dir.glob("train-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet shards found in {data_dir}")
        return load_dataset("parquet", data_files={split: [str(path) for path in files]}, split=split)

    return load_dataset(dataset_name, split=split)


def filter_target_age_range(dataset, min_age: float, max_age: float):
    def is_in_target_range(row):
        age = float(row["age"])
        return min_age <= age < max_age

    return dataset.filter(is_in_target_range)


def age_bin(age: float, bin_width: int) -> int:
    return int(float(age) // bin_width) * bin_width


def prepare_splits(dataset, dataset_image, val_fraction: float, test_fraction: float, seed: int, bin_width: int):
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    if not 0.0 <= test_fraction < 0.5:
        raise ValueError("--test-fraction must be between 0 and 0.5")
    if val_fraction + test_fraction >= 0.8:
        raise ValueError("--val-fraction plus --test-fraction must be less than 0.8")

    if "image" in dataset.features:
        dataset = dataset.cast_column("image", dataset_image(decode=False))

    rng = random.Random(seed)
    indices_by_bin = {}
    for index, row in enumerate(dataset):
        indices_by_bin.setdefault(age_bin(row["age"], bin_width), []).append(index)

    train_indices = []
    val_indices = []
    test_indices = []
    for indices in indices_by_bin.values():
        rng.shuffle(indices)
        test_count = round(len(indices) * test_fraction)
        val_count = round(len(indices) * val_fraction)

        if test_fraction > 0 and test_count == 0 and len(indices) >= 3:
            test_count = 1
        if val_count == 0 and len(indices) - test_count >= 2:
            val_count = 1

        test_indices.extend(indices[:test_count])
        val_indices.extend(indices[test_count : test_count + val_count])
        train_indices.extend(indices[test_count + val_count :])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return dataset.select(train_indices), dataset.select(val_indices), dataset.select(test_indices)


def age_distribution(dataset, bin_width: int) -> dict[str, int]:
    counts = Counter(age_bin(row["age"], bin_width) for row in dataset)
    return {f"{start}-{start + bin_width - 1}": counts[start] for start in sorted(counts)}


def sample_weight_by_bin(dataset, bin_width: int, cap: float, power: float) -> dict[int, float]:
    counts = Counter(age_bin(row["age"], bin_width) for row in dataset)
    if not counts:
        return {}

    average_count = sum(counts.values()) / len(counts)
    return {bin_start: min(cap, (average_count / count) ** power) for bin_start, count in counts.items()}


def image_payload_from_cell(cell):
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data is not None:
            return data
        path = cell.get("path")
        if path:
            return Path(path)
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell)
    if isinstance(cell, (str, Path)):
        return Path(cell)
    return cell


def image_to_array(np, image_module, image_cell):
    payload = image_payload_from_cell(image_cell)

    if isinstance(payload, Path):
        with image_module.open(payload) as image:
            image = image.convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), image_module.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.float32)

    if isinstance(payload, (bytes, bytearray, memoryview)):
        with image_module.open(io.BytesIO(bytes(payload))) as image:
            image = image.convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), image_module.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.float32)

    if hasattr(payload, "convert"):
        image = payload.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), image_module.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)

    raise ValueError(f"Unsupported image cell format: {type(image_cell)!r}")


def sample_generator(np, image_module, unidentified_image_error, dataset, weights_by_bin: dict[int, float] | None, bin_width: int):
    skipped = 0
    for row in dataset:
        try:
            age = float(row["age"])
            image = image_to_array(np, image_module, row["image"])
            if weights_by_bin is None:
                yield image, age
            else:
                yield image, age, float(weights_by_bin.get(age_bin(age, bin_width), 1.0))
        except (OSError, ValueError, unidentified_image_error) as exc:
            skipped += 1
            if skipped <= 10:
                print(f"Skipping unreadable image row {skipped}: {exc}")


def make_dataset(
    tf,
    np,
    image_module,
    unidentified_image_error,
    dataset,
    batch_size: int,
    shuffle: bool,
    repeat: bool,
    weights_by_bin: dict[int, float] | None = None,
    bin_width: int = 10,
):
    output_signature = [
        tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ]
    if weights_by_bin is not None:
        output_signature.append(tf.TensorSpec(shape=(), dtype=tf.float32))

    tf_dataset = tf.data.Dataset.from_generator(
        lambda: sample_generator(np, image_module, unidentified_image_error, dataset, weights_by_bin, bin_width),
        output_signature=tuple(output_signature),
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=batch_size * 32, reshuffle_each_iteration=True)
    if repeat:
        tf_dataset = tf_dataset.repeat()
    return tf_dataset.batch(batch_size).prefetch(AUTOTUNE)


def make_training_callbacks(
    keras,
    checkpoint_path: Path,
    log_path: Path,
    patience: int,
    append_log: bool,
    initial_best: float | None = None,
):
    checkpoint_kwargs = {}
    if initial_best is not None:
        checkpoint_kwargs["initial_value_threshold"] = initial_best

    return [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_mae_years",
            mode="min",
            save_best_only=True,
            **checkpoint_kwargs,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mae_years",
            mode="min",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae_years",
            mode="min",
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(log_path), append=append_log),
    ]


def compile_model(keras, model, learning_rate: float):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-5,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.Huber(delta=6.0),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae_years"),
            keras.metrics.RootMeanSquaredError(name="rmse_years"),
        ],
    )


def build_model(keras, weights: str | None, max_age: float):
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")

    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.04),
            keras.layers.RandomZoom(0.08),
            keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    x = augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    base.trainable = False

    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.35)(x)
    x = keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = keras.layers.Dropout(0.15)(x)
    x = keras.layers.Dense(1, activation="sigmoid")(x)
    outputs = keras.layers.Rescaling(max_age, name="age")(x)

    model = keras.Model(inputs, outputs)
    compile_model(keras, model, learning_rate=7e-4)
    return model, base


def configure_device(tf, requested_device: str):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if requested_device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
        return []

    if requested_device == "gpu" and not gpus:
        message = "No TensorFlow GPU device is available."
        if platform.system() == "Windows":
            message += (
                " TensorFlow 2.11+ does not use NVIDIA CUDA GPUs on native Windows; "
                "use WSL2/Linux for NVIDIA GPU training, or install/configure a "
                "DirectML-compatible TensorFlow stack."
            )
        raise SystemExit(message)

    return gpus


def train(args):
    np, tf, keras, load_dataset, dataset_image, image_module, unidentified_image_error = import_runtime()
    gpus = configure_device(tf, args.device)

    dataset = load_age_dataset(load_dataset, args.dataset, args.data_dir, args.split)
    source_rows = len(dataset)
    dataset = filter_target_age_range(dataset, TARGET_MIN_AGE, TARGET_MAX_AGE)
    filtered_rows = len(dataset)
    train_data, val_data, test_data = prepare_splits(
        dataset,
        dataset_image,
        args.val_fraction,
        args.test_fraction,
        args.seed,
        args.age_bin_width,
    )
    weights_by_bin = (
        sample_weight_by_bin(train_data, args.age_bin_width, args.sample_weight_cap, args.sample_weight_power)
        if args.use_sample_weights
        else None
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = len(train_data)
    val_rows = len(val_data)
    test_rows = len(test_data)
    if train_rows == 0:
        raise ValueError("Training split is empty. Reduce --val-fraction/--test-fraction or use more data.")
    if val_rows == 0:
        raise ValueError("Validation split is empty. Increase the dataset size or --val-fraction.")
    steps_per_epoch = math.ceil(train_rows / args.batch_size)

    print(f"TensorFlow: {tf.__version__}")
    print(f"Dataset: {args.dataset if args.data_dir is None else args.data_dir}")
    print(f"Target age range: {TARGET_MIN_AGE:g}-{TARGET_MAX_AGE:g} years (excluding {TARGET_MAX_AGE:g}+)")
    print(f"Rows after age filter: {filtered_rows:,} of {source_rows:,}")
    print(f"GPU devices: {[gpu.name for gpu in gpus] or 'none'}")
    print(f"Training rows: {train_rows:,}")
    print(f"Validation rows: {val_rows:,}")
    print(f"Test rows: {test_rows:,}")
    print(f"Sample weights: {'on' if weights_by_bin is not None else 'off'}")
    if weights_by_bin is not None:
        print(f"Sample weight cap: {args.sample_weight_cap:g}")
        print(f"Sample weight power: {args.sample_weight_power:g}")

    train_ds = make_dataset(
        tf,
        np,
        image_module,
        unidentified_image_error,
        train_data,
        args.batch_size,
        shuffle=True,
        repeat=True,
        weights_by_bin=weights_by_bin,
        bin_width=args.age_bin_width,
    )
    val_ds = make_dataset(
        tf,
        np,
        image_module,
        unidentified_image_error,
        val_data,
        args.batch_size,
        shuffle=False,
        repeat=False,
    )
    test_ds = make_dataset(
        tf,
        np,
        image_module,
        unidentified_image_error,
        test_data,
        args.batch_size,
        shuffle=False,
        repeat=False,
    ) if test_rows else None

    weights = None if args.weights == "none" else args.weights
    model, base = build_model(keras, weights=weights, max_age=TARGET_MAX_AGE)
    checkpoint_path = OUTPUT_DIR / "best_age_model.keras"
    log_path = OUTPUT_DIR / "training_log.csv"
    callbacks = make_training_callbacks(keras, checkpoint_path, log_path, args.patience, append_log=False)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )

    if args.fine_tune_epochs > 0:
        val_mae_history = history.history.get("val_mae_years") or [float("inf")]
        best_val_mae = min(val_mae_history)
        initial_best = best_val_mae if math.isfinite(best_val_mae) else None
        base.trainable = True
        trainable_base_layers = min(args.fine_tune_layers, len(base.layers))
        for layer in base.layers[:-trainable_base_layers]:
            layer.trainable = False
        compile_model(keras, model, learning_rate=args.fine_tune_learning_rate)
        fine_tune_callbacks = make_training_callbacks(
            keras,
            checkpoint_path,
            log_path,
            args.patience,
            append_log=True,
            initial_best=initial_best,
        )
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=args.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=fine_tune_callbacks,
        )
        history.history.update({f"fine_tune_{key}": value for key, value in history_ft.history.items()})

    if checkpoint_path.exists():
        model = keras.models.load_model(checkpoint_path)
    model.save(OUTPUT_DIR / "last_age_model.keras")
    validation_metrics = model.evaluate(val_ds, return_dict=True)
    test_metrics = model.evaluate(test_ds, return_dict=True) if test_ds is not None else {}
    save_split_predictions(
        tf,
        keras,
        np,
        image_module,
        unidentified_image_error,
        checkpoint_path,
        val_data,
        args.batch_size,
        OUTPUT_DIR / "validation_predictions.csv",
        OUTPUT_DIR / "validation_age_bin_metrics.csv",
        args.age_bin_width,
        TARGET_MAX_AGE,
    )
    if test_rows:
        save_split_predictions(
            tf,
            keras,
            np,
            image_module,
            unidentified_image_error,
            checkpoint_path,
            test_data,
            args.batch_size,
            OUTPUT_DIR / "test_predictions.csv",
            OUTPUT_DIR / "test_age_bin_metrics.csv",
            args.age_bin_width,
            TARGET_MAX_AGE,
        )

    summary = {
        "tensorflow": tf.__version__,
        "dataset": args.dataset if args.data_dir is None else str(args.data_dir),
        "device": args.device,
        "gpu_devices": [gpu.name for gpu in gpus],
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "test_rows": test_rows,
        "source_rows": source_rows,
        "excluded_rows": source_rows - filtered_rows,
        "sample_weights": bool(weights_by_bin),
        "sample_weight_cap": args.sample_weight_cap if weights_by_bin is not None else None,
        "sample_weight_power": args.sample_weight_power if weights_by_bin is not None else None,
        "sample_weights_by_bin": {
            f"{start}-{start + args.age_bin_width - 1}": float(weight)
            for start, weight in sorted((weights_by_bin or {}).items())
        },
        "fine_tune_layers": args.fine_tune_layers,
        "fine_tune_learning_rate": args.fine_tune_learning_rate,
        "target_age_range": {
            "min_inclusive": TARGET_MIN_AGE,
            "max_exclusive": TARGET_MAX_AGE,
        },
        "age_bin_width": args.age_bin_width,
        "train_age_distribution": age_distribution(train_data, args.age_bin_width),
        "validation_age_distribution": age_distribution(val_data, args.age_bin_width),
        "test_age_distribution": age_distribution(test_data, args.age_bin_width),
        "validation_metrics": {key: float(value) for key, value in validation_metrics.items()},
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "best_model": str(checkpoint_path),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def save_split_predictions(
    tf,
    keras,
    np,
    image_module,
    unidentified_image_error,
    model_path: Path,
    split_data,
    batch_size: int,
    predictions_path: Path,
    bin_metrics_path: Path,
    bin_width: int,
    max_age: float,
):
    model = keras.models.load_model(model_path)
    val_ds = make_dataset(
        tf,
        np,
        image_module,
        unidentified_image_error,
        split_data,
        batch_size,
        shuffle=False,
        repeat=False,
    )
    bin_stats = {}

    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["actual_age", "predicted_age", "absolute_error", "age_bin"])

        for images, ages in val_ds:
            predictions = model.predict(images, verbose=0).reshape(-1)
            for actual, predicted in zip(ages.numpy(), predictions):
                actual_age = float(actual)
                predicted_age = min(max_age, max(0.0, float(predicted)))
                absolute_error = abs(actual_age - predicted_age)
                bin_start = age_bin(actual_age, bin_width)
                writer.writerow(
                    [
                        round(actual_age, 2),
                        round(predicted_age, 2),
                        round(absolute_error, 2),
                        f"{bin_start}-{bin_start + bin_width - 1}",
                    ]
                )
                stats = bin_stats.setdefault(bin_start, {"count": 0, "absolute_error_sum": 0.0, "squared_error_sum": 0.0})
                stats["count"] += 1
                stats["absolute_error_sum"] += absolute_error
                stats["squared_error_sum"] += (actual_age - predicted_age) ** 2

    with bin_metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["age_bin", "count", "mae_years", "rmse_years"])
        for bin_start in sorted(bin_stats):
            stats = bin_stats[bin_start]
            count = stats["count"]
            writer.writerow(
                [
                    f"{bin_start}-{bin_start + bin_width - 1}",
                    count,
                    round(stats["absolute_error_sum"] / count, 4),
                    round(math.sqrt(stats["squared_error_sum"] / count), 4),
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Keras age predictor from face images.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--fine-tune-layers", type=int, default=50)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--age-bin-width", type=int, default=10)
    parser.add_argument("--use-sample-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-weight-cap", type=float, default=8.0)
    parser.add_argument("--sample-weight-power", type=float, default=1.15)
    args = parser.parse_args()

    if args.epochs < 0:
        parser.error("--epochs must be 0 or greater")
    if args.fine_tune_epochs < 0:
        parser.error("--fine-tune-epochs must be 0 or greater")
    if args.fine_tune_layers <= 0:
        parser.error("--fine-tune-layers must be greater than 0")
    if args.fine_tune_learning_rate <= 0:
        parser.error("--fine-tune-learning-rate must be greater than 0")
    if args.epochs == 0 and args.fine_tune_epochs == 0:
        parser.error("at least one of --epochs or --fine-tune-epochs must be greater than 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if args.patience < 0:
        parser.error("--patience must be 0 or greater")
    if args.age_bin_width <= 0:
        parser.error("--age-bin-width must be greater than 0")
    if args.sample_weight_cap <= 0:
        parser.error("--sample-weight-cap must be greater than 0")
    if args.sample_weight_power <= 0:
        parser.error("--sample-weight-power must be greater than 0")

    return args


if __name__ == "__main__":
    train(parse_args())
