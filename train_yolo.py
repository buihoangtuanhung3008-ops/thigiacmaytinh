"""Train a single-class YOLO detector from the exported Roboflow dataset."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = (
    PROJECT_DIR
    / "anh yolo"
    / "Warehouse Robot Docking and Path.v2-vatcanv1.yolov11 (1)"
    / "data.yaml"
)
DEFAULT_MODEL = PROJECT_DIR / "yolo11n.pt"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_FOLDERS = {
    "train": ("train",),
    "val": ("valid", "val"),
}


def _class_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names")
    if isinstance(names, list):
        return dict(enumerate(str(name) for name in names))
    if isinstance(names, dict):
        try:
            return {int(class_id): str(name) for class_id, name in names.items()}
        except ValueError as error:
            raise ValueError("Class IDs in data.yaml must be integers.") from error
    raise ValueError("data.yaml must define class names as a list or mapping.")


def _split_paths(
    dataset_root: Path, split: str
) -> tuple[Path, Path, list[Path]]:
    candidates: list[tuple[Path, Path]] = []
    for folder in SPLIT_FOLDERS[split]:
        candidates.extend(
            (
                (dataset_root / folder / "images", dataset_root / folder / "labels"),
                (dataset_root / "images" / folder, dataset_root / "labels" / folder),
            )
        )

    for image_dir, label_dir in candidates:
        if not image_dir.is_dir():
            continue
        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            if not label_dir.is_dir():
                raise FileNotFoundError(f"Missing labels directory: {label_dir}")
            return image_dir, label_dir, images

    expected = ", ".join(str(image_dir) for image_dir, _ in candidates)
    raise FileNotFoundError(f"No {split} images found. Checked: {expected}")


def _parse_label_file(
    path: Path,
    class_names: dict[int, str],
    target_class_id: int,
) -> tuple[list[str], Counter[str]]:
    converted_rows: list[str] = []
    counts: Counter[str] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        try:
            class_value = float(fields[0])
            class_id = int(class_value)
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"Invalid label at {path}:{line_number}") from error
        if class_value != class_id or class_id not in class_names:
            raise ValueError(f"Unknown class ID at {path}:{line_number}")
        if len(coordinates) < 4 or len(coordinates) % 2:
            raise ValueError(f"Invalid YOLO box or polygon at {path}:{line_number}")
        if any(value < 0 or value > 1 for value in coordinates):
            raise ValueError(f"Label coordinates must be normalized at {path}:{line_number}")

        class_name = class_names[class_id]
        counts[class_name] += 1
        if class_id != target_class_id:
            continue

        if len(coordinates) == 4:
            center_x, center_y, box_width, box_height = coordinates
        else:
            x_coordinates = coordinates[0::2]
            y_coordinates = coordinates[1::2]
            left, right = min(x_coordinates), max(x_coordinates)
            top, bottom = min(y_coordinates), max(y_coordinates)
            center_x, center_y = (left + right) / 2, (top + bottom) / 2
            box_width, box_height = right - left, bottom - top
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Empty object box at {path}:{line_number}")
        converted_rows.append(
            f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return converted_rows, counts


def _load_dataset(
    data_path: Path, target_name: str
) -> tuple[dict[int, str], int, dict[str, tuple[Path, Path, list[Path]]]]:
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid dataset configuration: {data_path}")
    class_names = _class_names(config)
    target_ids = [
        class_id
        for class_id, name in class_names.items()
        if name.strip().casefold() == target_name.casefold()
    ]
    if len(target_ids) != 1:
        raise ValueError(
            f"Expected exactly one class named {target_name!r}; found {len(target_ids)}."
        )

    splits = {
        split: _split_paths(data_path.parent, split)
        for split in SPLIT_FOLDERS
    }
    for split, (image_dir, label_dir, images) in splits.items():
        missing_labels = [
            image.stem for image in images
            if not (label_dir / f"{image.stem}.txt").is_file()
        ]
        if missing_labels:
            raise FileNotFoundError(
                f"{split} images are missing label files: {', '.join(missing_labels[:5])}"
            )
    return class_names, target_ids[0], splits


def _print_dataset_summary(
    class_names: dict[int, str],
    target_class_id: int,
    splits: dict[str, tuple[Path, Path, list[Path]]],
) -> None:
    print(f"Training class: {class_names[target_class_id]} (YOLO class 0)")
    for split, (_, label_dir, images) in splits.items():
        counts: Counter[str] = Counter()
        for image in images:
            _, image_counts = _parse_label_file(
                label_dir / f"{image.stem}.txt", class_names, target_class_id
            )
            counts.update(image_counts)
        if split == "train" and counts[class_names[target_class_id]] == 0:
            raise ValueError("No target objects were found in the training labels.")
        print(
            f"{split}: {len(images)} images, "
            f"{counts[class_names[target_class_id]]} target labels; "
            f"ignored floor labels: {counts['san'] + counts['sàn']}"
        )


def _prepare_training_data(
    temp_root: Path,
    class_names: dict[int, str],
    target_class_id: int,
    splits: dict[str, tuple[Path, Path, list[Path]]],
) -> Path:
    target_name = class_names[target_class_id]
    for split, (_, label_dir, images) in splits.items():
        image_output = temp_root / split / "images"
        label_output = temp_root / split / "labels"
        image_output.mkdir(parents=True)
        label_output.mkdir(parents=True)
        for image in images:
            shutil.copy2(image, image_output / image.name)
            rows, _ = _parse_label_file(
                label_dir / f"{image.stem}.txt", class_names, target_class_id
            )
            (label_output / f"{image.stem}.txt").write_text(
                "\n".join(rows), encoding="utf-8"
            )

    data_path = temp_root / "data.yaml"
    training_config = {
        "path": str(temp_root),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": [target_name],
    }
    data_path.write_text(
        yaml.safe_dump(training_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return data_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO to detect only vat_can from the anh yolo dataset."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--class-name", default="vat_can")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default=None, help="For example: cpu or 0 (GPU).")
    parser.add_argument("--name", default="vat_can")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate dataset paths and labels without starting training.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("--epochs and --batch must be positive; --imgsz must be >= 32.")
    if not args.data.is_file():
        parser.error(f"Dataset YAML not found: {args.data}")

    class_names, target_class_id, splits = _load_dataset(
        args.data.resolve(), args.class_name
    )
    _print_dataset_summary(class_names, target_class_id, splits)
    if args.check_only:
        print("Dataset check passed. Floor annotations are excluded from training.")
        return
    if not args.model.is_file():
        parser.error(f"YOLO model weights not found: {args.model}")

    with tempfile.TemporaryDirectory(prefix="yolo_vat_can_") as temp_dir:
        data_path = _prepare_training_data(
            Path(temp_dir), class_names, target_class_id, splits
        )
        model = YOLO(str(args.model.resolve()))
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=0,
            project=str(PROJECT_DIR / "runs" / "detect"),
            name=args.name,
        )
        best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
        print(f"Best weights: {best_weights}")


if __name__ == "__main__":
    main()
