"""Train YOLO instance segmentation from the Roboflow polygon annotations."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = (
    PROJECT_DIR
    / "anh yolo"
    / "Warehouse Robot Docking and Path.v2-vatcanv1.yolov11 (1)"
    / "data.yaml"
)
DEFAULT_MODEL = "yolo11n-seg.pt"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_FOLDERS = {"train": ("train",), "val": ("valid", "val")}
FLOOR_NAMES = {"san", "sàn", "floor", "ground"}


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


def _find_split(
    dataset_root: Path, split: str
) -> tuple[Path, Path, list[Path]]:
    for folder in SPLIT_FOLDERS[split]:
        for image_dir, label_dir in (
            (dataset_root / folder / "images", dataset_root / folder / "labels"),
            (
                dataset_root / "images" / folder,
                dataset_root / "labels" / folder,
            ),
        ):
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

    raise FileNotFoundError(f"No {split} images found under {dataset_root}.")


def _validate_polygon_labels(
    label_path: Path, class_names: dict[int, str]
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split()
        if not fields:
            continue
        try:
            class_value = float(fields[0])
            class_id = int(class_value)
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(
                f"Invalid segmentation label at {label_path}:{line_number}"
            ) from error
        if class_value != class_id or class_id not in class_names:
            raise ValueError(
                f"Unknown class ID at {label_path}:{line_number}"
            )
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(
                f"Segmentation labels must contain polygon points at "
                f"{label_path}:{line_number}."
            )
        if any(value < 0 or value > 1 for value in coordinates):
            raise ValueError(
                f"Polygon coordinates must be normalized at "
                f"{label_path}:{line_number}."
            )
        counts[class_id] += 1
    return counts


def _load_dataset(
    data_path: Path,
) -> tuple[dict[int, str], dict[str, tuple[Path, Path, list[Path]]]]:
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid dataset configuration: {data_path}")
    class_names = _class_names(config)
    if sorted(class_names) != list(range(len(class_names))):
        raise ValueError("Class IDs must be contiguous and start at 0.")
    if not any(name.strip().casefold() in FLOOR_NAMES for name in class_names.values()):
        raise ValueError("Dataset must include a floor class such as 'san'.")
    if not any(name.strip().casefold() == "vat_can" for name in class_names.values()):
        raise ValueError("Dataset must include the 'vat_can' class.")

    splits = {
        split: _find_split(data_path.parent, split)
        for split in SPLIT_FOLDERS
    }
    for split, (_, label_dir, images) in splits.items():
        split_counts: Counter[int] = Counter()
        for image in images:
            label_path = label_dir / f"{image.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing segmentation labels: {label_path}")
            split_counts.update(_validate_polygon_labels(label_path, class_names))
        if not split_counts:
            raise ValueError(f"No polygon annotations were found in {split}.")
        if split == "train":
            missing = [
                name for class_id, name in class_names.items()
                if split_counts[class_id] == 0
            ]
            if missing:
                raise ValueError(
                    "Training polygons are missing classes: "
                    + ", ".join(missing)
                )
    return class_names, splits


def _print_dataset_summary(
    class_names: dict[int, str],
    splits: dict[str, tuple[Path, Path, list[Path]]],
) -> None:
    for split, (_, label_dir, images) in splits.items():
        counts: Counter[int] = Counter()
        for image in images:
            counts.update(
                _validate_polygon_labels(
                    label_dir / f"{image.stem}.txt", class_names
                )
            )
        summary = ", ".join(
            f"{name}: {counts[class_id]}"
            for class_id, name in class_names.items()
        )
        print(f"{split}: {len(images)} images; polygon instances: {summary}")


def _prepare_dataset(
    temp_root: Path,
    class_names: dict[int, str],
    splits: dict[str, tuple[Path, Path, list[Path]]],
) -> Path:
    for split, (_, label_dir, images) in splits.items():
        image_output = temp_root / split / "images"
        label_output = temp_root / split / "labels"
        image_output.mkdir(parents=True)
        label_output.mkdir(parents=True)
        for image in images:
            shutil.copy2(image, image_output / image.name)
            shutil.copy2(
                label_dir / f"{image.stem}.txt",
                label_output / f"{image.stem}.txt",
            )

    data_path = temp_root / "data.yaml"
    training_config = {
        "path": str(temp_root),
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_names),
        "names": [class_names[class_id] for class_id in sorted(class_names)],
    }
    data_path.write_text(
        yaml.safe_dump(training_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return data_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO segmentation for floor and obstacle polygons."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--name", default="vat_can_seg")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate polygon annotations without starting training.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("--epochs and --batch must be positive; --imgsz must be >= 32.")
    if not args.data.is_file():
        parser.error(f"Dataset YAML not found: {args.data}")

    class_names, splits = _load_dataset(args.data.resolve())
    _print_dataset_summary(class_names, splits)
    if args.check_only:
        print("Segmentation dataset check passed; source polygon labels are preserved.")
        return

    from ultralytics import YOLO

    with tempfile.TemporaryDirectory(prefix="yolo_vat_can_seg_") as temp_dir:
        data_path = _prepare_dataset(Path(temp_dir), class_names, splits)
        model = YOLO(args.model)
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=0,
            project=str(PROJECT_DIR / "runs" / "segment"),
            name=args.name,
        )
        best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
        print(f"Best segmentation weights: {best_weights}")


if __name__ == "__main__":
    main()
