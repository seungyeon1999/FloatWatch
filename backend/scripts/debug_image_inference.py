from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis_service import (  # noqa: E402
    CLASS_CONFIDENCE_THRESHOLDS,
    INFERENCE_DEVICE,
    NET_MIN_CONFIDENCE,
    class_confidence_indices,
    detection_rows,
    filter_result,
    inference_environment_info,
    normalize_class_name,
    representative_image_predict_options,
    sha256_file,
)


def print_detections(title: str, rows: list[dict[str, object]], count_label: str) -> None:
    print(title)
    print(f"{count_label}={len(rows)}")
    for row in rows:
        x1, y1, x2, y2 = row["xyxy"]
        print(
            f"{row['index']} | class_id={row['class_id']} | class={row['class_name']} | "
            f"conf={row['confidence']:.4f} | box=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FloatWatch image inference diagnostics without creating an analysis record.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--representative", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    image_path = Path(args.image_path).resolve()
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"could not read image: {image_path}")

    model_artifact = argparse.Namespace(model_key=args.model_key, is_representative=args.representative)
    image_predict_options = representative_image_predict_options(model_artifact)
    predict_options = {"conf": args.confidence}
    class_thresholds = None
    if image_predict_options is not None:
        predict_options.update(image_predict_options)
        class_thresholds = CLASS_CONFIDENCE_THRESHOLDS

    print("[IMAGE INFERENCE DEBUG]")
    print(f"model_key={args.model_key}")
    print(f"is_representative={args.representative}")
    print(f"model_path={model_path}")
    print(f"model_sha256={sha256_file(model_path)}")
    print(f"source_image_path={image_path}")
    print(f"source_image_sha256={sha256_file(image_path)}")
    print(f"image_shape={frame.shape[1]}x{frame.shape[0]}x{frame.shape[2] if len(frame.shape) > 2 else 1}")
    print(f"device={INFERENCE_DEVICE}")
    print(f"conf={predict_options.get('conf')}")
    print(f"iou={predict_options.get('iou')}")
    print(f"imgsz={predict_options.get('imgsz')}")
    print(f"max_det={predict_options.get('max_det')}")
    print(f"class_thresholds={'enabled' if class_thresholds is not None else 'default'}")
    print(f"environment={inference_environment_info()}")

    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    result = model.predict(frame, **predict_options, device=INFERENCE_DEVICE, verbose=False)[0]

    raw_rows = detection_rows(result, names)
    print_detections("[RAW DETECTIONS]", raw_rows, "raw_detection_count")

    raw_class_ids = [int(row["class_id"]) for row in raw_rows]
    raw_confidences = [float(row["confidence"]) for row in raw_rows]
    kept = class_confidence_indices(raw_class_ids, raw_confidences, names, args.confidence, class_thresholds)
    kept_set = set(kept)
    print("[FILTER DEBUG]")
    for index, row in enumerate(raw_rows):
        class_name = str(row["class_name"])
        normalized = normalize_class_name(class_name)
        if class_thresholds is not None:
            threshold = class_thresholds.get(normalized, args.confidence)
        else:
            threshold = max(args.confidence, NET_MIN_CONFIDENCE) if normalized == "net" else args.confidence
        print(
            f"{class_name} conf={row['confidence']:.4f} threshold={threshold:.2f} "
            f"{'KEEP' if index in kept_set else 'REMOVE'}"
        )

    result = filter_result(result, kept)
    print_detections("[FINAL DETECTIONS]", detection_rows(result, names), "final_detection_count")


if __name__ == "__main__":
    main()
