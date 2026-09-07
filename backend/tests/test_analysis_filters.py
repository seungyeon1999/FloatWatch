from types import SimpleNamespace

import cv2
import numpy as np

from app.analysis_service import (
    CLASS_CONFIDENCE_THRESHOLDS,
    NET_MIN_CONFIDENCE,
    OpticalFlowBoxTracker,
    TemporalDetectionFilter,
    class_confidence_indices,
    draw_image_result_non_overlapping,
    draw_tracked_boxes,
    label_rectangles_overlap,
    place_label_rect,
    representative_image_predict_options,
)


def test_net_uses_stricter_confidence_threshold():
    names = {0: "Net", 1: "PET_Bottle"}

    kept = class_confidence_indices([0, 0, 1], [NET_MIN_CONFIDENCE - 0.01, NET_MIN_CONFIDENCE, 0.30], names, 0.25)

    assert kept == [1, 2]


def test_user_threshold_overrides_net_minimum_when_higher():
    kept = class_confidence_indices([0], [0.70], {0: "Net"}, 0.75)

    assert kept == []


def test_representative_image_thresholds_apply_per_class():
    names = {0: "Glass", 1: "Plastic_Buoy", 2: "Net", 3: "Styrofoam Piece"}

    kept = class_confidence_indices(
        [0, 0, 1, 1, 2, 2, 3, 3],
        [0.34, 0.35, 0.74, 0.75, 0.39, 0.40, 0.49, 0.50],
        names,
        0.10,
        CLASS_CONFIDENCE_THRESHOLDS,
    )

    assert kept == [1, 3, 5, 7]


def test_representative_image_predict_options_select_model_specific_size():
    assert representative_image_predict_options(SimpleNamespace(model_key="yolov8s", is_representative=True)) == {
        "conf": 0.10,
        "iou": 0.40,
        "imgsz": 800,
        "max_det": 300,
    }
    assert representative_image_predict_options(SimpleNamespace(model_key="yolov11s", is_representative=True))["imgsz"] == 1280
    assert representative_image_predict_options(SimpleNamespace(model_key="yolov26s", is_representative=True))["imgsz"] == 960
    assert representative_image_predict_options(SimpleNamespace(model_key="rt-detr", is_representative=True)) is None
    assert representative_image_predict_options(SimpleNamespace(model_key="yolov8s", is_representative=False)) is None


def test_image_label_prefers_box_top_when_available():
    rect = place_label_rect((20, 30, 70, 80), 52, 18, 200, 120, [])

    assert rect == (20, 12, 72, 30)


def test_image_label_moves_away_from_nearby_label_collision():
    used = [(20, 12, 72, 30)]

    rect = place_label_rect((22, 32, 72, 82), 52, 18, 200, 120, used)

    assert not label_rectangles_overlap(rect, used[0])


def test_image_label_stays_inside_top_and_side_edges():
    top_rect = place_label_rect((10, 2, 60, 40), 52, 18, 200, 120, [])
    edge_rect = place_label_rect((170, 50, 199, 90), 52, 18, 200, 120, [])

    assert top_rect[1] >= 0
    assert top_rect[1] >= 40
    assert edge_rect[0] >= 0
    assert edge_rect[2] <= 200


def test_image_annotation_keeps_all_detection_payloads_unchanged():
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    coordinates = [[float(index), 10.0, float(index + 20), 30.0] for index in range(10)]
    confidences = [0.75 + index * 0.01 for index in range(10)]
    class_ids = [float(index % 3) for index in range(10)]
    boxes = SimpleNamespace(
        xyxy=FakeTensor(coordinates),
        conf=FakeTensor(confidences),
        cls=FakeTensor(class_ids),
    )
    result = SimpleNamespace(boxes=boxes)
    frame = np.zeros((100, 160, 3), dtype=np.uint8)

    annotated = draw_image_result_non_overlapping(frame, result, {0: "Glass", 1: "Rope", 2: "Styrofoam_Piece"})

    assert annotated.shape == frame.shape
    assert np.count_nonzero(annotated) > 0
    assert result.boxes.xyxy.tolist() == coordinates
    assert result.boxes.conf.tolist() == confidences
    assert result.boxes.cls.tolist() == class_ids


def test_temporal_filter_requires_three_consecutive_overlapping_detections():
    temporal_filter = TemporalDetectionFilter(minimum_consecutive=3, iou_threshold=0.25)
    box = (10.0, 10.0, 40.0, 40.0)

    assert temporal_filter.update([(0, 7, box)]) == []
    assert temporal_filter.update([(0, 7, (11.0, 10.0, 41.0, 40.0))]) == []
    assert temporal_filter.update([(0, 7, (12.0, 11.0, 42.0, 41.0))]) == [0]


def test_temporal_filter_resets_after_a_missing_frame():
    temporal_filter = TemporalDetectionFilter(minimum_consecutive=3, iou_threshold=0.25)
    box = (10.0, 10.0, 40.0, 40.0)

    temporal_filter.update([(0, 7, box)])
    temporal_filter.update([(0, 7, box)])
    temporal_filter.update([])

    assert temporal_filter.update([(0, 7, box)]) == []


def test_optical_flow_tracker_moves_box_with_frame_content():
    first = np.zeros((100, 120, 3), dtype=np.uint8)
    for y in range(30, 61, 10):
        for x in range(25, 56, 10):
            cv2.circle(first, (x, y), 2, (255, 255, 255), -1)
    second = cv2.warpAffine(first, np.float32([[1, 0, 6], [0, 1, 3]]), (120, 100))
    tracker = OpticalFlowBoxTracker()
    tracker.reset(first, [{"class_id": 1, "confidence": 0.8, "box": (20.0, 25.0, 60.0, 65.0)}])

    tracked = tracker.update(second)

    x1, y1, x2, y2 = tracked[0]["box"]
    assert 25.0 <= x1 <= 27.0
    assert 27.0 <= y1 <= 29.0
    assert x2 - x1 == 40.0
    assert y2 - y1 == 40.0


def test_tracker_keeps_box_during_short_detection_gap():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    tracker = OpticalFlowBoxTracker()
    detection = {"class_id": 1, "confidence": 0.8, "box": (20.0, 20.0, 60.0, 60.0)}

    assert len(tracker.reconcile(frame, [detection])) == 1
    for _ in range(5):
        assert len(tracker.reconcile(frame, [])) == 1
    assert tracker.reconcile(frame, []) == []


def test_tracker_refreshes_grace_period_when_detection_returns():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    tracker = OpticalFlowBoxTracker()
    detection = {"class_id": 1, "confidence": 0.8, "box": (20.0, 20.0, 60.0, 60.0)}

    tracker.reconcile(frame, [detection])
    tracker.reconcile(frame, [])
    refreshed = tracker.reconcile(frame, [detection])

    assert len(refreshed) == 1
    assert refreshed[0]["missed_frames"] == 0


def test_draw_tracked_boxes_keeps_source_frame_unchanged():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    annotated = draw_tracked_boxes(
        frame,
        [{"class_id": 1, "confidence": 0.8, "box": (20.0, 20.0, 60.0, 60.0)}],
        {1: "PET_Bottle"},
    )

    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(annotated) > 0
