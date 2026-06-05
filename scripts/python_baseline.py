#!/usr/bin/env python3
"""Python baseline for the EdgeVisionRT tracking pipeline.

This is intentionally a plain Python/OpenCV baseline for comparison with the
optimized C++/NCNN implementation. It mirrors the broad runtime flow:

  video frame -> letterbox -> NCNN YOLOv8 -> NMS -> IoU tracker
  -> homography ground point -> line crossing -> annotated output

It is not an exact OCSort port. The tracker here is a small IoU tracker so the
script stays easy to run on the Pi and gives a useful "Python floor" for FPS.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


CLASS_NAMES = [
    "person",
    "helmet",
    "self_clothes",
    "safety_clothes",
    "head",
    "blur_head",
    "blur_clothes",
]


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int


@dataclass
class Track:
    bbox: np.ndarray
    track_id: int
    cls: int
    conf: float
    hits: int = 1
    age: int = 0
    last_ground: tuple[float, float] | None = None
    counted_in: bool = False
    counted_out: bool = False
    history: list[np.ndarray] = field(default_factory=list)


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.25, max_age: int = 30, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        for trk in self.tracks:
            trk.age += 1

        det_boxes = [np.array([d.x1, d.y1, d.x2, d.y2], dtype=np.float32) for d in detections]
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))

        pairs: list[tuple[float, int, int]] = []
        for di, det_box in enumerate(det_boxes):
            for ti, trk in enumerate(self.tracks):
                if detections[di].cls != trk.cls:
                    continue
                pairs.append((bbox_iou(det_box, trk.bbox), di, ti))
        pairs.sort(reverse=True, key=lambda item: item[0])

        for iou, di, ti in pairs:
            if iou < self.iou_threshold:
                break
            if di not in unmatched_dets or ti not in unmatched_tracks:
                continue
            det = detections[di]
            trk = self.tracks[ti]
            trk.bbox = det_boxes[di]
            trk.cls = det.cls
            trk.conf = det.conf
            trk.hits += 1
            trk.age = 0
            trk.history.append(trk.bbox.copy())
            unmatched_dets.remove(di)
            unmatched_tracks.remove(ti)

        for di in sorted(unmatched_dets):
            det = detections[di]
            trk = Track(
                bbox=det_boxes[di],
                track_id=self.next_id,
                cls=det.cls,
                conf=det.conf,
                history=[det_boxes[di].copy()],
            )
            self.next_id += 1
            self.tracks.append(trk)

        self.tracks = [trk for trk in self.tracks if trk.age <= self.max_age]
        return [trk for trk in self.tracks if trk.age == 0 and trk.hits >= self.min_hits]


class NcnnYolo:
    def __init__(
        self,
        param_path: Path,
        bin_path: Path,
        model_size: int,
        num_classes: int,
        conf_threshold: float,
        nms_threshold: float,
        threads: int,
    ):
        try:
            import ncnn  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Python package 'ncnn' is not installed. Install ncnn Python "
                "bindings on the Pi, or run the C++ app for NCNN inference."
            ) from exc

        self.ncnn = ncnn
        self.model_size = model_size
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))

    def infer(self, frame: np.ndarray) -> tuple[list[Detection], dict[str, float]]:
        t0 = time.perf_counter()
        chw, meta = letterbox_chw(frame, self.model_size)
        t_pre = time.perf_counter()

        mat = self.ncnn.Mat(chw).clone()
        ex = self.net.create_extractor()
        ex.input("in0", mat)
        ret, out = ex.extract("out0")
        if ret != 0:
            raise RuntimeError(f"NCNN extract failed with code {ret}")
        raw = np.array(out)
        t_inf = time.perf_counter()

        detections = decode_yolov8(raw, meta, self.num_classes, self.conf_threshold)
        detections = nms(detections, self.nms_threshold)
        t_post = time.perf_counter()

        return detections, {
            "preprocess_ms": (t_pre - t0) * 1000.0,
            "inference_ms": (t_inf - t_pre) * 1000.0,
            "postprocess_ms": (t_post - t_inf) * 1000.0,
        }


def letterbox_chw(frame: np.ndarray, model_size: int) -> tuple[np.ndarray, dict[str, float]]:
    src_h, src_w = frame.shape[:2]
    scale = min(model_size / src_w, model_size / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    pad_x = (model_size - new_w) // 2
    pad_y = (model_size - new_h) // 2

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((model_size, model_size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
    return chw, {
        "scale": scale,
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "orig_w": float(src_w),
        "orig_h": float(src_h),
        "model_size": float(model_size),
    }


def decode_yolov8(
    raw: np.ndarray,
    meta: dict[str, float],
    num_classes: int,
    conf_threshold: float,
) -> list[Detection]:
    arr = np.squeeze(raw)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    channels = 4 + num_classes
    if arr.shape[0] == channels and arr.shape[1] != channels:
        arr = arr.T
    if arr.shape[1] < channels:
        raise RuntimeError(f"Unexpected YOLO output shape: {raw.shape}")

    boxes = arr[:, :4].astype(np.float32)
    scores = arr[:, 4 : 4 + num_classes].astype(np.float32)
    cls_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(scores.shape[0]), cls_ids]

    keep = confs >= conf_threshold
    boxes = boxes[keep]
    cls_ids = cls_ids[keep]
    confs = confs[keep]

    detections: list[Detection] = []
    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    for box, cls_id, conf in zip(boxes, cls_ids, confs):
        cx, cy, w, h = box.tolist()
        x1 = (cx - w * 0.5 - pad_x) / scale
        y1 = (cy - h * 0.5 - pad_y) / scale
        x2 = (cx + w * 0.5 - pad_x) / scale
        y2 = (cy + h * 0.5 - pad_y) / scale
        x1 = float(np.clip(x1 / orig_w, 0.0, 1.0))
        y1 = float(np.clip(y1 / orig_h, 0.0, 1.0))
        x2 = float(np.clip(x2 / orig_w, 0.0, 1.0))
        y2 = float(np.clip(y2 / orig_h, 0.0, 1.0))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(x1, y1, x2, y2, float(conf), int(cls_id)))
    return detections


def nms(detections: list[Detection], threshold: float) -> list[Detection]:
    by_class: dict[int, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.cls, []).append(det)

    kept: list[Detection] = []
    for cls_dets in by_class.values():
        cls_dets.sort(key=lambda det: det.conf, reverse=True)
        while cls_dets:
            best = cls_dets.pop(0)
            kept.append(best)
            best_box = np.array([best.x1, best.y1, best.x2, best.y2], dtype=np.float32)
            cls_dets = [
                det
                for det in cls_dets
                if bbox_iou(best_box, np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32))
                <= threshold
            ]
    kept.sort(key=lambda det: det.conf, reverse=True)
    return kept


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def load_roi(path: Path | None) -> tuple[np.ndarray | None, list[tuple[float, float]], float]:
    if path is None:
        return None, [], 0.0
    data = json.loads(path.read_text(encoding="utf-8"))
    h = np.array(data["homography"], dtype=np.float64)
    polygon = [tuple(map(float, p)) for p in data.get("ground_polygon", [])]
    line_y = float(data.get("ground_crossing_line", 0.0))
    return h, polygon, line_y


def ground_point(bbox: np.ndarray, h: np.ndarray | None, frame_w: int, frame_h: int) -> tuple[float, float]:
    px = float((bbox[0] + bbox[2]) * 0.5 * frame_w)
    py = float(bbox[3] * frame_h)
    if h is None:
        return px, py
    src = np.array([[[px, py]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, h)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if not polygon:
        return True
    contour = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def draw(
    frame: np.ndarray,
    tracks: Iterable[Track],
    h: np.ndarray | None,
    polygon: list[tuple[float, float]],
    line_y: float,
    counts: tuple[int, int],
    fps: float,
) -> None:
    frame_h, frame_w = frame.shape[:2]
    if h is not None and polygon:
        inv = np.linalg.inv(h)
        pts = np.array([[polygon]], dtype=np.float32)
        img_pts = cv2.perspectiveTransform(pts, inv)[0].astype(np.int32)
        cv2.polylines(frame, [img_pts], True, (255, 0, 0), 2)
    if h is not None and line_y > 0:
        inv = np.linalg.inv(h)
        line = np.array([[[0.0, line_y], [400.0, line_y]]], dtype=np.float32)
        img_line = cv2.perspectiveTransform(line, inv)[0]
        cv2.line(frame, tuple(img_line[0].astype(int)), tuple(img_line[1].astype(int)), (0, 255, 255), 2)

    for trk in tracks:
        x1 = int(trk.bbox[0] * frame_w)
        y1 = int(trk.bbox[1] * frame_h)
        x2 = int(trk.bbox[2] * frame_w)
        y2 = int(trk.bbox[3] * frame_h)
        color = (0, 255, 0) if trk.cls == 0 else (255, 128, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {trk.track_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"FPS {fps:.1f} | IN {counts[0]} OUT {counts[1]}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python baseline for EdgeVisionRT.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("python_baseline_output.mp4"))
    parser.add_argument("--roi", type=Path, default=None)
    parser.add_argument("--param", type=Path, default=Path("models/PPE_model/model.ncnn.param"))
    parser.add_argument("--bin", type=Path, default=Path("models/PPE_model/model.ncnn.bin"))
    parser.add_argument("--width", type=int, default=0, help="Resize runtime frame width; 0 keeps source width.")
    parser.add_argument("--height", type=int, default=0, help="Resize runtime frame height; 0 keeps source height.")
    parser.add_argument("--model-size", type=int, default=640)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--person-id", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--tracker-iou", type=float, default=0.25)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-draw", action="store_true")
    parser.add_argument("--count-roi", action="store_true", help="Only count tracks whose ground point is inside ROI.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_out = src_fps if 0 < src_fps < 240 else 24.0
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read first frame: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.width > 0 and args.height > 0:
        out_w, out_h = args.width, args.height
    else:
        out_h, out_w = first.shape[:2]

    detector = NcnnYolo(
        args.param,
        args.bin,
        args.model_size,
        args.num_classes,
        args.conf,
        args.nms,
        args.threads,
    )
    tracker = IoUTracker(args.tracker_iou, args.max_age, args.min_hits)
    homography, polygon, line_y = load_roi(args.roi)

    writer = None
    if not args.no_draw:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps_out, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open output writer: {args.output}")

    counts_in = 0
    counts_out = 0
    timings: list[dict[str, float]] = []
    total_start = time.perf_counter()
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_index >= args.max_frames:
            break
        if args.width > 0 and args.height > 0:
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

        frame_start = time.perf_counter()
        detections, timing = detector.infer(frame)
        person_dets = [det for det in detections if det.cls == args.person_id]
        tracks = tracker.update(person_dets)

        for trk in tracks:
            ground = ground_point(trk.bbox, homography, frame.shape[1], frame.shape[0])
            if args.count_roi and not point_in_polygon(ground, polygon):
                continue
            if trk.last_ground is not None and line_y > 0:
                last_y = trk.last_ground[1]
                curr_y = ground[1]
                if last_y < line_y <= curr_y and not trk.counted_in:
                    counts_in += 1
                    trk.counted_in = True
                elif last_y >= line_y > curr_y and not trk.counted_out:
                    counts_out += 1
                    trk.counted_out = True
            trk.last_ground = ground

        frame_ms = (time.perf_counter() - frame_start) * 1000.0
        timing["frame_ms"] = frame_ms
        timings.append(timing)
        inst_fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0

        if writer is not None:
            annotated = frame.copy()
            draw(annotated, tracks, homography, polygon, line_y, (counts_in, counts_out), inst_fps)
            writer.write(annotated)

        frame_index += 1
        if frame_index % 30 == 0:
            avg_frame = np.mean([t["frame_ms"] for t in timings[-30:]])
            print(
                f"frame={frame_index} fps={1000.0 / avg_frame:.2f} "
                f"dets={len(person_dets)} tracks={len(tracks)} in={counts_in} out={counts_out}"
            )

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - total_start
    summary = {
        "frames": frame_index,
        "elapsed_s": elapsed,
        "fps": frame_index / elapsed if elapsed > 0 else 0.0,
        "avg_preprocess_ms": float(np.mean([t["preprocess_ms"] for t in timings])) if timings else 0.0,
        "avg_inference_ms": float(np.mean([t["inference_ms"] for t in timings])) if timings else 0.0,
        "avg_postprocess_ms": float(np.mean([t["postprocess_ms"] for t in timings])) if timings else 0.0,
        "avg_frame_ms": float(np.mean([t["frame_ms"] for t in timings])) if timings else 0.0,
        "count_in": counts_in,
        "count_out": counts_out,
        "output": str(args.output) if writer is not None else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
