"""Local files only; no camera/network access in the first experiment."""
from __future__ import annotations

import math
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def resolve_sources(source):
    path = Path(source).resolve()
    if path.is_dir():
        files = sorted(
            (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda p: p.name,
        )
        if not files:
            raise ValueError("No supported images in the input directory.")
        return files, "images"
    if not path.is_file():
        raise FileNotFoundError("Input file or image folder does not exist: " + str(path))
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path], "images"
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return [path], "video"
    raise ValueError("Unsupported input. Use a local image, image folder or video.")


def read_image(path):
    import cv2
    import numpy as np
    # cv2.imread can fail on Windows Unicode paths.
    frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Cannot decode image: " + str(path))
    return frame


def write_image(path, frame):
    import cv2
    extension = Path(path).suffix
    ok, encoded = cv2.imencode(extension, frame)
    if not ok:
        raise RuntimeError("Cannot encode preview image.")
    encoded.tofile(str(path))


def frames(files, kind, stride):
    import cv2
    if kind == "images":
        for path in files:
            yield {"source": path.name, "frame_index": 0, "video_time_s": None,
                   "fps": None, "image": read_image(path)}
        return

    capture = cv2.VideoCapture(str(files[0]))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video: " + str(files[0]))
        raw_fps = capture.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
        raw_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        expected_count = int(raw_count) if math.isfinite(raw_count) and raw_count > 0 else None
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                if index == 0:
                    raise ValueError("Video contains no decodable frames.")
                if expected_count is not None and index < expected_count - 1:
                    raise ValueError(
                        "Video decoding stopped before its advertised frame count; "
                        "check the codec or input file."
                    )
                break
            if index % stride == 0:
                yield {
                    "source": files[0].name, "frame_index": index,
                    "video_time_s": index / fps if fps else None,
                    "fps": fps, "image": frame,
                }
            index += 1
    finally:
        capture.release()


def annotate(frame, detections):
    import cv2
    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(x)) for x in detection["bbox_xyxy"]]
        color = (0, 100, 255) if detection["class_name"] == "fire" else (180, 180, 180)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.2f}".format(detection["class_name"], detection["confidence"])
        cv2.putText(output, label, (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return output


class PreviewWriter:
    def __init__(self, directory, kind, stride):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        self.stride = stride
        self.writer = None

    def add(self, frame_info, detections, ordinal):
        import cv2
        output = annotate(frame_info["image"], detections)
        if self.kind == "images":
            write_image(self.directory / f"{ordinal:06d}.jpg", output)
            return
        if self.writer is None:
            fps = frame_info["fps"]
            if fps is None:
                raise ValueError("Video FPS is unknown; rerun without --save-preview.")
            height, width = output.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.directory / "annotated.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                fps / self.stride, (width, height),
            )
            if not self.writer.isOpened():
                raise RuntimeError("MJPG preview writer unavailable; rerun without --save-preview.")
        self.writer.write(output)

    def close(self):
        if self.writer is not None:
            self.writer.release()
