"""Giao dien thu nghiem nhan dien vat the bang YOLO."""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

IMAGE_RESAMPLING = getattr(Image, "Resampling", Image).LANCZOS

APP_TITLE = "THỊ GIÁC MÁY TÍNH  |  NHẬN DIỆN VẬT CẢN"
MAX_FRAME_WIDTH = 960
YOLO_DEFAULT_WEIGHTS = "yolo11n.pt"
YOLO_IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.15
BACKGROUND = "#101820"
PANEL = "#182630"
TEXT = "#e7f0f3"
MUTED = "#91a4ad"
ACCENT = "#45d6b5"
OBSTACLE_COLOR = (55, 105, 255)
FLOOR_COLOR = (80, 210, 135)
UNKNOWN_COLOR = (220, 220, 220)
FONT_PATHS = (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf")
MAX_NOTIFICATION_HISTORY = 200
RESULT_POLL_INTERVAL_MS = 5
SEGMENTATION_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs",
    "segment",
    "floor_seg_demo",
    "weights",
    "best.pt",
)


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bounds: tuple[int, int, int, int]
    track_id: int | None
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class InferenceJob:
    generation: int
    tracking_session: int
    frame: np.ndarray
    is_video: bool
    weights: str
    tracker_name: str
    confidence: float


@dataclass(frozen=True)
class InferenceResult:
    generation: int
    is_video: bool
    image: np.ndarray | None = None
    detections: tuple[Detection, ...] = ()
    inference_fps: float = 0.0
    error: str | None = None


def draw_yolo_detections(
    image: np.ndarray,
    result: Any,
    *,
    include_floor: bool = False,
    draw_boxes: bool = True,
) -> list[Detection]:
    """Extract YOLO boxes and optionally draw them, excluding floor by default."""
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return []

    detections: list[Detection] = []
    for box_index, box in enumerate(boxes):
        class_id = int(box.cls[0].item())
        class_name = (
            str(names.get(class_id, class_id))
            if isinstance(names, dict)
            else str(names[class_id])
        )
        if (
            not include_floor
            and class_name.strip().casefold() in {"san", "sàn", "floor", "ground"}
        ):
            continue

        bounds = tuple(int(value) for value in box.xyxy[0].cpu().tolist())
        x1, y1, x2, y2 = bounds
        confidence = float(box.conf[0].item())
        track_id = int(box.id[0].item()) if box.id is not None else None
        mask: np.ndarray | None = None
        masks = getattr(result, "masks", None)
        mask_polygons = getattr(masks, "xy", None)
        mask_data = getattr(masks, "data", None)
        if mask_polygons is not None and len(mask_polygons) > box_index:
            polygon = np.asarray(mask_polygons[box_index], dtype=np.float32)
            if polygon.ndim == 2 and len(polygon) >= 3:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(
                    mask,
                    [np.rint(polygon).astype(np.int32)],
                    color=1,
                )
                mask = mask.astype(bool)
        if mask is None and mask_data is not None and len(mask_data) > box_index:
            mask_array = mask_data[box_index].detach().cpu().numpy()
            mask = cv2.resize(
                mask_array.astype(np.float32),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0.5
        if draw_boxes:
            track_label = f" ID{track_id}" if track_id is not None else ""
            label = f"{class_name} {confidence:.0%}{track_label}"
            cv2.rectangle(image, (x1, y1), (x2, y2), OBSTACLE_COLOR, 2)
            cv2.putText(
                image, label, (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, OBSTACLE_COLOR, 2, cv2.LINE_AA,
            )
        detections.append(
            Detection(class_name, confidence, bounds, track_id, mask)
        )
    return detections


@lru_cache(maxsize=8)
def _load_overlay_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.truetype("DejaVuSans.ttf", size=size)


def render_detection_view(
    image: np.ndarray,
    detections: tuple[Detection, ...],
    *,
    classification_view: bool,
) -> np.ndarray:
    rendered = image.copy()
    visible_detections = [
        detection for detection in detections
        if classification_view
        or detection.class_name.strip().casefold()
        not in {"san", "sàn", "floor", "ground"}
    ]

    if classification_view:
        tint = rendered.copy()
        for detection in sorted(
            visible_detections,
            key=lambda item: item.class_name.strip().casefold()
            not in {"san", "sàn", "floor", "ground"},
        ):
            color = (
                FLOOR_COLOR
                if detection.class_name.strip().casefold()
                in {"san", "sàn", "floor", "ground"}
                else OBSTACLE_COLOR
            )
            if detection.mask is not None:
                tint[detection.mask] = color
            else:
                cv2.rectangle(
                    tint, detection.bounds[:2], detection.bounds[2:], color, -1
                )
        rendered = cv2.addWeighted(tint, 0.22, rendered, 0.78, 0)

    labels: list[tuple[int, int, str, tuple[int, int, int]]] = []
    for detection in visible_detections:
        x1, y1, x2, y2 = detection.bounds
        is_floor = detection.class_name.strip().casefold() in {
            "san", "sàn", "floor", "ground"
        }
        color = FLOOR_COLOR if classification_view and is_floor else OBSTACLE_COLOR
        track_label = (
            f" ID{detection.track_id}" if detection.track_id is not None else ""
        )
        label = f"{detection.class_name} {detection.confidence:.0%}{track_label}"
        if detection.mask is None or not classification_view:
            cv2.rectangle(
                rendered, (x1, y1), (x2, y2), color,
                3 if classification_view else 2,
            )
        text_y = max(0, y1 - 25)
        cv2.rectangle(
            rendered, (x1, text_y), (min(rendered.shape[1] - 1, x1 + 320), y1),
            color, -1,
        )
        labels.append((x1 + 4, text_y + 2, label, color))

    legend_lines: list[tuple[str, tuple[int, int, int]]] = []
    if classification_view:
        has_masks = any(detection.mask is not None for detection in visible_detections)
        if has_masks:
            legend_lines = [
                ("Vật cản: mask model", OBSTACLE_COLOR),
                ("Sàn: mask model (nếu được phát hiện)", FLOOR_COLOR),
                ("Ngoài mask: nền / chưa phân loại", UNKNOWN_COLOR),
            ]
        else:
            legend_lines = [
                ("Phát hiện: box model", OBSTACLE_COLOR),
                ("Model này không xuất mask phân vùng", UNKNOWN_COLOR),
                ("Ngoài box: nền / chưa phân loại", UNKNOWN_COLOR),
            ]
        legend_width = max(0, min(rendered.shape[1] - 16, 390))
        legend_height = 18 + len(legend_lines) * 23
        cv2.rectangle(
            rendered, (8, 8), (8 + legend_width, 8 + legend_height),
            (18, 30, 40), -1,
        )
    if labels or legend_lines:
        pil_image = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        for x, y, text, _ in labels:
            draw.text(
                (x, y), text, font=_load_overlay_font(16),
                fill=(255, 255, 255),
            )
        for index, (text, color) in enumerate(legend_lines):
            y = 21 + index * 23
            rgb_color = (color[2], color[1], color[0])
            draw.rectangle((18, y - 8, 30, y + 4), fill=rgb_color)
            draw.text(
                (38, y - 10), text, font=_load_overlay_font(14),
                fill=(240, 245, 248),
            )
        rendered = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    return rendered


class ObstacleDetectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1220x780")
        self.root.minsize(860, 580)
        self.root.configure(bg=BACKGROUND)

        self.frame: np.ndarray | None = None
        self.render_image: np.ndarray | None = None
        self.latest_inference_image: np.ndarray | None = None
        self.latest_detections: tuple[Detection, ...] = ()
        self.current_source_name = ""
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.video: cv2.VideoCapture | None = None
        self.video_after_id: str | None = None
        self.video_playing = False
        self.video_inference_pending = False
        self.video_interval_ms = 33
        self.yolo_weights = YOLO_DEFAULT_WEIGHTS
        self.processing_fps = 0.0
        self.confidence = tk.DoubleVar(value=YOLO_CONFIDENCE)
        self.confidence_text = tk.StringVar(value=f"{YOLO_CONFIDENCE:.2f}")
        self.classification_view = tk.BooleanVar(value=False)
        self.model_text = tk.StringVar(value=f"Model: {YOLO_DEFAULT_WEIGHTS}")
        self.video_button_text = tk.StringVar(value="Chọn video")
        self.use_yolo = tk.BooleanVar(value=True)
        self.tracker_name = tk.StringVar(value="ByteTrack")
        self.weights_text = tk.StringVar(value=YOLO_DEFAULT_WEIGHTS)
        self.status_text = tk.StringVar(value="Mở ảnh hoặc video để bắt đầu.")
        self.metric_text = tk.StringVar(value="CHƯA CÓ DỮ LIỆU")
        self.notifications: list[tuple[str, str]] = []
        self.notified_track_ids: set[int] = set()
        self.notified_video_objects: list[
            tuple[str, tuple[int, int, int, int], float]
        ] = []
        self.notified_image_boxes: set[tuple[str, int, int, int, int]] = set()
        self._inference_jobs: queue.Queue[InferenceJob | None] = queue.Queue(maxsize=1)
        self._inference_results: queue.Queue[InferenceResult] = queue.Queue()
        self._inference_generation = 0
        self._tracking_session = 0
        self._closing = False
        self._inference_thread = threading.Thread(
            target=self._inference_worker,
            name="yolo-inference",
            daemon=True,
        )
        self._inference_thread.start()

        self._configure_style()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(RESULT_POLL_INTERVAL_MS, self._poll_inference_results)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TFrame", background=BACKGROUND
        )
        style.configure(
            "Panel.TFrame", background=PANEL
        )
        style.configure(
            "TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 10)
        )
        style.configure(
            "Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10)
        )
        style.configure(
            "Panel.TCheckbutton", background=PANEL, foreground=TEXT,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Title.TLabel", background=BACKGROUND, foreground=TEXT,
            font=("Segoe UI Semibold", 18),
        )
        style.configure(
            "Muted.TLabel", background=PANEL, foreground=MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Accent.TButton", font=("Segoe UI Semibold", 10), padding=(12, 9),
            background=ACCENT, foreground="#10232a",
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#72e7ce"), ("disabled", "#54736e")],
        )
        style.configure(
            "Secondary.TButton", font=("Segoe UI", 10), padding=(10, 8),
            background="#263943", foreground=TEXT,
        )
        style.map("Secondary.TButton", background=[("active", "#354f5a")])
        style.configure("TNotebook", background=PANEL, borderwidth=0)
        style.configure(
            "TNotebook.Tab", padding=(12, 7), background="#263943",
            foreground=TEXT,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", ACCENT)],
            foreground=[("selected", "#10232a")],
        )
        style.configure(
            "History.Treeview", background="#0b1116", fieldbackground="#0b1116",
            foreground=TEXT, rowheight=28,
        )
        style.configure(
            "History.Treeview.Heading", background=PANEL, foreground=TEXT,
        )

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(18, 14, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(side="left")
        ttk.Label(
            header, text="YOLO  •  OPENCV", foreground=ACCENT
        ).pack(side="right", padx=4)

        body = ttk.Frame(self.root, padding=(14, 6, 14, 14))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_preview_panel(body)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        sidebar = ttk.Frame(parent, style="Panel.TFrame", padding=14, width=250)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        sidebar.grid_propagate(False)

        ttk.Label(
            sidebar, text="DỮ LIỆU", style="Muted.TLabel"
        ).pack(anchor="w", pady=(2, 10))
        ttk.Button(
            sidebar, text="Chọn ảnh", style="Accent.TButton",
            command=self._open_image,
        ).pack(fill="x", pady=4)
        ttk.Button(
            sidebar, text="Chọn video", style="Secondary.TButton",
            command=self._open_video,
        ).pack(fill="x", pady=4)
        ttk.Button(
            sidebar, text="Mở ảnh mẫu", style="Secondary.TButton",
            command=self._open_sample,
        ).pack(fill="x", pady=4)

        ttk.Separator(sidebar).pack(fill="x", pady=10)
        ttk.Label(sidebar, text="NHẬN DIỆN", style="Muted.TLabel").pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Checkbutton(
            sidebar,
            text="Bật nhận diện YOLO",
            variable=self.use_yolo,
            command=self._on_yolo_toggle,
            style="Panel.TCheckbutton",
        ).pack(anchor="w", pady=3)
        ttk.Button(
            sidebar, text="Cấu hình model / tracker", style="Secondary.TButton",
            command=self._open_yolo_settings,
        ).pack(fill="x", pady=3)
        ttk.Label(
            sidebar, textvariable=self.model_text, style="Muted.TLabel",
            wraplength=215,
        ).pack(anchor="w", pady=(4, 2))
        ttk.Label(
            sidebar, text="Dùng weights đã train để nhận diện đúng vật thể cần tìm.",
            style="Muted.TLabel", wraplength=225,
        ).pack(anchor="w", pady=(0, 2))
        ttk.Checkbutton(
            sidebar,
            text="Chế độ phân vùng (thử nghiệm)",
            variable=self.classification_view,
            command=self._on_classification_view_toggle,
            style="Panel.TCheckbutton",
        ).pack(anchor="w", pady=(5, 2))
        ttk.Button(
            sidebar,
            text="Dùng model sàn + vật cản",
            style="Secondary.TButton",
            command=self._use_segmentation_model,
        ).pack(fill="x", pady=(2, 4))

        ttk.Separator(sidebar).pack(fill="x", pady=12)
        ttk.Label(sidebar, text="VIDEO", style="Muted.TLabel").pack(
            anchor="w", pady=(0, 4)
        )
        video_buttons = ttk.Frame(sidebar, style="Panel.TFrame")
        video_buttons.pack(fill="x")
        ttk.Button(
            video_buttons, textvariable=self.video_button_text,
            style="Secondary.TButton",
            command=self._toggle_video,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4), pady=4)
        ttk.Button(
            video_buttons, text="Đóng video", style="Secondary.TButton",
            command=self._stop_video,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0), pady=4)

        ttk.Separator(sidebar).pack(fill="x", pady=16)
        ttk.Label(
            sidebar, textvariable=self.metric_text, style="Panel.TLabel",
            font=("Segoe UI Semibold", 12),
        ).pack(anchor="w")
        ttk.Label(
            sidebar, textvariable=self.status_text, style="Muted.TLabel",
            wraplength=225, justify="left",
        ).pack(anchor="w", pady=(8, 2))
        ttk.Label(
            sidebar, text="Bản thử nghiệm, không dùng để điều khiển xe thật.",
            style="Muted.TLabel", wraplength=225, justify="left",
        ).pack(anchor="w", side="bottom", pady=(12, 0))

    def _build_preview_panel(self, parent: ttk.Frame) -> None:
        preview_panel = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        preview_panel.grid(row=0, column=1, sticky="nsew")
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(1, weight=1)
        self.content_tabs = ttk.Notebook(preview_panel)
        self.content_tabs.grid(row=1, column=0, sticky="nsew")

        image_tab = ttk.Frame(self.content_tabs, style="Panel.TFrame", padding=8)
        image_tab.rowconfigure(0, weight=1)
        image_tab.columnconfigure(0, weight=1)
        self.content_tabs.add(image_tab, text="Khung hình")
        self.canvas = tk.Canvas(image_tab, bg="#0b1116", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._render_frame)

        self.history_tab = ttk.Frame(self.content_tabs, style="Panel.TFrame", padding=8)
        self.content_tabs.add(self.history_tab, text="Lịch sử (0)")
        self.history_tab.rowconfigure(0, weight=1)
        self.history_tab.columnconfigure(0, weight=1)
        self.history_tree = ttk.Treeview(
            self.history_tab,
            columns=("time", "message"),
            show="headings",
            style="History.Treeview",
        )
        self.history_tree.heading("time", text="Thời gian")
        self.history_tree.heading("message", text="Thông báo")
        self.history_tree.column("time", width=90, stretch=False, anchor="center")
        self.history_tree.column("message", width=500, stretch=True)
        scrollbar = ttk.Scrollbar(
            self.history_tab, orient="vertical", command=self.history_tree.yview
        )
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        self.history_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn ảnh cần phân tích",
            filetypes=[
                ("Ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if not path:
            return

        self._load_image(path)

    def _open_sample(self) -> None:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = (
            os.path.join(project_dir, "anhhome.jpg"),
            os.path.join(project_dir, "anh", "anhhome.jpg"),
        )
        path = next(
            (candidate for candidate in candidates if os.path.isfile(candidate)), None
        )
        if path is None:
            messagebox.showerror(
                "Không tìm thấy ảnh mẫu",
                "Không tìm thấy anhhome.jpg trong thư mục chương trình hoặc thư mục anh.",
                parent=self.root,
            )
            return
        self._load_image(path)

    def _load_image(self, path: str) -> None:
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
        except OSError as error:
            messagebox.showerror("Không mở được ảnh", str(error), parent=self.root)
            return
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            messagebox.showerror(
                "Không mở được ảnh", "Định dạng ảnh không được hỗ trợ.",
                parent=self.root,
            )
            return
        self._release_video()
        self._invalidate_inference(reset_tracker=True)
        self.processing_fps = 0.0
        self._reset_notification_keys()
        self._set_frame(frame, os.path.basename(path))

    def _open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn video cần phân tích",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if not path:
            return
        self._release_video()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            messagebox.showerror(
                "Không mở được video", "Không thể đọc tệp video này.",
                parent=self.root,
            )
            return
        self._invalidate_inference(reset_tracker=True)
        self.processing_fps = 0.0
        self._reset_notification_keys()
        self.video = capture
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.video_interval_ms = max(1, int(1000 / fps)) if fps > 0 else 33
        self.video_playing = True
        self.video_button_text.set("Tạm dừng")
        self.status_text.set(os.path.basename(path))
        self._read_video_frame()

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width > MAX_FRAME_WIDTH:
            new_height = max(1, int(height * MAX_FRAME_WIDTH / width))
            return cv2.resize(frame, (MAX_FRAME_WIDTH, new_height))
        return frame

    def _set_frame(self, frame: np.ndarray, source_name: str | None = None) -> None:
        self.frame = self._prepare_frame(frame)
        if source_name:
            self.current_source_name = source_name
            self.status_text.set(source_name)
        self._analyze_current_frame()

    def _read_video_frame(self) -> None:
        self.video_after_id = None
        if (
            not self.video_playing
            or self.video is None
            or self.video_inference_pending
        ):
            return
        frame_started = time.perf_counter()
        ok, frame = self.video.read()
        if not ok:
            self._stop_video()
            self.status_text.set("Đã phát hết video.")
            return
        self.frame = self._prepare_frame(frame)
        self._analyze_current_frame()
        elapsed_ms = int((time.perf_counter() - frame_started) * 1000)
        delay_ms = max(1, self.video_interval_ms - elapsed_ms)
        if not self.video_inference_pending:
            self.video_after_id = self.root.after(delay_ms, self._read_video_frame)

    def _analyze_current_frame(self) -> None:
        if self.frame is None:
            return
        if not self.use_yolo.get():
            self.render_image = self.frame
            self.latest_inference_image = self.frame
            self.latest_detections: tuple[Detection, ...] = ()
            self.metric_text.set("VẬT THỂ: 0\nYOLO FPS: —")
            self._render_latest_detection_view()
            return

        self._submit_inference(self.frame, is_video=self.video is not None)
        if self.video is None:
            self.status_text.set("Đang phân tích ảnh...")
            self.render_image = self.frame
            self.latest_inference_image = self.frame
            self.latest_detections = ()
            self._render_latest_detection_view()

    def _submit_inference(self, frame: np.ndarray, *, is_video: bool) -> None:
        job = InferenceJob(
            generation=self._inference_generation,
            tracking_session=self._tracking_session,
            frame=frame.copy(),
            is_video=is_video,
            weights=self.yolo_weights,
            tracker_name=self.tracker_name.get(),
            confidence=self.confidence.get(),
        )
        if self._inference_jobs.full():
            try:
                self._inference_jobs.get_nowait()
            except queue.Empty:
                pass
        self._inference_jobs.put_nowait(job)
        if is_video:
            self.video_inference_pending = True

    def _inference_worker(self) -> None:
        model: Any | None = None
        model_key: tuple[str, str, int] | None = None
        while True:
            job = self._inference_jobs.get()
            if job is None:
                return

            key = (job.weights, job.tracker_name, job.tracking_session)
            try:
                if model is None or model_key != key:
                    yolo = import_module("ultralytics")
                    model = yolo.YOLO(job.weights)
                    model_key = key

                overlay = job.frame.copy()
                inference_started = time.perf_counter()
                if job.is_video:
                    tracker_file = (
                        "botsort.yaml"
                        if job.tracker_name == "BoT-SORT"
                        else "bytetrack.yaml"
                    )
                    results = model.track(
                        job.frame,
                        persist=True,
                        tracker=tracker_file,
                        conf=job.confidence,
                        imgsz=YOLO_IMAGE_SIZE,
                        verbose=False,
                    )
                else:
                    results = model.predict(
                        job.frame,
                        conf=job.confidence,
                        imgsz=YOLO_IMAGE_SIZE,
                        verbose=False,
                    )

                detections: list[Detection] = []
                if results:
                    detections = draw_yolo_detections(
                        overlay,
                        results[0],
                        include_floor=True,
                        draw_boxes=False,
                    )
                inference_fps = 1.0 / max(
                    time.perf_counter() - inference_started, 1e-6
                )
                self._inference_results.put(
                    InferenceResult(
                        generation=job.generation,
                        is_video=job.is_video,
                        image=job.frame.copy(),
                        detections=tuple(detections),
                        inference_fps=inference_fps,
                    )
                )
            except Exception as error:
                model = None
                model_key = None
                self._inference_results.put(
                    InferenceResult(
                        generation=job.generation,
                        is_video=job.is_video,
                        error=str(error),
                    )
                )

    def _poll_inference_results(self) -> None:
        latest_result: InferenceResult | None = None
        valid_results: list[InferenceResult] = []
        while True:
            try:
                result = self._inference_results.get_nowait()
            except queue.Empty:
                break
            if result.generation != self._inference_generation:
                continue
            if result.error is not None:
                if result.is_video:
                    self.video_inference_pending = False
                self._inference_generation += 1
                self.use_yolo.set(False)
                self.status_text.set(f"YOLO không chạy được: {result.error}")
                messagebox.showerror(
                    "Không thể chạy YOLO",
                    f"{result.error}\n\nKiểm tra thư viện Ultralytics và file weights đã chọn.",
                    parent=self.root,
                )
                latest_result = None
                valid_results.clear()
                break
            if result.is_video:
                self.video_inference_pending = False
            valid_results.append(result)
            latest_result = result

        if latest_result is not None and latest_result.image is not None:
            for result in valid_results:
                self._record_notifications(
                    list(result.detections), is_video=result.is_video
                )
            self.processing_fps = (
                latest_result.inference_fps
                if self.processing_fps == 0
                else self.processing_fps * 0.8 + latest_result.inference_fps * 0.2
            )
            self.latest_inference_image = latest_result.image
            self.latest_detections = latest_result.detections
            self._render_latest_detection_view()
            self.metric_text.set(
                f"VẬT THỂ: {sum(not self._is_floor_detection(item) for item in latest_result.detections)}\n"
                f"YOLO FPS: {self.processing_fps:.1f}"
            )
            if not latest_result.is_video:
                self.status_text.set(self.current_source_name)

        if not self._closing:
            if (
                self.video_playing
                and self.video is not None
                and not self.video_inference_pending
                and self.video_after_id is None
            ):
                inference_ms = (
                    1000 / latest_result.inference_fps
                    if latest_result is not None and latest_result.inference_fps > 0
                    else self.video_interval_ms
                )
                remaining_frame_delay = max(
                    1, int(self.video_interval_ms - inference_ms)
                )
                self.video_after_id = self.root.after(
                    remaining_frame_delay, self._read_video_frame
                )
            self.root.after(RESULT_POLL_INTERVAL_MS, self._poll_inference_results)

    def _record_notifications(
        self, detections: list[Detection], *, is_video: bool
    ) -> None:
        for detection in detections:
            if self._is_floor_detection(detection):
                continue
            if is_video:
                if (
                    detection.track_id is not None
                    and detection.track_id in self.notified_track_ids
                ):
                    continue
                if detection.track_id is not None:
                    self.notified_track_ids.add(detection.track_id)
                now = time.monotonic()
                self.notified_video_objects = [
                    item for item in self.notified_video_objects
                    if now - item[2] <= 5.0
                ]
                duplicate_index = next(
                    (
                        index
                        for index, (class_name, bounds, _) in enumerate(
                            self.notified_video_objects
                        )
                        if class_name == detection.class_name
                        and self._boxes_refer_to_same_object(bounds, detection.bounds)
                    ),
                    None,
                )
                if duplicate_index is not None:
                    self.notified_video_objects[duplicate_index] = (
                        detection.class_name,
                        detection.bounds,
                        now,
                    )
                    continue
                self.notified_video_objects.append(
                    (detection.class_name, detection.bounds, now)
                )
                source = (
                    f"ID {detection.track_id}"
                    if detection.track_id is not None
                    else "chưa tracking"
                )
            else:
                key = (detection.class_name, *detection.bounds)
                if key in self.notified_image_boxes:
                    continue
                self.notified_image_boxes.add(key)
                source = "Ảnh"

            timestamp = time.strftime("%H:%M:%S")
            message = (
                f"Phát hiện {detection.class_name} "
                f"({detection.confidence:.0%}) · {source}"
            )
            self.notifications.append((timestamp, message))
            self.history_tree.insert("", 0, values=(timestamp, message))
            if len(self.notifications) > MAX_NOTIFICATION_HISTORY:
                self.notifications.pop(0)
                oldest = self.history_tree.get_children()[-1]
                self.history_tree.delete(oldest)
            self.content_tabs.tab(
                self.history_tab, text=f"Lịch sử ({len(self.notifications)})"
            )

    @staticmethod
    def _is_floor_detection(detection: Detection) -> bool:
        return detection.class_name.strip().casefold() in {
            "san", "sàn", "floor", "ground"
        }

    @staticmethod
    def _boxes_refer_to_same_object(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> bool:
        x1, y1, x2, y2 = first
        a1, b1, a2, b2 = second
        intersection_width = max(0, min(x2, a2) - max(x1, a1))
        intersection_height = max(0, min(y2, b2) - max(y1, b1))
        intersection = intersection_width * intersection_height
        first_area = max(0, x2 - x1) * max(0, y2 - y1)
        second_area = max(0, a2 - a1) * max(0, b2 - b1)
        union = first_area + second_area - intersection
        if union and intersection / union >= 0.3:
            return True

        first_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        second_center = ((a1 + a2) / 2, (b1 + b2) / 2)
        center_distance = (
            (first_center[0] - second_center[0]) ** 2
            + (first_center[1] - second_center[1]) ** 2
        ) ** 0.5
        first_diagonal = (max(0, x2 - x1) ** 2 + max(0, y2 - y1) ** 2) ** 0.5
        second_diagonal = (max(0, a2 - a1) ** 2 + max(0, b2 - b1) ** 2) ** 0.5
        return center_distance <= max(24, min(first_diagonal, second_diagonal) * 0.5)

    def _reset_notification_keys(self) -> None:
        self.notified_track_ids.clear()
        self.notified_video_objects.clear()
        self.notified_image_boxes.clear()

    def _invalidate_inference(self, *, reset_tracker: bool = False) -> None:
        self._inference_generation += 1
        self.video_inference_pending = False
        if reset_tracker:
            self._tracking_session += 1

    def _on_yolo_toggle(self) -> None:
        self._invalidate_inference(reset_tracker=True)
        self._reset_notification_keys()
        if self.frame is not None:
            self._analyze_current_frame()
        elif self.use_yolo.get():
            self.status_text.set(
                f"YOLO sẽ dùng {os.path.basename(self.yolo_weights)} khi mở ảnh/video."
            )

    def _on_classification_view_toggle(self) -> None:
        self._render_latest_detection_view()

    def _use_segmentation_model(self) -> None:
        if not os.path.isfile(SEGMENTATION_WEIGHTS):
            messagebox.showerror(
                "Không tìm thấy model phân vùng",
                "Chưa có weights segmentation.\n"
                "Hãy chạy train_yolo_seg.py trước.",
                parent=self.root,
            )
            return

        self.classification_view.set(True)
        self.yolo_weights = SEGMENTATION_WEIGHTS
        self.weights_text.set(SEGMENTATION_WEIGHTS)
        self.model_text.set("Model: floor_seg_demo (sàn + vật cản)")
        self._invalidate_inference(reset_tracker=True)
        self._reset_notification_keys()
        self.status_text.set("Đã chọn model segmentation sàn + vật cản.")
        self._reanalyze_if_paused()

    def _render_latest_detection_view(self) -> None:
        if self.latest_inference_image is None:
            if self.frame is not None:
                self._render_frame(image=self.frame)
            return
        self.render_image = render_detection_view(
            self.latest_inference_image,
            self.latest_detections,
            classification_view=self.classification_view.get(),
        )
        self._render_frame(image=self.render_image)

    def _open_yolo_settings(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Cấu hình YOLO và tracker")
        dialog.geometry("460x290")
        dialog.resizable(False, False)
        dialog.configure(bg=PANEL)
        dialog.transient(self.root)
        dialog.grab_set()

        content = ttk.Frame(dialog, style="Panel.TFrame", padding=18)
        content.pack(fill="both", expand=True)
        ttk.Label(
            content, text="Weights đang chọn:", style="Panel.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            content, textvariable=self.weights_text, style="Muted.TLabel",
            wraplength=390,
        ).pack(anchor="w", pady=(2, 8))
        ttk.Button(
            content, text="Chọn file .pt", style="Secondary.TButton",
            command=self._select_yolo_weights,
        ).pack(anchor="w")
        tracker_row = ttk.Frame(content, style="Panel.TFrame")
        tracker_row.pack(fill="x", pady=(10, 4))
        ttk.Label(
            tracker_row, text="Tracker video:", style="Panel.TLabel"
        ).pack(side="left")
        tracker_picker = ttk.Combobox(
            tracker_row,
            textvariable=self.tracker_name,
            values=("ByteTrack", "BoT-SORT"),
            state="readonly",
            width=14,
        )
        tracker_picker.pack(side="right")
        tracker_picker.bind("<<ComboboxSelected>>", self._on_tracker_change)
        confidence_row = ttk.Frame(content, style="Panel.TFrame")
        confidence_row.pack(fill="x", pady=(8, 0))
        ttk.Label(
            confidence_row, text="Độ nhạy (confidence):", style="Panel.TLabel"
        ).pack(side="left")
        ttk.Label(
            confidence_row, textvariable=self.confidence_text, style="Panel.TLabel"
        ).pack(side="right")
        confidence_scale = ttk.Scale(
            content,
            variable=self.confidence,
            from_=0.05,
            to=0.75,
            command=self._update_confidence_label,
        )
        confidence_scale.pack(fill="x", pady=(2, 0))
        confidence_scale.bind("<ButtonRelease-1>", self._on_confidence_change)
        confidence_scale.bind("<KeyRelease>", self._on_confidence_change)
        ttk.Label(
            content,
            text="Ngưỡng mặc định thấp để ưu tiên bắt vật; có thể tăng báo nhầm. yolo11n.pt chỉ nhận diện các lớp phổ thông đã học, không nhận ra mọi vật thể.",
            style="Muted.TLabel", wraplength=390,
        ).pack(anchor="w", pady=(8, 0))

    def _update_confidence_label(self, value: str) -> None:
        self.confidence_text.set(f"{float(value):.2f}")

    def _on_confidence_change(self, _event: tk.Event) -> None:
        self._invalidate_inference()
        self._reanalyze_if_paused()

    def _select_yolo_weights(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn YOLO weights",
            filetypes=[("YOLO weights", "*.pt"), ("Tất cả tệp", "*.*")],
        )
        if not path:
            return
        self.yolo_weights = path
        self.weights_text.set(path)
        self.model_text.set(f"Model: {os.path.basename(path)}")
        self._invalidate_inference(reset_tracker=True)
        self._reset_notification_keys()
        self.status_text.set(f"Weights: {os.path.basename(path)}")
        self._reanalyze_if_paused()

    def _on_tracker_change(self, _event: tk.Event) -> None:
        self._invalidate_inference(reset_tracker=True)
        self.notified_track_ids.clear()
        self._reanalyze_if_paused()

    def _render_frame(self, _event: tk.Event | None = None, image: np.ndarray | None = None) -> None:
        if image is None:
            if self.frame is None:
                return
            image = self.render_image
            if image is None:
                image = self.frame
        canvas_width = max(1, _event.width) if _event is not None else max(
            1, self.canvas.winfo_width()
        )
        canvas_height = max(1, _event.height) if _event is not None else max(
            1, self.canvas.winfo_height()
        )
        height, width = image.shape[:2]
        scale = min(canvas_width / width, canvas_height / height)
        display_width = max(1, int(width * scale))
        display_height = max(1, int(height * scale))
        left = (canvas_width - display_width) // 2
        top = (canvas_height - display_height) // 2
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(rgb).resize(
            (display_width, display_height), IMAGE_RESAMPLING
        )
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(left, top, anchor="nw", image=self.preview_photo)

    def _reanalyze_if_paused(self) -> None:
        if self.frame is not None and not self.video_playing:
            self._analyze_current_frame()

    def _toggle_video(self) -> None:
        if self.video is None:
            self.status_text.set("Hãy chọn video trước.")
            return
        self.video_playing = not self.video_playing
        if self.video_playing:
            self.video_button_text.set("Tạm dừng")
            self._read_video_frame()
        elif self.video_after_id is not None:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None
            self.video_button_text.set("Phát tiếp")
        self.status_text.set("Video đang phát." if self.video_playing else "Video tạm dừng.")

    def _stop_video(self) -> None:
        self.video_playing = False
        self._release_video()

    def _release_video(self) -> None:
        if self.video_after_id is not None:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None
        self.video_playing = False
        if self.video is not None:
            self.video.release()
            self.video = None
        self.video_button_text.set("Chọn video")

    def _close(self) -> None:
        self._closing = True
        self._release_video()
        if self._inference_jobs.full():
            try:
                self._inference_jobs.get_nowait()
            except queue.Empty:
                pass
        self._inference_jobs.put_nowait(None)
        self.root.destroy()


def main() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(
            "Không thể khởi tạo giao diện Tkinter.\n"
            "Kiểm tra xem máy bạn có desktop GUI và Tkinter đã được cài đúng chưa.\n"
            f"Chi tiết: {exc}"
        ) from exc

    ObstacleDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()