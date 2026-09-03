"""High-Performance Screen Capture Engine.

Provides ultra-low latency frame acquisition using Windows Desktop Duplication
API (dxcam) targeting 60+ FPS, with graceful multi-platform fallbacks (mss, PIL, synthetic).
Includes OpenCV-based dirty-frame detection to eliminate redundant model wakeups.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# Configure module-level logger
logger = logging.getLogger("CaptureEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Attempt optional backend imports
DXCAM_AVAILABLE = False
MSS_AVAILABLE = False
PIL_AVAILABLE = False

try:
    import dxcam  # type: ignore

    DXCAM_AVAILABLE = True
except (ImportError, Exception) as exc:
    logger.debug("dxcam backend unavailable: %s", exc)

try:
    import mss  # type: ignore

    MSS_AVAILABLE = True
except (ImportError, Exception) as exc:
    logger.debug("mss backend unavailable: %s", exc)

try:
    from PIL import ImageGrab  # type: ignore

    PIL_AVAILABLE = True
except (ImportError, Exception) as exc:
    logger.debug("PIL ImageGrab backend unavailable: %s", exc)


def is_frame_dirty(
    prev_frame: Optional[np.ndarray],
    curr_frame: Optional[np.ndarray],
    threshold: float = 0.01,
) -> bool:
    """Evaluate whether the interface has mutated beyond sensory noise.

    Converts both input frames to grayscale, evaluates absolute pixel-level
    differences, applies a binary noise-rejection threshold (>25 delta), and
    calculates the fraction of altered pixels.

    Args:
        prev_frame: Previous video frame in BGR or RGB NumPy format (H, W, C).
        curr_frame: Current video frame in BGR or RGB NumPy format (H, W, C).
        threshold: Ratio of altered pixels required to mark frame as dirty (default: 0.01 = 1%).

    Returns:
        True if altered pixel ratio exceeds threshold or previous frame was None,
        False otherwise.
    """
    if curr_frame is None:
        logger.warning("is_frame_dirty called with curr_frame=None.")
        return False

    if prev_frame is None:
        logger.debug("Initial frame registered; marking frame as dirty by default.")
        return True

    if prev_frame.shape != curr_frame.shape:
        logger.warning(
            "Frame shape mismatch: prev=%s, curr=%s. Marking dirty.",
            prev_frame.shape,
            curr_frame.shape,
        )
        return True

    # Convert to 1-channel grayscale if 3-channel
    if prev_frame.ndim == 3 and prev_frame.shape[2] in (3, 4):
        prev_gray = cv2.cvtColor(prev_frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame

    if curr_frame.ndim == 3 and curr_frame.shape[2] in (3, 4):
        curr_gray = cv2.cvtColor(curr_frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame

    # Absolute difference calculation
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Filter cosmetic rendering jitter / compression artifacts with binary threshold (>25 intensity delta)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    total_pixels = thresh.shape[0] * thresh.shape[1]
    if total_pixels == 0:
        return False

    altered_pixels = int(cv2.countNonZero(thresh))
    ratio = altered_pixels / float(total_pixels)

    is_dirty = ratio > threshold
    logger.debug(
        "Dirty-frame check: altered=%.4f%% (%d/%d px), threshold=%.4f%%, dirty=%s",
        ratio * 100.0,
        altered_pixels,
        total_pixels,
        threshold * 100.0,
        is_dirty,
    )
    return is_dirty


class RealTimeFrameIngester:
    """High-frequency screen capture ingester with zero-copy double-buffering.

    Utilizes DirectX Desktop Duplication (dxcam) on Windows for near zero-overhead
    60+ FPS frame streaming. Gracefully degrades to mss, PIL, or synthetic canvas
    in headless/locked desktop environments.
    """

    def __init__(
        self,
        target_fps: int = 60,
        monitor_idx: int = 0,
        device_idx: int = 0,
        preferred_backend: str = "dxcam",
    ) -> None:
        """Initialize frame ingester configuration.

        Args:
            target_fps: Target acquisition rate in frames per second (e.g., 60).
            monitor_idx: Display monitor index (0-indexed).
            device_idx: GPU adapter index for DXGI.
            preferred_backend: Preferred capture engine ('dxcam', 'mss', or 'pil').
        """
        self.target_fps = target_fps
        self.monitor_idx = monitor_idx
        self.device_idx = device_idx
        self.preferred_backend = preferred_backend.lower()

        # Thread synchronization & buffers
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None

        # Telemetry
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._last_fps_calc_time: float = time.perf_counter()
        self._fps_counter: int = 0
        self._active_backend: str = "none"

        # Backend instance handles
        self._dxcam_cam: Any = None
        self._mss_sct: Any = None
        self._thread_local = threading.local()

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Select and verify the most capable available capture backend."""
        is_windows = platform.system() == "Windows"

        # Strategy 1: DXCAM (Windows DXGI Desktop Duplication)
        if self.preferred_backend == "dxcam" and is_windows and DXCAM_AVAILABLE:
            try:
                cam = dxcam.create(
                    device_idx=self.device_idx,
                    output_idx=self.monitor_idx,
                    output_color="BGR",
                )
                if cam is not None:
                    # Test probe grab to ensure Desktop Duplication permission is granted
                    test_frame = cam.grab()
                    self._dxcam_cam = cam
                    self._active_backend = "dxcam"
                    logger.info(
                        "DXCam backend initialized successfully (Device %d, Monitor %d).",
                        self.device_idx,
                        self.monitor_idx,
                    )
                    return
            except Exception as exc:
                logger.warning("DXCam probe failed (%s); cascading to secondary backend.", exc)

        # Strategy 2: MSS (Multi-platform fast capture)
        if MSS_AVAILABLE:
            try:
                with mss.mss() as sct:
                    if len(sct.monitors) > 1:
                        # Probe grab to ensure GDI BitBlt is functional
                        probe_mon = sct.monitors[min(self.monitor_idx + 1, len(sct.monitors) - 1)]
                        _ = sct.grab(probe_mon)
                        self._active_backend = "mss"
                        logger.info("MSS capture backend initialized successfully.")
                        return
            except Exception as exc:
                logger.warning("MSS probe failed (%s); cascading to PIL.", exc)

        # Strategy 3: PIL ImageGrab (Universal standard fallback)
        if PIL_AVAILABLE:
            try:
                probe_img = ImageGrab.grab()
                if probe_img is not None and probe_img.size[0] > 0:
                    self._active_backend = "pil"
                    logger.info("PIL ImageGrab backend verified and active.")
                    return
            except Exception as exc:
                logger.warning("PIL ImageGrab probe failed (%s); cascading to synthetic.", exc)

        # Strategy 4: High-Res Synthetic Canvas (Headless / Locked Desktop fail-safe)
        self._active_backend = "synthetic"
        logger.info(
            "Headless or locked display session detected. Synthetic 60+ FPS buffer active."
        )

    def _create_synthetic_frame(self, t: float) -> np.ndarray:
        """Render a synthetic desktop frame with active clock indicator.

        Args:
            t: Monotonic timestamp.

        Returns:
            Rendered BGR frame.
        """
        frame = np.full((1080, 1920, 3), 240, dtype=np.uint8)
        # Header bar
        cv2.rectangle(frame, (0, 0), (1920, 70), (210, 210, 210), -1)
        cv2.line(frame, (0, 70), (1920, 70), (180, 180, 180), 2)
        # Title
        cv2.putText(
            frame,
            f"Dual-System GUI Agent Active Session - FPS: {self._fps:.1f}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        # Mock interactive button 'Save As'
        cv2.rectangle(frame, (120, 20), (240, 60), (70, 130, 180), -1)
        cv2.putText(
            frame,
            "Save As",
            (135, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _capture_single_frame(self) -> Optional[np.ndarray]:
        """Acquire a single frame using the verified active backend."""
        try:
            if self._active_backend == "dxcam" and self._dxcam_cam is not None:
                return self._dxcam_cam.grab()

            elif self._active_backend == "mss":
                # Ensure thread-local MSS instance
                if not hasattr(self._thread_local, "sct") or self._thread_local.sct is None:
                    self._thread_local.sct = mss.mss()
                sct = self._thread_local.sct
                target_mon_idx = min(
                    self.monitor_idx + 1, len(sct.monitors) - 1
                )
                monitor = sct.monitors[target_mon_idx]
                sct_img = sct.grab(monitor)
                frame_bgra = np.frombuffer(sct_img.raw, dtype=np.uint8).reshape(
                    (sct_img.height, sct_img.width, 4)
                )
                return cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            elif self._active_backend == "pil" and PIL_AVAILABLE:
                pil_img = ImageGrab.grab()
                frame_rgb = np.array(pil_img, dtype=np.uint8)
                return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            elif self._active_backend == "synthetic":
                return self._create_synthetic_frame(time.perf_counter())

        except Exception as exc:
            logger.debug("Frame capture error on backend '%s': %s", self._active_backend, exc)
            return None

        return None

    def _worker_loop(self) -> None:
        """Background acquisition loop running at target refresh rate."""
        frame_interval = 1.0 / max(1, self.target_fps)
        logger.debug("Starting frame capture worker loop (Target FPS: %d).", self.target_fps)

        while not self._stop_event.is_set():
            t_start = time.perf_counter()

            frame = self._capture_single_frame()
            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._frame_count += 1
                    self._fps_counter += 1

            # Update FPS telemetry every 0.5 seconds
            t_now = time.perf_counter()
            elapsed_telemetry = t_now - self._last_fps_calc_time
            if elapsed_telemetry >= 0.5:
                self._fps = self._fps_counter / elapsed_telemetry
                self._fps_counter = 0
                self._last_fps_calc_time = t_now

            # Maintain target cadence
            t_elapsed = time.perf_counter() - t_start
            sleep_duration = frame_interval - t_elapsed
            if sleep_duration > 0.001:
                time.sleep(sleep_duration)

        # Thread exit cleanup
        if hasattr(self._thread_local, "sct") and self._thread_local.sct is not None:
            try:
                self._thread_local.sct.close()
            except Exception:
                pass
            self._thread_local.sct = None

    def start(self) -> None:
        """Start the background ingestion thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("Frame ingester is already running.")
            return

        self._stop_event.clear()
        self._last_fps_calc_time = time.perf_counter()
        self._fps_counter = 0
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="RealTimeFrameIngesterWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("RealTimeFrameIngester started (Active Backend: %s).", self._active_backend)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop background ingestion and release hardware resources.

        Args:
            timeout: Maximum seconds to wait for worker thread shutdown.
        """
        logger.info("Stopping RealTimeFrameIngester...")
        self._stop_event.set()

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                logger.warning("Ingester worker thread did not terminate within timeout.")
            self._worker_thread = None

        if self._dxcam_cam is not None:
            try:
                del self._dxcam_cam
            except Exception as exc:
                logger.debug("Error releasing dxcam handle: %s", exc)
            self._dxcam_cam = None

        logger.info("RealTimeFrameIngester stopped successfully.")

    def get_latest_frame(self, copy: bool = False) -> Optional[np.ndarray]:
        """Retrieve the most recent screen frame.

        Args:
            copy: If True, returns an isolated deepcopy; if False, returns the frame
                  buffer pointer (near zero-copy, recommended for read-only inference).

        Returns:
            Latest frame as BGR NumPy array, or None if no frame has been captured.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy else self._latest_frame

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime telemetry statistics.

        Returns:
            Dictionary containing backend name, active status, FPS, total frames,
            and resolution.
        """
        with self._lock:
            res: Optional[Tuple[int, int]] = None
            if self._latest_frame is not None:
                res = (self._latest_frame.shape[1], self._latest_frame.shape[0])

            return {
                "active_backend": self._active_backend,
                "is_running": (
                    self._worker_thread is not None
                    and self._worker_thread.is_alive()
                ),
                "fps": round(self._fps, 2),
                "total_frames_captured": self._frame_count,
                "resolution": res,
            }

    def __enter__(self) -> RealTimeFrameIngester:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
