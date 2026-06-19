# ============================================================================
# This file is derived from the ZZZSoundTrigger project.
# Original Author: ImLaoBJie
# Repository: https://github.com/ImLaoBJie/ZZZSoundTrigger
# License: GNU General Public License v3.0 (GPL-3.0)
#
# This file has been modified for integration into the ok-nte project.
# ============================================================================
import threading
import time
import warnings
from typing import Optional, cast

import librosa
import numpy as np
from scipy.signal import butter, correlate, filtfilt
from sklearn.preprocessing import scale

from ok import Logger
from src.sound_trigger.capture import MODE_PROCESS, AudioCaptureSource, create_capture_source
from src.sound_trigger.capture.base import CAPTURE_SAMPLE_RATE

warnings.filterwarnings("ignore", message="data discontinuity in recording")

logger = Logger.get_logger(__name__)


class SoundListener:
    used_sr = CAPTURE_SAMPLE_RATE
    sample_len = 0.2
    detection_interval = 0.025
    log_interval = 20
    default_process_name = "HTGame.exe"

    degree = 4
    cut_off = 1000

    def __init__(
        self,
        sample_path: str,
        counter_attack_sample_path: str,
        threshold: float = 0.13,
        counter_attack_threshold: float = 0.12,
        expansion_ratio: float = 1.0,
        is_allow_successive_trigger: bool = False,
        process_name: str = default_process_name,
    ):
        self.sample_path = sample_path
        self.counter_attack_sample_path = counter_attack_sample_path
        self.threshold = threshold
        self.counter_attack_threshold = counter_attack_threshold
        self.expansion_ratio = expansion_ratio
        self.is_allow_successive_trigger = is_allow_successive_trigger
        self.process_name = process_name

        self.is_computation_required = None
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None
        self._last_trigger_time = 0.0
        self._trigger_interval = 0.5

        self._sample_waveform = None
        self._counter_sample_waveform = None
        self._b = None
        self._a = None

        self.on_dodge_triggered = None
        self.on_counter_triggered = None
        self._capture: Optional[AudioCaptureSource] = None

        self._load_samples()

    def _load_samples(self):
        try:
            self._b, self._a = cast(
                tuple[np.ndarray, np.ndarray],
                butter(
                    self.degree,
                    self.cut_off,
                    btype="highpass",
                    output="ba",
                    fs=self.used_sr,
                ),
            )

            self._sample_waveform = self._normalize_waveform(
                self._load_and_cache(self.sample_path)
            )
            if self.counter_attack_sample_path:
                self._counter_sample_waveform = self._normalize_waveform(
                    self._load_and_cache(self.counter_attack_sample_path)
                )

            logger.info(f"Sound samples loaded: {self.used_sr}Hz")
        except Exception as e:
            message = (
                "Failed to load sound samples: "
                f"dodge={self.sample_path}, counter={self.counter_attack_sample_path}: {e}"
            )
            logger.error(message)
            raise RuntimeError(message) from e

    def _load_and_cache(self, path: str):
        import os

        cache_path = f"{path}_{self.used_sr}_{self.degree}_{self.cut_off}.npy"

        if os.path.exists(cache_path) and os.path.exists(path):
            if os.path.getmtime(cache_path) > os.path.getmtime(path):
                return np.load(cache_path)

        waveform, _ = librosa.load(path, sr=self.used_sr)
        waveform = self._filtering(waveform)
        np.save(cache_path, waveform)
        return waveform

    def _filtering(self, waveform):
        return filtfilt(self._b, self._a, waveform)

    @staticmethod
    def _normalize_waveform(waveform: np.ndarray) -> np.ndarray:
        return scale(waveform, with_mean=False)

    def _prepare_stream_waveform(self, stream_waveform: np.ndarray) -> np.ndarray:
        return self._normalize_waveform(self._filtering(stream_waveform))

    def matching(self, stream_waveform: np.ndarray, sample_waveform: np.ndarray):
        norm_stream_waveform = self._prepare_stream_waveform(stream_waveform)
        return self._match_normalized(norm_stream_waveform, sample_waveform)

    def _match_normalized(self, norm_stream_waveform: np.ndarray, norm_sample_waveform: np.ndarray):
        if norm_stream_waveform.shape[0] > norm_sample_waveform.shape[0]:
            correlation = (
                correlate(norm_stream_waveform, norm_sample_waveform, mode="same", method="fft")
                / norm_stream_waveform.shape[0]
            )
        else:
            correlation = (
                correlate(norm_sample_waveform, norm_stream_waveform, mode="same", method="fft")
                / norm_sample_waveform.shape[0]
            )

        max_corr = np.max(correlation) * self.expansion_ratio

        return max_corr

    def start(self):
        if self._running:
            logger.warning("SoundListener already running")
            return
        self._running = True
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()
        logger.info("SoundListener started successfully")

    def stop(self):
        logger.info(f"SoundListener stop called, current running: {self._running}")
        self._running = False
        if self._capture:
            self._capture.stop()
        if self._listener_thread:
            self._listener_thread.join(timeout=2.0)
        logger.info("SoundListener stopped")

    def _listen_loop(self):
        try:
            logger.info(f"Initializing WASAPI process audio capture for {self.process_name}...")

            check_count = 0
            max_samples = int(self.used_sr * self.sample_len)
            samples_per_check = max(1, int(self.used_sr * self.detection_interval))
            ring_buffer = np.zeros(max_samples * 2, dtype=np.float64)
            buffer_pos = 0
            total_written = 0
            samples_since_check = 0

            while self._running:
                if self._capture is None or not self._capture.is_alive():
                    if self._capture is not None:
                        logger.warning(
                            f"WASAPI process capture stopped; restarting: {self._capture.error}"
                        )
                        self._capture.stop()
                    self._capture = create_capture_source(
                        MODE_PROCESS,
                        process_name=self.process_name,
                    )
                    if not self._capture.start():
                        logger.warning(
                            "WASAPI process capture not ready for {}: {}".format(
                                self.process_name,
                                self._capture.error,
                            )
                        )
                        self._capture.stop()
                        self._capture = None
                        time.sleep(1.0)
                        continue
                    logger.info(f"Using audio capture source: {self._capture.name}")

                current_frame = self._capture.read(timeout=0.2)
                if current_frame is None or current_frame.size == 0:
                    continue

                if current_frame.shape[0] >= ring_buffer.shape[0]:
                    current_frame = current_frame[-ring_buffer.shape[0] :]

                end_pos = buffer_pos + current_frame.shape[0]
                if end_pos <= ring_buffer.shape[0]:
                    ring_buffer[buffer_pos:end_pos] = current_frame
                else:
                    first_part = ring_buffer.shape[0] - buffer_pos
                    ring_buffer[buffer_pos:] = current_frame[:first_part]
                    ring_buffer[: end_pos - ring_buffer.shape[0]] = current_frame[first_part:]

                buffer_pos = end_pos % ring_buffer.shape[0]
                total_written += current_frame.shape[0]
                samples_since_check += current_frame.shape[0]

                if total_written < max_samples or samples_since_check < samples_per_check:
                    continue

                samples_since_check = 0
                if buffer_pos >= max_samples:
                    window = ring_buffer[buffer_pos - max_samples : buffer_pos]
                else:
                    window = np.concatenate(
                        [
                            ring_buffer[-(max_samples - buffer_pos) :],
                            ring_buffer[:buffer_pos],
                        ]
                    )

                if self.is_computation_required and not self.is_computation_required():
                    continue

                norm_window = self._prepare_stream_waveform(window)
                dodge_score = self._match_normalized(norm_window, self._sample_waveform)
                counter_score = 0.0
                if self._counter_sample_waveform is not None:
                    counter_score = self._match_normalized(
                        norm_window,
                        self._counter_sample_waveform,
                    )

                self._check_triggers(dodge_score, counter_score)

                # self._draw_debug_visual(dodge_score, counter_score)

                check_count += 1
                if check_count % self.log_interval == 0:
                    logger.info(
                        "Audio monitoring - dodge_score: {:.4f} (threshold: {}), "
                        "counter_score: {:.4f} (threshold: {})".format(
                            dodge_score,
                            self.threshold,
                            counter_score,
                            self.counter_attack_threshold,
                        )
                    )
        except Exception as e:
            logger.error("Listener error", e)
        finally:
            if self._capture is not None:
                try:
                    self._capture.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop WASAPI process capture: {e}")
                self._capture = None
            self._running = False
            logger.info("Audio listener stopped")

    def _check_triggers(self, dodge_score, counter_score):
        now = time.time()
        if (
            not self.is_allow_successive_trigger
            and now - self._last_trigger_time < self._trigger_interval
        ):
            return

        if dodge_score > 0 and dodge_score > self.threshold:
            if self.on_dodge_triggered:
                logger.info(
                    "Dodge TRIGGERED! score: {:.4f}, threshold: {}".format(
                        dodge_score,
                        self.threshold,
                    )
                )
                self.on_dodge_triggered()
                self._last_trigger_time = now
                return

        if counter_score > 0 and counter_score > self.counter_attack_threshold:
            if self.on_counter_triggered:
                logger.info(
                    "Counter attack TRIGGERED! score: {:.4f}, threshold: {}".format(
                        counter_score,
                        self.counter_attack_threshold,
                    )
                )
                self.on_counter_triggered()
                self._last_trigger_time = now

    def _draw_debug_visual(self, dodge_score, counter_score):
        if not hasattr(self, "_visual_queue"):
            import queue

            self._visual_queue = queue.Queue(maxsize=1)
            self._mouse_x = -1

            def on_mouse(event, x, y, flags, param):
                if event == 0:  # cv2.EVENT_MOUSEMOVE
                    self._mouse_x = x

            def visual_worker():
                logger.info("Debug visual thread started")
                import cv2

                window_name = "Sound Listener Debug Wave"
                cv2.namedWindow(window_name)
                cv2.setMouseCallback(window_name, on_mouse)

                while self._running:
                    try:
                        # Timeout should be similar to detection_interval
                        d, c = self._visual_queue.get(timeout=0.1)
                        self._last_received_d, self._last_received_c = d, c
                        self._do_draw_debug_visual(d, c, update_history=True)
                    except Exception:
                        if self._running:
                            # If no data, redraw last state without updating history
                            last_d = getattr(self, "_last_received_d", 0.0)
                            last_c = getattr(self, "_last_received_c", 0.0)
                            self._do_draw_debug_visual(last_d, last_c, update_history=False)
                        continue
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                logger.info("Debug visual thread stopped")

            threading.Thread(target=visual_worker, daemon=True).start()

        self._last_d, self._last_c = dodge_score, counter_score

        try:
            # Use put_nowait to ensure we never block the audio loop
            # If the visual thread is slow, we just skip frames
            self._visual_queue.put_nowait((dodge_score, counter_score))
        except Exception:
            pass

    def _do_draw_debug_visual(self, dodge_score, counter_score, update_history=True):
        try:
            import cv2
            import numpy as np
        except ImportError:
            return

        if not hasattr(self, "_debug_history"):
            self._debug_history = {"dodge": [], "counter": []}
            self._max_history = 300
            self._debug_history["dodge"] = [0.0] * self._max_history
            self._debug_history["counter"] = [0.0] * self._max_history

        # Update history only if new data arrived
        if update_history:
            self._debug_history["dodge"].append(dodge_score)
            self._debug_history["counter"].append(counter_score)
            if len(self._debug_history["dodge"]) > self._max_history:
                self._debug_history["dodge"].pop(0)
                self._debug_history["counter"].pop(0)

        # Canvas settings
        width, height = 800, 400
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw grid and background
        canvas[:] = (20, 20, 20)
        for i in range(1, 10):
            y = int(height * i / 10)
            cv2.line(canvas, (0, y), (width, y), (40, 40, 40), 1)
        for i in range(1, 20):
            x = int(width * i / 20)
            cv2.line(canvas, (x, 0), (x, height), (30, 30, 30), 1)

        # Scale function
        max_val = max(0.5, self.threshold * 1.5, self.counter_attack_threshold * 1.5)

        def get_y(val):
            return int(height - (val / max_val) * height * 0.8) - 20

        # Draw thresholds
        d_y = get_y(self.threshold)
        c_y = get_y(self.counter_attack_threshold)
        cv2.line(canvas, (0, d_y), (width, d_y), (50, 50, 180), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, c_y), (width, c_y), (50, 180, 50), 1, cv2.LINE_AA)

        # Draw waves
        points_d = []
        points_c = []
        for i in range(self._max_history):
            x = int(i * (width / (self._max_history - 1)))
            points_d.append([x, get_y(self._debug_history["dodge"][i])])
            points_c.append([x, get_y(self._debug_history["counter"][i])])

        cv2.polylines(
            canvas, [np.array(points_d, np.int32)], False, (255, 100, 100), 2, cv2.LINE_AA
        )
        cv2.polylines(
            canvas, [np.array(points_c, np.int32)], False, (100, 255, 100), 2, cv2.LINE_AA
        )

        # Mouse interaction: Timeline and Values
        if hasattr(self, "_mouse_x") and 0 <= self._mouse_x < width:
            mx = self._mouse_x
            idx = int(mx * (self._max_history - 1) / width)
            if 0 <= idx < self._max_history:
                d_val = self._debug_history["dodge"][idx]
                c_val = self._debug_history["counter"][idx]

                # Draw vertical timeline line
                cv2.line(canvas, (mx, 0), (mx, height), (150, 150, 150), 1, cv2.LINE_AA)

                # Display detailed values at cursor
                info_text = f"T-{self._max_history - idx} | D: {d_val:.4f} | C: {c_val:.4f}"
                cv2.putText(
                    canvas,
                    info_text,
                    (min(mx + 10, width - 250), 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

        # Text labels
        cv2.putText(
            canvas,
            f"Dodge: {dodge_score:.3f} (T: {self.threshold:.3f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 100, 100),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Counter: {counter_score:.3f} (T: {self.counter_attack_threshold:.3f})",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 100),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Sound Listener Debug Wave", canvas)
        cv2.waitKey(1)
