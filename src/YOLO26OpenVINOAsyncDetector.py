import threading
import time

import numpy as np
from ok import Box, Logger
from openvino import AsyncInferQueue, Core, Layout, PartialShape, Type
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm

logger = Logger.get_logger(__name__)


class YOLO26OpenVINOAsyncDetector:
    _MAX_ACTIVE_RETIRED_INFER_QUEUES = 3
    _MAX_RETIRED_INFER_QUEUES = 10
    _RETIRED_QUEUE_KEEP_SECONDS = 3.0
    _SYNC_WAIT_TIMEOUT = 1.5

    def __init__(self, xml_path, num_requests=1):
        self.core = Core()
        model = self.core.read_model(model=xml_path)

        # 1. 配置预处理 (PPP) - 支持动态输入分辨率
        ppp = PrePostProcessor(model)

        # 声明输入的 Tensor 信息
        ppp.input().tensor().set_shape(PartialShape([1, -1, -1, 3])).set_element_type(
            Type.u8
        ).set_color_format(ColorFormat.BGR).set_layout(Layout("NHWC"))

        # 在预处理步骤中进行转换
        ppp.input().preprocess().convert_element_type(Type.f32).convert_color(
            ColorFormat.RGB
        ).resize(ResizeAlgorithm.RESIZE_LINEAR).scale([255.0, 255.0, 255.0])

        ppp.input().model().set_layout(Layout("NCHW"))
        model = ppp.build()

        # 2. 编译模型 (针对 AMD CPU 优化)
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": "2",  # 限制线程至 2，降低 CPU 峰值负载
        }
        self.compiled_model = self.core.compile_model(model, "CPU", config)

        self.model_h = 896
        self.model_w = 1536
        self.model_ratio = self.model_w / self.model_h

        # 3. 创建异步队列
        # 对于游戏辅助，jobs 建议设为 1 或 2，以保证最低延迟
        self.num_requests = num_requests
        self._state_lock = threading.RLock()
        self._retired_infer_queues = []
        self._active_queue_jobs = {}

        # 内部状态
        self.latest_results = None
        self.latest_image = None
        self.class_names = ["target"]  # 可根据 data.yaml 修改
        self.latency = 0.0  # 单次推理总耗时 (秒)
        self.job_id = 0
        self._force_next_submit = False
        self.infer_queue = self._create_infer_queue()

    def _create_infer_queue(self):
        infer_queue = AsyncInferQueue(self.compiled_model, jobs=self.num_requests)
        infer_queue.set_callback(self._callback)
        return infer_queue

    def _get_active_retired_count(self):
        with self._state_lock:
            return sum(
                1 for record in self._retired_infer_queues
                if self._active_queue_jobs.get(id(record["queue"]), 0) > 0
            )

    def _cleanup_retired_infer_queues(self):
        with self._state_lock:
            # 必须延迟销毁队列！如果因为 active_jobs 降为 0 就立刻从列表中移除对象，
            # 此时 C++ 回调线程可能尚未完全退出。Python GC 此时调用析构函数，
            # 将导致 GIL 锁死，造成软件完全卡死且无法恢复。
            now = time.monotonic()
            active_records = []
            inactive_records = []
            for record in self._retired_infer_queues:
                queue_id = id(record["queue"])
                active = self._active_queue_jobs.get(queue_id, 0) > 0
                still_warm = now - record["retired_at"] < self._RETIRED_QUEUE_KEEP_SECONDS
                if active or still_warm:
                    active_records.append(record)
                else:
                    inactive_records.append(record)

            keep_slots = max(0, self._MAX_RETIRED_INFER_QUEUES - len(active_records))
            inactive_to_keep = inactive_records[-keep_slots:] if keep_slots > 0 else []
            self._retired_infer_queues = active_records + inactive_to_keep

    def _mark_queue_job_started(self, infer_queue):
        with self._state_lock:
            queue_id = id(infer_queue)
            self._active_queue_jobs[queue_id] = (
                self._active_queue_jobs.get(queue_id, 0) + 1
            )
            return queue_id

    def _mark_queue_job_finished(self, queue_id):
        with self._state_lock:
            pending_jobs = self._active_queue_jobs.get(queue_id, 0) - 1
            if pending_jobs > 0:
                self._active_queue_jobs[queue_id] = pending_jobs
            else:
                self._active_queue_jobs.pop(queue_id, None)

    def _clear_queue_jobs(self, infer_queue):
        with self._state_lock:
            self._active_queue_jobs.pop(id(infer_queue), None)

    def _queue_has_active_jobs(self, infer_queue):
        with self._state_lock:
            return self._active_queue_jobs.get(id(infer_queue), 0) > 0

    def _cancel_queue_requests(self, infer_queue):
        try:
            for req in infer_queue:
                req.cancel()
        except Exception as e:
            logger.error("openvino cancel queue requests failed", e)

    def _retire_queue(self, infer_queue, cancel=True):
        retired_at = time.monotonic()
        if cancel:
            self._cancel_queue_requests(infer_queue)
            self._clear_queue_jobs(infer_queue)
        self._retired_infer_queues.append(
            {
                "queue": infer_queue,
                "retired_at": retired_at,
            }
        )

    def _try_rotate_busy_queue(self):
        with self._state_lock:
            self._cleanup_retired_infer_queues()
            if self.infer_queue.is_ready():
                return True
            if self._get_active_retired_count() >= self._MAX_ACTIVE_RETIRED_INFER_QUEUES:
                return False

            # Cancel current queue's requests so it stops stealing CPU immediately.
            self.job_id += 1
            self._retire_queue(self.infer_queue, cancel=True)
            self.infer_queue = self._create_infer_queue()
            return True

    def _callback(self, infer_request, user_data):
        """异步推理完成后的回调函数"""
        queue_id = user_data.get("queue_id")
        try:
            job_id = user_data.get("job_id", 0)
            if job_id < self.job_id:
                return

            start_time = user_data["start_time"]
            self.latency = time.time() - start_time

            detections = infer_request.get_output_tensor().data[0]

            box = user_data["box"]
            threshold = user_data["threshold"]
            target_label = user_data["label"]
            pad_x = user_data["pad_x"]
            pad_y = user_data["pad_y"]

            # 1. 画布相较于模型的缩放比例
            scale = user_data["target_w"] / self.model_w

            tmp_results = []
            for x1, y1, x2, y2, conf, cls_id in detections:
                if conf < threshold:
                    continue

                name = (
                    self.class_names[int(cls_id)]
                    if int(cls_id) < len(self.class_names)
                    else "unknown"
                )
                if target_label and name != target_label:
                    continue

                # 2. 从 AI 的坐标还原到带灰边的 Canvas 坐标
                canvas_x1 = x1 * scale
                canvas_y1 = y1 * scale
                canvas_w = (x2 - x1) * scale
                canvas_h = (y2 - y1) * scale

                # 3. 减去灰边的偏移量，得到在输入 input_crop 中的坐标
                # 再加上外面传进来的 Box 原图坐标，直接映射到全屏
                abs_x = int(canvas_x1 - pad_x + box.x)
                abs_y = int(canvas_y1 - pad_y + box.y)

                tmp_results.append(
                    Box(
                        x=abs_x,
                        y=abs_y,
                        width=int(canvas_w),
                        height=int(canvas_h),
                        confidence=float(conf),
                        name=name,
                    )
                )

            self.latest_results = tmp_results
            self.latest_image = user_data.get("image")
        except Exception as e:
            logger.error("openvino callback ignored failed/cancelled task", e)
        finally:
            if queue_id is not None:
                self._mark_queue_job_finished(queue_id)
            done_event = user_data.get("done_event")
            if done_event is not None:
                done_event.set()

    def debug_state(self):
        with self._state_lock:
            active_jobs = sum(self._active_queue_jobs.values())
            active_retired = self._get_active_retired_count()
            retired = len(self._retired_infer_queues)
            queue_ready = self.infer_queue.is_ready()
            latest_count = None if self.latest_results is None else len(self.latest_results)
            return (
                f"openvino(queue_ready={queue_ready}, active_jobs={active_jobs}, "
                f"retired={retired}, active_retired={active_retired}, "
                f"latest_count={latest_count}, latency={self.latency:.3f}, "
                f"force_next={self._force_next_submit}, job_id={self.job_id})"
            )

    def _detect(
        self,
        image,
        box: Box = None,
        threshold=0.5,
        label="target",
        force=False,
        mask_regions=None,
        done_event=None,
    ):
        """
        发起异步检测
        :param image: 全图 (numpy array)
        :param box: 指定检测区域的 Box 实例。如果为 None, 则检测全图。
        :param threshold: 置信度阈值
        :param label: 指定检测的类别名称
        :param force: 如果为 True，即使队列满也会丢弃旧结果并立刻提交新任务
        :param mask_regions: 需要屏蔽的全图归一化区域列表，格式为
            [(x1, y1, x2, y2), ...]。屏蔽会应用到推理画布，不修改原图。
        :return: list[Box] (返回的是上一帧或最近一次完成的结果)
        """

        submitted = False
        self._cleanup_retired_infer_queues()
        force_submit = force or self._force_next_submit
        if not self.infer_queue.is_ready():
            if not force_submit or not self._try_rotate_busy_queue():
                return self.latest_results, submitted

        h, w = image.shape[:2]

        if box is None:
            box = Box(x=0, y=0, width=w, height=h)

        # 1. 切片提取原始 ROI
        input_crop = image[
            max(0, box.y) : min(h, box.y + box.height),
            max(0, box.x) : min(w, box.x + box.width),
        ]

        crop_h, crop_w = input_crop.shape[:2]
        if crop_h == 0 or crop_w == 0:
            return self.latest_results, submitted  # 防止出界错误

        # 2. 补边逻辑：算出需要补多少灰边，让比例等于 model_ratio
        crop_ratio = crop_w / crop_h
        pad_x, pad_y = 0, 0

        if crop_ratio < self.model_ratio:
            # 框太瘦高了，左右补边
            target_h = crop_h
            target_w = int(crop_h * self.model_ratio)
            pad_x = (target_w - crop_w) // 2
        else:
            # 框太扁宽了，上下补边
            target_w = crop_w
            target_h = int(crop_w / self.model_ratio)
            pad_y = (target_h - crop_h) // 2

        # 3. 创建灰底画布并贴图 (耗时极短，保留 PPP 优势)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + crop_h, pad_x : pad_x + crop_w] = input_crop
        self._apply_canvas_mask(
            canvas,
            mask_regions,
            image_shape=(h, w),
            box=box,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        input_tensor = np.expand_dims(canvas, axis=0)

        with self._state_lock:
            self.job_id += 1
            current_job_id = self.job_id
            infer_queue = self.infer_queue
            queue_id = self._mark_queue_job_started(infer_queue)
            self._force_next_submit = False

        try:
            infer_queue.start_async(
                {0: input_tensor},
                {
                    "box": box,
                    "threshold": threshold,
                    "label": label,
                    "start_time": time.time(),
                    # 传给回调函数，用于减去补边的偏移
                    "pad_x": pad_x,
                    "pad_y": pad_y,
                    "target_w": target_w,  # 记录画布的总宽用于还原缩放
                    "job_id": current_job_id,
                    "queue_id": queue_id,
                    "image": image,
                    "done_event": done_event,
                },
            )
            submitted = True
        except Exception:
            self._mark_queue_job_finished(queue_id)
            if done_event is not None:
                done_event.set()
            raise

        return self.latest_results, submitted

    def detect(
        self,
        image,
        box: Box = None,
        threshold=0.5,
        label="target",
        force=False,
        mask_regions=None,
    ):
        """
        发起异步检测，返回上一帧或最近一次完成的结果。
        """

        results, _ = self._detect(
            image,
            box=box,
            threshold=threshold,
            label=label,
            force=force,
            mask_regions=mask_regions,
        )
        return results

    def wait(self, include_retired=False):
        """强制阻塞主线程，默认只等待当前推理队列完成。"""
        with self._state_lock:
            queues = [self.infer_queue]
            if include_retired:
                queues.extend(record["queue"] for record in self._retired_infer_queues)
        for infer_queue in queues:
            infer_queue.wait_all()
        self._cleanup_retired_infer_queues()

    def _apply_canvas_mask(self, canvas, mask_regions, image_shape, box, pad_x, pad_y):
        if not mask_regions:
            return

        image_h, image_w = image_shape
        canvas_h, canvas_w = canvas.shape[:2]
        crop_x1 = max(0, box.x)
        crop_y1 = max(0, box.y)
        crop_x2 = min(image_w, box.x + box.width)
        crop_y2 = min(image_h, box.y + box.height)

        for x1_ratio, y1_ratio, x2_ratio, y2_ratio in mask_regions:
            x1 = max(crop_x1, min(crop_x2, int(x1_ratio * image_w)))
            y1 = max(crop_y1, min(crop_y2, int(y1_ratio * image_h)))
            x2 = max(crop_x1, min(crop_x2, int(x2_ratio * image_w)))
            y2 = max(crop_y1, min(crop_y2, int(y2_ratio * image_h)))
            if x1 >= x2 or y1 >= y2:
                continue

            canvas_x1 = max(0, min(canvas_w, x1 - crop_x1 + pad_x))
            canvas_y1 = max(0, min(canvas_h, y1 - crop_y1 + pad_y))
            canvas_x2 = max(0, min(canvas_w, x2 - crop_x1 + pad_x))
            canvas_y2 = max(0, min(canvas_h, y2 - crop_y1 + pad_y))
            if canvas_x1 < canvas_x2 and canvas_y1 < canvas_y2:
                canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = 114

    def detect_sync(
        self, image, box=None, threshold=0.5, label="target", mask_regions=None
    ):
        """同步检测版本：发起请求后立即堵住，直到拿到本次请求的结果。"""
        done_event = threading.Event()
        # 同步调用通常用于确认当前帧状态。若等待旧 async 请求，回来的可能是旧帧、
        # 旧 ROI 或旧 mask 的结果，因此这里始终允许轮换忙队列并提交本次请求。
        _, submitted = self._detect(
            image,
            box=box,
            threshold=threshold,
            label=label,
            force=True,
            mask_regions=mask_regions,
            done_event=done_event,
        )
        if not submitted:
            return None
        if not done_event.wait(timeout=self._SYNC_WAIT_TIMEOUT):
            with self._state_lock:
                self.job_id += 1
                if (
                    self._active_queue_jobs.get(id(self.infer_queue), 0) > 0
                    and self._get_active_retired_count()
                    < self._MAX_ACTIVE_RETIRED_INFER_QUEUES
                ):
                    self._retire_queue(self.infer_queue, cancel=True)
                    self.infer_queue = self._create_infer_queue()
                retired_count = self._get_active_retired_count()
                active_jobs = sum(self._active_queue_jobs.values())
            logger.warning(
                "openvino sync detect timed out after "
                f"{self._SYNC_WAIT_TIMEOUT:.1f}s, retired_queues={retired_count}, "
                f"active_jobs={active_jobs}"
            )
            return None
        return self.latest_results

    def clear_cache(self):
        """清空缓存"""
        with self._state_lock:
            self.latest_results = None
            self.latest_image = None
            self.job_id += 1  # 增加 epoch，所有正在运行的旧任务的回调都会失效
            self._force_next_submit = True
            if (
                self._queue_has_active_jobs(self.infer_queue)
                and self._get_active_retired_count()
                < self._MAX_ACTIVE_RETIRED_INFER_QUEUES
            ):
                self._retire_queue(self.infer_queue, cancel=True)
                self.infer_queue = self._create_infer_queue()
        self._cleanup_retired_infer_queues()
