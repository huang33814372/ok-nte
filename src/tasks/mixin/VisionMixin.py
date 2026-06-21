import time
from typing import NamedTuple

import cv2
import numpy as np

from ok import BaseTask, Box


class RotatedTemplateCacheKey(NamedTuple):
    cache_id: object
    mask_shape: tuple[int, ...]
    mask_non_zero: int
    mask_hash: int
    angles: tuple[int, ...]
    min_non_zero: int


type RotatedTemplateCacheValue = list[tuple[int, np.ndarray]]


class SiftTemplateCacheKey(NamedTuple):
    feature_name: object
    template_shape: tuple[int, ...]
    template_dtype: str
    template_hash: int
    nfeatures: int


type SiftTemplateCacheValue = tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]


class VisionMixin(BaseTask):
    # cache_id/mask fingerprint/angle options -> [(angle, rotated mask), ...]
    _rotated_template_cache: dict[RotatedTemplateCacheKey, RotatedTemplateCacheValue] = {}

    # feature/template fingerprint/SIFT options -> (template keypoints, template descriptors)
    _sift_template_cache: dict[SiftTemplateCacheKey, SiftTemplateCacheValue] = {}

    def find_sift_feature(
        self,
        feature_name,
        box: Box | str | None = None,
        frame: np.ndarray | None = None,
        threshold=0.45,
        min_match_count=8,
        ratio=0.75,
        nfeatures=0,
    ) -> Box | None:
        """
        使用 SIFT 在指定区域内查找特征，适合模板和画面存在缩放差异的场景。

        :param feature_name: ok feature 名称。
        :param box: 搜索区域；None 表示整张图，也可传 box 名称。
        :param frame: 目标画面；None 时使用当前帧。
        :param threshold: RANSAC 内点比例阈值，越高越严格。
        :param min_match_count: 通过 Lowe ratio 过滤后的最少匹配点数量。
        :param ratio: Lowe ratio test 阈值，越低越严格。
        :param nfeatures: 传给 cv2.SIFT_create 的最大特征数，0 表示不限。
        :return: 匹配到的 Box；未找到时返回 None。Box 上额外带有 scale/match_count/inlier_count。
        """
        return self._find_sift_feature(
            feature_name=feature_name,
            box=box,
            frame=frame,
            threshold=threshold,
            min_match_count=min_match_count,
            ratio=ratio,
            nfeatures=nfeatures,
        )

    def _find_sift_feature(
        self,
        feature_name,
        box: Box | str | None = None,
        frame: np.ndarray | None = None,
        threshold=0.45,
        min_match_count=8,
        ratio=0.75,
        nfeatures=0,
    ) -> Box | None:
        start_time = time.time()

        def finish(result: Box | None, reason=None):
            cost_ms = (time.time() - start_time) * 1000
            if cost_ms > 100:
                if result is None:
                    self.log_debug(
                        f"find_sift_feature {feature_name} not found"
                        f" reason={reason} cost={cost_ms:.2f}ms"
                    )
                else:
                    self.log_debug(
                        f"find_sift_feature {feature_name} found {result}"
                        f" scale={getattr(result, 'scale', None)}"
                        f" matches={getattr(result, 'match_count', None)}"
                        f" inliers={getattr(result, 'inlier_count', None)}"
                        f" cost={cost_ms:.2f}ms"
                    )
            return result

        frame = self.frame if frame is None else frame
        if frame is None:
            return finish(None, "empty_frame")

        if isinstance(box, str):
            box = self.get_box_by_name(box)

        search_x, search_y = 0, 0
        if box is None:
            scene = frame
        else:
            search_x, search_y = box.x, box.y
            scene = box.crop_frame(frame)
            self.draw_boxes(feature_name, boxes=box, color="blue")

        if scene is None or scene.size == 0:
            return finish(None, "empty_scene")

        template = self.get_feature_by_name(feature_name).mat
        sift = cv2.SIFT_create(nfeatures=nfeatures)
        template_keypoints, template_descriptors = self._get_sift_template_data(
            feature_name, template, sift, nfeatures
        )
        if template_descriptors is None or len(template_keypoints) < min_match_count:
            return finish(None, "not_enough_template_keypoints")

        scene_gray = self._to_sift_gray(scene)
        scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_gray, None)
        if scene_descriptors is None or len(scene_keypoints) < min_match_count:
            return finish(None, "not_enough_scene_keypoints")

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = matcher.knnMatch(template_descriptors, scene_descriptors, k=2)
        good_matches = []
        for candidates in raw_matches:
            if len(candidates) < 2:
                continue
            first, second = candidates
            if first.distance < ratio * second.distance:
                good_matches.append(first)

        if len(good_matches) < min_match_count:
            return finish(None, "not_enough_good_matches")

        template_points = np.float32(
            [template_keypoints[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        scene_points = np.float32(
            [scene_keypoints[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(
            template_points, scene_points, cv2.RANSAC, 5.0
        )
        if homography is None or inlier_mask is None:
            return finish(None, "homography_failed")

        inlier_count = int(inlier_mask.ravel().sum())
        confidence = inlier_count / len(good_matches)
        if inlier_count < min_match_count or confidence < threshold:
            return finish(None, "low_inlier_confidence")

        template_height, template_width = template.shape[:2]
        corners = np.float32(
            [[0, 0], [template_width, 0], [template_width, template_height], [0, template_height]]
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        if not np.isfinite(projected).all():
            return finish(None, "invalid_projection")

        x, y, width, height = cv2.boundingRect(projected.astype(np.float32))
        if width <= 0 or height <= 0:
            return finish(None, "invalid_projected_box")

        scene_height, scene_width = scene.shape[:2]
        x = max(0, min(x, scene_width - 1))
        y = max(0, min(y, scene_height - 1))
        width = min(width, scene_width - x)
        height = min(height, scene_height - y)
        if width <= 0 or height <= 0:
            return finish(None, "projected_box_outside_scene")

        matched_box = Box(
            x + search_x,
            y + search_y,
            width,
            height,
            confidence=round(confidence, 3),
            name=feature_name,
        )
        matched_box.scale = round((width * height / (template_width * template_height)) ** 0.5, 3)
        matched_box.match_count = len(good_matches)
        matched_box.inlier_count = inlier_count
        self.draw_boxes(feature_name, boxes=matched_box, color="red")
        return finish(matched_box)

    def _get_sift_template_data(self, feature_name, template: np.ndarray, sift, nfeatures=0):
        template_key = SiftTemplateCacheKey(
            feature_name=feature_name,
            template_shape=template.shape,
            template_dtype=template.dtype.str,
            template_hash=hash(template.tobytes()),
            nfeatures=nfeatures,
        )
        cached = self._sift_template_cache.get(template_key)
        if cached is not None:
            return cached

        gray = self._to_sift_gray(template)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        data = (tuple(keypoints), descriptors)
        self._sift_template_cache[template_key] = data
        return data

    @staticmethod
    def _to_sift_gray(mat: np.ndarray):
        if mat.ndim == 2:
            gray = mat
        elif mat.shape[2] == 4:
            bgr = mat[:, :, :3].copy()
            bgr[mat[:, :, 3] == 0] = 0
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(mat[:, :, :3], cv2.COLOR_BGR2GRAY)

        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    def _find_rotated_template(
        self,
        feature_name,
        scene: np.ndarray,
        threshold=0.75,
        angle_range=range(-180, 180, 2),
        min_non_zero=20,
        template_angle=0,
    ):
        start_time = time.time()
        template = self.get_feature_by_name(feature_name).mat
        scene_mask = self._first_channel_mask(scene)
        if cv2.countNonZero(scene_mask) < min_non_zero:
            return [], (time.time() - start_time) * 1000

        best = None
        for angle, rotated_template in self._get_rotated_templates(
            template,
            angle_range=angle_range,
            min_non_zero=min_non_zero,
            cache_key=feature_name,
        ):
            th, tw = rotated_template.shape[:2]
            if th > scene_mask.shape[0] or tw > scene_mask.shape[1]:
                continue

            result = cv2.matchTemplate(scene_mask, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, score, _, top_left = cv2.minMaxLoc(result)
            if best is None or score > best["score"]:
                best = {
                    "center": (top_left[0] + tw // 2, top_left[1] + th // 2),
                    "angle": self._normalize_angle(angle + template_angle),
                    "match_angle": angle,
                    "score": score,
                }

        if best is None or best["score"] < threshold:
            return [], (time.time() - start_time) * 1000

        best["score"] = round(best["score"], 3)
        return [best], (time.time() - start_time) * 1000

    def _get_rotated_templates(
        self,
        template: np.ndarray,
        angle_range=range(-180, 180, 5),
        min_non_zero=20,
        cache_key=None,
    ):
        template_mask = self._trim_mask(self._first_channel_mask(template))
        angles = tuple(angle_range)
        template_key = RotatedTemplateCacheKey(
            cache_id=cache_key or id(template),
            mask_shape=template_mask.shape,
            mask_non_zero=cv2.countNonZero(template_mask),
            mask_hash=hash(template_mask.tobytes()),
            angles=angles,
            min_non_zero=min_non_zero,
        )
        cached = self._rotated_template_cache.get(template_key)
        if cached is not None:
            return cached

        templates = []
        for angle in angles:
            rotated = self._rotate_mask(template_mask, angle)
            rotated = self._trim_mask(rotated)
            if cv2.countNonZero(rotated) >= min_non_zero:
                templates.append((angle, rotated))

        self._rotated_template_cache[template_key] = templates
        return templates

    @staticmethod
    def _first_channel_mask(mat: np.ndarray):
        if mat.ndim == 2:
            return mat
        return mat[:, :, 0]

    @staticmethod
    def _normalize_angle(angle):
        return (angle + 180) % 360 - 180

    def _rotate_mask(self, mask: np.ndarray, angle):
        h, w = mask.shape[:2]
        center = (w / 2, h / 2)
        rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(rotate_matrix[0, 0])
        sin = abs(rotate_matrix[0, 1])
        new_w = int(round(h * sin + w * cos))
        new_h = int(round(h * cos + w * sin))
        rotate_matrix[0, 2] += new_w / 2 - center[0]
        rotate_matrix[1, 2] += new_h / 2 - center[1]
        return cv2.warpAffine(
            mask,
            rotate_matrix,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )

    def _trim_mask(self, mask):
        points = cv2.findNonZero(mask)
        if points is None:
            return mask
        x, y, w, h = cv2.boundingRect(points)
        return mask[y : y + h, x : x + w]

    def _find_contours_from_first_channel(self, bgr):
        bin_mat = bgr[:, :, 0]
        contours, _ = cv2.findContours(bin_mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _find_rotated_shape(self, target_contour, scene_contours, score_threshold=0.1):
        """
        target_contour: 要匹配的目标轮廓。
        scene_contours: 在场景中找到的候选轮廓。
        score_threshold: 越小越严格。通常 0.05-0.2 之间。
        """
        start_time = time.time()

        results = []
        for cnt in scene_contours:
            if cv2.contourArea(cnt) < 50:
                continue

            # 核心算法：比较两个形状的胡氏矩 (I1 模式最常用)
            # 返回值越小，匹配度越高（0 为完美匹配）
            score = cv2.matchShapes(target_contour, cnt, cv2.CONTOURS_MATCH_I1, 0.0)

            if score < score_threshold:
                # 计算重心和角度
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # 使用最小外接矩形获取角度
                    rect = cv2.minAreaRect(cnt)
                    angle = rect[2]  # 得到角度

                    results.append({"center": (cx, cy), "angle": angle, "score": round(score, 3)})

        # 按分数升序排列（得分越低越好）
        results = sorted(results, key=lambda x: x["score"])
        return results, (time.time() - start_time) * 1000
