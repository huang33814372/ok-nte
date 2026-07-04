import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from ok import Box

vision_module = importlib.import_module("src.tasks.mixin.VisionMixin")
VisionMixin = vision_module.VisionMixin


class _FakeSift:
    def detectAndCompute(self, _gray, _mask):
        keypoints = [vision_module.cv2.KeyPoint(float(index), float(index), 1) for index in range(4)]
        descriptors = np.zeros((4, 128), dtype=np.float32)
        return keypoints, descriptors


class _FakeMatcher:
    def knnMatch(self, _template_descriptors, _scene_descriptors, k=2):
        return [
            [
                SimpleNamespace(queryIdx=index, trainIdx=index, distance=0.1),
                SimpleNamespace(queryIdx=index, trainIdx=index, distance=1.0),
            ]
            for index in range(3)
        ]


class _VisionTask(VisionMixin):
    @property
    def frame(self):
        return self._frame


class TestVisionMixin(unittest.TestCase):
    def test_rotated_template_chamfer_tolerates_small_occlusion(self):
        task = object.__new__(_VisionTask)
        template_mask = np.zeros((12, 10), dtype=np.uint8)
        template_mask[:, :3] = 255
        template_mask[-3:, :] = 255
        template_mask[:3, :8] = 255
        template = np.dstack([template_mask, template_mask, template_mask])

        frame = np.zeros((30, 30, 3), dtype=np.uint8)
        frame[10:22, 8:18] = template
        frame[15:17, 9:11] = 0

        drawn_boxes = []
        task._frame = frame
        task.get_feature_by_name = lambda _name: SimpleNamespace(mat=template)
        task.draw_boxes = lambda boxes, color: drawn_boxes.append((boxes, color))

        result, _ = task.find_rotated_template(
            "unit_chamfer",
            box=Box(3, 4, 24, 24),
            angle_range=range(0, 1),
            threshold=0.8,
            min_coverage=0.8,
            frame_processor=lambda cropped: cropped[:, :, 0],
        )

        self.assertEqual(1, len(result))
        self.assertGreaterEqual(result[0]["coverage"], 0.9)
        self.assertEqual((10, 12), result[0]["center"])
        self.assertEqual(2, len(drawn_boxes))
        self.assertEqual("blue", drawn_boxes[0][1])
        self.assertEqual("red", drawn_boxes[1][1])
        self.assertEqual((8, 10), (drawn_boxes[1][0].x, drawn_boxes[1][0].y))

    def test_rotated_template_processor_runs_after_box_crop(self):
        task = object.__new__(_VisionTask)
        template_mask = np.full((4, 4), 255, dtype=np.uint8)
        template = np.dstack([template_mask, template_mask, template_mask])
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[3:7, 4:8] = template
        seen_shapes = []

        task._frame = frame
        task.get_feature_by_name = lambda _name: SimpleNamespace(mat=template)
        task.draw_boxes = lambda *args, **kwargs: None

        def processor(cropped):
            seen_shapes.append(cropped.shape[:2])
            return cropped[:, :, 0]

        result, _ = task.find_rotated_template(
            "unit_processor",
            box=Box(2, 1, 7, 8),
            angle_range=range(0, 1),
            min_non_zero=4,
            threshold=0.75,
            min_coverage=0.75,
            frame_processor=processor,
        )

        self.assertEqual([(8, 7)], seen_shapes)
        self.assertEqual(1, len(result))

    def test_sift_homography_requires_at_least_four_matches(self):
        task = object.__new__(_VisionTask)
        task._frame = np.zeros((32, 32, 3), dtype=np.uint8)
        task.get_original_feature_by_name = lambda _name: SimpleNamespace(
            mat=np.zeros((16, 16, 3), dtype=np.uint8)
        )
        task.draw_boxes = lambda *args, **kwargs: None
        task.log_debug = lambda _message: None

        with (
            patch.object(vision_module.cv2, "SIFT_create", return_value=_FakeSift()),
            patch.object(vision_module.cv2, "BFMatcher", return_value=_FakeMatcher()),
            patch.object(vision_module.cv2, "findHomography") as find_homography,
        ):
            result = task.find_sift_feature(
                "unit_sift_min_matches",
                min_match_count=3,
                small_target_retry=False,
            )

        self.assertIsNone(result)
        find_homography.assert_not_called()


if __name__ == "__main__":
    unittest.main()
