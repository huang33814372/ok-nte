from typing import cast

from ok import Box


def calculate_iou(box1: Box, box2: Box):
    """
    计算两个矩形框的交并比 (IoU)
    box 对象需具备 x, y, width, height 属性
    """
    x1_min, y1_min = box1.x, box1.y
    x1_max, y1_max = box1.x + box1.width, box1.y + box1.height
    x2_min, y2_min = box2.x, box2.y
    x2_max, y2_max = box2.x + box2.width, box2.y + box2.height

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = box1.area()
    area2 = box2.area()

    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0
    return inter_area / union_area


def suppress_boxes(data: list[Box] | list[list[Box]], iou_threshold=0.5) -> list[Box]:
    """
    非极大值抑制(NMS)通用函数：
    1. 如果传入 list[Box]: 执行单列表内部去重。
    2. 如果传入 list[list[Box]]: 仅在不同子列表之间去重，子列表内部保持原样。

    :param data: 特征对象列表 或 特征对象列表的列表
    :param iou_threshold: 重合度阈值 (0.0~1.0)，越高越允许重叠
    :return: 过滤后的扁平 Box 列表
    """
    if not data:
        return []

    if isinstance(data[0], list):
        nested_data = cast(list[list[Box]], data)
        tagged_boxes = [
            (box, group_id) for group_id, sub_list in enumerate(nested_data) for box in sub_list
        ]
    else:
        flat_data = cast(list[Box], data)
        tagged_boxes = [(box, i) for i, box in enumerate(flat_data)]

    tagged_boxes.sort(key=lambda item: getattr(item[0], "confidence", 0), reverse=True)

    keep: list[Box] = []
    suppressed = [False] * len(tagged_boxes)

    for i, (best_box, best_group) in enumerate(tagged_boxes):
        if suppressed[i]:
            continue

        keep.append(best_box)

        for j in range(i + 1, len(tagged_boxes)):
            if suppressed[j]:
                continue

            box, group = tagged_boxes[j]
            if best_group != group and calculate_iou(best_box, box) > iou_threshold:
                suppressed[j] = True

    return keep
