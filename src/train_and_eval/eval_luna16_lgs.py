import argparse
import json
import re
import math
from typing import Optional, Sequence, Tuple, Union


def extract_bboxes_from_parentheses(text: str):
    results = []

    # 提取 () 内的整体内容
    groups = re.findall(r"\(([^()]*)\)", text)

    for g in groups:
        # 直接提取所有数字（支持负数 / 浮点）
        nums = re.findall(r"-?\d+\.?\d*", g)

        if len(nums) != 4:
            continue

        bbox = [int(float(x)) for x in nums]
        results.append(bbox)

    if len(results) > 0:
        return results[0]
    else:
        return None


def iou_and_center_distance(
    bbox1,
    bbox2,
    eps: float = 1e-9,
) -> Tuple[float, float]:
    """
    Args:
        bbox1, bbox2: [x_min, y_min, x_max, y_max] or None
        eps: small value to avoid division by zero

    Returns:
        (iou, center_dist)
        If bbox1 is None or bbox2 is None or invalid => (-1.0, -1.0)
    """
    # None handling
    if bbox1 is None or bbox2 is None:
        return -1.0, -1.0

    # Basic validation
    if len(bbox1) != 4 or len(bbox2) != 4:
        return -1.0, -1.0

    x1_min, y1_min, x1_max, y1_max = map(float, bbox1)
    x2_min, y2_min, x2_max, y2_max = map(float, bbox2)

    # Invalid boxes (non-positive width/height)
    if x1_max <= x1_min or y1_max <= y1_min or x2_max <= x2_min or y2_max <= y2_min:
        return -1.0, -1.0

    # Intersection
    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_w * inter_h

    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + eps)

    # Center distance
    c1x, c1y = (x1_min + x1_max) / 2.0, (y1_min + y1_max) / 2.0
    c2x, c2y = (x2_min + x2_max) / 2.0, (y2_min + y2_max) / 2.0
    center_dist = math.hypot(c1x - c2x, c1y - c2y)

    return iou, center_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="work_dirs/Qwen3-VL-2B-Thinking/eval/output.jsonl")
    args = parser.parse_args()

    with open(args.input_path, 'r') as f:
        results = [json.loads(line) for line in f]

    correct_list = []
    wrong_list = []
    iou_list = []
    dist_list = []

    correct = 0
    correct_yes = 0
    cnt_yes = 0
    for item in results:
        gt = re.sub(r"<think>.*?</think>", "", item["labels"], flags=re.DOTALL)
        gt = re.sub(r"\([^()]*\)", "", gt)
        gt = gt.strip().lower()
        pred = re.sub(r"<think>.*?</think>", "", item['response'], flags=re.DOTALL)
        pred = re.sub(r"\([^()]*\)", "", pred)
        pred = pred.strip().lower()
        # print("pred:", item['response'], pred)
        # print("gt:", item["labels"], gt)

        if gt == "no":
            if pred == "no":
                correct += 1
        elif gt == "yes":
            cnt_yes += 1
            if pred == "yes":
                correct += 1
                correct_yes += 1

                gt_bbox = extract_bboxes_from_parentheses(item["labels"])
                pred_bbox = extract_bboxes_from_parentheses(item['response'])
                iou, center_dist = iou_and_center_distance(gt_bbox, pred_bbox)
                if iou > 0 and center_dist > 0:
                    iou_list.append(iou)
                    dist_list.append(center_dist)

    accuracy = correct / len(results)
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {correct_yes/cnt_yes}={correct_yes}/{cnt_yes}")
    print(len(iou_list), sum(iou_list) / len(iou_list))
    print(len(dist_list), sum(dist_list) / len(dist_list))

    # save_correct_path = args.input_path.replace(".jsonl", "_correct.json")
    # with open(save_correct_path, "w") as f:
    #     json.dump(correct_list, f, indent=4)

    # save_wrong_path = args.input_path.replace(".jsonl", "_wrong.json")
    # with open(save_wrong_path, "w") as f:
    #     json.dump(wrong_list, f, indent=4)
