import argparse
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="work_dirs/Qwen3-VL-2B-Thinking/eval/output.jsonl")
args = parser.parse_args()

with open(args.input_path, 'r') as f:
    results = [json.loads(line) for line in f]

correct_list = []
wrong_list = []

# correct = 0
# correct_yes = 0
# cnt_yes = 0
tp = 0
tn = 0
fp = 0
fn = 0
for item in results:
    gt = re.sub(r"<think>.*?</think>", "", item["labels"], flags=re.DOTALL)
    gt = re.sub(r"\([^()]*\)", "", gt)
    gt = gt.strip().lower()
    pred = re.sub(r"<think>.*?</think>", "", item['response'], flags=re.DOTALL)
    pred = re.sub(r"\([^()]*\)", "", pred)
    pred = pred.strip().lower()

    # if gt == "no":
    #     if pred == "no":
    #         correct += 1
    # elif gt == "yes":
    #     cnt_yes += 1
    #     if pred == "yes":
    #         correct += 1
    #         correct_yes += 1
    if gt == "yes":
        if pred == "yes":
            tp += 1
        elif pred == "no":
            fn += 1
    elif gt == "no":
        if pred == "no":
            tn += 1
        elif pred == "yes":
            fp += 1

# accuracy = correct / len(results)
# print(f"Accuracy: {accuracy}")
# print(f"Recall: {correct_yes/cnt_yes}={correct_yes}/{cnt_yes}")

print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
accuracy = (tp + tn) / len(results) if len(results) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# save tp,tn,fp,fn,acc,precsion,recall,f1 as csv
import csv
save_path = args.input_path.replace(".jsonl", ".csv")

with open(save_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "tp", "tn", "fp", "fn", "acc", "precision", "recall", "f1"])
    writer.writerow([args.input_path.split("/")[-3], tp, tn, fp, fn, accuracy, precision, recall, f1])

# save_correct_path = args.input_path.replace(".jsonl", "_correct.json")
# with open(save_correct_path, "w") as f:
#     json.dump(correct_list, f, indent=4)

# save_wrong_path = args.input_path.replace(".jsonl", "_wrong.json")
# with open(save_wrong_path, "w") as f:
#     json.dump(wrong_list, f, indent=4)
