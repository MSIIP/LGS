import json
import os

ds_type = "train"
luna16_path = f"path/to/LUNA16/luna16_filtered_2d_{ds_type}.json"
image_dir = "path/to/LUNA16/all_subsets_jpg"
save_path = f"work_dirs/dataset/luna16_lgs_{ds_type}.json"
prompt = "<image>\nIs there any lung nodule in this image?\nAnswer only Yes or No."

with open(luna16_path, 'r') as f:
    luna16_data = json.load(f)

data = []
for fileid, fileinfo in luna16_data.items():
    H, W = fileinfo["image_size"][0], fileinfo["image_size"][1]
    bbox = fileinfo["lung_nodule_25"]  # [y_min, y_max, x_min, x_max]
    y_min = int(bbox[0] / H * 1000)
    y_max = int(bbox[1] / H * 1000)
    x_min = int(bbox[2] / W * 1000)
    x_max = int(bbox[3] / W * 1000)

    # <|object_ref_start|>person<|object_ref_end|><|box_start|>(257,397),(313,572)<|box_end|>
    data.append({
        "id": f"{fileid}_pos",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"Yes ([{x_min}, {y_min}, {x_max}, {y_max}])"}
            # {"role": "assistant", "content": f"Yes (<|object_ref_start|>lung nodule<|object_ref_end|><|box_start|>({x_min},{y_min}),({x_max},{y_max})<|box_end|>)"}
        ],
        "images": [os.path.join(image_dir, fileinfo["pos_path"])],
        "image_size": fileinfo["image_size"],
        "spacing": fileinfo["spacing"],
        "lung_nodule": fileinfo["lung_nodule"],
        "lung_nodule_25": fileinfo["lung_nodule_25"],
    })
    data.append({
        "id": f"{fileid}_neg",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "No"}
        ],
        "images": [os.path.join(image_dir, fileinfo["neg_path"])],
        "image_size": fileinfo["image_size"],
        "spacing": fileinfo["spacing"],
        "lung_nodule": fileinfo["lung_nodule"],
        "lung_nodule_25": fileinfo["lung_nodule_25"],
    })

with open(save_path, 'w') as f:
    json.dump(data, f, indent=4)
