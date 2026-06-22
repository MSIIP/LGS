import json
import os

ds_type = "train"
luna16_path = f"path/to/LUNA16/luna16_filtered_2d_{ds_type}.json"
image_dir = "path/to/LUNA16/all_subsets_jpg"
save_path = f"work_dirs/dataset/luna16_cls_{ds_type}.json"
prompt = "<image>\nIs there any lung nodule in this image?\nAnswer only Yes or No."

with open(luna16_path, 'r') as f:
    luna16_data = json.load(f)

data = []
for fileid, fileinfo in luna16_data.items():
    data.append({
        "id": f"{fileid}_pos",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Yes"}
        ],
        "images": [os.path.join(image_dir, fileinfo["pos_path"])],
        "image_size": fileinfo["image_size"],
        "spacing": fileinfo["spacing"],
        "lung_nodule": fileinfo["lung_nodule"],
        "lung_nodule_25": fileinfo["lung_nodule_25"],
        "label": 1,
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
        "label": 0,
    })

with open(save_path, 'w') as f:
    json.dump(data, f, indent=4)
