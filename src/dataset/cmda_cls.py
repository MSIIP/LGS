import json
import os

ds_type = "train"
abnormal_name = "vestibular_schwannoma"
anns_path = f"path/to/CrossMoDA2021/cmda_mask_2d_{ds_type}.json"
save_path = f"work_dirs/dataset/cmda_cls_{ds_type}.json"
prompt = f"<image>\nIs there any {abnormal_name.replace('_', ' ')} in this image?\nAnswer only Yes or No."

with open(anns_path, 'r') as f:
    anns = json.load(f)

data = []
for fileid, fileinfo in anns.items():
    data.append({
        "id": f"{fileid}_pos",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Yes"}
        ],
        "images": [fileinfo["pos_path"]],
        "image_size": fileinfo["image_size"],
        "spacing": fileinfo["spacing"],
        abnormal_name: fileinfo[abnormal_name],
        "label": 1,
    })
    data.append({
        "id": f"{fileid}_neg",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "No"}
        ],
        "images": [fileinfo["neg_path"]],
        "image_size": fileinfo["image_size"],
        "spacing": fileinfo["spacing"],
        abnormal_name: fileinfo[abnormal_name],
        "label": 0,
    })

with open(save_path, 'w') as f:
    json.dump(data, f, indent=4)
