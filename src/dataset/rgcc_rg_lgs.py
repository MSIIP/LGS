import json
import random
from tqdm import tqdm


refer_path = "path/to/RadGenome-ChestCT/region_valid.json"
input_path = "path/to/RadGenome-ChestCT/report_region_valid.json"
output_raw_path = "work_dirs/dataset/rgcc_raw_valid.json"
output_wo_path = "work_dirs/dataset/rgcc_rg_valid.json"
output_with_path = "work_dirs/dataset/rgcc_lgs_valid.json"

region_list = {
    'thyroid': ["thyroid_left", "thyroid_right"],  # 1. 颈部/最上方
    'trachea and bronchie': ["trachea_and_bronchie"],  # 2. 气道中轴
    'lung': ["lung_left", "lung_right"],  # 3. 实质器官（呼吸）
    'pleura': ["lung_left", "lung_right"],  # 4. 肺部外层
    'mediastinum': [],  # 5. 胸腔中部区域
    'heart': ["heart"],  # 6. 纵隔内核心
    'esophagus': ["esophagus"],  # 7. 纵隔深层/消化道
    'breast': [],  # 8. 胸壁软组织
    'bone': [],  # 9. 骨骼支架
    'abdomen': []  # 10. 远端延伸
}


if __name__ == "__main__":
    with open(refer_path, "r") as f:
        refer_data = json.load(f)
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    data_raw = []
    data_wo = []
    data_with = []
    for d in tqdm(raw_data):
        fileid = d["videos"][0].split("/")[-1].replace(".npy", "")
        report = d["messages"][1]["content"]
        if fileid in refer_data:
            image_size = refer_data[fileid]["image_size"]
            for region in region_list:
                if f"<{region.replace(' ', '_')}>" in report:
                    bbox_str_list = []
                    for bbox_name in region_list[region]:
                        bbox = refer_data[fileid][bbox_name]
                        z_min = int(bbox[0] / image_size[0] * 1000)
                        z_max = int(bbox[1] / image_size[0] * 1000)
                        y_min = int(bbox[2] / image_size[1] * 1000)
                        y_max = int(bbox[3] / image_size[1] * 1000)
                        x_min = int(bbox[4] / image_size[2] * 1000)
                        x_max = int(bbox[5] / image_size[2] * 1000)

                        bbox_str = f"[{z_min}, {y_min}, {x_min}, {z_max}, {y_max}, {x_max}]"
                        bbox_str_list.append(bbox_str)

                    if len(bbox_str_list) > 0:
                        bbox_str = " ".join(bbox_str_list)
                        bbox_str = "(" + bbox_str + ")"
                        report = report.replace(
                            f"<{region.replace(' ', '_')}>",
                            f"<{region.replace(' ', '_')}> {bbox_str}",
                        )

            data_raw.append({
                "id": d["id"],
                "messages": [
                    {"role": "user", "content": d["messages"][0]["content"]},
                    {"role": "assistant", "content": "Findings:\n" + d["findings"] + "\n\nImpression:\n" + d["impression"]},
                ],
                "videos": d["videos"],
            })
            data_wo.append({
                "id": d["id"],
                "messages": [
                    {"role": "user", "content": d["messages"][0]["content"]},
                    {"role": "assistant", "content": d["messages"][1]["content"]},
                ],
                "videos": d["videos"],
            })
            data_with.append({
                "id": d["id"],
                "messages": [
                    {"role": "user", "content": d["messages"][0]["content"]},
                    {"role": "assistant", "content": report},
                ],
                "videos": d["videos"],
            })

    print(len(data_wo), len(data_with))
    with open(output_raw_path, "w") as f:
        json.dump(data_raw, f, indent=4)
    with open(output_wo_path, "w") as f:
        json.dump(data_wo, f, indent=4)
    with open(output_with_path, "w") as f:
        json.dump(data_with, f, indent=4)
