import json
import os
from tqdm import tqdm


image_dir = "path/to/CT-RATE/preprocessed_npy"
rgcc_path = "path/to/RadGenome-ChestCT/region_train.json"
save_path = "work_dirs/dataset/rgcc_det_train.json"

anatomy2name = {
    "lung_left": "left lung",
    "lung_right": "right lung",
    "trachea_and_bronchie": "trachea and bronchie",
    "heart": "heart",
    "esophagus": "esophagus",
    "thyroid_left": "left thyroid",
    "thyroid_right": "right thyroid",
}


def generate_Q(anatomy):
    prompt = f"Please locate the {anatomy2name[anatomy]} in the provided CT volume."
    return prompt.strip()


def generate_A(z_min, z_max, y_min, y_max, x_min, x_max):
    prompt = f"[{z_min}, {y_min}, {x_min}, {z_max}, {y_max}, {x_max}]"
    return prompt.strip()


if __name__ == "__main__":
    with open(rgcc_path, "r") as f:
        rgcc_data = json.load(f)

    anns = []
    for filename, info in tqdm(rgcc_data.items()):
        image_size = info["image_size"]
        data_type = filename.split("_")[0]
        patient_id = filename.split("_")[1]
        scan_id = filename.split("_")[2]
        reconstruction_id = filename.split("_")[3]
        file_path = os.path.join(
            image_dir,
            data_type,
            f"{data_type}_{patient_id}",
            f"{data_type}_{patient_id}_{scan_id}",
            f"{data_type}_{patient_id}_{scan_id}_{reconstruction_id}.npy",
        )

        for anatomy, bbox in info.items():
            if anatomy == "image_size":
                continue

            z_min = int((max(0, bbox[0]) / image_size[0]) * 1000)
            z_max = int((min(image_size[0] - 1, bbox[1]) / image_size[0]) * 1000)
            y_min = int((max(0, bbox[2]) / image_size[1]) * 1000)
            y_max = int((min(image_size[1] - 1, bbox[3]) / image_size[1]) * 1000)
            x_min = int((max(0, bbox[4]) / image_size[2]) * 1000)
            x_max = int((min(image_size[2] - 1, bbox[5]) / image_size[2]) * 1000)

            ann = {
                "id": f"{filename}_{anatomy}",
                "messages": [
                    {
                        "role": "user",
                        "content": "<video>\n" + generate_Q(anatomy),
                    },
                    {
                        "role": "assistant",
                        "content": generate_A(z_min, z_max, y_min, y_max, x_min, x_max),
                    },
                ],
                "videos": [file_path],
                "anatomy": anatomy,
                "image_size": image_size,
            }
            anns.append(ann)
    print(f"Total {len(anns)} annotations generated.")

    with open(save_path, "w") as f:
        json.dump(anns, f, indent=4)
