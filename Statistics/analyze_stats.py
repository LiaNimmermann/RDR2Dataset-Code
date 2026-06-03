import os
import json
from collections import Counter
import pickle

from tqdm import tqdm

without = True
wrong_only =False

def analyse_dataset(save_path, before):

    path = save_path + "/all_captures.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("id_metric_dict_times_min_mse.pkl", "rb") as f:
        metrics_mse_min = pickle.load(f)

    with open("id_metric_dict_png_exr_sim.pkl", "rb") as f:
        metrics_similarity = pickle.load(f)



    weather_counter = Counter()
    zone_counter = Counter()
    fine_class_counter = Counter()
    fine_classes_capture_counter = Counter()
    entities_per_capture_sum = 0
    capture_count = 0

    for _, capture_data in tqdm(data.items()):

        capture = capture_data.get("Capture", {})
        entities = capture_data.get("Entities", [])
        id = capture.get("ID")

        if without and metrics_mse_min[id]<0.0000001 and metrics_similarity[id]<0.0:
            continue

        if wrong_only and metrics_mse_min[id]>=0.0000001 and metrics_similarity[id]>=0.0:
            continue

        if not entities:
            continue

        weather = capture.get("Weather")
        zone = capture.get("Zone")
        if weather:
            weather_counter[weather] += 1
        if zone:
            zone_counter[zone] += 1

        capture_count += 1

        found_fine_classes = set()

        for entity in entities:
            entities_per_capture_sum += 1
            fine_class = entity.get("FineClassName")
            if fine_class:
                fine_class_counter[fine_class] += 1
                if not fine_class in found_fine_classes:
                    fine_classes_capture_counter[fine_class] += 1
                    found_fine_classes.add(fine_class)
                
        
        captured_fine_classes = set(fine_class_counter.keys())
        
        all_fine_classes_path = os.path.join(save_path, "gt_Fine_labelIds_mapping.json")
        with open(all_fine_classes_path, "r", encoding="utf-8") as f:
            all_fine_classes_json = json.load(f)

        all_fine_classes = set(all_fine_classes_json.keys())
        all_fine_classes.discard("background")

        missing_fine_classes = all_fine_classes - captured_fine_classes
        if missing_fine_classes:
            for missing_class in missing_fine_classes:
                fine_class_counter[missing_class] += 0


    results = {
    "weather_counts": dict(sorted(weather_counter.items(), key=lambda x: x[0])),
    "zone_counts": dict(sorted(zone_counter.items(), key=lambda x: x[0])),
    "class_counts": dict(sorted(fine_class_counter.items(), key=lambda x: x[0])),
    "fine_classes_capture_counts": dict(sorted(fine_classes_capture_counter.items(), key=lambda x: x[0])),
    "fine_classes_avg_entities_per_capture": {fine_class: fine_class_counter[fine_class] / fine_classes_capture_counter[fine_class] if fine_classes_capture_counter[fine_class] > 0 else 0 for fine_class in fine_class_counter},
    "avg_entities_per_capture": entities_per_capture_sum / capture_count if capture_count > 0 else 0
}
    if without:
        save_path = os.path.join("stats_summary_all_classes_wo_sim_mse.json")
    elif wrong_only:
        save_path = os.path.join("stats_summary_all_classes_wrong_only.json")
    else:
        save_path = os.path.join("stats_summary_all_classes_final.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)



analyse_dataset("/home/lnimmermann/Code/RDR2Dataset-Code/Statistics", False)
print("Done")
