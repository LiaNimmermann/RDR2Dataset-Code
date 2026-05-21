import os
import json
from collections import Counter

def analyse_dataset(save_path, before):

    path = save_path + "/all_captures.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    weather_counter = Counter()
    zone_counter = Counter()
    fine_class_counter = Counter()

    for _, capture_data in data.items():

        capture = capture_data.get("Capture", {})
        entities = capture_data.get("Entities", [])


        if not entities:
            continue

        weather = capture.get("Weather")
        zone = capture.get("Zone")
        if weather:
            weather_counter[weather] += 1
        if zone:
            zone_counter[zone] += 1


        for entity in entities:
            fine_class = entity.get("FineClassName")
            if fine_class:
                fine_class_counter[fine_class] += 1
        
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
}

    save_path = os.path.join(save_path, "stats_summary_all_classes_final.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)



analyse_dataset("/media/lstracke/T5 EVO/RDR2_dataset_processed", False)
