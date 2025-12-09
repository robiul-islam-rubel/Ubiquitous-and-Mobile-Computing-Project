import os
import json
import pandas as pd
from pathlib import Path
import re

def format_sign_name(name: str) -> str:
    """
    Format a traffic sign name according to the custom rule:
    - Keep camelCase if already present.
    - If single word → lowercase
    - If two words → first lowercase + second word capitalized
    - If space-separated multiwords → follow the camelCase rule
    """

    name = name.strip()

    # === if already camelCase like "doNotEnter" or "bicycleCrossing" ===
    if re.search(r"[a-z][A-Z]", name):
        return name

    # else normalize lowercase and split words
    words = name.lower().split()

    if len(words) == 1:
        return words[0]
    else:
        # capitalize all after the first
        return words[0] + ''.join(w.capitalize() for w in words[1:])

def evaluate_predictions(csv_path, json_folder):
    """
    Compare VLM predictions (traffic_sign_name) against ground-truth (Annotation tag)
    and compute per-image correctness and overall accuracy.
    """

    # --- Load ground truth CSV ---
    df = pd.read_csv(csv_path)
    gt_map = {}
    for _, row in df.iterrows():
        filename = Path(row["Filename"]).name
        gt_map[filename] = row["Annotation tag"]

    # Create a stem map for matching without extension
    gt_map_stem = {Path(k).stem: v for k, v in gt_map.items()}

    # --- Compare predictions ---
    results = []
    for json_file in sorted(Path(json_folder).glob("*.json")):
        fname_stem = Path(json_file.name).stem  # e.g., image0
        if fname_stem not in gt_map_stem:
            print(f"[WARN] No ground truth for {fname_stem}, skipping.")
            continue

        with open(json_file, "r") as f:
            pred = json.load(f)

        pred_name = format_sign_name(pred.get("traffic_sign_name", ""))
        conf = pred.get("confidence", 0.0)
        gt_name = format_sign_name(gt_map_stem[fname_stem])

        correct = (pred_name == gt_name)

        results.append({
            "filename": fname_stem,
            "ground_truth": gt_name,
            "prediction": pred_name,
            "confidence": conf,
            "correct": correct
        })

    # --- Compute accuracy ---
    df_res = pd.DataFrame(results)
    accuracy = df_res["correct"].mean() if not df_res.empty else 0.0

    print(f"\nEvaluated {len(df_res)} samples")
    print(f"Overall accuracy: {accuracy*100:.2f}%")

    df_res.to_csv("./1_Datasets/ModelPredictions.csv", index=False)

    return df_res, accuracy

if __name__ =="__main__":
    csv_path = "./1_Datasets/Classes.csv"          
    json_folder = "./1_Datasets/llama4scout"           

    df_results, acc = evaluate_predictions(csv_path, json_folder)
    print(df_results.head())
