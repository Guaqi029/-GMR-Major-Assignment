import argparse
import csv
import os
import pickle
from pathlib import Path

import numpy as np


METRIC_DEFINITIONS = {
    "joint_velocity_peak": {
        "zh": "关节速度峰值",
        "physical_meaning": "机器人所有关节角速度的最大绝对值，反映是否出现瞬时过快动作或尖峰。",
    },
    "joint_jerk_proxy": {
        "zh": "关节抖动代理",
        "physical_meaning": "关节速度变化率的平均绝对值，近似反映动作是否平滑、是否存在高频抖动。",
    },
    "contact_phase_wrist_jitter": {
        "zh": "接触阶段手腕抖动",
        "physical_meaning": "仅在接触阶段统计机器人手腕轨迹的高频变化强度，反映操作时手部是否稳定。",
    },
    "foot_sliding_proxy": {
        "zh": "足端滑移代理",
        "physical_meaning": "脚接近地面时的水平滑动速度均值，反映站立或支撑时脚底是否打滑。",
    },
    "joint_limit_proximity_ratio": {
        "zh": "关节逼近极限比例",
        "physical_meaning": "关节位置接近机械上下限的时间占比，反映姿态是否逼近不可执行或不自然区域。",
    },
    "human_robot_wrist_consistency": {
        "zh": "人机手腕运动一致性",
        "physical_meaning": "接触阶段人手腕与机器人手腕运动方向的平均相似度，反映操作语义是否被保留。",
    },
    "contact_ratio": {
        "zh": "接触阶段占比",
        "physical_meaning": "整段动作中被判定为接触阶段的帧比例，用于描述交互动作的时长占比。",
    },
}


def load_metrics(pkl_path):
    with open(pkl_path, "rb") as f:
        motion_data = pickle.load(f)
    metadata = motion_data.get("metadata") or {}
    metrics = metadata.get("motion_metrics") or {}
    contact_labels = metadata.get("contact_labels")
    if contact_labels is not None:
        contact_ratio = float(np.mean(np.asarray(contact_labels) > 0))
    else:
        contact_ratio = 0.0
    return {
        "file": str(pkl_path),
        "joint_velocity_peak": float(metrics.get("joint_velocity_peak", 0.0)),
        "joint_jerk_proxy": float(metrics.get("joint_jerk_proxy", 0.0)),
        "contact_phase_wrist_jitter": float(metrics.get("contact_phase_wrist_jitter", 0.0)),
        "foot_sliding_proxy": float(metrics.get("foot_sliding_proxy", 0.0)),
        "joint_limit_proximity_ratio": float(metrics.get("joint_limit_proximity_ratio", 0.0)),
        "human_robot_wrist_consistency": float(metrics.get("human_robot_wrist_consistency", 0.0)),
        "contact_ratio": contact_ratio,
        "phase_aware": bool(metadata.get("phase_aware", False)),
    }


def collect_results(folder):
    results = []
    for path in sorted(Path(folder).rglob("*.pkl")):
        results.append(load_metrics(path))
    return results


def write_csv(results, output_csv):
    fieldnames = [
        "file",
        "joint_velocity_peak",
        "joint_jerk_proxy",
        "contact_phase_wrist_jitter",
        "foot_sliding_proxy",
        "joint_limit_proximity_ratio",
        "human_robot_wrist_consistency",
        "contact_ratio",
        "phase_aware",
    ]
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def write_metric_definitions(output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "chinese_name", "physical_meaning"],
        )
        writer.writeheader()
        for metric, info in METRIC_DEFINITIONS.items():
            writer.writerow(
                {
                    "metric": metric,
                    "chinese_name": info["zh"],
                    "physical_meaning": info["physical_meaning"],
                }
            )


def print_summary(results):
    if not results:
        print("No result files found.")
        return
    peaks = np.array([row["joint_velocity_peak"] for row in results], dtype=float)
    jerks = np.array([row["joint_jerk_proxy"] for row in results], dtype=float)
    wrist_jitter = np.array([row["contact_phase_wrist_jitter"] for row in results], dtype=float)
    foot_sliding = np.array([row["foot_sliding_proxy"] for row in results], dtype=float)
    limit_ratio = np.array([row["joint_limit_proximity_ratio"] for row in results], dtype=float)
    wrist_consistency = np.array([row["human_robot_wrist_consistency"] for row in results], dtype=float)
    contacts = np.array([row["contact_ratio"] for row in results], dtype=float)
    print(f"Files: {len(results)}")
    print(f"Mean joint_velocity_peak: {peaks.mean():.6f} | {METRIC_DEFINITIONS['joint_velocity_peak']['zh']} | {METRIC_DEFINITIONS['joint_velocity_peak']['physical_meaning']}")
    print(f"Mean joint_jerk_proxy: {jerks.mean():.6f} | {METRIC_DEFINITIONS['joint_jerk_proxy']['zh']} | {METRIC_DEFINITIONS['joint_jerk_proxy']['physical_meaning']}")
    print(f"Mean contact_phase_wrist_jitter: {wrist_jitter.mean():.6f} | {METRIC_DEFINITIONS['contact_phase_wrist_jitter']['zh']} | {METRIC_DEFINITIONS['contact_phase_wrist_jitter']['physical_meaning']}")
    print(f"Mean foot_sliding_proxy: {foot_sliding.mean():.6f} | {METRIC_DEFINITIONS['foot_sliding_proxy']['zh']} | {METRIC_DEFINITIONS['foot_sliding_proxy']['physical_meaning']}")
    print(f"Mean joint_limit_proximity_ratio: {limit_ratio.mean():.6f} | {METRIC_DEFINITIONS['joint_limit_proximity_ratio']['zh']} | {METRIC_DEFINITIONS['joint_limit_proximity_ratio']['physical_meaning']}")
    print(f"Mean human_robot_wrist_consistency: {wrist_consistency.mean():.6f} | {METRIC_DEFINITIONS['human_robot_wrist_consistency']['zh']} | {METRIC_DEFINITIONS['human_robot_wrist_consistency']['physical_meaning']}")
    print(f"Mean contact_ratio: {contacts.mean():.6f} | {METRIC_DEFINITIONS['contact_ratio']['zh']} | {METRIC_DEFINITIONS['contact_ratio']['physical_meaning']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_csv", default="results/omomo_retarget_metrics.csv")
    parser.add_argument("--definitions_csv", default="results/omomo_metric_definitions.csv")
    args = parser.parse_args()

    results = collect_results(args.input_folder)
    write_csv(results, args.output_csv)
    write_metric_definitions(args.definitions_csv)
    print_summary(results)
    print(f"Saved metrics to {args.output_csv}")
    print(f"Saved metric definitions to {args.definitions_csv}")


if __name__ == "__main__":
    main()
