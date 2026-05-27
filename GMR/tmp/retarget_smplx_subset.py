import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)


DEFAULT_SOURCES = {
    "KIT": "/mnt/e/数据集/AMASS/KIT",
    "CMU": "/mnt/e/数据集/AMASS/CMU",
    "BMLmovi": "/mnt/e/数据集/AMASS/BMLmovi",
    "OMOMO": "/mnt/e/数据集/OMOMO/OMOMO_smplx",
}


def collect_motion_files(src_dir: str) -> list[str]:
    root = Path(src_dir)
    files = sorted(str(p) for p in root.rglob("*") if p.suffix in {".npz", ".pkl"})
    return [p for p in files if not p.endswith("_stagei.npz")]


def select_files(files: list[str], limit: int) -> list[str]:
    return files[:limit]


def retarget_one_file(smplx_file: str, robot: str, save_path: str, smplx_folder: Path) -> None:
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        smplx_file, smplx_folder
    )
    frame_list, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
    )

    qpos_list = [retarget.retarget(frame).copy() for frame in frame_list]
    qpos_array = np.array(qpos_list)

    motion_data = {
        "fps": aligned_fps,
        "root_pos": qpos_array[:, :3],
        "root_rot": qpos_array[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos_array[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)


def build_save_path(out_dir: str, dataset_name: str, src_file: str) -> str:
    stem = Path(src_file).stem
    return str(Path(out_dir) / dataset_name / f"{stem}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument(
        "--out_dir",
        default="/mnt/e/数据集/GMR_subset_outputs",
        help="Directory for retargeted robot motions.",
    )
    parser.add_argument(
        "--per_dataset",
        type=int,
        default=5,
        help="How many motions to retarget from each dataset.",
    )
    parser.add_argument(
        "--manifest_path",
        default="/mnt/e/数据集/GMR_subset_outputs/manifest.txt",
        help="Text file recording selected source files.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    smplx_folder = here.parent / "assets" / "body_models"

    selected = []
    for dataset_name, src_dir in DEFAULT_SOURCES.items():
        files = collect_motion_files(src_dir)
        chosen = select_files(files, args.per_dataset)
        if not chosen:
            print(f"[WARN] no valid motion files found for {dataset_name}: {src_dir}")
            continue
        selected.extend((dataset_name, f) for f in chosen)

    os.makedirs(Path(args.manifest_path).parent, exist_ok=True)
    with open(args.manifest_path, "w", encoding="utf-8") as f:
        for dataset_name, src_file in selected:
            f.write(f"{dataset_name}\t{src_file}\n")

    total = len(selected)
    print(f"selected {total} files")

    for idx, (dataset_name, src_file) in enumerate(selected, start=1):
        save_path = build_save_path(args.out_dir, dataset_name, src_file)
        print(f"[{idx}/{total}] retargeting {src_file}")
        retarget_one_file(src_file, args.robot, save_path, smplx_folder)
        print(f"[{idx}/{total}] saved {save_path}")

    print("done")


if __name__ == "__main__":
    main()
