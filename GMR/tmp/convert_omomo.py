import os
import pickle

import joblib
import numpy as np


MOTION_PATHS = [
    "/mnt/e/数据集/OMOMO/data/data/train_diffusion_manip_seq_joints24.p",
    "/mnt/e/数据集/OMOMO/data/data/test_diffusion_manip_seq_joints24.p",
]

TARGET_DIR = "/mnt/e/数据集/OMOMO/OMOMO_smplx"


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    for motion_path in MOTION_PATHS:
        print(f"loading {motion_path}")
        motion_data = joblib.load(motion_path)

        for smpl_data in motion_data.values():
            seq_name = smpl_data["seq_name"]
            num_frames = smpl_data["pose_body"].shape[0]

            poses = np.concatenate(
                [smpl_data["pose_body"], np.zeros((num_frames, 102))],
                axis=1,
            )
            smpl_data["poses"] = poses
            smpl_data["mocap_frame_rate"] = np.array(30)

            out_path = os.path.join(TARGET_DIR, f"{seq_name}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(smpl_data, f)

            print(f"saved {seq_name}")

    print("done")


if __name__ == "__main__":
    main()
