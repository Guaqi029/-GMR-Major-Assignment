import argparse
import json
import pathlib
import os
import multiprocessing as mp

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from natsort import natsorted
from rich import print
import torch
import pickle

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
    get_omomo_object_trajectory,
)
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import IK_CONFIG_ROOT
from general_motion_retargeting.utils.omomo_contact import detect_contact_phase
from general_motion_retargeting.utils.motion_smoothing import smooth_selected_dofs
import gc
import time
import psutil
import tracemalloc


def check_memory(threshold_gb=30):  # adjust based on your available memory
    mem = psutil.virtual_memory()
    used_memory_gb = (mem.total - mem.available) / (1024 ** 3)
    available_memory_gb = mem.available / (1024 ** 3)
    if available_memory_gb < threshold_gb:
        print(f"[WARNING] Memory usage:{used_memory_gb:.2f} GB, available:{available_memory_gb:.2f} GB, exceeding the threshold of {threshold_gb} GB.")
        return True
    return False


HERE = pathlib.Path(__file__).parent


def get_upper_body_dof_indices(retargeter):
    keywords = ("waist", "shoulder", "elbow", "wrist")
    indices = []
    for dof_name, dof_index in retargeter.robot_dof_names.items():
        if any(keyword in dof_name for keyword in keywords):
            robot_dof_index = dof_index - 6
            if robot_dof_index >= 0:
                indices.append(robot_dof_index)
    return sorted(set(indices))


def compute_motion_metrics(dof_pos, fps):
    if len(dof_pos) < 2:
        return {"joint_velocity_peak": 0.0, "joint_jerk_proxy": 0.0}
    velocity = np.diff(dof_pos, axis=0) * fps
    jerk_proxy = np.diff(velocity, axis=0) * fps if len(velocity) > 1 else np.zeros_like(velocity)
    return {
        "joint_velocity_peak": float(np.max(np.abs(velocity))),
        "joint_jerk_proxy": float(np.mean(np.abs(jerk_proxy))) if jerk_proxy.size else 0.0,
    }


def _first_body_index(body_names, candidates):
    for name in candidates:
        if name in body_names:
            return body_names.index(name)
    return None


def compute_contact_phase_wrist_jitter(body_pos, body_names, contact_labels, fps):
    if contact_labels is None or len(body_pos) < 2:
        return 0.0
    left_idx = _first_body_index(body_names, ("left_wrist_yaw_link", "left_wrist_pitch_link", "left_rubber_hand"))
    right_idx = _first_body_index(body_names, ("right_wrist_yaw_link", "right_wrist_pitch_link", "right_rubber_hand"))
    values = []
    for idx in (left_idx, right_idx):
        if idx is None:
            continue
        positions = body_pos[:, idx, :]
        velocity = np.diff(positions, axis=0) * fps
        if len(velocity) < 2:
            continue
        jerk = np.diff(velocity, axis=0) * fps
        valid = contact_labels[:-2]
        if np.any(valid):
            values.append(float(np.mean(np.linalg.norm(jerk[valid], axis=1))))
    return float(np.mean(values)) if values else 0.0


def compute_foot_sliding_proxy(body_pos, body_names, fps, height_threshold=0.04):
    foot_indices = [
        _first_body_index(body_names, ("left_toe_link", "left_ankle_roll_link", "left_foot")),
        _first_body_index(body_names, ("right_toe_link", "right_ankle_roll_link", "right_foot")),
    ]
    values = []
    for idx in foot_indices:
        if idx is None:
            continue
        foot_pos = body_pos[:, idx, :]
        horizontal_velocity = np.linalg.norm(np.diff(foot_pos[:, :2], axis=0) * fps, axis=1)
        near_ground = foot_pos[:-1, 2] < height_threshold
        if np.any(near_ground):
            values.append(float(np.mean(horizontal_velocity[near_ground])))
    return float(np.mean(values)) if values else 0.0


def compute_joint_limit_proximity_ratio(retargeter, qpos_array, threshold=0.1):
    values = []
    for joint_id in range(retargeter.model.njnt):
        joint_type = retargeter.model.jnt_type[joint_id]
        if joint_type not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            continue
        if not retargeter.model.jnt_limited[joint_id]:
            continue
        qpos_adr = retargeter.model.jnt_qposadr[joint_id]
        if qpos_adr >= qpos_array.shape[1]:
            continue
        joint_range = retargeter.model.jnt_range[joint_id]
        low, high = float(joint_range[0]), float(joint_range[1])
        span = high - low
        if span <= 0:
            continue
        joint_values = qpos_array[:, qpos_adr]
        normalized_margin = np.minimum(joint_values - low, high - joint_values) / span
        values.append(normalized_margin < threshold)
    if not values:
        return 0.0
    return float(np.mean(np.stack(values, axis=1)))


def compute_human_robot_wrist_consistency(body_pos, body_names, human_frames, contact_labels):
    if contact_labels is None or len(body_pos) < 2:
        return 0.0
    robot_indices = {
        "left_wrist": _first_body_index(body_names, ("left_wrist_yaw_link", "left_wrist_pitch_link", "left_rubber_hand")),
        "right_wrist": _first_body_index(body_names, ("right_wrist_yaw_link", "right_wrist_pitch_link", "right_rubber_hand")),
    }
    scores = []
    valid_pairs = contact_labels[:-1] & contact_labels[1:]
    if not np.any(valid_pairs):
        return 0.0
    for human_name, robot_idx in robot_indices.items():
        if robot_idx is None:
            continue
        robot_vel = np.diff(body_pos[:, robot_idx, :], axis=0)[valid_pairs]
        human_positions = np.array([np.asarray(frame[human_name][0], dtype=float) for frame in human_frames])
        human_vel = np.diff(human_positions, axis=0)[valid_pairs]
        for hv, rv in zip(human_vel, robot_vel):
            human_norm = np.linalg.norm(hv)
            robot_norm = np.linalg.norm(rv)
            if human_norm < 1e-6 or robot_norm < 1e-6:
                continue
            scores.append(float(np.dot(hv, rv) / (human_norm * robot_norm)))
    return float(np.mean(scores)) if scores else 0.0


def process_file(
    smplx_file_path,
    tgt_file_path,
    tgt_robot,
    SMPLX_FOLDER,
    tgt_folder,
    total_files,
    memory_threshold_gb,
    device,
    phase_aware,
    contact_threshold,
    contact_min_frames,
    smoothing_window,
    save_local_body_pos,
    compute_metrics,
    height_adjust,
    verbose=False,
):
    def log_memory(message):
        if verbose:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
            print(f"[MEMORY] {message}: {memory_usage:.2f} GB")
    
    # Start memory tracking if verbose
    if verbose:
        tracemalloc.start()
        
    # Initial checks (with optional logging)
    log_memory("Initial memory usage")
    
    num_pause = 0
    while check_memory(memory_threshold_gb):
        print(f"[PAUSE] Paused processing {smplx_file_path} to prevent memory overflow. num_pause: {num_pause}")
        time.sleep(60*2)
        num_pause += 1
        if num_pause > 10:
            print(f"[ERROR] Memory usage is still high after 10 pauses. Exiting.")
            return

    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(smplx_file_path, SMPLX_FOLDER)
        mocap_frame_rate = smplx_data["mocap_frame_rate"]
        log_memory("After loading SMPL-X data")
    except Exception as e:
        print(f"Error loading {smplx_file_path}: {e}")
        return
    
  
    tgt_fps = 30
    try:
        smplx_frame_data_list, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return

    object_positions = None
    contact_labels = None
    contact_min_distances = None
    if phase_aware:
        object_positions = get_omomo_object_trajectory(smplx_data, tgt_fps=tgt_fps)
        if object_positions is not None:
            contact_labels, contact_min_distances = detect_contact_phase(
                smplx_frame_data_list,
                object_positions,
                threshold=contact_threshold,
                min_frames=contact_min_frames,
            )
    
    # retarget
    retargeter = GMR(
        src_human="smplx",
        tgt_robot=tgt_robot,
        actual_human_height=actual_human_height,
        phase_aware=phase_aware,
    )
    qpos_list = []
    human_frames_for_metrics = []
    for frame_idx, smplx_frame_data in enumerate(smplx_frame_data_list):
        if phase_aware and contact_labels is not None and frame_idx < len(contact_labels):
            retargeter.set_phase("contact" if contact_labels[frame_idx] else None)
        elif phase_aware:
            retargeter.set_phase(None)
        qpos = retargeter.retarget(smplx_frame_data)
        qpos_list.append(qpos.copy())
        human_frames_for_metrics.append(retargeter.scaled_human_data)

    qpos_list = np.array(qpos_list)

    log_memory("After retargeting")

    dof_pos = qpos_list[:, 7:]
    if smoothing_window > 1:
        upper_body_dof_indices = get_upper_body_dof_indices(retargeter)
        dof_pos = smooth_selected_dofs(
            dof_pos,
            upper_body_dof_indices,
            window_size=smoothing_window,
        )
        qpos_list[:, 7:] = dof_pos

    try:
        root_pos = qpos_list[:, :3]
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
    num_frames = root_pos.shape[0]

    need_kinematics = save_local_body_pos or compute_metrics or height_adjust
    kinematics_model = None
    body_names = None
    local_body_pos = None
    final_body_pos = None
    if need_kinematics:
        kinematics_model = KinematicsModel(retargeter.xml_file, device=device)
        body_names = kinematics_model.body_names

        dof_tensor = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)

        if save_local_body_pos:
            fk_root_pos = torch.zeros((num_frames, 3), device=device)
            fk_root_rot = torch.zeros((num_frames, 4), device=device)
            fk_root_rot[:, -1] = 1.0
            local_body_pos, _ = kinematics_model.forward_kinematics(
                fk_root_pos, fk_root_rot, dof_tensor
            )

        if height_adjust:
            body_pos, _ = kinematics_model.forward_kinematics(
                torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
                dof_tensor,
            )
            ground_offset = 0.0
            lowerst_height = torch.min(body_pos[..., 2]).item()
            root_pos[:, 2] = root_pos[:, 2] - lowerst_height + ground_offset

        if compute_metrics:
            final_body_pos, _ = kinematics_model.forward_kinematics(
                torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
                dof_tensor,
            )
            final_body_pos = final_body_pos.detach().cpu().numpy()

        log_memory("After forward kinematics")
        
    ROOT_ORIGIN_OFFSET = True
    if ROOT_ORIGIN_OFFSET:
        # offset using the first frame
        root_pos[:, :2] -= root_pos[0, :2]
    metrics = None
    if compute_metrics:
        metrics = compute_motion_metrics(dof_pos, aligned_fps)
        metrics["contact_phase_wrist_jitter"] = compute_contact_phase_wrist_jitter(
            final_body_pos,
            body_names,
            contact_labels,
            aligned_fps,
        )
        metrics["foot_sliding_proxy"] = compute_foot_sliding_proxy(
            final_body_pos,
            body_names,
            aligned_fps,
        )
        metrics["joint_limit_proximity_ratio"] = compute_joint_limit_proximity_ratio(
            retargeter,
            qpos_list,
        )
        metrics["human_robot_wrist_consistency"] = compute_human_robot_wrist_consistency(
            final_body_pos,
            body_names,
            human_frames_for_metrics,
            contact_labels,
        )
        
    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy() if local_body_pos is not None else None,
        "link_body_list": body_names,
        "metadata": {
            "phase_aware": phase_aware,
            "contact_labels": contact_labels.astype(np.uint8) if contact_labels is not None else None,
            "contact_min_distances": contact_min_distances,
            "motion_metrics": metrics,
        },
    }


    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)
        
    # Progress print based on tgt_folder
    done = 0
    for root, _, files in os.walk(tgt_folder):
        done += len([f for f in files if f.endswith('.pkl')])
    print(f"Processed {done}/{total_files}: {tgt_file_path}")
    
    if verbose:
        # Get memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print("\nTop 10 memory-consuming lines:")
        for stat in top_stats[:10]:
            print(stat)
        
        tracemalloc.stop()
        
    # clean cache
    torch.cuda.empty_cache()
    gc.collect()
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--src_folder", type=str,
                        required=True,
                        )
    parser.add_argument("--tgt_folder", type=str,
                        required=True,
                        )
    
    parser.add_argument("--override", default=False, action="store_true")
    parser.add_argument("--num_cpus", default=4, type=int)
    parser.add_argument(
        "--memory_threshold_gb",
        default=2.0,
        type=float,
        help="Pause workers when available system memory drops below this threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device for forward kinematics, e.g. cpu or cuda:0. Defaults to cuda:0 when available, otherwise cpu.",
    )
    parser.add_argument("--phase_aware", default=False, action="store_true")
    parser.add_argument("--contact_threshold", default=0.35, type=float)
    parser.add_argument("--contact_min_frames", default=3, type=int)
    parser.add_argument("--smoothing_window", default=5, type=int)
    parser.add_argument("--fast_mode", default=False, action="store_true")
    parser.add_argument("--save_local_body_pos", default=False, action="store_true")
    parser.add_argument("--compute_metrics", default=False, action="store_true")
    parser.add_argument("--height_adjust", default=False, action="store_true")
    args = parser.parse_args()

    if args.fast_mode:
        args.smoothing_window = 1
        args.save_local_body_pos = False
        args.compute_metrics = False
        args.height_adjust = False
    
    # print the total number of cpus and gpus
    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Memory threshold: {args.memory_threshold_gb:.1f} GB")
    print(f"Phase-aware retargeting: {args.phase_aware}")
    print(f"Fast mode: {args.fast_mode}")
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    hard_motions_folder = HERE / ".." / "assets" / "hard_motions"

    verbose = False

    hard_motions_paths = [hard_motions_folder / "0.txt", 
                          hard_motions_folder / "1.txt"]
    hard_motions = []
    for hard_motions_path in hard_motions_paths:
        with open(hard_motions_path, "r") as f:
            for line in f:
                if "Motion:" in line:
                    motion_path = line.split(":")[1].strip()
                else:
                    continue
                motion_path = motion_path.split(",")[0].strip().split(".")[0]
                hard_motions.append(motion_path)
                
                
    args_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.npz"):
                continue
            if filename.endswith((".pkl", ".npz")):
                smplx_file_path = os.path.join(dirpath, filename)
                tgt_file_path = smplx_file_path.replace(src_folder, tgt_folder).replace(".npz", ".pkl")
                if not os.path.exists(tgt_file_path) or args.override:
                    args_list.append((smplx_file_path, tgt_file_path, args.robot, SMPLX_FOLDER, tgt_folder))
    print("full args_list:", len(args_list))
    
    # remove hard and infeasible motions
    exclude_file_content = ["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]
    
    new_args_list = []
    for arguments in args_list:
        motion_name = arguments[0].split("/")[-1].split('.')[0]
        if motion_name in hard_motions:
            continue
        if any(content in motion_name for content in exclude_file_content):
            continue
        new_args_list.append(arguments)
    args_list = new_args_list
    
    
    print("new args_list:", len(args_list))
    
    total_files = len(args_list)
    print(f"Total number of files to process: {total_files}")
    with mp.Pool(args.num_cpus) as pool:
        pool.starmap(
            process_file,
            [
                process_args
                + (
                    total_files,
                    args.memory_threshold_gb,
                    device,
                    args.phase_aware,
                    args.contact_threshold,
                    args.contact_min_frames,
                    args.smoothing_window,
                    args.save_local_body_pos,
                    args.compute_metrics,
                    args.height_adjust,
                    verbose,
                )
                for process_args in args_list
            ],
        )

    print("Done. Saved to ", tgt_folder)


if __name__ == "__main__":
    main()
