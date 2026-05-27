import argparse
import pathlib
import os
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
    get_omomo_object_trajectory,
)
from general_motion_retargeting.utils.omomo_contact import detect_contact_phase
from general_motion_retargeting.utils.motion_smoothing import smooth_selected_dofs

from rich import print


def get_upper_body_dof_indices(retargeter):
    keywords = ("waist", "shoulder", "elbow", "wrist")
    indices = []
    for dof_name, dof_index in retargeter.robot_dof_names.items():
        if any(keyword in dof_name for keyword in keywords):
            robot_dof_index = dof_index - 6
            if robot_dof_index >= 0:
                indices.append(robot_dof_index)
    return sorted(set(indices))

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/AMASS/KIT_572_dance_chacha11_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1Running_c3d/Run_C24_-_quick_side_step_left_stageii.npz",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "fourier_gr3"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--transparent_robot",
        default=False,
        action="store_true",
        help="Render the robot transparently so the human reference markers are easier to see.",
    )

    parser.add_argument(
        "--human_point_scale",
        type=float,
        default=0.2,
        help="Scale of the human reference markers.",
    )

    parser.add_argument(
        "--human_x_offset",
        type=float,
        default=0.8,
        help="Offset the human reference markers along X for side-by-side comparison.",
    )

    parser.add_argument(
        "--human_y_offset",
        type=float,
        default=0.0,
        help="Offset the human reference markers along Y for side-by-side comparison.",
    )

    parser.add_argument(
        "--human_z_offset",
        type=float,
        default=0.0,
        help="Offset the human reference markers along Z for side-by-side comparison.",
    )

    parser.add_argument("--phase_aware", default=False, action="store_true")
    parser.add_argument("--contact_threshold", default=0.35, type=float)
    parser.add_argument("--contact_min_frames", default=3, type=int)
    parser.add_argument("--smoothing_window", default=5, type=int)

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        phase_aware=args.phase_aware,
    )
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=int(args.transparent_robot),
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
    

    object_positions = None
    contact_labels = None
    if args.phase_aware:
        object_positions = get_omomo_object_trajectory(smplx_data, tgt_fps=tgt_fps)
        if object_positions is not None:
            contact_labels, _ = detect_contact_phase(
                smplx_data_frames,
                object_positions,
                threshold=args.contact_threshold,
                min_frames=args.contact_min_frames,
            )

    qpos_list = []
    human_frames_for_view = []
    for frame_idx, smplx_frame_data in enumerate(smplx_data_frames):
        if args.phase_aware and contact_labels is not None and frame_idx < len(contact_labels):
            retarget.set_phase("contact" if contact_labels[frame_idx] else None)
        elif args.phase_aware:
            retarget.set_phase(None)
        qpos = retarget.retarget(smplx_frame_data)
        qpos_list.append(qpos.copy())
        human_frames_for_view.append(retarget.scaled_human_data)

    qpos_list = np.array(qpos_list)
    if args.smoothing_window > 1:
        upper_body_dof_indices = get_upper_body_dof_indices(retarget)
        qpos_list[:, 7:] = smooth_selected_dofs(
            qpos_list[:, 7:],
            upper_body_dof_indices,
            window_size=args.smoothing_window,
        )

    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
    
    # Start the viewer
    i = -1

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        qpos = qpos_list[i]

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=human_frames_for_view[i],
            # human_motion_data=smplx_data,
            human_point_scale=args.human_point_scale,
            human_pos_offset=np.array([args.human_x_offset, args.human_y_offset, args.human_z_offset]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )
            
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
            "metadata": {
                "phase_aware": args.phase_aware,
                "contact_labels": contact_labels.astype(np.uint8) if contact_labels is not None else None,
            },
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    
    robot_motion_viewer.close()
