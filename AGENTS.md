# Repository Guidelines

## Project Structure & Module Organization
- `GMR/general_motion_retargeting/`: core Python package (IK, kinematics, retargeting pipeline, utilities).
- `GMR/scripts/`: runnable entry points for conversion, batch processing, and visualization (for example `smplx_to_robot.py`, `bvh_to_robot.py`, `vis_robot_motion.py`).
- `GMR/general_motion_retargeting/ik_configs/`: JSON mapping files such as `smplx_to_g1.json` and `bvh_to_talos.json`.
- `GMR/assets/`: robot XML/URDF definitions, meshes, and body-model assets.
- `GMR/third_party/`: vendored dependencies; avoid editing unless syncing upstream.
- `reference/`: paper/reference material, not runtime code.

## Build, Test, and Development Commands
Run commands from `GMR/`.

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
```

- `python scripts/smplx_to_robot.py --smplx_file <in.npz> --robot unitree_g1 --save_path <out.pkl>`: single-file SMPL-X retargeting.
- `python scripts/bvh_to_robot.py --bvh_file <in.bvh> --robot unitree_g1 --save_path <out.pkl>`: single-file BVH retargeting.
- `python scripts/smplx_to_robot_dataset.py` / `python scripts/bvh_to_robot_dataset.py`: dataset-level conversion.
- `python scripts/vis_robot_motion.py --robot <robot> --robot_motion_path <out.pkl>`: visual verification.

## Coding Style & Naming Conventions
- Use Python 3.10+, 4-space indentation, and PEP 8 naming (`snake_case` for functions/files, `PascalCase` for classes).
- Keep script names action-oriented: `<input>_to_<target>.py` and `vis_*.py`.
- Keep new robot configs in `ik_configs/` with existing naming pattern (`smplx_to_<robot>.json`).
- Prefer small, explicit functions in core modules; keep CLI parsing in `scripts/`.

## Testing Guidelines
- No formal pytest suite is currently configured; use script-based regression checks.
- Validate with at least one conversion command and one visualization command before opening a PR.
- For difficult motions, use examples in `GMR/TEST_MOTIONS.md` and record pass/fail notes in PR description.

## Commit & Pull Request Guidelines
- Current history is minimal (`init`), so use clear conventional-style commit messages, e.g. `feat: add smplx_to_openloong ik config` or `fix: clamp knee joint limits in retarget loop`.
- Keep commits focused (code, config, and assets only when directly related).
- PRs should include: purpose, key changes, commands run, affected robot/data formats, and screenshots or short videos for motion-quality changes.
- Link related issues and call out large asset additions explicitly.

## Security & Configuration Tips
- Do not commit proprietary motion data, credentials, or licensed body-model files.
- Keep large generated outputs (`.pkl`, videos) out of Git unless required for reproducible benchmarks.
