# 5-Day GMR 实验落地计划（仅 GMR，不跑完整 TWIST）

## 目标与范围
- 目标：在 MuJoCo 中完成 GMR 动作重定向实验，做一个小创新并形成可写入课程报告的结果。
- 范围：只使用 `GMR` 仓库，不做真实机器人部署，不复现 TWIST 全链路 RL 训练。

## 创新点建议（选 1 个）
- 方案A（推荐）：调整 `ik_configs` 中关键身体部位（pelvis/feet/hands）的 position/rotation 权重，降低脚滑与抖动。
- 方案B：对输出关节角添加简单平滑（如移动平均）并比较平滑性与跟踪误差。

## Day 1：环境与数据准备
- 创建环境并安装：`conda create -n gmr python=3.10 -y && conda activate gmr && pip install -e .`
- 下载数据：SMPL-X body models、AMASS（主）、OMOMO（可选补充）。
  - 先下这 4 个就够你做一次作业实验了（体量和多样性平衡）：
  1. CMU（动作多、经典）
  2. KIT（过渡动作丰富）
  3. BMLmovi（多人多动作）
  4. EyesJapanDataset（动作类型补充）
  如果时间和硬盘还够，再加：
  5. HumanEva
  6. Transitions_mocap
- 验证目录：`assets/body_models/smplx/SMPLX_*.pkl` 可被脚本读取。
- 产出：环境可运行、至少 20 条动作样本可用。

## Day 2：基线跑通与样本集固定
- 跑通基线命令：
  - `python scripts/smplx_to_robot.py --smplx_file <file> --robot unitree_g1 --save_path <out.pkl> --rate_limit`
  - `python scripts/vis_robot_motion.py --robot unitree_g1 --robot_motion_path <out.pkl>`
- 固定实验子集：从 AMASS 选 20 条（行走/挥手/下蹲/转身）。
- 产出：基线结果视频与 pkl 文件。

## Day 3：实现创新方案
- 方案A：复制并修改一个配置（如 `smplx_to_g1.json`），形成 `smplx_to_g1_tuned.json`。
- 方案B：在离线结果上增加平滑后处理脚本（输出新 pkl）。
- 对同一 20 条样本跑“基线 vs 改进”。
- 产出：两组可对比结果。

## Day 4：评估与可视化
- 指标（至少 3 个）：
  - 关节速度均值/峰值（平滑性）
  - 关节速度突变次数（抖动代理）
  - 失败率（明显穿地/失稳/动作崩坏条数）
- 生成表格与图：每条动作一行，汇总平均值与标准差。
- 产出：`results.csv`、对比图、关键案例截图/GIF。

## Day 5：报告定稿
- 报告结构：摘要、方法、实验设置、结果、讨论、局限与未来工作。
- 最少内容清单：
  - 数据来源与筛选规则
  - 命令与参数（可复现）
  - 基线 vs 改进定量表
  - 2~4 个定性案例图
- 产出：可提交版 PDF/Word 报告。

## 建议目录结构
```text
project_root/
  experiments/
    data_split/
    outputs_baseline/
    outputs_tuned/
    metrics/
    figures/
  reference/
    5-day-gmr-experiment-plan.md
```

## 风险与止损
- 若 AMASS 下载慢：先用少量 OMOMO/LAFAN1 样本跑通流程。
- 若指标实现超时：保留“失败率 + 关节速度统计”两项核心指标。
- 若调参效果不明显：改为“不同 `--rate_limit`/权重策略对稳定性影响”的对比研究。
