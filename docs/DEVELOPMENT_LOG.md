# CAT404 音乐驱动 3D 角色动画系统 — 开发记录

> 项目：An Interactive Music-Driven 3D Character Animation System Using Multi-Level Musical Features
> 记录日期：2026-10-20 · 迭代：Iteration 2（多级映射 + 动作库 + 混合 + 评估 + 交互界面）
> 代码位置：`CAT404_music_animation_starter/`（本轮在副本中开发，最后同步回桌面项目目录）

---

## 0. 本轮工作摘要

上一轮项目已经打通了 “音频 → 节拍 → 事件 → Blender 方块动画” 的最小闭环，并导入了 AIST++ 骨架。但对照 proposal，动作库、多级映射、动作混合、受控实验、可编辑映射规则这些核心内容都还没有实现，动画对象也还是方块。

本轮按 proposal 的迭代计划（5.9 节）完成了：

| 迭代 | 内容 | 状态 |
|---|---|---|
| 1 | 音乐分析与特征提取（beat/tempo/energy/onset/structure） | ✅ 重写并完成 |
| 2 | 动作数据集处理与动作库 | ✅ 1 408 段 AIST++ → 16 737 个动作片段 |
| 3 | 节拍基线动画 | ✅ 保留并统一到新映射引擎（`beat_only`） |
| 4 | 多级音乐-动作映射 | ✅ 五层映射 + 4 个消融条件 |
| 5 | 动作混合 | ✅ Slerp + 时间缩放 + DTW 对齐 + 地面约束 |
| 6 | 交互式界面 | ✅ Streamlit 五页界面（规则可编辑、A/B 评分） |
| 7 | 系统集成与评估 | ✅ BAS / 表现力指标 / 24 组受控实验 |

产出的关键数字：

- 动作库：1 408 个 AIST++ 序列，10 个舞种，**16 737** 个 2 秒片段，描述符 9 维。
- 音乐分析：4 首测试曲（120 / 105.9 / 150 BPM，含 1 首 116 秒的段落测试曲），40–225 个节拍，1–6 个段落。
- 受控实验：4 首曲 × 6 条件 = **24 次完整生成与评估**（59 秒完成，不含渲染）。
- 结果：beat-only 的节拍 F1 更高（0.663 vs 0.514），multi-level 的表现力指标全面更高（能量相关 0.170 vs −0.041；节拍响应相关 0.315 vs −0.126；4/4 曲目胜出），与 proposal 的假设一致。

---

## 1. 起点盘点：接手时的项目状态

在桌面目录 `CAT404_music_animation_starter/` 中已有以下内容（本轮未删除，保留兼容）：

| 文件 | 作用 | 评价 |
|---|---|---|
| `src/music_analysis.py` | librosa 提取 beat / RMS / onset，输出 CSV + 图 | 可用，但只有 3 个特征、无段落、无节拍级表 |
| `src/beat_baseline.py` | 节拍 → `animation_events.csv`（时间 + 强度） | 可用，是最初的 “beat-only” 雏形 |
| `src/multi_level_mapping.py` | RMS 0.7 + onset 0.3 合成强度 | 只有 “强度” 一个维度，没有速度/段落/动作选择 |
| `src/compare_mappings.py` | 比较两份事件强度的差异 | 只比较数值，不是 BAS 评估 |
| `src/animate_aist_skeleton.py` | matplotlib 3D 骨架 GIF | 数据可视化，不参与动画生成 |
| `blender/animate_cube.py`, `animate_multi_level.py` | 方块随节拍弹跳 | 占位动画 |
| `blender/import_aist_skeleton.py` | 球+圆柱骨架关键帧 | 可视化用 |
| `blender/create_aist_rigged.py` | 从 AIST++ 首帧建人形网格 + 11 骨骼 + 自动蒙皮 | 重要的技术基础，本轮复用其思路 |
| `blender/render_from_events.py` | 方块动画 → MP4（VSE 混音） | 本轮复用其 “scene strip + sound strip” 混音方案 |
| `app.py` | Streamlit：上传 WAV → 事件 → 方块 MP4 | 功能单薄，映射不可编辑 |
| `data/aistplusplus/keypoints3d/` | 1 408 个 pkl（约 917 MB） | 动作库原料 |
| `data/audio/test_song.wav` | 23.28 s，120 BPM | 唯一测试曲 |
| `outputs/` | 若干 mp4 / csv | 旧版结果 |

**结论**：已有的是 “节拍驱动的占位动画”，proposal 要求的是 “多级特征驱动的、可从动作库检索并混合、可交互编辑、可评估” 的系统。差距集中在第 2–7 迭代。

---

## 2. 差距分析（proposal vs 接手状态）

| Proposal 要求 | 接手状态 | 本轮实现 | 状态 |
|---|---|---|---|
| 5.2 音乐分析：beat/tempo/RMS/onset/spectral/chroma/MFCC/structure | 仅 beat/RMS/onset | `src/music_features.py`：30 FPS 对齐的帧级特征 + 节拍级特征表 + 基于 SSM/checkerboard 的段落切分 | ✅ |
| 5.3 多级映射：beat→时间、tempo→速度、energy→强度、structure→动作类型 | 只有强度加权 | `src/mapping.py`：五层显式规则 + 声明的映射权重 + 逐槽决策解释 | ✅ |
| 5.4 动作库（AIST++） | 只读单个序列做可视化 | `src/motion_library.py`：全部 1 408 序列切分为 16 737 个片段并计算 9 维描述符 | ✅ |
| 5.5 动作选择 | 无 | 规则化最近邻检索 + 多样性惩罚 + 段落类别约束 | ✅ |
| 5.6 动作混合：Slerp / DTW / 足部修正 | 无 | `src/motion_blend.py`：方向空间 Slerp、时间缩放、DTW 对齐、地面/足部约束、强度缩放 | ✅ |
| 5.7 交互界面：速度/强度/风格/时间轴/关键帧 + 映射规则可编辑 | 仅 2 个滑条 | `app.py` 五页界面：分析 / 映射预览 / 生成渲染 / 评估 / 文档，规则权重全部可调，A/B 评分落盘 | ✅（关键帧编辑未做，见限制） |
| 5.8 Blender 3D 环境 | 方块 | `blender/render_character.py`：人形网格 + 21 骨骼绑定 + 逐帧重定向 + 混音 MP4 + 可选 FBX | ✅ |
| 8. 评估：BAS + 感知同步/表现力 + 特征消融 | 无数值评估 | `src/beat_alignment.py` + `src/experiment.py`：BAS、F1、能量相关、节拍响应相关、动态范围、多样性；6 条件 × 4 曲 | ✅（用户研究仅提供工具，未收集数据） |
| 8. 导出标准动画文件 | 无 | Blender FBX 导出（默认关闭，可 `--export-fbx`），`motion.npz` 含根轨迹/方向/节点 | ✅（BVH 未实现） |

---

## 3. 系统架构

```
音频 WAV
  │
  ▼
src/music_features.py ──────► data/features/<song>/
  │  • 30 FPS 帧级特征（rms/onset/centroid/flux/band ratios）
  │  • 节拍级特征表（ioi/bpm_local/energy_norm/onset_norm/section_class）
  │  • 段落（SSM + checkerboard novelty → 边界 → 能量等级 0/1/2）
  │  • feature_plot.png / summary.json
  ▼
src/mapping.py ─────────────► outputs/runs/<run>/plan.csv
  │  beat → 槽位（slot_beats=2）
  │  tempo → 检索目标 + 有界播放速率
  │  energy → 动作增益 + 能量族检索
  │  onset → 混合时长
  │  structure → 动作族约束
  ▼
data/library/（motion_library.py 一次性构建）
  │  joints.npy / root.npy / bone_lengths.npy / rest_joints.npy
  │  motion_index.csv（16 737 行 × 描述符）
  ▼
src/motion_blend.py ───────► outputs/runs/<run>/motion.npz
  │  时间缩放 → 强度缩放（骨骼角度）→ Slerp 交叉淡入 → DTW 对齐
  │  → 根轨迹连续 + 舞台锚定 → 垂直速度限制 → 足部离地约束
  ▼
blender/render_character.py ─► character.mp4（+ character.blend / FBX）
  │  人形网格（刚性部件）→ 21 骨骼 → 逐帧姿态重定向 → EEVEE → VSE 混音
  ▼
src/beat_alignment.py ──────► metrics.json（BAS / F1 / 相关 / 动态范围）
  ▼
src/experiment.py ──────────► outputs/experiments/*.csv + docs/EXPERIMENT_REPORT.md
```

所有中间产物都落在以 “曲目 + 条件” 命名的 run 目录里，任意一个实验数字都能回溯到 `plan.csv` 与 `mapping_explanation.txt`。

---

## 4. 迭代记录（本次实际执行）

### 4.1 运行环境与一个关键坑

- 项目自带 `.venv`（Python 3.14.2），已装 librosa 1.0.0、numpy 2.5.4、pandas 3.0.6、scipy 1.18.1、matplotlib 3.11.2、streamlit 1.66.0、soundfile 0.14.0。Blender 5.2.2 位于 `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`。
- **坑：`librosa.load` 在受控环境下无限挂起。** 用 `faulthandler` dump 调用栈定位到 `numba.core.caching.ensure_cache_path → tempfile.NamedTemporaryFile`：numba 想在 `site-packages/__pycache__` 写 JIT 缓存，被文件沙箱拒绝后阻塞（不是抛错）。
  **修复**：设置 `NUMBA_CACHE_DIR` 到可写目录，librosa 立即恢复正常（首次 beat tracking 约 16 s，之后 <0.2 s）。
  这一点只影响受限环境的命令行运行；用户本机直接运行不受影响，但为稳妥起见在复现命令中保留该环境变量。
- 本项目的数据根目录支持外部化：`CAT404_DATA_ROOT`（可写数据）与 `CAT404_AIST_ROOT`（只读的 AIST++ 关键点目录），见 `src/paths.py`。这样代码库不需要携带 917 MB 数据集。

### 4.2 音乐分析模块（`src/music_features.py`）

实现要点：

1. **帧率对齐**：`hop_length = round(sr / fps) = 735`（22050/30），因此第 *i* 帧严格对应 *i/30* 秒，音乐帧与动画帧共用一条时间轴，避免重采样误差。
2. **特征集**：RMS、onset strength、spectral centroid、bandwidth、rolloff、flatness、spectral flux、低/中/高频能量占比（20–250 / 250–2000 / 2000–11025 Hz）。chroma 与 MFCC 只用于段落分析，**不参与映射**（保证每个动画决策都能追溯到具名特征）。
3. **节拍级表**：对每个节拍插值得到 rms/onset/centroid/flux/low_ratio，并加入 `ioi`、`bpm_local = 60/ioi`、拍号位置（4/4 下拍相位由低频能量最大原则估计）、段落 id/标签/能量等级；再做 10/90 分位鲁棒归一化。
4. **段落切分**：beat-synchronous chroma + MFCC → 余弦自相似矩阵 → Gaussian checkerboard 核 novelty → `peak_pick` → 合并过短段落；段落按平均能量排序映射为 class 0/1/2（calm/mid/energetic）。
5. **输出**：`features_frame.csv`、`features_beat.csv`、`beats.csv`（兼容旧脚本）、`structure.json`、`summary.json`、`feature_plot.png`。

验证（`data/audio/test_song.wav`）：

```
Song       : test_song.wav
Duration   : 23.28 s
Tempo      : 120.00 BPM
Beats      : 40
Sections   : 1
Output     : data/features/test_song
```

短曲只切出 1 个段落符合预期（少于 16 拍不做切分）；`demo_structure.wav`（quiet/mid/loud/mid/quiet）切出 6 段，说明段落模块有效。

### 4.3 动作库（`src/motion_library.py`）

处理流程：

1. 读取 `keypoints3d_optim`（720 帧 @ 60 FPS，COCO-17），隔帧降采样到 30 FPS。
2. 坐标变换 `(x, y, z) → (x, z, y) × 0.01`，得到 Z-up、米制坐标；按序列第 2 百分位踝高把地面归零。
3. 根节点 = 髋中心，局部位姿 = 关节 − 根；根轨迹单独保存。
4. 以 2.0 s 窗口、1.0 s 步长切分，计算每个片段的描述符：

| 描述符 | 含义 | 用途 |
|---|---|---|
| `energy` | 平均关节速度 | 能量族 + 检索 |
| `arm_activity` / `leg_activity` | 上肢/下肢运动占比 | 分析 |
| `root_speed` | 根节点水平速度 | 检索（位移型动作） |
| `bounce` | 髋部垂直幅度 | 分析 |
| `extent` | 姿态空间散布 | 分析 |
| `rhythmicity` | 速度自相关峰值（0.3–1.2 s lag） | 偏好有节律的动作 |
| `internal_bpm` | 自相关峰 lag 推出的动作内在 BPM | **tempo 层的检索目标** |
| `turn` | 累计身体偏航 | 分析 |

5. 归一化：`*_n` 使用 5/95 分位；`energy_class` 按能量分位三等分（保证三类均衡）；rest pose 取能量最低 10% 片段首帧的中位数（比随机帧更接近站立中性姿态）。
6. 输出：`joints.npy`（N×T×17×3, float16）、`root.npy`（float32）、`bone_lengths.npy`（N×21）、`rest_joints.npy`、`motion_index.csv`、`library_summary.json`。

实际构建结果：

```
Clips      : 16737
Sequences  : 1408
Genres     : BR HO JB JS KR LH LO MH PO WA
Energy     : 0.00021 - 2.55598
```

单次构建约 3 分钟（远程磁盘读取 917 MB pkl）。

### 4.4 映射引擎（`src/mapping.py` + `config/mapping_rules.json`）

设计上把 proposal 的 “音乐特征 → 动画参数” 写成**显式规则**，规则全部在 JSON 中，可被界面覆盖：

| 层 | 实现 | 权重/参数 |
|---|---|---|
| beat | 2 拍一个槽位，槽位边界对齐第一个下拍 | `slot_beats` |
| tempo | 检索偏好 `internal_bpm` 接近当前 BPM 的片段；播放速率为节拍网格拟合速率 × 有界 tempo 修正 | `ref_bpm`、`exponent`、`min/max_rate`、`weight` |
| energy | 动作增益 `gain = lerp(min_gain, max_gain, energy_norm)`；同时作为能量族检索目标 | `weight`、`min/max_gain` |
| onset | 混合时长 `blend = lerp(max_blend, min_blend, onset_norm)`（重音越强过渡越短） | `min/max_blend` |
| structure | 段落能量等级限制候选池（池 < 12 时回退全库） | `strict` |

检索打分：

```
score = w_tempo * |tempo_n - target_tempo_n|
      + w_energy * |energy_n - target_energy|
      + w_rhythm * (1 - rhythmicity_n)
      + diversity_penalty * [clip 在最近 6 槽出现过]
      + genre_penalty * [与最近片段同源文件]
      + 0.01 * 确定性抖动（避免完全重复）
```

6 个条件由 `_FEATURE_FLAGS` 关闭对应层实现，**其余代码完全相同**（这是受控比较公平性的关键）。

每个槽位同时生成一行人类可读解释，例如：

```
slot 00 | 0.07-1.10s | 2 beats @ 116.2 BPM | energy=0.61 onset=0.71 section=A(class 1)
        | rate=0.97 gain=1.10 blend=0.17s | clip 1894 (House, gHO_sBM_cAll_d19_mHO3_ch04.pkl, e=0.0280, tempo=120 BPM)
```

验证（`test_song.wav`, multi_level）：20 个槽位，平均 rate 0.981，平均 gain 0.993，平均混合 0.263 s，18 个不同片段。

### 4.5 动作合成与混合（`src/motion_blend.py` + `src/kinematics.py`）

**表示方法**：动作不再是原始关节位置，而是 *根轨迹 + 21 根骨骼的单位方向 + 骨骼长度*。

理由：直接对关节位置做线性交叉淡入会出现“肢体缩短/脚底打滑”；方向空间可以用 Slerp 做真正的姿态插值，并且能量 → 动作幅度可以定义为 “相对于 rest 姿态的角度缩放”，可解释且不会破坏骨骼长度。

**组装流程**（每个槽位）：

1. 取片段 → 按首帧髋轴做朝向归一化（所有片段都朝 +Y，避免每次切换人物转头）。
2. 时间缩放：`speed = 片段帧数 / (槽位帧数 + 混合帧数) × clip(rate, 0.85, 1.15)`；采样位置用 Slerp（方向）与线性插值（根）。
3. 强度：`scale_direction(dir, rest_dir, gain)` —— 把 rest→当前方向的角度乘以 gain（gain<1 收敛、>1 夸张）。
4. 根位置：把新片段根轨迹的 XY 平移到当前已合成轨迹上（位置连续）。
5. **舞台锚定**：AIST++ 片段本身会满场跑动，保留 45% 的位移（`root_anchor`），既保留重心移动又不会走出镜头。
6. 交叉淡入：先快照被覆盖区间（`base_directions/base_root/base_filled`），再用 `slerp(base, new, w)` 与 `lerp(base_root, new_root, w)`，`w` 为 smoothstep。**这一步是本轮最重要的修复**，见第 7 节。
7. 片段覆盖区间延长到 “槽位 + 全计划最大混合帧数”，保证下一个片段淡入时一定有已填充的姿态作为基准。
8. 后处理：3 帧方向平滑 → 垂直速度限制（0.02 m/帧）→ 足部离地约束（踝/趾最低点保持在 0.07 m 以上，先膨胀再平滑）→ 最终 FK。

输出 `motion.npz`（root / directions / lengths / nodes / joints / energy）、`motion_summary.json`、`plan.csv`。

### 4.6 Blender 角色渲染（`blender/render_character.py`）

- 用首帧 21 个节点生成人形：头、躯干、骨盆、颈、上臂/前臂/手、大腿/小腿/脚，并在肩/肘/髋/膝加关节球遮住接缝。
- **刚性蒙皮**：每个部件 100% 绑定到对应骨骼（`_tag_group`），关节球同理。之前尝试过 “voxel remesh + 自动权重/距离权重”，会出现躯干尖刺与头部拉长（见第 7 节），刚性部件是稳定且可解释的选择。
- 骨架：21 根骨骼，head/tail 直接取首帧节点；`pose_bone.matrix` 按 “rest 方向 → 目标方向” 的旋转差重定向，逐帧写 location + rotation_quaternion（不再写 scale，避免拉伸）。
- 场景：EEVEE（自动探测 `BLENDER_EEVEE_NEXT`/`BLENDER_EEVEE`）、三点面光、深色地面、45–50 mm 相机；16 TAA 采样以换取可接受的渲染速度。
- 输出：直接渲染动画场景，或建立 mix scene（Scene strip + Sound strip）输出带音轨的 MP4；`--blend-out` 保存 .blend，`--export-fbx` 导出 FBX，`--stills N` 只出静帧（快速质检）。

单帧静帧渲染约 0.4 s；634 帧 960×540 的完整渲染约 6–10 分钟（视机器负载）。

### 4.7 评估与实验（`src/beat_alignment.py`, `src/experiment.py`）

指标（全部在 `src/beat_alignment.py` 中实现并写明定义）：

| 指标 | 定义 | 目的 |
|---|---|---|
| BAS（Gaussian） | 对每个音乐节拍，取到最近 “运动节拍” 的距离 d，`mean(exp(-d²/(2·0.15²)))` | 节拍对齐 |
| Beat F1 | 0.15 s 容差内运动节拍精确率与音乐节拍召回率的调和平均 | 节拍覆盖 |
| Energy corr | 运动能量曲线与音乐 RMS 曲线（插值到 30 FPS）的 Pearson 相关 | 表现力/能量同步 |
| Beat response corr | 每个音乐节拍后 0.12 s 的运动能量 vs 该拍音乐能量，二者的相关 | “重拍是否跳得更用力” |
| Dynamic range | 运动能量 90/10 分位比 | 动态幅度 |
| Clip variety | 不同片段数 / 槽位数 | 动作丰富度 |

运动节拍由平均关节速度信号的局部极大值检测（阈值 = 均值 + 0.5 标准差，最小间隔 0.2 s），**不依赖音乐端节拍器**，避免循环论证。

实验条件：`beat_only`、`multi_level`、`no_tempo`、`no_energy`、`no_onset`、`no_structure`；曲目 4 首（见 4.2 与 4.8 的变体生成）。

### 4.8 测试曲目变体（`src/make_test_variants.py`）

只有一首 23 秒测试曲无法支撑消融实验，因此用 librosa 时间伸缩生成不同速度版本，并拼接出段落结构：

| 文件 | 时长 | 检测 BPM | 节拍 | 段落 | 说明 |
|---|---|---|---|---|---|
| `test_song.wav` | 23.28 s | 120.0 | 40 | 1 | 原始 |
| `demo_100bpm.wav` | 27.38 s | 105.9 | 42 | 2 | 0.85× 时间伸缩 |
| `demo_145bpm.wav` | 19.40 s | 150.0 | 42 | 1 | 1.20× 时间伸缩 |
| `demo_structure.wav` | 116.38 s | 120.0 | 225 | 6 | quiet/mid/loud/mid/quiet 拼接 |

### 4.9 交互界面（`app.py`）

Streamlit 五页：

1. **Music analysis**：时长/速度/节拍/段落指标、特征总览图、段落表、节拍级特征表。
2. **Mapping plan**：可选 6 种条件；侧栏可实时调整 12 个映射参数（槽位拍数、参考 BPM、tempo/energy 权重、增益上下限、onset 混合上下限、多样性惩罚、舞台锚定、足部锁定等）；生成后展示每个槽位的 rate/gain/blend/clip，并可导出 `plan.csv` 与 `rules.json`，**逐条解释每一次决策**。
3. **Motion & render**：一键合成动作（秒级），显示 4 项指标与 8 帧姿态预览，可下载 `motion.npz`，可调用 Blender 渲染并播放/下载 MP4。
4. **Evaluation**：同一首歌自动跑 beat-only 与 multi-level 并给出差值表；提供**盲评 A/B 表单**（同步性/表现力 1–5 分），结果追加写入 `outputs/user_study/responses.csv`，可直接用于 proposal 的用户研究。
5. **Documentation**：映射表、流水线说明、报告下载。

用 `streamlit.testing.v1.AppTest` 做的无头冒烟测试通过（无异常，5 个 tab、5 个按钮全部渲染）：

```
App rendered without exceptions
tabs: ['1. Music analysis', '2. Mapping plan', '3. Motion & render', '4. Evaluation', '5. Documentation']
buttons: ['Re-analyse music', 'Build mapping plan', 'Generate motion', 'Run both conditions on this song', 'Save rating']
```

（测试进程退出时出现一次 `PermissionError`，来自 Python 清理临时目录，属于受限环境现象，与应用逻辑无关。）

---

## 5. 关键设计决策与权衡

| 决策 | 选择 | 理由 / 代价 |
|---|---|---|
| 动作表示 | 根轨迹 + 骨骼方向 + 骨长 | 支持 Slerp、角度缩放与稳定 FK；代价是不能直接保留原始关节的细微长度变化 |
| 动作检索 | 规则化加权最近邻 | 可解释、零训练；代价是无法学习长程编排 |
| tempo 映射 | 选择内在 BPM 匹配的片段 + 有界速率修正 | 节拍网格拟合本身已让速度随 BPM 变化，若再乘 (bpm/120) 会“双重计数”；因此 tempo 层同时承担 “选什么速度的动作” |
| 结构与动作族 | 3 个能量族（calm/mid/energetic） | 与段落能量等级天然对应；代价是族内多样性仍靠检索权重维持 |
| 混合 | 方向空间 Slerp + smoothstep 权重 | 真正插值、无肢体塌陷；代价是每帧最多两段重叠，无法做多段加权 |
| 角色网格 | 刚性部件蒙皮 | 稳定、无尖刺；代价是关节处略显“木偶感” |
| 渲染 | EEVEE 16 采样、960×540 | 可在一台笔记本上 6–10 分钟出片；代价是画质一般 |
| 评估 | 运动节拍由速度峰值检测 | 与音乐端独立的度量；代价是对“平滑动作”可能低估节拍数 |

---

## 6. 实验结果（4 曲 × 6 条件，24 次运行）

### 6.1 条件均值

| 条件 | BAS | Beat F1 | 能量相关 | 节拍响应相关 | 动态范围 | 片段多样性 |
|---|---|---|---|---|---|---|
| beat_only | 0.684 | **0.663** | −0.041 | −0.126 | 3.03 | 0.351 |
| multi_level | 0.661 | 0.514 | **0.170** | **0.315** | 3.72 | **0.747** |
| no_tempo | 0.668 | 0.552 | 0.161 | 0.333 | 3.58 | 0.718 |
| no_energy | 0.661 | 0.607 | 0.091 | 0.078 | 3.91 | 0.582 |
| no_onset | 0.641 | 0.470 | 0.161 | 0.319 | 3.34 | 0.747 |
| no_structure | **0.683** | 0.515 | 0.184 | **0.366** | **5.15** | **0.846** |

### 6.2 配对差值（相对 beat-only，4 首曲）

| 条件 | ΔBAS | ΔF1 | Δ能量相关（胜场） | Δ节拍响应（胜场） | Δ多样性 |
|---|---|---|---|---|---|
| multi_level | −0.024 | −0.149 | **+0.211（4/4）** | **+0.441（4/4）** | +0.396 |
| no_tempo | −0.016 | −0.112 | +0.202（3/4） | +0.459（4/4） | +0.367 |
| no_energy | −0.023 | −0.057 | +0.132（4/4） | +0.204（4/4） | +0.230 |
| no_onset | −0.043 | −0.194 | +0.202（4/4） | +0.446（4/4） | +0.396 |
| no_structure | −0.001 | −0.148 | +0.225（3/4） | +0.492（4/4） | +0.495 |

### 6.3 解读

1. **同步性：beat-only 略优。** 两者的槽位都落在节拍网格上，但 beat-only 使用固定 0.30 s 混合、始终选中性片段，运动更强、加速度峰更密集，因而运动节拍检测的 F1 更高。多级映射为了表现力牺牲了一部分 “节拍命中率”（0.663 → 0.514）。
2. **表现力：multi-level 明显更好。** 能量相关从 −0.041 提升到 +0.170（4/4 曲胜），节拍响应相关从 −0.126 提升到 +0.315（4/4 曲胜），片段多样性从 0.35 提升到 0.75。这正是 proposal 假设的方向：**多级映射的价值主要体现在 “重拍动作更强、弱拍动作更轻”**。
3. **消融：三个层各有作用。**
   - 去掉 energy：节拍响应相关跌到 0.078（−0.24），说明 **energy 层是表现力的主要来源**。
   - 去掉 onset：Beat F1 跌到 0.470（−0.19），说明 **onset 层影响过渡与同步**。
   - 去掉 structure：动态范围升到 5.15、能量相关略升，说明在当前样本上 **structure 层的收益不明显**（段落只有 1–6 段，类别约束反而限制了候选池）。
   - 去掉 tempo：影响最小（ΔBAS −0.016），因为节拍网格本身已经让速度跟随 BPM。
4. **统计功效不足**：只有 4 首歌，Wilcoxon 需要 ≥5 对，故 p 值为 n/a；报告只给出描述性统计与胜场数。

---

## 7. 问题与修复记录（按发现顺序）

| # | 现象 | 根因 | 修复 | 验证 |
|---|---|---|---|---|
| 1 | `librosa.load` 挂起数分钟 | numba JIT 缓存写入被沙箱拒绝后阻塞 | 设置 `NUMBA_CACHE_DIR` | 加载 1.26 s、beat tracking 15.7 s（首次） |
| 2 | 写完代码后数据目录不可写 | 桌面目录在沙箱外 | 数据根目录外部化（`CAT404_DATA_ROOT` / `CAT404_AIST_ROOT`），在可写副本中开发 | 特征写入 workspace 成功 |
| 3 | 合成动作在槽位边界出现 0.2–0.4 m 的关节跳变（能量曲线尖峰 0.28） | ① Slerp 对近平行反向方向退化（零向量归一化）② 混合时以“已混合值”为基准递归混合，单帧推进过大 ③ 上一片段覆盖结束后下一片段权重仍 <1，出现未填充帧 | 反向方向加 1e-3 扰动；改为快照 `base` 后与 `new` 做真交叉淡入；片段覆盖延长 `max_blend_frames`；位置采样改为 clamp 不循环 | 能量峰值 0.255 → 0.077；最大单帧位移 0.44 → 0.30 m（剩余为快速摆臂，属数据本身） |
| 4 | 脚陷入地面 5–9 cm | 踝关节离地约 7 cm，原先把踝高夹到 0；且足部修正用了限速前的旧节点 | 引入 `foot_clearance=0.07`；先限速再重算节点再修正；用膨胀+平滑保证覆盖峰值 | 最低足部高度 −0.086 → 0.000 m |
| 5 | 人物整体偏高/偏低、出画 | 相机目标与距离按髋高估计，透视下脚被裁切 | 相机目标固定 0.95 m，距离 4.3 m，45–50 mm | 静帧全身入画 |
| 6 | voxel remesh 后躯干出现尖刺、头部被拉成锥形 | 自动权重把躯干顶点绑到手臂骨骼，骨骼独立无父子约束，旋转时被拉飞 | 改为刚性部件蒙皮（每部件 100% 单骨骼）+ 关节球遮缝；不再对骨骼写 scale | 静帧无尖刺、头形正常 |
| 7 | 一次完整渲染 20 分钟仍未完成 | EEVEE 默认 64 TAA 采样 + 720p | `taa_render_samples=16`、960×540 | 单帧约 0.4 s，整片 6–10 分钟 |
| 8 | `pandas.to_markdown` 报错 | venv 无 tabulate | 自写 `md_table()` | 报告正常生成 |
| 9 | AppTest 找不到 app.py | 相对路径按测试文件解析 | 用 `Path(__file__).parents[1]` 绝对路径 | 冒烟测试通过 |

---

## 8. 已知限制

1. **用户研究尚未收集数据**：界面已提供盲评表单与 CSV 落盘，但 proposal 中的 “user-perceived synchronization / expressiveness” 仍需招募被试。
2. **样本量小**：4 首短曲（19–116 s）且其中 3 首由同一首测试曲变换而来，统计结论是描述性的；需要更多真实歌曲。
3. **同步性未超过 beat-only**：BAS 与 F1 略低，这是 “表现力换同步” 的取舍；可尝试在重拍处强制片段起点对齐（硬约束）而非仅靠槽位。
4. **structure 层收益不明显**：段落数少时类别约束限制了候选池；可改为软约束（类别只作为打分项）或加入段落内动作重复性约束。
5. **动作空间为 COCO-17**：手指、脚趾、面部缺失；AIST++ 只有 10 个街舞舞种。
6. **刚性蒙皮**：视觉上关节略硬，无肌肉/布料形变；如需更高质量可换 SMPL-X + 学习式蒙皮。
7. **无真正的足部 IK**：只做垂直离地约束，快速移动时仍有轻微滑步。
8. **无关键帧编辑**：proposal 5.7 的 “keyframe editor” 尚未实现（当前可编辑的是映射规则与整体参数）。
9. **BVH 导出未实现**：已支持 FBX（Blender 自带导出器，需在界面外手动传参）与 NPZ。
10. **中文报告**中的图表为英文标签，正式提交前可本地化。

---

## 9. 下一步计划（对应 proposal 时间表 M6–M8）

| 优先级 | 任务 | 说明 |
|---|---|---|
| P0 | 用户研究 | 用第 4 页盲评表单收集 ≥12 名被试 × 2 条件 × 3 曲，计算配对检验与效应量 |
| P0 | 更多真实歌曲 | 至少 8 首、覆盖 90–160 BPM，重新跑 `src/experiment.py` |
| P1 | 重拍硬对齐 | 在槽位内按 onset 强度选择片段起始相位，提升 BAS/F1 |
| P1 | 结构层软约束 | 类别作为打分项而非硬过滤，观察消融变化 |
| P1 | 关键帧编辑 | 在时间轴上插入用户姿态约束，用方向空间插值生成过渡 |
| P2 | 足部 IK | 两骨 IK + 接触检测，消除滑步 |
| P2 | BVH 导出 | 由骨骼方向/层级直接写 BVH，便于导入 Unity/UE |
| P2 | 渲染质量 | 若机器允许，切回 720p / 32 采样出最终演示片 |

---

## 10. 复现命令（Windows PowerShell）

```powershell
cd <项目根目录>
$env:NUMBA_CACHE_DIR = "$PWD\.numba_cache"      # 仅在受限环境下需要
$env:CAT404_AIST_ROOT = "D:\path\to\aistplusplus\keypoints3d"   # 数据集在外部时

# 1. 动作库（一次性，约 3 分钟）
.\.venv\Scripts\python.exe src\motion_library.py

# 2. 测试曲变体（一次性）
.\.venv\Scripts\python.exe src\make_test_variants.py

# 3. 单曲全流程（分析 → 映射 → 合成 → 指标）
.\.venv\Scripts\python.exe src\pipeline.py data\audio\test_song.wav --mode multi_level

# 4. 受控实验（4 曲 × 6 条件，约 1 分钟）
.\.venv\Scripts\python.exe src\experiment.py

# 5. Blender 渲染
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --factory-startup `
  --python blender\render_character.py -- `
  --motion outputs\runs\test_song_multi_level\motion.npz `
  --audio data\audio\test_song.wav `
  --output outputs\runs\test_song_multi_level\character.mp4 `
  --resolution 960x540

# 6. 交互界面
.\.venv\Scripts\streamlit.exe run app.py
```

---

## 11. 文件清单（本轮新增/修改）

| 路径 | 说明 |
|---|---|
| `src/paths.py` | 路径与数据根目录解析 |
| `src/music_features.py` | 音乐分析（帧级/节拍级/段落） |
| `src/kinematics.py` | 21 节点骨架、方向 Slerp、角度缩放、FK |
| `src/motion_library.py` | AIST++ → 16 737 片段动作库 |
| `src/mapping.py` | 五层映射 + 6 条件 + 检索 |
| `src/motion_blend.py` | 时间缩放、Slerp 混合、DTW、地面约束 |
| `src/beat_alignment.py` | BAS / F1 / 相关性指标 |
| `src/experiment.py` | 受控实验与消融、报告生成 |
| `src/pipeline.py` | 单曲全流程编排（含渲染） |
| `src/preview_motion.py` | 动作静帧预览 |
| `src/make_test_variants.py` | 测试曲变体 |
| `blender/render_character.py` | 角色绑定、重定向、MP4/FBX 渲染 |
| `config/mapping_rules.json` | 可编辑映射规则 |
| `app.py` | Streamlit 交互界面（重写） |
| `tests/test_app_smoke.py` | 界面无头冒烟测试 |
| `tests/probe_librosa.py`, `tests/debug_*.py` | 诊断脚本（可保留，非运行依赖） |
| `tools/make_diagram.py` | 生成报告用流水线图 |
| `tools/make_progress_report.py` | 生成 `docs/PROGRESS_REPORT.docx` |
| `tools/extract_frame.py` | 从渲染 MP4 抽帧做质检 |
| `docs/DEVELOPMENT_LOG.md` | 本文件 |
| `docs/EXPERIMENT_REPORT.md` | 自动生成的实验报告 |
| `docs/PROGRESS_REPORT.md` / `.docx` | 阶段性进展报告 |

已渲染的视频（供报告/演示使用）：

- `outputs/runs/smoke_multilevel/character_multilevel.mp4`：多级映射，21.1 s，960×540，含音轨（H.264 + AAC）。
- `outputs/runs/exp_test_song_beat_only/character_beat_only.mp4`：节拍基线对照，同样规格。
- `outputs/runs/smoke_multilevel/character.blend`：可直接在 Blender 中打开的绑定场景。
