# 阶段性进展报告（摘要）

> 完整排版版本（含图表、表格）见 `docs/PROGRESS_REPORT.docx`。
> 详细开发过程、决策与问题修复见 `docs/DEVELOPMENT_LOG.md`。
> 自动生成的实验数据见 `docs/EXPERIMENT_REPORT.md`。

## 1. 本轮完成内容

按 proposal 第 5.9 节的迭代计划完成第 1–7 次迭代：

| 迭代 | 内容 | 结果 |
|---|---|---|
| 1 | 音乐分析与特征提取 | 30 FPS 帧级特征、节拍级特征表、段落切分 |
| 2 | 动作数据集与动作库 | 1 408 个 AIST++ 序列 → 16 737 个 2 秒片段 |
| 3 | 节拍基线动画 | 统一为 `beat_only` 条件 |
| 4 | 多级音乐-动作映射 | beat / tempo / energy / onset / structure 五层 + 4 个消融 |
| 5 | 动作混合 | Slerp、时间缩放、强度缩放、DTW、地面约束 |
| 6 | 交互式界面 | 5 页 Streamlit，映射规则可编辑，盲评表单 |
| 7 | 系统集成与评估 | BAS/F1/相关性指标，4 曲 × 6 条件 = 24 次实验 |

## 2. 关键结果

| 条件 | BAS | Beat F1 | 能量相关 | 节拍响应相关 | 片段多样性 |
|---|---|---|---|---|---|
| Beat-only | 0.684 | 0.663 | −0.041 | −0.126 | 0.351 |
| Multi-level | 0.661 | 0.514 | 0.170 | 0.315 | 0.747 |

- 多级映射在表现力指标上 4/4 曲目胜出（能量相关 +0.211，节拍响应相关 +0.441）。
- 消融：energy 层是表现力的主要来源（去掉后节拍响应相关 0.315 → 0.078）；
  onset 层影响同步（去掉后 Beat F1 0.514 → 0.470）；structure 层在当前样本上收益不明显。
- 同步性方面 beat-only 略优，说明本系统用少量同步性换取了明显的表现力提升。

## 3. 当前限制

用户研究尚未收集数据（界面已就绪）；样本仅 4 首短曲；尚缺关键帧编辑、足部 IK、BVH 导出；
角色为刚性蒙皮，关节略硬。

## 4. 复现

```powershell
python src\motion_library.py
python src\pipeline.py data\audio\test_song.wav --mode multi_level
python src\experiment.py
streamlit run app.py
```
