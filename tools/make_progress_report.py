"""Generate the stage progress report (DOCX) for the CAT404 project.

Run with the bundled Python that provides python-docx:

    <runtime>/python.exe tools/make_progress_report.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PROJECT = Path(__file__).resolve().parents[1]
DOCS = PROJECT / "docs"
RUN = PROJECT / "outputs" / "runs" / "smoke_multilevel"
EXP = PROJECT / "outputs" / "experiments"


# ---------------------------------------------------------------------------
# Figures (the diagram itself is produced by tools/make_diagram.py)
# ---------------------------------------------------------------------------

def ensure_diagram(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run tools/make_diagram.py with the project virtualenv first."
        )


# ---------------------------------------------------------------------------
# DOCX helpers
# ---------------------------------------------------------------------------

def set_cjk(run, name: str = "Microsoft YaHei") -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def style_document(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for level, size in ((1, 16), (2, 13), (3, 11.5)):
        style = document.styles[f"Heading {level}"]
        style.font.name = "Microsoft YaHei"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")


def add_caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    set_cjk(run)


def add_table(document: Document, headers, rows, widths=None, font_size: int = 9) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    header_cells = table.rows[0].cells
    for index, header in enumerate(headers):
        header_cells[index].text = ""
        run = header_cells[index].paragraphs[0].add_run(header)
        run.bold = True
        run.font.size = Pt(font_size)
        set_cjk(run)

    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = ""
            run = cells[index].paragraphs[0].add_run(str(value))
            run.font.size = Pt(font_size)
            set_cjk(run)

    if widths:
        for index, width in enumerate(widths):
            table.columns[index].width = Cm(width)
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Cm(width)

    document.add_paragraph()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build() -> Path:
    DOCS.mkdir(parents=True, exist_ok=True)
    diagram = DOCS / "pipeline_diagram.png"
    ensure_diagram(diagram)

    document = Document()
    section = document.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    style_document(document)

    # Title block
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("CAT404W Software Engineering Major Project")
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    set_cjk(run)

    heading = document.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run("音乐驱动 3D 角色动画系统 — 阶段性进展报告")
    run.font.size = Pt(20)
    run.bold = True
    run.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
    set_cjk(run)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(
        "An Interactive Music-Driven 3D Character Animation System\n"
        "Using Multi-Level Musical Features"
    )
    run.font.size = Pt(11)
    run.italic = True
    set_cjk(run)

    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run("学生：Yixuan Guo　|　导师：Dr. Justtina Anantha Jothi AP C.M.E John\n"
                       "School of Computer Sciences, Universiti Sains Malaysia\n"
                       "报告日期：2026-10-20　|　迭代：Iteration 2")
    run.font.size = Pt(9.5)
    set_cjk(run)

    document.add_paragraph()

    # 1. Summary
    document.add_heading("1. 本轮工作摘要", level=1)
    document.add_paragraph(
        "本报告记录按 proposal 第 5.9 节迭代计划开展的第二次迭代。上一轮已经完成 "
        "“音频 → 节拍 → 事件 → Blender 方块动画” 的最小闭环，但动作库、多级映射、"
        "动作混合、受控实验和可编辑映射规则仍未实现。本轮补齐了 proposal 第 5.2–5.8 节"
        "描述的全部核心模块，并用 4 首测试曲 × 6 种映射条件完成了 24 次受控生成与评估。"
    )
    document.add_paragraph(
        "与上一轮相比，系统现在能够：从 16 737 个 AIST++ 动作片段中按音乐特征检索动作；"
        "在节拍网格上以 Slerp 混合出连续动画；用绑定好的 3D 人形在 Blender 中渲染带音轨的 "
        "MP4；并在交互界面里逐条查看、修改每一次映射决策。"
    )

    add_table(
        document,
        ["迭代", "内容", "本轮结果"],
        [
            ["1", "音乐分析与特征提取", "完成：30 FPS 帧级特征 + 节拍级特征表 + 段落切分"],
            ["2", "动作数据集与动作库", "完成：1 408 序列 → 16 737 片段，9 维描述符"],
            ["3", "节拍基线动画", "完成：统一为 beat_only 条件，与多级共享全部代码"],
            ["4", "多级音乐-动作映射", "完成：beat/tempo/energy/onset/structure 五层 + 4 个消融"],
            ["5", "动作混合", "完成：Slerp、时间缩放、DTW 对齐、足部与地面约束"],
            ["6", "交互式界面", "完成：5 页 Streamlit，规则可编辑，盲评表单"],
            ["7", "系统集成与评估", "完成：BAS/F1/相关性指标，24 组实验"],
        ],
        widths=[2.0, 5.5, 8.5],
    )

    document.add_picture(str(diagram), width=Cm(16.4))
    add_caption(document, "图 1　系统流水线与各阶段产物")

    # 2. Design
    document.add_heading("2. 系统设计", level=1)
    document.add_paragraph(
        "系统采用 proposal 提出的“显式规则 + 现有动作库”路线，不训练生成模型。"
        "这样做的目的是让每一个动画决策都能追溯到某个具体音乐特征，并让用户可以直接修改映射规则。"
    )

    add_table(
        document,
        ["音乐特征", "动画参数", "实现方式"],
        [
            ["Beat 节拍", "动作发生的时刻", "2 拍一个槽位，槽位边界对齐第一个下拍"],
            ["Tempo 速度", "动作的快慢", "检索内在 BPM 匹配的片段；播放速率按节拍网格拟合并有界修正"],
            ["RMS 能量", "动作的强度", "动作增益（骨骼角度相对 rest 姿态缩放）+ 能量族检索"],
            ["Onset 起音", "过渡的锐利程度", "重叠片段之间的混合时长，起音越强混合越短"],
            ["Structure 段落", "动作类型/风格族", "段落能量等级（calm/mid/energetic）限制候选池"],
        ],
        widths=[3.4, 4.4, 8.2],
    )

    document.add_paragraph(
        "动作在“方向空间”中合成：根轨迹 + 21 根骨骼的单位方向 + 骨骼长度。"
        "相对直接插值关节位置，这一表示让 Slerp 混合、强度缩放和正向运动学重建都保持稳定，"
        "不会出现肢体缩短或关节塌陷。"
    )

    document.add_heading("3. 实现进展", level=1)

    document.add_heading("3.1 音乐分析", level=2)
    document.add_paragraph(
        "Librosa 提取 RMS、onset、谱质心、谱通量、频段能量占比等特征，hop_length 取 735 使帧率"
        "严格等于 30 FPS，音乐帧与动画帧共用一条时间轴。段落切分采用 beat-synchronous chroma/MFCC "
        "自相似矩阵 + checkerboard 核 novelty + 峰值检测，段落按平均能量映射为 3 个等级。"
        "chroma 与 MFCC 只用于段落分析，不参与动画映射。"
    )

    document.add_heading("3.2 动作库", level=2)
    document.add_paragraph(
        "1 408 个 AIST++ 序列（COCO-17，60 FPS）降采样到 30 FPS，转换为 Z-up 米制坐标并按踝高归零，"
        "以 2 秒窗口、1 秒步长切分为 16 737 个片段。每个片段计算能量、上肢/下肢活动度、根速度、"
        "髋部弹跳、姿态散布、节律强度、内在 BPM、累计转向共 9 个描述符，并给出能量三分位类别。"
    )

    document.add_heading("3.3 映射与检索", level=2)
    document.add_paragraph(
        "映射规则全部写在 config/mapping_rules.json 中，界面可以实时覆盖。每个槽位的检索打分是"
        "“tempo 距离、energy 距离、节律惩罚、重复惩罚、确定性抖动”的加权和，因此每次选片都有"
        "可读的解释文本。六个实验条件只改变“哪些层是激活的”，其余代码完全相同。"
    )

    document.add_heading("3.4 动作混合", level=2)
    document.add_paragraph(
        "按节拍网格做时间缩放，按能量做角度缩放，用 smoothstep 权重在重叠区做方向 Slerp 交叉淡入，"
        "用 DTW 在 ±3 帧内对齐进入片段，最后做根轨迹连续、舞台锚定、垂直速度限制与足部离地约束。"
        "调试过程中修复了四类会造成明显跳变的问题（递归混合、覆盖空洞、反向方向退化、踝高偏移），"
        "运动能量的单帧尖峰由 0.255 降到 0.077。"
    )

    document.add_heading("3.5 Blender 角色渲染", level=2)
    document.add_paragraph(
        "用首帧节点生成人形网格（头、躯干、骨盆、四肢、手、脚、关节球），每个部件刚性绑定到对应骨骼；"
        "21 根骨骼逐帧按“rest 方向 → 目标方向”的旋转差重定向，再叠加相机、三点面光、舞台地面，"
        "通过 Scene strip + Sound strip 输出带音轨的 MP4，也可保存 .blend 或导出 FBX。"
    )

    # 4. Experiment
    document.add_heading("4. 评估与实验结果", level=1)
    document.add_paragraph(
        "评估指标包括 BAS（Gaussian，σ = 0.15 s）、节拍 F1、运动能量与音乐 RMS 的相关、"
        "逐拍运动能量与音乐能量的相关、动态范围和片段多样性。运动节拍由关节速度峰值独立检测，"
        "不依赖音乐端节拍器。4 首测试曲（120 BPM / 105.9 BPM / 150 BPM / 116 s 段落测试曲）"
        "× 6 条件共 24 次运行。"
    )

    add_table(
        document,
        ["条件", "BAS", "Beat F1", "能量相关", "节拍响应相关", "多样性"],
        [
            ["Beat-only", "0.684", "0.663", "−0.041", "−0.126", "0.351"],
            ["Multi-level", "0.661", "0.514", "0.170", "0.315", "0.747"],
            ["No tempo", "0.668", "0.552", "0.161", "0.333", "0.718"],
            ["No energy", "0.661", "0.607", "0.091", "0.078", "0.582"],
            ["No onset", "0.641", "0.470", "0.161", "0.319", "0.747"],
            ["No structure", "0.683", "0.515", "0.184", "0.366", "0.846"],
        ],
        widths=[3.6, 2.4, 2.6, 2.8, 3.2, 2.4],
    )

    if (EXP / "experiment_summary.png").exists():
        document.add_picture(str(EXP / "experiment_summary.png"), width=Cm(16.0))
        add_caption(document, "图 2　各条件在四项指标上的均值与标准差（4 首测试曲）")

    document.add_paragraph(
        "主要发现：Beat-only 的节拍 F1 更高（0.663 对 0.514），而 Multi-level 的表现力指标全面领先"
        "（能量相关 −0.041 → 0.170，节拍响应相关 −0.126 → 0.315，4/4 曲目胜出，片段多样性 0.351 → 0.747）。"
        "消融显示 energy 层是表现力的主要来源（去掉后节拍响应相关降到 0.078），onset 层影响同步"
        "（去掉后 F1 降到 0.470），而 structure 层在当前样本上的收益不明显。"
    )

    # 5. Interface / artefacts
    document.add_heading("5. 交互界面与可视化", level=1)
    document.add_paragraph(
        "Streamlit 界面分为音乐分析、映射计划、动作与渲染、评估、文档五页。"
        "侧栏可实时调整 12 个映射参数；映射页展示每个槽位的决策与解释；评估页可一键比较两种条件，"
        "并提供盲评 A/B 表单把评分写入 CSV，可直接用于 proposal 的用户研究。"
    )

    preview = RUN / "motion_preview.png"
    if preview.exists():
        document.add_picture(str(preview), width=Cm(16.4))
        add_caption(document, "图 3　合成动作的 8 帧姿态与运动能量曲线（黄线为槽位边界）")

    still = RUN / "still_01.png"
    if still.exists():
        document.add_picture(str(still), width=Cm(14.0))
        add_caption(document, "图 4　Blender 中绑定并渲染的人形角色")

    # 6. Issues
    document.add_heading("6. 主要问题与修复", level=1)
    add_table(
        document,
        ["问题", "原因", "修复"],
        [
            ["librosa 加载挂起", "numba JIT 缓存写入被拒绝", "设置 NUMBA_CACHE_DIR"],
            ["槽位边界关节跳变 0.2–0.4 m", "递归混合 + 片段覆盖空洞 + 反向方向退化",
             "真交叉淡入 + 覆盖延长 + 反向扰动"],
            ["脚陷入地面 5–9 cm", "踝高偏移与旧节点参与修正", "7 cm 离地约束 + 先限速再修正"],
            ["躯干尖刺、头部拉长", "voxel remesh 后自动权重把躯干绑到手臂骨",
             "刚性部件蒙皮 + 关节球遮缝"],
            ["整片渲染过慢", "EEVEE 64 采样 + 720p", "16 采样 + 960×540"],
        ],
        widths=[4.6, 6.0, 5.4],
    )

    # 7. Limitations
    document.add_heading("7. 限制与后续工作", level=1)
    for text in (
        "用户研究尚未收集数据：界面与记录格式已就绪，需招募 ≥12 名被试、每人 2 条件 × 3 曲。",
        "样本量小：4 首短曲中 3 首由同一首变换而来，统计结论为描述性，需要更多真实歌曲。",
        "同步性未超过基线：multi-level 以约 0.02–0.15 的 BAS/F1 换取明显更高的表现力，"
        "后续可在重拍处加入片段起点硬约束。",
        "structure 层收益不明显，建议改为软约束并增加段落内动作重复性。",
        "尚缺关键帧编辑、足部 IK 与 BVH 导出；角色为刚性蒙皮，关节偏硬。",
    ):
        paragraph = document.add_paragraph(text, style="List Bullet")

    document.add_heading("8. 复现方式", level=1)
    document.add_paragraph(
        "环境：项目自带 .venv（Python 3.14，librosa 1.0、numpy 2.5、pandas 3.0、scipy 1.18、"
        "streamlit 1.66）与 Blender 5.2.2。完整命令见 docs/DEVELOPMENT_LOG.md 第 10 节。"
    )
    document.add_paragraph(
        "核心命令：python src/motion_library.py（构建动作库）；"
        "python src/pipeline.py data/audio/test_song.wav --mode multi_level（单曲全流程）；"
        "python src/experiment.py（24 组受控实验）；streamlit run app.py（交互界面）。"
    )

    output = DOCS / "PROGRESS_REPORT.docx"
    document.save(output)
    return output


if __name__ == "__main__":
    print(build())
