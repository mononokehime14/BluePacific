# -*- coding: utf-8 -*-
"""
Build the share-out deck: "From Embedding Models to Agentic RAG"
Source outline: RAG and Agentic AI.md (same content as the .docx)
Style: clean academic, white background, dark titles, large placeholder areas for the author to fill in.
"""
import copy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

import os
OUT = os.environ.get(
    "DECK_OUT",
    r"C:\Users\renha\workspace\projects\Learning\Embedding and RAG\Embedding and Agentic RAG.pptx")

# ---------------------------------------------------------------- palette
INK        = RGBColor(0x1B, 0x25, 0x33)   # near-black headings
BODY       = RGBColor(0x33, 0x40, 0x50)   # body text
MUTED      = RGBColor(0x7A, 0x86, 0x96)   # captions / hints
ACCENT     = RGBColor(0x1F, 0x5C, 0xA8)   # blue accent
ACCENT2    = RGBColor(0xC0, 0x50, 0x2A)   # warm accent (warnings / weak points)
RULE       = RGBColor(0xD8, 0xDE, 0xE6)   # hairlines
PLACE_BG   = RGBColor(0xF7, 0xF9, 0xFC)   # placeholder fill
PLACE_LINE = RGBColor(0xBB, 0xC7, 0xD6)
CHIP_BG    = RGBColor(0xEC, 0xF2, 0xFA)

FONT = "Microsoft YaHei"
MONO = "Consolas"

SW, SH = 13.333, 7.5           # 16:9 inches
M      = 0.62                  # left/right margin


# ---------------------------------------------------------------- helpers
def txbox(slide, x, y, w, h, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf


def style(p, size=14, bold=False, color=BODY, font=FONT, italic=False,
          space_after=6, align=PP_ALIGN.LEFT, line=1.25):
    p.alignment = align
    p.space_after = Pt(space_after)
    p.line_spacing = line
    for r in p.runs:
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = font
        # east-asian font
        rPr = r._r.get_or_add_rPr()
        ea = rPr.find(qn('a:ea'))
        if ea is None:
            ea = rPr.makeelement(qn('a:ea'), {})
            rPr.append(ea)
        ea.set('typeface', font)
    return p


def para(tf, text, first=False, **kw):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.text = text
    return style(p, **kw)


def rich(tf, segments, first=False, size=14, space_after=6, line=1.25,
         align=PP_ALIGN.LEFT, color=BODY, font=FONT):
    """segments: list of (text, {overrides})"""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    for text, ov in segments:
        r = p.add_run()
        r.text = text
        r.font.size = Pt(ov.get("size", size))
        r.font.bold = ov.get("bold", False)
        r.font.italic = ov.get("italic", False)
        r.font.color.rgb = ov.get("color", color)
        r.font.name = ov.get("font", font)
        rPr = r._r.get_or_add_rPr()
        ea = rPr.find(qn('a:ea'))
        if ea is None:
            ea = rPr.makeelement(qn('a:ea'), {})
            rPr.append(ea)
        ea.set('typeface', ov.get("font", font))
    p.alignment = align
    p.space_after = Pt(space_after)
    p.line_spacing = line
    return p


def rect(slide, x, y, w, h, fill=None, line=None, lw=0.75, radius=None):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        shp.adjustments[0] = radius
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(lw)
    shp.shadow.inherit = False
    if shp.has_text_frame:
        shp.text_frame.word_wrap = True
    return shp


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def base(slide, section="", number=None, title=None, kicker=None):
    """Slide chrome: top accent rule, section label, big title."""
    rect(slide, 0, 0, SW, 0.085, fill=ACCENT)
    if section:
        _, tf = txbox(slide, M, 0.34, 8.6, 0.3)
        para(tf, section.upper(), first=True, size=10.5, bold=True,
             color=ACCENT, space_after=0)
    if number is not None:
        _, tf = txbox(slide, SW - M - 1.6, 0.34, 1.6, 0.3)
        para(tf, str(number), first=True, size=10.5, color=MUTED,
             align=PP_ALIGN.RIGHT, space_after=0)
    if title:
        _, tf = txbox(slide, M, 0.72, SW - 2 * M, 0.72)
        para(tf, title, first=True, size=27, bold=True, color=INK, space_after=0)
        rect(slide, M, 1.5, 1.15, 0.055, fill=ACCENT)
    if kicker:
        _, tf = txbox(slide, M, 1.66, SW - 2 * M, 0.36)
        para(tf, kicker, first=True, size=13, color=MUTED, space_after=0)
    return slide


def placeholder(slide, x, y, w, h, label, hint, lines=3):
    """A dashed-ish content area the author fills in (图 / 表 / 代码 / 案例)."""
    rect(slide, x, y, w, h, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
    _, tf = txbox(slide, x + 0.22, y + 0.18, w - 0.44, h - 0.36)
    para(tf, label, first=True, size=13, bold=True, color=ACCENT, space_after=4)
    para(tf, hint, size=11.5, color=MUTED, line=1.2, space_after=0)
    for _ in range(lines):
        p = tf.add_paragraph()
        p.text = "· "
        style(p, size=11.5, color=PLACE_LINE, space_after=2)


def bullets(tf, items, size=14, first_flag=True):
    """items: str  or  (text, level)  or  (text, level, {ov})"""
    first = first_flag
    for it in items:
        ov = {}
        if isinstance(it, str):
            text, lvl = it, 0
        elif len(it) == 2:
            text, lvl = it
        else:
            text, lvl = it[0], it[1]
            ov = it[2]
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = ("— " if lvl == 1 else "") + ("" if lvl else "▪ ") + text
        style(p, size=ov.get("size", size if lvl == 0 else size - 1.5),
              bold=ov.get("bold", False),
              color=ov.get("color", BODY if lvl == 0 else MUTED),
              space_after=ov.get("space_after", 7 if lvl == 0 else 5),
              line=1.22)
    return tf


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def table(slide, x, y, w, rows, col_w=None, header=True, size=12,
          row_h=0.42, head_h=0.46):
    """rows: list of list of str; first row = header when header=True"""
    nrows, ncols = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y),
                                 Inches(w), Inches(head_h + row_h * (nrows - 1)))
    tbl = shp.table
    tbl.first_row = header
    tbl.horz_banding = True
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(Inches(w) * cw / total))
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(head_h if (header and ri == 0) else row_h)
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = Inches(0.09)
            cell.margin_right = Inches(0.07)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if header and ri == 0:
                cell.fill.fore_color.rgb = ACCENT
            else:
                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if ri % 2 else RGBColor(0xF4, 0xF7, 0xFB)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = val
            style(p, size=size, bold=(header and ri == 0),
                  color=RGBColor(0xFF, 0xFF, 0xFF) if (header and ri == 0) else BODY,
                  space_after=0, line=1.1)
    return tbl


def timeline(slide, x, y, w, items, h=0.62):
    """Horizontal chevron chain. items: list of (title, sub)"""
    n = len(items)
    gap = 0.12
    cw = (w - gap * (n - 1)) / n
    for i, (t, s) in enumerate(items):
        cx = x + i * (cw + gap)
        rect(slide, cx, y, cw, h, fill=CHIP_BG, line=PLACE_LINE, lw=0.75, radius=0.16)
        _, tf = txbox(slide, cx + 0.1, y + 0.08, cw - 0.2, h - 0.16)
        para(tf, t, first=True, size=12.5, bold=True, color=ACCENT,
             align=PP_ALIGN.CENTER, space_after=1)
        para(tf, s, size=9.5, color=MUTED, align=PP_ALIGN.CENTER, space_after=0, line=1.05)


# ---------------------------------------------------------------- deck
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)

# ---------------------------------------------------------------- 1 cover
s = blank(prs)
rect(s, 0, 0, SW, 2.55, fill=RGBColor(0x12, 0x2A, 0x48))
rect(s, 0, 2.55, SW, 0.07, fill=ACCENT)
_, tf = txbox(s, M + 0.1, 0.62, SW - 2 * M, 0.4)
para(tf, "TECH SHARING  ·  LLM RETRIEVAL STACK", first=True, size=12,
     bold=True, color=RGBColor(0x8F, 0xB8, 0xE8), space_after=0)

_, tf = txbox(s, M + 0.1, 1.08, SW - 2 * M - 0.4, 1.3)
para(tf, "From Embedding Models to Agentic RAG", first=True, size=40,
     bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), space_after=0)
para(tf, "从 BERT 到 Qwen3-Embedding,从 Native RAG 到 Agentic RAG",
     size=15, color=RGBColor(0xB8, 0xCD, 0xE6), space_after=0)

_, tf = txbox(s, M + 0.1, 3.0, SW - 2 * M, 1.5)
rich(tf, [("分享人:", {"bold": True, "color": INK}), ("  ______________     ", {}),
          ("日期:", {"bold": True, "color": INK}), ("  ______ 年 ___ 月 ___ 日", {})],
     first=True, size=13, space_after=10)
rich(tf, [("时长:", {"bold": True, "color": INK}),
          ("  约 60 min(Embedding 35 min + Agentic RAG 25 min)      ", {}),
          ("形式:", {"bold": True, "color": INK}), ("  线上 / 线下分享会", {})],
     size=13, space_after=10)
rich(tf, [("配套材料:", {"bold": True, "color": INK}),
          ("  RAG and Agentic AI.md(大纲原文)", {})], size=13, space_after=0)

rect(s, M + 0.1, 5.0, SW - 2 * M - 0.2, 1.5, fill=PLACE_BG, line=PLACE_LINE,
     lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.34, 5.18, SW - 2 * M - 0.68, 1.2)
para(tf, "开场白占位(30 秒,建议先讲)", first=True, size=12.5, bold=True,
     color=ACCENT, space_after=5)
para(tf, "· 为什么讲这个:检索质量决定了 RAG 的上限,而 Embedding 决定检索质量。", size=11.5, color=BODY, space_after=3)
para(tf, "· 目标:听完能回答「我的场景该选哪个 embedding 模型」「我的 RAG 卡在哪一环,该用哪个 agentic 技巧」。", size=11.5, color=BODY, space_after=0)
notes(s, "封面。建议先把「两条主线」讲清楚:第一条线是 Embedding 模型的技术演进(表征能力),"
         "第二条线是 RAG 的架构演进(控制流)。两条线交汇点就是:更好的 embedding 让检索更准,"
         "更强的 agent 让检索更「会挑时机、会改问题、会自我纠正」。")

# ---------------------------------------------------------------- 2 agenda
s = blank(prs)
base(s, "Agenda", 2, "今天讲什么", "两条主线:表征能力(Embedding)× 控制流(RAG 架构)")
rect(s, M, 2.15, (SW - 2 * M - 0.3) / 2, 4.3, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.28, 2.38, (SW - 2 * M - 0.3) / 2 - 0.56, 3.9)
para(tf, "PART 1  ·  Embedding Model 演进", first=True, size=17, bold=True,
     color=ACCENT, space_after=12)
bullets(tf, [
    "BERT — Encoder-only 起点:双向 MLM + NSP",
    "SBERT — 用 Siamese 网络解决句子级相似度",
    "MTEB — 统一的评测基准(从「炼丹」到「比榜」)",
    "BGE — 中文第一梯队:RetroMAE + 对比学习 + 微调",
    "NV-Embed — Decoder-only + Latent Attention",
    "Qwen3-Embedding — 统一 Embedding / Reranker 双模",
], size=13.5, first_flag=False)
notes(s, "议程页。强调两部分的动机不一样:Part 1 是「模型怎么变强」,Part 2 是「系统怎么变聪明」。"
         "可以提一句:工业界落地的瓶颈往往不是模型不够强,而是系统不会在正确的时机做正确的事。")

rect(s, M + (SW - 2 * M - 0.3) / 2 + 0.3, 2.15, (SW - 2 * M - 0.3) / 2, 4.3,
     fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + (SW - 2 * M - 0.3) / 2 + 0.58, 2.38,
              (SW - 2 * M - 0.3) / 2 - 0.56, 3.9)
para(tf, "PART 2  ·  From RAG to Agentic RAG", first=True, size=17, bold=True,
     color=ACCENT2, space_after=12)
bullets(tf, [
    "Native RAG 的两个结构性弱点",
    "ReAct RAG → Agentic RAG 的范式转变",
    "查询侧:Query Rewrite / Multi-Query",
    "知识库侧:Graph RAG / RAPTOR",
    "检索后侧:Reranker / Self-correction / Corrective RAG",
    "编排侧:Adaptive RAG(以及 Backup:FLARE / SELF-RAG)",
], size=13.5, first_flag=False)
notes(s, "注意踩点:每个技巧都问一句「它插在 RAG 流水线的哪一环」。这样听众不会把它们记成一堆名词。")

# ---------------------------------------------------------------- 3 section divider
s = blank(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x12, 0x2A, 0x48))
rect(s, 0, 3.05, 2.2, 0.09, fill=ACCENT)
_, tf = txbox(s, M + 0.2, 3.35, SW - 2 * M, 1.4)
para(tf, "PART 1", first=True, size=15, bold=True,
     color=RGBColor(0x8F, 0xB8, 0xE8), space_after=8)
para(tf, "Embedding Model 演进", size=38, bold=True,
     color=RGBColor(0xFF, 0xFF, 0xFF), space_after=8)
para(tf, "BERT → SBERT → MTEB → BGE → NV-Embed → Qwen3-Embedding",
     size=14, color=RGBColor(0xB8, 0xCD, 0xE6), space_after=0)
notes(s, "过渡页。一句话定调:embedding 模型的目标从来没有变过——把语义压到一个"
         "「距离就有意义」的向量空间。变的是训练目标、规模和评测方式。")

# ---------------------------------------------------------------- 4 BERT intro
s = blank(prs)
timeline(s, M, 2.12, SW - 2 * M, [
    ("BERT", "Encoder-only,双向"), ("SBERT", "句子级相似度"),
    ("MTEB", "统一评测"), ("BGE", "中文强,三阶段"),
    ("NV-Embed", "Decoder-only"), ("Qwen3-Emb", "双模统一"),
], h=0.72)
base(s, "Part 1 · Embedding Model: BERT", 4, "BERT:Encoder-only 的起点",
     "2018 · 把「预训练 + 微调」范式带入 NLP")
rect(s, M, 3.05, 6.5, 3.4, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 3.26, 6.0, 3.0)
para(tf, "四个关键设计", first=True, size=15, bold=True, color=ACCENT, space_after=9)
bullets(tf, [
    "Encoder-only Transformer:只用编码器堆叠,没有自回归解码器",
    "Bidirectional:双向 self-attention,每个 token 同时看左右上下文",
    "Masked Language Model (MLM):随机 mask 15% token,预测被 mask 的词",
    "输出是「每个 token 一个向量」,不是「一句话一个向量」",
], size=13.5, first_flag=False)
placeholder(s, M + 6.9, 3.05, SW - 2 * M - 6.9, 3.4,
            "占位:架构图 / 数据规模", "建议放 BERT-base vs BERT-large 参数表(L=12/24, H=768/1024, A=12/16, ~110M/340M)",
            lines=4)
notes(s, "重点埋一个伏笔:MLA/MLM 是 token 级任务,所以 BERT 原生输出的是 token 向量,"
         "要拿到 sentence embedding 需要额外手段(CLS 或 mean pooling)——这正是后面 SBERT 要解决的问题。"
         "可以提一句 BERT 的输入表示:token + segment + position 三种 embedding 相加。")

# ---------------------------------------------------------------- 5 BERT pretraining
s = blank(prs)
base(s, "Part 1 · Embedding Model: BERT", 5, "BERT 的预训练:两个自监督任务",
     "How pretraining is done")
rect(s, M, 2.12, 6.5, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 6.0, 3.9)
para(tf, "1. Bidirectional MLM", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "随机选 15% 的 token 做预测;其中 80% 换成 [MASK]、10% 换随机词、10% 保持不变",
    "为什么不是 100% [MASK]:避免预训练与微调阶段的输入分布不一致",
    "为什么是双向:能看到左右文,比 GPT 式「只看左边」更适合做理解任务",
], size=13, first_flag=False)
para(tf, "2. Next Sentence Prediction (NSP)", size=15, bold=True,
     color=ACCENT, space_after=8)
bullets(tf, [
    "输入两个句子 A / B,预测 B 是否是 A 的真实下一句(50% 正例)",
    "目的:让模型学到句子之间的关系 —— 对 QA、NLI 这类任务有用",
    "后续研究(RoBERTa)发现 NSP 收益有限,SOP(Sentence-Order Prediction)更好",
], size=13, first_flag=False)
placeholder(s, M + 6.9, 2.12, SW - 2 * M - 6.9, 4.33,
            "占位:MLM 掩码示例 / 预训练流程", "建议画:输入串 → mask 位置 → Encoder → 只对被 mask 位置算 loss",
            lines=6)
notes(s, "这一页要讲出「为什么这两个任务能学到语义」。MLM 学到的是词汇/句法/常识的分布式表示;"
         "NSP 学到句子级连贯性。同时点出局限:NSP 提供的是「句子级」监督但也引入了噪声,"
         "而真正的句子语义相似度并没有被直接优化——这是 SBERT 的动机。")

# ---------------------------------------------------------------- 6 BERT finetune
s = blank(prs)
base(s, "Part 1 · Embedding Model: BERT", 6, "BERT 的微调:带标注的下游任务",
     "How fine-tuning is done —— 含 Question-Answer Pair")
rect(s, M, 2.12, 6.6, 2.22, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.30, 6.1, 1.9)
para(tf, "标准流程", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "预训练权重 + 任务头(classification / span extraction)→ 端到端微调",
    "QA 任务(SQuAD 式):输入 [CLS] Question [SEP] Passage [SEP],输出答案 span 的 start / end 位置",
    "训练目标:cross-entropy;小学习率(2e-5 ~ 5e-5)、3~4 个 epoch 即可收敛",
], size=13, first_flag=False)
rect(s, M, 4.44, 6.6, 2.01, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 4.60, 6.1, 1.75)
para(tf, "作为 Embedding 模型用时的坑", size=14, bold=True, color=ACCENT2, space_after=7)
bullets(tf, [
    "BERT 没有为「句子相似度」微调过 → CLS 向量各向异性(anisotropy),余弦相似度几乎无区分度",
    "常见补救:取 mean pooling、或对向量做 whitening / 标准化",
    "同一批句子两两组合(Cross-Encoder)效果好但 O(n²) 推理,无法做海量检索",
], size=12, first_flag=False)
placeholder(s, M + 7.0, 2.12, SW - 2 * M - 7.0, 4.33,
            "占位:微调数据样例 / 代码片段", "建议贴:Question-Answer Pair 样例(question, passage, label)、以及 3~5 行 HuggingFace 微调代码",
            lines=6)
notes(s, "关键过渡点:讲清「BERT 微调 ≠ 得到 embedding 模型」。微调后的 BERT 做分类很强,"
         "但直接拿 CLS 向量算相似度效果很差(Sentence-BERT 论文里 STS 上 Spearman 只有 0.5x)。"
         "这就引出下一节 SBERT。")

# ---------------------------------------------------------------- 7 SBERT why
s = blank(prs)
base(s, "Part 1 · Embedding Model: SBERT", 7, "BERT 为什么做不好句子相似度",
     "Why BERT fails at semantic text similarity (sentence level)")
rect(s, M, 2.12, 6.2, 2.22, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 2.30, 5.7, 1.9)
para(tf, "根本原因:训练目标不匹配", first=True, size=14, bold=True, color=ACCENT2, space_after=6)
para(tf, "MLM / NSP 优化的是 token 级和句子级「是否相连」,从未优化「两句意思有多接近」;"
         "没有监督信号把「语义近」映射到「向量近」。", size=12, color=BODY, space_after=0, line=1.2)
table(s, M, 4.52, 6.2, [
    ["", "Cross-Encoder", "Bi-Encoder (SBERT)"],
    ["输入", "[Q][SEP][D] 一起进模型", "Q 和 D 分别独立编码"],
    ["交互", "逐 token 交叉注意力", "只在最后算向量相似度"],
    ["精度", "高(STS 上更好)", "略低但非常接近"],
    ["速度", "n 个文档 → n 次前向", "文档向量可离线预计算"],
    ["1 万句检索", "~65 小时", "~5 秒(约 4.6 万倍)"],
    ["用途", "重排 Reranker(小候选集)", "召回 / 大规模检索"],
], col_w=[1.0, 1.6, 1.7], size=10.5, row_h=0.32, head_h=0.36)
placeholder(s, M + 6.6, 2.12, SW - 2 * M - 6.6, 4.33,
            "占位:两种结构对比图", "建议画:上=Cross-Encoder(Q、D 拼一起→1 个打分);下=Bi-Encoder(Q→u, D→v → cosine)",
            lines=6)
notes(s, "这一页是全 Part 1 的技术核心。务必讲清 Cross-Encoder 与 Bi-Encoder 的本质区别:"
         "「交互发生在哪里」。Cross-Encoder 的交互在中间层,所以精度高但不可预计算;"
         "Bi-Encoder 把交互推迟到最后一步(点积/余弦),代价是精度略降,收益是可预计算 → 可上 ANN 索引。"
         "表格里的 65 小时 vs 5 秒来自 SBERT 原论文(10k 句子)。")

# ---------------------------------------------------------------- 8 SBERT siamese
s = blank(prs)
base(s, "Part 1 · Embedding Model: SBERT", 8, "Siamese 网络与训练目标",
     "Siamese Network, training loss")
rect(s, M, 2.12, 6.5, 2.5, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 6.0, 2.1)
para(tf, "网络结构", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "共享权重的双塔(Siamese / twin network):同一个 BERT 编码器分别编码句子 A、B",
    "Pooling 层:取 mean 或 max(默认 mean)把 token 向量压成定长句向量;CLS 也可用",
    "默认输出 768 维,可选再过一个全连接层降到 256/512 维",
], size=13, first_flag=False)
rect(s, M, 4.82, 6.5, 1.63, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 5.00, 6.0, 1.7)
para(tf, "训练技巧", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "分类目标:softmax 三分类(entailment / neutral / contradiction),推理时用 cosine",
    "大 batch + 长训练步数很关键(原论文 batch 16 太小,后续 SBERT 用 1024)",
], size=13, first_flag=False)
placeholder(s, M + 6.9, 2.12, SW - 2 * M - 6.9, 4.33,
            "占位:Siamese 结构图 + 三个 loss 公式", "建议画:左塔 BERT→pooling→u,右塔共享权重→pooling→v,中间 similarity;下方写 cosine / MSE / triplet 三种 loss",
            lines=6)
notes(s, "三种 loss 各自适用场景要讲清:\n"
         "1) Classification objective:u、v、|u-v| 拼接后过 softmax,适合有 NLI 标注的场景(SNLI+MNLI 达 100 万对)。\n"
         "2) Regression objective:直接对 cosine 相似度做 MSE 回归,适合有 1~5 分相似度标注。\n"
         "3) Triplet objective:max(||s_a - s_p|| - ||s_a - s_n|| + ε, 0),锚点-正例-负例。\n"
         "另外强调:in-batch negatives 是让大 batch 生效的关键——同 batch 内其他样本即天然负例。")

# ---------------------------------------------------------------- 9 SBERT perf
s = blank(prs)
base(s, "Part 1 · Embedding Model: SBERT", 9, "效果:从 0.5 到 0.8 的跃升",
     "Performance")
table(s, M, 2.15, 7.4, [
    ["方法", "STS 任务表现", "说明"],
    ["Avg. GloVe embeddings", "Spearman ≈ 0.52", "词向量平均,弱基线"],
    ["BERT CLS / mean 向量", "≈ 0.51 ~ 0.60", "未针对相似度训练"],
    ["SBERT (base)", "≈ 0.76", "Siamese + NLI 数据微调"],
    ["SBERT (roberta-large)", "≈ 0.79", "更大基座、更长训练"],
], col_w=[1.5, 1.1, 1.4], size=11.5, row_h=0.38, head_h=0.4)
_, tf = txbox(s, M, 5.05, 7.4, 1.4)
para(tf, "结论", first=True, size=14, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "同样的 BERT 权重,换一个训练目标,句子相似度能力直接跳一个档",
    "并且保留了「可预计算向量 + ANN 检索」的工程可行性 → 现代 RAG 的底座",
], size=12.5, first_flag=False)
placeholder(s, M + 7.8, 2.15, SW - 2 * M - 7.8, 4.3,
            "占位:STS 各数据集柱状图", "数据来源:Sentence-BERT 论文(Reimers & Gurevych, 2019) Table 2 / 官方文档;建议贴论文原图",
            lines=6)
notes(s, "数据务必标注来源(论文里的具体数值随设置不同会有浮动),现场可以说「数量级」而不是精确值。"
         "核心 takeaway:embedding 模型的强弱,很大程度上是训练目标(损失函数 + 数据构造)决定的,而不是只靠更大基座。")

# ---------------------------------------------------------------- 10 MTEB
s = blank(prs)
base(s, "Part 1 · Embedding Model: MTEB", 10, "MTEB:统一的 Embedding 评测基准",
     "Massive Text Embedding Benchmark —— 让模型可比")
rect(s, M, 2.12, 6.3, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "是什么", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "MTEB 论文(2022)开源的大规模 embedding 评测套件",
    "覆盖 8 类任务:Retrieval、Reranking、Clustering、Classification、Pair Classification、STS、Summarization、Bitext Mining",
    "初版约 56 个数据集、112 种语言;后续持续扩展到上百个数据集",
    "同时提供 C-MTEB(中文)与 MMTEB(多语言),Leaderboard 公开排名",
], size=13, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:MTEB 任务/数据规模与榜单截图", "建议贴:8 类任务示意 + 当前 MTEB Leaderboard 前几名(Multilingual / Chinese 两个榜)",
            lines=6)
notes(s, "为什么需要它:在 MTEB 之前,每个模型都在自己那 1~2 个数据集上报数,不可比。"
         "MTEB 的价值是「统一口径」。同时要提醒两个常见误用:\n"
         "1) 榜单高分 ≠ 你的业务场景效果,一定要用自己的数据做小规模评测;\n"
         "2) 注意区分「检索榜(Retrieval)」和「STS 榜」,前者更贴近 RAG。")

# ---------------------------------------------------------------- 11 BGE C-MTEB
s = blank(prs)
base(s, "Part 1 · Embedding Model: BGE", 11, "BGE:中文场景的第一梯队",
     "BAAI General Embedding —— 从 C-MTEB 说起")
rect(s, M, 2.12, 6.3, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "C-MTEB(中文 MTEB)", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "BGE 团队构建的中文评测集:6 类任务、约 35 个数据集,覆盖检索/排序/聚类/分类/STS/句对分类",
    "数据来自中文开源数据集,统一 re-split 与格式化,让中文模型第一次有了可比口径",
    "BGE 系列包含:英文 bge-large-en、中文 bge-large-zh、多语言 bge-m3,以及轻量 bge-small/base",
], size=13, first_flag=False)
para(tf, "为什么中文要单独立榜", size=14, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "分词与语义分布不同,英文榜高分模型在中文上可能明显掉点",
    "中文检索常涉及短 query + 长文档、专有名词,评测集要贴近这个分布",
], size=13, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:C-MTEB 榜单 / BGE 模型规格表", "建议贴:bge-large-zh-v1.5 在 C-MTEB 的均分与各任务分,以及 bge-m3 的 dense/sparse/colbert 三输出示意",
            lines=6)
notes(s, "可以顺带提 bge-m3 的三个卖点:多语言(100+)、多粒度(最长 8192 token)、"
         "多向量输出(dense + sparse + ColBERT 式 multi-vector),一次编码支持三种检索模式。"
         "BGE 也是国内 RAG 项目最常见的中文 embedding 选择。")

# ---------------------------------------------------------------- 12 BGE stage1
s = blank(prs)
base(s, "Part 1 · Embedding Model: BGE", 12, "三阶段训练(1/3):RetroMAE 预训练",
     "Stage 1 —— 先把基座的语言与语义理解做扎实")
rect(s, M, 2.12, 6.4, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.9, 3.9)
para(tf, "RetroMAE 是什么", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "一种面向检索场景的 MAE(Masked AutoEncoder)式预训练方法",
    "Encoder 侧:对输入做高比例 mask(如 15%~30%),编码出句子表示",
    "Decoder 侧:用「浅层 + 非对称」解码器,从句子表示重建被 mask 的原始 token",
    "关键点:decoder 只用于预训练,推理时丢弃,所以不增加线上成本",
], size=13, first_flag=False)
para(tf, "为什么有效", size=14, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "重建任务迫使 encoder 的句向量必须携带「可还原全文」的完整信息 → 表示更充分",
    "相比纯 MLM,更贴近「用一个向量代表整段文本」的目标,为后续对比学习打好初始化",
], size=13, first_flag=False)
placeholder(s, M + 6.8, 2.12, SW - 2 * M - 6.8, 4.33,
            "占位:RetroMAE 流程图", "建议画:输入 →(高比例 mask)→ Encoder → 句向量 → 非对称浅层 Decoder → 重建被 mask token;标注「decoder 推理时丢弃」",
            lines=6)
notes(s, "阶段定位要讲清:三阶段是「预训练打底 → 对比学习对齐 → 有监督微调提精度」的递进关系,"
         "每一步都在上一阶段的权重上继续。RetroMAE 属于第一阶段,目标是让基座会「压缩理解」。")

# ---------------------------------------------------------------- 13 BGE stage2
s = blank(prs)
base(s, "Part 1 · Embedding Model: BGE", 13, "三阶段训练(2/3):对比学习",
     "Stage 2 —— 用海量无标注文本对把语义对齐到向量距离")
rect(s, M, 2.12, 6.4, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.9, 3.9)
para(tf, "训练设置", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "目标函数:InfoNCE 式对比损失(同 SBERT 思路,但规模与数据构造不同)",
    "数据:大规模弱监督文本对 —— 标题-正文、问答对、同义改写、社区问答等",
    "温度系数 τ 控制分布锐度;大 batch(如 1024+)保证 in-batch negatives 足够多",
], size=13, first_flag=False)
para(tf, "两个工程要点", size=14, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "In-batch negatives:同 batch 内其他样本的文档当负例,零额外前向成本",
    "困难负例(hard negatives):相似但不匹配的文档对训练收益最大,需要专门挖掘",
], size=13, first_flag=False)
placeholder(s, M + 6.8, 2.12, SW - 2 * M - 6.8, 4.33,
            "占位:对比学习示意 + InfoNCE 公式", "建议画:锚点 q 与正例 d+ 拉近、与 batch 内负例 d- 推远;旁边写 -log(exp(sim(q,d+)/τ) / Σ exp(sim(q,d_i)/τ))",
            lines=6)
notes(s, "这一页的重点是「数据构造」而不是公式。可以强调:对比学习的效果上限很大程度由"
         "「正例够不够真、负例够不够难」决定。in-batch negatives 是免费的,hard negatives 是花钱的但收益更大。")

# ---------------------------------------------------------------- 14 BGE stage3 + perf
s = blank(prs)
base(s, "Part 1 · Embedding Model: BGE", 14, "三阶段训练(3/3):有监督微调 + 效果",
     "Stage 3 —— 高质量标注数据,把精度再抬一档")
rect(s, M, 2.12, 6.3, 2.5, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 2.1)
para(tf, "微调阶段", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "数据:人工/半人工标注的高质量检索三元组(query, pos, neg)",
    "负例构造:先用当前模型召回 top-k 近似文档,再挑选「看似相关但不正确」的作为 hard negative",
    "训练目标仍是对比损失,但数据更干净、负例更困难 → 检索指标显著提升",
    "可加 instruction 前缀(如「为这个句子生成表示用于检索」),提升跨任务泛化",
], size=12.5, first_flag=False)
table(s, M, 4.82, 6.3, [
    ["阶段", "目标", "数据"],
    ["1 RetroMAE", "基座表示能力", "大规模无标注语料"],
    ["2 对比学习", "语义对齐", "弱监督文本对"],
    ["3 微调", "检索精度", "高质量标注 + 难负例"],
], col_w=[1.2, 1.2, 1.6], size=10.5, row_h=0.33, head_h=0.36)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:BGE 在 C-MTEB / MTEB 的分数对比", "建议贴:bge-large-zh-v1.5 与 bge-m3 的均分、以及相对 OpenAI text-embedding 系列的中文对比(注明来源与评测时间)",
            lines=6)
notes(s, "把三阶段串成一个故事:先学会「压缩理解」(RetroMAE),再学会「距离有意义」(对比学习),"
         "最后用标注数据把「难的边界」抠清楚(微调 + 难负例)。"
         "提醒:榜单数字有时效性,建议现场强调趋势而不是绝对分数。")

# ---------------------------------------------------------------- 15 NV-Embed
s = blank(prs)
base(s, "Part 1 · Embedding Model: NV-Embed", 15, "NV-Embed:Decoder-only 也能做 Embedding",
     "NVIDIA · 把 LLM 直接改造成 embedding 模型")
rect(s, M, 2.12, 6.3, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "三个关键改动", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Decoder-only 基座:直接用 Mistral-7B 这类自回归 LLM 做 embedding,不做架构改造",
    "Latent Attention Layer:在最后加一层 latent attention,把变长的 token 表示聚合成一个定长向量(替代 mean pooling,且去掉因果 mask 让 token 双向可见)",
    "两阶段对比训练:Stage 1 用大规模弱监督数据(检索式 + 非检索式任务混合)微调;Stage 2 用高质量数据精调",
], size=12.5, first_flag=False)
rect(s, M, 5.05, 6.3, 1.4, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 5.2, 5.8, 1.15)
para(tf, "反直觉的点:第二阶段不用 in-batch negatives", first=True, size=13,
     bold=True, color=ACCENT2, space_after=5)
para(tf, "第二阶段改用「同一 batch 内同一 query 的其他正例 / 跨设备(across-device)负例」等策略后效果更好 —— "
         "说明当基座足够强、数据足够干净时,粗暴的 in-batch negative 反而引入假负例噪声。",
     size=11.5, color=BODY, space_after=0, line=1.18)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:Latent Attention 结构图 + MTEB 分数", "建议画:LLM 各层 hidden → latent attention(可学习 query 向量做 cross-attn)→ 单个 embedding;并贴论文在 MTEB 上的均分与任务数",
            lines=6)
notes(s, "这一页最大的启发:embedding 模型不一定要「双塔 + pooling」的经典形态,"
         "一个通用 LLM 加一层聚合就能打到榜一。同时点出趋势:embedding 与 LLM 的边界在模糊化,"
         "后面 Qwen3-Embedding 也是同一条路线(用 LLM 基座统一 embedding 与 reranker)。")

# ---------------------------------------------------------------- 16 Qwen3-Embedding intro
s = blank(prs)
base(s, "Part 1 · Embedding Model: Qwen3-Embedding", 16, "Qwen3-Embedding:一个基座,两种模型",
     "Introduction —— 以 LLM 为底座统一 Embedding 与 Reranking")
rect(s, M, 2.12, 6.3, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "家族与能力", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "模型规格:0.6B / 4B / 8B(Embedding),另有对应的 Reranker 版本",
    "以 Qwen3 系列 LLM 为初始化基座,继承多语言与长文本能力",
    "超长上下文:支持 32k token;向量维度支持自定义(常见 1024/2560/4096,可裁剪到更低维)",
    "覆盖 100+ 语言(含编程语言),检索、聚类、分类、双语挖掘等多任务",
], size=13, first_flag=False)
para(tf, "设计思想:Embedding 与 Reranker 同源", size=14, bold=True,
     color=ACCENT, space_after=6)
bullets(tf, [
    "同一个基座,只用不同的「输出头」和「拼接方式」区分两种用途",
    "好处:训练信号可互相迁移,线上可复用同一套推理基础设施",
], size=13, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:模型规格表 / 多语言与榜单成绩", "建议贴:0.6B/4B/8B 的维度、层数、上下文长度,以及 MTEB Multilingual、MTEB Code 的分数",
            lines=6)
notes(s, "承接 NV-Embed:继续强调「LLM 基座做 embedding」这条路线。"
         "Qwen3-Embedding 的工程价值在于嵌入维度可定制 + 32k 上下文 + 多语言,"
         "对中文和代码检索都比较友好,是目前开源自建 RAG 的热门选择。")

# ---------------------------------------------------------------- 17 Qwen3-Embedding mechanics
s = blank(prs)
base(s, "Part 1 · Embedding Model: Qwen3-Embedding", 17, "输入格式、Reranker 与训练损失",
     "拿 EOS 当句向量,Reranker 只输出 Yes / No")
rect(s, M, 2.12, 6.2, 2.35, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 2.30, 5.7, 2.0)
para(tf, "Embedding model:输入 → EOS Pooling", first=True, size=14,
     bold=True, color=ACCENT, space_after=6)
rich(tf, [("{Inst}{Q}<EOS>", {"font": MONO, "bold": True, "color": INK}),
          ("   →   ", {"color": MUTED}),
          ("取 <EOS> 位置的隐状态作为句向量", {"color": BODY})],
     size=12.5, space_after=5)
para(tf, "用最后一个 token(而非 mean/CLS)做 pooling:与 decoder-only 的因果注意力特性一致 —— "
         "序列末尾的表示已经「看过」全部前文,天然是整段的压缩。", size=11.5, color=BODY,
     space_after=0, line=1.18)
rect(s, M, 4.62, 6.2, 1.83, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 4.80, 5.7, 1.5)
para(tf, "Reranker:输入 → LM Head → Yes / No", first=True, size=14,
     bold=True, color=ACCENT, space_after=6)
rich(tf, [("{Inst}{Q}{Doc}", {"font": MONO, "bold": True, "color": INK}),
          ("  →  ", {"color": MUTED}),
          ("LM Head", {"font": MONO, "color": INK}), ("  →  ", {"color": MUTED}),
          ("比较 “Yes” / “No” 两个 token 的 logit", {"color": BODY})],
     size=12.5, space_after=5)
para(tf, "对两个 logit 做 softmax 得到相关性分数。本质是把「相关性判断」变成一个二分类问题,"
         "复用了 LLM 原本的语言建模能力,不需要额外的打分头。", size=11.5, color=BODY,
     space_after=0, line=1.18)
rect(s, M + 6.6, 2.12, SW - 2 * M - 6.6, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 6.86, 2.34, SW - 2 * M - 7.12, 3.9)
para(tf, "训练损失:InfoNCE 风格对比损失", first=True, size=14, bold=True,
     color=ACCENT, space_after=8)
bullets(tf, [
    "query 与正例 doc 更近、与 batch 内负例更远",
    "Reranker 侧同样用对比式目标,只把「相似度」换成 Yes/No 概率",
    "配套指令(instruction)可分别优化检索、分类、聚类等不同任务的向量空间",
], size=12.5, first_flag=False)
placeholder(s, M + 6.86, 4.35, SW - 2 * M - 7.12, 1.9,
            "占位:公式与代码", "建议贴:InfoNCE 公式、`get_detailed_instruct()` 用法示例、以及 embed/rerank 各 3 行调用代码",
            lines=2)
notes(s, "这一页是全 Part 1 的收束,把「三种 pooling 思路」做个对照:\n"
         "BERT 用 CLS、SBERT 用 mean pooling、Qwen3 用 EOS(last token)。\n"
         "可以让听众记住一句话:pooling 的选择取决于基座的注意力结构 —— 双向编码器适合 mean,"
         "因果解码器适合取最后一位。\n"
         "Reranker 用 Yes/No logit 是个很优雅的设计,现场可以演示一下 rerank 前后 top-5 的变化。")

# ---------------------------------------------------------------- 18 Part 2 divider
s = blank(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x3A, 0x1E, 0x14))
rect(s, 0, 3.05, 2.2, 0.09, fill=ACCENT2)
_, tf = txbox(s, M + 0.2, 3.35, SW - 2 * M, 1.4)
para(tf, "PART 2", first=True, size=15, bold=True,
     color=RGBColor(0xE8, 0xB8, 0x9A), space_after=8)
para(tf, "From RAG to Agentic RAG", size=38, bold=True,
     color=RGBColor(0xFF, 0xFF, 0xFF), space_after=8)
para(tf, "当检索质量到瓶颈之后,提升点在「控制流」而不只是「模型」",
     size=14, color=RGBColor(0xE6, 0xCD, 0xB8), space_after=0)
notes(s, "过渡页。可以说一句串联的话:Part 1 我们一直在优化「检索得准不准」,"
         "Part 2 我们换一个角度——让系统自己决定「什么时候检索、检索什么、检索完信不信」。")

# ---------------------------------------------------------------- 19 native RAG
s = blank(prs)
base(s, "Part 2 · Agentic RAG", 19, "Native RAG 的两个结构性弱点",
     "先看清「朴素流水线」到底在哪里漏水")
_, tf = txbox(s, M, 2.12, SW - 2 * M, 0.9)
para(tf, "Naive RAG 流水线:用户 Query →(embedding)→ 向量库 Top-K → 拼进 Prompt → LLM 生成",
     first=True, size=13, bold=True, color=INK, space_after=6)
para(tf, "每一步都是单向、一次性、无反馈的 —— 没有一步会「回头检查」。",
     size=12, color=MUTED, space_after=0)
rect(s, M, 3.1, 6.3, 3.35, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 3.32, 5.8, 2.95)
para(tf, "弱点 1:Chunk size 直接决定信息完整性", first=True, size=14,
     bold=True, color=ACCENT2, space_after=8)
bullets(tf, [
    "切得太小:段落信息被切断,指代(它/该方案/上述)失去先行词,答案残缺",
    "切得太大:一个 chunk 混入多个主题,向量被「平均」掉,召回噪声升高",
    "固定长度切分忽略了文档结构(标题层级、表格、代码块)",
    "长度还牵动成本与上下文预算:chunk 越大,Top-K 塞进 prompt 越贵",
], size=12.5, first_flag=False)
rect(s, M + 6.7, 3.1, SW - 2 * M - 6.7, 3.35, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 6.96, 3.32, SW - 2 * M - 6.96, 2.95)
para(tf, "弱点 2:语义检索直接决定召回文档", first=True, size=14,
     bold=True, color=ACCENT2, space_after=8)
bullets(tf, [
    "「语义相似」不等于「能回答问题」——追问式 query 与答案段落措辞差异大",
    "单一向量表达不了多意图 query(如「对比 A 和 B 的定价」)",
    "对精确匹配需求(型号、编号、专有名词)天然弱于 BM25 / 关键词检索",
    "Top-K 一旦召错,后面再强的 LLM 也只能在错误上下文里编",
], size=12.5, first_flag=False)
notes(s, "这一页是 Part 2 的动机页,要讲得痛。建议现场举一个自己踩过的例子:"
         "比如用户问「A 方案的报价是多少」,而文档里写的是「该方案预算约 X 万元」,"
         "指代 + 措辞双重错位,导致召不回来。\n"
         "结论句:Native RAG 的问题是「没有反馈回路」,所以后面所有 agentic 技巧本质都是"
         "在流水线的某个位置插入一个「决策 / 修正」环节。")

# ---------------------------------------------------------------- 20 summary graph
s = blank(prs)
base(s, "Part 2 · Agentic RAG", 20, "Agentic RAG 全景图:技巧插在哪一环",
     "Query 侧 / 检索侧 / 知识库侧 / 检索后侧 / 编排侧")
# five columns
cols = [
    ("查询侧", ACCENT, ["Query Rewrite", "Multi-Query", "HyDE", "Query Decomposition"]),
    ("检索侧", ACCENT, ["Hybrid Search", "Reranker", "Metadata Filter", "ANN 调参"]),
    ("知识库侧", ACCENT2, ["Graph RAG", "RAPTOR", "Hierarchical Index", "多模态/表格索引"]),
    ("检索后侧", ACCENT2, ["Self-Reflection", "Corrective RAG", "上下文压缩", "引用与溯源"]),
    ("编排侧", RGBColor(0x6B, 0x3F, 0xA0), ["Adaptive RAG", "ReAct Loop", "Router / Planner", "FLARE · SELF-RAG"]),
]
cw = (SW - 2 * M - 4 * 0.18) / 5
for i, (name, color, items) in enumerate(cols):
    cx = M + i * (cw + 0.18)
    rect(s, cx, 2.25, cw, 0.5, fill=color, radius=0.12)
    _, tf = txbox(s, cx, 2.38, cw, 0.3)
    para(tf, name, first=True, size=13.5, bold=True,
         color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER, space_after=0)
    rect(s, cx, 2.75, cw, 2.55, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.06)
    _, tf = txbox(s, cx + 0.16, 2.92, cw - 0.32, 2.25)
    first = True
    for it in items:
        para(tf, it, first=first, size=11.5, color=BODY, space_after=7, line=1.12)
        first = False
rect(s, M, 5.55, SW - 2 * M, 1.85, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 5.74, SW - 2 * M - 0.52, 1.5)
para(tf, "怎么读这张图:一条 query 进来到答案出去,每个环节都能加一个「智能」环节;"
         "它们的共同点是引入反馈与决策 —— 要么改问题(查询侧),要么改证据(知识库/检索侧),"
         "要么改答案(检索后侧),要么决定整条流程走不走(编排侧)。",
     first=True, size=12.5, color=INK, space_after=8, line=1.25)
para(tf, "接下来按环逐一展开:Query Rewrite / Multi-Query → Adaptive RAG → Graph RAG → RAPTOR → Reranker → Self-correction & Corrective RAG。",
     size=12, color=MUTED, space_after=0)
notes(s, "这是全场的信息密度最高的一页,建议停 2 分钟。"
         "讲法:不要逐个念名词,而是强调「分类逻辑」——四类动作 + 一个总编排。"
         "听众带走这张分类框架,比记住 10 个名词更有用。")

# ---------------------------------------------------------------- 21 query rewrite
s = blank(prs)
base(s, "Part 2 · Agentic RAG", 21, "查询侧(1):Query Rewrite",
     "用户说的话 ≠ 能召回的 query")
rect(s, M, 2.12, 6.3, 2.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.32, 5.8, 1.85)
para(tf, "要解决什么", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "指代消解:「它多少钱」→ 补全为「XX 方案多少钱」",
    "口语 → 术语:把用户口语映射到知识库里的正式表述",
    "多轮上下文补全:把历史对话压缩进当前 query",
], size=12.5, first_flag=False)
table(s, M, 4.5, 6.3, [
    ["方法", "做法"],
    ["LLM 重写", "一次 LLM 调用,把原 query 改写为检索友好形式"],
    ["HyDE", "先让 LLM 生成一段假想答案,再用它去检索(文档空间更接近)"],
    ["Step-back", "先问一个更抽象的上位问题,再检索(提升推理题召回)"],
], col_w=[1.0, 2.3], size=10.5, row_h=0.34, head_h=0.36)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:重写前后对比示例", "建议贴 1~2 个真实 case:原 query、重写后 query、原本召回什么 / 重写后召回什么(附相似度分数)",
            lines=6)
notes(s, "强调 Query Rewrite 是「性价比最高」的一个技巧:一次额外 LLM 调用(成本低),"
         "但常常能解决多轮对话和口语化查询这两大类失败。\n"
         "注意风险:重写可能引入幻觉或改变用户意图,建议对关键实体做校验,"
         "或者保留原 query 一起检索(融合结果)。")

# ---------------------------------------------------------------- 22 multi-query
s = blank(prs)
base(s, "Part 2 · Agentic RAG", 22, "查询侧(2):Multi-Query 与查询分解",
     "一个向量召回不全,那就召多路再融合")
rect(s, M, 2.12, 6.3, 2.5, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.32, 5.8, 2.15)
para(tf, "Multi-Query", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "用 LLM 把原 query 改写成 N 个不同角度/措辞的 query",
    "分别检索,合并候选集,再做去重与重排(常见 RRF 倒数排名融合)",
    "收益:召回率提升,缓解「一句话只有一个向量」的表达瓶颈",
    "代价:N 倍检索开销 + 需要融合排序;N 一般取 3~5 即可",
], size=12.5, first_flag=False)
rect(s, M, 4.92, 6.3, 1.53, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 5.1, 5.8, 1.2)
para(tf, "Query Decomposition(相关问题)", first=True, size=13, bold=True,
     color=ACCENT, space_after=5)
para(tf, "把复合问题拆成子问题,分别检索后分别回答再汇总。例如「A 和 B 哪个更适合冷启动场景」"
         "→ 拆成「A 的冷启动能力」「B 的冷启动能力」两次检索。", size=11.5, color=BODY,
     space_after=0, line=1.18)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:Multi-Query 流程图 / RRF 公式", "建议画:query → LLM 生成 q1..q3 → 三路检索 → 合并去重 → rerank → top-k;并写 RRF: score = Σ 1/(k + rank_i)",
            lines=6)
notes(s, "Multi-Query 与 Reranker 是天然的一对:Multi-Query 负责「提高召回」,"
         "Reranker 负责「把多路召回的水分挤掉」。现场建议一起讲,因为只用 Multi-Query 不重排,"
         "准确率可能反而下降(噪声变多)。")

# ---------------------------------------------------------------- 23 adaptive RAG
s = blank(prs)
base(s, "Part 2 · Agentic RAG", 23, "编排侧:Adaptive RAG",
     "让系统自己决定「要不要检索、检索几次、检索多深」")
rect(s, M, 2.12, 6.3, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "核心思想", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "不是所有问题都需要检索。「1+1 等于几」走参数化知识就够,检索反而引入噪声和延迟",
    "典型做法:先用一个轻量分类器/LLM 判断 query 复杂度,路由到三条路径之一",
    "路径 A(不检索):直接由 LLM 回答",
    "路径 B(单步检索):一次检索 + 生成",
    "路径 C(多步检索):迭代检索、分解子问题、必要时重整 query",
    "判断依据可以来自 query 分类、检索结果的相关性分数、或生成答案的置信度",
], size=12.5, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:Adaptive RAG 路由流程图", "建议画:Query →(复杂度分类器)→ No Retrieval / Single-step / Multi-step 三条分支,并标注各自的延迟与成本差异",
            lines=6)
notes(s, "Adaptive RAG 是「编排侧」的代表:它管的不是某一步怎么做,而是整条流程走不走、走多久。\n"
         "工程上最大的价值是成本与延迟优化——把简单问题从检索路径摘出去,"
         "既省 token 又降延迟。现场可以给出自己场景里的路由比例(比如 30% 无需检索)。")

# ---------------------------------------------------------------- 24 graph RAG
s = blank(prs)
base(s, "Part 2 · Agentic RAG", 24, "知识库侧(1):Graph RAG",
     "当答案需要跨多个文档「连起来」时,向量检索会失效")
rect(s, M, 2.12, 6.3, 2.35, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.32, 5.8, 2.0)
para(tf, "解决什么问题", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "全局性问题:「整个语料库的主要主题是什么」—— 单个 chunk 里没有这个答案",
    "多跳问题:「A 的供应商的 CEO 还投资了哪些公司」—— 需要沿关系链走",
    "向量 Top-K 只能给「相似的片段」,给不了「关系结构」",
], size=12.5, first_flag=False)
rect(s, M, 4.77, 6.3, 1.68, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 4.96, 5.8, 1.35)
para(tf, "典型做法(Microsoft GraphRAG)", first=True, size=14, bold=True,
     color=ACCENT, space_after=7)
bullets(tf, [
    "用 LLM 从每个 chunk 抽实体 + 关系 + 声称(claim),构建知识图谱",
    "对社区(community)做分层聚类,并为每个社区预生成摘要",
    "查询时:局部检索走实体邻域;全局问题走社区摘要 map-reduce",
], size=12, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:知识图谱示意 + 社区分层", "建议画:实体节点+关系边的小图,外面用不同颜色的圈表示社区层级;并贴一个全局问题的 map-reduce 结果示例",
            lines=6)
notes(s, "必须讲清 GraphRAG 的代价:索引阶段 LLM 调用量极大(要遍历所有 chunk 做抽取 + 社区摘要),"
         "成本可能是普通向量索引的 10 倍以上,而且文档更新后重建昂贵。\n"
         "结论:只在「全局摘要类问题」或「多跳关系类问题」占比高的场景才值得上,"
         "不要为了技术时髦而默认上 GraphRAG。")

# ---------------------------------------------------------------- 25 RAPTOR
s = blank(prs)
base(s, "Part 2 · Agentic RAG", 25, "知识库侧(2):RAPTOR",
     "Recursive Abstractive Processing for Tree-Organized Retrieval")
rect(s, M, 2.12, 6.3, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "基本思路", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "把文档自底向上做「递归摘要」,构建一棵多层的树",
    "叶子层:原始 chunk;上层:对相邻 chunk 聚类后的 LLM 摘要",
    "递归进行直到形成根节点——根节点就是整篇文档的抽象摘要",
    "所有层级的节点都各自生成 embedding,一起进向量库",
], size=12.5, first_flag=False)
para(tf, "检索时", size=14, bold=True, color=ACCENT, space_after=7)
bullets(tf, [
    "一次查询可同时命中原 chunk(细节)和上层摘要(全局)",
    "等价于「让检索器自己决定要看细节还是看梗概」",
    "常见增强:用 collapsed tree(所有节点平铺一层)或 tree traversal 两种检索策略",
], size=12.5, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:RAPTOR 树结构图", "建议画:底部 6~8 个 chunk → 上一层 3 个聚类摘要 → 顶层 1 个根摘要,并标出 query 命中的两个不同层级节点",
            lines=6)
notes(s, "RAPTOR 与 GraphRAG 的对比讲一句:两者都在「知识库侧」加结构,"
         "但 GraphRAG 抽的是「实体关系图」,RAPTOR 建的是「摘要层级树」。"
         "RAPTOR 更适合「同一主题分散在多段、有的问细节有的问全局」的文档集,"
         "索引成本远低于 GraphRAG。")

# ---------------------------------------------------------------- 26 reranker
s = blank(prs)
base(s, "Part 2 · Agentic RAG", 26, "检索侧:Reranker",
     "召回可以宽,排序必须准")
rect(s, M, 2.12, 6.2, 2.4, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.32, 5.7, 2.05)
para(tf, "两阶段检索(Recall → Precision)", first=True, size=14, bold=True,
     color=ACCENT, space_after=8)
bullets(tf, [
    "第一阶段(召回):Bi-Encoder + ANN,从百万级里快速取 top-100,追求「不漏」",
    "第二阶段(重排):Cross-Encoder 逐个精算 query-doc 相关性,输出 top-5,追求「最准」",
    "为什么必须两段:Cross-Encoder 精度高但无法预计算,只能作用在小候选集上",
], size=12.5, first_flag=False)
rect(s, M, 4.72, 6.2, 1.73, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 4.9, 5.7, 1.4)
para(tf, "工程实践要点", first=True, size=13, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "候选集大小是主要调参项:50~100 通常是最佳区间,再往上收益递减、延迟线性增长",
    "常用模型:bge-reranker 系列、Qwen3-Reranker、Cohere Rerank 等",
], size=11.5, first_flag=False)
placeholder(s, M + 6.6, 2.12, SW - 2 * M - 6.6, 4.33,
            "占位:Rerank 前后 top-k 对比", "建议贴真实 case:同一 query 的 rerank 前 top-5 与后 top-5,标出相似度/相关性分数变化,以及最终答案质量差异",
            lines=6)
notes(s, "这是最容易见效、也最常被忽视的一环。很多团队把精力花在换 embedding 模型上,"
         "但加一个 reranker 的提升往往比换模型更大,而且改动最小。\n"
         "现场给个数字:常见的 nDCG / Recall@5 提升幅度。注意延迟成本(Cross-Encoder 是逐对计算)。")

# ---------------------------------------------------------------- 27 self-correction
s = blank(prs)
base(s, "Part 2 · Agentic RAG", 27, "检索后侧:Self-correction 与 ReAct 循环",
     "检索完不等于结束 —— 要检查证据够不够、答案站不站得住")
rect(s, M, 2.12, 6.3, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "Self-correction 的两个检查点", first=True, size=14, bold=True,
     color=ACCENT, space_after=8)
bullets(tf, [
    "检索后检查:召回的文档真的相关吗?不相关就重写 query 重检索",
    "生成后检查:答案的每一句有引用支撑吗?无支撑就标注或重答",
    "实现方式:额外的 LLM 调用做 critique,或训练专门的判别模型",
], size=12.5, first_flag=False)
para(tf, "ReAct 循环:Reason → Act → Observe", size=14, bold=True,
     color=ACCENT, space_after=7)
bullets(tf, [
    "把「检索」变成 agent 的一个 tool,由模型自己决定调用时机与参数",
    "多轮循环:想一步 → 调工具 → 看结果 → 再决定下一步",
    "与 Native RAG 的本质区别:检索次数和 query 内容不再由代码写死",
], size=12.5, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:ReAct 轨迹示例(Thought/Action/Observation)", "建议贴一条真实的多轮轨迹,展示模型如何先查 A、发现不够、再查 B;并标注每步的延迟与 token 消耗",
            lines=6)
notes(s, "要点:Self-correction 的收益与代价都很明显——它能把「答错」变成「答不出」,"
         "但循环次数失控会导致延迟和成本飙升。必须设上限(最多 N 轮)和预算。\n"
         "可以引用 Reflexion / Self-RAG 的思路:把 critique 本身也变成模型的输出之一。")

# ---------------------------------------------------------------- 28 corrective RAG
s = blank(prs)
base(s, "Part 2 · Agentic RAG", 28, "检索后侧:Corrective RAG (CRAG)",
     "给检索结果加一个「质量闸门」")
rect(s, M, 2.12, 6.3, 2.25, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.32, 5.8, 1.9)
para(tf, "三步流程", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "① 评估:一个轻量评估器给每个召回的文档打相关性分(模型自评或专门训练的检索评估器)",
    "② 分档:按分数把结果分为 Correct / Incorrect / Ambiguous 三档",
    "③ 动作:按档位决定后续流程(纠正检索 / 走 web 搜索 / 直接生成)",
], size=12, first_flag=False)
table(s, M, 4.55, 6.3, [
    ["档位", "含义", "动作"],
    ["Correct", "相关且可靠", "直接用于生成"],
    ["Incorrect", "不相关", "丢弃,改写 query 重检索或转向外部搜索"],
    ["Ambiguous", "部分相关", "两者结合:知识库 + 外部搜索融合"],
], col_w=[0.9, 1.1, 2.0], size=10.5, row_h=0.34, head_h=0.36)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:CRAG 流程框图", "建议画:Query → Retriever → Evaluator(打分)→ 三分支(Correct 精炼 / Incorrect 弃用+改写 / Ambiguous 融合)→ Generator;可加 web search 作为备选知识源",
            lines=6)
notes(s, "CRAG 与 Self-RAG 的差别值得点一句:CRAG 是「外部评估器 + 动作分档」的工程化方案,"
         "不要求重新训练模型;Self-RAG 是「训练模型自己吐 reflection token」的方案,"
         "效果更强但要重训。\n"
         "实践建议:先上 CRAG 式的规则/评估器方案,验证有效再考虑 Self-RAG。")

# ---------------------------------------------------------------- 29 warp-up
s = blank(prs)
base(s, "Wrap-up", 29, "小结与落地建议",
     "从「换模型」到「加回路」")
rect(s, M, 2.12, 6.4, 2.4, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.32, 5.9, 2.05)
para(tf, "Part 1 一句话总结", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "BERT 给了双向理解,但没有句子级目标 → SBERT 用 Siamese + 对比学习补上",
    "MTEB 让模型可比,BGE 让中文可用,Decoder-only 路线(NV-Embed / Qwen3)让 LLM 直接变成检索器",
    "选型顺序:先看榜单找候选 → 再用自己的业务数据做小评测 → 最后看维度/延迟/成本",
], size=12.5, first_flag=False)
rect(s, M + 6.8, 2.12, SW - 2 * M - 6.8, 2.4, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 7.06, 2.32, SW - 2 * M - 7.06, 2.05)
para(tf, "Part 2 一句话总结", first=True, size=14, bold=True, color=ACCENT2, space_after=8)
bullets(tf, [
    "Native RAG 的瓶颈在「没有反馈」:切分固定、检索单路、召错就编",
    "Agentic RAG = 在流水线各环节插入决策与修正:改问题 / 改证据 / 改答案 / 改流程",
    "技巧不是越多越好,先量测瓶颈在哪一环,再对症下药",
], size=12.5, first_flag=False)
rect(s, M, 4.67, SW - 2 * M, 1.78, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 4.86, SW - 2 * M - 0.52, 1.45)
para(tf, "Q&A / 讨论占位", first=True, size=13, bold=True, color=ACCENT, space_after=6)
para(tf, "· 我们自己的业务场景里,最想先试哪一个技巧?为什么?", size=11.5, color=BODY, space_after=3)
para(tf, "· 如果只能在「换更好的 embedding」和「加一个 reranker」之间选一个,你选哪个?为什么?", size=11.5, color=BODY, space_after=3)
para(tf, "· 讨论:GraphRAG 的索引成本在你的数据量下能接受吗?", size=11.5, color=BODY, space_after=0)
notes(s, "收尾页。留出 5~10 分钟 Q&A。可以准备一个「现场演示」作为彩蛋:"
         "同一个问题在 Native RAG 与加了 rerank + multi-query 后答案的对比。")

# ---------------------------------------------------------------- 30 orchestration frameworks
s = blank(prs)
base(s, "Part 2 · Agentic RAG", 30, "编排框架对比:LangGraph、LlamaIndex 与其他主流选择",
     "生产里的「Agentic RAG」很大程度上是框架选择:谁来决定下一步,状态存在哪")
table(s, M, 2.14, SW - 2 * M, [
    ["框架", "范式", "核心机制", "适合场景"],
    ["纯 Python", "手写控制流",
     "一步一个函数,循环和状态自己管",
     "先做诚实基线,不要一上来就引入依赖"],
    ["LangGraph", "图 / 状态机",
     "节点 + 边 + 共享的类型化 State;checkpointer 持久化每一步;原生 interrupt 支持 human-in-the-loop",
     "控制流需要显式、可审计 —— Agentic RAG 的默认选择"],
    ["LlamaIndex Workflows", "事件驱动步骤",
     "Step 订阅并发出事件(async);retriever / index / parser / query engine 一等公民",
     "以检索和索引为核心的流水线,需要自定义控制流"],
    ["CrewAI", "角色多智能体",
     "Agent 带角色、目标、工具;sequential 或 hierarchical 流程派发任务",
     "多角色协作(调研 → 分析 → 撰写),原型速度最快"],
    ["AutoGen / AG2", "对话式多智能体",
     "Agent 之间对话,group chat 由 manager 调度;带代码执行沙箱",
     "研究型多智能体对话、需要执行代码的场景"],
    ["Pydantic AI", "类型安全 Agent",
     "Pydantic 校验结构化输出与依赖注入;配 FastAPI 很自然",
     "要求 schema 保证、不想要魔法抽象"],
    ["Semantic Kernel", "Plugin / Planner",
     "Skill 与 plugin 交由 planner 编排;.NET 与企业集成强",
     "企业技术栈,尤其是 .NET 与已有 connector"],
    ["DSPy", "声明式优化",
     "声明 module 与 signature,给一个 metric,compile 出 prompt —— 是优化而不是编排",
     "流程稳定后,继续打磨 prompt / few-shot 质量"],
], col_w=[1.45, 1.25, 2.75, 2.45], size=10, row_h=0.47, head_h=0.34)
_, tf = txbox(s, M, 6.55, SW - 2 * M, 0.7)
para(tf, "怎么选:1) 只需要一次检索 + 一次 LLM 调用,就别上框架;2) 需要分支、重试、人工审批 → 图(LangGraph);"
         "3) 难点在检索与索引 → LlamaIndex;4) 有多个明确角色 → CrewAI 或 AutoGen;"
         "5) 瓶颈是 prompt 质量 → DSPy;6) 无论选谁,一定要有 tracing、checkpointing 和可回放的状态。",
     first=True, size=11, bold=True, color=INK, space_after=0, line=1.18)
notes(s, "定位:这一页放在小结之后,因为它回答的是全场最自然的「下一句」提问 —— "
         "「这些听着工程量不小,那我们用什么搭?」\n"
         "开场先给框架,别急着念名字:这些框架都只做两件事 —— 决定下一步做什么,以及把状态放在哪。"
         "差异在三处:(a) 显式图 vs. 由 Agent 行为涌现;(b) 检索/索引层有多成熟;"
         "(c) 跑挂之后 trace 是否可观测、可回放。\n"
         "给听众的工程建议:先用纯 Python 把 Agentic RAG 原型跑起来,感受到痛点了再引入 LangGraph(管流程)"
         "或 LlamaIndex(管检索)。框架锁定是真的成本 —— trace、checkpoint 和 state schema 比 star 数重要得多。\n"
         "不要在会场上演变成框架之争:如果有人问「哪个最好」,用底部那条决策线回答,而不是给排名。")

# ---------------------------------------------------------------- 31 backup divider
s = blank(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x22, 0x24, 0x2A))
_, tf = txbox(s, M + 0.2, 3.1, SW - 2 * M, 1.3)
para(tf, "BACKUP", first=True, size=15, bold=True,
     color=RGBColor(0x9A, 0xA4, 0xB2), space_after=8)
para(tf, "备用材料:FLARE 与 SELF-RAG", size=36, bold=True,
     color=RGBColor(0xFF, 0xFF, 0xFF), space_after=8)
para(tf, "时间富余或有人问到「检索时机怎么判断」时展开", size=13,
     color=RGBColor(0xB8, 0xBE, 0xC8), space_after=0)
notes(s, "备用页。如果前面超时,这两页不讲;如果有人追问 Adaptive RAG 里的「检索触发信号」,"
         "就直接跳到这两页。")

# ---------------------------------------------------------------- 31 FLARE
s = blank(prs)
base(s, "Backup", 31, "FLARE:用 LLM 自己的置信度决定要不要检索",
     "Forward-Looking Active REtrieval")
rect(s, M, 2.12, 6.3, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "核心思想", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "不再「先生成完整答案再检查」,而是「边生成边判断」",
    "生成到某一步时,如果模型对下一个内容没把握(low confidence),就触发一次检索",
    "检索后再继续生成 —— 检索时机由模型的不确定性驱动,而不是由固定的流程决定",
], size=12.5, first_flag=False)
rect(s, M, 4.6, 6.3, 1.85, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 4.78, 5.8, 1.5)
para(tf, "两种触发方式", first=True, size=13, bold=True, color=ACCENT, space_after=6)
rich(tf, [("Direct:", {"bold": True, "color": INK}),
          ("生成下一句之前先看它的 confidence score(概率),如果偏低,就拿「这句话本身」去检索。",
           {"color": BODY})], size=11.5, space_after=5, line=1.18)
rich(tf, [("Instruct:", {"bold": True, "color": INK}),
          ("根据当前已生成的句子,另外用一个 LLM 生成「用来 search 的 query」,再检索,把结果并回来继续生成。",
           {"color": BODY})], size=11.5, space_after=0, line=1.18)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:FLARE 生成-检索交错流程图", "建议画:生成 … → confidence 低 → 构造 query → 检索 → 回填上下文 → 继续生成;标注 Direct / Instruct 两条分支",
            lines=6)
notes(s, "FLARE 的价值在于「按需检索」:一次回答里可能只触发 1~2 次检索,而不是每一步都查。\n"
         "工程注意:要用 token 概率来算置信度,低概率 token 也可以作为「该查了」的信号;"
         "另外频繁触发检索会让延迟不可控,需要设阈值与次数上限。")

# ---------------------------------------------------------------- 32 SELF-RAG
s = blank(prs)
base(s, "Backup", 32, "SELF-RAG:训练模型自己做 critique",
     "Self-Reflective Retrieval-Augmented Generation")
rect(s, M, 2.12, 6.3, 4.33, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "和 FLARE / CRAG 的区别", first=True, size=14, bold=True,
     color=ACCENT, space_after=8)
bullets(tf, [
    "FLARE 靠模型输出概率来触发检索;CRAG 外挂一个评估器打分",
    "SELF-RAG 走第三条路:专门训练一个模型,让它在生成过程中自己产出 reflection token",
], size=12.5, first_flag=False)
para(tf, "四类 reflection token", size=14, bold=True, color=ACCENT, space_after=7)
bullets(tf, [
    "Retrieve:当前需不需要检索(触发 retrieval trigger)",
    "IsRel:检索到的段落相不相关",
    "IsSup:生成的这句话有没有被证据支撑",
    "IsUse:整体回答对用户有没有用",
], size=12.5, first_flag=False)
para(tf, "训练方式", size=14, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "用 GPT-4 给语料打上 reflection token 的标注,再做有监督微调",
    "推理时可以用这些 token 的权重做加权解码,把「有证据支撑」的候选提到前面",
], size=12.5, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 4.33,
            "占位:reflection token 插入位置示意", "建议画:生成序列中交替出现 [Retrieve] / 段落 / [IsRel] / 句子 / [IsSup] / [IsUse];并对比「开/关检索」两种输出",
            lines=6)
notes(s, "这是最「重」的方案:需要重训模型 + 用强模型造标注,成本高。\n"
         "但它把「自我纠正」从外挂流程变成了模型的内在能力,推理时不再需要多次额外 LLM 调用。\n"
         "现场结论:除非有明确的长期投入,否则先上 CRAG 式的外挂方案验证价值。")

# ---------------------------------------------------------------- 33 thanks
s = blank(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x12, 0x2A, 0x48))
_, tf = txbox(s, M + 0.2, 2.85, SW - 2 * M, 1.8)
para(tf, "Thanks & Q&A", first=True, size=40, bold=True,
     color=RGBColor(0xFF, 0xFF, 0xFF), space_after=10)
para(tf, "参考资料:SBERT (Reimers & Gurevych, 2019) · MTEB (Muennighoff et al.) · "
         "BGE / C-MTEB (BAAI) · NV-Embed (NVIDIA) · Qwen3-Embedding (Alibaba) · "
         "RAPTOR · GraphRAG (Microsoft) · CRAG · FLARE · SELF-RAG",
     size=11.5, color=RGBColor(0x9F, 0xB8, 0xD4), space_after=0, line=1.3)
rect(s, M + 0.2, 5.05, SW - 2 * M - 0.4, 1.0, fill=RGBColor(0x1B, 0x3A, 0x5E), radius=0.05)
_, tf = txbox(s, M + 0.45, 5.25, SW - 2 * M - 0.9, 0.7)
para(tf, "占位:联系方式 / 内部分享材料链接 / 代码仓库地址", first=True, size=12,
     bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), space_after=0)
notes(s, "致谢页。提醒自己把「参考资料出处」念一遍,尤其是榜单数据要说明评测时间。")

prs.save(OUT)
print("saved:", OUT)
print("slides:", len(prs.slides.__iter__.__self__._sldIdLst))
