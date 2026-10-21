# -*- coding: utf-8 -*-
"""
Build the ENGLISH share-out deck: "From Embedding Models to Agentic RAG"
Source outline: RAG and Agentic AI.md (same content as the .docx)
Style: clean academic, white background, dark titles, large placeholder areas for the author to fill in.
All slide content is English (US corporate setting).
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

import os
OUT = os.environ.get(
    "DECK_OUT",
    r"C:\Users\renha\workspace\projects\Learning\Embedding and RAG\Embedding and Agentic RAG (EN).pptx")

# ---------------------------------------------------------------- palette
INK        = RGBColor(0x1B, 0x25, 0x33)
BODY       = RGBColor(0x33, 0x40, 0x50)
MUTED      = RGBColor(0x7A, 0x86, 0x96)
ACCENT     = RGBColor(0x1F, 0x5C, 0xA8)
ACCENT2    = RGBColor(0xC0, 0x50, 0x2A)
RULE       = RGBColor(0xD8, 0xDE, 0xE6)
PLACE_BG   = RGBColor(0xF7, 0xF9, 0xFC)
PLACE_LINE = RGBColor(0xBB, 0xC7, 0xD6)
CHIP_BG    = RGBColor(0xEC, 0xF2, 0xFA)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Segoe UI"
MONO = "Consolas"

SW, SH = 13.333, 7.5
M      = 0.62


# ---------------------------------------------------------------- helpers
def txbox(slide, x, y, w, h, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf


def style(p, size=14, bold=False, color=BODY, font=FONT, italic=False,
          space_after=6, align=PP_ALIGN.LEFT, line=1.2):
    p.alignment = align
    p.space_after = Pt(space_after)
    p.line_spacing = line
    for r in p.runs:
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = font
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
    rect(slide, 0, 0, SW, 0.085, fill=ACCENT)
    if section:
        _, tf = txbox(slide, M, 0.34, 9.4, 0.3)
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
    rect(slide, x, y, w, h, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
    _, tf = txbox(slide, x + 0.22, y + 0.18, w - 0.44, h - 0.36)
    para(tf, label, first=True, size=13, bold=True, color=ACCENT, space_after=4)
    para(tf, hint, size=11.5, color=MUTED, line=1.2, space_after=0)
    for _ in range(lines):
        p = tf.add_paragraph()
        p.text = "\u00b7 "
        style(p, size=11.5, color=PLACE_LINE, space_after=2)


def bullets(tf, items, size=14, first_flag=True):
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
        p.text = ("\u2013 " if lvl == 1 else "") + ("" if lvl else "\u25aa  ") + text
        style(p, size=ov.get("size", size if lvl == 0 else size - 1.5),
              bold=ov.get("bold", False),
              color=ov.get("color", BODY if lvl == 0 else MUTED),
              space_after=ov.get("space_after", 7 if lvl == 0 else 5),
              line=1.15)
    return tf


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def table(slide, x, y, w, rows, col_w=None, header=True, size=12,
          row_h=0.34, head_h=0.38):
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
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if header and ri == 0:
                cell.fill.fore_color.rgb = ACCENT
            else:
                cell.fill.fore_color.rgb = WHITE if ri % 2 else RGBColor(0xF4, 0xF7, 0xFB)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = val
            style(p, size=size, bold=(header and ri == 0),
                  color=WHITE if (header and ri == 0) else BODY,
                  space_after=0, line=1.08)
    return tbl


def timeline(slide, x, y, w, items, h=0.62):
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
para(tf, "TECH SHARING  \u00b7  LLM RETRIEVAL STACK", first=True, size=12,
     bold=True, color=RGBColor(0x8F, 0xB8, 0xE8), space_after=0)
_, tf = txbox(s, M + 0.1, 1.08, SW - 2 * M - 0.4, 1.3)
para(tf, "From Embedding Models to Agentic RAG", first=True, size=40,
     bold=True, color=WHITE, space_after=0)
para(tf, "BERT \u2192 SBERT \u2192 MTEB \u2192 BGE \u2192 NV-Embed \u2192 Qwen3-Embedding, "
         "and from Native RAG to Agentic RAG",
     size=14, color=RGBColor(0xB8, 0xCD, 0xE6), space_after=0)
_, tf = txbox(s, M + 0.1, 3.0, SW - 2 * M, 1.5)
rich(tf, [("Presenter: ", {"bold": True, "color": INK}), ("______________     ", {}),
          ("Date: ", {"bold": True, "color": INK}), ("______________", {})],
     first=True, size=13, space_after=10)
rich(tf, [("Duration: ", {"bold": True, "color": INK}),
          ("~60 min (35 min embeddings + 25 min agentic RAG)      ", {}),
          ("Format: ", {"bold": True, "color": INK}), ("in-person / virtual", {})],
     size=13, space_after=10)
rich(tf, [("Companion material: ", {"bold": True, "color": INK}),
          ("RAG and Agentic AI.md (original outline)", {})], size=13, space_after=0)
rect(s, M + 0.1, 5.0, SW - 2 * M - 0.2, 1.5, fill=PLACE_BG, line=PLACE_LINE,
     lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.34, 5.18, SW - 2 * M - 0.68, 1.2)
para(tf, "Opening note placeholder (~30 sec)", first=True, size=12.5, bold=True,
     color=ACCENT, space_after=5)
para(tf, "Why this topic: retrieval quality sets the ceiling for any RAG system, and embeddings set retrieval quality.",
     size=11.5, color=BODY, space_after=3)
para(tf, "Goal: by the end you can answer \u201cwhich embedding model should we pick?\u201d and "
         "\u201cwhere is our RAG pipeline leaking, and which agentic technique fixes it?\u201d",
     size=11.5, color=BODY, space_after=0)
notes(s, "Cover slide. Frame the two storylines up front: (1) embeddings = representation quality, "
         "(2) RAG architecture = control flow. They meet in the middle: better embeddings make retrieval "
         "more accurate, smarter agents make retrieval better timed, better phrased, and self-correcting.")

# ---------------------------------------------------------------- 2 agenda
s = blank(prs)
base(s, "Agenda", 2, "What we will cover", "Two storylines: representation quality (embeddings) x control flow (RAG)")
colw = (SW - 2 * M - 0.3) / 2
rect(s, M, 2.15, colw, 4.3, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.28, 2.38, colw - 0.56, 3.9)
para(tf, "PART 1  \u00b7  The embedding model evolution", first=True, size=17,
     bold=True, color=ACCENT, space_after=12)
bullets(tf, [
    "BERT \u2014 the encoder-only starting point: bidirectional MLM + NSP",
    "SBERT \u2014 Siamese networks solve sentence-level similarity",
    "MTEB \u2014 one benchmark to compare every model",
    "BGE \u2014 strong on Chinese: RetroMAE + contrastive + fine-tuning",
    "NV-Embed \u2014 decoder-only LLM + latent attention",
    "Qwen3-Embedding \u2014 one base model for embedder and reranker",
], size=13, first_flag=False)
rect(s, M + colw + 0.3, 2.15, colw, 4.3, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + colw + 0.58, 2.38, colw - 0.56, 3.9)
para(tf, "PART 2  \u00b7  From RAG to Agentic RAG", first=True, size=17,
     bold=True, color=ACCENT2, space_after=12)
bullets(tf, [
    "The two structural weaknesses of Native RAG",
    "ReAct RAG \u2192 Agentic RAG: what actually changes",
    "Query side: Query Rewrite / Multi-Query",
    "Knowledge base side: Graph RAG / RAPTOR",
    "Post-retrieval side: Reranker / Self-correction / Corrective RAG",
    "Orchestration: Adaptive RAG (plus backup: FLARE / SELF-RAG)",
], size=13, first_flag=False)
notes(s, "Agenda slide. The two halves have different motivations: Part 1 is \u201chow did the models get better\u201d, "
         "Part 2 is \u201chow did the system get smarter\u201d. Worth saying out loud: in production the bottleneck is "
         "usually not model quality, it is that the system does not do the right thing at the right moment.")

# ---------------------------------------------------------------- 3 divider
s = blank(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x12, 0x2A, 0x48))
rect(s, 0, 3.05, 2.2, 0.09, fill=ACCENT)
_, tf = txbox(s, M + 0.2, 3.35, SW - 2 * M, 1.4)
para(tf, "PART 1", first=True, size=15, bold=True,
     color=RGBColor(0x8F, 0xB8, 0xE8), space_after=8)
para(tf, "The Embedding Model Evolution", size=36, bold=True, color=WHITE, space_after=8)
para(tf, "BERT \u2192 SBERT \u2192 MTEB \u2192 BGE \u2192 NV-Embed \u2192 Qwen3-Embedding",
     size=14, color=RGBColor(0xB8, 0xCD, 0xE6), space_after=0)
notes(s, "Divider. One-line framing: the goal of an embedding model has never changed \u2014 compress meaning "
         "into a vector space where distance is meaningful. What changed is the training objective, the scale, "
         "and how we measure it.")

# ---------------------------------------------------------------- 4 BERT intro
s = blank(prs)
timeline(s, M, 2.12, SW - 2 * M, [
    ("BERT", "Encoder-only"), ("SBERT", "Sentence-level"),
    ("MTEB", "Benchmark"), ("BGE", "3-stage training"),
    ("NV-Embed", "Decoder-only"), ("Qwen3-Emb", "Embed + Rerank"),
], h=0.72)
base(s, "Part 1 \u00b7 Embedding Model: BERT", 4, "BERT: the encoder-only starting point",
     "2018 \u00b7 the \u201cpretrain then fine-tune\u201d paradigm for NLP")
rect(s, M, 3.05, 6.5, 3.4, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 3.26, 6.0, 3.0)
para(tf, "Four key design choices", first=True, size=15, bold=True,
     color=ACCENT, space_after=9)
bullets(tf, [
    "Encoder-only Transformer: a stack of encoders, no autoregressive decoder",
    "Bidirectional: full self-attention, every token sees both left and right context",
    "Masked Language Model (MLM): mask 15% of tokens and predict them",
    "Output is one vector per token, not one vector per sentence",
], size=13.5, first_flag=False)
placeholder(s, M + 6.9, 3.05, SW - 2 * M - 6.9, 3.4,
            "Placeholder: architecture / scale",
            "Suggested: BERT-base vs BERT-large spec table (L=12/24, H=768/1024, A=12/16, ~110M/340M params)",
            lines=4)
notes(s, "Plant the seed here: MLM is a token-level objective, so BERT natively returns token vectors. "
         "Getting a sentence embedding requires an extra step (CLS or mean pooling) \u2014 which is exactly the "
         "problem SBERT later solves. Also worth mentioning the input representation: "
         "token + segment + position embeddings, summed.")

# ---------------------------------------------------------------- 5 BERT pretraining
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: BERT", 5, "How BERT is pretrained: two self-supervised tasks",
     "No labels needed \u2014 the text supervises itself")
rect(s, M, 2.12, 6.5, 5.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 6.0, 3.9)
para(tf, "1. Bidirectional MLM", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Pick 15% of tokens; of those, 80% become [MASK], 10% become a random word, 10% stay unchanged",
    "Why not mask 100%: it avoids a train/inference mismatch in the input distribution",
    "Why bidirectional: seeing both sides beats left-only (GPT-style) for understanding tasks",
], size=12.5, first_flag=False)
para(tf, "2. Next Sentence Prediction (NSP)", size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Feed two segments A and B; predict whether B truly follows A (50% positive)",
    "Purpose: teach sentence-to-sentence relationships, useful for QA and NLI",
    "Later work (RoBERTa) found NSP weak; Sentence-Order Prediction (SOP) works better",
], size=12.5, first_flag=False)
placeholder(s, M + 6.9, 2.12, SW - 2 * M - 6.9, 5.2,
            "Placeholder: MLM masking example / pretraining flow",
            "Suggested: input string \u2192 masked positions \u2192 Encoder \u2192 loss computed only on masked positions",
            lines=6)
notes(s, "Spend a moment on why these two tasks teach semantics. MLM learns distributed lexical, syntactic "
         "and world knowledge; NSP learns discourse coherence. Then name the limitation: NSP gives sentence-level "
         "supervision, but semantic similarity itself is never optimized directly \u2014 that is the motivation for SBERT.")

# ---------------------------------------------------------------- 6 BERT finetune
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: BERT", 6, "How BERT is fine-tuned: labeled downstream data",
     "Including question-answer pairs")
rect(s, M, 2.12, 6.6, 2.22, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.30, 6.1, 1.9)
para(tf, "Standard recipe", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Pretrained weights + task head (classification / span extraction) \u2192 end-to-end fine-tuning",
    "QA (SQuAD style): input [CLS] Question [SEP] Passage [SEP]; output start / end positions of the answer span",
    "Objective: cross-entropy; small LR (2e-5 to 5e-5) converges in 3-4 epochs",
], size=13, first_flag=False)
rect(s, M, 4.44, 6.6, 2.88, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 4.60, 6.1, 2.6)
para(tf, "The catch: BERT is not an off-the-shelf embedding model", size=14,
     bold=True, color=ACCENT2, space_after=7)
bullets(tf, [
    "BERT was never trained on sentence similarity, so CLS vectors are anisotropic and cosine similarity barely discriminates",
    "Common mitigations: mean pooling, or whitening / normalizing the vectors",
    "Feeding pairs jointly (a Cross-Encoder) works well but is O(n\u00b2) at inference \u2014 not viable for large-scale retrieval",
], size=12, first_flag=False)
placeholder(s, M + 7.0, 2.12, SW - 2 * M - 7.0, 5.2,
            "Placeholder: fine-tuning data sample / code",
            "Suggested: a question-answer pair example (question, passage, label) plus 3-5 lines of HuggingFace fine-tuning code",
            lines=6)
notes(s, "This is the key transition. Fine-tuning BERT makes a great classifier but a poor embedding model: "
         "the Sentence-BERT paper reports STS Spearman around 0.5x for vanilla BERT CLS/mean embeddings. "
         "That gap is exactly what SBERT closes.")

# ---------------------------------------------------------------- 7 SBERT why
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: SBERT", 7, "Why BERT is bad at sentence-level similarity",
     "The training objective simply does not match the task")
rect(s, M, 2.12, 6.2, 2.22, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 2.30, 5.7, 1.9)
para(tf, "Root cause: a mismatch of objectives", first=True, size=14, bold=True,
     color=ACCENT2, space_after=6)
para(tf, "MLM and NSP optimize token-level and sentence-level coherence. Neither ever optimizes "
         "\u201chow close in meaning are these two sentences\u201d, so nothing pushes similar meanings "
         "to nearby vectors.", size=12, color=BODY, space_after=0, line=1.2)
table(s, M, 4.52, 6.2, [
    ["", "Cross-Encoder", "Bi-Encoder (SBERT)"],
    ["Input", "[Q][SEP][D] encoded together", "Q and D encoded separately"],
    ["Interaction", "Token-level cross-attention", "Only at the final similarity step"],
    ["Accuracy", "Higher on STS", "Slightly lower, very close"],
    ["Speed", "n documents = n forward passes", "Doc vectors precomputed offline"],
    ["10k sentences", "~65 hours", "~5 seconds (~46,000x faster)"],
    ["Best for", "Reranking a small candidate set", "Recall / large-scale retrieval"],
], col_w=[1.0, 1.6, 1.7], size=10.5, row_h=0.32, head_h=0.36)
placeholder(s, M + 6.6, 2.12, SW - 2 * M - 6.6, 5.2,
            "Placeholder: architecture comparison",
            "Suggested: top = Cross-Encoder (Q and D concatenated \u2192 one score); bottom = Bi-Encoder (Q \u2192 u, D \u2192 v \u2192 cosine)",
            lines=6)
notes(s, "This is the technical heart of Part 1. The real distinction is WHERE the interaction happens. "
         "A Cross-Encoder interacts in the middle layers, so it is accurate but not precomputable. A Bi-Encoder "
         "defers all interaction to a final dot product: slightly less accurate, but vectors can be precomputed "
         "and indexed with ANN. The 65 hours vs 5 seconds figures come from the SBERT paper (10k sentences).")

# ---------------------------------------------------------------- 8 SBERT siamese
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: SBERT", 8, "The Siamese network and its training objectives",
     "Shared weights, pooled vectors, and three loss families")
rect(s, M, 2.12, 6.5, 2.5, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 6.0, 2.1)
para(tf, "Network structure", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Two towers with shared weights (Siamese): the same encoder embeds sentence A and sentence B",
    "Pooling layer: mean or max (mean by default) compresses token vectors into one fixed-length sentence vector; CLS also works",
    "Default output is 768-dim, optionally projected down to 256/512",
], size=13, first_flag=False)
rect(s, M, 4.82, 6.5, 2.5, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 5.00, 6.0, 2.2)
para(tf, "Engineering notes", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Large batches matter: the original batch size of 16 was too small, SBERT later trained with 1024",
    "In-batch negatives are free: every other example in the batch is a natural negative",
], size=13, first_flag=False)
placeholder(s, M + 6.9, 2.12, SW - 2 * M - 6.9, 5.2,
            "Placeholder: Siamese diagram + the three losses",
            "Suggested: left tower BERT\u2192pooling\u2192u, right tower (shared weights)\u2192pooling\u2192v, similarity in the middle; below, write the cosine / MSE / triplet objectives",
            lines=6)
notes(s, "Explain when each objective applies:\n"
         "1) Classification objective: concatenate u, v and |u-v|, then softmax \u2014 ideal when NLI-style labels exist "
         "(SNLI + MNLI give ~1M pairs).\n"
         "2) Regression objective: MSE directly on the cosine score \u2014 ideal when you have 1-5 similarity ratings.\n"
         "3) Triplet objective: max(||s_a - s_p|| - ||s_a - s_n|| + margin, 0) with anchor / positive / negative.\n"
         "Also stress that in-batch negatives are what make a large batch pay off.")

# ---------------------------------------------------------------- 9 SBERT perf
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: SBERT", 9, "Results: from roughly 0.5 to roughly 0.8",
     "Same BERT weights, a different objective")
table(s, M, 2.15, 7.4, [
    ["Approach", "STS performance", "Notes"],
    ["Avg. GloVe embeddings", "Spearman \u2248 0.52", "Averaged word vectors; weak baseline"],
    ["BERT CLS / mean vectors", "\u2248 0.51 - 0.60", "Never trained for similarity"],
    ["SBERT (base)", "\u2248 0.76", "Siamese + NLI fine-tuning"],
    ["SBERT (roberta-large)", "\u2248 0.79", "Bigger backbone, longer training"],
], col_w=[1.5, 1.1, 1.4], size=11.5, row_h=0.38, head_h=0.4)
_, tf = txbox(s, M, 5.05, 7.4, 1.4)
para(tf, "Takeaway", first=True, size=14, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "Same BERT weights, different training objective \u2014 sentence similarity jumps a full tier",
    "And it keeps the engineering property that made RAG possible: precomputed vectors + ANN search",
], size=12.5, first_flag=False)
placeholder(s, M + 7.8, 2.15, SW - 2 * M - 7.8, 4.3,
            "Placeholder: per-dataset STS bar chart",
            "Source: Sentence-BERT paper (Reimers & Gurevych, 2019), Table 2 / official docs; the original figure works well here",
            lines=6)
notes(s, "Always cite the source; quote magnitudes rather than exact numbers because the figures move with settings. "
         "The real lesson: how strong an embedding model is depends heavily on the training objective "
         "(loss + how pairs and negatives are constructed), not only on a bigger backbone.")

# ---------------------------------------------------------------- 10 MTEB
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: MTEB", 10, "MTEB: one benchmark for every embedding model",
     "Massive Text Embedding Benchmark \u2014 making models comparable")
rect(s, M, 2.12, 6.3, 5.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "What it is", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "An open evaluation suite for text embeddings (MTEB paper, 2022)",
    "Eight task types: Retrieval, Reranking, Clustering, Classification, Pair Classification, STS, Summarization, Bitext Mining",
    "Started at ~56 datasets across 112 languages; has since grown to well over a hundred datasets",
    "Comes with C-MTEB (Chinese) and MMTEB (multilingual), plus a public leaderboard",
], size=13, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: task taxonomy + leaderboard snapshot",
            "Suggested: the 8 task types, plus the current top entries on the Multilingual and Chinese leaderboards",
            lines=6)
notes(s, "Why it matters: before MTEB, every model reported numbers on its own one or two datasets, so nothing "
         "was comparable. MTEB's value is a shared yardstick. Two common misuses to call out:\n"
         "1) A high leaderboard score does not mean it works on your data \u2014 always run a small eval on your own corpus;\n"
         "2) Know whether you are reading the Retrieval board or the STS board \u2014 Retrieval is what maps to RAG.")

# ---------------------------------------------------------------- 11 BGE C-MTEB
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: BGE", 11, "BGE: the strong option for Chinese",
     "BAAI General Embedding \u2014 starting from C-MTEB")
rect(s, M, 2.12, 6.3, 5.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "C-MTEB (Chinese MTEB)", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Chinese evaluation suite built by the BGE team: 6 task types, ~35 datasets covering retrieval, reranking, clustering, classification, STS, pair classification",
    "Datasets are re-split and standardized so Chinese models are finally comparable",
    "The BGE family spans English (bge-large-en), Chinese (bge-large-zh), multilingual (bge-m3), plus small/base sizes",
], size=13, first_flag=False)
para(tf, "Why Chinese needs its own board", size=14, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "Tokenization and semantic distribution differ; a top English model can drop noticeably on Chinese",
    "Chinese retrieval often means short queries against long documents with many proper nouns \u2014 the eval set should reflect that",
], size=13, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: C-MTEB scores / model spec table",
            "Suggested: bge-large-zh-v1.5 average and per-task C-MTEB scores, plus bge-m3's dense / sparse / ColBERT outputs",
            lines=6)
notes(s, "Worth mentioning bge-m3's three selling points: multilingual (100+ languages), multi-granularity "
         "(up to 8192 tokens), and multi-vector output (dense + sparse + ColBERT-style multi-vector) so one "
         "encoding supports three retrieval modes. In practice BGE is the most common Chinese embedding choice, "
         "and if your audience works in English only, say so and pivot quickly to the training pipeline.")

# ---------------------------------------------------------------- 12 BGE stage1
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: BGE", 12, "Three-stage training (1/3): RetroMAE pretraining",
     "Stage 1 \u2014 build a strong language and semantics foundation first")
rect(s, M, 2.12, 6.4, 5.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.30, 5.9, 4.5)
para(tf, "What RetroMAE is", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "A masked-autoencoder pretraining method designed with retrieval in mind",
    "Encoder side: heavily mask the input (e.g. 15-30%) and encode a sentence representation",
    "Decoder side: a shallow, asymmetric decoder reconstructs the masked tokens from that single representation",
    "Key point: the decoder exists only for pretraining and is discarded at inference \u2014 zero serving cost",
], size=13, first_flag=False)
para(tf, "Why it works", size=14, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "Reconstruction forces the sentence vector to carry enough information to restore the whole input, so representations get richer",
    "Compared with plain MLM it is closer to the real goal \u2014 one vector representing a whole passage \u2014 and gives contrastive training a better starting point",
], size=12.5, first_flag=False)
placeholder(s, M + 6.8, 2.12, SW - 2 * M - 6.8, 5.2,
            "Placeholder: RetroMAE diagram",
            "Suggested: input \u2192 (heavy masking) \u2192 Encoder \u2192 sentence vector \u2192 asymmetric shallow decoder \u2192 reconstruct masked tokens; label the decoder as discarded at inference",
            lines=6)
notes(s, "Make the staging explicit: the three stages build on each other \u2014 pretrain the foundation, "
         "align the space with contrastive learning, then sharpen precision with supervised fine-tuning. "
         "RetroMAE is stage 1, teaching the backbone to compress and understand.")

# ---------------------------------------------------------------- 13 BGE stage2
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: BGE", 13, "Three-stage training (2/3): contrastive learning",
     "Stage 2 \u2014 align semantics with vector distance using large unlabeled pairs")
rect(s, M, 2.12, 6.4, 5.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.9, 3.9)
para(tf, "Training setup", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Objective: an InfoNCE-style contrastive loss (same family as SBERT, different scale and data construction)",
    "Data: large-scale weakly supervised pairs \u2014 title-body, QA pairs, paraphrases, community Q&A",
    "Temperature \u03c4 controls sharpness; large batches (1024+) guarantee enough in-batch negatives",
], size=13, first_flag=False)
para(tf, "Two engineering levers", size=14, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "In-batch negatives: documents from other examples in the batch serve as negatives at no extra forward-pass cost",
    "Hard negatives: documents that look relevant but are wrong give the biggest training signal, and must be mined deliberately",
], size=13, first_flag=False)
placeholder(s, M + 6.8, 2.12, SW - 2 * M - 6.8, 5.2,
            "Placeholder: contrastive setup + InfoNCE",
            "Suggested: anchor q pulled toward d+ and pushed from the batch negatives d-, next to -log(exp(sim(q,d+)/\u03c4) / \u03a3 exp(sim(q,d_i)/\u03c4))",
            lines=6)
notes(s, "The point of this slide is data construction, not the formula. Stress that the ceiling of contrastive "
         "learning is largely set by how real the positives are and how hard the negatives are. In-batch negatives "
         "are free; mined hard negatives cost effort but pay more.")

# ---------------------------------------------------------------- 14 BGE stage3 + perf
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: BGE", 14, "Three-stage training (3/3): supervised fine-tuning + results",
     "Stage 3 \u2014 high-quality labeled data pushes precision up again")
rect(s, M, 2.12, 6.3, 2.78, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.30, 5.8, 2.6)
para(tf, "Fine-tuning stage", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Data: high-quality (query, positive, negative) triples, human or semi-automatically labeled",
    "Negative mining: retrieve top-k near-misses with the current model, then keep the ones that look relevant but are wrong as hard negatives",
    "Same contrastive objective, but cleaner data and harder negatives \u2192 clear retrieval gains",
    "Optional instruction prefixes (e.g. \u201crepresent this sentence for retrieval\u201d) improve cross-task generalization",
], size=12.5, first_flag=False)
table(s, M, 5.06, 6.3, [
    ["Stage", "Goal", "Data"],
    ["1 RetroMAE", "Foundation representation", "Large unlabeled corpus"],
    ["2 Contrastive", "Semantic alignment", "Weakly supervised pairs"],
    ["3 Fine-tuning", "Retrieval precision", "Curated labels + hard negatives"],
], col_w=[1.2, 1.2, 1.6], size=10.5, row_h=0.33, head_h=0.36)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: BGE scores on C-MTEB / MTEB",
            "Suggested: bge-large-zh-v1.5 and bge-m3 averages, plus a comparison against OpenAI text-embedding models (cite source and evaluation date)",
            lines=6)
notes(s, "Tie the three stages into one story: first learn to compress and understand (RetroMAE), then learn "
         "that distance means meaning (contrastive), then sharpen the hard boundaries with labels (fine-tuning "
         "with hard negatives). Remind the audience that leaderboard numbers are time-sensitive \u2014 talk trends, "
         "not absolute scores.")

# ---------------------------------------------------------------- 15 NV-Embed
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: NV-Embed", 15, "NV-Embed: decoder-only models can embed too",
     "NVIDIA \u00b7 turning a plain LLM into an embedding model")
rect(s, M, 2.12, 6.3, 2.78, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.30, 5.8, 2.6)
para(tf, "Three key changes", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Decoder-only backbone: use Mistral-7B-class autoregressive LLMs as embedders with no architectural surgery",
    "Latent attention layer: a final attention layer aggregates variable-length token states into one fixed vector, replacing mean pooling and removing the causal mask so tokens see each other bidirectionally",
    "Two-stage contrastive training: stage 1 on large weakly supervised data mixing retrieval and non-retrieval tasks; stage 2 on high-quality curated data",
], size=12.5, first_flag=False)
rect(s, M, 5.02, 6.3, 2.30, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 5.16, 5.8, 2.0)
para(tf, "The counterintuitive part: stage 2 drops in-batch negatives", first=True,
     size=13, bold=True, color=ACCENT2, space_after=5)
para(tf, "Switching stage 2 to alternative negatives (other positives of the same query in the batch, "
         "or across-device negatives) worked better. When the backbone is strong and the data is clean, "
         "naive in-batch negatives inject false-negative noise.", size=11.5, color=BODY,
     space_after=0, line=1.18)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: latent attention diagram + MTEB scores",
            "Suggested: LLM hidden states \u2192 latent attention (a learned query vector doing cross-attention) \u2192 single embedding; next to it, the paper's MTEB average and task count",
            lines=6)
notes(s, "The big takeaway: an embedding model does not have to be the classic dual-encoder plus pooling shape. "
         "A general-purpose LLM with one aggregation layer can top the leaderboard. Also note the trend: the "
         "boundary between embedding models and LLMs is dissolving \u2014 Qwen3-Embedding takes the same route by "
         "unifying embedder and reranker on one base model.")

# ---------------------------------------------------------------- 16 Qwen3 intro
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: Qwen3-Embedding", 16, "Qwen3-Embedding: one backbone, two models",
     "Unifying embedding and reranking on a single LLM base")
rect(s, M, 2.12, 6.3, 5.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "Family and capabilities", first=True, size=15, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Sizes: 0.6B / 4B / 8B embedders, with matching reranker variants",
    "Initialized from the Qwen3 LLM family, inheriting multilingual and long-context ability",
    "Long context: 32k tokens; configurable embedding dimension (commonly 1024 / 2560 / 4096, trimmable lower)",
    "100+ natural languages plus programming languages, across retrieval, clustering, classification and bitext mining",
], size=13, first_flag=False)
para(tf, "Design idea: embedder and reranker share a base", size=14, bold=True,
     color=ACCENT, space_after=6)
bullets(tf, [
    "Same backbone, different output head and concatenation pattern for the two roles",
    "Benefit: training signal transfers between them, and production can reuse one serving stack",
], size=12.5, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: spec table + benchmark scores",
            "Suggested: dimension, layers and context length for 0.6B / 4B / 8B, plus MTEB Multilingual and MTEB Code scores",
            lines=6)
notes(s, "Continues the NV-Embed thread: LLM backbones as embedders. The practical appeal of Qwen3-Embedding "
         "is custom embedding dimensions, 32k context and broad language coverage, which makes it a popular "
         "self-hosted choice for multilingual and code retrieval.")

# ---------------------------------------------------------------- 17 Qwen3 mechanics
s = blank(prs)
base(s, "Part 1 \u00b7 Embedding Model: Qwen3-Embedding", 17, "Input format, reranker, and training loss",
     "EOS pooling for embeddings, a Yes/No token for reranking")
rect(s, M, 2.12, 6.2, 2.55, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 2.30, 5.7, 2.2)
para(tf, "Embedding model: input \u2192 EOS pooling", first=True, size=14,
     bold=True, color=ACCENT, space_after=6)
rich(tf, [("{Inst}{Q}<EOS>", {"font": MONO, "bold": True, "color": INK}),
          ("   \u2192   ", {"color": MUTED}),
          ("take the hidden state at <EOS> as the sentence vector", {"color": BODY})],
     size=12.5, space_after=5)
para(tf, "Pooling from the final token rather than mean or CLS matches the causal attention of a decoder-only "
         "model: the last position has already attended to the entire prefix, so it is a natural summary of the input.",
     size=11.5, color=BODY, space_after=0, line=1.18)
rect(s, M, 4.79, 6.2, 2.53, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 4.98, 5.7, 2.2)
para(tf, "Reranker: input \u2192 LM head \u2192 Yes / No", first=True, size=14,
     bold=True, color=ACCENT, space_after=6)
rich(tf, [("{Inst}{Q}{Doc}", {"font": MONO, "bold": True, "color": INK}),
          ("  \u2192  ", {"color": MUTED}),
          ("LM head", {"font": MONO, "color": INK}), ("  \u2192  ", {"color": MUTED}),
          ("compare the logits of \u201cYes\u201d and \u201cNo\u201d", {"color": BODY})],
     size=12.5, space_after=5)
para(tf, "Softmax over those two logits gives the relevance score. Relevance judgment becomes a binary "
         "classification problem, reusing the LLM's own language modeling head \u2014 no extra scoring head needed.",
     size=11.5, color=BODY, space_after=0, line=1.18)
rect(s, M + 6.6, 2.12, SW - 2 * M - 6.6, 5.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 6.86, 2.34, SW - 2 * M - 7.12, 3.9)
para(tf, "Training loss: InfoNCE-style contrastive loss", first=True, size=14,
     bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Pull each query toward its positive document and away from in-batch negatives",
    "The reranker uses a contrastive-style objective too, with Yes/No probability standing in for similarity",
    "Instruction prefixes shape the vector space for retrieval, classification or clustering",
], size=12.5, first_flag=False)
placeholder(s, M + 6.86, 4.62, SW - 2 * M - 7.12, 2.7,
            "Placeholder: formula and code",
            "Suggested: the InfoNCE formula, a get_detailed_instruct() usage sample, and 3 lines each calling embed / rerank",
            lines=2)
notes(s, "This closes Part 1. Compare the three pooling philosophies side by side:\n"
         "BERT uses CLS, SBERT uses mean pooling, Qwen3 uses the last token (EOS).\n"
         "The line to remember: pooling follows the attention structure of the backbone \u2014 bidirectional encoders "
         "suit mean pooling, causal decoders suit the final position.\n"
         "The Yes/No logit reranker is elegant; if you have a demo, show top-5 before and after reranking.")

# ---------------------------------------------------------------- 18 Part 2 divider
s = blank(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x3A, 0x1E, 0x14))
rect(s, 0, 3.05, 2.2, 0.09, fill=ACCENT2)
_, tf = txbox(s, M + 0.2, 3.35, SW - 2 * M, 1.4)
para(tf, "PART 2", first=True, size=15, bold=True,
     color=RGBColor(0xE8, 0xB8, 0x9A), space_after=8)
para(tf, "From RAG to Agentic RAG", size=36, bold=True, color=WHITE, space_after=8)
para(tf, "Once retrieval quality plateaus, the next win comes from control flow, not just the model",
     size=14, color=RGBColor(0xE6, 0xCD, 0xB8), space_after=0)
notes(s, "Divider. Bridge sentence: Part 1 was about making retrieval more accurate. Part 2 changes the angle \u2014 "
         "let the system decide when to retrieve, what to retrieve, and whether to trust what came back.")

# ---------------------------------------------------------------- 19 native RAG
s = blank(prs)
base(s, "Part 2 \u00b7 Agentic RAG", 19, "Two structural weaknesses of Native RAG",
     "Look at where the naive pipeline actually leaks")
_, tf = txbox(s, M, 2.12, SW - 2 * M, 0.9)
para(tf, "Naive RAG: user query \u2192 (embed) \u2192 vector store top-K \u2192 stuff into prompt \u2192 LLM generates",
     first=True, size=13, bold=True, color=INK, space_after=6)
para(tf, "Every step is one-directional, single-shot, and has no feedback \u2014 nothing ever looks back.",
     size=12, color=MUTED, space_after=0)
rect(s, M, 3.1, 6.3, 3.35, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 3.32, 5.8, 2.95)
para(tf, "Weakness 1: chunk size decides what information survives", first=True,
     size=14, bold=True, color=ACCENT2, space_after=8)
bullets(tf, [
    "Too small: paragraphs get cut in half, pronouns lose their antecedents (\u201cit\u201d, \u201cthat plan\u201d, \u201cthe above\u201d), answers come out incomplete",
    "Too large: one chunk mixes several topics, so its vector is an average of everything and recall gets noisier",
    "Fixed-length splitting ignores document structure (heading hierarchy, tables, code blocks)",
    "Chunk size also drives cost: the larger the chunk, the more expensive top-K is to fit into the prompt",
], size=12.5, first_flag=False)
rect(s, M + 6.7, 3.1, SW - 2 * M - 6.7, 3.35, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 6.96, 3.32, SW - 2 * M - 6.96, 2.95)
para(tf, "Weakness 2: semantic search decides which documents you get", first=True,
     size=14, bold=True, color=ACCENT2, space_after=8)
bullets(tf, [
    "\u201cSemantically similar\u201d is not the same as \u201ccontains the answer\u201d \u2014 a follow-up question and its answer passage rarely share wording",
    "A single vector cannot represent a multi-intent query (e.g. \u201ccompare the pricing of A and B\u201d)",
    "Embeddings are naturally weaker than BM25 / keyword search on exact matches: model numbers, IDs, proper nouns",
    "If top-K is wrong, even a strong LLM can only hallucinate inside the wrong context",
], size=12.5, first_flag=False)
notes(s, "This is the motivation slide for Part 2 \u2014 make the pain concrete. Use a real example from your own "
         "project: a user asks \u201cwhat is the price of plan A\u201d while the document says \u201cthis plan is budgeted at "
         "about X\u201d, so both the pronoun and the wording are misaligned and retrieval misses it.\n"
         "Closing line: Native RAG's problem is the missing feedback loop, so every agentic technique is really "
         "a decision or correction step inserted somewhere along the pipeline.")

# ---------------------------------------------------------------- 20 summary graph
s = blank(prs)
base(s, "Part 2 \u00b7 Agentic RAG", 20, "Agentic RAG map: where each technique plugs in",
     "Query side / retrieval side / knowledge base side / post-retrieval side / orchestration")
cols = [
    ("Query side", ACCENT, ["Query Rewrite", "Multi-Query", "HyDE", "Query Decomposition"]),
    ("Retrieval side", ACCENT, ["Hybrid Search", "Reranker", "Metadata Filter", "ANN tuning"]),
    ("Knowledge base", ACCENT2, ["Graph RAG", "RAPTOR", "Hierarchical index", "Table / multimodal index"]),
    ("Post-retrieval", ACCENT2, ["Self-Reflection", "Corrective RAG", "Context compression", "Citations & grounding"]),
    ("Orchestration", RGBColor(0x6B, 0x3F, 0xA0), ["Adaptive RAG", "ReAct loop", "Router / Planner", "FLARE \u00b7 SELF-RAG"]),
]
cw = (SW - 2 * M - 4 * 0.18) / 5
for i, (name, color, items) in enumerate(cols):
    cx = M + i * (cw + 0.18)
    rect(s, cx, 2.25, cw, 0.5, fill=color, radius=0.12)
    _, tf = txbox(s, cx, 2.38, cw, 0.3)
    para(tf, name, first=True, size=13, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, space_after=0)
    rect(s, cx, 2.75, cw, 2.55, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.06)
    _, tf = txbox(s, cx + 0.16, 2.92, cw - 0.32, 2.25)
    first = True
    for it in items:
        para(tf, it, first=first, size=11.5, color=BODY, space_after=7, line=1.12)
        first = False
rect(s, M, 5.55, SW - 2 * M, 1.85, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 5.74, SW - 2 * M - 0.52, 1.5)
para(tf, "How to read this: from query in to answer out, every stage can gain an \u201cintelligent\u201d step. "
         "What they share is feedback and decision-making \u2014 either fix the question (query side), fix the evidence "
         "(knowledge base and retrieval side), fix the answer (post-retrieval side), or decide whether the whole "
         "flow runs at all (orchestration).",
     first=True, size=12.5, color=INK, space_after=8, line=1.25)
para(tf, "Next, one ring at a time: Query Rewrite / Multi-Query \u2192 Adaptive RAG \u2192 Graph RAG \u2192 RAPTOR "
         "\u2192 Reranker \u2192 Self-correction & Corrective RAG.",
     size=12, color=MUTED, space_after=0)
notes(s, "This is the densest slide in the deck \u2014 pause here for a couple of minutes. Do not read the terms one "
         "by one; emphasize the classification logic: four kinds of action plus one orchestrator. An audience that "
         "leaves with this framework retains far more than an audience that memorized ten nouns.")

# ---------------------------------------------------------------- 21 query rewrite
s = blank(prs)
base(s, "Part 2 \u00b7 Agentic RAG", 21, "Query side (1): Query Rewrite",
     "What the user says is not what retrieves well")
rect(s, M, 2.12, 6.3, 2.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.32, 5.8, 1.85)
para(tf, "Problems it solves", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Coreference: \u201chow much is it?\u201d \u2192 \u201chow much is plan X?\u201d",
    "Colloquial to terminology: map everyday wording onto the vocabulary your corpus actually uses",
    "Multi-turn context: compress conversation history into the current query",
], size=12.5, first_flag=False)
table(s, M, 4.5, 6.3, [
    ["Method", "How it works"],
    ["LLM rewrite", "One LLM call turns the raw query into a retrieval-friendly form"],
    ["HyDE", "Generate a hypothetical answer first, then retrieve with it (closer to document space)"],
    ["Step-back", "Ask a more abstract parent question first, then retrieve (helps reasoning questions)"],
], col_w=[1.0, 2.3], size=10.5, row_h=0.34, head_h=0.36)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: before / after rewrite examples",
            "Suggested: 1-2 real cases showing the original query, the rewrite, and what each one retrieved (with similarity scores)",
            lines=6)
notes(s, "Query Rewrite is the highest return-on-effort technique here: one extra LLM call, and it routinely fixes "
         "both multi-turn and colloquial-query failures.\n"
         "Watch the risk: a rewrite can hallucinate or drift from user intent. Validate key entities, or keep the "
         "original query in the mix and fuse the results.")

# ---------------------------------------------------------------- 22 multi-query
s = blank(prs)
base(s, "Part 2 \u00b7 Agentic RAG", 22, "Query side (2): Multi-Query and decomposition",
     "If one vector cannot recall everything, recall along several paths")
rect(s, M, 2.12, 6.3, 2.78, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.28, 5.8, 2.6)
para(tf, "Multi-Query", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "An LLM rewrites the query into N variants with different angles and phrasing",
    "Retrieve with each, merge the candidate sets, then dedupe and rerank (RRF reciprocal rank fusion is common)",
    "Benefit: higher recall, working around the expressive limit of a single query vector",
    "Cost: N times the retrieval work plus a fusion step; N of 3-5 is usually enough",
], size=12.5, first_flag=False)
rect(s, M, 4.92, 6.3, 2.4, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 5.10, 5.8, 2.05)
para(tf, "Related: query decomposition", first=True, size=13, bold=True,
     color=ACCENT, space_after=5)
para(tf, "Split a compound question into sub-questions, retrieve and answer each, then synthesize. "
         "For example \u201cwhich of A or B is better for cold start?\u201d becomes two retrievals: "
         "cold-start capability of A, and of B.", size=11.5, color=BODY, space_after=0, line=1.18)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: Multi-Query flow / RRF formula",
            "Suggested: query \u2192 LLM generates q1..q3 \u2192 three retrievals \u2192 merge and dedupe \u2192 rerank \u2192 top-k; with RRF: score = \u03a3 1/(k + rank_i)",
            lines=6)
notes(s, "Multi-Query and the reranker are a natural pair: Multi-Query raises recall, the reranker squeezes the "
         "water out of the merged set. Present them together, because Multi-Query without reranking can actually "
         "hurt precision \u2014 you just added noise.")

# ---------------------------------------------------------------- 23 adaptive RAG
s = blank(prs)
base(s, "Part 2 \u00b7 Agentic RAG", 23, "Orchestration: Adaptive RAG",
     "Let the system decide whether to retrieve, how often, and how deep")
rect(s, M, 2.12, 6.3, 5.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "Core idea", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Not every question needs retrieval. \u201cWhat is 1+1\u201d is fine from parametric knowledge, and retrieving there only adds noise and latency",
    "Typical design: a lightweight classifier or LLM judges query complexity, then routes to one of three paths",
    "Path A (no retrieval): answer directly from the LLM",
    "Path B (single-step): one retrieval, then generate",
    "Path C (multi-step): iterative retrieval, sub-question decomposition, query reformulation as needed",
    "The signal can come from query classification, retrieval relevance scores, or generation confidence",
], size=12.5, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: Adaptive RAG routing diagram",
            "Suggested: query \u2192 (complexity classifier) \u2192 no-retrieval / single-step / multi-step branches, annotated with the latency and cost of each",
            lines=6)
notes(s, "Adaptive RAG is the flagship of the orchestration ring: it does not decide how a step is done, it decides "
         "whether and how long the pipeline runs.\n"
         "The biggest engineering win is cost and latency \u2014 pulling easy questions off the retrieval path saves "
         "tokens and time. Share your own routing split, e.g. 30% of queries need no retrieval.")

# ---------------------------------------------------------------- 24 graph RAG
s = blank(prs)
base(s, "Part 2 \u00b7 Agentic RAG", 24, "Knowledge base side (1): Graph RAG",
     "When the answer requires connecting facts across documents, vector search fails")
rect(s, M, 2.12, 6.3, 2.35, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.32, 5.8, 2.0)
para(tf, "Problems it solves", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Global questions: \u201cwhat are the main themes of the whole corpus?\u201d \u2014 no single chunk contains that answer",
    "Multi-hop questions: \u201cwhich other companies did the CEO of A's supplier invest in?\u201d \u2014 requires walking a chain of relations",
    "Vector top-K returns similar fragments; it cannot return relational structure",
], size=12.5, first_flag=False)
rect(s, M, 4.77, 6.3, 1.68, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 4.90, 5.8, 1.8)
para(tf, "Typical approach (Microsoft GraphRAG)", first=True, size=14, bold=True,
     color=ACCENT, space_after=7)
bullets(tf, [
    "An LLM extracts entities, relations and claims from every chunk to build a knowledge graph",
    "Hierarchical communities are detected and summarized ahead of time",
    "At query time: local search walks entity neighborhoods; global questions use community summaries in map-reduce",
], size=11.5, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: knowledge graph + community hierarchy",
            "Suggested: a small node-and-edge graph with colored rings showing community levels, plus an example map-reduce result for a global question",
            lines=6)
notes(s, "Be explicit about the cost: indexing calls the LLM over every chunk for extraction and then again for "
         "community summaries, so it can be 10x or more the cost of a plain vector index, and rebuilding after "
         "document changes is expensive.\n"
         "Conclusion: adopt GraphRAG only when global-summary or multi-hop questions are a large share of your "
         "traffic. Do not default to it because it is fashionable.")

# ---------------------------------------------------------------- 25 RAPTOR
s = blank(prs)
base(s, "Part 2 \u00b7 Agentic RAG", 25, "Knowledge base side (2): RAPTOR",
     "Recursive Abstractive Processing for Tree-Organized Retrieval")
rect(s, M, 2.12, 6.3, 5.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.30, 5.8, 4.5)
para(tf, "The idea", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Summarize documents recursively, bottom-up, to build a multi-level tree",
    "Leaf level: the original chunks; upper levels: LLM summaries of clustered neighboring chunks",
    "Recurse until a root node forms \u2014 the root is an abstract summary of the whole document",
    "Every node at every level gets its own embedding and lives in the same vector store",
], size=12.5, first_flag=False)
para(tf, "At retrieval time", size=14, bold=True, color=ACCENT, space_after=7)
bullets(tf, [
    "One query can hit both original chunks (detail) and higher-level summaries (the big picture)",
    "Effectively the retriever decides for itself whether to read details or the gist",
    "Common variants: collapsed tree (all nodes flattened into one level) or tree traversal",
], size=12, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: RAPTOR tree diagram",
            "Suggested: 6-8 chunks at the bottom \u2192 3 cluster summaries \u2192 1 root summary on top, with one query shown hitting nodes at two different levels",
            lines=6)
notes(s, "Contrast with GraphRAG in one sentence: both add structure on the knowledge base side, but GraphRAG "
         "extracts an entity-relation graph while RAPTOR builds a summary hierarchy tree. RAPTOR fits corpora "
         "where one topic spans many passages and some questions are detail-oriented while others are global, "
         "and its indexing cost is far below GraphRAG's.")

# ---------------------------------------------------------------- 26 reranker
s = blank(prs)
base(s, "Part 2 \u00b7 Agentic RAG", 26, "Retrieval side: Reranker",
     "Recall can be broad, but ranking has to be precise")
rect(s, M, 2.12, 6.2, 2.4, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.32, 5.7, 2.05)
para(tf, "Two-stage retrieval (recall then precision)", first=True, size=14,
     bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Stage 1 (recall): a bi-encoder plus ANN quickly pulls top-100 from millions \u2014 optimized for not missing anything",
    "Stage 2 (rerank): a cross-encoder scores each query-document pair precisely and returns top-5 \u2014 optimized for being right",
    "Why two stages are mandatory: cross-encoders are accurate but cannot be precomputed, so they only run on a small candidate set",
], size=12.5, first_flag=False)
rect(s, M, 4.72, 6.2, 2.6, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 4.90, 5.7, 2.25)
para(tf, "Practical notes", first=True, size=13, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "Candidate set size is the main tuning knob: 50-100 is usually the sweet spot; beyond that gains flatten while latency grows linearly",
    "Common models: the bge-reranker family, Qwen3-Reranker, Cohere Rerank",
], size=11.5, first_flag=False)
placeholder(s, M + 6.6, 2.12, SW - 2 * M - 6.6, 5.2,
            "Placeholder: top-k before vs after reranking",
            "Suggested: a real case with the same query showing top-5 before and after rerank, the score changes, and the difference in final answer quality",
            lines=6)
notes(s, "This is the highest-leverage and most frequently skipped step. Teams spend weeks swapping embedding "
         "models when adding a reranker would deliver a bigger gain with a far smaller change.\n"
         "Give a number from your own setup (a typical nDCG or Recall@5 lift) and be honest about the latency "
         "cost, since a cross-encoder scores every pair.")

# ---------------------------------------------------------------- 27 self-correction
s = blank(prs)
base(s, "Part 2 \u00b7 Agentic RAG", 27, "Post-retrieval side: Self-correction and the ReAct loop",
     "Retrieving is not the end \u2014 check the evidence and the answer")
rect(s, M, 2.12, 6.3, 5.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.34, 5.8, 3.9)
para(tf, "Two checkpoints for self-correction", first=True, size=14, bold=True,
     color=ACCENT, space_after=8)
bullets(tf, [
    "Check after retrieval: are the retrieved documents actually relevant? If not, rewrite the query and retrieve again",
    "Check after generation: is every claim supported by a citation? If not, flag it or regenerate",
    "Implementation: an extra LLM call acting as critic, or a purpose-trained discriminator model",
], size=12.5, first_flag=False)
para(tf, "The ReAct loop: Reason \u2192 Act \u2192 Observe", size=14, bold=True,
     color=ACCENT, space_after=7)
bullets(tf, [
    "Retrieval becomes one of the agent's tools, and the model decides when to call it and with what arguments",
    "Multiple turns: think, call a tool, read the result, decide the next step",
    "The real difference from Native RAG: how many retrievals happen and what gets asked are no longer hardcoded",
], size=12, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: ReAct trace (Thought/Action/Observation)",
            "Suggested: a real multi-turn trace where the model queries A, finds it insufficient, then queries B; annotate latency and token cost per step",
            lines=6)
notes(s, "Both the upside and the cost are obvious: self-correction converts wrong answers into refusals, but "
         "runaway loops blow up latency and spend. Always cap the number of iterations and set a budget.\n"
         "Reference the Reflexion / Self-RAG line of work: turning critique into part of the model's own output.")

# ---------------------------------------------------------------- 28 corrective RAG
s = blank(prs)
base(s, "Part 2 \u00b7 Agentic RAG", 28, "Post-retrieval side: Corrective RAG (CRAG)",
     "Put a quality gate on the retrieval results")
rect(s, M, 2.12, 6.3, 2.25, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.32, 5.8, 1.9)
para(tf, "Three-step flow", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "1) Evaluate: a lightweight evaluator scores the relevance of each retrieved document (self-assessment or a trained retrieval evaluator)",
    "2) Bucket: sort results into Correct / Incorrect / Ambiguous by score",
    "3) Act: choose the follow-up per bucket (correct, or fall back to web search, or generate directly)",
], size=12, first_flag=False)
table(s, M, 4.55, 6.3, [
    ["Bucket", "Meaning", "Action"],
    ["Correct", "Relevant and reliable", "Use directly for generation"],
    ["Incorrect", "Irrelevant", "Discard, rewrite the query and retrieve, or fall back to external search"],
    ["Ambiguous", "Partly relevant", "Combine both: knowledge base plus external search fusion"],
], col_w=[0.9, 1.1, 2.0], size=10.5, row_h=0.34, head_h=0.36)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: CRAG flow diagram",
            "Suggested: query \u2192 retriever \u2192 evaluator (scoring) \u2192 three branches (correct refines, incorrect discards and rewrites, ambiguous fuses) \u2192 generator; web search as an alternate knowledge source",
            lines=6)
notes(s, "Call out the difference from Self-RAG: CRAG is an engineered pattern with an external evaluator and "
         "bucketed actions, and it needs no model retraining. Self-RAG trains the model to emit reflection tokens "
         "itself, which is stronger but requires retraining.\n"
         "Practical advice: start with the CRAG-style evaluator plus rules, prove the value, and only then consider Self-RAG.")

# ---------------------------------------------------------------- 29 wrap-up
s = blank(prs)
base(s, "Wrap-up", 29, "Summary and how to apply this",
     "From swapping models to adding feedback loops")
rect(s, M, 2.12, 6.4, 2.62, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.30, 5.9, 2.48)
para(tf, "Part 1 in one sentence", first=True, size=14, bold=True,
     color=ACCENT, space_after=8)
bullets(tf, [
    "BERT brought bidirectional understanding but had no sentence-level objective, so SBERT added Siamese networks and contrastive learning",
    "MTEB made models comparable, BGE made Chinese practical, and the decoder-only route (NV-Embed, Qwen3) turned LLMs into retrievers directly",
    "Selection order: shortlist from benchmarks, run a small eval on your own data, then weigh dimension, latency and cost",
], size=12.5, first_flag=False)
rect(s, M + 6.8, 2.12, SW - 2 * M - 6.8, 2.62, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 7.06, 2.30, SW - 2 * M - 7.06, 2.48)
para(tf, "Part 2 in one sentence", first=True, size=14, bold=True,
     color=ACCENT2, space_after=8)
bullets(tf, [
    "Native RAG's bottleneck is the absence of feedback: fixed splitting, single-path retrieval, and hallucination when recall is wrong",
    "Agentic RAG adds decisions and corrections along the pipeline: fix the question, fix the evidence, fix the answer, or change the flow",
    "More techniques is not better \u2014 measure where the bottleneck is, then treat that stage",
], size=12.5, first_flag=False)
rect(s, M, 4.87, SW - 2 * M, 1.78, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 5.06, SW - 2 * M - 0.52, 1.45)
para(tf, "Q&A / discussion placeholder", first=True, size=13, bold=True,
     color=ACCENT, space_after=6)
para(tf, "\u00b7 Which technique should we try first in our own use case, and why?", size=11.5, color=BODY, space_after=3)
para(tf, "\u00b7 If you could only pick one: a better embedding model, or a reranker? Why?", size=11.5, color=BODY, space_after=3)
para(tf, "\u00b7 Discussion: is GraphRAG's indexing cost acceptable at our data volume?", size=11.5, color=BODY, space_after=0)
notes(s, "Wrap-up. Leave 5-10 minutes for Q&A. If you want a closer, prepare a live demo: the same question "
         "answered by Native RAG versus multi-query plus reranking.")

# ---------------------------------------------------------------- 30 orchestration frameworks
s = blank(prs)
base(s, "Part 2 \u00b7 Agentic RAG", 30, "Orchestration frameworks: LangGraph, LlamaIndex and friends",
     "\u201cAgentic RAG\u201d in production is mostly a framework choice: who decides the next step, and where state lives")
table(s, M, 2.14, SW - 2 * M, [
    ["Framework", "Paradigm", "Core idea", "Best for"],
    ["Plain Python", "Hand-rolled",
     "A function per step; you own the loop and the state",
     "Getting an honest baseline before any dependency"],
    ["LangGraph", "Graph / state machine",
     "Nodes + edges + shared typed state; checkpointer persists every step; built-in interrupts for human-in-the-loop",
     "Explicit, auditable control flow \u2014 the default choice for agentic RAG"],
    ["LlamaIndex Workflows", "Event-driven steps",
     "Steps subscribe to and emit events (async); first-class retrievers, indexes, parsers, query engines",
     "Retrieval- and index-centric pipelines that need custom control"],
    ["CrewAI", "Role-based multi-agent",
     "Agents carry roles, goals and tools; sequential or hierarchical process delegates tasks",
     "Multi-role workflows (researcher \u2192 analyst \u2192 writer) fast to prototype"],
    ["AutoGen / AG2", "Conversational multi-agent",
     "Agents converse; group chat with a manager; code execution sandbox",
     "Research-style multi-agent dialogue and code-execution agents"],
    ["Pydantic AI", "Type-safe agents",
     "Pydantic-validated structured outputs and typed dependencies; works well behind FastAPI",
     "Teams that want schema guarantees and minimal magic"],
    ["Semantic Kernel", "Plugins / planner",
     "Skills and plugins composed by a planner; strong .NET and enterprise integration",
     "Enterprise stacks, especially .NET, with existing connectors"],
    ["DSPy", "Declarative optimization",
     "Declare modules and signatures, supply a metric, compile to find prompts \u2014 optimizing, not orchestrating",
     "Tuning prompt/few-shot quality once the flow is stable"],
], col_w=[1.45, 1.25, 2.75, 2.45], size=10, row_h=0.47, head_h=0.34)
_, tf = txbox(s, M, 6.55, SW - 2 * M, 0.7)
para(tf, "How to choose: 1) if you only need one retrieval and one LLM call, do not adopt a framework at all. "
         "2) Branching, retries, or human approval \u2192 a graph (LangGraph). 3) Retrieval and indexing are the hard "
         "part \u2192 LlamaIndex. 4) Several distinct roles \u2192 CrewAI or AutoGen. 5) Prompt quality is the bottleneck "
         "\u2192 DSPy. 6) Whatever you pick, insist on tracing, checkpointing and replayable state.",
     first=True, size=11, bold=True, color=INK, space_after=0, line=1.18)
notes(s, "Positioning: this slide sits after the wrap-up because it answers the natural next question \u2014 "
         "\u201cthis all sounds like a lot of plumbing, what do we build it with?\u201d\n"
         "Lead with the framing line: every one of these frameworks does the same two jobs \u2014 it decides what "
         "happens next, and it holds the state. The differentiation is (a) explicit graph versus emergent "
         "agent behavior, (b) how mature the retrieval/indexing layer is, (c) how observable and replayable a "
         "failed run is.\n"
         "Sound engineering advice to give out loud: prototype agentic RAG with plain Python first, feel the pain, "
         "then adopt LangGraph (for flow) or LlamaIndex (for retrieval). Framework lock-in is real \u2014 the trace, "
         "the checkpoint and the state schema matter more than the star count.\n"
         "Do not start a framework war in the room: if someone asks \u201cwhich is best?\u201d, answer with the decision "
         "guide at the bottom rather than a ranking.")

# ---------------------------------------------------------------- 31 backup divider
s = blank(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x22, 0x24, 0x2A))
_, tf = txbox(s, M + 0.2, 3.1, SW - 2 * M, 1.3)
para(tf, "BACKUP", first=True, size=15, bold=True,
     color=RGBColor(0x9A, 0xA4, 0xB2), space_after=8)
para(tf, "Backup: FLARE and SELF-RAG", size=36, bold=True, color=WHITE, space_after=8)
para(tf, "Use these if we have time left, or if someone asks how retrieval timing is decided",
     size=13, color=RGBColor(0xB8, 0xBE, 0xC8), space_after=0)
notes(s, "Backup slides. If the talk runs long, skip these two. If someone pushes on \u201chow does the system decide "
         "when to retrieve?\u201d, jump straight here.")

# ---------------------------------------------------------------- 31 FLARE
s = blank(prs)
base(s, "Backup", 31, "FLARE: let the LLM's own confidence trigger retrieval",
     "Forward-Looking Active REtrieval")
rect(s, M, 2.12, 6.3, 2.35, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.32, 5.8, 2.0)
para(tf, "Core idea", first=True, size=14, bold=True, color=ACCENT, space_after=8)
bullets(tf, [
    "Instead of generating a full answer and checking afterwards, generate and judge as you go",
    "When the model is unsure about what comes next (low confidence), it triggers a retrieval",
    "Generation then resumes with the new evidence \u2014 timing is driven by model uncertainty rather than a fixed pipeline",
], size=12.5, first_flag=False)
rect(s, M, 4.6, 6.3, 1.85, fill=PLACE_BG, line=PLACE_LINE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.24, 4.78, 5.8, 1.5)
para(tf, "Two ways to trigger", first=True, size=13, bold=True, color=ACCENT, space_after=6)
rich(tf, [("Direct: ", {"bold": True, "color": INK}),
          ("before generating the next sentence, look at its confidence score (probability). If it is low, "
           "search with that sentence itself as the query.", {"color": BODY})],
     size=11.5, space_after=5, line=1.18)
rich(tf, [("Instruct: ", {"bold": True, "color": INK}),
          ("use the sentences generated so far to have a separate LLM produce a search query, retrieve with it, "
           "and merge the results back before continuing.", {"color": BODY})],
     size=11.5, space_after=0, line=1.18)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: interleaved generate-retrieve flow",
            "Suggested: generate ... \u2192 low confidence \u2192 build query \u2192 retrieve \u2192 merge context \u2192 continue generating; show the Direct and Instruct branches",
            lines=6)
notes(s, "FLARE's value is retrieval on demand: an answer may trigger only one or two retrievals instead of "
         "querying at every step.\n"
         "Engineering notes: use token probabilities to compute confidence, and low-probability tokens are "
         "themselves a useful \u201ctime to look it up\u201d signal. Frequent triggering makes latency unpredictable, "
         "so set thresholds and a hard cap on retrievals.")

# ---------------------------------------------------------------- 32 SELF-RAG
s = blank(prs)
base(s, "Backup", 32, "SELF-RAG: train the model to critique itself",
     "Self-Reflective Retrieval-Augmented Generation")
rect(s, M, 2.12, 6.3, 5.2, fill=None, line=RULE, lw=1.0, radius=0.03)
_, tf = txbox(s, M + 0.26, 2.30, 5.8, 4.7)
para(tf, "How it differs from FLARE and CRAG", first=True, size=14, bold=True,
     color=ACCENT, space_after=8)
bullets(tf, [
    "FLARE uses the model's output probabilities to trigger retrieval; CRAG bolts on an external evaluator",
    "SELF-RAG takes a third route: train a model that emits reflection tokens as part of its own generation",
], size=12.5, first_flag=False)
para(tf, "Four kinds of reflection token", size=14, bold=True, color=ACCENT, space_after=7)
bullets(tf, [
    "Retrieve: does this step need retrieval at all (the retrieval trigger)",
    "IsRel: is the retrieved passage relevant",
    "IsSup: is the generated sentence supported by the evidence",
    "IsUse: is the overall response useful to the user",
], size=12.5, first_flag=False)
para(tf, "How it is trained", size=14, bold=True, color=ACCENT, space_after=6)
bullets(tf, [
    "A strong model (GPT-4 class) annotates the corpus with reflection tokens, then the target model is supervised-fine-tuned",
    "At inference, those token weights can drive weighted decoding that favors candidates with evidence support",
], size=12.5, first_flag=False)
placeholder(s, M + 6.7, 2.12, SW - 2 * M - 6.7, 5.2,
            "Placeholder: reflection token placement",
            "Suggested: a generation sequence alternating tokens and content, e.g. [Retrieve] / passage / [IsRel] / sentence / [IsSup] / [IsUse]; compare output with retrieval on and off",
            lines=6)
notes(s, "This is the heaviest option: it needs retraining plus annotation from a strong model, which is expensive.\n"
         "In exchange, self-correction moves from a bolt-on pipeline into an intrinsic model capability, so "
         "inference no longer needs repeated extra LLM calls.\n"
         "Recommendation: unless there is a long-term commitment, validate the idea first with a CRAG-style "
         "bolt-on approach.")

# ---------------------------------------------------------------- 33 thanks
s = blank(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x12, 0x2A, 0x48))
_, tf = txbox(s, M + 0.2, 2.85, SW - 2 * M, 1.8)
para(tf, "Thanks & Q&A", first=True, size=40, bold=True, color=WHITE, space_after=10)
para(tf, "References: SBERT (Reimers & Gurevych, 2019) \u00b7 MTEB (Muennighoff et al.) \u00b7 "
         "BGE / C-MTEB (BAAI) \u00b7 NV-Embed (NVIDIA) \u00b7 Qwen3-Embedding (Alibaba) \u00b7 "
         "RAPTOR \u00b7 GraphRAG (Microsoft) \u00b7 CRAG \u00b7 FLARE \u00b7 SELF-RAG",
     size=11.5, color=RGBColor(0x9F, 0xB8, 0xD4), space_after=0, line=1.3)
rect(s, M + 0.2, 5.05, SW - 2 * M - 0.4, 1.0, fill=RGBColor(0x1B, 0x3A, 0x5E), radius=0.05)
_, tf = txbox(s, M + 0.45, 5.25, SW - 2 * M - 0.9, 0.7)
para(tf, "Placeholder: contact info / internal links / repo URL", first=True,
     size=12, bold=True, color=WHITE, space_after=0)
notes(s, "Closing slide. Read the reference list aloud, and always state the evaluation date when quoting "
         "leaderboard numbers.")

prs.save(OUT)
print("saved:", OUT)
print("slides:", len(prs.slides._sldIdLst))
