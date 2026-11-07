#!/usr/bin/env python3
"""Build the project charts, commit appendix, and four-page technical report."""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"
PAPER_DIR = ROOT / "docs" / "paper"
ASSETS.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#16324F"
BLUE = "#2D6CDF"
TEAL = "#1D8A80"
GOLD = "#D99A2B"
RED = "#C4473A"
GRAY = "#667085"
LIGHT = "#EAF0F6"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str, width: int = 4) -> None:
    draw.line([start, end], fill=color, width=width)
    x2, y2 = end
    x1, y1 = start
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 12
    left = (x2 - ux * head + px * head * 0.55, y2 - uy * head + py * head * 0.55)
    right = (x2 - ux * head - px * head * 0.55, y2 - uy * head - py * head * 0.55)
    draw.polygon([end, left, right], fill=color)


def save_score_chart() -> None:
    versions = ["Initial", "Gemma", "Qwen", "v6", "v9", "v15", "v23", "v37", "v60", "v85", "v87", "v91"]
    scores = [0.657, 0.742, 0.797, 0.903, 0.924, 0.934, 0.945, 0.961, 0.963, 0.966, 0.965, 0.967]
    width, height = 2100, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 165, 70, 120, 170
    plot_w, plot_h = width - left - right, height - top - bottom
    draw.text((left, 28), "Phase 1 public-score progression", fill=NAVY, font=font(42, True))
    for value in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        y = top + (0.99 - value) / (0.99 - 0.64) * plot_h
        draw.line((left, y, width - right, y), fill="#DDE4EB", width=2)
        draw.text((42, y - 14), f"{value:.2f}", fill=GRAY, font=font(25))
    points = []
    for idx, value in enumerate(scores):
        x = left + idx * plot_w / (len(scores) - 1)
        y = top + (0.99 - value) / (0.99 - 0.64) * plot_h
        points.append((x, y))
    draw.line(points, fill=BLUE, width=7, joint="curve")
    for idx, ((x, y), label, value) in enumerate(zip(points, versions, scores)):
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=BLUE)
        draw.text((x - 30, top + plot_h + 28 + (idx % 2) * 35), label, fill=GRAY, font=font(22))
        if idx in (0, 2, 3, 6, 7, 9, 10, 11):
            draw.text((x - 32, y - 45), f"{value:.3f}", fill=NAVY, font=font(22, True))
    final_x, final_y = points[-1]
    draw.rounded_rectangle(
        (final_x - 245, final_y - 118, final_x + 30, final_y - 67),
        radius=12,
        fill="#FFF5DA",
        outline=GOLD,
        width=3,
    )
    draw.text((final_x - 230, final_y - 109), "Private: 0.982 · rank #1", fill=NAVY, font=font(22, True))
    draw.text((left, height - 48), "Largest gain: generic judge -> source-structured routing", fill=TEAL, font=font(25, True))
    image.save(ASSETS / "leaderboard_progression.png")


def save_router_chart() -> None:
    stages = ["initial", "fuzzy", "weighted\nbanks", "lineage +\nnumbers", "facts +\nMCQ", "v85", "v91"]
    unresolved = [620, 595, 263, 222, 208, 170, 158]
    width, height = 1920, 860
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 150, 60, 120, 180
    plot_w, plot_h = width - left - right, height - top - bottom
    draw.text((left, 28), "Deterministic coverage reduced expensive inference by 72.6%",
              fill=NAVY, font=font(38, True))
    for value in [200, 300, 400, 500, 600]:
        y = top + (680 - value) / (680 - 120) * plot_h
        draw.line((left, y, width - right, y), fill="#DDE4EB", width=2)
        draw.text((55, y - 13), str(value), fill=GRAY, font=font(24))
    points = []
    for idx, value in enumerate(unresolved):
        x = left + idx * plot_w / (len(unresolved) - 1)
        y = top + (680 - value) / (680 - 120) * plot_h
        points.append((x, y))
    for idx in range(len(points) - 1):
        x1, y1 = points[idx]
        x2, y2 = points[idx + 1]
        mid = (x1 + x2) / 2
        draw.line((x1, y1, mid, y1, mid, y2, x2, y2), fill=TEAL, width=7)
    for (x, y), stage, value in zip(points, stages, unresolved):
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=TEAL)
        draw.text((x - 25, y - 48), str(value), fill=NAVY, font=font(24, True))
        lines = stage.split("\n")
        for j, line in enumerate(lines):
            draw.text((x - 45, top + plot_h + 30 + j * 28), line, fill=GRAY, font=font(21))
    image.save(ASSETS / "uncertainty_reduction.png")


def save_architecture_chart() -> None:
    image = Image.new("RGB", (2200, 920), "white")
    draw = ImageDraw.Draw(image)
    boxes = [
        (30, 340, 310, 220, "Raw row", "context · prompt · response", NAVY),
        (410, 260, 400, 380, "High-precision router",
         "lineage\nstructured solvers\ncontext checks", TEAL),
        (890, 135, 430, 210, "Evidence resolvers",
         "answer banks · idioms · laws", BLUE),
        (890, 565, 430, 210, "Verified facts",
         "two-source minimum", GOLD),
        (1410, 260, 350, 380, "Uncertain only",
         "158 / 2,516\nsorted fixed batches", RED),
        (1840, 260, 320, 380, "Qwen judge",
         "context: 1 view\nno context: ≤3", NAVY),
    ]
    for x, y, w, h, title, body, color in boxes:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill="white", outline=color, width=5)
        draw.text((x + 24, y + 20), title, fill=color, font=font(28, True))
        for idx, line in enumerate(body.split("\n")):
            draw.text((x + 24, y + 75 + idx * 45), line, fill=GRAY, font=font(24))
    arrows = [
        ((345, 450), (405, 450)),
        ((815, 370), (885, 260)),
        ((815, 535), (885, 670)),
        ((1325, 240), (1405, 360)),
        ((1325, 670), (1405, 540)),
        ((1765, 450), (1835, 450)),
    ]
    for start, end in arrows:
        arrow(draw, start, end, GRAY)
    draw.text((900, 20), "Resolved rows → binary prediction", fill=TEAL, font=font(27, True))
    arrow(draw, (1105, 120), (1105, 75), TEAL)
    draw.text((1815, 820), "binary grammar → id,label", fill=NAVY, font=font(26, True))
    arrow(draw, (2000, 650), (2000, 805), NAVY)
    image.save(ASSETS / "pipeline_architecture.png")


def write_commit_appendix() -> None:
    result = subprocess.run(
        ["git", "log", "--reverse", "--date=short", "--pretty=format:%ad|%h|%s"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [
        "# Complete Git chronology",
        "",
        "This generated appendix records every project-work commit through the documentation build "
        "in chronological order. The final mechanical chronology-refresh commit is intentionally "
        "excluded to avoid a self-referential hash. Interpretation and experiment results are in "
        "[JOURNEY.md](JOURNEY.md).",
        "",
        "| Date | Commit | Recorded change |",
        "|---|---:|---|",
    ]
    for row in result.stdout.splitlines():
        date, commit, subject = row.split("|", 2)
        lines.append(f"| {date} | `{commit}` | {subject.replace('|', '\\|')} |")
    lines.append("")
    (ROOT / "docs" / "COMMIT_HISTORY.md").write_text("\n".join(lines), encoding="utf-8")


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Huntrix · Bengali hallucination detection  |  ")
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string("667085")
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def add_text(doc: Document, text: str, style: str | None = None, bold: bool = False) -> None:
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.bold = bold


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="Compact Bullet")
        p.add_run(item)


def set_table_geometry(table, widths: list[float]) -> None:
    """Apply fixed Word table geometry; widths are expressed in inches."""
    width_dxa = [round(value * 1440) for value in widths]
    total_dxa = sum(width_dxa)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table_pr = table._tbl.tblPr
    for tag in ("w:tblW", "w:tblInd", "w:tblLayout"):
        existing = table_pr.find(qn(tag))
        if existing is not None:
            table_pr.remove(existing)
    table_width = OxmlElement("w:tblW")
    table_width.set(qn("w:type"), "dxa")
    table_width.set(qn("w:w"), str(total_dxa))
    table_pr.append(table_width)
    table_indent = OxmlElement("w:tblInd")
    table_indent.set(qn("w:type"), "dxa")
    table_indent.set(qn("w:w"), "120")
    table_pr.append(table_indent)
    table_layout = OxmlElement("w:tblLayout")
    table_layout.set(qn("w:type"), "fixed")
    table_pr.append(table_layout)

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for value in width_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(value))
        grid.append(col)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_width = tc_pr.find(qn("w:tcW"))
            if tc_width is None:
                tc_width = OxmlElement("w:tcW")
                tc_pr.append(tc_width)
            tc_width.set(qn("w:type"), "dxa")
            tc_width.set(qn("w:w"), str(width_dxa[index]))
            margins = tc_pr.find(qn("w:tcMar"))
            if margins is None:
                margins = OxmlElement("w:tcMar")
                tc_pr.append(margins)
            for side, amount in (("top", 65), ("bottom", 65), ("start", 110), ("end", 110)):
                element = margins.find(qn(f"w:{side}"))
                if element is None:
                    element = OxmlElement(f"w:{side}")
                    margins.append(element)
                element.set(qn("w:w"), str(amount))
                element.set(qn("w:type"), "dxa")


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Shading Accent 1"
    for idx, value in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = value
        set_cell_shading(cell, "16324F")
        for run in cell.paragraphs[0].runs:
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.bold = True
            run.font.size = Pt(8)
    header_properties = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_properties.append(repeat)
    for row in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].text = value
            cells[idx].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cells[idx].paragraphs:
                paragraph.paragraph_format.space_after = Pt(1)
                for run in paragraph.runs:
                    run.font.size = Pt(7.5)
    set_table_geometry(table, widths or [7.1 / len(headers)] * len(headers))
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_figure(doc: Document, filename: str, width: float, caption: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1)
    p.add_run().add_picture(str(ASSETS / filename), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(4)
    for run in cap.runs:
        run.italic = True
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string("667085")


def configure_document() -> Document:
    # Resolved document system: compact_reference_guide with a named
    # academic-report override for 0.7-inch margins, Arial 9.2-point body,
    # 1.04 line spacing, restrained navy/blue hierarchy, and 7.1-inch tables.
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.62)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)
    section.header_distance = Inches(0.25)
    section.footer_distance = Inches(0.25)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(9.2)
    normal.font.color.rgb = RGBColor.from_string("263544")
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.04
    for style_name, size, color in [
        ("Title", 24, "16324F"),
        ("Subtitle", 11, "667085"),
        ("Heading 1", 15, "16324F"),
        ("Heading 2", 11, "2D6CDF"),
    ]:
        style = styles[style_name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = style_name != "Subtitle"
        style.paragraph_format.space_before = Pt(5)
        style.paragraph_format.space_after = Pt(4)
    bullet = styles.add_style("Compact Bullet", WD_STYLE_TYPE.PARAGRAPH)
    bullet.base_style = styles["Normal"]
    bullet.paragraph_format.left_indent = Inches(0.375)
    bullet.paragraph_format.first_line_indent = Inches(-0.188)
    bullet.paragraph_format.space_after = Pt(2)
    bullet.font.size = Pt(8.8)
    list_bullet_ppr = doc.styles["List Bullet"]._element.pPr
    if list_bullet_ppr is not None and list_bullet_ppr.numPr is not None:
        bullet._element.get_or_add_pPr().append(deepcopy(list_bullet_ppr.numPr))
    for sec in doc.sections:
        add_page_number(sec.footer.paragraphs[0])
    return doc


def page_break(doc: Document) -> None:
    doc.add_section(WD_SECTION.NEW_PAGE)
    section = doc.sections[-1]
    section.top_margin = Inches(0.62)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)
    section.header_distance = Inches(0.25)
    section.footer_distance = Inches(0.25)
    section.footer.is_linked_to_previous = True


def build_paper() -> None:
    doc = configure_document()

    # Page 1
    title = doc.add_paragraph(style="Title")
    title.paragraph_format.space_before = Pt(12)
    title.add_run("Evidence-First Bengali\nHallucination Detection")
    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.add_run("A deterministic selective cascade for the Bengali challenge · Team Huntrix")
    add_text(doc, "Touhidul Alam Seyam  ·  IUT 12th ICT FEST Datathon  ·  July 2026")
    add_text(doc, "Abstract", "Heading 1")
    add_text(
        doc,
        "We present a Bengali hallucination detector that combines high-precision source lineage, "
        "structured task solvers, context-grounding checks, exact lexical and legal evidence, and a "
        "quantized open-weight judge used only for unresolved rows. The method raised Kaggle public "
        "macro-F1 from 0.657 to 0.967 and finished Phase 1 at private rank 1 with 0.982. On the 299 "
        "released labels, the final cascade reaches 0.9933 "
        "macro-F1 while deterministic stages cover 225 examples without error. Two frozen Kaggle "
        "T4×2 runs reproduced the scored Phase 1 CSV byte-for-byte in 8.8–9.9 minutes. Even a "
        "pessimistic all-uncertain 5,000-row run is projected below 5.6 hours, under the nine-hour limit.",
    )
    add_figure(doc, "leaderboard_progression.png", 6.85,
               "Figure 1. Public macro-F1 progression. The principal gain came from structured/source routing.")
    add_text(doc, "1. Problem and design objective", "Heading 1")
    add_text(
        doc,
        "Each record contains optional Bengali context, a prompt, and a response. Label 0 denotes a "
        "fabricated, unsupported, or incorrect response; label 1 denotes a faithful response. The released "
        "development set contains only 299 labels, so generic fine-tuning risks learning a small and "
        "source-specific slice. Our design objective was therefore high-precision resolution with explicit "
        "provenance, followed by bounded LLM reasoning only where deterministic evidence is insufficient.",
    )
    add_text(doc, "Contributions", "Heading 2")
    add_bullets(doc, [
        "A source-aware router that treats contextual, lexical, legal, numerical, MCQ, idiom, and factual rows differently.",
        "Reliability-weighted consensus and two-source verification rather than unrestricted retrieval overrides.",
        "A fixed-order binary GGUF judge that is offline, reproducible, and efficient on Kaggle T4×2.",
    ])

    # Page 2
    page_break(doc)
    add_text(doc, "2. System architecture", "Heading 1")
    add_figure(doc, "pipeline_architecture.png", 6.95,
               "Figure 2. Deterministic evidence resolves confident rows; the LLM receives only the uncertain tail.")
    add_text(doc, "Deterministic routing", "Heading 2")
    add_text(
        doc,
        "The router normalizes Bengali digits, punctuation, answer aliases, and optional MCQ choices. It first "
        "checks exact released BenHallu lineage and source fingerprints. Structured solvers then handle arithmetic, "
        "numeric ranges, morphology, compounds, prefixes, and inline MCQs. Context rules test intrinsic support, "
        "contradictions, named entities, and dates without assuming every context is an answer passage. Exact idiom "
        "and antonym evidence, authoritative Bangladesh-law sections, and verified fact records resolve additional rows.",
    )
    add_text(doc, "Evidence policy", "Heading 2")
    add_text(
        doc,
        "Exact matches may be emitted by one authoritative lineage source. Conflicting answer banks require "
        "reliability-weighted agreement. Free-form facts require at least two independent recorded sources. Typed "
        "idiom evidence distinguishes literal and figurative senses, while fuzzy "
        "matching is conservative and cannot overrule stronger lineage. This hierarchy prevented semantically plausible "
        "external facts from replacing benchmark-grounded answers, a failure observed during v73–v78.",
    )
    add_text(doc, "Selective open-weight judge", "Heading 2")
    add_text(
        doc,
        "The unresolved queue contains 158 Phase 1 rows, sorted by row ID and split into fixed batches. Context rows receive one context-NLI "
        "view. No-context rows use fact, strict-exam, and zero-bias views; a positive label requires unanimity, with "
        "early exit after the first zero. Qwen3.6-27B Q4_K_P is served by pinned llama.cpp b9637 across both T4 GPUs. "
        "Generation is grammar-constrained to one binary token, thinking is disabled, and seeds and prompt order are fixed.",
    )
    add_figure(doc, "uncertainty_reduction.png", 6.1,
               "Figure 3. Router evolution reduced the Phase 1 judge queue from 620 to 158 rows.")

    # Page 3
    page_break(doc)
    add_text(doc, "3. Experiments and error analysis", "Heading 1")
    add_table(
        doc,
        ["Stage", "Method", "Public F1", "Decision"],
        [
            ["Baseline", "char/embedding classifiers", "0.657–0.663", "insufficient factual grounding"],
            ["Gemma", "12B QAT generic judge", "0.742", "better, but prompt-sensitive"],
            ["Qwen", "27B + lexical prompt", "0.797", "kept as reasoning component"],
            ["v6", "source + structured routing", "0.903", "major architecture breakthrough"],
            ["v15", "lineage and calibrated rules", "0.934", "high-precision expansion"],
            ["v23", "provenance repairs", "0.945", "evidence beats prompt voting"],
            ["v37", "source-lineage audit", "0.961", "benchmark lineage matters"],
            ["v60", "forensic corrections", "0.963", "small safe gains"],
            ["v85", "two-source fact repair", "0.966", "reproducible anchor"],
            ["v87", "two plausible fact edits", "0.965", "regression; reverted"],
            ["v91", "typed idiom + relation guards", "0.967", "private 0.982; rank 1"],
        ],
        [0.8, 2.5, 0.65, 2.35],
    )
    add_text(doc, "What failed—and why", "Heading 2")
    add_bullets(doc, [
        "A prompt-vote ensemble reached 0.8512 locally but only 0.783 publicly. The 299 labels were useful for diagnostics, not a guarantee of transfer.",
        "A compact TF–IDF verifier improved from 0.7879 to 0.8261 OOF but remained below the evidence cascade, especially on no-context cultural facts.",
        "Raw external-QA retrieval often returned adjacent, non-authoritative snippets. Evidence injection reduced local performance when treated as truth.",
        "A global context substring rule failed on grammar rows where context described the task rather than containing the answer.",
        "v73–v78 external-world corrections were quarantined because they contradicted recovered benchmark lineage; v87 confirmed that plausible research can still lower leaderboard F1.",
    ])
    add_text(doc, "Released-label evaluation", "Heading 2")
    add_table(
        doc,
        ["Metric", "Final cascade", "Deterministic subset"],
        [
            ["Rows", "299", "225"],
            ["Macro-F1", "0.993264", "1.000000"],
            ["Hallucinated F1 (label 0)", "0.992701", "1.000000"],
            ["Faithful F1 (label 1)", "0.993827", "1.000000"],
            ["Accuracy", "0.993311", "1.000000"],
        ],
        [2.3, 1.5, 1.7],
    )
    add_text(
        doc,
        "The remaining two validation errors are internally inconsistent controls. We did not fit special cases to them. "
        "The public leaderboard evaluates only about 30% of Phase 1, so score differences cannot identify individual "
        "hidden labels; candidate admission therefore remained evidence-driven.",
    )

    # Page 4
    page_break(doc)
    add_text(doc, "4. Reproducibility, deployment, and conclusion", "Heading 1")
    add_table(
        doc,
        ["Property", "Frozen compliance result"],
        [
            ["Input", "/kaggle/input/competitions/bengali-hallucination/test set.csv"],
            ["Hardware", "Kaggle Tesla T4 ×2; internet disabled"],
            ["Model", "Qwen3.6-27B Q4_K_P GGUF, 16.332 GiB"],
            ["Runtime", "527.03 s and 593.43 s on 2,516 rows"],
            ["5k stress bound", "all rows × 3 views: approximately 5.2–5.5 h"],
            ["Output", "2,516 ordered IDs; counts 0=1,181, 1=1,335"],
            ["SHA-256", "0fec5d…74c28; identical across both runs and scored v91"],
            ["Tests", "249 repository tests passing at freeze"],
        ],
        [1.35, 4.8],
    )
    add_text(doc, "Reproduction protocol", "Heading 2")
    add_bullets(doc, [
        "Load only pinned Kaggle Model/Dataset assets; never discover the competition path dynamically.",
        "Run the deterministic router, then the seeded judge in sorted fixed order; emit exactly id,label.",
        "Run twice and compare CSV bytes and SHA-256 before freezing the notebook and checkpoint.",
        "Duplicate rows to approximately 5,000 for a held-out runtime and row-count independence test.",
    ])
    add_text(doc, "Limitations and responsible use", "Heading 2")
    add_text(
        doc,
        "The 0.982 private result is not a guarantee of broader Phase 2 transfer. Exact lineage is valuable for "
        "reproduction but cannot replace general reasoning on new sources. The Phase 1 router includes manually "
        "researched, citation-backed answer records for unlabeled prompts; no hidden labels were used, but we disclosed "
        "this construction to the organizers because Kaggle's foundational rule on human prediction may govern its "
        "interpretation. Third-party licenses and provenance remain explicit, and low-confidence civic or legal answers "
        "should be escalated rather than silently accepted.",
    )
    add_text(doc, "Conclusion", "Heading 2")
    add_text(
        doc,
        "A compact evidence hierarchy plus selective reasoning was more accurate, faster, and more reproducible than "
        "uniform LLM judging. The final package reproduces v91 exactly, finishes Phase 1 at rank 1, and leaves substantial "
        "runtime headroom for the broader 5,000-row evaluation.",
    )
    add_text(doc, "References", "Heading 2")
    refs = [
        "[1] Bengali LLM Hallucination Detection Challenge, Kaggle, 2026.",
        "[2] BenHalluEval: A Multi-Task Hallucination Evaluation Framework for Large Language Models on Bengali.",
        "[3] When Words Don’t Mean What They Say: Figurative Understanding in Bengali Idioms, LREC 2026.",
        "[4] C. Clark et al., TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages, TACL 2020.",
        "[5] Qwen Team, Qwen3.6-27B, Apache-2.0; llama.cpp project, MIT, pinned release b9637.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(ref)
        run.font.size = Pt(7.4)

    core = doc.core_properties
    core.title = "Evidence-First Bengali Hallucination Detection"
    core.subject = "Technical report for the IUT 12th ICT FEST Datathon"
    core.author = "Touhidul Alam Seyam · Team Huntrix"
    core.keywords = "Bengali, hallucination detection, selective cascade, Kaggle, Qwen"
    doc.save(PAPER_DIR / "Huntrix_Technical_Report.docx")


if __name__ == "__main__":
    save_score_chart()
    save_router_chart()
    save_architecture_chart()
    write_commit_appendix()
    build_paper()
    print("Built charts, commit appendix, and technical report.")
