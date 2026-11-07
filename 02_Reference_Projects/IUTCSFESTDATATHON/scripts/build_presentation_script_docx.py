from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "HUNTRIX_PRESENTATION_SCRIPT.md"
OUTPUT = ROOT / "output" / "presentation" / "Huntrix_Presentation_Script.docx"

BLUE = "1F5A7A"
TEAL = "2A8C82"
GOLD = "D59B32"
RED = "B5483A"
INK = "0F172A"
MUTED = "526174"
LINE = "D8E0EA"
PALE_BLUE = "E7EEF4"
PALE_TEAL = "DDF2EE"
PALE_GOLD = "FBF0D8"
WHITE = "FFFFFF"


def set_run_font(run, name="Calibri", size=11, color=INK, bold=False, italic=False):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:ascii"), name)
    rpr.rFonts.set(qn("w:hAnsi"), name)
    rpr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    run.bold = bold
    run.italic = italic
    return run


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_dxa):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


def set_paragraph_border(paragraph, color=LINE, size=6, space=3):
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), str(space))
    bottom.set(qn("w:color"), color)
    p_bdr.append(bottom)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_run_font(paragraph.add_run("Team Huntrix  |  "), size=9, color=MUTED)
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instr, separate, text, end))


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for name, size, color, before, after in (
        ("Heading 1", 17, BLUE, 18, 10),
        ("Heading 2", 14, BLUE, 14, 7),
        ("Heading 3", 12, "1F4D78", 10, 5),
    ):
        style = styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True


def add_header_footer(section):
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    header = section.header
    p = header.paragraphs[0]
    p.text = ""
    set_run_font(p.add_run("TEAM HUNTRIX  |  FINAL PRESENTATION SCRIPT"), size=9, color=BLUE, bold=True)
    set_paragraph_border(p)
    p = section.footer.paragraphs[0]
    p.text = ""
    add_page_number(p)


def configure_page(section):
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.82)
    section.bottom_margin = Inches(0.78)
    section.left_margin = Inches(0.88)
    section.right_margin = Inches(0.88)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)


def add_callout(doc, title, body, fill=PALE_BLUE, accent=BLUE):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    set_run_font(p.add_run(title + "\n"), size=10.5, color=accent, bold=True)
    set_run_font(p.add_run(body), size=10.5, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def parse_source():
    text = SOURCE.read_text(encoding="utf-8")
    allocation_match = re.search(
        r"## Presenter allocation\n\n(.*?)(?=\n## Slide 1:)",
        text,
        flags=re.S,
    )
    allocation_block = allocation_match.group(1) if allocation_match else ""
    allocation = []
    for line in allocation_block.splitlines():
        if not line.startswith("|") or "---" in line or "Presenter" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 3:
            allocation.append(tuple(cells))

    slides = []
    matches = list(re.finditer(r"^## Slide (\d+): (.+)$", text, flags=re.M))
    end_of_slides = text.find("\n## Likely questions")
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else end_of_slides
        block = text[start:end].strip()
        presenter = re.search(r"\*\*Presenter:\*\*\s*(.+)", block)
        target = re.search(r"\*\*Target:\*\*\s*(.+)", block)
        handoff = re.search(r"\*\*Handoff:\*\*\s*(.+)", block)
        body = re.sub(r"\*\*Presenter:\*\*.*\n?", "", block)
        body = re.sub(r"\*\*Target:\*\*.*\n?", "", body)
        body = re.sub(r"\*\*Handoff:\*\*.*\n?", "", body).strip()
        body = re.sub(r"\n{2,}", "\n\n", body)
        slides.append(
            {
                "number": int(match.group(1)),
                "title": match.group(2).strip(),
                "presenter": presenter.group(1).strip() if presenter else "",
                "target": target.group(1).strip() if target else "",
                "script": body,
                "handoff": handoff.group(1).strip() if handoff else "",
            }
        )

    questions = []
    q_match = re.search(r"## Likely questions\n\n(.*?)(?=\n## Source map)", text, flags=re.S)
    if q_match:
        q_block = q_match.group(1)
        q_heads = list(re.finditer(r"^### (.+)$", q_block, flags=re.M))
        for idx, match in enumerate(q_heads):
            start = match.end()
            end = q_heads[idx + 1].start() if idx + 1 < len(q_heads) else len(q_block)
            questions.append((match.group(1).strip(), q_block[start:end].strip()))

    sources = []
    s_match = re.search(r"## Source map\n\n(.*)$", text, flags=re.S)
    if s_match:
        for line in s_match.group(1).splitlines():
            match = re.match(r"(\d+)\.\s+(.+)", line)
            if match:
                sources.append((match.group(1), match.group(2)))
    return allocation, slides, questions, sources


def add_script_paragraph(doc, text):
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    created = []
    for part in paragraphs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(7)
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.keep_together = True
        set_run_font(p.add_run(part), size=11.5, color=INK)
        created.append(p)
    return created


def build():
    allocation, slides, questions, sources = parse_source()
    doc = Document()
    section = doc.sections[0]
    configure_page(section)
    configure_styles(doc)
    add_header_footer(section)

    # Editorial cover for a rehearsal script.
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(92)
    p.paragraph_format.space_after = Pt(18)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("FINAL ROUND • 10-MINUTE TALK"), size=11, color=GOLD, bold=True)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("Team Huntrix\nPresentation Script"), size=28, color=INK, bold=True)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(24)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(
        p.add_run("Evidence-first selective inference for Bengali hallucination detection"),
        size=14,
        color=BLUE,
    )

    add_callout(
        doc,
        "Delivery target",
        "Rehearsed content: 9 minutes 15 seconds across two presenters, with one handoff and 45 seconds of safety before the 10-minute hard stop.",
        PALE_TEAL,
        TEAL,
    )

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(26)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(
        p.add_run("Touhidul Alam Seyam  •  MD. Abtahee Kabir"),
        size=10.5,
        color=INK,
        bold=True,
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("Presenter-ready rehearsal edition  |  23 July 2026"), size=10, color=MUTED)

    doc.add_page_break()
    doc.add_heading("Run of show", level=1)
    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    set_table_geometry(table, [2900, 1700, 1700, 3060])
    set_repeat_header(table.rows[0])
    for cell, text in zip(table.rows[0].cells, ("Presenter", "Slides", "Time", "Primary arc")):
        cell.text = text
        set_cell_shading(cell, PALE_BLUE)
        for run in cell.paragraphs[0].runs:
            set_run_font(run, size=10.5, color=BLUE, bold=True)
    arcs = [
        "Problem, router, solvers, claim verification",
        "Judge, evidence, experiments, Phase 2, close",
    ]
    for idx, row_data in enumerate(allocation):
        row = table.add_row()
        prevent_row_split(row)
        for cell, text in zip(row.cells, (*row_data, arcs[idx] if idx < len(arcs) else "")):
            set_run_font(cell.paragraphs[0].add_run(text), size=10, color=INK, bold=(cell is row.cells[0]))

    add_callout(
        doc,
        "Speaking rule",
        "Do not race the diagrams. State the claim first, then point to the mechanism or number on screen. Finish the final sentence before handing off.",
        PALE_GOLD,
        GOLD,
    )
    add_callout(
        doc,
        "Technical wording to keep exact",
        "The frozen notebook uses one local llama.cpp server whose model is tensor-split equally across two T4 GPUs.",
        PALE_BLUE,
        BLUE,
    )

    previous_presenter = None
    for slide in slides:
        presenter_changed = slide["presenter"] != previous_presenter
        if presenter_changed:
            doc.add_page_break()
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(3)
            set_run_font(p.add_run("PRESENTER BLOCK"), size=10, color=GOLD, bold=True)
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(10)
            set_run_font(p.add_run(slide["presenter"]), size=22, color=INK, bold=True)
            previous_presenter = slide["presenter"]

        # Split the two presenter blocks at stable boundaries so both DOCX and
        # exported PDF retain predictable rehearsal pages.
        if slide["number"] in {4, 7, 11, 14}:
            doc.add_page_break()
        heading = doc.add_heading(
            f"Slide {slide['number']}: {slide['title']}",
            level=2,
        )
        heading.paragraph_format.keep_with_next = True
        meta = doc.add_paragraph()
        meta.paragraph_format.space_after = Pt(6)
        meta.paragraph_format.keep_with_next = True
        set_run_font(meta.add_run(f"Presenter  {slide['presenter']}"), size=10, color=TEAL, bold=True)
        set_run_font(meta.add_run(f"     Target  {slide['target']}"), size=10, color=GOLD, bold=True)

        created = add_script_paragraph(doc, slide["script"])
        if created:
            heading.paragraph_format.keep_with_next = True
            meta.paragraph_format.keep_with_next = True
        if slide["handoff"]:
            add_callout(doc, "Handoff", slide["handoff"], PALE_TEAL, TEAL)

    section = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_page(section)
    add_header_footer(section)
    doc.add_heading("Likely questions after the talk", level=1)
    for idx, (question, answer) in enumerate(questions, 1):
        q = doc.add_paragraph()
        q.paragraph_format.space_before = Pt(10)
        q.paragraph_format.space_after = Pt(4)
        q.paragraph_format.keep_with_next = True
        set_run_font(q.add_run(f"Q{idx}. {question}"), size=12, color=INK, bold=True)
        a = doc.add_paragraph()
        a.paragraph_format.space_after = Pt(7)
        a.paragraph_format.keep_together = True
        set_run_font(a.add_run("Suggested answer: "), size=10.5, color=TEAL, bold=True)
        set_run_font(a.add_run(answer), size=10.5, color=INK)

    section = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_page(section)
    add_header_footer(section)
    doc.add_heading("Source map", level=1)
    add_callout(
        doc,
        "Use",
        "These numbers match the compact citations shown in the deck. Full bibliographic details remain in the technical report.",
        PALE_BLUE,
        BLUE,
    )
    for number, source in sources:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.paragraph_format.space_after = Pt(7)
        p.paragraph_format.keep_together = True
        set_run_font(p.add_run(f"{number}. "), size=10.5, color=GOLD, bold=True)
        set_run_font(p.add_run(source), size=10.5, color=INK)

    add_callout(
        doc,
        "Final delivery cue",
        "Finish on the open questions and runtime evidence, pause, look at the judges, and say: “We are ready for your questions.”",
        PALE_TEAL,
        TEAL,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
