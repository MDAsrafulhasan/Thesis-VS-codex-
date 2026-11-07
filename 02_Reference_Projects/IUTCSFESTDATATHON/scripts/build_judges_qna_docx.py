from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "qna" / "Huntrix_Judges_QA_Preparation.docx"

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
PALE_RED = "F7E5E1"
WHITE = "FFFFFF"


def set_run_font(run, name="Calibri", size=11, color=INK, bold=False, italic=False):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
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
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
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


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_paragraph_border(paragraph, color=BLUE, size=10, space=4):
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
    fld_char = OxmlElement("w:fldChar")
    fld_char.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.extend((fld_char, instr, fld_sep, text, fld_end))


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
        ("Heading 1", 16, BLUE, 18, 10),
        ("Heading 2", 13, BLUE, 14, 7),
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

    for style_name, fill, color in (
        ("Answer Label", PALE_TEAL, TEAL),
        ("Evidence Label", PALE_BLUE, BLUE),
        ("Caution Label", PALE_RED, RED),
    ):
        if style_name not in styles:
            style = styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)
        else:
            style = styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(10.5)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(3)
        style.paragraph_format.space_after = Pt(3)
        style.paragraph_format.keep_with_next = True

    if "Question" not in styles:
        question = styles.add_style("Question", WD_STYLE_TYPE.PARAGRAPH)
    else:
        question = styles["Question"]
    question.font.name = "Calibri"
    question.font.size = Pt(12)
    question.font.bold = True
    question.font.color.rgb = RGBColor.from_string(INK)
    question.paragraph_format.space_before = Pt(12)
    question.paragraph_format.space_after = Pt(4)
    question.paragraph_format.keep_with_next = True


def add_header_footer(doc):
    section = doc.sections[0]
    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_run_font(p.add_run("TEAM HUNTRIX  |  JUDGE Q&A HANDBOOK"), size=9, color=BLUE, bold=True)
    set_paragraph_border(p, color=LINE, size=6, space=3)
    footer = section.footer
    add_page_number(footer.paragraphs[0])


def add_callout(doc, title, body, fill=PALE_BLUE, accent=BLUE):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    set_run_font(p.add_run(title + "\n"), size=11, color=accent, bold=True)
    set_run_font(p.add_run(body), size=10.5, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Inches(0.38)
        p.paragraph_format.first_line_indent = Inches(-0.19)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.25
        set_run_font(p.add_run(item), size=11, color=INK)


def add_numbered(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.left_indent = Inches(0.38)
        p.paragraph_format.first_line_indent = Inches(-0.19)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.25
        set_run_font(p.add_run(item), size=11, color=INK)


def add_labelled_paragraph(doc, label, text, label_color=TEAL):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_together = True
    set_run_font(p.add_run(label + " "), size=10.5, color=label_color, bold=True)
    set_run_font(p.add_run(text), size=10.5, color=INK)
    return p


def add_question(doc, number, question, answer, pressed=None, evidence=None, caution=None):
    p = doc.add_paragraph(style="Question")
    set_run_font(p.add_run(f"Q{number}. {question}"), size=12, color=INK, bold=True)
    card = [p]
    card.append(add_labelled_paragraph(doc, "Short answer:", answer, TEAL))
    if pressed:
        card.append(add_labelled_paragraph(doc, "If they press:", pressed, BLUE))
    if evidence:
        card.append(add_labelled_paragraph(doc, "Evidence to cite:", evidence, GOLD))
    if caution:
        card.append(add_labelled_paragraph(doc, "Do not claim:", caution, RED))
    for paragraph in card[:-1]:
        paragraph.paragraph_format.keep_with_next = True
    for paragraph in card:
        paragraph.paragraph_format.keep_together = True


def add_quick_facts(doc):
    doc.add_heading("Quick facts to memorize", level=1)
    rows = [
        ("Task", "Binary Bengali hallucination detection: 0 = hallucinated, 1 = faithful"),
        ("Official score language", "Competition page reports macro-F1; the rules also emphasize label-0 F1. We tracked both."),
        ("Released labels", "299 rows"),
        ("Phase 1 test", "2,516 rows"),
        ("Final Phase 1 public", "0.967, public rank #5"),
        ("Final Phase 1 private", "0.982, private rank #1"),
        ("Released-set validation", "Macro-F1 0.993264; accuracy 0.993311"),
        ("Deterministic released subset", "225 rows; macro-F1 and label-0 F1 both 1.000000"),
        ("Phase 1 v91 routing", "158 unresolved rows sent to Qwen; 2,358 resolved before the judge"),
        ("Model", "Qwen3.6-27B Q4_K_P, 16.332 GiB GGUF"),
        ("Serving", "One local llama.cpp server, model tensor-split 1,1 across two T4 GPUs"),
        ("Generation", "Temperature 0, thinking disabled, stable ID/view seeds, grammar restricted to 0 or 1"),
        ("Phase 1 reproducibility", "Three full T4×2 runs were byte-identical to scored submission 54836339"),
        ("Frozen Phase 1 runtime", "440.87–593.43 seconds; organizer form conservatively reported 0.17 h / 612 s"),
        ("Phase 2 held-out run", "5,000 rows, T4×2, 3:52:01.91 end to end"),
        ("Phase 2 routing", "207 deterministic (4.14%); 4,793 judged (95.86%)"),
        ("Phase 2 official result", "0.90144, private rank #7 of 32 teams"),
        ("Training", "No fine-tuning and no team-trained checkpoint"),
        ("Internet/API", "Offline. The HTTP endpoint is loopback 127.0.0.1 to the local model server, not an external API."),
    ]
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    set_table_geometry(table, [2600, 6760])
    headers = table.rows[0].cells
    headers[0].text = "Fact"
    headers[1].text = "Exact value or wording"
    set_repeat_table_header(table.rows[0])
    for cell in headers:
        set_cell_shading(cell, PALE_BLUE)
        for run in cell.paragraphs[0].runs:
            set_run_font(run, size=10.5, color=BLUE, bold=True)
    for idx, (label, value) in enumerate(rows):
        row = table.add_row()
        prevent_row_split(row)
        cells = row.cells
        if idx % 2:
            for cell in cells:
                set_cell_shading(cell, "F7F9FC")
        p0, p1 = cells[0].paragraphs[0], cells[1].paragraphs[0]
        set_run_font(p0.add_run(label), size=10, color=INK, bold=True)
        set_run_font(p1.add_run(value), size=10, color=INK)
    doc.add_paragraph()


SECTIONS = [
    (
        "1. Task framing and research contribution",
        [
            (
                "What problem are you solving?",
                "We decide whether a Bengali model response is hallucinated or faithful, sometimes against a supplied context and sometimes from general or culturally situated knowledge.",
                "The same binary label hides several reasoning tasks: context entailment, numbers, options, morphology, idioms, law, history, literature, and open-domain facts. Our method first identifies the evidence standard appropriate to the row.",
                "Competition task definition; Slide 2; report Introduction.",
                "Do not reduce the task to generic text classification.",
            ),
            (
                "What is the central idea of your method?",
                "Use exact methods for exact cases and abstain to a language-model judge for the uncertain remainder.",
                "Every deterministic stage returns 0, 1, or null. Null is a deliberate abstention, not an error. Only rows that survive the task and evidence gates reach Qwen.",
                "Slides 3–4; report Equation 1 and Figure 1.",
                "Do not say the rules always outperform the LLM.",
            ),
            (
                "What is novel here?",
                "The contribution is the selective combination of task routing, relation-typed evidence, claim-level context verification, explicit abstention, and a reproducible offline fallback.",
                "The individual ingredients are known. The research contribution is the decision contract and admission discipline: a rule may decide only when its entity, predicate, value, provenance, and conflict checks are explicit.",
                "Slides 3, 7, and 9; report Method.",
                "Do not claim that routing, RAG, or abstention was invented by us.",
            ),
            (
                "Why is this better than one large prompt?",
                "One prompt applies one evidence policy to heterogeneous tasks. The router lets a numeric contradiction, an idiom meaning, and a historical fact use different proof standards.",
                "Our uniform Qwen version reached 0.797 publicly. Introducing source- and task-conditioned routing at v6 coincided with the largest observed step, to 0.903.",
                "Slide 10; EXPERIMENT.log v6.",
                "The version comparison is cumulative, not a controlled one-factor ablation.",
            ),
            (
                "Why call the system evidence-first?",
                "A deterministic decision is admitted only when the evidence type and requested relation are explicit; otherwise the model judges the row.",
                "In Phase 2 most rows reached the model because exact evidence coverage collapsed. The name describes the admission policy, not a promise that evidence will cover every distribution.",
                "Slides 9 and 14.",
                "Do not imply that 95.86% of Phase 2 was evidence-resolved.",
            ),
            (
                "What is the main scientific finding?",
                "Matching the decision method to the task mattered more than repeatedly changing the judge prompt.",
                "The strongest evidence is the progression from uniform judging to selective routing, the regression caused by weakly typed fact edits, and the safe abstention behavior under Phase 2 source shift.",
                "Slides 10–14; DECISION.log and EXPERIMENT.log.",
                "Do not present leaderboard movement as causal proof.",
            ),
            (
                "Why is Bengali hallucination detection different?",
                "Bengali tasks combine multilingual reasoning with culturally situated facts, morphology, idioms, and local civic or legal knowledge that globally dominant models may not represent reliably.",
                "The competition explicitly includes cultural-distance bands and Phase 2 sources such as Banglapedia, government services, Bangladesh law, NCTB textbooks, newspapers, and literature.",
                "Problem.md; organizer Phase 2 preview.",
                "Avoid broad claims that Bengali models are generally unreliable without measured evidence.",
            ),
            (
                "What exactly does label 0 mean?",
                "Label 0 means the response is hallucinated, unsupported, factually wrong, or answers the wrong requested relation. Label 1 means faithful.",
                "A fact can be true but still receive 0 if it does not answer the prompt—for example, returning a film title when the question asks for its release year.",
                "Released sample 4; Slide 6.",
                "Do not accidentally reverse the labels during Q&A.",
            ),
        ],
    ),
    (
        "2. Router, solvers, and relation verification",
        [
            (
                "How does the router work?",
                "It normalizes the row, applies structured and source-specific stages in a fixed priority, records accepted decisions, and marks everything else uncertain for the judge.",
                "The order protects context contradictions and exact relations from being overridden by weaker similarity. Every decision is stored with stage and trace information.",
                "scripts/phase2_pipeline.py; Slides 4 and 9.",
                "Do not describe the router as a learned classifier; the frozen notebook does not load one.",
            ),
            (
                "What does abstention mean operationally?",
                "A stage returns null when it lacks enough evidence to emit 0 or 1. The row remains in the unresolved queue and proceeds to the next stage or final judge.",
                "This prevents an approximate match from becoming a confident label and is the main safety valve under domain shift.",
                "Slides 3–4.",
                "Do not call abstention a third output label; the final CSV is still binary.",
            ),
            (
                "What structured solvers did you use?",
                "Separate solvers handle numbers, dates, ranges, multiple-choice options, Bengali morphology, and idiom or lexical relations.",
                "They share the same contract: parse the requested entity and predicate, normalize the candidate value, compare under task-specific equivalence, and abstain on missing or conflicting fields.",
                "Slide 5; structured override tests.",
                "Do not imply every row in those categories is automatically resolved.",
            ),
            (
                "How do you prevent answering the wrong relation?",
                "A candidate source must match the entity and the requested predicate before its value is considered.",
                "We explicitly distinguish start versus end, birth versus death, writing versus publication, construction start versus completion, and river source versus confluence. Unknown or mismatched relations are neutral or abstain.",
                "Slide 9; relation signatures in DECISION.log.",
                "Semantic similarity alone is not sufficient evidence.",
            ),
            (
                "Explain the Runway example.",
                "The context contains both the film title Runway and its release year 2010. The prompt asks for the year, but the response gives the title. The entity is related, but the predicate and response type are wrong, so the label is 0.",
                "This example demonstrates why retrieval overlap can be high while the answer is still invalid.",
                "Competition-released sample 4; Slide 6.",
                "Do not call the response factually fabricated; it is relation-inappropriate.",
            ),
            (
                "How are context rows checked?",
                "The response is reduced to atomic claims and compared with the most relevant context clause using anchors such as entity, number, negation, and relation.",
                "Any contradiction yields 0. Label 1 requires support for all factual claims and no unsupported addition. Intermediate cases abstain.",
                "Slide 7; report Figure 2a.",
                "Do not say global word overlap decides context rows.",
            ),
            (
                "What happens when the response is partly correct?",
                "If any atomic factual claim contradicts the context or reliable expected value, the response is labeled 0.",
                "For context rows, a mostly copied answer with one changed year, person, quantity, or polarity is still a hallucination. If support is incomplete rather than contradictory, the deterministic stage abstains.",
                "Claim aggregation rule in Slide 7.",
                "Do not average claim correctness into a soft score in the frozen system.",
            ),
            (
                "How do numeric rules work?",
                "They normalize Bengali and Arabic digits, compound Bengali number words, units, dates, ranges, and requested relations before comparing values.",
                "The v91 relation-bound rules require exact numeric multisets for ranges or a unique relation-aligned clause for contradictions. Superlatives and ambiguous multi-clause cases abstain.",
                "DECISION.log: relation-bound numeric signatures; 19/19 released calibration.",
                "Do not say a number appearing anywhere in the context is enough.",
            ),
            (
                "How do idiom rules avoid literal/figurative confusion?",
                "The prompt is first typed as asking for literal or figurative meaning, then matched against the corresponding field of the idiom record.",
                "Duplicate and alias records must agree. Ambiguous or conflicting records abstain. This was calibrated on a clean exact-headword subset before promotion.",
                "Typed idiom matcher; Slide 5.",
                "Do not flatten all idiom aliases into a single dictionary value.",
            ),
            (
                "How are conflicts handled?",
                "If equally trusted records disagree about the same entity and predicate, the deterministic stage abstains.",
                "A stronger exact source can resolve a weaker candidate, but similarity or an unrelated relation cannot override an admitted exact decision.",
                "Slide 9 conflict gate.",
                "Do not hide disagreement by majority vote over dependent sources.",
            ),
            (
                "Why does stage order matter?",
                "A broad but weak rule must not override a narrow high-confidence rule. Priority encodes evidence trust and task specificity.",
                "Structured parsing and exact lineage run before fuzzy proposal stages. Context contradictions receive precedence over external plausibility.",
                "phase2_pipeline.py stage order.",
                "Do not claim the order was arbitrary.",
            ),
            (
                "Does the system produce explanations?",
                "It produces machine-readable decision traces rather than natural-language rationales.",
                "For deterministic rows the trace records route, source family, matched relation, value comparison, and verdict. For judged rows it records each fixed prompt view and binary output.",
                "router_trace.json and judge_trace.json design.",
                "Do not equate a trace with a faithful chain of thought.",
            ),
        ],
    ),
    (
        "3. Data, evidence, retrieval, and leakage concerns",
        [
            (
                "Did you use RAG?",
                "Not as an end-to-end generative RAG system. Retrieval proposed candidates, while typed verification decided whether any candidate was admissible.",
                "Broad retrieval repeatedly found the right topic but wrong relation. We therefore made fuzzy and hybrid retrieval proposal-only unless independent relation-aligned evidence agreed.",
                "Slide 11; hybrid retrieval audit in DECISION.log.",
                "Do not say retrieved top-1 text directly determines the label.",
            ),
            (
                "What external data did you use?",
                "Public Bengali and multilingual QA, lexical, idiom, Wikipedia, textbook, news, legal, history, literature, exam, and structured-fact resources.",
                "The public repository contains DATA_SOURCES.md, CITATIONS.md, and THIRD_PARTY_NOTICES.md with links and license status. Sources with unclear redistribution terms were used for audit or retrieval rather than copied indiscriminately.",
                "Public submission repository data register; report External Resources.",
                "Do not claim every source is equally authoritative or equally licensed.",
            ),
            (
                "Did exact-source matching amount to test-label leakage?",
                "No hidden labels were available or used. The system compares input rows against public or competition-permitted source records and recomputes a label from entity, predicate, and value.",
                "We did not store a Phase 1 prediction vector or manually assign test-row labels. Exact lineage is external evidence, not a ground-truth label. We disclosed the method to organizers.",
                "METHODOLOGY_DISCLOSURE.md; organizer clarification; freeze manifest.",
                "Do not say there was no manual research at all; public facts were researched and cited.",
            ),
            (
                "Did you manually label test rows?",
                "No. We did not create an ID-to-label map or use unreleased ground truth.",
                "Some reusable factual records were researched from public sources. At inference, the rule still checks the row’s requested relation and candidate response. A record is not itself a test label.",
                "Public README methodology disclosure.",
                "Do not blur the distinction between researching a fact and assigning a test label.",
            ),
            (
                "How did you discover that some questions matched public datasets?",
                "By normalizing prompt, context, and answer fields and auditing exact or high-precision lineage against public datasets.",
                "The discovery came from dataset/source comparison, not from leaderboard feedback. Near matches were subjected to relation signatures and conservative thresholds.",
                "Answer-bank construction and lineage audit logs.",
                "Do not imply we inferred matches from private leaderboard behavior.",
            ),
            (
                "Why require two sources for free-form facts?",
                "Independent corroboration reduces the chance that one noisy page or copied error becomes a deterministic label.",
                "The two-source rule applies to free-form verified facts. Exact benchmark lineage, supplied context, idiom records, or legislation use source-specific admission rules instead.",
                "Report Evidence Admission and Conflict Resolution.",
                "Do not count mirrors or duplicates as independent evidence.",
            ),
            (
                "How do you assess source independence?",
                "By source family or repository rather than raw URL count, and by checking that the records support the same entity-relation claim.",
                "The frozen rule excludes relation mismatches from consensus and treats unknown relations as neutral rather than supportive.",
                "Hybrid retrieval audit; typed evidence schema.",
                "Do not claim perfect independence detection; copied web content remains a limitation.",
            ),
            (
                "How do you handle stale or time-sensitive facts?",
                "The frozen system is conservative: conflicting or ambiguous evidence abstains to the judge.",
                "A production version should add timestamps, source versioning, and explicit validity intervals, especially for government services, fees, offices, and recent news.",
                "Report Limitations and Phase 2 source table.",
                "Do not claim the evidence base is permanently current.",
            ),
            (
                "Why not use the internet during inference?",
                "Phase 2 required offline Kaggle execution. All model and evidence assets were attached and version-pinned before the run.",
                "Offline execution also makes the result reproducible and prevents live pages from changing between runs.",
                "Competition rules; frozen notebook.",
                "Do not suggest live scraping or external search occurs in the submitted notebook.",
            ),
            (
                "What is inside the runtime dataset?",
                "The deterministic router, source answer banks, lexical and idiom evidence, released benchmark lineage, verified facts, legal evidence, scripts, manifests, and licenses.",
                "The model and CUDA llama.cpp executable are attached separately. The runtime dataset does not contain a saved Phase 1 prediction vector.",
                "Public README and freeze manifest.",
                "Do not call the runtime dataset a trained checkpoint.",
            ),
            (
                "What prevents a fuzzy match from making a wrong decision?",
                "High lexical gates, exact numeric consistency, relation preservation, and—where required—independent consensus.",
                "If those conditions are not met, fuzzy retrieval remains a proposal and the row is left unresolved.",
                "Fuzzy source override tests; relation-safe retrieval audit.",
                "Do not quote one threshold without explaining the relation guard.",
            ),
            (
                "Why not simply retrieve Wikipedia for every question?",
                "Many prompts concern idioms, morphology, legal clauses, exam options, or supplied contexts where Wikipedia is not the correct evidence source.",
                "Even for factual questions, topically relevant text can contain many neighboring relations. Retrieval must be conditioned on what the question actually asks.",
                "Task taxonomy and Runway example.",
                "Do not treat a single corpus as universal ground truth.",
            ),
        ],
    ),
    (
        "4. Qwen judge and inference design",
        [
            (
                "Why Qwen3.6-27B?",
                "It offered strong multilingual Bengali reasoning while remaining feasible as a 16.332 GiB quantized open-weight model on Kaggle T4×2.",
                "The alternative was not chosen on size alone. We tested smaller and different judges, but the larger gain came from routing. Qwen became the bounded fallback rather than the whole pipeline.",
                "Slide 8; experiment progression.",
                "Do not claim Qwen is universally best for Bengali.",
            ),
            (
                "Did you fine-tune Qwen?",
                "No. We used a public quantized base model with fixed prompts and constrained decoding.",
                "There is therefore no team-trained checkpoint or training notebook to reproduce. The model path and version are pinned in the inference notebook.",
                "Public README; frozen notebook.",
                "Do not imply prompt design is fine-tuning.",
            ),
            (
                "How is the model deployed on two T4 GPUs?",
                "One local llama.cpp server loads the model with layer split mode and tensor split 1,1 across both T4 GPUs.",
                "The exact command uses 99 GPU layers, 8,192 context size, flash attention, Q8 key/value cache, and parallelism 1. Requests are processed deterministically in sorted order.",
                "phase2_v91_inference.py server_command.",
                "Do not say there are two independent servers or two model replicas.",
            ),
            (
                "Are you calling an API?",
                "Only a local loopback HTTP interface at 127.0.0.1 to the llama.cpp process inside the same Kaggle notebook.",
                "There is no internet connection, paid service, external endpoint, token, or closed model. HTTP is merely the local process interface.",
                "Frozen notebook health check and /v1/chat/completions call.",
                "Do not answer with an unqualified “yes”; distinguish local interface from prohibited external API use.",
            ),
            (
                "What prompts does the judge use?",
                "Context rows use one grounding view. No-context rows use up to three fixed views: strict exam checking, a cautious hallucination check, and a general factual check.",
                "A no-context row receives label 1 only if all executed views are positive. The first 0 stops further views, saving runtime.",
                "gguf_selective_judge.py; Slide 8.",
                "Do not say three independent models vote; these are views of one model.",
            ),
            (
                "Why unanimous positive voting?",
                "Faithful label 1 requires stronger support, while any clear contradiction is enough for hallucinated label 0.",
                "The policy matches the task’s emphasis on catching hallucinations and permits early exit. It is intentionally conservative and should be evaluated with per-class metrics.",
                "Judge consensus function; Slide 8.",
                "Do not claim the policy is unbiased between classes.",
            ),
            (
                "Why disable thinking?",
                "The task needs a stable binary decision, and long reasoning would increase latency and variability without a way to validate the chain of thought.",
                "Thinking is disabled in both the server and request, temperature is 0, and a grammar restricts output to 0 or 1.",
                "Frozen server command and call_server payload.",
                "Do not claim that the model performs no internal computation.",
            ),
            (
                "Why max_tokens 2 if the output is one token?",
                "The grammar only allows a binary label; the small allowance is an implementation buffer for tokenization while still preventing explanations.",
                "The parser rejects responses without a valid isolated 0 or 1, and the notebook retries failed requests with the same prompt and seed.",
                "gguf_selective_judge.py.",
                "Do not say the model emits explanations that are later discarded.",
            ),
            (
                "How is determinism enforced?",
                "Fixed row order, prompt order, batch boundaries, temperature 0, stable seeds derived from row ID and view, one server slot, and grammar-constrained output.",
                "We then verified determinism empirically: three full Phase 1 runs produced byte-identical CSVs matching the scored submission hash.",
                "Slide 12; freeze manifest.",
                "Seeds alone are not the whole reproducibility story.",
            ),
            (
                "Why only one request worker?",
                "Serial scheduling made request order and server-slot behavior reproducible and still met the nine-hour limit.",
                "The model itself is split across both GPUs. `row-batch-size=4` organizes fixed groups and tracing, but rows are intentionally judged serially with `--parallel 1`.",
                "Notebook command and gguf_selective_judge.py loop.",
                "Do not describe this as four-way concurrent inference.",
            ),
            (
                "Does the judge see the full context?",
                "The deterministic router can inspect the row data, while the LLM prompt includes up to 1,800 characters of cleaned context.",
                "This bound controls runtime and context length. It is also a limitation: answer-bearing evidence late in a long passage can be missed and should be addressed with better clause retrieval in future work.",
                "make_prompt in gguf_selective_judge.py.",
                "Do not claim the LLM always sees an untruncated document.",
            ),
            (
                "What happens if the model server fails?",
                "The notebook fails fast rather than silently writing an invalid submission.",
                "It validates asset paths, checks two T4 GPUs, polls the local health endpoint, retries individual requests, writes partial traces, and always terminates the server in a finally block.",
                "Frozen inference notebook.",
                "Do not claim recovery from every possible Kaggle infrastructure failure.",
            ),
        ],
    ),
    (
        "5. Evaluation, experiments, and scores",
        [
            (
                "What metric did you optimize?",
                "We tracked macro-F1, accuracy, and class-specific performance—especially label 0—because the competition materials emphasize both macro-F1 and the hallucinated class.",
                "When discussing official results, we report the displayed public and private competition scores without trying to resolve inconsistent wording in the rule pages.",
                "Problem.md; report Protocol.",
                "Do not report accuracy alone.",
            ),
            (
                "What were your final Phase 1 scores?",
                "Public macro-F1 was 0.967 at rank 5; private macro-F1 was 0.982 at rank 1.",
                "The public split covered only part of the test set, so the difference does not identify which rows were right or wrong.",
                "Final leaderboard record in EXPERIMENT.log.",
                "Do not infer private labels from the score difference.",
            ),
            (
                "How strong was local validation?",
                "On 299 released labels, macro-F1 was 0.993264 and accuracy was 0.993311.",
                "The deterministic subset covered 225 rows with perfect macro-F1 and label-0 F1. The full cascade made two errors, both documented rather than patched by row ID.",
                "Report Results.",
                "Do not call 299 rows a large or definitive validation set.",
            ),
            (
                "Why not force local F1 to 1.0?",
                "Because the remaining errors exposed inconsistent or relation-malformed controls, and patching them by row would reduce generalization.",
                "Both Qwen and Gemma rejected the problematic answers for defensible semantic reasons. We preserved the general relation rule rather than memorizing exceptions.",
                "DECISION.log bottleneck audit.",
                "Do not imply the released labels were broadly wrong; only specific controls were inconsistent.",
            ),
            (
                "Are your version results ablations?",
                "No. They are cumulative version comparisons that locate design pivots and regressions.",
                "A controlled ablation would change one factor while keeping data, routing, prompts, and evaluation fixed. We did not have enough labels or submission budget to run a complete factorial study.",
                "Slide 10 caption and script.",
                "Do not use causal language such as “this rule alone added 0.058.”",
            ),
            (
                "What was the biggest improvement?",
                "The largest observed public step was from uniform Qwen at 0.797 to v6 at 0.903, when source and structured routing were introduced.",
                "Because v6 contained multiple changes, we describe this as a pivot associated with routing, not a clean isolated effect.",
                "Slide 10; experiment ledger.",
                "Do not claim exact causality.",
            ),
            (
                "What failed?",
                "Prompt voting, a TF-IDF verifier used as a scorer, raw QA retrieval, and plausible but weakly typed fact edits all failed or regressed.",
                "Each failure changed policy: stop prompt churn, keep TF-IDF only as a fallback, make retrieval proposal-only, and require source plus predicate for fact edits.",
                "Slide 11; EXPERIMENT.log.",
                "Do not hide negative results; they strengthen the methodology story.",
            ),
            (
                "Why did v87 regress?",
                "It changed two labels using plausible factual evidence without sufficiently binding that evidence to the requested predicate.",
                "The public score moved from 0.966 to 0.965, so we reverted the edits and tightened the admission rule.",
                "Slide 11; v87 log entry.",
                "Do not claim the leaderboard revealed the exact incorrect rows.",
            ),
            (
                "How did you avoid leaderboard probing?",
                "We used submissions to evaluate complete hypotheses, not systematic label flips designed to recover ground truth.",
                "A rounded aggregate change was treated only as evidence about the candidate pipeline. We maintained a decision log and did not convert public feedback into an ID-to-label map.",
                "Experiment ledger and competition fair-play rule.",
                "Do not describe individual submissions as label queries.",
            ),
            (
                "Why are public and private scores different?",
                "They evaluate different hidden subsets, so sampling and domain composition can produce different aggregate F1.",
                "The private improvement supports the final pipeline at the aggregate level but does not prove a particular rule or row was correct.",
                "Final leaderboard record.",
                "Do not say the private set was easier without evidence.",
            ),
            (
                "How do you know the system reproduced the scored result?",
                "The notebook-generated CSV had the same 2,516 IDs, label counts, and SHA-256 hash as competition submission 54836339.",
                "Three T4×2 runs were byte-identical, including the final public notebook version.",
                "Freeze manifest; Slide 12.",
                "Do not rely only on a rounded leaderboard score as proof of reproduction.",
            ),
            (
                "What would a stronger evaluation look like?",
                "A larger stratified set with row-level labels, domain and cultural-distance breakdowns, controlled ablations, calibration curves, and repeated hardware measurements.",
                "For Phase 2, the highest-value analysis is errors on the 4,793 judged rows versus the 207 deterministic rows once ground truth is available.",
                "Report Limitations; Slide 15.",
                "Do not claim we already performed unavailable Phase 2 error analysis.",
            ),
        ],
    ),
    (
        "6. Phase 2 domain shift and runtime",
        [
            (
                "What happened in the Phase 2 organizer run?",
                "The notebook processed 5,000 held-out rows on T4×2 in 3:52:01.91 and passed final output validation.",
                "The router resolved 207 rows and sent 4,793 to Qwen. The run used 42.97% of the nine-hour budget and left 5:07:58.09 of headroom.",
                "Organizer-executed huntrix-1.ipynb; Slide 13.",
                "Do not imply this runtime result is an accuracy score.",
            ),
            (
                "What was the Phase 2 score?",
                "The official Phase 2 result was 0.90144, private rank #7 of 32 teams.",
                "That aggregate score is verified by the organizer result sheet. Because row-level labels were not released, we cannot honestly attribute errors to individual routes, domains, or examples.",
                "Official Phase 2 result sheet; Slide 13.",
                "Do not infer row-level accuracy, per-route accuracy, or per-domain accuracy from the aggregate score.",
            ),
            (
                "Why did deterministic coverage fall from 93.72% to 4.14%?",
                "Phase 2 deliberately used a broader source distribution, so Phase 1 exact lineage and narrow evidence banks matched far fewer rows.",
                "The router responded as designed: it abstained instead of forcing approximate source matches. That transferred work to the general judge.",
                "Organizer source preview; Slide 14.",
                "Do not describe the coverage collapse as a surprise or hide it.",
            ),
            (
                "Does the coverage collapse prove Phase 1 overfitting?",
                "It shows that the exact-lineage layer was distribution-specific. The full system was designed with a general fallback precisely because that layer would not transfer uniformly.",
                "We should separate two claims: Phase 1 exact evidence was highly effective on its source mix; Phase 2 generalization depended mostly on the judge and context/structured solvers. The official aggregate Phase 2 score was 0.90144, but row-level labels were not released.",
                "Slides 13–15.",
                "Do not deny the phase-specific nature of the evidence layer.",
            ),
            (
                "If Qwen judged 95.86%, why not use Qwen for all Phase 2 rows?",
                "The same frozen notebook had to reproduce Phase 1 and run Phase 2 without dataset-specific branching.",
                "Exact deterministic decisions are cheaper and traceable when available, while abstention ensures unknown sources reach Qwen. Removing them would change the frozen system without evidence of a benefit.",
                "Frozen evaluation contract.",
                "Do not claim 207 deterministic rows necessarily improved Phase 2 accuracy without labels.",
            ),
            (
                "Why was Phase 2 much slower than your 5,000-row projection?",
                "The projection assumed the Phase 1 routing mix, where only 158 of 2,516 rows needed the judge. Phase 2 sent 4,793 of 5,000 rows to the judge.",
                "The per-row model cost dominated, so a source-shifted queue was far more important than raw row count. The actual run still remained under nine hours.",
                "Phase 1 and Phase 2 routing counts.",
                "Do not present the same-mix projection as a guaranteed runtime.",
            ),
            (
                "Could runtime be reduced further?",
                "Yes: safe request parallelism, prefix caching analysis, shorter prompts, calibrated one-view routing, or a faster quantization could reduce the 4,793-row judge cost.",
                "Any optimization must preserve output stability and accuracy. Our frozen system preferred deterministic scheduling because it already had more than five hours of headroom.",
                "Phase 2 runtime breakdown.",
                "Do not promise a speedup without a T4×2 benchmark.",
            ),
            (
                "What sources caused the shift?",
                "The announced mix included Common Crawl, recent newspapers, Bengali Wikipedia, Banglapedia, government services, Bangladesh legislation, NCTB textbooks, and literature.",
                "The held-out notebook itself reveals routing coverage, not source labels for every error. We cannot attribute accuracy or failures by domain without organizer annotations.",
                "Organizer Phase 2 preview.",
                "Do not infer source-specific scores from the runtime trace.",
            ),
            (
                "How does the pipeline handle unseen sources?",
                "It skips source-lineage decisions unless exact admission conditions are satisfied, while context and structured checks can still run. The remainder reaches the fixed judge.",
                "Unknown is treated as insufficient evidence, not as hallucinated or faithful by default.",
                "Slide 14 routing policy.",
                "Do not say unseen automatically means label 0.",
            ),
            (
                "What did final output validation check?",
                "Exactly 5,000 ordered IDs, two columns named id and label, no missing values, and binary integer labels.",
                "The notebook preserves input IDs rather than reconstructing them from row count.",
                "Final validation cell in organizer run.",
                "Do not use label balance as a correctness check.",
            ),
        ],
    ),
    (
        "7. Reproducibility, packaging, and compliance",
        [
            (
                "Can anyone click Run All on the Kaggle notebook?",
                "Yes, provided the three pinned public inputs are attached: the runtime dataset, Qwen model version, and llama.cpp CUDA dataset, with T4×2 selected and internet disabled.",
                "The notebook checks exact asset paths and hardware before inference. Missing or wrong versions fail with a clear error rather than silently substituting files.",
                "Public README and RUNBOOK.md.",
                "Do not say it works without the required attached assets.",
            ),
            (
                "Why is the test path hard-coded?",
                "The organizers required the literal competition path so they could replace the mounted file for Phase 2 without notebook changes.",
                "The notebook does not search directories, construct filenames dynamically, or branch on row count.",
                "Frozen TEST_PATH cell.",
                "Do not tell judges to edit the path for Phase 2.",
            ),
            (
                "What exactly is frozen?",
                "Notebook version, model version, runtime dataset version, CUDA executable version, commands, hashes, prompts, and output contract.",
                "The freeze manifest provides immutable references and expected hashes. Evaluation artifacts were not modified after the deadline.",
                "FREEZE_MANIFEST.json and immutable GitHub commit.",
                "Do not point to a moving branch as the authoritative frozen artifact.",
            ),
            (
                "Why pin llama.cpp?",
                "Inference kernels, chat templates, grammar behavior, and GPU memory use can change across runtime versions.",
                "Pinning build b9637 and its binary hash reduces a major source of silent drift.",
                "Public CUDA dataset and manifest.",
                "Do not assume any recent llama.cpp binary reproduces the result.",
            ),
            (
                "Why were repeated runs needed?",
                "A fixed seed does not guarantee identical end-to-end output when scheduling, kernels, model servers, and file ordering can vary.",
                "Byte-level comparison of complete CSVs tests the actual artifact that is scored.",
                "Slide 12; three-run reproduction.",
                "Do not treat one successful run as determinism proof.",
            ),
            (
                "Did the notebook contain saved predictions?",
                "No. It reads the raw input, initializes unresolved rows conservatively, recomputes router decisions, judges every unresolved row, and writes a new CSV.",
                "The runtime manifest explicitly excludes a Phase 1 prediction vector, and the same pipeline accepted the 5,000-row replacement dataset.",
                "Package manifest and Phase 2 run.",
                "Do not confuse answer banks with a saved label vector.",
            ),
            (
                "How do you know IDs are not hard-coded?",
                "The notebook reads IDs from the input and validates uniqueness and output order. The organizer’s 5,000-row replacement completed successfully.",
                "Decision traces store observed IDs for audit, but labels are recomputed from row content and evidence.",
                "Input/output validation cells.",
                "Do not say traces are used as future predictions.",
            ),
            (
                "Why require exactly T4×2?",
                "That is the environment we validated and the notebook explicitly checks. The model is configured for an equal split across two T4 GPUs.",
                "Supporting P100 or a different GPU layout would require new memory and runtime validation, which was outside the frozen submission.",
                "Hardware check in notebook.",
                "Do not claim hardware portability that was not tested.",
            ),
            (
                "What licenses apply?",
                "Original Huntrix code is MIT; third-party model, datasets, and runtime components retain their own licenses and notices.",
                "The public package includes third-party notices and license files. Sources with uncertain redistribution terms were not indiscriminately republished.",
                "LICENSE, THIRD_PARTY_NOTICES.md, DATA_SOURCES.md.",
                "Do not say all external data is MIT.",
            ),
            (
                "Did you obey the no-external-API rule?",
                "Yes. The submitted inference runs offline with open weights and local attached assets.",
                "The loopback llama.cpp endpoint is inside the notebook process environment. No paid API, Hugging Face token, internet request, or proprietary service is used.",
                "Notebook internet disabled; local endpoint.",
                "Do not let the word “API” remain ambiguous.",
            ),
            (
                "Why is there no training notebook?",
                "Because the final solution did not train or fine-tune a model.",
                "The deterministic assets are constructed data and code, while Qwen is a public pinned base model. The inference notebook and data-source documentation are the relevant reproducibility artifacts.",
                "Public README.",
                "Do not invent a training process for formality.",
            ),
            (
                "How was the package checked before submission?",
                "Automated tests, asset existence checks, manifest hashing, output schema validation, repeated Kaggle runs, and byte comparison with the scored CSV.",
                "The full suite had 249 passing tests at freeze, including 225 deterministic controls.",
                "Slide 12 and freeze record.",
                "Do not claim tests prove accuracy on unseen domains.",
            ),
        ],
    ),
    (
        "8. Limitations, ethics, and future work",
        [
            (
                "What is the biggest limitation?",
                "Only 299 labeled development rows were available, and Phase 2 provided an aggregate score without row-level labels.",
                "That prevents reliable per-domain calibration, controlled ablations, or row-level analysis of the 4,793 judged cases.",
                "Slide 15; report Limitations.",
                "Do not use leaderboard aggregates as a substitute for labeled error analysis.",
            ),
            (
                "Is the system suitable for government or legal deployment?",
                "Not without human escalation, continuously versioned evidence, and domain-specific evaluation.",
                "The competition pipeline demonstrates detection and reproducible inference, but civic and legal facts can be time-sensitive, contested, or high consequence.",
                "Problem framing and report Limitations.",
                "Do not present a competition model as production assurance.",
            ),
            (
                "Can the evidence base contain bias or errors?",
                "Yes. Public corpora can carry annotation errors, copied misinformation, historical framing bias, and uneven cultural coverage.",
                "Typed predicates, provenance, conflict abstention, and source-specific rules reduce risk but do not eliminate it.",
                "Evidence admission design.",
                "Do not call public data ground truth by default.",
            ),
            (
                "What would you change first with Phase 2 labels?",
                "Partition errors by route, context status, source family, and model view, then calibrate the judged tail.",
                "The first comparison should be 207 deterministic versus 4,793 judged rows, followed by errors caused by context truncation, relation swaps, stale facts, and class bias.",
                "Slide 15 next analyses.",
                "Do not start by adding more rules before knowing the error distribution.",
            ),
            (
                "Would you fine-tune in the next version?",
                "Possibly, but only with a larger, source-diverse labeled set and held-out domain splits.",
                "A useful target would be calibration or reranking of the uncertain tail, not memorization of 299 examples. Any fine-tune would need a separate checkpoint, training notebook, and T4×2 benchmark.",
                "Training decision rationale.",
                "Do not say fine-tuning is inherently bad.",
            ),
            (
                "What ablations should be run next?",
                "Remove each major route family in turn, compare one-view versus three-view judging, test predicate gates, and evaluate evidence-only, judge-only, and cascade variants on the same labeled split.",
                "Report macro-F1, label-0 F1, coverage, latency, and calibration separately for context and no-context rows.",
                "Experiment limitations.",
                "Do not reuse the public leaderboard as the ablation set.",
            ),
            (
                "How would you improve long-context handling?",
                "Retrieve relation-aligned clauses from the full supplied context before prompting the judge.",
                "The selector should be validated to preserve negation, entities, numbers, and the requested predicate, rather than taking the first 1,800 characters.",
                "Current context truncation in judge prompt.",
                "Do not simply increase context length without a runtime benchmark.",
            ),
            (
                "How would you improve speed?",
                "Calibrate safe parallel requests, cache common prompt prefixes, reduce redundant views, or test a faster model or quantization.",
                "The objective is not speed alone; every change must preserve the frozen output contract and be benchmarked on Kaggle T4×2.",
                "Phase 2 judge dominated 13,747.50 seconds.",
                "Do not promise linear speedup from two GPUs; they currently hold one split model.",
            ),
            (
                "What result would falsify your hypothesis?",
                "If a uniformly prompted judge matched or exceeded the cascade across source-shifted labeled sets at equal runtime, the selective-routing benefit would not hold.",
                "Likewise, if relation gates reduced recall without reducing relation-swap errors, their added complexity would not be justified.",
                "Research hypothesis from Slide 1.",
                "Do not make the hypothesis unfalsifiable.",
            ),
            (
                "What should the audience remember?",
                "A hallucination detector should know when its evidence answers the requested relation—and know when it does not know.",
                "Our strongest engineering result is that this policy reproduced Phase 1 exactly and remained executable under a severe Phase 2 coverage shift.",
                "Slides 6, 12, and 14.",
                "Do not replace this with a rank-focused closing.",
            ),
        ],
    ),
]


RAPID_FIRE = [
    ("How many rows did Qwen judge in Phase 1?", "158 unique rows. Some no-context rows required multiple prompt views."),
    ("How many Phase 1 rows were deterministic?", "2,358 of 2,516, or approximately 93.72%."),
    ("How many Phase 2 rows were deterministic?", "207 of 5,000, or 4.14%."),
    ("How many Phase 2 rows went to Qwen?", "4,793, or 95.86%."),
    ("What was Phase 2 throughput?", "Approximately 0.349 judged rows per second for the judge stage."),
    ("How long did routing take in Phase 2?", "73.1774 seconds."),
    ("How long did model startup take after routing?", "98.0729 seconds; cumulative model-ready time was 171.2503 seconds."),
    ("How long did the judge stage take?", "13,747.5044 seconds, or about 3 hours 49 minutes 7.5 seconds."),
    ("What were Phase 2 output counts?", "2,582 zeros and 2,418 ones. These counts do not imply accuracy."),
    ("What were Phase 1 output counts?", "1,181 zeros and 1,335 ones."),
    ("What is the scored submission reference?", "Kaggle submission 54836339."),
    ("What is the expected v91 CSV hash?", "0fec5d610f4e7b94ab6c46e2094d58e940b0f36781e9fa98d5466d2135474c28."),
    ("What is the model context size?", "8,192 tokens at the server; the judge prompt caps cleaned supplied context at 1,800 characters."),
    ("What is the request temperature?", "0."),
    ("What is the request grammar?", "A grammar allowing only 0 or 1."),
    ("Is thinking enabled?", "No. It is disabled in the server and request."),
    ("How many retries?", "Up to two retries after the first attempt, with the same seed and prompt."),
    ("How are seeds chosen?", "A stable SHA-256-derived seed from row ID and prompt view."),
    ("Is inference concurrent?", "No at the request level: one worker and server parallelism 1. The model is split across both GPUs."),
    ("What is tensor split?", "1,1 across the two T4 GPUs in layer split mode."),
    ("What if context is NULL?", "The row follows structured/source routes and, if unresolved, up to three no-context judge views."),
    ("What if context exists?", "Context-specific deterministic checks run; unresolved rows receive one context-grounding judge view."),
    ("What decides label 1 for no-context judging?", "All executed views must be 1."),
    ("What decides early label 0?", "The first no-context judge view that returns 0 stops evaluation."),
    ("Does retrieval decide labels?", "Only after strict typed admission; broad or fuzzy retrieval alone is proposal-only."),
    ("What is the fallback if evidence conflicts?", "Abstain to the judge."),
    ("Did you train an encoder?", "No trained model is part of the final frozen pipeline."),
    ("Did you use paid APIs?", "No."),
    ("Did you use internet during evaluation?", "No."),
    ("Can the notebook accept a different row count?", "Yes. It reads N from the literal mounted test file and preserves its IDs."),
    ("Why 0.17 hours in the form?", "It is a conservative 612-second report above the slowest measured Phase 1 run of 593.43 seconds."),
    ("What did three identical runs prove?", "Execution and CSV reproducibility for Phase 1, not unseen-domain accuracy."),
    ("What did Phase 2 runtime prove?", "The frozen package completed 5,000 shifted rows under nine hours, not that its predictions were correct."),
    ("Was the private score manually reproduced from hidden labels?", "No. The notebook reproduced the scored prediction file from raw rows and frozen public assets."),
    ("What should you say if a number is unknown?", "“We did not measure that. What we can verify is …” followed by the closest documented evidence."),
]


def build_document():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.82)
    section.bottom_margin = Inches(0.78)
    section.left_margin = Inches(0.88)
    section.right_margin = Inches(0.88)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)
    configure_styles(doc)
    add_header_footer(doc)

    # Editorial-cover pattern for a practical technical handbook.
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(92)
    p.paragraph_format.space_after = Pt(18)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("FINAL ROUND PREPARATION"), size=11, color=GOLD, bold=True)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("Team Huntrix\nJudge Q&A Handbook"), size=28, color=INK, bold=True)

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
        "Purpose",
        "Prepare both presenters to answer technical, methodological, compliance, runtime, and limitation questions clearly—without overclaiming.",
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
    set_run_font(p.add_run("Prepared 23 July 2026  |  Internal rehearsal edition"), size=10, color=MUTED)
    doc.add_page_break()

    doc.add_heading("How to use this handbook", level=1)
    add_callout(
        doc,
        "The four-part answer pattern",
        "Start with the claim. Name the mechanism. Give one measured proof point. End with the limitation. Most answers should take 20–35 seconds unless the judge asks for detail.",
        PALE_BLUE,
        BLUE,
    )
    add_numbered(
        doc,
        [
            "Each presenter rehearses the questions assigned to their slide block, then learns the Quick facts page.",
            "A teammate asks the red-team version of each question. The speaker answers without reading.",
            "Stop any answer that exceeds 45 seconds. Repeat it using claim → mechanism → evidence → limitation.",
            "If the requested fact is not measured, say so directly and pivot to the nearest verified result.",
            "Never contradict the frozen notebook, paper, deck, public repository, or organizer-run runtime record.",
        ],
    )
    add_callout(
        doc,
        "Critical wording",
        "The notebook uses one local llama.cpp server with the model tensor-split 1,1 across two T4 GPUs. The server exposes a local loopback HTTP endpoint. That is not an external or paid API call.",
        PALE_GOLD,
        GOLD,
    )
    add_callout(
        doc,
        "Safe sentence when you do not know",
        "“We did not measure that, so I do not want to infer it. What we can verify is …”",
        PALE_RED,
        RED,
    )

    doc.add_heading("Suggested Q&A ownership", level=2)
    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    set_table_geometry(table, [2400, 2600, 4360])
    set_repeat_table_header(table.rows[0])
    hdr = table.rows[0].cells
    for cell, text in zip(hdr, ("Presenter", "Primary slides", "Questions to lead")):
        cell.text = text
        set_cell_shading(cell, PALE_BLUE)
        for run in cell.paragraphs[0].runs:
            set_run_font(run, size=10.5, color=BLUE, bold=True)
    ownership = [
        ("Seyam", "1–7", "Task framing, selective cascade, solvers, relation and context verification"),
        ("Abtahee", "8–15", "Judge, evidence, experiments, reproducibility, Phase 2, limitations and close"),
    ]
    for presenter, slides, lead in ownership:
        cells = table.add_row().cells
        for cell, text in zip(cells, (presenter, slides, lead)):
            p = cell.paragraphs[0]
            set_run_font(p.add_run(text), size=10, color=INK, bold=(cell is cells[0]))
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    set_run_font(
        p.add_run("Note: "),
        size=9.5,
        color=RED,
        bold=True,
    )
    set_run_font(
        p.add_run("This is speaking ownership based on the deck, not a substitute for truthful authorship or contribution statements."),
        size=9.5,
        color=MUTED,
    )

    doc.add_page_break()
    add_quick_facts(doc)

    doc.add_heading("A 30-second system explanation", level=1)
    doc.paragraphs[-1].paragraph_format.page_break_before = True
    p = doc.add_paragraph()
    p.paragraph_format.keep_together = True
    set_run_font(
        p.add_run(
            "Our system is a selective cascade. It first applies task-specific deterministic checks for supplied context, numbers, options, morphology, idioms, laws, and exact public-source lineage. A deterministic rule can emit a label only when the entity, requested predicate, value, provenance, and conflict checks agree; otherwise it abstains. The unresolved rows go to a quantized Qwen3.6-27B judge running offline through one llama.cpp server split across two T4 GPUs. This reproduced our 2,516-row Phase 1 submission byte-for-byte and completed the organizer’s 5,000-row Phase 2 run in 3 hours 52 minutes."
        ),
        size=11,
        color=INK,
    )

    doc.add_heading("A 90-second technical explanation", level=1)
    p = doc.add_paragraph()
    p.paragraph_format.keep_together = True
    set_run_font(
        p.add_run(
            "The benchmark is heterogeneous even though the output is binary. We therefore normalize each Bengali row and route it by evidence type. Context rows are reduced to claim-level support, contradiction, or unknown states. Structured solvers normalize Bengali numbers, dates, ranges, options, morphology, and literal-versus-figurative idiom meaning. Public evidence is typed as an entity, predicate, value, source family, and provenance record. Retrieval may propose evidence, but a deterministic label is admitted only when entity, predicate, and value align and no equally trusted source conflicts. Every uncertain case returns null and moves to the final judge. The judge uses one grounding view for context rows and up to three fixed no-context views, with unanimous-positive voting and early exit on the first zero. Temperature is zero, thinking is disabled, output is grammar-constrained to 0 or 1, and row order, prompt order, batches, and seeds are fixed. The result was 0.967 public and 0.982 private in Phase 1. In the shifted Phase 2 run, deterministic coverage fell from 93.72% to 4.14%, so Qwen judged 4,793 rows; the notebook still completed within the nine-hour limit and achieved an official score of 0.90144, rank 7 of 32. Because row-level labels were not released, we do not claim route-level or domain-level accuracy."
        ),
        size=11,
        color=INK,
    )

    question_number = 1
    for section_title, questions in SECTIONS:
        doc.add_page_break()
        doc.add_heading(section_title, level=1)
        for question, answer, pressed, evidence, caution in questions:
            add_question(doc, question_number, question, answer, pressed, evidence, caution)
            question_number += 1

    doc.add_page_break()
    doc.add_heading("9. Rapid-fire factual checks", level=1)
    add_callout(
        doc,
        "Rule",
        "Answer these in one sentence. Do not add a second claim unless the judge asks for it.",
        PALE_GOLD,
        GOLD,
    )
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    set_table_geometry(table, [4200, 5160])
    headers = table.rows[0].cells
    headers[0].text = "Question"
    headers[1].text = "One-sentence answer"
    set_repeat_table_header(table.rows[0])
    for cell in headers:
        set_cell_shading(cell, PALE_BLUE)
        for run in cell.paragraphs[0].runs:
            set_run_font(run, size=10.5, color=BLUE, bold=True)
    for idx, (question, answer) in enumerate(RAPID_FIRE):
        row = table.add_row()
        prevent_row_split(row)
        cells = row.cells
        if idx % 2:
            for cell in cells:
                set_cell_shading(cell, "F7F9FC")
        set_run_font(cells[0].paragraphs[0].add_run(question), size=9.5, color=INK, bold=True)
        set_run_font(cells[1].paragraphs[0].add_run(answer), size=9.5, color=INK)

    doc.add_page_break()
    doc.add_heading("10. Red-team rehearsal round", level=1)
    add_callout(
        doc,
        "How to run this round",
        "One teammate reads the hostile wording exactly. The speaker must acknowledge the valid concern, answer with evidence, and state the limit—without becoming defensive.",
        PALE_RED,
        RED,
    )
    red_team = [
        (
            "“Your Phase 2 deterministic coverage collapsed. Is this not just a Phase 1 memorization system?”",
            "The exact-lineage layer was phase-specific, and the 4.14% Phase 2 coverage demonstrates that openly. The safety property is that it abstained rather than forcing approximate labels. The frozen general judge then handled 95.86% of rows, completed under the runtime limit, and the system scored 0.90144 overall. Without row-level labels, we cannot isolate how much each route contributed.",
        ),
        (
            "“You got private rank 1 because you found the source data. Why is that research?”",
            "Source lineage contributed to Phase 1, but the research question is how to admit that evidence safely. Broad matching regressed; the useful method was typed relation verification, conflict handling, traceable abstention, and a fallback that survived source shift. We disclose both the benefit and the limitation.",
        ),
        (
            "“Is your answer bank just manual test labeling with another name?”",
            "No. A label vector maps test IDs directly to 0 or 1. Our records describe public facts or public-dataset question-answer lineage with provenance. The inference code still checks the input entity, requested predicate, and response value, and it abstains on mismatch. No hidden label was available.",
        ),
        (
            "“You say no APIs, but your code calls /v1/chat/completions.”",
            "That endpoint is served at 127.0.0.1 by the llama.cpp process inside the offline Kaggle notebook. It is a local inter-process interface, not an external service. Internet is disabled and no token or paid model is used.",
        ),
        (
            "“Your slide says multiple version gains. Where is the ablation?”",
            "We do not call those controlled ablations. They are cumulative scored versions that identify pivots and regressions. A proper ablation requires a larger labeled set and fixed evaluation conditions; we list that as next work.",
        ),
        (
            "“Why should we trust 0.993 on only 299 rows?”",
            "It is a development diagnostic, not a population guarantee. We report the sample size, class metrics, and two remaining errors, and we separate it from public, private, and Phase 2 evidence.",
        ),
        (
            "“If Qwen did almost everything in Phase 2, what did your method add?”",
            "It added a disciplined fallback boundary, reproducible packaging, and safe behavior when exact evidence disappeared. Whether the 207 deterministic Phase 2 labels improved accuracy is unknown without labels, and we will not claim otherwise.",
        ),
        (
            "“Why did your runtime projection miss by hours?”",
            "The projection assumed the Phase 1 routing mix. Phase 2 changed the unresolved rate from 6.28% to 95.86%, so model calls—not row count—dominated. The actual notebook still used less than half of the nine-hour budget.",
        ),
        (
            "“Why did Phase 2 fall below the Phase 1 private score?”",
            "The source distribution shifted sharply: deterministic coverage fell from 93.72% to 4.14%, moving 95.86% of rows to the general judge. The official result was still 0.90144 and rank 7 of 32, but without row-level labels we cannot assign the loss to specific domains or routes.",
        ),
        (
            "“Would this be safe for citizens asking about government services?”",
            "Not yet as an autonomous authority. Civic information changes and has real consequences. A production system needs timestamped authoritative sources, confidence calibration, monitoring, and human escalation.",
        ),
    ]
    for idx, (attack, response) in enumerate(red_team, 1):
        p = doc.add_paragraph(style="Question")
        set_run_font(p.add_run(f"Challenge {idx}. {attack}"), size=12, color=RED, bold=True)
        add_labelled_paragraph(doc, "Grounded response:", response, TEAL)

    doc.add_page_break()
    doc.add_heading("11. Final rehearsal checklist", level=1)
    checklist = [
        "Both presenters can state labels correctly: 0 hallucinated, 1 faithful.",
        "Both presenters can explain the router in 30 seconds without reading.",
        "Both presenters know 299, 2,516, 158, 5,000, 207, 4,793, 0.967, 0.982, 0.90144, rank 7 of 32, and 3:52:01.91.",
        "Both presenters say one llama.cpp server tensor-split across two T4 GPUs.",
        "Both presenters can distinguish local loopback HTTP from an external API.",
        "Both presenters can distinguish public fact research from manual test-row labeling.",
        "Both presenters can explain why retrieval is proposal-only.",
        "Both presenters can explain why the Runway response is label 0.",
        "Both presenters can state that version comparisons are not controlled ablations.",
        "Both presenters can state the official Phase 2 aggregate result and explain that row-level labels were not released.",
        "Both presenters can name one limitation before being asked.",
        "Each presenter has a truthful two-sentence description of their own contribution.",
        "The team has agreed who answers architecture, data, model, runtime, compliance, and limitations questions.",
        "No answer depends on private leaderboard row inference or unpublished labels.",
        "Answers stop after one claim, one mechanism, one proof point, and one limitation.",
    ]
    for item in checklist:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.1)
        p.paragraph_format.space_after = Pt(6)
        set_run_font(p.add_run("☐  "), name="Arial Unicode MS", size=11, color=TEAL, bold=True)
        set_run_font(p.add_run(item), size=11, color=INK)

    doc.add_heading("Artifact map for evidence during Q&A", level=1)
    doc.paragraphs[-1].paragraph_format.page_break_before = True
    artifact_rows = [
        ("Method and diagrams", "Huntrix_Final_Presentation.pptx; Huntrix_Technical_Report.pdf"),
        ("Exact inference behavior", "kaggle_notebooks/phase2_v91_inference/phase2_v91_inference.py"),
        ("Router and stage order", "scripts/phase2_pipeline.py"),
        ("Judge prompts and decoding", "scripts/gguf_selective_judge.py"),
        ("Experiment history", "EXPERIMENT.log"),
        ("Decision rationale and failures", "DECISION.log"),
        ("Frozen versions and hashes", "FREEZE_MANIFEST.json"),
        ("External data and licenses", "DATA_SOURCES.md; THIRD_PARTY_NOTICES.md; CITATIONS.md"),
        ("Phase 2 runtime", "PHASE2_RUNTIME_EVIDENCE.md; organizer-run huntrix-1.ipynb"),
        ("Compliance disclosure", "METHODOLOGY_DISCLOSURE.md; public README"),
    ]
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    set_table_geometry(table, [2900, 6460])
    set_repeat_table_header(table.rows[0])
    for cell, text in zip(table.rows[0].cells, ("Question area", "Primary evidence artifact")):
        cell.text = text
        set_cell_shading(cell, PALE_BLUE)
        for run in cell.paragraphs[0].runs:
            set_run_font(run, size=10.5, color=BLUE, bold=True)
    for left, right in artifact_rows:
        row = table.add_row()
        prevent_row_split(row)
        cells = row.cells
        set_run_font(cells[0].paragraphs[0].add_run(left), size=10, color=INK, bold=True)
        set_run_font(cells[1].paragraphs[0].add_run(right), size=10, color=INK)

    doc.add_paragraph()
    add_callout(
        doc,
        "Closing line",
        "“Our strongest claim is not that every source was covered. It is that the system required the right relation before trusting evidence, abstained when that evidence disappeared, reproduced its scored output, and completed the shifted held-out run within budget.”",
        PALE_TEAL,
        TEAL,
    )

    # Keep question cards and table headers visually stable.
    for paragraph in doc.paragraphs:
        if paragraph.style.name in {"Question", "Heading 1", "Heading 2", "Heading 3"}:
            paragraph.paragraph_format.keep_with_next = True
        if paragraph.style.name == "Question":
            paragraph.paragraph_format.page_break_before = False

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_document()
