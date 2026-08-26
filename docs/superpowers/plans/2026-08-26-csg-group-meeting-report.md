# CSG Group Meeting Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and visually verify one 6-8 page Chinese Word report that explains the CSG model and embeds the three supplied figures.

**Architecture:** A reproducible Python builder will assemble a styled A4 DOCX from repository-grounded model facts and the three source PNGs. A separate verification pass will audit document structure, render every page to PNG, inspect every rendered page, and iterate on the builder until the final layout is clean.

**Tech Stack:** Bundled Codex Python runtime, `python-docx`, Pillow, OOXML helpers, LibreOffice via `render_docx.py`.

---

## File Structure

- Create: `reports/build_csg_group_meeting_report.py` - deterministic document builder and all report prose/style definitions.
- Create: `reports/csg_group_meeting_report_2026-08-26.docx` - final user deliverable.
- Create: `reports/qa_csg_report_v1/` - renderer output used only for visual QA.
- Read: `models/convlstm_ls_topography_joint_gated_simvp.py` - source of model structure and contracts.
- Read: `tests/test_convlstm_ls_topography_joint_gated_simvp.py` - source of engineering validation coverage.
- Read: `docs/superpowers/specs/2026-08-26-csg-group-meeting-report-design.md` - approved content and layout boundary.
- Read: the three user-supplied PNG files in `C:/Users/29737/AppData/Local/Temp/` - embedded figures.

### Task 1: Resolve Document Runtime And Authoring Rules

**Files:**
- Read: `C:/Users/29737/.codex/plugins/cache/openai-primary-runtime/documents/26.819.11345/skills/documents/references/design_presets.md`
- Read: `C:/Users/29737/.codex/plugins/cache/openai-primary-runtime/documents/26.819.11345/skills/documents/tasks/create_edit.md`
- Read: `C:/Users/29737/.codex/plugins/cache/openai-primary-runtime/documents/26.819.11345/skills/documents/tasks/images_figures.md`
- Read: `C:/Users/29737/.codex/plugins/cache/openai-primary-runtime/documents/26.819.11345/skills/documents/tasks/verify_render.md`

- [ ] **Step 1: Confirm bundled executables and packages**

Use these workspace dependency paths:

```text
Python: C:/Users/29737/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe
Node:   C:/Users/29737/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe
```

Run:

```powershell
& 'C:\Users\29737\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -c "import docx, PIL; print(docx.__version__, PIL.__version__)"
```

Expected: exit code 0 and both package versions printed.

- [ ] **Step 2: Verify all three image inputs before authoring**

Run `Get-Item` for these exact paths and confirm nonzero lengths:

```text
C:/Users/29737/AppData/Local/Temp/codex-clipboard-4cb30153-cdde-48a5-b61c-bafc71fe033b.png
C:/Users/29737/AppData/Local/Temp/codex-clipboard-b588e1a7-ff38-42b1-82e7-905d2c72eccb.png
C:/Users/29737/AppData/Local/Temp/codex-clipboard-ec8e412a-73d6-4739-8013-471ae85ff815.png
```

Expected dimensions are approximately `1096x437`, `1092x485`, and `1073x773` pixels.

- [ ] **Step 3: Mark the DOCX create operation exactly once**

Run from the bundled documents package root:

```powershell
& 'C:\Users\29737\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' container_tools/mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format docx
```

Expected: exit code 0 before the first authoring command.

### Task 2: Build The CSG Report

**Files:**
- Create: `reports/build_csg_group_meeting_report.py`
- Create: `reports/csg_group_meeting_report_2026-08-26.docx`

- [ ] **Step 1: Implement deterministic document helpers**

Create helpers with these responsibilities and signatures:

```python
from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, RGBColor


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def keep_with_next(paragraph) -> None:
    paragraph.paragraph_format.keep_with_next = True


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instruction, separate, end))


def add_heading(document, number: str, title: str, level: int = 1):
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.add_run(f"{number}  {title}")
    keep_with_next(paragraph)
    return paragraph


def add_callout(document, label: str, body: str, accent: str):
    table = document.add_table(rows=1, cols=2)
    table.autofit = False
    table.columns[0].width = Cm(2.4)
    table.columns[1].width = Cm(14.7)
    label_cell, body_cell = table.rows[0].cells
    set_cell_shading(label_cell, accent)
    set_cell_shading(body_cell, "F3F6F9")
    for cell in (label_cell, body_cell):
        set_cell_margins(cell)
    label_run = label_cell.paragraphs[0].add_run(label)
    label_run.bold = True
    label_run.font.color.rgb = RGBColor(255, 255, 255)
    body_cell.paragraphs[0].add_run(body)
    return table


def add_figure(document, path: Path, caption: str, width_cm: float):
    image_paragraph = document.add_paragraph()
    image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_paragraph.paragraph_format.keep_with_next = True
    image_paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
    caption_paragraph = document.add_paragraph(caption, style="Caption")
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.keep_with_next = True
    source = document.add_paragraph("来源：本次实验平台截图。")
    source.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return image_paragraph, caption_paragraph, source
```

Apply one resolved `standard_business_brief` preset consistently: A4 portrait,
`1.8 cm` side margins, `1.7 cm` top/bottom margins, Chinese body font
`Microsoft YaHei` or available CJK fallback, restrained `#1F2937` text, blue
`#1F77B4`, orange `#FF7F0E`, green `#2CA02C`, and light gray `#EEF2F6` rules.
Set explicit paragraph spacing, table widths, cell margins, image widths, and
header/footer dimensions rather than inheriting Word defaults.

- [ ] **Step 2: Write the eight-page content sequence**

Build these sections with explicit page breaks so each page has one purpose:

```text
1. Cover: CSG 条件时空门控模型 / 火星臭氧多步预测组会汇报 / 2026-08-26
2. 研究问题与核心结论: motivation, three model comparison, evidence boundary
3. Model architecture: input contracts, main pipeline, Figure 1
4. Joint gate: Ls harmonics, MOLA scaling, fusion equation, initialization
5. Overall results: Figure 2, approximate RMSE ranking, relative interpretation
6. Multi-step behavior: Figure 3, early/medium/long lead observations
7. Engineering review: 170,338 parameters, +8.19%, tests, strengths, limitations
8. Conclusions and next experiment matrix: baseline/current/additive-residual,
   at least three seeds, lead/Ls/elevation stratification
```

Use the exact current-model equation:

```text
J = GELU(W_e(E) + f_Ls(Ls) + f_topo(MOLA))
G = tanh(W_g(J))
E_gated = E * (1 + s * G),  s in [0, 1]
```

State that chart-derived values are approximate. Do not invent author,
institution, optimizer, epoch count, split ratios, statistical significance, or
non-RMSE metrics.

- [ ] **Step 3: Embed and caption the figures**

Use the PNGs in this order:

```text
Figure 1: CSG 架构示意图
Figure 2: SimVP、ConvLSTM-SimVP 与 CSG 的总体 RMSE 对比
Figure 3: 三种模型在 20 个预测步上的 RMSE 变化
```

Preserve aspect ratio, center every image, keep each caption with its image,
and add source text `来源：本次实验平台截图。` below the caption. Add descriptive
alternative text to each drawing in the final OOXML.

- [ ] **Step 4: Generate the DOCX**

Run:

```powershell
& 'C:\Users\29737\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'D:\_HQZOZONE\reports\build_csg_group_meeting_report.py'
```

Expected: `reports/csg_group_meeting_report_2026-08-26.docx` exists, is larger
than 100 KB, and contains no visible placeholders.

### Task 3: Structural And Accessibility Verification

**Files:**
- Verify: `reports/csg_group_meeting_report_2026-08-26.docx`

- [ ] **Step 1: Audit document structure with python-docx**

Run a read-only script that asserts:

```python
from docx import Document

doc = Document(r"D:\_HQZOZONE\reports\csg_group_meeting_report_2026-08-26.docx")
assert len(doc.inline_shapes) == 3
assert len(doc.sections) == 1
assert any("CSG" in p.text for p in doc.paragraphs)
assert any("E_gated" in p.text for p in doc.paragraphs)
assert not any(token in p.text for p in doc.paragraphs for token in ("TBD", "TODO", "待补充"))
print(len(doc.paragraphs), len(doc.tables), len(doc.inline_shapes))
```

Expected: exit code 0 and exactly three inline images.

- [ ] **Step 2: Run the packaged accessibility audit**

Run:

```powershell
& 'C:\Users\29737\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/a11y_audit.py 'D:\_HQZOZONE\reports\csg_group_meeting_report_2026-08-26.docx'
```

Expected: no missing image descriptions and no unmarked header-row issues. Fix
safe findings in the builder and regenerate rather than patching the final file
ad hoc.

### Task 4: Render And Inspect Every Page

**Files:**
- Verify: `reports/csg_group_meeting_report_2026-08-26.docx`
- Create: `reports/qa_csg_report_v1/page-*.png`

- [ ] **Step 1: Render DOCX to page PNGs**

Run from the documents skill package root:

```powershell
& 'C:\Users\29737\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' render_docx.py 'D:\_HQZOZONE\reports\csg_group_meeting_report_2026-08-26.docx' --output_dir 'D:\_HQZOZONE\reports\qa_csg_report_v1' --emit_pdf
```

Expected: 6-8 nonempty `page-*.png` files and a nonempty QA PDF.

- [ ] **Step 2: Inspect every page image at original detail**

Open every `page-*.png` with `view_image(detail="original")`. Confirm:

```text
- no clipped or overlapping Chinese text
- no figure distortion, cropping, or unreadable labels
- no headings stranded at page bottoms
- no captions split from figures
- no unexpected blank pages or large dead zones
- header, footer, and page numbers remain inside margins
- table borders, padding, and column widths are visually balanced
```

- [ ] **Step 3: Iterate without destructive cleanup**

If a defect exists, modify only `reports/build_csg_group_meeting_report.py`,
regenerate the DOCX, and render to a new directory such as
`reports/qa_csg_report_v2`. Do not recursively remove prior QA output. Repeat
until every latest page passes visual inspection.

### Task 5: Final Verification And Handoff

**Files:**
- Verify: `reports/csg_group_meeting_report_2026-08-26.docx`

- [ ] **Step 1: Run final file and content checks**

Run `Get-Item` for the final DOCX, rerun the structural assertions against the
latest build, and confirm the latest renderer exited with code 0. Record the
final page count and output size.

- [ ] **Step 2: Inspect repository changes**

Run:

```powershell
git status --short -- 'reports/build_csg_group_meeting_report.py' 'reports/csg_group_meeting_report_2026-08-26.docx'
```

Expected: only the report builder and requested DOCX are new in this scope.
Do not stage, commit, or modify unrelated user files.

- [ ] **Step 3: Deliver the DOCX only**

Return one output citation for
`D:/_HQZOZONE/reports/csg_group_meeting_report_2026-08-26.docx`. Mention the
page count, the three embedded figures, and that every rendered page was
visually checked. Do not link QA PNGs or the QA PDF unless requested.
