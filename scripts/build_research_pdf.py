#!/usr/bin/env python3
"""Build a five-page PDF exclusively from a completed measured comparison.

Requires reportlab (available in the bundled document runtime). No placeholder
results are generated if comparison.json is missing or incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("clean", "burst_missing", "limb_missing", "long_missing", "jitter")
VARIANTS = ("baseline", "recovery", "rejection", "combined")
NAMES = {
    "baseline": "Original model",
    "recovery": "Joint recovery",
    "rejection": "Rejection only",
    "combined": "Recovery + rejection",
}
SHORT = {
    "clean": "Original poses",
    "burst_missing": "Short joint gaps",
    "limb_missing": "Short limb gaps",
    "long_missing": "Long joint gaps",
    "jitter": "Coordinate noise",
}
Q1_URL = "https://www.scimagojr.com/journalvalue.php?area=1700&category=1707&ord=desc&order=item"
PAPER_URL = "https://doi.org/10.1016/j.patcog.2023.109528"
REPO_URL = "https://github.com/LukasHedegaard/continual-skeletons"


def clean(text):
    return str(text).translate(
        str.maketrans(
            {
                "\u2011": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u00a0": " ",
                "\u03c3": "sigma",
            }
        )
    )


def pct(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f} pp"


def validate(data):
    if data.get("status") != "complete":
        raise ValueError("Only a completed measured comparison can be published")
    if {c["id"] for c in data["conditions"]} != set(CONDITIONS):
        raise ValueError("All five conditions must be present")
    for condition in data["conditions"]:
        if {r["variant"] for r in condition["results"]} != set(VARIANTS):
            raise ValueError("All four variants must be present in every condition")
        for row in condition["results"]:
            if row["n_samples"] != data["protocol"]["test_count"]:
                raise ValueError("Inconsistent test sample count")
            for key in ("accuracy", "macro_f1", "coverage", "accepted_accuracy"):
                number = row[key]
                if number is None and key == "accepted_accuracy":
                    continue
                if not math.isfinite(number) or not 0 <= number <= 1:
                    raise ValueError(f"Invalid metric: {key}")
    if not data["equivalence"].get("passed"):
        raise ValueError("The recorded clip/continual equivalence check did not pass")


def build(data, output, digest):
    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph, Table, TableStyle

    font_dir = Path(reportlab.__file__).parent / "fonts"
    for name, filename in (
        ("ReportSans", "Vera.ttf"),
        ("ReportSans-Bold", "VeraBd.ttf"),
        ("ReportSans-Oblique", "VeraIt.ttf"),
        ("ReportSans-BoldOblique", "VeraBI.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily(
        "ReportSans",
        normal="ReportSans",
        bold="ReportSans-Bold",
        italic="ReportSans-Oblique",
        boldItalic="ReportSans-BoldOblique",
    )
    width, height = A4
    margin = 44
    content = width - 2 * margin
    ink = colors.HexColor("#172033")
    muted = colors.HexColor("#536275")
    purple = colors.HexColor("#6D28D9")
    pale = colors.HexColor("#F3EFFC")
    line = colors.HexColor("#DCE3EB")
    slate = colors.HexColor("#52647A")
    c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
    c.setTitle("PoseLab - Measured Human Activity Recognition Research Results")
    c.setAuthor("PoseLab Mini Project-II")
    c.setSubject(
        "Real Kinetics-skeleton pilot; pretrained CoST-GCN* and bounded causal pose recovery"
    )
    styles = {
        "body": ParagraphStyle(
            "body",
            fontName="ReportSans",
            fontSize=10,
            leading=14,
            textColor=ink,
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "small", fontName="ReportSans", fontSize=8.5, leading=11.5, textColor=muted
        ),
        "title": ParagraphStyle(
            "title", fontName="ReportSans-Bold", fontSize=29, leading=33, textColor=ink
        ),
        "h1": ParagraphStyle(
            "h1", fontName="ReportSans-Bold", fontSize=23, leading=28, textColor=ink
        ),
        "h2": ParagraphStyle(
            "h2", fontName="ReportSans-Bold", fontSize=12, leading=16, textColor=purple
        ),
        "cell": ParagraphStyle(
            "cell", fontName="ReportSans", fontSize=8.5, leading=11, textColor=ink
        ),
        "header": ParagraphStyle(
            "header",
            fontName="ReportSans-Bold",
            fontSize=8.3,
            leading=10.5,
            textColor=ink,
        ),
        "code": ParagraphStyle(
            "code", fontName="Courier", fontSize=8, leading=11, textColor=ink
        ),
    }
    p = data["protocol"]
    settings = data["calibration"]
    config = settings["recovery"]
    by_condition = {item["id"]: item for item in data["conditions"]}

    def metric(condition, variant):
        return next(
            r for r in by_condition[condition]["results"] if r["variant"] == variant
        )

    def paragraph(text, y, style="body", gap=10, x=margin, available=content):
        text = clean(text)
        block = Paragraph(text, styles[style])
        _, block_height = block.wrap(available, height)
        if y - block_height < 53:
            raise ValueError(f"Page content overflow near: {text[:80]}")
        block.drawOn(c, x, y - block_height)
        return y - block_height - gap

    def section(title, y):
        return paragraph(escape(title), y, "h2", gap=6)

    def start(page, label, title):
        c.setFillColor(purple)
        c.rect(0, height - 8, width, 8, fill=1, stroke=0)
        c.setFont("ReportSans-Bold", 8)
        c.setFillColor(muted)
        c.drawString(
            margin,
            height - 35,
            f"POSELAB  /  MEASURED RESEARCH PILOT  /  {label.upper()}",
        )
        y = paragraph(escape(title), height - 65, "h1", gap=15)
        c.setStrokeColor(line)
        c.line(margin, 39, width - margin, 39)
        c.setFillColor(muted)
        c.setFont("ReportSans", 8)
        c.drawString(margin, 25, "IIIT Dharwad | Mini Project-II | Aug-Nov 2026")
        c.drawRightString(width - margin, 25, f"{page} / 5")
        return y

    def table(values, col_widths, y, spans=(), cell_size=8.5, padding=7):
        body_style = ParagraphStyle(
            "table_body", parent=styles["cell"], fontSize=cell_size
        )
        formatted = [
            [
                Paragraph(
                    escape(clean(value)).replace("\n", "<br/>"),
                    styles["header"] if row_index == 0 else body_style,
                )
                for value in row
            ]
            for row_index, row in enumerate(values)
        ]
        t = Table(formatted, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), pale),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), padding),
            ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, line),
        ]
        for row_index in range(1, len(values)):
            commands.append(("LINEBELOW", (0, row_index), (-1, row_index), 0.35, line))
        for start_row, end_row in spans:
            commands.extend(
                [
                    ("SPAN", (0, start_row), (0, end_row)),
                    (
                        "BACKGROUND",
                        (0, start_row),
                        (0, end_row),
                        colors.HexColor("#F7F9FC"),
                    ),
                ]
            )
        t.setStyle(TableStyle(commands))
        _, table_height = t.wrap(content, height)
        if y - table_height < 53:
            raise ValueError("Table would run beyond the page margin")
        t.drawOn(c, margin, y - table_height)
        return y - table_height - 15

    def bullet(text, y):
        return paragraph("<b>-</b> " + escape(text), y, gap=7)

    # Page 1: a substantive first page, with method and evidence clearly scoped.
    y = start(
        1,
        "Project and method",
        "Human Activity Recognition\nwith Missing-Joint Recovery",
    )
    y = paragraph(
        "<b>A measured extension of a published skeleton-action model.</b> We investigate whether limited repair of temporarily missing pose joints helps an unchanged pretrained classifier, and whether calibrated rejection makes accepted predictions more reliable.",
        y,
    )
    y = section("Base paper and verified journal status", y)
    y = paragraph(
        f'<b>Continual Spatio-Temporal Graph Convolutional Networks</b><br/>Lukas Hedegaard, Negar Heidari and Alexandros Iosifidis.<br/><i>Pattern Recognition</i> 140 (2023), 109528. <link href="{PAPER_URL}" color="#6D28D9">Published paper</link>.',
        y,
    )
    y = paragraph(
        f'<b>Q1 evidence:</b> SCImago/SJR <b>2024</b>, category <b>Computer Vision and Pattern Recognition</b>; journal SJR 2.058, Q1. This verifies that named ranking/year, not 2023 JCR status. <link href="{escape(Q1_URL)}" color="#6D28D9">SCImago category listing</link>.',
        y,
        "small",
    )
    y = section("The implementation in one pipeline", y)
    y = paragraph(
        "Real OpenPose skeleton sequence &gt; optional bounded causal repair &gt; authors' fixed CoST-GCN* classifier &gt; optional calibrated rejection.",
        y,
    )
    y = paragraph(
        "The model combines information across body joints and across time. The paper's continual formulation reuses intermediate calculations during streaming. Our measurements use its weight-compatible clip path, with a recorded numerical check against the continual definition.",
        y,
    )
    y = section("What is ours", y)
    y = paragraph(
        "We implement pose repair, confidence decay, bounded gap handling, rejection calibration, controlled corruption experiments, ablations and a reviewable comparison. The pretrained classifier and its weights are credited to the authors; they were not trained by this project.",
        y,
    )
    selected = (
        "Confidence-gated hold (beta = 0)"
        if config["beta"] == 0
        else "Alpha-beta motion extrapolation"
    )
    details = [
        ["Frozen setting", "Measured configuration"],
        ["Calibration-selected recipe", settings["selected_candidate"]],
        ["Repair behavior", selected],
        [
            "Gap and update parameters",
            f"Maximum {config['max_gap_frames']} frames; alpha {config['alpha']}; beta {config['beta']}",
        ],
        [
            "Prior confidence / decay",
            f"Minimum repair confidence {config.get('min_repair_confidence', 0)}; decay {config['confidence_decay']}",
        ],
        [
            "Coordinate guards",
            f"Speed <= {config['max_speed']}; displacement <= {config['max_displacement']} (input units)",
        ],
        [
            "Person-slot reset guard",
            (
                "Disabled"
                if config.get("identity_jump_threshold") is None
                else f"Torso jump > {config['identity_jump_threshold']} input units resets prior state"
            ),
        ],
    ]
    y = table(details, [175, content - 175], y)
    y = paragraph(
        f"Measured run: {escape(data['generated_at'])}. {p['calibration_count']} calibration clips; {p['test_count']} disjoint test clips; {p['class_count']} classifier outputs.",
        y,
        "small",
    )
    c.showPage()

    # Page 2: primary paired comparison and a vector chart.
    y = start(2, "Primary result", "Recovery gains and individual regressions")
    y = paragraph(
        "The primary contrast isolates recovery without rejection. Both methods see the same held-out clips, the same preprocessing and the same added corruption masks. All five conditions are retained.",
        y,
    )
    plot_bottom = y - 240
    left = margin + 34
    plot_width = content - 44
    plot_height = 175
    largest = max(
        metric(condition, variant)["accuracy"] * 100
        for condition in CONDITIONS
        for variant in ("baseline", "recovery")
    )
    upper = max(40, math.ceil(largest / 10) * 10 + 10)
    c.setFont("ReportSans", 8)
    for tick in range(0, upper + 1, 10):
        yy = plot_bottom + tick / upper * plot_height
        c.setStrokeColor(line)
        c.line(left, yy, left + plot_width, yy)
        c.setFillColor(muted)
        c.drawRightString(left - 7, yy - 3, str(tick))
    c.setFont("ReportSans-Bold", 8.5)
    c.setFillColor(ink)
    c.drawString(left, plot_bottom + plot_height + 23, "All-input accuracy (%)")
    group_width = plot_width / len(CONDITIONS)
    for index, condition in enumerate(CONDITIONS):
        mid = left + group_width * (index + 0.5)
        for offset, variant, color in (
            (-19, "baseline", slate),
            (3, "recovery", purple),
        ):
            value = metric(condition, variant)["accuracy"] * 100
            bar_height = value / upper * plot_height
            c.setFillColor(color)
            c.rect(mid + offset, plot_bottom, 18, bar_height, fill=1, stroke=0)
            c.setFont("ReportSans-Bold", 8)
            c.drawCentredString(
                mid + offset + 9, plot_bottom + bar_height + 5, f"{value:.1f}"
            )
        label = SHORT[condition].split(" ")
        c.setFillColor(ink)
        c.setFont("ReportSans", 8)
        c.drawCentredString(mid, plot_bottom - 14, " ".join(label[:2]))
        if len(label) > 2:
            c.drawCentredString(mid, plot_bottom - 25, " ".join(label[2:]))
    for xx, color, label in (
        (margin + 130, slate, "Original model"),
        (margin + 280, purple, "With joint recovery"),
    ):
        c.setFillColor(color)
        c.rect(xx, plot_bottom - 48, 9, 9, fill=1, stroke=0)
        c.setFillColor(ink)
        c.drawString(xx + 15, plot_bottom - 47, label)
    y = plot_bottom - 70
    values = [["Condition", "Difference", "Paired 95% interval", "Fixed / regressed"]]
    for condition in CONDITIONS:
        paired = by_condition[condition]["paired_recovery"]
        values.append(
            [
                SHORT[condition],
                pp(paired["accuracy_gain"]),
                f"{pp(paired['ci95'][0])} to {pp(paired['ci95'][1])}",
                f"{paired['improved_clips']} / {paired['regressed_clips']}",
            ]
        )
    y = table(values, [128, 75, 176, content - 379], y)
    y = paragraph(
        "Differences are percentage points, not relative percentage improvements. A paired bootstrap resamples clip pairs; intervals crossing zero do not establish a consistent gain in this pilot. Corrected and regressed clips are both shown.",
        y,
        "small",
    )
    noise = by_condition["jitter"]["paired_recovery"]
    noise_uncertainty = (
        "The paired interval includes zero, so a consistent improvement is not established here."
        if noise["ci95"][0] <= 0 <= noise["ci95"][1]
        else "Its paired interval describes this pilot, not all future noisy inputs."
    )
    y = paragraph(
        f"<b>Noise stress test:</b> recovery changes accuracy by {pp(noise['accuracy_gain'])}. {noise_uncertainty} Gap repair is not a general-purpose denoising solution.",
        y,
    )
    c.showPage()

    # Page 3: complete four-way ablations, without hiding rejection's cost.
    y = start(3, "Complete ablation", "Read accuracy alongside coverage")
    y = paragraph(
        "All-input accuracy and macro-F1 count abstentions as errors. Accepted accuracy uses only clips receiving a prediction. A higher accepted accuracy at lower coverage is a tradeoff, not proof of a better classifier.",
        y,
    )
    values = [
        [
            "Condition",
            "Variant",
            "All-input\naccuracy",
            "Macro-F1",
            "Accepted\naccuracy",
            "Coverage",
        ]
    ]
    spans = []
    for condition in CONDITIONS:
        start_row = len(values)
        for index, variant in enumerate(VARIANTS):
            row = metric(condition, variant)
            values.append(
                [
                    SHORT[condition] if index == 0 else "",
                    NAMES[variant],
                    pct(row["accuracy"]),
                    pct(row["macro_f1"]),
                    pct(row["accepted_accuracy"]),
                    pct(row["coverage"]),
                ]
            )
        spans.append((start_row, len(values) - 1))
    y = table(values, [92, 108, 78, 65, 85, content - 428], y, spans, cell_size=8)
    y = paragraph(
        f"Macro-F1 averages the {p['classes_present']} true classes represented in the test subset. The model retains all {p['class_count']} outputs. {p['empty_clips_retained']} empty test clips are retained.",
        y,
        "small",
    )
    y = paragraph(
        f"Frozen rejection thresholds: original inputs {settings['baseline_threshold']:.3f}; repaired inputs {settings['recovery_threshold']:.3f}. Threshold selection seeks the highest calibration accepted accuracy subject to at least 80% calibration coverage; this does not guarantee that coverage on unseen conditions.",
        y,
        "small",
    )
    c.showPage()

    # Page 4: precise claims and timing scope.
    y = start(4, "Interpretation and limits", "What this experiment establishes")
    for condition in CONDITIONS:
        result = by_condition[condition]["paired_recovery"]
        gain = result["accuracy_gain"]
        statement = (
            "improved" if gain > 0 else "worsened" if gain < 0 else "did not change"
        )
        interval_note = (
            "interval includes zero"
            if result["ci95"][0] <= 0 <= result["ci95"][1]
            else "paired interval excludes zero"
        )
        y = bullet(
            f"{SHORT[condition]}: accuracy difference {pp(gain)}; {interval_note}.",
            y,
        )
    y = section("Measured processing cost", y - 4)
    timings = [["Condition", "Original ms/clip", "Recovery ms/clip"]]
    for condition in CONDITIONS:
        timings.append(
            [
                SHORT[condition],
                f"{metric(condition, 'baseline')['latency_ms']:.2f}",
                f"{metric(condition, 'recovery')['latency_ms']:.2f}",
            ]
        )
    y = table(timings, [content - 230, 115, 115], y)
    y = paragraph(escape(p["timing_scope"]), y, "small")
    y = section("Limits that must stay in the presentation", y)
    for limitation in [
        "Pose coordinates and labels are real Kinetics data. Joint masking and coordinate noise are synthetic perturbations; physical camera occlusion has not been validated.",
        "This is a sampled pilot, not a reproduction of the full paper benchmark or a state-of-the-art claim. Upstream validation data may have influenced the authors' checkpoint selection.",
        "Confidence-ranked person slots may change identities. Long gaps, abrupt motion and person swaps remain difficult. The optional identity guard is not a full tracker.",
        "The new 400-class classifier has no validated webcam integration. The existing MediaPipe six-activity dashboard is a separate demo, not this research checkpoint.",
    ]:
        y = bullet(limitation, y)
    c.showPage()

    # Page 5: reproducibility and concise viva answers.
    y = start(5, "Reproduce and present", "A result you can inspect and explain")
    y = section("Reproduction", y)
    y = paragraph(
        "Clone the project repository and install its recorded research dependencies. Setup downloads a pinned upstream checkout and verifies the checkpoint hash. The data utility downloads a fixed real-data subset using byte ranges rather than the full archive.",
        y,
    )
    commands = "python -m pip install -r requirements-research.txt<br/>python scripts/setup_research.py<br/>python scripts/research_data.py<br/>python -m research.benchmark --phase all<br/>python scripts/render_research_report.py"
    y = paragraph(commands, y, "code", gap=12)
    y = paragraph(
        "The Colab notebook is notebooks/Research_Comparison.ipynb. Keep comparison.json, predictions.csv, calibration.json, design.json and the data manifest together. Selection is deterministic; never choose test clips based on whether predictions succeed.",
        y,
        "small",
    )
    y = section("Short viva answers", y)
    for question, answer in [
        (
            "Why a graph network?",
            "A body is naturally a graph of connected joints. Spatial layers combine neighboring joints; temporal layers learn movement across frames.",
        ),
        (
            "What is the contribution?",
            "A bounded repair and rejection pipeline plus a controlled, reproducible ablation study around the authors' unchanged pretrained classifier.",
        ),
        (
            "Why is repair causal?",
            "It uses previous and current measurements only. Filling a gap using a future observation would violate the streaming constraint.",
        ),
        (
            "Why not compare our number directly with the paper?",
            "The paper uses full benchmark splits. This smaller pilot and its added corruptions answer a different, explicitly scoped question.",
        ),
        (
            "Why show negative results?",
            "They identify the method's limits. A useful experiment must show regressions and unchanged conditions as well as improvements.",
        ),
    ]:
        y = paragraph(f"<b>{escape(question)}</b> {escape(answer)}", y, gap=8)
    y = section("Provenance and evidence", y)
    eq = data["equivalence"]
    y = paragraph(
        f"Numerical clip/continual check: passed. Maximum logit differences: clip vs continual {eq['clip_vs_continual_max_abs']:.3g}; clip vs step {eq['clip_vs_step_max_abs']:.3g}. This is a check on the recorded calibration input, not validation of every possible stream.",
        y,
        "small",
    )
    y = paragraph(
        f'<link href="{PAPER_URL}" color="#6D28D9">Published base paper</link> | <link href="{REPO_URL}" color="#6D28D9">Authors\' implementation</link> | <link href="{escape(Q1_URL)}" color="#6D28D9">SCImago 2024 category evidence</link>',
        y,
        "small",
    )
    y = paragraph(
        "Upstream commit: "
        + p["upstream_commit"]
        + "<br/>Checkpoint SHA256: "
        + p["checkpoint_sha256"]
        + "<br/>Comparison JSON SHA256: "
        + digest,
        y,
        "small",
    )
    c.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "results/research/comparison.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/pdf/PoseLab_Research_Results.pdf"
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.exit(
            1,
            "Measured comparison.json is not ready. No PDF or placeholder results were generated.\n",
        )
    try:
        raw = args.input.read_bytes()
        data = json.loads(raw)
        validate(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        build(data, args.output, hashlib.sha256(raw).hexdigest())
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(1, f"Cannot create measured report: {exc}\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
