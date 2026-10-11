#!/usr/bin/env python3
"""Render the measured comparison JSON into charts, HTML, and research notes.

No examples or placeholder scores are substituted when results are missing.
All numerical reporting is derived solely from comparison.json.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEAM_ROLL_NUMBERS = "23bcs019, 23bcs007, 23bcs021"
MENTOR = "SUNIL CK"
PAPER = "https://doi.org/10.1016/j.patcog.2023.109528"
REPOSITORY = "https://github.com/LukasHedegaard/continual-skeletons"
VARIANT_IDS = ("baseline", "recovery", "rejection", "combined")
CONDITION_IDS = ("clean", "burst_missing", "limb_missing", "long_missing", "jitter")
COLORS = ("#475569", "#6d28d9", "#0f766e", "#c2410c")
SHORT_LABELS = {
    "clean": "Original poses",
    "burst_missing": "Short joint gaps",
    "limb_missing": "Short limb gaps",
    "long_missing": "Long joint gaps",
    "jitter": "Coordinate noise",
}


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f} pp"


def escaped(value):
    return html.escape(str(value), quote=True)


def validate(payload):
    if payload.get("status") != "complete":
        raise ValueError("The benchmark must report status='complete' before rendering")
    protocol = payload["protocol"]
    if protocol["test_count"] <= 0 or protocol["calibration_count"] <= 0:
        raise ValueError("Calibration and test counts must be positive")
    if {c["id"] for c in payload["conditions"]} != set(CONDITION_IDS):
        raise ValueError("Expected all five measured conditions, including noise")
    if {v["id"] for v in payload["variants"]} != set(VARIANT_IDS):
        raise ValueError("Expected all four benchmark variants")
    for condition in payload["conditions"]:
        if {r["variant"] for r in condition["results"]} != set(VARIANT_IDS):
            raise ValueError(f"Incomplete variants for {condition['id']}")
        for row in condition["results"]:
            if row["n_samples"] != protocol["test_count"]:
                raise ValueError("Variant sample count differs from the protocol")
            for field in ("accuracy", "macro_f1", "coverage", "accepted_accuracy"):
                value = row[field]
                if value is None and field == "accepted_accuracy":
                    continue
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError(f"Invalid {field} in {condition['id']}")
        paired = condition["paired_recovery"]
        values = [paired["accuracy_gain"], *paired["ci95"]]
        if len(values) != 3 or not all(math.isfinite(x) for x in values):
            raise ValueError("Missing or invalid paired confidence interval")
    return payload


def ordered(payload):
    by_id = {c["id"]: c for c in payload["conditions"]}
    return [by_id[key] for key in CONDITION_IDS]


def rows(condition):
    by_id = {row["variant"]: row for row in condition["results"]}
    return [by_id[key] for key in VARIANT_IDS]


def takeaways(payload):
    statements = []
    for condition in ordered(payload):
        paired = condition["paired_recovery"]
        gain = paired["accuracy_gain"]
        low, high = paired["ci95"]
        if gain > 0:
            outcome = "increased"
        elif gain < 0:
            outcome = "decreased"
        else:
            outcome = "did not change"
        interval = (
            "The interval includes zero, so this pilot does not establish a consistent gain."
            if low <= 0 <= high
            else "This interval concerns the sampled clips under this fixed corruption protocol; it is not evidence of universal superiority."
        )
        statements.append(
            f"{condition['label']}: recovery {outcome} all-input accuracy (difference {pp(gain)}) "
            f"(paired 95% interval {pp(low)} to {pp(high)}). {interval}"
        )
    return statements


def plot(payload, output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".build/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    conditions = ordered(payload)
    names = {v["id"]: v["label"] for v in payload["variants"]}
    positions = np.arange(len(conditions))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12.5, 5.4), layout="constrained")
    for index, variant in enumerate(VARIANT_IDS):
        values = [
            next(r for r in c["results"] if r["variant"] == variant)["accuracy"] * 100
            for c in conditions
        ]
        bars = ax.bar(
            positions + (index - 1.5) * width,
            values,
            width,
            label=names[variant],
            color=COLORS[index],
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
    ax.set_xticks(positions, [SHORT_LABELS[c["id"]] for c in conditions])
    ax.set_ylabel("Accuracy on all test clips (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Measured comparison across every test condition",
        loc="left",
        fontweight="bold",
        pad=16,
    )
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.savefig(output / "accuracy_comparison.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(16, 4.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    markers = ("o", "s", "^", "D")
    for ax, condition in zip(axes, conditions):
        for index, row in enumerate(rows(condition)):
            value = row["accepted_accuracy"]
            if value is not None:
                ax.scatter(
                    100 * row["coverage"],
                    100 * value,
                    color=COLORS[index],
                    marker=markers[index],
                    s=55,
                    alpha=0.85,
                    label=names[row["variant"]],
                    zorder=4 - index / 10,
                )
        ax.set_title(SHORT_LABELS[condition["id"]], fontsize=10, pad=10)
        ax.set_xlim(0, 104)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Accuracy among accepted clips (%)")
    fig.supxlabel(
        "Coverage: percentage of all test clips receiving a prediction", fontsize=11
    )
    fig.suptitle(
        "Rejection changes coverage — read both axes together",
        fontweight="bold",
        fontsize=14,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(output / "coverage_tradeoff.png", dpi=180, facecolor="white")
    plt.close(fig)


def recovery_explanation(payload):
    config = payload["calibration"]["recovery"]
    if config["beta"] == 0:
        chosen = "The selected setting has beta = 0, so the deployed repair holds the last observed position rather than extrapolating motion."
    else:
        chosen = "The selected setting updates a bounded velocity estimate and extrapolates it only during short gaps."
    guard = (
        "The optional torso-jump guard is disabled in these measured runs."
        if config.get("identity_jump_threshold") is None
        else f"The torso-jump guard resets a person slot above the configured threshold {config['identity_jump_threshold']}; it is not an identity tracker."
    )
    decay = (
        "Repair preserves the last observed confidence score."
        if config["confidence_decay"] == 1
        else f"Repair multiplies the last observed confidence by {config['confidence_decay']} for each missing frame."
    )
    gate = (
        f"Repair can start only when the last real observation has confidence at least {config['min_repair_confidence']}."
        if config.get("min_repair_confidence", 0) > 0
        else "There is no additional last-observation confidence gate for repair."
    )
    return " ".join([chosen, decay, gate, guard])


def markdown_results(payload, source_hash):
    protocol = payload["protocol"]
    names = {v["id"]: v["label"] for v in payload["variants"]}
    output = [
        "# Measured research results\n",
        f"**Team roll numbers:** {TEAM_ROLL_NUMBERS}\n\n**Mentor:** {MENTOR}\n",
        f"Generated from `results/research/comparison.json`, measured at **{payload['generated_at']}**. Source SHA256: `{source_hash}`.\n",
        f"The pilot uses **{protocol['calibration_count']} calibration clips** and **{protocol['test_count']} separate test clips** from real {protocol['dataset']} data. The classifier retains **{protocol['class_count']} output classes**; **{protocol['classes_present']} true classes** occur in this test subset. **{protocol['empty_clips_retained']} empty clips** are retained.\n",
        f"Base paper: [Continual Spatio-Temporal Graph Convolutional Networks]({PAPER}), *Pattern Recognition* 140 (2023), 109528. [Authors' code]({REPOSITORY}).\n",
        "## Complete measured table\n",
        "Accuracy and macro-F1 count abstentions as incorrect. Accepted accuracy is conditional on receiving a prediction and must always be read together with coverage. Macro-F1 averages the true classes present in this test subset; it is not a 400-class macro average.\n",
        "| Condition | Variant | All-input accuracy | Macro-F1 | Accepted accuracy | Coverage | Processing ms/clip |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in ordered(payload):
        for row in rows(condition):
            output.append(
                f"| {condition['label']} | {names[row['variant']]} | {pct(row['accuracy'])} | {pct(row['macro_f1'])} | {pct(row['accepted_accuracy'])} | {pct(row['coverage'])} | {row['latency_ms']:.2f} |"
            )
    output.extend(
        [
            "\n## Recovery compared with the original model\n",
            "These are paired accuracy differences in **percentage points**, using identical test clips and corruption masks. Rejection is excluded from this particular contrast.\n",
            "| Condition | Accuracy difference | Paired 95% interval | Corrected clips | Regressed clips |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for condition in ordered(payload):
        value = condition["paired_recovery"]
        output.append(
            f"| {condition['label']} | {pp(value['accuracy_gain'])} | {pp(value['ci95'][0])} to {pp(value['ci95'][1])} | {value['improved_clips']} | {value['regressed_clips']} |"
        )
    output.extend(
        [
            "",
            *["- " + sentence for sentence in takeaways(payload)],
            "\n## What was selected\n",
            f"Calibration selected **{payload['calibration']['selected_candidate']}**. {recovery_explanation(payload)}\n",
            "```json",
            json.dumps(payload["calibration"]["recovery"], indent=2),
            "```",
            "\n## Scope and limitations\n",
            *["- " + text for text in limitations(payload)],
            "\nThe HTML report and both charts are in `results/research`. Per-clip predictions and frozen calibration settings are separate benchmark outputs.\n",
        ]
    )
    return "\n".join(output)


def limitations(payload):
    protocol = payload["protocol"]
    return [
        "The pose coordinates and action labels are real public data. Missing-joint blocks and coordinate jitter are synthetic perturbations of those real poses. Physical occlusion in new camera recordings has not been validated.",
        "This is a sampled robustness pilot, not reproduction of the paper's full benchmark, and it does not establish state-of-the-art performance or worldwide methodological novelty.",
        "Calibration and test clips are disjoint within the upstream validation partition. That partition may have informed the authors' pretrained checkpoint selection. This is not a newly collected subject-independent test set.",
        "All methods use identical pretrained weights, preprocessing, clips and corruption masks. Repair and rejection settings were frozen on calibration data before test evaluation; the classifier was not retrained.",
        protocol["timing_scope"],
        "The research classifier is not integrated into a validated webcam pipeline. The existing MediaPipe six-activity demo is a separate system; its outputs are not results from this 400-class checkpoint.",
        "Kinetics uses OpenPose 18-joint 2D coordinates and confidence. MediaPipe 33-joint landmarks and NTU 25-joint 3D skeletons are not interchangeable inputs without an adapter and fresh validation.",
        "Person slots are ranked by detector confidence and can switch identities. Bounded extrapolation cannot reliably repair long gaps, abrupt motion, or identity swaps. Reliable visible observations are not denoised by this recovery method.",
        "A confidence threshold improves the reliability of accepted predictions only by a tradeoff with coverage; it does not prove the classifier itself became more accurate.",
    ]


def markdown_viva(payload):
    p = payload["protocol"]
    config = payload["calibration"]["recovery"]
    settings = payload["calibration"]
    return f"""# Research viva guide

**Team roll numbers:** {TEAM_ROLL_NUMBERS}

**Mentor:** {MENTOR}

## A short explanation of the project

We use an existing pretrained graph neural network to recognize human actions from body-joint sequences. We investigate what happens when a pose detector temporarily loses joints. Our extension repairs short gaps using earlier observations and can reject predictions when evidence is insufficient. We compare the original model and three ablations on identical real pose sequences with controlled corruption.

## What does the base model do?

The base paper is [Continual Spatio-Temporal Graph Convolutional Networks]({PAPER}), published in *Pattern Recognition* in 2023. Its code is at the [authors' repository]({REPOSITORY}). A skeleton is represented as a graph: joints are nodes and body connections are edges. Spatial graph convolutions combine information across joints; temporal convolutions learn how that information changes across frames. CoST-GCN reuses intermediate computation as new frames arrive, reducing redundant work in streaming inference.

This project uses the authors' pretrained **single joint-stream CoST-GCN\\*** checkpoint. The classifier accepts 18-joint 2D OpenPose data with x, y and confidence channels and produces {p['class_count']} activity scores. We did not train this classifier ourselves. Our benchmark uses the weight-compatible clip path and records a numerical equivalence check against the continual definition. Its timing is batch processing per 300-frame clip, not live camera frames per second.

## What did we add?

The extension is a causal, bounded pose-recovery module and a calibrated rejection policy. Causal means it uses current and earlier observations, never future observations. For each joint/person slot, an alpha-beta tracker maintains position and velocity. Alpha adjusts the estimated position when a new observation arrives; beta adjusts velocity. Missing joints can be filled only for a bounded number of frames, with bounded displacement and a configurable confidence multiplier. Long gaps remain invalid. Reliable observed coordinates pass through unchanged.

Calibration selected **{settings['selected_candidate']}**, with alpha **{config['alpha']}**, beta **{config['beta']}**, and maximum gap **{config['max_gap_frames']} frames**. {recovery_explanation(payload)} This is a practical extension and a measured experiment relative to this base paper, not a claim that interpolation, tracking, or rejection has never been studied before.

## Why compare four variants?

1. **Original model** measures the unchanged checkpoint.
2. **Joint recovery only** isolates whether repaired input improves recognition.
3. **Rejection only** measures the effect of withholding uncertain predictions without repairing input.
4. **Recovery + rejection** measures their combination.

Without these ablations, an apparent improvement could come only from declining to answer harder cases. Recovery-only accuracy is therefore the main paired contrast. Rejection thresholds were selected separately for the original and recovered inputs on calibration data.

## How is the comparison fair?

We use {p['calibration_count']} calibration clips and {p['test_count']} different test clips. Selection is deterministic from the official validation partition, without selecting clips because predictions are correct. The classifier still predicts all {p['class_count']} classes; {p['classes_present']} true classes occur in this sampled test set. All variants receive the same clips and identical corruption masks. We freeze settings before evaluating test labels. The {p['empty_clips_retained']} empty test clips remain in the analysis.

The conditions are original poses, short joint gaps, short whole-limb gaps, longer joint gaps and coordinate noise. The pose data are real, but the added masks/noise are simulated. This evaluates controlled pose corruption, not demonstrated robustness to physical occlusion in a webcam scene.

## What do the metrics mean?

- **All-input accuracy:** correct predictions divided by every test clip; abstentions count as errors.
- **Macro-F1:** the average class F1 over true classes represented in this subset, giving rare and frequent represented classes equal weight.
- **Coverage:** the fraction of all clips receiving a prediction.
- **Accepted accuracy:** correctness among accepted clips only; always report it with coverage.
- **Percentage-point difference:** recovery accuracy minus original-model accuracy, rather than relative percentage improvement.
- **Paired 95% bootstrap interval:** resample the same clip pairs to estimate uncertainty in the recovery-minus-baseline difference. An interval crossing zero does not establish a consistent improvement in this pilot.

The rejection thresholds are **{settings['baseline_threshold']:.3f}** for original inputs and **{settings['recovery_threshold']:.3f}** for repaired inputs. They maximize calibration accepted accuracy subject to at least 80% calibration coverage. That constraint does not guarantee 80% coverage on every unseen corruption condition.

## What should we say about the actual results?

Use the complete tables in [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md), including losses and unchanged results. Never quote only the condition with the largest gain. A short-gap repair method may help missing joints while leaving noisy observed coordinates unchanged or making some cases worse. A negative result defines where this implementation is unsuitable; it must remain in the presentation.

## What are the practical limits?

The pilot is smaller than the paper's full test set. Upstream validation clips may have influenced the authors' checkpoint selection. The repair uses confidence-ranked person slots rather than stable identities, so a slot can change people. Long gaps cannot be reconstructed reliably, pose-only classification misses object/context information, and measured batch timings exclude pose estimation and browser overhead. We do not claim full benchmark reproduction, state-of-the-art accuracy, medical suitability, or validated real-time webcam performance.

The earlier MediaPipe six-activity dashboard is a separate demo. The new 400-class research checkpoint expects OpenPose's exact topology and preprocessing; merely renaming joints or supplying MediaPipe landmarks would not establish correctness. A future webcam adapter needs separate data and evaluation.

## What is the research question?

Can bounded causal repair improve a fixed pretrained skeleton-action classifier under short pose-detection gaps, and what accuracy-versus-coverage tradeoff results from calibrated rejection? We answer that question with an unchanged model, controlled perturbations, ablations, held-out clips and uncertainty intervals.
"""


def render_html(payload, output, source_hash):
    protocol = payload["protocol"]
    variant_names = {v["id"]: v["label"] for v in payload["variants"]}
    table = []
    contrasts = []
    for condition in ordered(payload):
        for index, row in enumerate(rows(condition)):
            table.append(
                "<tr>"
                + (
                    f'<th rowspan="4">{escaped(condition["label"])}</th>'
                    if index == 0
                    else ""
                )
                + f'<td>{escaped(variant_names[row["variant"]])}</td><td>{pct(row["accuracy"])}</td>'
                + f'<td>{pct(row["macro_f1"])}</td><td>{pct(row["accepted_accuracy"])}</td>'
                + f'<td>{pct(row["coverage"])}</td><td>{row["latency_ms"]:.2f}</td></tr>'
            )
        paired = condition["paired_recovery"]
        tone = (
            "up"
            if paired["accuracy_gain"] > 0
            else "down" if paired["accuracy_gain"] < 0 else "neutral"
        )
        contrasts.append(
            f'<tr><th>{escaped(condition["label"])}</th><td class="{tone}">{pp(paired["accuracy_gain"])}</td><td>{pp(paired["ci95"][0])} to {pp(paired["ci95"][1])}</td><td>{paired["improved_clips"]}</td><td>{paired["regressed_clips"]}</td></tr>'
        )

    def embed(filename):
        return (
            "data:image/png;base64,"
            + base64.b64encode((output / filename).read_bytes()).decode()
        )

    conditions = {c["id"]: c for c in payload["conditions"]}
    baseline_clean = rows(conditions["clean"])[0]["accuracy"]
    brief = "".join(f"<li>{escaped(text)}</li>" for text in takeaways(payload))
    limits = "".join(f"<li>{escaped(text)}</li>" for text in limitations(payload))
    settings = escaped(json.dumps(payload["calibration"], indent=2))
    equivalence = escaped(json.dumps(payload["equivalence"], indent=2))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PoseLab · Measured research results</title>
<style>
:root{{--ink:#172033;--muted:#586779;--line:#dce3eb;--purple:#6d28d9}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}}main{{max-width:1240px;margin:auto;padding:48px 28px 72px}}header{{background:#172033;color:white;padding:40px;border-radius:22px}}.eyebrow{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#c4b5fd;font-weight:750}}h1{{font-size:clamp(30px,4vw,48px);line-height:1.12;max-width:900px;margin:16px 0}}header p{{max-width:880px;color:#d8e1ec}}.meta{{font-size:13px;color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}}.card,section{{background:white;border:1px solid var(--line);border-radius:16px}}.card{{padding:20px}}.card strong{{display:block;font-size:28px;line-height:1.2;margin:10px 0;color:var(--purple)}}.card span{{font-size:13px;color:var(--muted)}}section{{margin-top:22px;padding:28px}}h2{{font-size:24px;line-height:1.3;margin:0 0 12px}}h3{{margin-top:24px}}p{{margin:10px 0}}img{{display:block;width:100%;height:auto;margin:18px 0}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5}}th,td{{padding:11px 12px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}}th:first-child,td:first-child{{text-align:left}}thead th{{color:#4b5563;background:#f7f9fc;font-size:12px}}tbody th{{text-align:left;min-width:175px;max-width:245px}}td:nth-child(2){{text-align:left}}.up{{color:#116d50;font-weight:750}}.down{{color:#b42318;font-weight:750}}.neutral{{color:#475569;font-weight:750}}li{{margin:9px 0}}a{{color:#6d28d9}}pre{{overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f9fc;padding:18px;border-radius:10px;font-size:12px}}summary{{cursor:pointer;font-weight:650;margin:12px 0}}.note{{border-left:4px solid #a78bfa;padding:12px 18px;background:#f5f3ff;border-radius:0 8px 8px 0}}footer{{font-size:12px;overflow-wrap:anywhere;margin-top:24px;color:var(--muted)}}@media(max-width:760px){{main{{padding:20px 12px}}header,section{{padding:22px}}.cards{{grid-template-columns:repeat(2,1fr)}}}}@media print{{body{{background:white}}main{{padding:0}}header{{border-radius:0}}section{{break-inside:avoid}}.scroll{{overflow:visible}}table{{font-size:10px}}}}
</style></head><body><main>
<header><div class="eyebrow">PoseLab / Mini Project-II / Measured pilot</div><h1>Can short-gap pose recovery improve activity recognition?</h1><p>A fixed pretrained CoST-GCN* classifier, real public pose sequences, and a controlled comparison that includes gains, regressions and rejected predictions.</p><p><strong>Team roll numbers:</strong> {escaped(TEAM_ROLL_NUMBERS)}<br><strong>Mentor:</strong> {escaped(MENTOR)}</p><p>Measured at {escaped(payload['generated_at'])}. No example scores or simulated activity labels.</p></header>
<div class="cards"><div class="card"><span>Disjoint calibration / test</span><strong>{protocol['calibration_count']} / {protocol['test_count']}</strong><span>Real clips from upstream validation</span></div><div class="card"><span>Original model · original poses</span><strong>{pct(baseline_clean)}</strong><span>All-input top-1 accuracy</span></div><div class="card"><span>Short joint gaps · recovery difference</span><strong>{pp(conditions['burst_missing']['paired_recovery']['accuracy_gain'])}</strong><span>Percentage points, paired clips</span></div><div class="card"><span>Test classes / classifier outputs</span><strong>{protocol['classes_present']} / {protocol['class_count']}</strong><span>Full output head retained</span></div></div>
<section><h2>The measured result</h2><ul>{brief}</ul><p class="note">Added joint masks and coordinate noise are synthetic stress conditions applied to <strong>real pose data</strong>. These results do not validate physical camera occlusion or a new webcam classifier.</p><img src="{embed('accuracy_comparison.png')}" alt="Accuracy chart for all five conditions and four variants"></section>
<section><h2>Every condition, every variant</h2><p>All-input accuracy and macro-F1 count abstentions as errors. Accepted accuracy describes only clips receiving predictions; coverage shows how many clips remain. Macro-F1 averages the {protocol['classes_present']} true classes present in this test subset.</p><div class="scroll"><table><thead><tr><th>Condition</th><th>Variant</th><th>All-input accuracy</th><th>Macro-F1</th><th>Accepted accuracy</th><th>Coverage</th><th>ms / clip</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><p class="meta">{escaped(protocol['timing_scope'])}</p></section>
<section><h2>Recovery gains and regressions</h2><p>This contrast isolates recovery, without rejection. Intervals use paired clip resampling; they describe this sampled protocol rather than guaranteed performance on new people or cameras.</p><div class="scroll"><table><thead><tr><th>Condition</th><th>Accuracy difference</th><th>Paired 95% interval</th><th>Corrected clips</th><th>Regressed clips</th></tr></thead><tbody>{''.join(contrasts)}</tbody></table></div></section>
<section><h2>The cost of rejecting uncertain predictions</h2><p>Moving toward higher accepted accuracy while reducing coverage is a tradeoff. A rejected clip is still counted as incorrect in all-input accuracy. Overlapping markers indicate identical or very similar measurements.</p><img src="{embed('coverage_tradeoff.png')}" alt="Accepted accuracy versus coverage for all variants under each condition"></section>
<section><h2>What changed in the system</h2><p>The authors' pretrained weights remain unchanged. Our module estimates position and velocity from past joint observations, repairs bounded gaps, carries a configurable confidence score during repair, and optionally withholds low-confidence predictions. Calibration selected <strong>{escaped(payload['calibration']['selected_candidate'])}</strong>.</p><p>{escaped(recovery_explanation(payload))}</p><p>{escaped(protocol['description'])}</p><details><summary>Frozen calibration settings and scores</summary><pre>{settings}</pre></details><details><summary>Clip / continual equivalence check</summary><pre>{equivalence}</pre></details></section>
<section><h2>Limits of the evidence</h2><ul>{limits}</ul></section>
<footer><p>Base paper: <a href="{PAPER}">Continual Spatio-Temporal Graph Convolutional Networks</a>, Pattern Recognition 140 (2023), 109528. <a href="{REPOSITORY}">Authors' repository</a>.</p><p>Upstream commit: {escaped(protocol['upstream_commit'])}<br>Checkpoint SHA256: {escaped(protocol['checkpoint_sha256'])}<br>Comparison JSON SHA256: {source_hash}</p><p>Hardware: {escaped(protocol['hardware'])}. Device: {escaped(protocol['device'])}. Batch size: {protocol['batch_size']}. Test clips retained without pose: {protocol['empty_clips_retained']}.</p></footer>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "results/research/comparison.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results/research")
    parser.add_argument("--docs", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.exit(
            1,
            f"Measured comparison is missing: {args.input}\nRun research.benchmark first; no report or substitute results were generated.\n",
        )
    try:
        raw = args.input.read_bytes()
        payload = validate(json.loads(raw))
        source_hash = hashlib.sha256(raw).hexdigest()
        args.out.mkdir(parents=True, exist_ok=True)
        args.docs.mkdir(parents=True, exist_ok=True)
        plot(payload, args.out)
        (args.out / "report.html").write_text(
            render_html(payload, args.out, source_hash)
        )
        (args.docs / "RESEARCH_RESULTS.md").write_text(
            markdown_results(payload, source_hash)
        )
        (args.docs / "RESEARCH_VIVA.md").write_text(markdown_viva(payload))
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(1, f"Cannot render measured research report: {exc}\n")
    print(f"Saved standalone report and two charts to {args.out}")
    print(f"Saved measured results and viva guide to {args.docs}")


if __name__ == "__main__":
    main()
