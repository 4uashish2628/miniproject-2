#!/usr/bin/env python3
"""Build the project-specific presentation study guide from verified evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results/research/comparison.json").read_text())
assert DATA["status"] == "complete"
DIGEST = hashlib.sha256(
    (ROOT / "results/research/comparison.json").read_bytes()
).hexdigest()
if DIGEST != "346808497b94c5ee389f38629a4e46429b4895611430b73fae048e29643986b9":
    raise ValueError(
        "The measured experiment changed; review the guide's worked examples before regenerating it."
    )
PAGES = []


def page(title, subtitle, *blocks):
    PAGES.append((title, subtitle, blocks))


def h(text):
    return ("heading", text)


def p(text):
    return ("paragraph", text)


def callout(text):
    return ("callout", text)


def table(rows, widths):
    return ("table", rows, widths)


def qa(question, answer):
    return ("qa", question, answer)


page(
    "Own the presentation",
    "START HERE / YOUR PROJECT IN ONE MINUTE",
    p(
        "<b>PoseLab: Robust Human Activity Recognition using Pose Estimation and Missing-Joint Recovery</b><br/>Indian Institute of Information Technology Dharwad | Mini Project-II | Aug-Nov 2026<br/><b>Team:</b> 23bcs019, 23bcs007, 23bcs021<br/><b>Mentor:</b> SUNIL CK"
    ),
    callout(
        "We use a pretrained skeleton-action model and ask whether repairing brief missing-joint gaps improves recognition. We keep the classifier fixed, add a confidence-gated causal hold, and compare four variants on the same real pose clips. The strongest evidence is for short gaps: joint-dropout accuracy rises from 11.50% to 15.25%. This is a controlled robustness pilot, with failures and uncertainty reported."
    ),
    h("The five things every team member must explain"),
    p(
        "1. Pose estimation finds body joints; activity recognition classifies their movement.<br/>2. A graph model combines connected joints and changes across time.<br/>3. Our selected method holds a previously reliable joint for at most five missing frames.<br/>4. We chose settings on 200 calibration clips and evaluated 400 different clips.<br/>5. Better short-gap results do not prove that every condition or a live webcam improved."
    ),
    h("Read this in three passes"),
    table(
        [
            ["Pass", "Pages", "What to do"],
            [
                "Learn",
                "2-9",
                "Draw the pipeline and explain each decision without reading.",
            ],
            [
                "Rehearse",
                "10-13",
                "Practise the eight slides, dashboard demo and code tour.",
            ],
            [
                "Defend",
                "14-18",
                "Answer viva questions, then use the final revision sheet.",
            ],
        ],
        [65, 55, 387],
    ),
    p(
        "This guide matches the current research deck and measured run. The scripts are speaking prompts, not text to memorise word for word. Read the linked paper as well. If a question goes beyond the experiment, distinguish what is measured from what you would test next."
    ),
    p(
        "<b>Evidence snapshot:</b> experiment dated 11 October 2026; guide prepared 12 October 2026. The earlier six-activity MediaPipe lab is separate and must not be presented as this research model."
    ),
)

page(
    "The concepts, in plain language",
    "FOUNDATIONS / LEARN THE MEANING BEFORE THE ABBREVIATION",
    table(
        [
            ["Term", "Meaning in this project"],
            [
                "Pose estimation",
                "Finding positions of body joints in an image. The research data already contain OpenPose estimates; our benchmark does not run a new image detector.",
            ],
            [
                "Activity recognition",
                "Mapping a sequence of poses to an action label. One frame may look similar during sitting down and standing up; movement helps distinguish them.",
            ],
            [
                "Skeleton",
                "A compact body representation: joints with coordinates and confidence, plus a known connection structure.",
            ],
            [
                "Graph / node / edge",
                "A graph represents relationships. Each body joint is a node. A shoulder-to-elbow connection is an edge.",
            ],
            [
                "Spatial / temporal",
                "Spatial means relationships between joints within a frame. Temporal means how features change across frames.",
            ],
            [
                "GCN",
                "A graph convolutional network learns how to combine features from connected joints. It learns useful patterns rather than using a hand-written rule for each action.",
            ],
            [
                "Pretrained checkpoint",
                "Saved learned model parameters. We load the authors' parameters; we do not train or fine-tune the classifier in this experiment.",
            ],
            [
                "Inference / calibration",
                "Inference makes a prediction. Here calibration selects repair settings and rejection thresholds using labelled development clips; it is not neural-network training.",
            ],
            [
                "Ablation",
                "A controlled comparison that removes or adds one component to identify what that component contributes.",
            ],
            [
                "Causal / imputation",
                "Causal repair uses current and past observations only. Imputation fills a missing value with an estimate, not recovered ground truth.",
            ],
        ],
        [112, 395],
    ),
    callout(
        "A wrist position alone is ambiguous. The wrist relative to the elbow, and how both move over time, gives the classifier richer evidence. Draw three connected joints in three consecutive frames to explain this."
    ),
    p(
        "<b>Do not mix confidence types:</b> joint confidence describes pose-detector evidence; the classifier's maximum softmax score describes its preference among action classes. Neither automatically equals the probability of being correct."
    ),
)

page(
    "The paper and our extension",
    "READING MAP / SEPARATE THE AUTHORS' WORK FROM OURS",
    p(
        "<b>Base reference [1]:</b> Lukas Hedegaard, Negar Heidari and Alexandros Iosifidis, <i>Continual Spatio-Temporal Graph Convolutional Networks</i>, Pattern Recognition 140 (2023), 109528. DOI: 10.1016/j.patcog.2023.109528."
    ),
    p(
        "The paper addresses repeated computation when graph-based action models repeatedly process overlapping windows. Its continual formulation reuses intermediate work as frames arrive. It studies CoST-GCN and related attention models, weight transfer, and efficiency. Read the abstract and introduction for motivation; Section II-A for graphs; Sections III-A, III-B and III-D for cached convolution, aligned residuals and construction; Section III-F for limits; and Section IV for evaluation. Figures 2, 3 and 5 help explain the architecture. [1]"
    ),
    h("What to say about the research gap"),
    callout(
        "Our project asks a complementary question: for a fixed pretrained skeleton classifier, can a small causal input-repair module reduce the damage caused by short pose-detection gaps? We evaluate that extension with matched inputs, ablations and uncertainty. We are not claiming that missing-data repair was invented by us."
    ),
    table(
        [
            ["Reused from the authors", "Added in this project"],
            [
                "Model definitions, pretrained single joint-stream checkpoint, original Kinetics preprocessing.",
                "Confidence-gated short-gap recovery, slot-reset safeguard and optional calibrated rejection.",
            ],
            [
                "400-class action output and 18-joint OpenPose layout.",
                "Deterministic pilot split, controlled stress conditions, paired comparisons and saved evidence.",
            ],
            [
                "Their continual model formulation and its research motivation.",
                "An inference adapter, equivalence check, dashboard, reports and explanation of failures.",
            ],
        ],
        [253, 254],
    ),
    h("The exact Q1 wording"),
    p(
        "Say: 'The paper was published in Pattern Recognition; our documented journal-quartile evidence is SCImago/SJR 2024 Q1.' Q1 refers to a journal's category/year ranking, not a grade attached to an individual article. Do not call arXiv a Q1 journal, or change the claim to publication-year JCR without matching evidence. The 2024 SJR listing is also available in a university-library copy [3]. Bring the ranking evidence with the DOI; a supervisor-specific ranking requirement still needs to match it."
    ),
    p(
        "The authors' repository is linked in [2]. Their best multi-stream or full-benchmark numbers are not directly comparable to our sampled single-stream pilot. Our baseline is the same checkpoint evaluated by us on the same damaged clips."
    ),
)

page(
    "Trace one clip through the system",
    "ARCHITECTURE / THE ACTUAL RESEARCH PATH",
    callout(
        "Public OpenPose skeleton clip -> original preprocessing -> fixed corruption condition -> optional causal recovery -> unchanged pretrained graph classifier -> optional rejection -> metrics and saved dashboard results."
    ),
    h("The input shape you should know"),
    p(
        "The model input is <b>N x 3 x 300 x 18 x 2</b>. N is the batch size; the three channels are x, y and joint confidence; 300 is the frame-slot window; 18 is the OpenPose joint count; two is the number of person slots. A single clip has shape 3 x 300 x 18 x 2. Short clips include zero padding. Recovery processes the real clip length rather than repairing padding."
    ),
    h("The network explanation"),
    p(
        "Think of a spatial layer as gathering information from neighbouring joints, transforming it with learned weights, and combining it into new features. Temporal layers combine those features across nearby frames. Deeper layers represent longer and more complex motion patterns. Pooling summarises the features, and the final classifier produces 400 class scores. We use the authors' joint stream, not an ensemble of joints, bones and motion streams. [2; local upstream adapter]"
    ),
    ("diagram",),
    h("Three details that prevent common mistakes"),
    p(
        "<b>Continual inference is not continual learning.</b> Inference reuses computation/state when inputs arrive; learning would update weights. Our pretrained weights remain fixed.<br/><b>Our measured execution is clip-based.</b> The adapter uses a weight-compatible clip path and checks its output against continual and step execution. It does not establish live-camera FPS.<br/><b>Person slots are not identities.</b> People are ranked by pose confidence. A slot may switch people; the reset guard is only a conservative safeguard."
    ),
    h("Why not feed MediaPipe joints directly?"),
    p(
        "The research checkpoint expects a specific 18-joint topology, ordering, coordinates and preprocessing. The older webcam lab uses 33 MediaPipe landmarks and a different classifier. Mapping their names alone does not validate a compatible input distribution. A new adapter and separate camera evaluation are future work."
    ),
    p(
        "<b>Evidence to open:</b> research/data.py for the original feeder, research/upstream.py for checkpoint loading and equivalence, and results/research/equivalence.json for the recorded check."
    ),
)

page(
    "Explain our recovery on a whiteboard",
    "METHOD / THE SELECTED CONFIDENT_HOLD_5 SETTING",
    p(
        "For each joint and person slot, maintain the last real position, its confidence and the number of consecutive missing frames. At a reliable new observation, update this memory. If the joint disappears briefly, reuse that earlier position only when the confidence and gap conditions allow it."
    ),
    table(
        [
            ["Rule", "Actual selected setting"],
            [
                "What counts as observed?",
                "Current joint confidence > 0. Observed coordinates pass through unchanged; min_confidence is 0.",
            ],
            [
                "When can repair start?",
                "There is a previous real observation with confidence at least 0.5.",
            ],
            [
                "How long is the hold?",
                "At most five consecutive missing frames; the sixth remains invalid.",
            ],
            [
                "What position is filled?",
                "The last observed position. Alpha = 1 and beta = 0, so the selected tracker does not extrapolate velocity.",
            ],
            [
                "What confidence is filled?",
                "Last real confidence multiplied by 0.9 to the power of the gap length.",
            ],
            [
                "When is a slot reset?",
                "Median jump > 0.15 across at least two commonly visible torso anchors in adjacent frames. This can miss a swap or reset on fast motion.",
            ],
        ],
        [132, 375],
    ),
    h("Worked example: a wrist disappears"),
    p(
        "Suppose the last real wrist is at (0.20, -0.10), confidence 0.80. The next two frames are missing. Fill the same position with scores 0.80 x 0.9 = <b>0.72</b> and 0.80 x 0.9 squared = <b>0.648</b>. At five missing frames the score is about <b>0.4724</b>. Frame six stays zero. A new observed wrist immediately supplies its own coordinates again."
    ),
    p(
        "The 0.5 gate applies to the <b>last real</b> observation, not the decayed value. Therefore a fifth-frame filled score below 0.5 can still be valid under the rule. A currently visible joint with score 0.2 passes through, but cannot seed later repair. Never claim that all low-confidence visible points are removed."
    ),
    callout(
        "This works only if a recently reliable position is still a useful approximation. Brief gaps can benefit; rapid motion makes the held position stale. Confidence decay expresses reduced evidence to the network, but is not a calibrated probability of the repaired coordinate being correct."
    ),
    p(
        "The general code also supports velocity-based candidates and speed/displacement limits. Those are not the selected intervention: beta = 0 makes the chosen method a bounded hold. Read CausalPoseRecovery.step in research/recovery.py."
    ),
)

page(
    "How we made the comparison fair",
    "EXPERIMENT / WHAT WAS FIXED AND WHAT WAS CHANGED",
    p(
        "<b>Data:</b> 600 real public Kinetics-skeleton clips from the upstream official validation partition. We use 200 for choosing settings and 400 disjoint clips for extension evaluation. The model retains all 400 outputs; 251 true classes occur in the evaluation subset. Two empty clips remain included."
    ),
    p(
        "<b>Selection:</b> sort clips by SHA256 of seed 2026 plus clip ID, then allocate calibration and evaluation clips. Selection does not depend on whether a prediction is right, and there is no empty-pose or class filtering. This is a deterministic clip split, not a participant-independent split or a full dataset reproduction."
    ),
    table(
        [
            ["Condition", "What it probes"],
            [
                "Original poses",
                "Preservation of baseline behaviour, including natural detector gaps.",
            ],
            [
                "Short joint gaps",
                "Target 25% missing observed joint-frames, using 5-frame blocks.",
            ],
            [
                "Short limb gaps",
                "Target 25% missing observed joint-frames using limb groups and 5-frame blocks.",
            ],
            [
                "Long joint gaps",
                "Target 25% missing observed joint-frames using 15-frame blocks.",
            ],
            [
                "Coordinate noise",
                "Gaussian coordinate jitter with standard deviation 0.02.",
            ],
        ],
        [124, 383],
    ),
    p(
        "The coordinates and labels are real. Added masks and noise are controlled synthetic damage. '25%' is a target over observed joint-frame entries; realised dropout can vary with group/block boundaries and availability. Blocks can overlap or join, so a five-frame block does not guarantee every resulting gap is exactly five frames. Recovery receives neither the clean target nor the hidden corruption mask."
    ),
    table(
        [
            ["Variant", "Purpose"],
            ["Original model", "Unchanged checkpoint, no added recovery or rejection."],
            ["Recovery only", "Isolate input repair while answering every clip."],
            ["Rejection only", "Isolate withholding uncertain predictions."],
            [
                "Recovery + rejection",
                "Measure the combined pipeline and its coverage cost.",
            ],
        ],
        [137, 370],
    ),
    p(
        "<b>Calibration:</b> seven repair candidates were compared across original poses, short joint gaps and short limb gaps. The highest mean all-input accuracy selected confident_hold_5. There were three development rounds on the same calibration set, retained in the records. Thresholds were selected separately for raw/repaired predictions, then settings and implementation hashes were frozen before evaluation. All variants share checkpoint, preprocessing, clips and corruption masks."
    ),
)

conditions = DATA["conditions"]
labels = {
    "clean": "Original poses",
    "burst_missing": "Short joint gaps",
    "limb_missing": "Short limb gaps",
    "long_missing": "Long joint gaps",
    "jitter": "Coordinate noise",
}


def metric(condition, variant):
    return next(r for r in condition["results"] if r["variant"] == variant)


rows = [["Condition", "Original", "Recovery", "Gain", "Paired 95% interval"]]
for c in conditions:
    d = c["paired_recovery"]
    rows.append(
        [
            labels[c["id"]],
            f'{metric(c,"baseline")["accuracy"]*100:.2f}%',
            f'{metric(c,"recovery")["accuracy"]*100:.2f}%',
            f'{d["accuracy_gain"]*100:+.2f} pp',
            f'{d["ci95"][0]*100:+.2f} to {d["ci95"][1]*100:+.2f} pp',
        ]
    )
page(
    "Know the results, not just the chart",
    "MEASURED EVIDENCE / SAME 400 CLIPS IN EACH ROW",
    table(rows, [124, 66, 70, 65, 182]),
    h("The main result, with arithmetic"),
    p(
        "For short joint gaps, the original model gets <b>46/400</b> correct; recovery gets <b>61/400</b>. The difference is 15 correct clips, or <b>3.75 percentage points</b>. Recovery corrects 21 baseline errors but also breaks six previously correct predictions: 21 - 6 = 15. This explains why both improvements and regressions must be shown."
    ),
    p(
        "The relative increase is (15.25 - 11.50) / 11.50, about 32.6%. If used, label it explicitly as relative. For the main talk, prefer the two absolute scores and +3.75 percentage points; saying 'accuracy improved by 32.6%' alone is misleading."
    ),
    h("The second result and the limits"),
    p(
        "Short limb gaps improve from <b>61/400 to 74/400</b>: 20 corrected clips and seven regressions, net +13, or +3.25 percentage points. Both short-gap intervals exclude zero in this pilot. Original poses and long gaps each gain only one net correct clip. Noise gains six, but its interval touches zero. These other conditions do not establish a consistent gain."
    ),
    p(
        "Long-gap macro-F1 falls from <b>13.21% to 12.95%</b> even though accuracy rises slightly. Different metrics capture different behaviour. Keep this result visible. Original poses are detector estimates, not perfect ground truth, so natural gaps can still be repaired."
    ),
    h("How to defend the low accuracy"),
    p(
        "A 400-class pose-only task is difficult: skeletons omit objects and scene context, and detections can be incomplete. Those are plausible contributors, not causes isolated by this experiment. The measured clean baseline is 28.75%, and corrupted accuracy remains low. Our claim is a matched short-gap robustness gain; the model is not ready for reliable general deployment."
    ),
    p(
        "<b>Processing cost:</b> short-joint processing rises from 30.87 to 43.58 ms per 300-frame clip; short-limb processing from 31.04 to 43.68 ms. Roughly 13 ms extra here is a cost, not a speed improvement. These batch timings exclude pose estimation, transfers, I/O and browser work."
    ),
)

page(
    "Metrics, rejection and uncertainty",
    "DEFENCE / WHY THE DENOMINATOR MATTERS",
    table(
        [
            ["Metric", "Definition and interpretation"],
            [
                "All-input accuracy",
                "Correct / all evaluation clips. Rejected predictions count as incorrect.",
            ],
            [
                "Coverage",
                "Accepted clips / all clips. This measures how often an answer is returned, not how accurate the joints are.",
            ],
            [
                "Accepted accuracy",
                "Correct accepted predictions / accepted clips. Read it together with coverage; it can rise just by rejecting difficult cases.",
            ],
            [
                "Precision / recall / F1",
                "For a class: precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN). F1 balances precision and recall.",
            ],
            [
                "Macro-F1",
                "Mean F1 across the 251 true classes represented in this subset. Each represented class gets equal weight. It is not a macro average over all 400 outputs.",
            ],
        ],
        [122, 385],
    ),
    h("Actual rejection example: short joint gaps"),
    p(
        "Recovery alone: <b>61/400 = 15.25%</b>, answering all clips.<br/>Recovery plus rejection: <b>314/400 = 78.50% coverage</b>, with <b>60/314 = 19.11% accepted accuracy</b> and <b>60/400 = 15.00% all-input accuracy</b>. It removes 86 answers, including one correct answer. It did not make the underlying recovered classifier more accurate."
    ),
    p(
        "The selected action-score threshold is <b>0.08</b> for both variants. Accept only if the maximum softmax score reaches the threshold and some genuinely observed pose exists. Threshold search maximises calibration accepted accuracy subject to at least 80% pooled calibration coverage. That is not an 80% guarantee on every test condition. 'Calibrated rejection' means a selected operating threshold, not probability-calibrated softmax scores."
    ),
    h("Explain a paired bootstrap in 30 seconds"),
    p(
        "For each clip record +1 if recovery alone fixes it, -1 if it breaks it, and 0 otherwise. Resample these matched clip differences with replacement 3,000 times; take the 2.5th and 97.5th percentiles of the mean differences. Keeping the pair together accounts for both methods seeing the same clip. An interval containing zero leaves the direction uncertain under this procedure."
    ),
    p(
        "These are per-condition intervals with no correction for five comparisons. They capture resampling uncertainty for this subset and fixed corruption protocol, not uncertainty over every seed, subject, camera or deployment. Repeated calibration on the same 200 clips can also overfit development decisions; a larger independent study would strengthen the evidence."
    ),
)

page(
    "Make a precise contribution claim",
    "LIMITATIONS / CONFIDENT DOES NOT MEAN EXAGGERATED",
    table(
        [
            ["Avoid saying", "Say this instead"],
            [
                "We beat the original paper.",
                "Our extension improves this unchanged checkpoint on our matched short-gap pilot.",
            ],
            [
                "We built/trained CoST-GCN.",
                "We reused the authors' architecture and pretrained weights, and implemented/evaluated the surrounding extension.",
            ],
            [
                "Our detector recovers true joints.",
                "Our module supplies a bounded estimate for missing input; we measured action recognition, not ground-truth joint reconstruction error.",
            ],
            [
                "We solved occlusion.",
                "We tested synthetic missing-joint blocks on real poses. Physical occlusion needs new camera experiments.",
            ],
            [
                "We used motion prediction.",
                "The chosen configuration holds the last reliable joint; velocity-based candidates were considered but not selected.",
            ],
            [
                "It works in real time.",
                "The repair is causal; our measured benchmark is clip-based. End-to-end live latency was not established.",
            ],
            [
                "Noise was removed.",
                "Visible coordinates pass through. We did not demonstrate denoising.",
            ],
            [
                "Confidence is the chance of being right.",
                "It is a model or detector score; probability calibration was not established.",
            ],
            [
                "The webcam proves these results.",
                "The webcam lab is a separate prototype. The research dashboard displays recorded measured evidence.",
            ],
        ],
        [190, 317],
    ),
    h("The strongest contribution statement"),
    callout(
        "The contribution is a small, interpretable robustness extension and a reproducible controlled evaluation: confidence-gated causal holding, bounded repair, a slot reset, calibrated rejection, four ablations and paired results. The strongest evidence concerns short gaps; the study also exposes limited long-gap value, regressions and coverage costs."
    ),
    p(
        "Next steps, in order: test physically occluded recordings with compatible pose extraction; evaluate more clips and corruption seeds; measure pose error where ground truth exists; compare against additional standard imputation baselines; validate identity tracking; then integrate and time an end-to-end camera pipeline. These are proposals, not completed results."
    ),
)

page(
    "Your speaking script: slides 1-4",
    "REHEARSAL / AIM FOR ABOUT 8 MINUTES INCLUDING THE DEMO",
    qa(
        "Slide 1 - Title | about 25 seconds",
        "Good morning. Our project is robust human activity recognition using pose estimation and missing-joint recovery. Our team is 23bcs019, 23bcs007 and 23bcs021, guided by SUNIL CK. We investigate whether a small input-repair module can improve an existing action classifier when pose joints briefly disappear.",
    ),
    qa(
        "Slide 2 - Base paper | about 45 seconds",
        "Our base paper is Continual Spatio-Temporal Graph Convolutional Networks, published in Pattern Recognition in 2023. It focuses on efficient skeleton-action inference. We use its pretrained single joint-stream model, with 18 OpenPose joints and 400 action outputs. We retain the checkpoint and add a missing-joint recovery stage. This lets us study robustness without retraining a large model.",
    ),
    qa(
        "Slide 3 - The intervention | about 65 seconds",
        "A pose detector may briefly lose a wrist, ankle or limb. Our selected method remembers the last real observation and holds its position for at most five missing frames, provided its confidence was at least 0.5. The filled confidence decays by 0.9 each frame. It uses no future frames. The selected beta is zero, so this is a hold, not motion extrapolation. A torso-jump guard can reset stale state, and an optional classifier-score threshold can withhold uncertain predictions.",
    ),
    qa(
        "Slide 4 - Fair comparison | about 55 seconds",
        "We selected 200 calibration clips and 400 disjoint evaluation clips from real Kinetics-skeleton validation data. Settings were chosen on calibration clips and frozen before evaluation. The classifier still has 400 outputs; 251 labels occur in our subset. Every variant sees the same checkpoint, clips and corruption masks. We compare original input, recovery only, rejection only and their combination across five conditions. The input poses are real; added masks and noise are synthetic stress tests.",
    ),
    h("What to point at"),
    p(
        "Slide 2: point to the reused model and the extension separately. Slide 3: trace one missing-joint gap with your finger. Slide 4: emphasise the split and identical conditions before showing any gains. If time is short, shorten background; keep the method and fair-comparison explanation."
    ),
    callout(
        "Transition to results: 'Now that the classifier and inputs are controlled, the difference in recognition tells us what our added recovery stage changes.'"
    ),
)

page(
    "Your speaking script: slides 5-8",
    "REHEARSAL / SHOW THE GAIN AND ITS BOUNDARY",
    qa(
        "Slide 5 - Accuracy across conditions | about 55 seconds",
        "Our clearest improvement is under short gaps. With short joint loss, accuracy increases from 11.50% to 15.25%; with short limb loss, from 15.25% to 18.50%. These are all-input scores, so no difficult examples were removed. Original poses and long gaps show only small gains. Our extension adds processing cost, and absolute accuracy remains limited on this 400-class task.",
    ),
    qa(
        "Slide 6 - Paired gains and uncertainty | about 50 seconds",
        "For short joint gaps, 21 wrong predictions become correct, while six correct predictions become wrong. That is 15 additional correct clips out of 400, or 3.75 percentage points. The paired interval is +1.50 to +6.25 points. The short-limb interval is also above zero. The intervals for the other conditions include zero, so we do not claim consistent improvement there. These are pilot intervals, not proof of universal superiority.",
    ),
    qa(
        "Slide 7 - Rejection and coverage | about 45 seconds",
        "Rejection changes which clips get an answer. On short joint gaps, recovery plus rejection has 19.11% accuracy among accepted clips, but answers only 78.50% of clips. Its all-input accuracy is 15.00%, compared with 15.25% for recovery alone. We report both accuracy and coverage so rejection cannot hide the difficult examples. The threshold is selected on calibration data.",
    ),
    qa(
        "Slide 8 - Limits, demo and conclusion | about 40 seconds before demo",
        "This is a small controlled robustness study using a reused pretrained model. It does not reproduce the full paper or validate physical camera occlusion. Person slots can switch, long gaps remain hard and some predictions regress. I will show both an improved and a failed example in the dashboard. The separate webcam lab uses a different model and is not evidence for these scores.",
    ),
    h("Close after the 90-second demonstration"),
    callout(
        "Our pilot supports a specific conclusion: a bounded confidence-gated hold helps this fixed model under short pose gaps. The benefit is measurable but limited, costs processing time, and needs broader real-camera validation. The code, per-clip predictions and full results are available for inspection."
    ),
    p(
        "These timings give approximately 7 minutes 50 seconds including the demo. Use the actual time assigned by your committee. Practise once slowly and once with interruptions; avoid reading the tables aloud row by row."
    ),
)

page(
    "Run a clean, honest demonstration",
    "DEMO / A 90-SECOND PATH WITH A BACKUP",
    table(
        [
            ["Time", "Action", "What to explain"],
            [
                "0-15 s",
                "Open the research dashboard at http://127.0.0.1:8765/",
                "This displays our saved experiment. It is not training the model live.",
            ],
            [
                "15-35 s",
                "Select the short joint-gap condition; point to baseline and recovery cards.",
                "11.50% to 15.25%, +3.75 percentage points on identical clips.",
            ],
            [
                "35-55 s",
                "Go to Pose viewer, choose Improved and press Play; use the frame slider.",
                "Orange joints are filled estimates; captions are predictions for the whole clip, not per-frame predictions.",
            ],
            [
                "55-70 s",
                "Choose Regressed or Both incorrect.",
                "Repair can help the input and still fail to recognise the action, or make it worse.",
            ],
            [
                "70-90 s",
                "Show the coverage panel, complete table and Protocol; point to Download results.",
                "All conditions, withheld predictions and experiment provenance are available.",
            ],
        ],
        [60, 192, 255],
    ),
    p(
        "The viewer shows the first person slot; the classifier uses both. A selected example explains behaviour, but average improvement comes from the full paired evaluation. Do not infer accuracy from one attractive animation."
    ),
    h("Before walking into the room"),
    p(
        "Open the current research PPTX, this guide, the five-page measured report and the dashboard. Test the exact laptop, browser, charger and projector. Verify team IDs and mentor spelling. Check that all five conditions and example controls load. Keep the report and slides locally for an offline fallback. The dashboard can show saved research results without downloading the model or data again."
    ),
    h("Start the existing project on this Mac"),
    ("code", "cd /Users/ashishsingh/Desktop/miniproject\n.venv/bin/python run.py"),
    p(
        "If port 8765 is already in use by this project, use its open dashboard. If another service occupies it, use <b>.venv/bin/python run.py --port 8766</b> and change the browser URL. For a fresh machine, follow the installation section of README.md before presentation day. Do not rerun the benchmark during the talk: explicit reruns overwrite saved experiment outputs."
    ),
    callout(
        "If the browser fails: 'The dashboard is a viewer for the same saved measurements. I can show the PDF table and the per-clip evidence instead.' Continue with the results, without claiming a live computation happened."
    ),
)

page(
    "Know where the evidence lives",
    "CODE TOUR / SIX PLACES YOU SHOULD BE ABLE TO OPEN",
    table(
        [
            ["File or function", "What to explain when it is opened"],
            [
                "research/recovery.py\nCausalPoseRecovery.step",
                "Find the observed mask, state update, five-frame condition, last-score gate, decayed confidence and expired-state reset. Explain why beta = 0 holds the position.",
            ],
            [
                "research/corruptions.py\napply_corruption",
                "How repeatable missing-joint blocks, limb masks and noise are added; the original input remains available for fair comparison.",
            ],
            [
                "research/benchmark.py\ncalibrate / evaluate",
                "Calibration selects settings; evaluation checks split separation and frozen hashes. Four variants use the same corrupted samples.",
            ],
            [
                "research/benchmark.py\nmetric / paired_ci",
                "All-input versus accepted accuracy, coverage, represented-class macro-F1, corrected/regressed counts and paired bootstrap.",
            ],
            [
                "research/upstream.py\nload_model / verify_equivalence",
                "Loads the pinned upstream definitions and checkpoint; compares clip/continual/step outputs to check the inference bridge.",
            ],
            [
                "research/data.py\nload_partition",
                "Uses the upstream feeder and checks raw-data hashes and labels, preserving the expected layout.",
            ],
        ],
        [194, 313],
    ),
    h("The saved audit trail"),
    p(
        "<b>results/research/comparison.json:</b> every measured row, protocol and example.<br/><b>results/research/predictions.csv:</b> per-clip predictions for checking failures and paired changes.<br/><b>results/research/calibration.json and design.json:</b> chosen settings, development history and frozen implementation/checkpoint hashes.<br/><b>results/research/data-manifest.json:</b> selected clips and provenance.<br/><b>results/research/equivalence.json:</b> adapter check; clip-versus-step maximum difference about 0.00000477, below the 0.0005 tolerance."
    ),
    p(
        "Numerical agreement on the checked input supports the adapter, but does not prove correctness for every possible input or reproduce the paper's full experiment. Fixed seeds and hashes improve traceability; timings and small floating-point differences can still vary by hardware."
    ),
    callout(
        "Practise tracing one test clip: its input and mask -> repaired sequence -> two predictions -> whether it is corrected or regressed -> its contribution to the aggregate score. This is a stronger defence than memorising a directory tree."
    ),
)

page(
    "Viva: fundamentals and implementation",
    "QUESTIONS 1-9 / ANSWER IN TWO OR THREE SENTENCES",
    qa(
        "1. Why use pose instead of RGB video?",
        "Pose makes body geometry and movement explicit and reduces dependence on background appearance. It also discards objects and context, which can be essential for some Kinetics actions. This is a trade-off, not proof that pose is always better.",
    ),
    qa(
        "2. Why a graph model?",
        "A skeleton naturally has connected joints. A graph model learns from those relationships; temporal layers add motion information. We reuse a paper-backed pretrained model rather than compare every model family.",
    ),
    qa(
        "3. What does the star in CoST-GCN* refer to?",
        "The authors use the star for the modified architecture with temporal stride one and no padding. Our pretrained single joint-stream checkpoint matches that variant. [2]",
    ),
    qa(
        "4. Is this an LSTM or a Kalman filter?",
        "The classifier is a graph convolutional model. The general recovery implementation has an alpha-beta tracker, but the selected setting reduces to a confidence-gated hold. It is not a full Kalman filter: we do not maintain uncertainty covariance or compute Kalman gains.",
    ),
    qa(
        "5. Why not interpolate between the last and next visible point?",
        "That uses a future observation and introduces look-ahead. Our repair is causal. Offline interpolation is a valid comparison to investigate, but it is not the method used for these results.",
    ),
    qa(
        "6. Why five frames and confidence 0.5?",
        "They are part of the configuration selected from seven candidates using the calibration set. They are not universal constants. The limited candidate search and reuse of one calibration set are constraints of this pilot.",
    ),
    qa(
        "7. What if a joint was never seen?",
        "There is no trusted history, so the module cannot fill it. Likewise, a last real confidence below 0.5 cannot seed repair, and a gap beyond five frames remains invalid.",
    ),
    qa(
        "8. Does the person reset solve identity tracking?",
        "No. It checks adjacent visible torso anchors for a large median jump and clears the slot's state. It can miss swaps or confuse fast movement with a swap.",
    ),
    qa(
        "9. Are you improving pose accuracy?",
        "We evaluated action classification, not joint-coordinate error against ground truth. A visually plausible repaired joint is still an estimate; pose reconstruction quality needs its own labelled evaluation.",
    ),
)

page(
    "Viva: experimental design and metrics",
    "QUESTIONS 10-18 / EXPECT THESE FROM A CAREFUL EXAMINER",
    qa(
        "10. Where is your training set?",
        "The classifier was already trained by the paper's authors. We perform no classifier training. Our 200 calibration clips choose the extension settings; 400 separate clips evaluate the extension.",
    ),
    qa(
        "11. Is your test set completely independent?",
        "It is disjoint from our calibration clips, but both subsets come from the upstream official validation partition. That partition may have influenced the authors' checkpoint selection. This is not a new subject-independent deployment test.",
    ),
    qa(
        "12. Why only 400 evaluation clips?",
        "It makes a reproducible pilot feasible on our resources while keeping the full 400-output classifier. It limits per-class support and precision; a larger evaluation is needed before broader claims.",
    ),
    qa(
        "13. Did you choose only cases that improve?",
        "No. The clip split is hash-based, all five conditions are reported, empty clips are kept, and settings were frozen before evaluation. The demo's selected examples are illustrative and are not the statistical evidence.",
    ),
    qa(
        "14. How do we know recovery causes the difference?",
        "The paired comparison changes only the recovery stage while holding model, inputs and masks fixed. It supports an effect under this experiment's conditions. It does not establish the same effect in all datasets or cameras.",
    ),
    qa(
        "15. Can you compare 15.25% with the paper's best score?",
        "Not directly. Dataset size, single versus multiple streams, corruption and evaluation protocol differ. Compare our original and recovered model on the same 400 clips; do not present that as beating the published benchmark.",
    ),
    qa(
        "16. Why did rejection raise accuracy?",
        "It raised accuracy among accepted predictions by withholding some difficult cases. Coverage fell; in the short-joint example all-input accuracy actually fell from 15.25% to 15.00% after rejection. Both denominators must be shown.",
    ),
    qa(
        "17. Does a 95% interval above zero prove superiority?",
        "It is evidence of a positive paired difference under this sampled protocol. It is not a guarantee on every future dataset. The intervals are per condition, unadjusted for multiple comparisons, and use a fixed corruption protocol.",
    ),
    qa(
        "18. Why is F1 not the same as accuracy?",
        "Accuracy counts correct clips. Macro-F1 weighs represented classes equally and balances their precision and recall. With uneven, small class counts the two can move differently, as the long-gap result shows.",
    ),
)

page(
    "Viva: contribution, failures and next steps",
    "QUESTIONS 19-27 / DEFEND THE SCOPE WITHOUT DODGING",
    qa(
        "19. Isn't holding the last point too simple?",
        "The primitive is simple and established. The project contribution is its bounded confidence-gated implementation and a controlled study of when it helps a fixed model, including ablations, paired uncertainty and failures. We do not claim a new family of tracking algorithms.",
    ),
    qa(
        "20. What did your team build rather than copy?",
        "The project adds the recovery/rejection experiment, deterministic data pipeline, corruption controls, numerical adapter checks, measured comparison and dashboard/reporting. Attribute the pretrained model, architecture and base preprocessing to the authors. Each member should describe only work they actually did and understand.",
    ),
    qa(
        "21. Did AI help you?",
        "Describe the tools actually used and follow your institute's disclosure requirements. Do not claim manual authorship of code you cannot explain. The defensible part is understanding the implementation, verifying the experiment and being clear about reused work.",
    ),
    qa(
        "22. Why does it fail on long gaps?",
        "After five missing frames we stop filling; a held position also becomes less plausible as motion continues. The long-gap condition has only a one-clip net accuracy gain and lower macro-F1, so it offers little evidence of benefit.",
    ),
    qa(
        "23. Why did noise accuracy rise if you do not denoise?",
        "Observed noisy coordinates are not smoothed by this method. Recovery can still act on naturally missing entries in those clips and alter downstream predictions. This run does not isolate the reason; the noise interval touches zero, so we do not claim a consistent noise benefit.",
    ),
    qa(
        "24. Why can 15-frame dropout score above 5-frame dropout?",
        "The separately seeded masks place missing entries differently, even with the same target fraction. Gap length alone is not isolated by comparing these two aggregate scores. Multiple matched severity trials would be needed to establish a monotonic effect.",
    ),
    qa(
        "25. Is a five-frame hold 0.17 seconds?",
        "Only if the actual sampling rate is 30 FPS. Our rule is specified in frame steps; do not assume a universal duration or convert batch clip time into streaming FPS.",
    ),
    qa(
        "26. Could this detect falls or support medical decisions?",
        "Those applications require task-specific data, validation and reliability requirements. This project demonstrates an action-recognition robustness experiment, not a validated safety-critical system.",
    ),
    qa(
        "27. What would you improve first?",
        "Run a larger independent study with multiple corruption seeds and real physically occluded compatible pose recordings, then add stronger imputation baselines and identity handling. Keep a new held-out evaluation set separate from development.",
    ),
)

page(
    "A practical rehearsal plan",
    "PREPARATION / EVERY MEMBER SHOULD UNDERSTAND THE WHOLE STORY",
    table(
        [
            ["Study block", "Task", "You are ready when..."],
            [
                "45 minutes",
                "Read pages 1-6 and the paper sections on page 3.",
                "You can draw the graph, pipeline and five-frame hold without notes.",
            ],
            [
                "30 minutes",
                "Read pages 7-9; calculate the short-gap and rejection examples yourself.",
                "You can distinguish percentage points, accepted accuracy and coverage.",
            ],
            [
                "30 minutes",
                "Open the six code locations on page 13 and follow one clip.",
                "You can show the actual conditions in code and connect them to the result.",
            ],
            [
                "30 minutes",
                "Rehearse pages 10-12 with the real slides and dashboard.",
                "You finish within your time allowance and can continue from the PDF if needed.",
            ],
            [
                "30 minutes",
                "Ask one another the viva questions without the answer visible.",
                "Each answer includes the claim, the evidence and any relevant limit.",
            ],
        ],
        [83, 218, 206],
    ),
    h("Suggested speaking split, not a claim about past contributions"),
    p(
        "<b>23bcs019:</b> slides 1-2, motivation and base paper.<br/><b>23bcs007:</b> slides 3-4, method and fair experiment.<br/><b>23bcs021:</b> slides 5-8 and demo, results and limitations.<br/>Swap these roles if another arrangement fits you better. Everyone should know the pitch, selected settings, main result and prototype/research distinction."
    ),
    h("Three whiteboard exercises"),
    p(
        "<b>A.</b> Draw shoulder -> elbow -> wrist across three time steps. Explain spatial and temporal information.<br/><b>B.</b> Start at confidence 0.8, remove the next six frames, and write the filled scores and stopping point. Repeat with confidence 0.2.<br/><b>C.</b> Use 46 original correct clips, 21 corrections and six regressions. Derive 61 corrected-model successes, 15.25% accuracy and +3.75 percentage points."
    ),
    h("How to handle a question you cannot answer"),
    callout(
        "State what the experiment does establish, name what is missing, then propose the check. Example: 'We did not measure that on live camera input. This result is from stored OpenPose clips. To answer it, I would run the compatible pose pipeline on new recordings and measure end-to-end latency and accuracy.'"
    ),
    p(
        "Before submission, fill the remaining actual names/department and institute template details. The supplied guidelines require the report soft copy at least three working days before each presentation; obtain the actual schedule from Table I rather than guessing a deadline."
    ),
)

page(
    "Last-minute revision and reading links",
    "KEEP THIS PAGE OPEN BEFORE YOU PRESENT",
    table(
        [
            ["Remember", "Exact fact"],
            [
                "Base model",
                "Authors' pretrained single joint-stream CoST-GCN*; 400 action outputs.",
            ],
            [
                "Data",
                "18 OpenPose joints; x, y, confidence; 300 frame slots; two person slots.",
            ],
            [
                "Split",
                "200 calibration + 400 evaluation; 251 true classes; two empty clips retained.",
            ],
            [
                "Selected recovery",
                "Last real confidence >= 0.5; alpha 1; beta 0; hold <= 5 frames; confidence decay 0.9.",
            ],
            [
                "Short-joint result",
                "11.50% -> 15.25%; +3.75 pp; 21 corrected / 6 regressed; interval +1.50 to +6.25 pp.",
            ],
            [
                "Short-limb result",
                "15.25% -> 18.50%; +3.25 pp; interval +0.75 to +5.75 pp.",
            ],
            [
                "Rejection",
                "Threshold 0.08. Joint-gap combined: 19.11% accepted accuracy at 78.50% coverage; 15.00% overall.",
            ],
            [
                "Boundary",
                "Controlled pose damage, not validated camera occlusion; clip timing, not live FPS; old webcam lab is separate.",
            ],
        ],
        [132, 375],
    ),
    h("Read in this order"),
    p(
        '<b>[1] Paper:</b> <link href="https://arxiv.org/html/2203.11009v2" color="#6D28D9">arxiv.org/html/2203.11009v2</link><br/><b>Published identity:</b> <link href="https://doi.org/10.1016/j.patcog.2023.109528" color="#6D28D9">doi.org/10.1016/j.patcog.2023.109528</link><br/><b>[2] Authors\' implementation:</b> <link href="https://github.com/LukasHedegaard/continual-skeletons" color="#6D28D9">github.com/LukasHedegaard/continual-skeletons</link>'
    ),
    p(
        '<b>[3] Q1 evidence:</b> <link href="https://www.scimagojr.com/journalvalue.php?area=1700&amp;category=1707&amp;ord=desc&amp;order=item" color="#6D28D9">SCImago computer-vision journal listing</link>. <link href="https://kniznica.umb.sk/app/cmsSiteBoxAttachment.php?ID=6322&amp;cmsDataID=0" color="#6D28D9">University-library copy of the SJR 2024 table</link> includes Pattern Recognition (ISSN 00313203), Q1. The live SCImago page was unavailable to the research tool during this guide refresh; retain the year-specific evidence. This is not verification of publication-year JCR.'
    ),
    p(
        '<b>Our repository:</b> <link href="https://github.com/4uashish2628/miniproject-2" color="#6D28D9">github.com/4uashish2628/miniproject-2</link><br/><b>Local reading:</b> docs/RESEARCH_VIVA.md, docs/RESEARCH_RESULTS.md and the code/evidence files on page 13. Use the current research presentation, not the older presentation draft. The Colab reproduction notebook is notebooks/Research_Comparison.ipynb.'
    ),
    p(
        f'<b>Evidence integrity:</b> comparison.json SHA256<br/><font size="7">{DIGEST}</font><br/>This guide explains the saved experiment and does not rerun it or change its results. Links checked or located on 12 October 2026.'
    ),
)


def build(output):
    import reportlab
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph, Table, TableStyle

    fonts = Path(reportlab.__file__).parent / "fonts"
    for name, file in [
        ("Guide", "Vera.ttf"),
        ("Guide-Bold", "VeraBd.ttf"),
        ("Guide-Italic", "VeraIt.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily(
        "Guide",
        normal="Guide",
        bold="Guide-Bold",
        italic="Guide-Italic",
        boldItalic="Guide-Bold",
    )
    ink, muted, purple = map(colors.HexColor, ["#172033", "#556276", "#6D28D9"])
    pale, border = map(colors.HexColor, ["#F4F0FB", "#DFE4EC"])
    width, height = A4
    margin = 44
    available = width - margin * 2
    styles = {
        "body": ParagraphStyle(
            "body", fontName="Guide", fontSize=10, leading=14.2, textColor=ink
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName="Guide-Bold",
            fontSize=11.5,
            leading=15,
            textColor=purple,
        ),
        "title": ParagraphStyle(
            "title", fontName="Guide-Bold", fontSize=23, leading=28, textColor=ink
        ),
        "cell": ParagraphStyle(
            "cell", fontName="Guide", fontSize=9.2, leading=12.7, textColor=ink
        ),
        "th": ParagraphStyle(
            "th", fontName="Guide-Bold", fontSize=8.8, leading=12, textColor=ink
        ),
        "code": ParagraphStyle(
            "code", fontName="Courier", fontSize=8.7, leading=12, textColor=ink
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
    c.setTitle("PoseLab - Presentation and Viva Study Guide")
    c.setAuthor("23bcs019, 23bcs007, 23bcs021 | Mentor: SUNIL CK")
    stats = []
    for number, (title, subtitle, blocks) in enumerate(PAGES, 1):
        c.setFillColor(purple)
        c.rect(0, height - 7, width, 7, fill=1, stroke=0)
        c.setFont("Guide-Bold", 7.5)
        c.setFillColor(muted)
        c.drawString(margin, height - 31, "POSELAB / PRESENTATION AND VIVA GUIDE")
        y = height - 54

        def draw_text(text, style="body", gap=9, inset=0, background=False):
            nonlocal y
            block = Paragraph(text, styles[style])
            _, bh = block.wrap(available - 2 * inset, height)
            total = bh + (16 if background else 0)
            if y - total < 57:
                raise ValueError(
                    f"Page {number} overflow ({y-total:.1f}) near {text[:70]}"
                )
            if background:
                c.setFillColor(pale)
                c.roundRect(margin, y - total, available, total, 5, fill=1, stroke=0)
                block.drawOn(c, margin + inset, y - 8 - bh)
            else:
                block.drawOn(c, margin + inset, y - bh)
            y -= total + gap

        draw_text(escape(title), "title", 9)
        draw_text(f'<font size="7.4" color="#556276">{escape(subtitle)}</font>', gap=14)
        c.bookmarkPage(f"page-{number}")
        c.addOutlineEntry(title, f"page-{number}", level=0)
        for block in blocks:
            if block[0] == "heading":
                draw_text(escape(block[1]), "heading", 5)
            elif block[0] == "paragraph":
                draw_text(block[1])
            elif block[0] == "callout":
                draw_text(block[1], inset=11, background=True, gap=11)
            elif block[0] == "qa":
                draw_text(f"<b>{escape(block[1])}</b><br/>{escape(block[2])}", gap=11)
            elif block[0] == "code":
                draw_text(
                    escape(block[1]).replace("\n", "<br/>"),
                    "code",
                    inset=11,
                    background=True,
                )
            elif block[0] == "diagram":
                c.setFont("Guide", 7.5)
                points = []
                for frame in range(3):
                    x = margin + 128 + frame * 125
                    nodes = [
                        (x, y - 11),
                        (x + 10, y - 33),
                        (x + 27, y - 55 + frame * 9),
                    ]
                    points.append(nodes)
                    c.setStrokeColor(purple)
                    c.setLineWidth(1.8)
                    for first, second in zip(nodes, nodes[1:]):
                        c.line(*first, *second)
                    c.setFillColor(purple)
                    for px, py in nodes:
                        c.circle(px, py, 3, fill=1, stroke=0)
                    c.setFillColor(muted)
                    c.drawCentredString(x + 12, y - 77, f"Frame {frame + 1}")
                c.setLineWidth(0.6)
                c.setStrokeColor(muted)
                c.setDash(3, 3)
                for before, after in zip(points, points[1:]):
                    for a, b in zip(before, after):
                        c.line(a[0] + 5, a[1], b[0] - 5, b[1])
                c.setDash()
                for label, (_, py) in zip(["Shoulder", "Elbow", "Wrist"], points[0]):
                    c.drawString(margin, py - 3, label)
                y -= 91
                draw_text(
                    '<font size="8">Simplified arm graph: purple links connect joints; dashed links connect corresponding joints across time.</font>',
                    gap=10,
                )
            elif block[0] == "table":
                cells = [
                    [
                        Paragraph(
                            escape(str(v)).replace("\n", "<br/>"),
                            styles["th" if ri == 0 else "cell"],
                        )
                        for v in row
                    ]
                    for ri, row in enumerate(block[1])
                ]
                t = Table(cells, colWidths=block[2])
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), pale),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 7),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("LINEBELOW", (0, 0), (-1, -1), 0.35, border),
                        ]
                    )
                )
                _, th = t.wrap(available, height)
                if y - th < 57:
                    raise ValueError(f"Table overflow on page {number}")
                t.drawOn(c, margin, y - th)
                y -= th + 12
        c.setStrokeColor(border)
        c.line(margin, 39, width - margin, 39)
        c.setFillColor(muted)
        c.setFont("Guide", 7.5)
        c.drawString(
            margin, 25, "IIIT Dharwad | Measured research pilot | 12 October 2026"
        )
        c.drawRightString(width - margin, 25, f"{number} / {len(PAGES)}")
        stats.append({"page": number, "title": title, "bottom_y": round(y, 1)})
        c.showPage()
    c.save()
    print(json.dumps({"output": str(output), "pages": stats}, indent=2))


if __name__ == "__main__":
    build(ROOT / "output/pdf/PoseLab_Presentation_Study_Guide.pdf")
