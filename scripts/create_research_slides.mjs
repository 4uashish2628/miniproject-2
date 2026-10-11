/** Generate the research deck from the frozen, measured comparison JSON.
 * Requires the Codex presentation runtime. Environment variables may override
 * POSELAB_RUNTIME and POSELAB_PRESENTATION_SKILL on another machine.
 * Usage: node scripts/create_research_slides.mjs [comparison.json] [output.pptx]
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RUNTIME = process.env.POSELAB_RUNTIME ?? path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies');
process.env.RUNTIME_NODE_MODULES ??= path.join(RUNTIME, 'node/node_modules');
const SKILL = process.env.POSELAB_PRESENTATION_SKILL ?? path.join(os.homedir(), '.codex/plugins/cache/openai-primary-runtime/presentations/26.1009.11318/skills/presentations');
const inputPath = path.resolve(process.argv[2] ?? path.join(ROOT, 'results/research/comparison.json'));
const finalPath = path.resolve(process.argv[3] ?? path.join(ROOT, 'deliverables/PoseLab_Research_Presentation.pptx'));
const BUILD = path.join(ROOT, '.build/research-slides');
const revision = new Date().toISOString().replace(/[:.]/g, '-');
const result = JSON.parse(await fs.readFile(inputPath, 'utf8'));
if (result.status !== 'complete') throw new Error('A complete measured comparison is required. No placeholder metrics are supported.');
const ids = ['clean', 'burst_missing', 'limb_missing', 'long_missing', 'jitter'];
const conditions = ids.map(id => result.conditions.find(c => c.id === id));
if (conditions.some(c => !c)) throw new Error('All five planned conditions are required.');
const variant = (condition, id) => {
  const row = condition.results.find(r => r.variant === id);
  if (!row || !Number.isFinite(row.accuracy) || !Number.isFinite(row.coverage)) throw new Error(`Missing measured ${id} result in ${condition.id}`);
  return row;
};
for (const c of conditions) for (const id of ['baseline', 'recovery', 'rejection', 'combined']) variant(c, id);
if (result.calibration.selected_candidate !== 'confident_hold_5') throw new Error('Update the method explanation for the selected calibration recipe.');
const protocol = result.protocol;
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(path.join(RUNTIME, 'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
const family = resolvePresentationFont({ fontFamily: 'Arial' });
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = { ink: '#182D35', muted: '#65767B', paper: '#F5F7F4', green: '#255C4D', lime: '#D6E998', nav: '#142E31', baseline: '#829DA9', recovery: '#2D6A56', line: '#D6DED9', loss: '#A53730' };
const pct = n => Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : 'No accepted clips';
const pp = n => `${n > 0 ? '+' : ''}${(n * 100).toFixed(2)}`;
const date = new Date(result.generated_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'Asia/Kolkata' });
const labels = ['Original poses', 'Joint loss, 5 frames', 'Limb loss, 5 frames', 'Joint loss, 15 frames', 'Coordinate noise'];
const citations = {
  paper: 'Hedegaard, Heidari and Iosifidis. Continual spatio-temporal graph convolutional networks. Pattern Recognition 140 (2023), 109528. https://doi.org/10.1016/j.patcog.2023.109528',
  repo: 'Official source: https://github.com/LukasHedegaard/continual-skeletons',
  q1: 'SCImago Journal & Country Rank, 2024 data, Computer Vision and Pattern Recognition category. Pattern Recognition, ISSN 0031-3203, SJR 2.058, Q1. Primary indexed ranking: https://www.scimagojr.com/journalvalue.php?area=1700&category=1707&ord=desc&order=item . The live page restricted direct access during verification. This is a 2024 SCImago/SJR claim, not a claim about the 2023 JCR edition.',
  data: 'Kinetics Skeleton data distributed with ST-GCN, official validation partition. Source: https://github.com/yysijie/st-gcn and ' + protocol.source_url,
};
function text(slide, content, x, y, w, h, size = 28, color = C.ink, bold = false) {
  const shape = slide.shapes.add({ geometry: 'textbox', position: { left: x, top: y, width: w, height: h }, fill: 'none', line: { fill: 'none', width: 0 } });
  shape.text = content;
  shape.text.style = { typeface: family, fontSize: size, color, bold, autoFit: 'none' };
  shape.text.insets = 0;
  return shape;
}
function slide(title, notes, dark = false) {
  const s = deck.slides.add();
  s.background.fill = dark ? C.nav : C.paper;
  if (title) text(s, title, 68, 48, 1144, 65, 44, dark ? '#FFFFFF' : C.ink, true);
  text(s, 'PoseLab  /  IIIT Dharwad', 70, 668, 800, 28, 17, dark ? '#BACCC5' : C.muted);
  text(s, String(deck.slides.items.length), 1170, 664, 44, 32, 21, dark ? C.lime : C.muted);
  s.speakerNotes.textFrame.setText(notes);
  return s;
}
function table(s, values, widths, top = 180, height = 350, size = 25) {
  const t = s.tables.add({ rows: values.length, columns: widths.length, left: 70, top, width: 1140, height, columnWidths: widths, values });
  t.cells.block({ row: 0, column: 0, rowCount: values.length, columnCount: widths.length }).assign({ textStyle: { typeface: family, fontSize: size, color: C.ink }, margins: { left: 15, right: 12, top: 12, bottom: 12 } });
  t.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: widths.length }).assign({ fill: C.green, textStyle: { typeface: family, fontSize: size, color: '#FFFFFF', bold: true } });
  for (let i = 1; i < values.length; i++) t.cells.block({ row: i, column: 0, rowCount: 1, columnCount: widths.length }).assign({ fill: i % 2 ? '#FFFFFF' : '#EBF0EC' });
  t.borders.assign({ fill: C.line, width: 1, style: 'solid' });
  return t;
}
function node(s, label, x, y, w, h, emphasis = false) {
  const a = s.shapes.add({ geometry: 'rect', position: { left: x, top: y, width: w, height: h }, fill: emphasis ? C.green : '#FFFFFF', line: { fill: C.green, width: 2 } });
  a.text = label;
  a.text.style = { typeface: family, fontSize: 27, color: emphasis ? '#FFFFFF' : C.ink, alignment: 'center', verticalAlignment: 'middle', autoFit: 'none' };
  a.text.insets = 14;
  return a;
}
function connect(s, a, b) { s.shapes.connect(a, b, { kind: 'straight', fromSide: 'right', toSide: 'left', line: { fill: C.green, width: 2.5, style: 'solid' }, tail: { type: 'triangle', width: 'sm', length: 'sm' } }); }

let s = slide('', `Mini Project II, seventh semester, August to November 2026. Team roll numbers: 23bcs019, 23bcs007, 23bcs021. Mentor: SUNIL CK. We evaluate an extension around the authors' pretrained graph model. Measured results generated ${result.generated_at}.\n${citations.paper}\n${citations.repo}`, true);
text(s, 'IIIT DHARWAD', 72, 65, 1000, 40, 24, C.lime, true);
text(s, 'PoseLab', 72, 167, 1120, 95, 76, '#FFFFFF', true);
text(s, 'Human activity recognition\nwith missing-joint recovery', 72, 291, 1120, 148, 48, '#FFFFFF');
text(s, 'A study using an existing Q1 paper and its public model', 72, 451, 1110, 40, 28, '#D9E4DE');
text(s, 'Team: 23bcs019, 23bcs007, 23bcs021', 72, 506, 1110, 36, 26, '#FFFFFF');
text(s, 'Mentor: SUNIL CK', 72, 548, 1110, 36, 26, '#FFFFFF');
text(s, `Mini Project II  /  Aug–Nov 2026\nMeasured experiment: ${date}`, 72, 602, 1080, 58, 23, C.lime);

s = slide('Research foundation and project question', `${citations.paper}\n${citations.repo}\n${citations.q1}\nThe paper targets efficient continual action recognition. Our research question concerns tolerance to missing or noisy pose joints. We add and evaluate a small causal recovery component without retraining the classifier. We do not claim that joint holding is a new algorithm or that this sample reproduces the paper's full benchmark. The ranking refers to the journal and database year, not an arXiv page.`);
text(s, 'Continual spatio-temporal graph\nconvolutional networks', 72, 160, 1120, 103, 37, C.green, true);
text(s, 'Hedegaard, Heidari and Iosifidis\nPattern Recognition 140 (2023), 109528', 72, 282, 1110, 86, 28);
text(s, 'Q1 journal in SCImago/SJR 2024', 72, 391, 1110, 48, 30, C.green, true);
text(s, 'Our question', 72, 480, 1110, 44, 29, C.ink, true);
text(s, 'Can a short memory of reliable joints improve activity\nrecognition when pose estimates temporarily disappear?', 72, 535, 1110, 95, 31, C.ink);

s = slide('Confidence-gated short-gap hold', `The deployed calibration winner is ${result.calibration.selected_candidate}. Hold the last observed joint position for at most five missing frames when the last real pose confidence was at least 0.5. Multiply its confidence by 0.9 for each held frame. Reset the person-slot memory if the torso anchor jumps by more than 0.15 in normalized coordinates. The method uses no future frames. Alpha 1 and beta 0 mean there is no velocity extrapolation in the chosen method. Velocity-based candidates were considered on calibration only. The original and recovered branches use the same frozen pretrained checkpoint. A separate optional threshold can reject low-confidence class predictions.\n${citations.repo}`);
const input = node(s, 'Pose sequence\nwith missing joints', 70, 199, 293, 126);
const original = node(s, 'Original\nclassifier', 522, 146, 300, 105);
const recovery = node(s, 'Hold reliable joints\nfor up to 5 frames', 474, 335, 351, 113, true);
const classifier = node(s, 'Same\nclassifier', 931, 335, 278, 113);
connect(s, input, original); connect(s, input, recovery); connect(s, recovery, classifier);
text(s, 'Unchanged pretrained weights', 887, 165, 322, 80, 25, C.muted);
text(s, 'Last real confidence ≥ 0.5', 70, 502, 555, 43, 29, C.green, true);
text(s, 'Confidence decays × 0.9 per frame', 70, 553, 555, 75, 27);
text(s, 'Torso movement resets memory', 700, 502, 510, 43, 29, C.green, true);
text(s, 'No future frames or retraining', 700, 553, 510, 75, 27);

s = slide('A fixed comparison on real pose sequences', `${citations.data}\n${citations.repo}\nSelection: ${protocol.selection}\nSubset limitation: ${protocol.subset_limitation}\nCheckpoint: ${protocol.checkpoint}. SHA256 ${protocol.checkpoint_sha256}. Source commit ${protocol.upstream_commit}. Seed ${protocol.seed}.\nSeven recovery recipes were compared on three calibration conditions over three development rounds, without using held-out test predictions. Calibration also selected the rejection thresholds. The test uses identical clips and deterministic corruptions for all variants. The test is a sample from the official validation partition, which is not an independent new-person dataset. Empty clips remain in the sample. The classifier retains all 400 possible labels.`);
text(s, `${protocol.calibration_count}`, 72, 160, 470, 103, 76, C.green, true);
text(s, 'Calibration clips', 72, 278, 495, 47, 31, C.ink, true);
text(s, 'Choose recovery settings\nand rejection thresholds', 72, 342, 495, 96, 29, C.muted);
text(s, `${protocol.test_count}`, 705, 160, 470, 103, 76, C.green, true);
text(s, 'Disjoint test clips', 705, 278, 495, 47, 31, C.ink, true);
text(s, 'Same clips and corruptions\nfor every variant', 705, 342, 495, 96, 29, C.muted);
text(s, `Kinetics Skeleton validation sample  /  ${protocol.classes_present} true classes`, 72, 483, 1130, 50, 30);
text(s, 'Five conditions: original poses, short joint loss, limb loss,\nlonger joint loss and coordinate noise', 72, 554, 1110, 85, 28, C.muted);

s = slide('Accuracy across five test conditions', `Measured results from ${inputPath}, generated ${result.generated_at}. Accuracy is the number of correct predictions divided by all ${protocol.test_count} clips. Neither plotted variant abstains. Every point uses the same clip and corruption in both branches. Joint and limb masks target 25% of eligible visible joints in blocks of 5 frames. The longer-gap condition uses 15-frame blocks. Coordinate noise has normalized standard deviation 0.02. Actual masked fractions are available in the JSON and dashboard. The clean condition uses the original detector's poses, which can already contain missing joints.\n${conditions.map((c, i) => `${labels[i]}: original ${pct(variant(c, 'baseline').accuracy)}, recovery ${pct(variant(c, 'recovery').accuracy)}. Macro F1: ${pct(variant(c, 'baseline').macro_f1)} versus ${pct(variant(c, 'recovery').macro_f1)}.`).join('\n')}`);
text(s, `${protocol.test_count} test clips in each condition. All predictions count.`, 72, 131, 1135, 40, 26, C.muted);
const chart = s.charts.add('bar', {
  position: { left: 66, top: 190, width: 1148, height: 400 },
  categories: ['Original\nposes', 'Joint loss\n5 frames', 'Limb loss\n5 frames', 'Joint loss\n15 frames', 'Coordinate\nnoise'],
  series: [
    { name: 'Original model', values: conditions.map(c => variant(c, 'baseline').accuracy), fill: C.baseline },
    { name: 'With recovery', values: conditions.map(c => variant(c, 'recovery').accuracy), fill: C.recovery },
  ],
  hasLegend: true, legend: { position: 'bottom', textStyle: { fontSize: 24, fill: C.ink } },
  barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 25 },
  yAxis: { min: 0, max: 1, majorUnit: 0.2, numberFormatCode: '0%', textStyle: { fontSize: 22, fill: C.muted }, majorGridlines: { fill: C.line, width: 1, style: 'solid' } },
  xAxis: { textStyle: { fontSize: 23, fill: C.ink } },
  dataLabels: { showValue: true, position: 'outEnd', numberFormatCode: '0.00%', textStyle: { fontSize: 22, fill: C.ink } },
  chartFill: C.paper, plotAreaFill: C.paper,
});
applyPresentationChartFont(chart, { fontFamily: family });
text(s, 'Loss conditions target 25% of visible joints. Noise uses σ = 0.02.', 72, 614, 1135, 37, 24, C.muted);

s = slide('Paired gains and uncertainty', `Each row compares recovery with original predictions on the same test clips. The gain is in percentage points, not relative percent. A corrected clip was wrong under the baseline and correct after recovery. A regressed clip was correct before recovery and wrong after it. The remaining clips did not change correctness. The 95% interval comes from a paired clip bootstrap with 3000 resamples. These are per-condition intervals, without adjustment for five simultaneous comparisons. They capture sample uncertainty in this validation subset, not all deployment uncertainty.\n${conditions.map((c,i) => `${labels[i]}: ${JSON.stringify(c.paired_recovery)}`).join('\n')}`);
text(s, 'Recovery minus original accuracy, in percentage points', 72, 130, 1125, 42, 27, C.muted);
const pairedTable = table(s, [['Condition', 'Gain', '95% interval', 'Corrected', 'Regressed'], ...conditions.map((c, i) => {
  const p = c.paired_recovery;
  return [labels[i], pp(p.accuracy_gain), `${pp(p.ci95[0])} to ${pp(p.ci95[1])}`, String(p.improved_clips), String(p.regressed_clips)];
})], [340, 135, 285, 185, 195], 192, 351, 25);
conditions.forEach((c, i) => pairedTable.cells.block({ row: i + 1, column: 1, rowCount: 1, columnCount: 1 }).assign({ textStyle: { typeface: family, fontSize: 25, color: c.paired_recovery.accuracy_gain < 0 ? C.loss : C.green, bold: true } }));
text(s, 'Intervals containing zero leave the direction uncertain.', 72, 576, 1125, 49, 28, C.ink);

s = slide('Rejection changes accuracy and coverage', `The classifier can abstain when its maximum softmax score is below a calibration-selected threshold. The values are scores, not calibrated probabilities. Original threshold ${result.calibration.baseline_threshold}, recovery threshold ${result.calibration.recovery_threshold}. Calibration prefers coverage of at least 80%, which does not guarantee test coverage. Accepted accuracy excludes rejected clips and must be read together with coverage. The overall accuracy and macro F1 count rejected clips as failures. Rejection is not evidence of recognizing unseen action classes.\n${conditions.map((c,i) => `${labels[i]}: rejection only ${JSON.stringify(variant(c,'rejection'))}. Combined ${JSON.stringify(variant(c,'combined'))}.`).join('\n')}`);
text(s, 'Accepted accuracy describes only the clips the system labels', 72, 130, 1135, 43, 26, C.muted);
table(s, [['Condition', 'Rejection only\nAccuracy', 'Coverage', 'Recovery + rejection\nAccuracy', 'Coverage'], ...conditions.map((c, i) => {
  const a = variant(c, 'rejection'), b = variant(c, 'combined');
  return [labels[i], pct(a.accepted_accuracy), pct(a.coverage), pct(b.accepted_accuracy), pct(b.coverage)];
})], [320, 215, 140, 310, 155], 195, 373, 24);
text(s, `Score thresholds: ${result.calibration.baseline_threshold.toFixed(2)} original, ${result.calibration.recovery_threshold.toFixed(2)} with recovery`, 72, 598, 1120, 40, 26, C.ink);

const positive = conditions.filter(c => c.paired_recovery.accuracy_gain > 0).length;
const negative = conditions.filter(c => c.paired_recovery.accuracy_gain < 0).length;
s = slide('Limits and demonstration', `Use the dashboard at http://127.0.0.1:8765/research after launching the project. Compare all five conditions, then inspect Improved, Regressed and Both incorrect example sequences. The pose playback is a visualization of real keypoint input and recovered keypoints. Prediction captions refer to the full clip, not each displayed frame. Download the JSON to inspect the precise values and provenance. The separate /lab webcam prototype uses MediaPipe and a different classifier. It is not a measured live implementation of the paper's 400-class model.\nMeasured gains are positive for ${positive} of five conditions and negative for ${negative}, with per-condition intervals on slide 6. The project supports a bounded study, not a claim of general superiority. Limitations include validation sampling, synthetic damage, person-slot changes, and limited visibility into deployment behavior. ${protocol.timing_scope}`);
text(s, 'What the experiment supports', 72, 163, 540, 55, 30, C.green, true);
text(s, 'A measured study of a small\nrecovery extension', 72, 243, 525, 104, 31);
text(s, 'Sampled validation clips\nSynthetic missing joints and noise\nPerson slots can change identity', 72, 402, 550, 149, 27, C.muted);
text(s, 'Dashboard demonstration', 715, 163, 496, 55, 30, C.green, true);
text(s, 'Open /research\nCompare all five conditions\nPlay corrected and failed examples\nInspect or download the results', 715, 243, 493, 231, 29);
text(s, 'The webcam lab is a separate prototype.', 715, 528, 493, 76, 25, C.muted);

await fs.mkdir(BUILD, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const candidatePath = path.join(BUILD, `candidate-${revision}.pptx`);
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
const candidateHash = createHash('sha256').update(await fs.readFile(candidatePath)).digest('hex');
let validation;
try { validation = await finalizePresentation({
  workspaceDir: ROOT, candidatePath, finalPath,
  pythonExecutable: path.join(RUNTIME, 'python/bin/python3'),
  integrityValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit', '--require-native-table-slide', '6', '--require-native-table-slide', '7'],
  explicitTotalSlideCount: 8,
  requiredNativeTableOwnerSlides: [6, 7], requiredNativeChartOwnerSlides: [5],
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: 'design', families: [family] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(BUILD, `validation-${revision}.json`),
}); } finally {
  // The finalizer puts workbook snapshots in workspace-root scratch folders.
  // Archive only snapshots whose recorded input hash matches this invocation.
  for (const name of await fs.readdir(ROOT)) {
    if (!name.startsWith('.chart-data-')) continue;
    const scratch = path.join(ROOT, name);
    const receipt = await fs.readFile(path.join(scratch, 'chart-data-snapshot.json'), 'utf8').then(JSON.parse).catch(() => null);
    if (receipt?.input_sha256 === candidateHash) await fs.rename(scratch, path.join(BUILD, name));
  }
}
const finalDeck = await PresentationFile.importPptx(await FileBlob.load(finalPath));
for (let i = 0; i < finalDeck.slides.items.length; i++) {
  const blob = await finalDeck.export({ slide: finalDeck.slides.items[i], format: 'png', scale: 1 });
  await fs.writeFile(path.join(BUILD, `slide-${i + 1}.png`), new Uint8Array(await blob.arrayBuffer()));
}
const montage = await finalDeck.export({ format: 'webp', montage: true });
const previewPath = path.join(path.dirname(finalPath), 'PoseLab_Research_Preview.webp');
await fs.writeFile(previewPath, new Uint8Array(await montage.arrayBuffer()));
await fs.writeFile(path.join(BUILD, 'deck-source.json'), JSON.stringify({ inputPath, generated_at: result.generated_at, finalPath, previewPath, slides: 8, validation }, null, 2));
console.log(JSON.stringify({ finalPath, previewPath, slides: finalDeck.slides.items.length }));
