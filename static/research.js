/* Research dashboard. Every plotted number comes from research-results.json. */
"use strict";

(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const byId = (id) => document.getElementById(id);
  const defaults = [
    {
      id: "baseline",
      label: "Baseline",
      description: "The pretrained classifier without our additions.",
    },
    {
      id: "recovery",
      label: "Recovery only",
      description: "Pose recovery before the same classifier.",
    },
    {
      id: "rejection",
      label: "Rejection only",
      description: "The original input, with unreliable predictions rejected.",
    },
    {
      id: "combined",
      label: "Combined",
      description: "Pose recovery and prediction rejection together.",
    },
  ];
  let experiment = null;
  let metric = "accuracy";
  let animation = null;
  let playing = false;
  let frame = 0;
  let lastTick = 0;
  let activeExample = null;

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }

  function svgElement(tag, attrs = {}, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs))
      node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  const isRate = (value) =>
    typeof value === "number" &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 1;
  const percentage = (value) =>
    isRate(value) ? `${(value * 100).toFixed(2)}%` : "—";
  const count = (value) =>
    Number.isInteger(value) && value >= 0 ? value.toLocaleString() : "—";
  const numberOrDash = (value) =>
    typeof value === "number" && Number.isFinite(value) ? value : "—";
  const formatValue = (value) =>
    value === null || value === undefined || value === ""
      ? "Not recorded"
      : typeof value === "object"
        ? JSON.stringify(value)
        : String(value);
  const variant = (id) =>
    experiment?.variants.find((item) => item.id === id) ||
    defaults.find((item) => item.id === id);
  const result = (condition, id) =>
    condition?.results?.find((item) => item.variant === id);
  const selectedCondition = () =>
    experiment?.conditions.find(
      (condition) => condition.id === byId("condition-select").value,
    );

  function delta(value, baseline) {
    if (!isRate(value) || !isRate(baseline))
      return { text: "—", className: "delta-neutral" };
    const points = (value - baseline) * 100;
    const rounded = Math.round(points * 100) / 100;
    return {
      text: `${rounded > 0 ? "+" : rounded < 0 ? "−" : ""}${Math.abs(rounded).toFixed(2)} pp`,
      className:
        rounded > 0
          ? "delta-positive"
          : rounded < 0
            ? "delta-negative"
            : "delta-neutral",
    };
  }

  function addMetadata(label, value, isHash = false) {
    const row = element("div");
    row.append(
      element("dt", label),
      element("dd", formatValue(value), isHash ? "hash" : ""),
    );
    byId("protocol-details").append(row);
  }

  function safeLink(value) {
    try {
      const url = new URL(value, window.location.origin);
      return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch {
      return null;
    }
  }

  function renderProtocol() {
    const protocol = experiment.protocol || {};
    byId("protocol-details").replaceChildren();
    const rows = [
      ["Dataset", protocol.dataset],
      ["Split", protocol.split],
      ["Test examples", protocol.test_count ?? protocol.sample_count],
      ["Calibration examples", protocol.calibration_count],
      ["Classes present", protocol.classes_present],
      ["Model classes", protocol.class_count],
      ["Random seed", protocol.seed],
      ["Selection", protocol.selection],
      ["Checkpoint", protocol.checkpoint],
      ["Checkpoint SHA-256", protocol.checkpoint_sha256, true],
      ["Upstream commit", protocol.upstream_commit, true],
      ["Experiment date", experiment.generated_at],
    ];
    for (const row of rows) addMetadata(...row);
    if (protocol.source_url) {
      const url = safeLink(protocol.source_url);
      if (url) {
        const row = element("div");
        const dd = element("dd");
        const link = element("a", "Open dataset / source ↗");
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        dd.append(link);
        row.append(element("dt", "Data source"), dd);
        byId("protocol-details").append(row);
      }
    }
    if (protocol.description) addMetadata("Protocol", protocol.description);
    if (protocol.corruptions) addMetadata("Corruptions", protocol.corruptions);
    if (protocol.latency_description || protocol.timing_scope)
      byId("latency-note").textContent =
        protocol.latency_description || protocol.timing_scope;
    if (protocol.macro_f1_scope)
      addMetadata("Macro F1 scope", protocol.macro_f1_scope);
    if (protocol.hardware) addMetadata("Hardware", protocol.hardware);
    if (protocol.device) addMetadata("Device", protocol.device);
    if (protocol.batch_size) addMetadata("Batch size", protocol.batch_size);
    if (experiment.calibration?.selected_candidate)
      addMetadata(
        "Recovery selected",
        experiment.calibration.selected_candidate,
      );
    if (experiment.calibration?.frozen_at)
      addMetadata("Settings frozen", experiment.calibration.frozen_at);
    if (protocol.subset_limitation)
      byId("subset-limitation").textContent = protocol.subset_limitation;
    const notes = Array.isArray(experiment.notes) ? experiment.notes : [];
    byId("experiment-notes").replaceChildren(
      ...notes.map((note) => element("li", note)),
    );
    if (!notes.length)
      byId("experiment-notes").append(
        element(
          "li",
          "Use the supplied experiment record when interpreting these results.",
        ),
      );
    if (experiment.generated_at)
      byId("updated-at").textContent = `Experiment: ${experiment.generated_at}`;
  }

  function renderVariants() {
    byId("variant-descriptions").replaceChildren();
    experiment.variants.forEach((item, index) => {
      const card = element("article", null, "variant-description");
      card.append(
        element("span", String(index + 1).padStart(2, "0"), "variant-number"),
        element("h3", item.label),
        element("p", item.description),
      );
      byId("variant-descriptions").append(card);
    });
  }

  function renderSelectedCondition() {
    const condition = selectedCondition();
    const baseline = result(condition, "baseline");
    const recovery = result(condition, "recovery");
    const combined = result(condition, "combined");
    const protocol = experiment.protocol || {};
    byId("condition-description").textContent =
      ({
        clean: "Real recorded poses with no added corruption.",
        burst_missing: "Temporarily hide individual joints in five-frame blocks, using the same missing-joint pattern for every method.",
        limb_missing: "Temporarily hide connected arm or leg joints in five-frame blocks, using identical inputs for every method.",
        long_missing: "Hide individual joints in longer fifteen-frame blocks to test the limits of short-gap repair.",
        jitter: "Add random coordinate noise with standard deviation 0.02 in normalized image units, using identical noise for every method.",
      })[condition?.id] || "All variants use the same examples for this condition.";
    if (
      isRate(condition?.mean_actual_dropout) &&
      condition.mean_actual_dropout > 0
    )
      byId("condition-description").textContent +=
        ` Mean actual masked joint-frames: ${percentage(condition.mean_actual_dropout)}.`;
    byId("baseline-accuracy").textContent = percentage(baseline?.accuracy);
    byId("recovery-accuracy").textContent = percentage(recovery?.accuracy);
    byId("combined-coverage").textContent = percentage(combined?.coverage);
    const difference = delta(recovery?.accuracy, baseline?.accuracy);
    byId("accuracy-delta").textContent =
      difference.text === "—"
        ? "Difference not available"
        : `${difference.text} vs baseline`;
    byId("accuracy-delta").className = difference.className;
    byId("coverage-detail").textContent = isRate(combined?.coverage)
      ? `Recovery + rejection: ${((1 - combined.coverage) * 100).toFixed(1)}% rejected`
      : "Accepted predictions / all examples";
    byId("test-count").textContent = count(
      protocol.test_count ?? protocol.sample_count ?? baseline?.n_samples,
    );
    byId("class-count").textContent = Number.isInteger(protocol.classes_present)
      ? `${count(protocol.classes_present)} classes in this test subset`
      : "Same examples for every variant";
    byId("coverage-condition").textContent =
      condition?.label || condition?.id || "Selected condition";
    const paired = condition?.paired_recovery;
    const interval = paired?.ci95;
    const validInterval =
      Array.isArray(interval) &&
      interval.length === 2 &&
      interval.every(
        (value) => typeof value === "number" && Number.isFinite(value),
      );
    byId("paired-recovery").hidden = !validInterval;
    if (validInterval) {
      const points = (value) =>
        `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value * 100).toFixed(2)}`;
      byId("paired-title").textContent =
        `Recovery difference · ${condition.label || condition.id}`;
      byId("paired-interval").textContent =
        `${difference.text} accuracy; 95% paired bootstrap interval: ${points(interval[0])} to ${points(interval[1])} percentage points.`;
      byId("paired-counts").textContent =
        `${count(paired.improved_clips)} clips changed from wrong to correct; ${count(paired.regressed_clips)} changed from correct to wrong.`;
      byId("paired-scope").textContent =
        `${interval[0] <= 0 && interval[1] >= 0 ? "The interval includes zero; this pilot does not establish a consistent gain. " : ""}Clip-level uncertainty for this test subset and fixed corruption seed. ${paired.method || "Paired clip bootstrap"}.`;
    }
    const container = byId("coverage-chart");
    container.className = "";
    container.replaceChildren();
    for (const item of experiment.variants) {
      const measured = result(condition, item.id);
      const row = element("div", null, "coverage-row");
      const title = element("div", null, "coverage-title");
      title.append(
        element("span", item.label),
        element("strong", percentage(measured?.accepted_accuracy)),
      );
      const meter = element("div", null, "coverage-meter");
      meter.setAttribute("aria-hidden", "true");
      const fill = element("span");
      fill.style.width = `${isRate(measured?.coverage) ? measured.coverage * 100 : 0}%`;
      meter.append(fill);
      const meta = element("div", null, "coverage-meta");
      meta.append(
        element("span", "Accepted accuracy"),
        element("span", `${percentage(measured?.coverage)} coverage`),
      );
      row.append(title, meter, meta);
      container.append(row);
    }
  }

  function renderChart() {
    const conditions = experiment.conditions;
    const rowHeight = 85;
    const height = conditions.length * rowHeight + 44;
    const left = 133;
    const plotWidth = 357;
    const svg = svgElement("svg", {
      viewBox: `0 0 560 ${height}`,
      role: "img",
      "aria-labelledby": "chart-title chart-desc",
    });
    svg.append(
      svgElement(
        "title",
        { id: "chart-title" },
        `${metric === "accuracy" ? "Accuracy" : "Macro F1"} across corruption conditions`,
      ),
    );
    svg.append(
      svgElement(
        "desc",
        { id: "chart-desc" },
        "Grouped bars compare the original model with joint recovery, without rejection. Exact values are in the complete comparison table below.",
      ),
    );
    for (let tick = 0; tick <= 100; tick += 25) {
      const x = left + (tick / 100) * plotWidth;
      svg.append(
        svgElement("line", {
          x1: x,
          x2: x,
          y1: 12,
          y2: height - 29,
          stroke: "#e9eee8",
          "stroke-width": 1,
        }),
      );
      svg.append(
        svgElement(
          "text",
          {
            x,
            y: height - 10,
            fill: "#6c7d76",
            "font-size": 10,
            "text-anchor": "middle",
          },
          `${tick}%`,
        ),
      );
    }
    conditions.forEach((condition, index) => {
      const y = index * rowHeight + 22;
      const label = String(condition.label || condition.id);
      const lines = [""];
      for (const word of label.split(" ")) {
        const last = lines.length - 1;
        if (lines[last] && `${lines[last]} ${word}`.length > 22)
          lines.push(word);
        else lines[last] += `${lines[last] ? " " : ""}${word}`;
      }
      const title = svgElement("text", {
        x: left - 13,
        y: y + 17 - (Math.min(lines.length, 3) - 1) * 6,
        fill: "#425b50",
        "font-size": 11,
        "text-anchor": "end",
      });
      lines
        .slice(0, 3)
        .forEach((line, lineIndex) =>
          title.append(
            svgElement("tspan", { x: left - 13, dy: lineIndex ? 13 : 0 }, line),
          ),
        );
      title.append(svgElement("title", {}, label));
      svg.append(title);
      for (const [offset, id, color] of [
        [0, "baseline", "#829da9"],
        [24, "recovery", "#2d6a56"],
      ]) {
        const value = result(condition, id)?.[metric];
        if (isRate(value))
          svg.append(
            svgElement("rect", {
              x: left,
              y: y + offset,
              width: value * plotWidth,
              height: 16,
              rx: 3,
              fill: color,
            }),
          );
        svg.append(
          svgElement(
            "text",
            {
              x: left + (isRate(value) ? value * plotWidth : 0) + 7,
              y: y + offset + 12,
              fill: "#364d42",
              "font-size": 11,
            },
            percentage(value),
          ),
        );
      }
    });
    byId("performance-chart").className = "";
    byId("performance-chart").replaceChildren(svg);
  }

  function renderTable() {
    const body = byId("results-body");
    body.replaceChildren();
    for (const condition of experiment.conditions) {
      const heading = element("tr", null, "condition-heading");
      const cell = element("th", condition.label || condition.id);
      cell.colSpan = 7;
      cell.scope = "colgroup";
      heading.append(cell);
      body.append(heading);
      const baseline = result(condition, "baseline");
      for (const item of experiment.variants) {
        const measured = result(condition, item.id);
        const row = element(
          "tr",
          null,
          item.id === "combined" ? "combined-row" : "",
        );
        const label = element("th");
        label.scope = "row";
        label.append(element("span", item.label, "table-variant"));
        if (Number.isInteger(measured?.n_samples))
          label.append(
            element(
              "span",
              `${count(measured.n_samples)} examples`,
              "table-samples",
            ),
          );
        row.append(label);
        for (const key of [
          "accuracy",
          "macro_f1",
          "coverage",
          "accepted_accuracy",
        ])
          row.append(element("td", percentage(measured?.[key])));
        const difference = delta(measured?.accuracy, baseline?.accuracy);
        row.append(
          element(
            "td",
            item.id === "baseline" ? "Reference" : difference.text,
            item.id === "baseline" ? "delta-neutral" : difference.className,
          ),
        );
        const latency = measured?.latency_ms;
        const time = element(
          "td",
          typeof latency === "number" &&
            Number.isFinite(latency) &&
            latency >= 0
            ? `${latency.toFixed(1)} ms`
            : "—",
        );
        if (
          measured &&
          (measured.classifier_ms !== undefined ||
            measured.recovery_ms !== undefined)
        )
          time.title = `Classifier: ${numberOrDash(measured.classifier_ms)} ms; recovery: ${numberOrDash(measured.recovery_ms)} ms. Offline clip measurement.`;
        row.append(time);
        body.append(row);
      }
    }
  }

  function frameCount() {
    return Math.min(
      activeExample?.before?.length || 0,
      activeExample?.after?.length || 0,
    );
  }

  function pointCoordinates(point) {
    if (
      !Array.isArray(point) ||
      point.length < 2 ||
      !Number.isFinite(point[0]) ||
      !Number.isFinite(point[1])
    )
      return null;
    if (point.length >= 3 && (!Number.isFinite(point[2]) || point[2] <= 0))
      return null;
    return [72 + point[0] * 296, 22 + point[1] * 296];
  }

  function drawSkeleton(id, points, color) {
    const svg = byId(id);
    const drawing = svg.querySelector(".pose-drawing");
    drawing.replaceChildren();
    const coordinates = Array.isArray(points)
      ? points.map(pointCoordinates)
      : [];
    const edges = Array.isArray(activeExample?.edges)
      ? activeExample.edges
      : [];
    for (const edge of edges) {
      if (!Array.isArray(edge)) continue;
      const a = coordinates[edge[0]],
        b = coordinates[edge[1]];
      if (a && b)
        drawing.append(
          svgElement("line", {
            x1: a[0],
            y1: a[1],
            x2: b[0],
            y2: b[1],
            stroke: color,
            "stroke-width": 3,
            "stroke-linecap": "round",
            opacity: 0.78,
          }),
        );
    }
    for (const [index, point] of coordinates.entries())
      if (point)
        drawing.append(
          svgElement("circle", {
            cx: point[0],
            cy: point[1],
            r: 3.5,
            fill:
              id === "pose-after" &&
              !pointCoordinates(activeExample.before[frame]?.[index])
                ? "#d78044"
                : color,
            stroke: "#fff",
            "stroke-width": 1.2,
          }),
        );
    if (!coordinates.some(Boolean))
      drawing.append(
        svgElement(
          "text",
          {
            x: 220,
            y: 170,
            fill: "#6c7d76",
            "font-size": 12,
            "text-anchor": "middle",
          },
          "No valid joints in this frame",
        ),
      );
  }

  function drawExample() {
    if (!activeExample) return;
    drawSkeleton("pose-before", activeExample.before[frame], "#7a939f");
    drawSkeleton("pose-after", activeExample.after[frame], "#2d6a56");
    byId("example-frame").value = String(frame);
    byId("frame-label").textContent = `Frame ${frame + 1} / ${frameCount()}`;
    const before = activeExample.prediction_baseline;
    const after = activeExample.prediction_recovery;
    byId("before-prediction").textContent =
      before === undefined || before === null
        ? "Prediction not attached"
        : `Original: ${before}`;
    byId("after-prediction").textContent =
      after === undefined || after === null
        ? "Prediction not attached"
        : `Recovery: ${after}`;
    byId("example-rejection-note").textContent =
      activeExample.prediction_combined === undefined ||
      activeExample.prediction_combined === null
        ? ""
        : `After adding rejection, the combined method returns: ${activeExample.prediction_combined}.`;
  }

  function stopAnimation() {
    playing = false;
    if (animation !== null) cancelAnimationFrame(animation);
    animation = null;
    byId("play-example").textContent = "Play";
  }

  function advance(timestamp) {
    if (!playing) return;
    const fps =
      typeof activeExample?.fps === "number" && activeExample.fps > 0
        ? activeExample.fps
        : 30;
    const interval = 1000 / Math.min(fps, 60);
    if (!lastTick) lastTick = timestamp;
    if (timestamp - lastTick >= interval) {
      frame =
        (frame + Math.floor((timestamp - lastTick) / interval)) % frameCount();
      lastTick = timestamp;
      drawExample();
    }
    animation = requestAnimationFrame(advance);
  }

  const conditionLabel = (id) => ({
    clean: "Original poses",
    burst_missing: "Short joint gaps",
    limb_missing: "Short limb gaps",
    long_missing: "Long joint gaps",
    jitter: "Coordinate noise",
  })[id] || id;

  function selectExample() {
    stopAnimation();
    activeExample = experiment.examples.find(
      (example) => example.id === byId("example-select").value,
    );
    // Start at the first observed frame; the dataset includes empty frame 0.
    frame = Math.max(
      0,
      activeExample.before.findIndex((points) =>
        Array.isArray(points) && points.some((point) => pointCoordinates(point)),
      ),
    );
    byId("example-frame").max = String(Math.max(0, frameCount() - 1));
    byId("play-example").disabled = frameCount() < 2;
    const description = [
      activeExample.category ? `Outcome: ${activeExample.category}` : null,
      activeExample.label ? `Ground truth: ${activeExample.label}` : null,
      activeExample.condition ? `Condition: ${conditionLabel(activeExample.condition)}` : null,
      activeExample.id ? `Example: ${activeExample.id}` : null,
    ]
      .filter(Boolean)
      .join(" · ");
    byId("example-description").textContent =
      activeExample.description || description;
    drawExample();
  }

  function renderExamples() {
    const examples = (
      Array.isArray(experiment.examples) ? experiment.examples : []
    ).filter(
      (item) =>
        Array.isArray(item.before) &&
        item.before.length &&
        Array.isArray(item.after) &&
        item.after.length,
    );
    experiment.examples = examples.map((item, index) => ({
      ...item,
      id: String(item.id ?? index),
    }));
    if (!examples.length) return;
    byId("example-empty").hidden = true;
    byId("example-content").hidden = false;
    byId("example-selector-wrap").hidden = false;
    byId("example-select").replaceChildren();
    for (const example of experiment.examples) {
      const option = element(
        "option",
        example.label
          ? `${example.category ? `${example.category}: ` : ""}${example.label} · ${conditionLabel(example.condition) || example.id}`
          : example.id,
      );
      option.value = example.id;
      byId("example-select").append(option);
    }
    for (const id of ["pose-before", "pose-after"]) {
      const grid = byId(id).querySelector(".pose-grid-lines");
      for (let x = 0; x <= 440; x += 40)
        grid.append(
          svgElement("line", {
            x1: x,
            x2: x,
            y1: 0,
            y2: 340,
            stroke: "#e6ece2",
            "stroke-width": 1,
          }),
        );
      for (let y = 0; y <= 340; y += 40)
        grid.append(
          svgElement("line", {
            x1: 0,
            x2: 440,
            y1: y,
            y2: y,
            stroke: "#e6ece2",
            "stroke-width": 1,
          }),
        );
    }
    selectExample();
  }

  function showPending(message) {
    byId("run-status").textContent = "Results pending";
    byId("run-status").classList.add("pending");
    if (message) byId("load-status").textContent = message;
  }

  async function loadExperiment() {
    try {
      const response = await fetch("/static/research-results.json", {
        cache: "no-store",
      });
      if (!response.ok) {
        showPending(
          response.status === 404
            ? "No experiment file is available yet. Run the research evaluation to populate this page with measured results."
            : `Results could not be loaded (HTTP ${response.status}). No measurements are being displayed.`,
        );
        return;
      }
      const data = await response.json();
      if (!["complete", "completed"].includes(data.status)) {
        showPending(
          data.message ||
            "The experiment is not complete yet. No final measurements are being displayed.",
        );
        return;
      }
      if (
        !Array.isArray(data.conditions) ||
        !data.conditions.length ||
        !data.conditions.some(
          (condition) =>
            Array.isArray(condition.results) &&
            condition.results.some((item) => isRate(item.accuracy)),
        )
      ) {
        showPending(
          "The results file contains no valid completed measurements. Check the evaluation output before interpreting this page.",
        );
        return;
      }
      experiment = data;
      experiment.variants = defaults.map((item) => ({
        ...item,
        ...(Array.isArray(data.variants)
          ? data.variants.find((entry) => entry.id === item.id)
          : {}),
      }));
      byId("run-status").textContent = "Measured results";
      byId("run-status").classList.remove("pending");
      byId("load-status").className = "notice loaded";
      byId("load-status").textContent =
        "Completed local experiment. These are measured results for the documented test subset, not the paper's full benchmark scores.";
      byId("download-results").removeAttribute("aria-disabled");
      const select = byId("condition-select");
      select.replaceChildren();
      for (const condition of experiment.conditions) {
        const option = element("option", condition.label || condition.id);
        option.value = condition.id;
        select.append(option);
      }
      select.disabled = false;
      const corruption = experiment.conditions.find(
        (condition) => condition.id === "burst_missing",
      );
      if (corruption) select.value = corruption.id;
      renderProtocol();
      renderVariants();
      renderSelectedCondition();
      renderChart();
      renderTable();
      renderExamples();
    } catch (error) {
      showPending(
        "The experiment results could not be read. Start the local application and check that its results file is valid JSON. No measurements have been invented.",
      );
      console.error("Research results could not be loaded", error);
    }
  }

  byId("condition-select").addEventListener("change", renderSelectedCondition);
  document.querySelectorAll("[data-metric]").forEach((button) =>
    button.addEventListener("click", () => {
      metric = button.dataset.metric;
      document.querySelectorAll("[data-metric]").forEach((item) => {
        item.classList.toggle("selected", item === button);
        item.setAttribute("aria-pressed", String(item === button));
      });
      if (experiment) renderChart();
    }),
  );
  byId("example-select").addEventListener("change", selectExample);
  byId("example-frame").addEventListener("input", (event) => {
    stopAnimation();
    frame = Number(event.target.value);
    drawExample();
  });
  byId("play-example").addEventListener("click", () => {
    if (playing) {
      stopAnimation();
      return;
    }
    if (!activeExample || frameCount() < 2) return;
    playing = true;
    lastTick = 0;
    byId("play-example").textContent = "Pause";
    animation = requestAnimationFrame(advance);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopAnimation();
  });
  loadExperiment();
})();
