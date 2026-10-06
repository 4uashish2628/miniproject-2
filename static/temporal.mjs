// Temporal state is isolated so it can be tested without a camera or browser.
export class LabelSmoother {
  constructor(required = 3) {
    this.required = required;
    this.reset();
  }
  reset() {
    this.label = "unknown";
    this.candidate = null;
    this.hits = 0;
  }
  update(label) {
    if (label === "unknown") {
      this.reset();
      return this.label;
    }
    if (label === this.candidate) this.hits++;
    else {
      this.candidate = label;
      this.hits = 1;
    }
    // Abstain during transitions instead of displaying a stale activity.
    this.label = this.hits >= this.required ? label : "unknown";
    return this.label;
  }
}
export class RepCounter {
  constructor() {
    this.reset();
  }
  reset() {
    this.counts = { squats: 0, jumping_jacks: 0 };
    this.phase = "unarmed";
    this.activity = null;
    this.last = -Infinity;
  }
  interrupt() {
    this.phase = "unarmed";
    this.activity = null;
  }
  update(label, f, t) {
    if (!["squats", "jumping_jacks"].includes(label) || !f) {
      this.interrupt();
      return;
    }
    if (label !== this.activity) {
      this.activity = label;
      this.phase = "unarmed";
    }
    let closed, opened;
    if (label === "squats") {
      closed = Math.min(f.left_knee, f.right_knee) > 0.86;
      opened = Math.max(f.left_knee, f.right_knee) < 0.73;
    } else {
      closed =
        f.ankle_spread < 0.85 &&
        Math.max(f.left_wrist_height, f.right_wrist_height) < -0.2;
      opened =
        f.ankle_spread > 1.2 &&
        Math.min(f.left_wrist_height, f.right_wrist_height) > 0.4;
    }
    if (this.phase === "unarmed" && closed) this.phase = "ready";
    else if (this.phase === "ready" && opened) this.phase = "open";
    else if (this.phase === "open" && closed) {
      if (t - this.last > 0.6) {
        this.counts[label]++;
        this.last = t;
      }
      this.phase = "ready";
    }
  }
}
export function instantGeometry(p, aspect = 1) {
  const core = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  if (
    !p ||
    core.some(
      (i) =>
        p[i][3] < 0.5 ||
        p[i][0] < -0.05 ||
        p[i][0] > 1.05 ||
        p[i][1] < -0.05 ||
        p[i][1] > 1.05,
    )
  )
    return null;
  const xy = p.map((v) => [v[0] * aspect, v[1]]);
  const hip = [(xy[23][0] + xy[24][0]) / 2, (xy[23][1] + xy[24][1]) / 2];
  const shoulder = [(xy[11][0] + xy[12][0]) / 2, (xy[11][1] + xy[12][1]) / 2];
  const scale = Math.hypot(hip[0] - shoulder[0], hip[1] - shoulder[1]);
  if (scale < 0.06) return null;
  function angle(a, b, c) {
    const u = [xy[a][0] - xy[b][0], xy[a][1] - xy[b][1]],
      v = [xy[c][0] - xy[b][0], xy[c][1] - xy[b][1]];
    const d = Math.hypot(...u) * Math.hypot(...v);
    return d < 1e-7
      ? NaN
      : Math.acos(Math.min(1, Math.max(-1, (u[0] * v[0] + u[1] * v[1]) / d))) /
          Math.PI;
  }
  const f = {
    left_knee: angle(23, 25, 27),
    right_knee: angle(24, 26, 28),
    ankle_spread: Math.abs(xy[27][0] - xy[28][0]) / scale,
    left_wrist_height: (xy[11][1] - xy[15][1]) / scale,
    right_wrist_height: (xy[12][1] - xy[16][1]) / scale,
  };
  return Object.values(f).every(Number.isFinite) ? f : null;
}
