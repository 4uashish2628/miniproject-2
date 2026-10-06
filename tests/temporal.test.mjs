import test from "node:test";
import assert from "node:assert/strict";
import {
  LabelSmoother,
  RepCounter,
  instantGeometry,
} from "../static/temporal.mjs";
test("smoothing confirms three consecutive predictions and clears uncertainty", () => {
  const s = new LabelSmoother();
  assert.equal(s.update("squats"), "unknown");
  s.update("squats");
  assert.equal(s.update("squats"), "squats");
  assert.equal(s.update("standing"), "unknown");
  assert.equal(s.update("unknown"), "unknown");
  assert.equal(s.update("squats"), "unknown");
});
test("squat count requires a complete armed cycle and ignores jitter", () => {
  const c = new RepCounter();
  const up = { left_knee: 0.95, right_knee: 0.95 },
    down = { left_knee: 0.6, right_knee: 0.6 };
  c.update("squats", down, 0);
  c.update("squats", up, 1);
  assert.equal(c.counts.squats, 0);
  c.update("squats", down, 2);
  c.update("squats", up, 3);
  assert.equal(c.counts.squats, 1);
  for (let i = 0; i < 5; i++)
    c.update("squats", { left_knee: 0.8, right_knee: 0.8 }, 3.1 + i * 0.1);
  assert.equal(c.counts.squats, 1);
  c.update("squats", down, 5);
  c.update("unknown", null, 5.5);
  c.update("squats", up, 6);
  assert.equal(c.counts.squats, 1);
});
test("jumping jack state is independent of squat state", () => {
  const c = new RepCounter(),
    closed = {
      ankle_spread: 0.6,
      left_wrist_height: -0.5,
      right_wrist_height: -0.5,
    },
    open = {
      ankle_spread: 1.6,
      left_wrist_height: 0.8,
      right_wrist_height: 0.8,
    };
  c.update("jumping_jacks", closed, 0);
  c.update("jumping_jacks", open, 1);
  c.update("jumping_jacks", closed, 2);
  assert.equal(c.counts.jumping_jacks, 1);
  assert.equal(c.counts.squats, 0);
  c.reset();
  assert.equal(c.counts.jumping_jacks, 0);
});
test("missing or overlapping joints do not produce valid live geometry", () => {
  assert.equal(instantGeometry(null), null);
  assert.equal(
    instantGeometry(Array.from({ length: 33 }, () => [0.5, 0.5, 0, 1])),
    null,
  );
});
