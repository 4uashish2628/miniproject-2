"""Procedural skeletons for software demonstrations only, never research data."""

import math


def skeleton(label, t):
    p = [[0.5, 0.3, 0, 0.99] for _ in range(33)]

    def put(i, x, y):
        p[i] = [x, y, 0, 0.99]

    put(0, 0.5, 0.16)
    for a in range(1, 11):
        put(a, 0.5 + (a % 2 - 0.5) * 0.03, 0.16 + (a // 4) * 0.015)
    coords = {
        11: (0.41, 0.29),
        12: (0.59, 0.29),
        13: (0.36, 0.43),
        14: (0.64, 0.43),
        15: (0.34, 0.56),
        16: (0.66, 0.56),
        23: (0.44, 0.54),
        24: (0.56, 0.54),
        25: (0.43, 0.72),
        26: (0.57, 0.72),
        27: (0.43, 0.92),
        28: (0.57, 0.92),
    }
    u = (1 - math.cos(t * math.pi * 1.3)) / 2
    if label in ("squats", "sitting"):
        bend = u if label == "squats" else 0.95
        for i in (11, 12, 13, 14, 15, 16, 23, 24):
            x, y = coords[i]
            coords[i] = (x, y + 0.16 * bend)
        coords[25] = (0.43 - 0.12 * bend, 0.72 + 0.025 * bend)
        coords[26] = (0.57 + 0.12 * bend, 0.72 + 0.025 * bend)
        put(0, 0.5, 0.16 + 0.16 * bend)
    if label == "jumping_jacks":
        for side in (-1, 1):
            shoulder = 11 if side < 0 else 12
            coords[shoulder + 2] = (
                0.5 + side * (0.15 + 0.07 * math.sin(u * math.pi)),
                0.43 - 0.22 * u,
            )
            coords[shoulder + 4] = (
                0.5 + side * (0.16 * (1 - u) + 0.09 * u),
                0.56 - 0.47 * u,
            )
        coords[27] = (0.43 - 0.18 * u, 0.92)
        coords[28] = (0.57 + 0.18 * u, 0.92)
        coords[25] = (0.43 - 0.08 * u, 0.72)
        coords[26] = (0.57 + 0.08 * u, 0.72)
    if label == "marching":
        for i, phase in ((25, 0), (26, math.pi)):
            lift = max(0, math.sin(t * math.pi * 1.6 + phase)) * 0.16
            x, y = coords[i]
            coords[i] = (x, y - lift)
            x, y = coords[i + 2]
            coords[i + 2] = (x + 0.025, y - lift)
    if label == "waving":
        coords[14] = (0.68, 0.26)
        coords[16] = (0.69 + 0.1 * math.sin(t * math.pi * 3), 0.12)
    for i, (x, y) in coords.items():
        put(i, x, y)
    for i in (17, 19, 21):
        p[i] = p[15].copy()
    for i in (18, 20, 22):
        p[i] = p[16].copy()
    for i in (29, 31):
        p[i] = [p[27][0] - 0.02, p[27][1] + 0.015, 0, 0.99]
    for i in (30, 32):
        p[i] = [p[28][0] + 0.02, p[28][1] + 0.015, 0, 0.99]
    return p


def demo_clip(label, duration=8, phase=0):
    return [
        {"t": i / 10, "landmarks": skeleton(label, i / 10 + phase)}
        for i in range(round(duration * 10) + 1)
    ]
