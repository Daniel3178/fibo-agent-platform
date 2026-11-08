#!/usr/bin/env python
"""Summarise MoE gate health from gating-diagnostics outputs (read-only).

Usage:
    python gate_health.py RUN_DIR [RUN_DIR ...]
    python gate_health.py BENCHMARK_DIR          # every subdirectory holding epoch_metrics.csv

RUN_DIR is a directory written by `train_neural_SDE.py --gating-diagnostics` and/or
`analyze_gating` (epoch_metrics.csv, val_metrics.json, real_gates.csv, events.csv).
Thresholds in `verdict` are heuristics, not results from the literature.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERTS = ("Bounce", "Break", "Hover")
PROBS = ("p_bounce", "p_break", "p_hover")


def entropy_norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1)
    return float((-(p * np.log(p)).sum(1) / np.log(3)).mean())


def winner_shares(p):
    return np.bincount(np.asarray(p).argmax(1), minlength=3) / len(p)


def mutual_information(a, b):
    """I(a; b) / H(a) for two discrete label arrays (0 = independent, 1 = b determines a)."""
    joint = pd.crosstab(a, b, normalize=True).to_numpy()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())
    h = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    return mi / h if h > 0 else np.nan


def summarise(run):
    row = {"run": run.name}
    epochs = pd.read_csv(run / "epoch_metrics.csv")
    last = epochs[epochs.split == "validation"].sort_values("epoch").iloc[-1]
    row["epoch"] = int(last.epoch)
    row["val_entropy"] = last.entropy
    row["val_confidence"] = last.confidence
    for col, name in zip(PROBS, EXPERTS):
        row[f"val_win_{name}"] = last[col + "_winner_share"]
    row["val_mean_pi_var"] = float(np.mean([last[c + "_variance"] for c in PROBS]))
    # I(X; E)/log 3 = H(pbar)/log 3 - mean H(pi)/log 3: ~0 means every input gets the same gate.
    pbar = np.array([[last[c] for c in PROBS]])
    row["val_mi_input_expert"] = entropy_norm(pbar) - last.entropy

    metrics = run / "val_metrics.json"
    if metrics.exists():
        values = json.loads(metrics.read_text())
        values = values[0] if isinstance(values, list) else values
        row["val_loss_sde"] = values.get("val/loss_sde", np.nan)

    gates = run / "real_gates.csv"
    if gates.exists():
        frame = pd.read_csv(gates, usecols=[*PROBS, "distance", "movement"])
        p = frame[list(PROBS)].to_numpy()
        row["real_entropy"] = entropy_norm(p)
        for share, name in zip(winner_shares(p), EXPERTS):
            row[f"real_win_{name}"] = share
        # Bounce drift sign = sign(distance), Break = sign(movement): where they agree the experts are interchangeable.
        row["real_sign_agree"] = float((np.sign(frame.distance) == np.sign(frame.movement)).mean())

    events = run / "events.csv"
    if events.exists():
        ev = pd.read_csv(events)
        ev = ev[ev.event.isin(EXPERTS)]
        # Mean gate of the matching expert on its own events minus on the other events.
        contrast = [ev.loc[ev.event == name, col].mean() - ev.loc[ev.event != name, col].mean()
                    for col, name in zip(PROBS, EXPERTS)]
        row["event_contrast"] = float(np.mean(contrast))
        row["event_mi"] = mutual_information(ev.event.to_numpy(), ev[list(PROBS)].to_numpy().argmax(1))
    return row


def verdict(row):
    wins = [row[f"val_win_{name}"] for name in EXPERTS]
    flags = []
    if max(wins) >= 0.95:
        flags.append("hard-collapse")
    elif min(wins) < 0.02:
        flags.append("starved-expert")
    if row["val_entropy"] >= 0.85 and row["val_confidence"] < 0.6:
        flags.append("uniform-soft")
    if row.get("event_contrast", np.inf) < 0.05:
        flags.append("non-semantic")
    if not flags and min(wins) >= 0.1:
        flags.append("healthy")
    return ",".join(flags) or "mixed"


def find_runs(paths):
    runs = []
    for path in map(Path, paths):
        if (path / "epoch_metrics.csv").exists():
            runs.append(path)
        else:
            runs.extend(sorted(p.parent for p in path.glob("*/epoch_metrics.csv")))
    return runs


def main(argv):
    runs = find_runs(argv or ["."])
    if not runs:
        sys.exit("No epoch_metrics.csv found in the given directories.")
    table = pd.DataFrame([summarise(run) for run in runs]).set_index("run")
    table["verdict"] = [verdict(row) for _, row in table.iterrows()]
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(table.T if len(table) > 1 else table.iloc[0].to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
