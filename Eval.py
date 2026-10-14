#!/usr/bin/env python3
"""Eval harness. Run: python eval.py context_notes.csv [shipment_records.csv]

A. Labeled verdict set (traps from the brief) scored for exact verdict + note id.
B. Synthetic end-to-end check of weekly maths, baselines and flagging.
C. Optional: 3 untouched runs on the real file, hash-compared.
"""
import csv
import os
import sys
import tempfile
from datetime import date

import assistant as A

notes_csv = sys.argv[1]
note_rows = A.verify_annotations(notes_csv)
index, idf = A.build_index(note_rows)
W = lambda s: date.fromisoformat(s)

# ---- A. labeled verdicts: (route, week, own_rise, expected_note_id or None)
LABELED = [
    ("Ahmedabad-Mumbai", "2025-01-20", 0.295, "N002"),   # festival week, correct route+week
    ("Ahmedabad-Mumbai", "2025-01-27", 0.30, None),      # week after festival: out of window
    ("Mumbai-Pune",      "2025-01-20", 0.30, None),      # N002 is for a different route
    ("Chennai-Bangalore", "2025-02-24", 0.30, "N001"),   # flood window start
    ("Chennai-Bangalore", "2025-03-03", 0.30, "N001"),   # flood window, 2nd week
    ("Chennai-Bangalore", "2025-03-10", 0.25, None),     # window ended Mar 8; N009 says normal
    ("Chennai-Bangalore", "2025-03-17", 0.25, None),     # N009 'back to normal'
    ("Chennai-Bangalore", "2025-02-17", 0.30, None),     # before window
    ("Delhi-Jaipur",     "2024-11-11", 0.355, None),     # no note at all
    ("Mumbai-Pune",      "2025-09-15", 0.092, None),     # N006 'stable demand' must not justify
    ("Mumbai-Delhi",     "2024-07-29", 0.20, None),      # N005 'costs not affected'
    ("Kolkata-Bhubaneswar", "2025-06-09", 0.20, None),   # N008 'no disruptions'
    ("Mumbai-Pune",      "2025-05-12", 0.07, "N003"),    # diesel, plausible magnitude, in window
    ("Mumbai-Pune",      "2025-05-12", 0.30, None),      # diesel cannot explain +30%
    ("Mumbai-Pune",      "2025-06-09", 0.07, None),      # beyond assumed 4-week diesel window
    ("Mumbai-Pune",      "2025-10-27", 0.20, None),      # N010 'absorbed, no rate change'
]
tp = fp = fn = tn = bad = 0
for route, wk, rise, want in LABELED:
    got, _ = A.verdict(route, W(wk), rise, A.rank_notes(route, W(wk), index, idf))
    ok = got == want
    bad += not ok
    tp += bool(got) and ok
    fp += bool(got) and got != want
    fn += (want is not None) and got is None
    tn += (not got) and (want is None)
    print(("PASS " if ok else "FAIL ") + f"{route} {wk} rise={rise:+.0%} expected={want} got={got}")
print(f"A. verdicts: {len(LABELED)-bad}/{len(LABELED)} correct | justified TP={tp} FP(fabricated)={fp} FN={fn} TN={tn}")
assert fp == 0, "a wrongly-justified row is the worst failure mode"

# ---- B. synthetic end-to-end: two Short routes, flat 4.0, one spike week
def synth(path):
    rows, n = [], 1
    start = W("2025-01-06")  # a Monday
    for wk in range(12):
        for route, (o, d) in {"A-B": ("A", "B"), "C-D": ("C", "D")}.items():
            for day in (1, 3):  # two shipments / week (Tue, Thu)
                dt = date.fromordinal(start.toordinal() + 7 * wk + day).isoformat()
                rate = 4.0 * (1.5 if (route == "A-B" and wk == 10) else 1.0)
                rows.append([f"S{n:04d}", o, d, "Short", "X", 10, 100, round(rate * 10 * 100), dt, "T"])
                n += 1
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("shipment_id,origin,destination,route_type,material,quantity_tonnes,distance_km,freight_cost_inr,shipment_date,transporter".split(","))
        w.writerows(rows)

tmp = tempfile.mkdtemp()
sp, op = os.path.join(tmp, "s.csv"), os.path.join(tmp, "o.csv")
synth(sp)
res = A.run(sp, notes_csv, op)
assert len(res) == 1, f"expected exactly 1 flag, got {len(res)}"
r = res[0]
assert (r["route"], r["week_of"]) == ("A-B", "2025-03-17"), r
assert r["cost_per_tonne_km"] == "6.00" and r["vs_own_history"].startswith("+50.0%"), r
assert r["vs_similar_routes"].startswith("+50.0%"), r
assert r["flagged"] == "Yes" and r["matched_note_id"] == ""
print("B. synthetic maths + flagging: PASS ->", r["vs_own_history"], "|", r["vs_similar_routes"])

# ---- C. reproducibility on real data (optional)
if len(sys.argv) > 2:
    hashes = []
    for i in range(3):
        p = os.path.join(tmp, f"run{i}.csv")
        A.run(sys.argv[2], notes_csv, p)
        hashes.append(A.file_hash(p))
    print("C. 3-run sha256:", *hashes, sep="\n   ")
    assert len(set(hashes)) == 1, "runs differ!"
    print("C. identical across 3 runs: PASS")