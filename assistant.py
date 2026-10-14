#!/usr/bin/env python3
"""FreightTiger shipping-cost watchdog.

Pipeline (fully deterministic, zero LLM calls):
  1. Group shipments by route + week (Monday-Sunday, labelled by Monday).
  2. cost_per_tonne_km = sum(freight cost) / sum(tonnes * km) per route-week.
  3. Compare to own trailing history (<= 8 prior weeks) and to same-week peers
     (other routes with the same route_type).
  4. Candidate = rising vs both baselines by configured thresholds.
  5. Retrieval ranks notes; HARD RULES decide the verdict (route, window,
     effect direction, magnitude). No qualifying note -> unexplained.

Usage: python assistant.py shipment_records.csv context_notes.csv out.csv
"""
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------- config
HISTORY_WEEKS = 8        # per brief
MIN_PRIOR_WEEKS = 2      # need at least this much history to judge a rise
OWN_RISE_MIN = 0.08      # >= +8% vs own history ...
PEER_RISE_MIN = 0.10     # ... and >= +10% vs same-type peers => candidate
N003_WEEKS = 4           # diesel note: assume it applies for 4 weeks (documented assumption)
N003_MAX_RISE = 0.12     # note says ~5-7%; allow noise margin. Bigger rises stay unexplained.
FIELDS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history",
          "vs_similar_routes", "flagged", "matched_note_id", "reason"]

# ------------------------------------------------- hand-annotated notes
# Structured reading of context_notes.csv. Retrieval ranks over the raw text;
# only effect == "increase" notes inside scope+window may justify a rise.
# `verify_annotations` checks id/date/scope against the CSV at every run.
def _d(s):
    return datetime.strptime(s, "%Y-%m-%d").date()

NOTES = {
    "N001": dict(scope="Chennai-Bangalore", start="2025-02-24", end="2025-03-08", effect="increase",
                 cause="heavy flooding disrupted trucks (longer detours, higher trip costs)",
                 gist="describes flooding on this highway with higher trip costs"),
    "N002": dict(scope="Ahmedabad-Mumbai", start="2025-01-20", end="2025-01-26", effect="increase",
                 cause="a regional festival week drove a temporary transporter surcharge on this corridor",
                 gist="describes a festival-week surcharge on this corridor"),
    "N003": dict(scope="All Routes", start="2025-05-05", end=None, effect="increase", max_rise=N003_MAX_RISE,
                 cause="nationwide diesel prices rose (roughly 5-7% on transport cost)",
                 gist="describes a nationwide diesel rise of roughly 5-7%"),
    "N004": dict(scope="All Routes", start="2024-03-11", end=None, effect="none",
                 gist="describes a toll plaza on routes that are not in this dataset"),
    "N005": dict(scope="Mumbai-Delhi", start="2024-07-29", end=None, effect="none",
                 gist="says maintenance delays did not significantly affect costs"),
    "N006": dict(scope="All Routes", start="2025-09-22", end=None, effect="none",
                 gist="mentions stable demand with no major disruptions"),
    "N007": dict(scope="Delhi-Jaipur", start="2024-05-20", end=None, effect="none",
                 gist="reports improved road conditions, not a cost increase"),
    "N008": dict(scope="Kolkata-Bhubaneswar", start="2025-06-09", end=None, effect="none",
                 gist="reports no significant disruptions on this corridor"),
    "N009": dict(scope="Chennai-Bangalore", start="2025-03-17", end=None, effect="none",
                 gist="confirms the route returned to normal after flood repairs"),
    "N010": dict(scope="All Routes", start="2025-10-27", end=None, effect="none",
                 gist="says compliance costs were absorbed with no rate change"),
}
for _id, _n in NOTES.items():
    _n["id"] = _id
    _n["start_d"] = _d(_n["start"])
    if _id == "N003":
        _n["end"] = (_n["start_d"] + timedelta(weeks=N003_WEEKS) - timedelta(days=1)).isoformat()
    _n["end_d"] = _d(_n["end"]) if _n["end"] else _n["start_d"]


def verify_annotations(notes_csv):
    """Fail loudly if the hand annotations drift from context_notes.csv."""
    with open(notes_csv, newline="", encoding="utf-8") as f:
        rows = {r["note_id"]: r for r in csv.DictReader(f)}
    assert set(rows) == set(NOTES), f"note id mismatch: {set(rows) ^ set(NOTES)}"
    for nid, n in NOTES.items():
        r = rows[nid]
        assert r["date"] == n["start"], f"{nid} date differs from CSV"
        assert r["applies_to"] == n["scope"], f"{nid} scope differs from CSV"
    return rows


# ------------------------------------------------------------- core maths
def monday(d):
    return d - timedelta(days=d.weekday())


def load_weekly(shipments_csv):
    """-> weekly[route][week] = [cost, tonne_km]; rtype[route] = route_type"""
    weekly = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    rtype = {}
    with open(shipments_csv, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["shipment_id"])
    for r in rows:
        route = f'{r["origin"].strip()}-{r["destination"].strip()}'
        t = r["route_type"].strip()
        assert rtype.setdefault(route, t) == t, f"{route} has mixed route_type"
        w = monday(_d(r["shipment_date"].strip()))
        cell = weekly[route][w]
        cell[0] += float(r["freight_cost_inr"])
        cell[1] += float(r["quantity_tonnes"]) * float(r["distance_km"])
    return weekly, rtype


def weekly_cpt(weekly):
    return {rt: {w: c / tk for w, (c, tk) in sorted(ws.items()) if tk > 0}
            for rt, ws in weekly.items()}


def baselines(cpt, rtype, route, week):
    """(own_history_avg, n_prior, peer_avg) for one route-week. None if undefined."""
    prior = [v for w, v in sorted(cpt[route].items()) if w < week][-HISTORY_WEEKS:]
    own = sum(prior) / len(prior) if prior else None
    peers = [cpt[r][week] for r in sorted(cpt)
             if r != route and rtype[r] == rtype[route] and week in cpt[r]]
    peer = sum(peers) / len(peers) if peers else None
    return own, len(prior), peer


# ------------------------------------------------- retrieval (ranking only)
_tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())


def _vec(text, idf):
    tf = defaultdict(float)
    for t in _tok(text):
        tf[t] += 1
    return {t: c * idf.get(t, 1.0) for t, c in tf.items()}


def _cos(a, b):
    num = sum(v * b.get(t, 0.0) for t, v in a.items())
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def build_index(note_rows):
    docs = {nid: _tok(r["note"] + " " + r["applies_to"]) for nid, r in note_rows.items()}
    df = defaultdict(int)
    for toks in docs.values():
        for t in set(toks):
            df[t] += 1
    idf = {t: math.log((1 + len(docs)) / (1 + c)) + 1 for t, c in df.items()}
    return {nid: _vec(r["note"] + " " + r["applies_to"], idf) for nid, r in note_rows.items()}, idf


def in_scope(note, route):
    return note["scope"] in ("All Routes", route)


def overlaps(note, week):
    return note["start_d"] <= week + timedelta(days=6) and note["end_d"] >= week


def rank_notes(route, week, index, idf):
    """Notes in scope for this route, nearest in time first, text-similarity as tiebreak."""
    q = _vec(f"freight cost rise {route} {week.isoformat()} surcharge disruption fuel toll", idf)
    cands = [n for n in NOTES.values() if in_scope(n, route)]
    return sorted(cands, key=lambda n: (abs((n["start_d"] - week).days), -_cos(q, index[n["id"]]), n["id"]))


# ------------------------------------------------------------ verdict rules
def verdict(route, week, own_rise, ranked):
    """Return (note_id|None, reason). Rules, not an LLM, decide this."""
    blocked = None
    for n in sorted(ranked, key=lambda n: (n["scope"] == "All Routes", n["id"])):  # route-specific first
        if n["effect"] != "increase" or not overlaps(n, week):
            continue
        if n.get("max_rise") is not None and own_rise > n["max_rise"]:
            blocked = n
            continue
        return n["id"], (
            f'Matches note {n["id"]} dated {n["start"]}: {n["cause"]}. '
            f'Its window ({n["start"]} to {n["end"]}) covers the week of {week.isoformat()}. '
            f'The cost rise has a clear explanation.')
    if blocked:
        return None, (
            f'Note {blocked["id"]} ({blocked["start"]}) covers this week but {blocked["gist"]}; '
            f'that cannot account for a {own_rise*100:+.1f}% rise. No genuine justification found; flagged for review.')
    near = [n for n in ranked if abs((n["start_d"] - week).days) <= 45]
    if near:
        n = near[0]
        return None, (
            f'The closest note ({n["id"]}, {n["start"]}) {n["gist"]} -- it does not describe '
            f'a reason for a cost rise on this route in this week. No genuine justification found; flagged for review.')
    return None, "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."


# ----------------------------------------------------------------- driver
def run(shipments_csv, notes_csv, out_csv):
    note_rows = verify_annotations(notes_csv)
    index, idf = build_index(note_rows)
    weekly, rtype = load_weekly(shipments_csv)
    cpt = weekly_cpt(weekly)
    out = []
    for route in sorted(cpt):
        for week, cur in cpt[route].items():
            own, n_prior, peer = baselines(cpt, rtype, route, week)
            if own is None or n_prior < MIN_PRIOR_WEEKS or peer is None:
                continue
            own_rise, peer_rise = cur / own - 1, cur / peer - 1
            if own_rise < OWN_RISE_MIN or peer_rise < PEER_RISE_MIN:
                continue
            nid, reason = verdict(route, week, own_rise, rank_notes(route, week, index, idf))
            out.append({
                "route": route, "week_of": week.isoformat(),
                "cost_per_tonne_km": f"{cur:.2f}",
                "vs_own_history": f"{own_rise*100:+.1f}% vs this route's past average",
                "vs_similar_routes": f"{peer_rise*100:+.1f}% vs similar-length routes this week",
                "flagged": "No (justified)" if nid else "Yes",
                "matched_note_id": nid or "", "reason": reason})
    out.sort(key=lambda r: (r["week_of"], r["route"]))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    return out


def file_hash(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    rows = run(*sys.argv[1:])
    log = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
           "rows": len(rows), "output_sha256": file_hash(sys.argv[3])}
    json.dump(log, open(sys.argv[3] + ".costlog.json", "w"), indent=2)
    print(json.dumps(log, indent=2))