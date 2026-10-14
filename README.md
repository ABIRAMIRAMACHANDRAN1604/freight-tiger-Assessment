# Shipping-cost watchdog

```
python assistant.py shipment_records.csv context_notes.csv output.csv   # writes output.csv + output.csv.costlog.json
python eval.py context_notes.csv shipment_records.csv                   # labeled checks + 3-run hash diff
```
Python 3.9+, standard library only.

## How it works
1. **Group** by route (`origin-destination`) and week (Monday label). `route_type` is used as given.
2. **Cost/tonne-km** per route-week = sum(cost) / sum(tonnes x km) (ratio of sums, as the brief defines it).
3. **vs own history**: mean of the route's weekly values over the up-to-8 most recent prior weeks that have shipments, strictly before the current week. With fewer than 8, all prior weeks are used (no padding). A row needs >= 2 prior weeks to be judged.
4. **vs similar routes**: mean of other routes' values, same week, same `route_type`, self excluded.
5. **Candidate** if >= +8% vs own history AND >= +10% vs peers (`OWN_RISE_MIN`, `PEER_RISE_MIN`). Output contains candidate weeks only, matching the sample file.
6. **Verdict** (rules, not an LLM). A note justifies a rise only if all hold: effect is *increase*, scope is this route or All Routes, note window overlaps the week, magnitude plausible. Otherwise `Yes` with blank `matched_note_id`.
7. **Retrieval**: TF-IDF cosine over note text plus time proximity ranks in-scope notes. It picks the "closest note" quoted in unexplained reasons. It never grants a justification.

## Assumptions to challenge
- Notes were annotated by hand (10 notes) into scope / window / effect; `verify_annotations` fails the run if they drift from the CSV.
- N001 window is Feb 24 - Mar 8 from its text, so week of 2025-03-10 is NOT justified (N009 says normal).
- N003 (diesel) has no end date: I assume 4 weeks, and it only explains rises <= +12% (note says ~5-7%). Bigger rises stay unexplained.
- Thresholds are first guesses; tune them on the real file via the eval before trusting the flag count.

## Trust and reproducibility
- `eval.py` scores 16 labeled verdict cases (wrong route, wrong week, "stable demand", "costs not affected", "back to normal", diesel magnitude/window) and a synthetic end-to-end check. Fabricated-justification count must be 0.
- No randomness, no LLM, sorted iteration: three runs give identical sha256 (printed by `eval.py`).

## Cost log (one full run)
LLM calls: 0 | input tokens: 0 | output tokens: 0 | cost: $0. Reason text is templated from the matched note. The optional Q&A stretch goal is not built; it would be the only place an LLM (temperature 0, grounded on this output CSV) is needed.
## Final results
- Thresholds: OWN_RISE_MIN = 0.08, PEER_RISE_MIN = 0.10
- Output: 9 rows (6 flagged, 3 justified)
- Eval: 16/16 labeled verdicts correct, 0 fabricated justifications
- Reproducibility: 3 untouched runs, identical sha256
  2950d63bd45fd9ecb7f1e82b884320080b68a3f06dfb53762cc283f73f0283b1
- Cost (one full run): 0 LLM calls, 0 input tokens, 0 output tokens, $0