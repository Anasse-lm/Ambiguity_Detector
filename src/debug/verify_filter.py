"""
verify_filter.py
----------------
Verifies the structural-token filter in PlaceholderBridge on 30 XAI stories.
Reports: mean tokens filtered per story, bridge selection fill rate, per-label stats.

Usage:
    PYTHONPATH=src python3 src/debug/verify_filter.py
"""
import sys
import io
import json
import collections
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from req_ambiguity.xai.bridge import PlaceholderBridge, normalize_token

ROOT = Path(__file__).resolve().parents[2]


def main():
    xai_dir = ROOT / "outputs" / "xai" / "json"
    out_file = ROOT / "outputs" / "debug" / "filter_verification_report.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(xai_dir.glob("*.json"), key=lambda f: f.stat().st_size, reverse=True)
    samples = all_files[:30]

    if not samples:
        print("No XAI JSON files found.")
        return

    bridge = PlaceholderBridge()

    # Counters
    total_stories = 0
    stories_with_empty_selection = 0
    total_filtered_count = 0
    total_evidence_count = 0
    per_label_placeholders: dict = collections.defaultdict(collections.Counter)
    per_label_fallback: dict = collections.defaultdict(int)
    story_rows = []

    report_lines = []

    def log(line=""):
        print(line, flush=True)
        report_lines.append(line)

    SEP = "=" * 70

    for path in samples:
        raw = json.load(open(path, "r", encoding="utf-8"))
        story_text = raw.get("original_text", "")
        total_stories += 1

        story_filtered = 0
        story_evidence = 0
        story_all_selections = []

        log(f"\n{SEP}")
        log(f"  Story : {story_text}")
        log(f"  File  : {path.name}")
        log(SEP)

        for label, exp in raw.get("label_explanations", {}).items():
            # Build evidence_tokens in the (token, score) format
            top_ev = exp.get("top_evidence_tokens", [])
            evidence_pairs = [(t["token"], abs(t["score"])) for t in top_ev]

            if not evidence_pairs:
                continue

            story_evidence += len(evidence_pairs)

            # Run match_evidence with verbose=False (we do our own logging)
            results = bridge.match_evidence(label, evidence_pairs, story_text, verbose=False)

            # Manually compute filter stats
            kept, removed = bridge._filter_structural(evidence_pairs, label)
            story_filtered += len(removed)

            # Log per-label
            raw_toks = [f"{t} ({s:.3f})" for t, s in evidence_pairs]
            kept_toks = [f"{t} ({s:.3f})" for t, s in kept]
            removed_toks = [t for t, s in removed]
            selected = [r["placeholder"] for r in results]
            fallback_flags = [r.get("via_fallback", False) for r in results]

            log(f"  Label : {label}")
            log(f"    Raw evidence   : {raw_toks}")
            log(f"    Filtered out   : {removed_toks}")
            log(f"    Kept evidence  : {kept_toks}")
            log(f"    Selections     : {selected}  (fallback={any(fallback_flags)})")

            story_all_selections.extend(selected)
            for ph in selected:
                per_label_placeholders[label][ph] += 1
            if any(fallback_flags):
                per_label_fallback[label] += 1

        total_filtered_count += story_filtered
        total_evidence_count += story_evidence

        if not story_all_selections:
            stories_with_empty_selection += 1

        story_rows.append({
            "story": story_text[:60],
            "filtered": story_filtered,
            "evidence": story_evidence,
            "selections": story_all_selections,
        })

    # ---- Aggregate ----
    log(f"\n{'#' * 70}")
    log("  AGGREGATE VERIFICATION REPORT")
    log(f"{'#' * 70}")
    log(f"  Stories analysed             : {total_stories}")
    log(f"  Total IG evidence tokens     : {total_evidence_count}")
    log(f"  Total structural filtered    : {total_filtered_count}")
    log(f"  Mean filtered per story      : {total_filtered_count / max(total_stories, 1):.2f}")
    log(f"  Mean evidence per story      : {total_evidence_count / max(total_stories, 1):.2f}")
    log(f"  Filter rate (%)              : {total_filtered_count / max(total_evidence_count, 1) * 100:.1f}%")
    log(f"  Stories with empty selection : {stories_with_empty_selection} / {total_stories}  "
        f"({stories_with_empty_selection / max(total_stories, 1) * 100:.1f}%)")
    log(f"  Stories with non-empty sel.  : {total_stories - stories_with_empty_selection} / {total_stories}  "
        f"({(total_stories - stories_with_empty_selection) / max(total_stories, 1) * 100:.1f}%)")

    log(f"\n  Per-label top placeholders selected:")
    for label, counter in sorted(per_label_placeholders.items()):
        top = counter.most_common(3)
        fallbacks = per_label_fallback.get(label, 0)
        log(f"    {label}:")
        for ph, cnt in top:
            log(f"      {ph:<35} selected {cnt} times")
        log(f"      Fallback selections: {fallbacks}")

    log(f"\n  ASSESSMENT:")
    mean_filtered = total_filtered_count / max(total_stories, 1)
    fill_rate = (total_stories - stories_with_empty_selection) / max(total_stories, 1) * 100
    if mean_filtered < 2:
        log("  ⚠  Low filtering: mean < 2 tokens removed per story. "
            "Filter may not be effective.")
    elif mean_filtered > 6:
        log("  ⚠  High filtering: mean > 6 tokens removed per story. "
            "Filter may be too aggressive.")
    else:
        log(f"  ✓  Filter rate is within target range ({mean_filtered:.2f} tokens/story).")

    if fill_rate < 80:
        log(f"  ⚠  Bridge fill rate {fill_rate:.1f}% < 80% target. "
            "Many stories have empty placeholder selections.")
    else:
        log(f"  ✓  Bridge fill rate {fill_rate:.1f}% meets or exceeds 80% target.")

    log(f"{'#' * 70}\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nVerification report saved to {out_file}")


if __name__ == "__main__":
    main()
