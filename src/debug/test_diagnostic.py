"""
test_diagnostic.py
------------------
Batch-runs AttributionDiagnostic on 30 sample XAI JSON files and saves
the printed reports to outputs/debug/diagnostic_samples.txt.

Usage:
    PYTHONPATH=src python3 src/debug/test_diagnostic.py
"""
import sys
import io
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from req_ambiguity.xai.attribution_diagnostic import AttributionDiagnostic


def main():
    root = Path(__file__).resolve().parents[2]
    xai_dir = root / "outputs" / "xai" / "json"
    output_file = root / "outputs" / "debug" / "diagnostic_samples.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Pick 30 sample files that have rich data (larger file size)
    all_files = sorted(xai_dir.glob("*.json"), key=lambda f: f.stat().st_size, reverse=True)
    samples = all_files[:30]

    if not samples:
        print("No XAI JSON files found.")
        return

    diagnostic = AttributionDiagnostic()
    all_output = []

    # Aggregate stats
    struct_pcts = []
    assessments = {"HEALTHY": 0, "BORDERLINE": 0, "WARNING": 0}
    skipped = 0

    for path in samples:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        story_text = raw.get("original_text", "")

        # Build xai_results in the format that the Streamlit pipeline produces
        xai_results = {}
        for label, exp in raw.get("label_explanations", {}).items():
            tokens = [t["word"] for t in exp.get("word_level_attributions", [])]
            scores = [t["score"] for t in exp.get("word_level_attributions", [])]
            top_ev = [(t["token"], t["score"]) for t in exp.get("top_evidence_tokens", [])]
            xai_results[label] = {
                "tokens": tokens,
                "scores": scores,
                "top_evidence_tokens": top_ev,
                "predicted_probability": exp.get("predicted_probability", 0.0),
            }

        if not xai_results:
            skipped += 1
            continue

        # Capture stdout while running diagnostic
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        try:
            result = diagnostic.analyze_and_print(xai_results, story_text)
        finally:
            sys.stdout = old_stdout

        captured = buffer.getvalue()
        print(captured, end="")  # echo to real terminal too
        all_output.append(f"=== {path.name} ===\n{captured}\n")

        if result:
            struct_pcts.append(result.get("structural_pct", 0.0))
            a = result.get("assessment", "")
            if a in assessments:
                assessments[a] += 1

    # ---- Aggregate summary ----
    n = len(struct_pcts)
    SEP = "=" * 64
    summary = (
        f"\n{SEP}\n"
        f"  AGGREGATE SUMMARY — {n} stories analysed\n"
        f"{SEP}\n"
        f"  Skipped (no label data)  : {skipped}\n"
        f"  HEALTHY   (struct ≤33%)  : {assessments['HEALTHY']}  ({assessments['HEALTHY']/max(n,1)*100:.1f}%)\n"
        f"  BORDERLINE (33–50%)      : {assessments['BORDERLINE']}  ({assessments['BORDERLINE']/max(n,1)*100:.1f}%)\n"
        f"  WARNING   (struct >50%)  : {assessments['WARNING']}  ({assessments['WARNING']/max(n,1)*100:.1f}%)\n"
        f"\n"
        f"  Mean structural attribution : {sum(struct_pcts)/max(n,1):.1f}%\n"
        f"  Min  structural attribution : {min(struct_pcts):.1f}%\n"
        f"  Max  structural attribution : {max(struct_pcts):.1f}%\n"
        f"{SEP}\n"
    )
    print(summary)
    all_output.append(summary)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_output))

    print(f"Full report saved to {output_file}")


if __name__ == "__main__":
    main()
