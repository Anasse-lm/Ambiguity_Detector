"""
attribution_diagnostic.py
--------------------------
Structural-vs-content attribution diagnostic for the IG explainability module.

Prints a per-story report to stdout (the terminal where Streamlit is running)
every time a story is analyzed. Does NOT modify the Streamlit UI.
"""
from __future__ import annotations

import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_token(token: str) -> str:
    """Strip BPE prefix markers and lowercase."""
    # DeBERTa uses '▁' (U+2581), BERT uses '##', GPT-style uses 'Ġ'
    token = token.lstrip("\u2581")   # DeBERTa sentencepiece marker
    token = re.sub(r"^##", "", token)  # BERT WordPiece marker
    token = token.lstrip("\u0120")   # Ġ (GPT-2 / RoBERTa marker)
    return token.lower().strip()


_STRUCTURAL_TOKENS_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "structural_tokens.yaml"
)


class AttributionDiagnostic:
    """
    Computes structural-vs-content attribution fractions from an XAI record
    and prints a formatted diagnostic report to stdout.

    The diagnostic is purely informational — it does NOT raise on error and
    does NOT modify any pipeline state.
    """

    def __init__(self, structural_tokens_path: str | Path = _STRUCTURAL_TOKENS_PATH):
        path = Path(structural_tokens_path)
        if not path.exists():
            print(f"[AttributionDiagnostic] WARNING: {path} not found. Diagnostic disabled.")
            self.structural_tokens: set = set()
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.structural_tokens = {t.lower() for t in data.get("structural_tokens", [])}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_and_print(
        self,
        xai_record: Dict[str, Any],
        story_text: str,
    ) -> Dict[str, Any]:
        """
        Analyze the XAI record and print a structural-attribution diagnostic.

        Parameters
        ----------
        xai_record : dict
            The in-memory XAI output produced by the Streamlit pipeline.
            Expected shape::

                {
                  label: {
                    "tokens": List[str],
                    "scores": List[float],
                    "top_evidence_tokens": List[(token, score)]
                  }
                }

        story_text : str
            Original user-story text (for display only).

        Returns
        -------
        dict with diagnostic summary (for optional session logging).
        """
        try:
            return self._run(xai_record, story_text)
        except Exception as exc:  # noqa: BLE001
            print(f"[AttributionDiagnostic] Non-fatal error: {exc}")
            return {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, xai_record: Dict[str, Any], story_text: str) -> Dict[str, Any]:
        active_labels = [k for k in xai_record if isinstance(xai_record[k], dict)]

        if not active_labels:
            print("\n[AttributionDiagnostic] No active labels for this story; diagnostic skipped.")
            return {}

        SEP = "=" * 64

        print(f"\n{SEP}")
        print("  IG ATTRIBUTION DIAGNOSTIC")
        print(f"  Story: {story_text}")
        print(SEP)

        per_label_breakdown: Dict[str, Dict] = {}
        all_tokens_seen: Dict[str, float] = {}   # token -> max |score| across labels

        global_struct_mass = 0.0
        global_content_mass = 0.0
        global_total_mass = 0.0

        for label in active_labels:
            label_data = xai_record[label]
            tokens: List[str] = label_data.get("tokens", [])
            scores: List[float] = label_data.get("scores", [])
            prob: float = label_data.get("predicted_probability", 0.0)

            if not tokens or not scores:
                continue

            pairs: List[Tuple[str, float]] = list(zip(tokens, [abs(s) for s in scores]))

            struct_mass = 0.0
            content_mass = 0.0
            struct_count = 0
            content_count = 0

            for tok, abs_score in pairs:
                norm = _normalize_token(tok)
                if norm in self.structural_tokens:
                    struct_mass += abs_score
                    struct_count += 1
                else:
                    content_mass += abs_score
                    content_count += 1
                # track globally (for top-5 lists)
                all_tokens_seen[tok] = max(all_tokens_seen.get(tok, 0.0), abs_score)

            total_mass = struct_mass + content_mass
            struct_pct = (struct_mass / total_mass * 100) if total_mass > 0 else 0.0
            content_pct = (content_mass / total_mass * 100) if total_mass > 0 else 0.0
            assessment = self._assess(struct_pct)

            global_struct_mass += struct_mass
            global_content_mass += content_mass
            global_total_mass += total_mass

            per_label_breakdown[label] = {
                "prob": prob,
                "struct_pct": round(struct_pct, 1),
                "content_pct": round(content_pct, 1),
                "struct_count": struct_count,
                "content_count": content_count,
                "assessment": assessment,
            }

        # ---- overall numbers ----
        n_total = len(all_tokens_seen)
        n_struct = sum(1 for t in all_tokens_seen if _normalize_token(t) in self.structural_tokens)
        n_content = n_total - n_struct
        overall_struct_pct = (
            global_struct_mass / global_total_mass * 100
        ) if global_total_mass > 0 else 0.0
        overall_content_pct = 100.0 - overall_struct_pct
        overall_assessment = self._assess(overall_struct_pct)

        # ---- print summary ----
        print(f"  Total tokens (excl. [CLS], [SEP], padding): {n_total}")
        print(f"  Structural tokens : {n_struct} ({n_struct/max(n_total,1)*100:.1f}%)")
        print(f"  Content tokens    : {n_content} ({n_content/max(n_total,1)*100:.1f}%)")
        print()
        print(f"  Total absolute attribution mass : {global_total_mass:.3f}")
        print(f"  Structural attribution mass     : {global_struct_mass:.3f} ({overall_struct_pct:.1f}%)")
        print(f"  Content attribution mass        : {global_content_mass:.3f} ({overall_content_pct:.1f}%)")
        print()
        print("  Per-label breakdown:")
        for label, br in per_label_breakdown.items():
            short = label.replace("Ambiguity", "")
            print(
                f"    {short:<22} (prob={br['prob']:.2f}): "
                f"struct={br['struct_pct']:.1f}%, content={br['content_pct']:.1f}%"
                f" — {br['assessment']}"
            )

        # ---- top-5 filtered vs unfiltered ----
        sorted_all = sorted(all_tokens_seen.items(), key=lambda x: x[1], reverse=True)
        top5_unfiltered = sorted_all[:5]
        top5_filtered = [
            (t, s) for t, s in sorted_all
            if _normalize_token(t) not in self.structural_tokens
        ][:5]

        print()
        print("  Top 5 evidence tokens (after filtering structural):")
        for i, (tok, sc) in enumerate(top5_filtered, 1):
            print(f"    {i}. {_normalize_token(tok):<20} (attribution: {sc:.3f})")

        print()
        print("  Top 5 evidence tokens (BEFORE filtering structural):")
        for i, (tok, sc) in enumerate(top5_unfiltered, 1):
            is_struct = _normalize_token(tok) in self.structural_tokens
            tag = " [STRUCTURAL]" if is_struct else ""
            print(f"    {i}. {_normalize_token(tok):<20} (attribution: {sc:.3f}){tag}")

        # ---- final assessment ----
        print()
        if overall_assessment == "HEALTHY":
            print(
                f"  ASSESSMENT: HEALTHY — structural attribution is below the 33% warning\n"
                f"  threshold. The model is primarily attending to content tokens."
            )
        elif overall_assessment == "BORDERLINE":
            print(
                f"  ASSESSMENT: BORDERLINE — structural attribution is {overall_struct_pct:.1f}%\n"
                f"  (33–50% range). Monitor this pattern across more stories."
            )
        else:
            print(
                f"  ASSESSMENT: WARNING — structural attribution is {overall_struct_pct:.1f}% "
                f"(above 33%\n"
                f"  threshold). The model may be exploiting template patterns. Consider\n"
                f"  re-evaluation on non-template data and document this finding in the\n"
                f"  threats-to-validity section of the thesis."
            )
        print(SEP + "\n")

        # ---- return dict ----
        return {
            "story_text": story_text,
            "active_labels": active_labels,
            "total_tokens": n_total,
            "structural_tokens_count": n_struct,
            "content_tokens_count": n_content,
            "total_attribution": round(global_total_mass, 4),
            "structural_attribution": round(global_struct_mass, 4),
            "content_attribution": round(global_content_mass, 4),
            "structural_pct": round(overall_struct_pct, 2),
            "content_pct": round(overall_content_pct, 2),
            "per_label_breakdown": per_label_breakdown,
            "top_evidence_filtered": [(t, round(s, 4)) for t, s in top5_filtered],
            "top_evidence_unfiltered": [(t, round(s, 4)) for t, s in top5_unfiltered],
            "assessment": overall_assessment,
        }

    @staticmethod
    def _assess(struct_pct: float) -> str:
        if struct_pct <= 33.0:
            return "HEALTHY"
        if struct_pct <= 50.0:
            return "BORDERLINE"
        return "WARNING"
