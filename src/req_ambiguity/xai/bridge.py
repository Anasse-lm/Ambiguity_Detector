"""
bridge.py
---------
PlaceholderBridge: maps IG evidence tokens to placeholder candidates
via the trigger map, with structural-token filtering applied before lookup.

Structural token filtering takes precedence over trigger_map.yaml:
if a token is in structural_tokens.yaml, it is removed BEFORE the
trigger lookup fires. See configs/structural_tokens.yaml for the
precedence rule documentation.
"""
import re
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Token normalization
# ---------------------------------------------------------------------------

def normalize_token(token: str) -> str:
    """Normalize a token for comparison against the structural list.

    Handles DeBERTa BPE prefix markers, casing, and punctuation.

    Parameters
    ----------
    token : raw token string from the tokenizer (may contain ▁, ##, Ġ prefix)

    Returns
    -------
    Normalized lowercase string suitable for set-membership checks.
    """
    if not token:
        return ""
    # Strip BPE prefix markers
    token = token.lstrip("\u2581")      # DeBERTa ▁ (U+2581)
    token = re.sub(r"^##", "", token)  # BERT WordPiece ##
    token = token.lstrip("\u0120")     # GPT-2 / RoBERTa Ġ (U+0120)
    # Lowercase
    token = token.lower()
    # Strip leading/trailing punctuation
    token = token.rstrip(",.!?;:'\"")
    token = token.lstrip(",.!?;:'\"")
    return token.strip()


# ---------------------------------------------------------------------------
# PlaceholderBridge
# ---------------------------------------------------------------------------

class PlaceholderBridge:
    """
    Maps Integrated Gradients evidence tokens to placeholder candidates.

    Steps (per label):
      1. Receive raw IG evidence tokens (token, score) pairs.
      2. Filter out structural tokens (template scaffold words).
      3. Run trigger-map lookup on the filtered evidence.
      4. If no trigger fires, fall back to the first placeholder for the label.
    """

    def __init__(
        self,
        trigger_map_path: str = "configs/trigger_map.yaml",
        placeholders_path: str = "configs/placeholders.yaml",
    ):
        root = Path(__file__).resolve().parents[3]

        with open(root / trigger_map_path, "r", encoding="utf-8") as f:
            self.trigger_map = yaml.safe_load(f)

        with open(root / placeholders_path, "r", encoding="utf-8") as f:
            self.placeholders = yaml.safe_load(f)

        # Build lookup: label -> list of placeholder names
        self.label_to_placeholders: Dict[str, List[str]] = {}
        for label, ph_list in self.placeholders.items():
            if isinstance(ph_list, list):
                self.label_to_placeholders[label] = [ph["name"] for ph in ph_list]

        # Build lookup: placeholder name -> list of trigger words (normalized)
        self.placeholder_to_triggers: Dict[str, List[str]] = {}
        for ph_name, trigger_def in self.trigger_map.items():
            if not isinstance(trigger_def, dict):
                continue  # skip metadata keys like 'version'
            triggers = []
            for t in trigger_def.get("triggers", []):
                if isinstance(t, dict) and "word" in t:
                    triggers.append(normalize_token(t["word"]))
                elif isinstance(t, str):
                    triggers.append(normalize_token(t))
            self.placeholder_to_triggers[ph_name] = triggers

        # Load structural tokens for evidence filtering
        structural_path = root / "configs" / "structural_tokens.yaml"
        if structural_path.exists():
            with open(structural_path, "r", encoding="utf-8") as f:
                st_data = yaml.safe_load(f)
            # Pre-normalize the structural set once at init time
            self.normalized_structural_tokens: set = {
                normalize_token(t) for t in st_data.get("structural_tokens", [])
            }
        else:
            print(
                "[PlaceholderBridge] WARNING: structural_tokens.yaml not found; "
                "no filtering applied.",
                flush=True,
            )
            self.normalized_structural_tokens = set()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _filter_structural(
        self,
        tokens_with_scores: List[Tuple[str, float]],
        label: str,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Split (token, score) pairs into kept and removed lists.

        Parameters
        ----------
        tokens_with_scores : List of (raw_token, score)
        label              : label name (for logging)

        Returns
        -------
        (kept, removed) — both are List[(raw_token, score)]
        """
        kept, removed = [], []
        for tok, score in tokens_with_scores:
            norm = normalize_token(tok)
            if norm in self.normalized_structural_tokens or norm == "":
                removed.append((tok, score))
            else:
                kept.append((tok, score))
        return kept, removed

    def _print_filter_report(
        self,
        story_snippet: str,
        label: str,
        raw_pairs: List[Tuple[str, float]],
        kept_pairs: List[Tuple[str, float]],
        removed_pairs: List[Tuple[str, float]],
        selected_placeholders: List[str],
    ) -> None:
        """Print the verbose per-label filter report to stdout."""
        SEP = "=" * 37
        print(f"\n{SEP} BRIDGE FILTER REPORT {SEP}", flush=True)
        print(f"Label   : {label}", flush=True)
        print(f"Story   : {story_snippet[:60]}{'...' if len(story_snippet) > 60 else ''}", flush=True)
        print(f"Raw IG evidence tokens (top {len(raw_pairs)}):", flush=True)
        for i, (tok, sc) in enumerate(raw_pairs, 1):
            norm = normalize_token(tok)
            status = "FILTERED" if norm in self.normalized_structural_tokens or norm == "" else "KEPT    "
            print(f"  {i:2}. {tok:<20}  score={sc:+.4f}  [{status}]", flush=True)

        print(
            f"Tokens filtered: {len(removed_pairs)} of {len(raw_pairs)} "
            f"({len(removed_pairs)/max(len(raw_pairs),1)*100:.0f}%)",
            flush=True,
        )
        print(f"Final evidence for trigger lookup (top {min(5, len(kept_pairs))}):", flush=True)
        for i, (tok, sc) in enumerate(kept_pairs[:5], 1):
            print(f"  {i:2}. {tok:<20}  score={sc:+.4f}", flush=True)
        print(f"Selected placeholders: {selected_placeholders}", flush=True)
        print("=" * 95 + "\n", flush=True)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def match_evidence(
        self,
        label: str,
        evidence_tokens: List[Tuple[str, float]],
        story_text: str = "",
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Match IG evidence tokens against the trigger map for the given label.

        Parameters
        ----------
        label           : ambiguity label name
        evidence_tokens : List of (raw_token, abs_attribution_score)
        story_text      : original story text (for verbose logging)
        verbose         : if True, print the bridge filter report to stdout

        Returns
        -------
        List of dicts: {placeholder, match_score, matched_evidence, via_fallback}
        """
        results = []
        if label not in self.placeholders:
            return results

        placeholder_defs = self.placeholders[label]

        # 1. Filter structural tokens; preserve (token, score) pairs
        kept_pairs, removed_pairs = self._filter_structural(evidence_tokens, label)
        extracted_tokens = [normalize_token(tok) for tok, _ in kept_pairs]

        # 2. Trigger-map lookup on filtered tokens
        matched_any = False
        for pdef in placeholder_defs:
            placeholder_name = pdef["name"]
            triggers = self.placeholder_to_triggers.get(placeholder_name, [])

            # Substring matching between normalized evidence and normalized triggers
            matched_evidence = []
            for norm_tok in extracted_tokens:
                for trig in triggers:
                    if trig and norm_tok and (trig in norm_tok or norm_tok in trig):
                        matched_evidence.append(norm_tok)

            if matched_evidence:
                matched_any = True
                results.append({
                    "placeholder": placeholder_name,
                    "match_score": 1.0,
                    "matched_evidence": list(set(matched_evidence)),
                    "via_fallback": False,
                })

        # 3. Fallback if nothing matched
        if not matched_any and placeholder_defs:
            results.append({
                "placeholder": placeholder_defs[0]["name"],
                "match_score": 0.0,
                "matched_evidence": [],
                "via_fallback": True,
            })

        # 4. Verbose report
        if verbose:
            selected = [r["placeholder"] for r in results]
            self._print_filter_report(
                story_text, label, evidence_tokens, kept_pairs, removed_pairs, selected
            )

        return results

    def select_placeholders(
        self,
        active_labels: List[str],
        evidence_tokens_per_label: Dict[str, List[str]],
    ) -> List[str]:
        """
        For each active label, find placeholders whose trigger words overlap
        with the IG-derived evidence tokens for that label (after filtering).

        Parameters
        ----------
        active_labels              : list of predicted label names above threshold
        evidence_tokens_per_label  : dict mapping label -> list of raw token strings

        Returns
        -------
        Deduplicated list of matched placeholder names.
        """
        selected = []
        for label in active_labels:
            candidate_placeholders = self.label_to_placeholders.get(label, [])
            raw_tokens = evidence_tokens_per_label.get(label, [])

            # Convert strings to (token, 1.0) pairs for the shared filter helper
            raw_pairs = [(tok, 1.0) for tok in raw_tokens]
            kept_pairs, removed_pairs = self._filter_structural(raw_pairs, label)

            if removed_pairs:
                removed_words = [tok for tok, _ in removed_pairs]
                print(
                    f"[PlaceholderBridge.select_placeholders] Filtered {len(removed_words)} "
                    f"structural token(s) for '{label}': {sorted(removed_words)}",
                    flush=True,
                )

            filtered_norms = [normalize_token(tok) for tok, _ in kept_pairs]

            for placeholder in candidate_placeholders:
                triggers = self.placeholder_to_triggers.get(placeholder, [])
                if any(
                    trig and norm_tok and (trig in norm_tok or norm_tok in trig)
                    for norm_tok in filtered_norms
                    for trig in triggers
                ):
                    selected.append(placeholder)

        return list(set(selected))  # deduplicate
