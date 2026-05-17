import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional


class PromptBuilder:
    """
    Constructs the prompt for the Refinement module.

    Pass 1 (initial):   build_prompt()        — sends only the bridge-selected
                        placeholders with their FULL metadata (description,
                        trigger, clarification_template, before/after example).
    Pass 2 (follow-up): build_followup_prompt() — user has answered the
                        clarification questions; Gemini replaces placeholders
                        with the real values provided by the user.
    """

    def __init__(
        self,
        template_path: str = "configs/refinement_prompt_template.txt",
        placeholders_path: str = "configs/placeholders.yaml",
    ):
        # Load the prompt template
        template_file = Path(template_path)
        if not template_file.exists():
            raise FileNotFoundError(f"Prompt template missing: {template_path}")
        with open(template_file, "r", encoding="utf-8") as f:
            self.template = f.read()

        # Load the placeholders mapping — keep ALL metadata
        placeholders_file = Path(placeholders_path)
        if not placeholders_file.exists():
            raise FileNotFoundError(f"Placeholders file missing: {placeholders_path}")
        with open(placeholders_file, "r", encoding="utf-8") as f:
            placeholders_map = yaml.safe_load(f)

        # Flat dict: name -> full metadata dict
        self.placeholder_metadata: Dict[str, Dict] = {}
        # Flat dict: name -> description (kept for backward compat with chip rendering)
        self.placeholder_descriptions: Dict[str, str] = {}

        for category, ph_list in placeholders_map.items():
            if isinstance(ph_list, list):
                for ph in ph_list:
                    name = ph["name"]
                    self.placeholder_metadata[name] = ph
                    self.placeholder_descriptions[name] = ph.get("description", "").strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_placeholder_block(self, allowed_placeholders: List[str]) -> str:
        """
        Build the ALLOWED PLACEHOLDERS section of the prompt using the
        FULL metadata from placeholders.yaml (description, trigger,
        clarification_template, before/after example).

        Only placeholders in `allowed_placeholders` are included.
        """
        blocks = []
        for ph_name in allowed_placeholders:
            meta = self.placeholder_metadata.get(ph_name)
            if not meta:
                blocks.append(f"{ph_name}: (no metadata available)")
                continue

            desc = meta.get("description", "").strip()
            trigger = meta.get("trigger", "").strip()
            clarification = meta.get("clarification_template", "").strip()
            example = meta.get("example", {})
            before = example.get("before", "")
            after = example.get("after", "")

            lines = [f"{ph_name}:"]
            lines.append(f"  Description          : {desc}")
            if trigger:
                lines.append(f"  When to use          : {trigger}")
            if clarification:
                lines.append(f"  Clarification to ask : {clarification}")
            if before and after:
                lines.append(f"  Example before       : {before}")
                lines.append(f"  Example after        : {after}")

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Pass 1: initial refinement
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        original_story: str,
        active_labels: List[str],
        evidence_tokens: List[str],
        allowed_placeholders: List[str],
        previous_error: Optional[str] = None,
    ) -> str:
        """
        Build the initial refinement prompt.

        Only `allowed_placeholders` (bridge-selected) are included in the prompt
        with their full metadata. Gemini must not use any placeholder outside
        this list.
        """
        active_labels_str = ", ".join(active_labels)
        evidence_tokens_str = ", ".join(f'"{tok}"' for tok in evidence_tokens)
        allowed_placeholders_str = self._build_placeholder_block(allowed_placeholders)

        base_prompt = self.template.format(
            ORIGINAL_STORY=original_story,
            ACTIVE_LABELS=active_labels_str,
            EVIDENCE_TOKENS=evidence_tokens_str,
            ALLOWED_PLACEHOLDERS=allowed_placeholders_str,
        )

        if previous_error:
            retry_instruction = (
                "\n\n============================================================\n"
                "RETRY INSTRUCTION\n"
                "============================================================\n"
                "Your previous response had the following issue:\n"
                f"{previous_error}\n\n"
                "Please correct the issue and return only valid JSON with legal "
                "placeholders from the allowed list."
            )
            return base_prompt + retry_instruction

        return base_prompt

    # ------------------------------------------------------------------
    # Pass 2: follow-up with user answers
    # ------------------------------------------------------------------

    def build_followup_prompt(
        self,
        original_story: str,
        refined_story: str,
        clarification_questions: List[str],
        user_answers: Dict[str, str],
    ) -> str:
        """
        Build a follow-up prompt where the user has answered the clarification
        questions. Gemini replaces each <TBD_...> placeholder with the real
        value the user provided.

        Parameters
        ----------
        original_story          : the original ambiguous story
        refined_story           : the placeholder-filled story from Pass 1
        clarification_questions : list of questions Gemini asked in Pass 1
        user_answers            : dict mapping question text -> user's answer

        Returns
        -------
        A prompt string that instructs Gemini to produce the final concrete story.
        """
        qa_lines = []
        for i, q in enumerate(clarification_questions):
            answer = user_answers.get(q, "").strip()
            if answer:
                qa_lines.append(f"  Q{i+1}: {q}")
                qa_lines.append(f"  A{i+1}: {answer}")
            else:
                qa_lines.append(f"  Q{i+1}: {q}")
                qa_lines.append(f"  A{i+1}: (not answered — keep placeholder)")

        qa_block = "\n".join(qa_lines)

        prompt = (
            "You are a requirements engineering assistant. A user story was previously\n"
            "refined by replacing vague terms with structured placeholders. The\n"
            "stakeholder has now answered the clarification questions. Your task is to\n"
            "produce the FINAL concrete story by substituting each placeholder with\n"
            "the stakeholder's answer.\n\n"
            "RULES:\n"
            "1. Replace every <TBD_...> placeholder that has a non-empty answer with\n"
            "   the stakeholder's answer, integrated naturally into the sentence.\n"
            "2. Keep placeholders that have no answer (leave them as-is).\n"
            "3. Preserve the user-story sentence structure.\n"
            "4. Do NOT add information beyond what the stakeholder provided.\n"
            "5. Return ONLY a JSON object with exactly these keys:\n"
            '   {"final_story": "<the fully resolved story>",\n'
            '    "remaining_placeholders": ["<TBD_X>", ...],\n'
            '    "resolution_summary": "<one sentence describing what was clarified>"}\n\n'
            "============================================================\n"
            "ORIGINAL STORY:\n"
            f"{original_story}\n\n"
            "PLACEHOLDER-REFINED STORY (from Pass 1):\n"
            f"{refined_story}\n\n"
            "STAKEHOLDER ANSWERS:\n"
            f"{qa_block}\n\n"
            "============================================================\n"
            "Return ONLY the JSON object, nothing else. No markdown fences.\n"
        )
        return prompt

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    def build_retry_prompt(
        self,
        original_story: str,
        active_labels: List[str],
        evidence_tokens: List[str],
        allowed_placeholders: List[str],
        previous_error: str,
    ) -> str:
        return self.build_prompt(
            original_story, active_labels, evidence_tokens,
            allowed_placeholders, previous_error=previous_error
        )

    def render_for_inspection(
        self,
        original_story: str,
        active_labels: List[str],
        evidence_tokens: List[str],
        allowed_placeholders: List[str],
    ) -> str:
        return self.build_prompt(original_story, active_labels, evidence_tokens, allowed_placeholders)
