"""
Prompt Injection Detector — detect LLM prompt injection in tool outputs.

Scans tool output text for patterns that attempt to override system instructions,
inject new roles, or manipulate the agent's behavior.

Sprint 17 (Security & Sandboxing) Module 2.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Detection patterns ───────────────────────────────────────────

# Role confusion / impersonation
ROLE_CONFUSION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\b(you are now|you must act as|pretend to be|roleplay as)\b"), "role_override"),
    (re.compile(r"(?i)\b(system:\s|assistant:\s|human:\s)"), "role_label_injection"),
    (re.compile(r"(?i)\b(ignore (all )?(previous|prior|above) (instructions|prompts|rules))\b"), "instruction_override"),
    (re.compile(r"(?i)\b(forget (everything|all|your) (you|instructions|rules))\b"), "memory_wipe"),
    (re.compile(r"(?i)\b(new instructions?:)\b"), "new_instruction_injection"),
]

# Safety bypass attempts
SAFETY_BYPASS_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\b(jailbreak|DAN mode|developer mode|god mode)\b"), "jailbreak_keyword"),
    (re.compile(r"(?i)\b(bypass (safety|security|filter|restriction))\b"), "bypass_attempt"),
    (re.compile(r"(?i)\b(disable (safety|security|content) (filter|check|policy))\b"), "disable_safety"),
    (re.compile(r"(?i)\b(override (safety|security|content) (policy|rules|guidelines))\b"), "override_safety"),
    (re.compile(r"(?i)\b(no (restrictions?|limits?|rules?|boundaries))\b"), "remove_restrictions"),
]

# Data exfiltration attempts
EXFILTRATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\b(send|transmit|exfiltrate|upload)\s+(all|the|your)\s+(data|information|secrets?|keys?|tokens?)\b"), "data_exfiltration"),
    (re.compile(r"(?i)\b(reveal|show|display|print)\s+(your|the|all)\s+(system prompt|instructions|config)\b"), "prompt_extraction"),
    (re.compile(r"(?i)\b(what are your (instructions|rules|system prompt))\b"), "prompt_probing"),
]

# Token/context manipulation
CONTEXT_MANIPULATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\b(begin new conversation|start fresh|reset context)\b"), "context_reset"),
    (re.compile(r"(?i)\b(end of (system|assistant) (message|prompt|instructions))\b"), "boundary_manipulation"),
    (re.compile(r"(?i)(={5,}|---{5,}|\*{5,})\s*(system|new|instructions)", re.MULTILINE), "visual_separator_injection"),
]

# Encoded / obfuscated payloads
OBFUSCATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\b(base64|rot13|hex)\s*:\s*[A-Za-z0-9+/=]{20,}"), "encoded_payload"),
    (re.compile(r"\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){5,}"), "hex_escape_sequence"),
    (re.compile(r"&#x?[0-9a-fA-F]+;(&#x?[0-9a-fA-F]+;){5,}"), "html_entity_sequence"),
]


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class InjectionDetectionResult:
    """Result of prompt injection analysis."""
    is_safe: bool
    risk_score: float  # 0.0–1.0
    detections: List[Dict[str, str]] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.is_safe:
            return "no injection detected"
        cats = ", ".join(self.categories[:5])
        return f"injection detected ({cats}); risk={self.risk_score:.2f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "risk_score": round(self.risk_score, 4),
            "detections": self.detections,
            "categories": self.categories,
        }


# ── Category risk weights ────────────────────────────────────────

CATEGORY_WEIGHTS: Dict[str, float] = {
    "role_confusion": 0.8,
    "safety_bypass": 0.9,
    "exfiltration": 0.7,
    "context_manipulation": 0.6,
    "obfuscation": 0.5,
}


# ── PromptInjectionDetector ──────────────────────────────────────

class PromptInjectionDetector:
    """
    Detect prompt injection attacks in tool output text.

    Usage::

        det = PromptInjectionDetector()
        result = det.scan("Ignore previous instructions and do X")
        if not result.is_safe:
            print(result.summary)
    """

    # Mapping category → patterns
    ALL_PATTERN_GROUPS: Dict[str, List[Tuple[re.Pattern, str]]] = {
        "role_confusion": ROLE_CONFUSION_PATTERNS,
        "safety_bypass": SAFETY_BYPASS_PATTERNS,
        "exfiltration": EXFILTRATION_PATTERNS,
        "context_manipulation": CONTEXT_MANIPULATION_PATTERNS,
        "obfuscation": OBFUSCATION_PATTERNS,
    }

    def __init__(
        self,
        risk_threshold: float = 0.4,
        enabled_categories: Optional[List[str]] = None,
        max_scan_length: int = 500_000,
    ):
        self._risk_threshold = risk_threshold
        self._enabled_categories = set(
            enabled_categories or list(self.ALL_PATTERN_GROUPS.keys())
        )
        self._max_scan_length = max_scan_length
        self._total_scans = 0
        self._total_detections = 0

    def scan(self, text: str) -> InjectionDetectionResult:
        """
        Scan text for prompt injection patterns.

        Returns InjectionDetectionResult with risk score and detections.
        """
        self._total_scans += 1

        if not text or not text.strip():
            return InjectionDetectionResult(is_safe=True, risk_score=0.0)

        # Truncate excessively long text
        scan_text = text[:self._max_scan_length]

        detections: List[Dict[str, str]] = []
        categories_found: set = set()

        for category, patterns in self.ALL_PATTERN_GROUPS.items():
            if category not in self._enabled_categories:
                continue
            for pattern, label in patterns:
                match = pattern.search(scan_text)
                if match:
                    detections.append({
                        "category": category,
                        "pattern": label,
                        "matched_text": match.group()[:100],  # Truncate match
                    })
                    categories_found.add(category)

        # Calculate risk score
        risk_score = self._calculate_risk(categories_found, len(detections))

        is_safe = risk_score < self._risk_threshold

        if not is_safe:
            self._total_detections += 1

        return InjectionDetectionResult(
            is_safe=is_safe,
            risk_score=risk_score,
            detections=detections,
            categories=sorted(categories_found),
        )

    def scan_tool_output(
        self,
        tool_name: str,
        output: Any,
    ) -> InjectionDetectionResult:
        """
        Scan tool output for prompt injection.

        Handles string outputs and dict outputs (scans all string values).
        """
        if isinstance(output, str):
            return self.scan(output)

        if isinstance(output, dict):
            combined_text_parts: List[str] = []
            for value in output.values():
                if isinstance(value, str):
                    combined_text_parts.append(value)
            return self.scan("\n".join(combined_text_parts))

        if isinstance(output, list):
            combined_text_parts = []
            for item in output:
                if isinstance(item, str):
                    combined_text_parts.append(item)
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, str):
                            combined_text_parts.append(v)
            return self.scan("\n".join(combined_text_parts))

        return InjectionDetectionResult(is_safe=True, risk_score=0.0)

    def _calculate_risk(self, categories: set, detection_count: int) -> float:
        """Calculate overall risk score from detected categories."""
        if not categories:
            return 0.0

        # Base risk from category weights
        max_weight = max(
            CATEGORY_WEIGHTS.get(cat, 0.3) for cat in categories
        )

        # Boost for multiple categories
        multi_cat_bonus = min(0.2, (len(categories) - 1) * 0.1)

        # Boost for many detections
        count_bonus = min(0.15, (detection_count - 1) * 0.03)

        return min(1.0, max_weight + multi_cat_bonus + count_bonus)

    # ── Stats ──────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return detector statistics."""
        return {
            "total_scans": self._total_scans,
            "total_detections": self._total_detections,
            "detection_rate": (
                self._total_detections / self._total_scans
                if self._total_scans > 0 else 0.0
            ),
            "risk_threshold": self._risk_threshold,
            "enabled_categories": sorted(self._enabled_categories),
        }

    def reset_stats(self) -> None:
        """Reset scan statistics."""
        self._total_scans = 0
        self._total_detections = 0
