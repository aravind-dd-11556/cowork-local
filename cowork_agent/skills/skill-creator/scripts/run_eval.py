#!/usr/bin/env python3
"""Run skill evaluations and output results as JSON."""
import sys
import json
import time
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class EvalResult:
    prompt: str
    expected_skill: str
    actual_skill: str | None
    correct: bool
    latency_ms: float
    confidence: float

@dataclass
class EvalSummary:
    total: int
    correct: int
    accuracy: float
    avg_latency_ms: float
    p95_latency_ms: float
    false_positives: int
    false_negatives: int
    results: list[dict]

def load_eval_cases(eval_file: str) -> list[dict]:
    """Load evaluation cases from a JSON file.

    Expected format:
    [
        {"prompt": "Create a Word document", "expected_skill": "docx"},
        {"prompt": "What is 2+2?", "expected_skill": null},
        ...
    ]
    """
    with open(eval_file) as f:
        return json.load(f)

def run_eval(eval_cases: list[dict], skill_registry) -> EvalSummary:
    """Run evaluation cases against a skill registry."""
    results = []
    latencies = []
    false_positives = 0
    false_negatives = 0
    correct_count = 0

    for case in eval_cases:
        prompt = case["prompt"]
        expected = case.get("expected_skill")

        start = time.monotonic()
        # Simulate skill matching - in real use, call registry.match(prompt)
        matched = skill_registry.match(prompt) if skill_registry else None
        elapsed_ms = (time.monotonic() - start) * 1000

        actual = matched.name if matched else None
        confidence = matched.confidence if matched and hasattr(matched, 'confidence') else 0.0
        is_correct = actual == expected

        if not is_correct:
            if expected is None and actual is not None:
                false_positives += 1
            elif expected is not None and actual is None:
                false_negatives += 1
        else:
            correct_count += 1

        latencies.append(elapsed_ms)
        results.append(EvalResult(
            prompt=prompt,
            expected_skill=expected or "none",
            actual_skill=actual or "none",
            correct=is_correct,
            latency_ms=round(elapsed_ms, 2),
            confidence=confidence,
        ))

    sorted_latencies = sorted(latencies)
    p95_idx = int(len(sorted_latencies) * 0.95)

    return EvalSummary(
        total=len(results),
        correct=correct_count,
        accuracy=round(correct_count / len(results) * 100, 1) if results else 0,
        avg_latency_ms=round(statistics.mean(latencies), 2) if latencies else 0,
        p95_latency_ms=round(sorted_latencies[p95_idx], 2) if sorted_latencies else 0,
        false_positives=false_positives,
        false_negatives=false_negatives,
        results=[asdict(r) for r in results],
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_eval.py <eval_cases.json> [--verbose]")
        print("\neval_cases.json format:")
        print('[{"prompt": "Create a spreadsheet", "expected_skill": "xlsx"}, ...]')
        sys.exit(1)

    eval_file = sys.argv[1]
    verbose = "--verbose" in sys.argv

    cases = load_eval_cases(eval_file)
    print(f"Loaded {len(cases)} eval cases from {eval_file}")

    # Without a real registry, just output the cases for manual review
    print(json.dumps({
        "message": "Eval cases loaded. Connect to SkillRegistry for actual evaluation.",
        "cases_count": len(cases),
        "cases": cases if verbose else cases[:5],
    }, indent=2))

if __name__ == "__main__":
    main()
