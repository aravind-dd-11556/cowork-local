"""
Test Coverage Collector — parse pytest results and generate coverage reports.

Ingests pytest JSON reports, tracks module-level coverage, identifies
uncovered modules, and exports reports in multiple formats.

Sprint 16 (Testing & Observability Hardening) Module 5.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Enums & data classes ─────────────────────────────────────────

class CheckStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class CheckResult:
    """Result of a single test."""
    test_id: str
    test_path: str = ""
    class_name: str = ""
    method_name: str = ""
    status: CheckStatus = CheckStatus.PASSED
    duration_ms: float = 0.0
    error_message: str = ""
    covered_modules: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_path": self.test_path,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "error_message": self.error_message,
            "covered_modules": self.covered_modules,
        }


@dataclass
class CheckSuite:
    """Aggregated results from a test run."""
    name: str
    results: List[CheckResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def test_count(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == CheckStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == CheckStatus.FAILED)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == CheckStatus.SKIPPED)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == CheckStatus.ERROR)

    @property
    def duration_ms(self) -> float:
        return sum(r.duration_ms for r in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.test_count if self.test_count > 0 else 0.0

    @property
    def modules_covered(self) -> Set[str]:
        modules: Set[str] = set()
        for r in self.results:
            modules.update(r.covered_modules)
        return modules

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "test_count": self.test_count,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "duration_ms": round(self.duration_ms, 2),
            "pass_rate": round(self.pass_rate, 4),
            "modules_covered": sorted(self.modules_covered),
        }


@dataclass
class CoverageDetail:
    """Coverage detail for a single module."""
    module_name: str
    test_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    test_ids: List[str] = field(default_factory=list)

    @property
    def coverage_score(self) -> float:
        """Simple coverage score: ratio of passed tests to total."""
        return self.pass_count / self.test_count if self.test_count > 0 else 0.0


# ── CoverageCollector ───────────────────────────────────────

class CoverageCollector:
    """
    Collects and analyzes test results for coverage reporting.

    Usage::

        collector = CoverageCollector()
        suite = collector.collect_from_pytest_json(report_data)
        summary = collector.generate_summary()
        uncovered = collector.identify_uncovered_modules(all_modules, threshold=0.8)
    """

    def __init__(self):
        self._suites: List[CheckSuite] = []
        self._known_modules: Set[str] = set()

    # ── Collection ────────────────────────────────────────────

    def collect_from_pytest_json(self, report: Dict[str, Any]) -> CheckSuite:
        """
        Parse a pytest JSON report into a CheckSuite.

        Expected format (from pytest-json-report):
        {
            "tests": [
                {
                    "nodeid": "tests/test_foo.py::TestBar::test_baz",
                    "outcome": "passed",
                    "duration": 0.042,
                    ...
                }
            ],
            "summary": {...}
        }
        """
        tests = report.get("tests", [])
        results: List[CheckResult] = []

        for test in tests:
            nodeid = test.get("nodeid", "")
            parts = nodeid.split("::")

            result = CheckResult(
                test_id=nodeid,
                test_path=parts[0] if parts else "",
                class_name=parts[1] if len(parts) > 1 else "",
                method_name=parts[-1] if len(parts) > 2 else (parts[1] if len(parts) > 1 else ""),
                status=self._parse_status(test.get("outcome", "passed")),
                duration_ms=test.get("duration", 0.0) * 1000,
                error_message=self._extract_error(test),
                covered_modules=self._infer_modules(parts[0] if parts else ""),
            )
            results.append(result)

        suite = CheckSuite(
            name=report.get("root", "test_run"),
            results=results,
        )
        self._suites.append(suite)
        return suite

    def collect_results(self, results: List[CheckResult], name: str = "manual") -> CheckSuite:
        """Collect a list of CheckResult objects directly."""
        suite = CheckSuite(name=name, results=results)
        self._suites.append(suite)
        return suite

    def register_known_modules(self, modules: List[str]) -> None:
        """Register modules that should be tracked for coverage."""
        self._known_modules.update(modules)

    # ── Analysis ──────────────────────────────────────────────

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a coverage summary across all collected suites."""
        if not self._suites:
            return {"total_suites": 0, "total_tests": 0}

        all_results: List[CheckResult] = []
        for suite in self._suites:
            all_results.extend(suite.results)

        total = len(all_results)
        passed = sum(1 for r in all_results if r.status == CheckStatus.PASSED)
        failed = sum(1 for r in all_results if r.status == CheckStatus.FAILED)
        skipped = sum(1 for r in all_results if r.status == CheckStatus.SKIPPED)
        errors = sum(1 for r in all_results if r.status == CheckStatus.ERROR)

        return {
            "total_suites": len(self._suites),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
            "total_duration_ms": round(sum(r.duration_ms for r in all_results), 2),
            "modules_covered": sorted(set().union(*(s.modules_covered for s in self._suites))),
        }

    def get_coverage_by_module(self, module_name: str) -> CoverageDetail:
        """Get test coverage detail for a specific module."""
        detail = CoverageDetail(module_name=module_name)

        for suite in self._suites:
            for result in suite.results:
                if module_name in result.covered_modules:
                    detail.test_count += 1
                    detail.test_ids.append(result.test_id)
                    if result.status == CheckStatus.PASSED:
                        detail.pass_count += 1
                    elif result.status in (CheckStatus.FAILED, CheckStatus.ERROR):
                        detail.fail_count += 1

        return detail

    def identify_uncovered_modules(
        self,
        all_modules: List[str],
        threshold: float = 0.8,
    ) -> List[str]:
        """
        Find modules with coverage below the threshold.

        Parameters
        ----------
        all_modules : list
            All known module names.
        threshold : float
            Minimum coverage score (0.0–1.0).
        """
        uncovered: List[str] = []
        for mod in all_modules:
            detail = self.get_coverage_by_module(mod)
            if detail.coverage_score < threshold:
                uncovered.append(mod)
        return uncovered

    def get_slowest_tests(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest tests across all suites."""
        all_results: List[CheckResult] = []
        for suite in self._suites:
            all_results.extend(suite.results)

        sorted_results = sorted(all_results, key=lambda r: r.duration_ms, reverse=True)
        return [r.to_dict() for r in sorted_results[:top_n]]

    def get_failed_tests(self) -> List[Dict[str, Any]]:
        """Get all failed/errored tests."""
        failed: List[CheckResult] = []
        for suite in self._suites:
            for r in suite.results:
                if r.status in (CheckStatus.FAILED, CheckStatus.ERROR):
                    failed.append(r)
        return [r.to_dict() for r in failed]

    def compare_suites(
        self, suite_a: CheckSuite, suite_b: CheckSuite,
    ) -> Dict[str, Any]:
        """Compare two test suites to detect changes."""
        return {
            "suite_a": suite_a.name,
            "suite_b": suite_b.name,
            "test_count_delta": suite_b.test_count - suite_a.test_count,
            "pass_rate_delta": round(suite_b.pass_rate - suite_a.pass_rate, 4),
            "duration_delta_ms": round(suite_b.duration_ms - suite_a.duration_ms, 2),
            "new_modules": sorted(suite_b.modules_covered - suite_a.modules_covered),
            "removed_modules": sorted(suite_a.modules_covered - suite_b.modules_covered),
        }

    # ── Export ─────────────────────────────────────────────────

    def export_report(self, format: str = "json") -> str:
        """Export coverage report.  Formats: json, markdown."""
        if format == "markdown":
            return self._export_markdown()
        return self._export_json()

    def _export_json(self) -> str:
        data = self.generate_summary()
        data["suites"] = [s.to_dict() for s in self._suites]
        data["failed_tests"] = self.get_failed_tests()
        data["slowest_tests"] = self.get_slowest_tests(10)
        return json.dumps(data, indent=2, default=str)

    def _export_markdown(self) -> str:
        summary = self.generate_summary()
        lines = [
            "# Test Coverage Report",
            "",
            f"**Total tests:** {summary['total_tests']}",
            f"**Passed:** {summary['passed']}",
            f"**Failed:** {summary['failed']}",
            f"**Pass rate:** {summary['pass_rate']*100:.1f}%",
            f"**Duration:** {summary['total_duration_ms']:.0f}ms",
            "",
        ]

        failed = self.get_failed_tests()
        if failed:
            lines.append("## Failed Tests")
            lines.append("")
            for f in failed:
                lines.append(f"- `{f['test_id']}`: {f['error_message'][:100]}")
            lines.append("")

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all collected data."""
        self._suites.clear()

    # ── Internal ──────────────────────────────────────────────

    @staticmethod
    def _parse_status(outcome: str) -> CheckStatus:
        mapping = {
            "passed": CheckStatus.PASSED,
            "failed": CheckStatus.FAILED,
            "skipped": CheckStatus.SKIPPED,
            "error": CheckStatus.ERROR,
        }
        return mapping.get(outcome.lower(), CheckStatus.ERROR)

    @staticmethod
    def _extract_error(test: Dict[str, Any]) -> str:
        """Extract error message from pytest test record."""
        # Try call.longrepr first, then setup.longrepr
        for phase in ("call", "setup", "teardown"):
            phase_data = test.get(phase, {})
            if isinstance(phase_data, dict):
                longrepr = phase_data.get("longrepr", "")
                if longrepr:
                    return str(longrepr)[:500]
        return ""

    @staticmethod
    def _infer_modules(test_path: str) -> List[str]:
        """
        Infer which core modules a test file covers based on naming convention.

        e.g. tests/test_p15.py → infers from test class/method names
             tests/test_agent.py → ["agent"]
        """
        if not test_path:
            return []
        # Strip directory and extension
        name = test_path.rsplit("/", 1)[-1]
        name = name.replace("test_", "").replace(".py", "")

        # Sprint test files (test_p1.py, test_p15.py) cover multiple modules
        if name.startswith("p") and name[1:].isdigit():
            return [f"sprint_{name}"]
        return [name]
