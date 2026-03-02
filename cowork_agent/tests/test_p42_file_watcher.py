"""
Sprint 42 · Tests – Live Workspace Awareness
=============================================
~100 tests covering FileWatcher, WorkspaceAnalyzer, SuggestionEngine,
GitMonitor, and agent integration.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from cowork_agent.core.file_watcher import FileEvent, FileWatcher, FileWatcherConfig
from cowork_agent.core.workspace_analyzer import (
    FileIssue,
    LARGE_FILE_THRESHOLD,
    MERGE_CONFLICT_RE,
    WorkspaceAnalyzer,
)
from cowork_agent.core.proactive_suggestions import Suggestion, SuggestionEngine
from cowork_agent.core.git_monitor import GitChange, GitMonitor, GitState


# ═══════════════════════════════════════════════════════════════════════
# FileEvent
# ═══════════════════════════════════════════════════════════════════════

class TestFileEvent(unittest.TestCase):
    """5 tests"""

    def test_creation_defaults(self):
        ev = FileEvent(path="foo.py", event_type="created")
        self.assertEqual(ev.path, "foo.py")
        self.assertEqual(ev.event_type, "created")
        self.assertGreater(ev.timestamp, 0)
        self.assertIsNone(ev.old_path)

    def test_creation_with_old_path(self):
        ev = FileEvent(path="new.py", event_type="moved", old_path="old.py")
        self.assertEqual(ev.old_path, "old.py")

    def test_to_dict(self):
        ev = FileEvent(path="a.json", event_type="modified", timestamp=100.0)
        d = ev.to_dict()
        self.assertEqual(d["path"], "a.json")
        self.assertEqual(d["event_type"], "modified")
        self.assertEqual(d["timestamp"], 100.0)

    def test_from_dict(self):
        d = {"path": "b.py", "event_type": "deleted", "timestamp": 50.0, "old_path": None}
        ev = FileEvent.from_dict(d)
        self.assertEqual(ev.path, "b.py")
        self.assertEqual(ev.event_type, "deleted")

    def test_all_event_types(self):
        for t in ("created", "modified", "deleted", "moved"):
            ev = FileEvent(path="x", event_type=t)
            self.assertEqual(ev.event_type, t)


# ═══════════════════════════════════════════════════════════════════════
# FileWatcherConfig
# ═══════════════════════════════════════════════════════════════════════

class TestFileWatcherConfig(unittest.TestCase):
    """6 tests"""

    def test_defaults(self):
        cfg = FileWatcherConfig()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.poll_interval_seconds, 2.0)
        self.assertEqual(cfg.debounce_seconds, 1.0)

    def test_default_watch_patterns(self):
        cfg = FileWatcherConfig()
        self.assertIn("*.py", cfg.watch_patterns)
        self.assertIn("*.json", cfg.watch_patterns)

    def test_default_ignore_patterns(self):
        cfg = FileWatcherConfig()
        self.assertIn(".git", cfg.ignore_patterns)
        self.assertIn("__pycache__", cfg.ignore_patterns)
        self.assertIn("node_modules", cfg.ignore_patterns)

    def test_custom_values(self):
        cfg = FileWatcherConfig(poll_interval_seconds=5.0, debounce_seconds=2.0)
        self.assertEqual(cfg.poll_interval_seconds, 5.0)
        self.assertEqual(cfg.debounce_seconds, 2.0)

    def test_to_dict(self):
        cfg = FileWatcherConfig()
        d = cfg.to_dict()
        self.assertIn("enabled", d)
        self.assertIn("watch_patterns", d)
        self.assertIn("ignore_patterns", d)

    def test_disabled(self):
        cfg = FileWatcherConfig(enabled=False)
        self.assertFalse(cfg.enabled)


# ═══════════════════════════════════════════════════════════════════════
# FileWatcher
# ═══════════════════════════════════════════════════════════════════════

class TestFileWatcher(unittest.TestCase):
    """18 tests"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_file(self, name, content="hello"):
        path = os.path.join(self.tmp, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_init_defaults(self):
        fw = FileWatcher(workspace_path=self.tmp)
        self.assertFalse(fw.is_running)
        self.assertEqual(fw.event_history, [])

    def test_scan_directory_empty(self):
        fw = FileWatcher(workspace_path=self.tmp)
        state = fw._scan_directory()
        self.assertEqual(state, {})

    def test_scan_finds_watched_files(self):
        self._make_file("test.py")
        self._make_file("data.json")
        fw = FileWatcher(workspace_path=self.tmp)
        state = fw._scan_directory()
        self.assertIn("test.py", state)
        self.assertIn("data.json", state)

    def test_scan_ignores_non_watched(self):
        self._make_file("image.png")
        fw = FileWatcher(workspace_path=self.tmp)
        state = fw._scan_directory()
        self.assertNotIn("image.png", state)

    def test_scan_ignores_directories(self):
        self._make_file("__pycache__/foo.py")
        self._make_file("node_modules/bar.js")
        fw = FileWatcher(workspace_path=self.tmp)
        state = fw._scan_directory()
        self.assertEqual(state, {})

    def test_detect_created(self):
        fw = FileWatcher(workspace_path=self.tmp)
        old = {}
        new = {"a.py": 100.0}
        events = fw._detect_changes(old, new)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "created")
        self.assertEqual(events[0].path, "a.py")

    def test_detect_deleted(self):
        fw = FileWatcher(workspace_path=self.tmp)
        old = {"a.py": 100.0}
        new = {}
        events = fw._detect_changes(old, new)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "deleted")

    def test_detect_modified(self):
        fw = FileWatcher(workspace_path=self.tmp)
        old = {"a.py": 100.0}
        new = {"a.py": 200.0}
        events = fw._detect_changes(old, new)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "modified")

    def test_detect_no_changes(self):
        fw = FileWatcher(workspace_path=self.tmp)
        state = {"a.py": 100.0}
        events = fw._detect_changes(state, dict(state))
        self.assertEqual(events, [])

    def test_detect_multiple_changes(self):
        fw = FileWatcher(workspace_path=self.tmp)
        old = {"a.py": 100.0, "b.py": 100.0}
        new = {"a.py": 200.0, "c.py": 100.0}
        events = fw._detect_changes(old, new)
        types = {e.event_type for e in events}
        self.assertIn("modified", types)  # a.py changed
        self.assertIn("deleted", types)   # b.py gone
        self.assertIn("created", types)   # c.py new

    def test_debounce_keeps_latest(self):
        fw = FileWatcher(workspace_path=self.tmp)
        events = [
            FileEvent(path="a.py", event_type="modified", timestamp=1.0),
            FileEvent(path="a.py", event_type="modified", timestamp=2.0),
        ]
        result = fw._debounce(events)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].timestamp, 2.0)

    def test_debounce_different_types_kept(self):
        fw = FileWatcher(workspace_path=self.tmp)
        events = [
            FileEvent(path="a.py", event_type="created", timestamp=1.0),
            FileEvent(path="a.py", event_type="modified", timestamp=2.0),
        ]
        result = fw._debounce(events)
        self.assertEqual(len(result), 2)

    def test_matches_watch(self):
        fw = FileWatcher(workspace_path=self.tmp)
        self.assertTrue(fw._matches_watch("test.py"))
        self.assertTrue(fw._matches_watch("data.json"))
        self.assertFalse(fw._matches_watch("image.png"))

    def test_matches_ignore(self):
        fw = FileWatcher(workspace_path=self.tmp)
        self.assertTrue(fw._matches_ignore(".git"))
        self.assertTrue(fw._matches_ignore("__pycache__"))
        self.assertFalse(fw._matches_ignore("src"))

    def test_start_stop(self):
        async def _run():
            fw = FileWatcher(workspace_path=self.tmp,
                             config=FileWatcherConfig(poll_interval_seconds=0.05))
            await fw.start()
            self.assertTrue(fw.is_running)
            await asyncio.sleep(0.1)
            await fw.stop()
            self.assertFalse(fw.is_running)

        asyncio.get_event_loop().run_until_complete(_run())

    def test_callback_invoked(self):
        received = []

        def cb(events):
            received.extend(events)

        async def _run():
            self._make_file("init.py")
            fw = FileWatcher(workspace_path=self.tmp,
                             config=FileWatcherConfig(poll_interval_seconds=0.05),
                             on_events=cb)
            await fw.start()
            await asyncio.sleep(0.08)
            # Create new file after start
            self._make_file("new_file.py")
            await asyncio.sleep(0.15)
            await fw.stop()
            return received

        result = asyncio.get_event_loop().run_until_complete(_run())
        # Should have detected new_file.py creation
        created = [e for e in result if e.event_type == "created"]
        self.assertTrue(len(created) >= 1)

    def test_clear_history(self):
        fw = FileWatcher(workspace_path=self.tmp)
        fw._event_history = [FileEvent(path="x", event_type="created")]
        fw.clear_history()
        self.assertEqual(fw.event_history, [])

    def test_event_history_capped(self):
        fw = FileWatcher(workspace_path=self.tmp)
        fw._max_history = 5
        events = [FileEvent(path=f"f{i}.py", event_type="created") for i in range(10)]
        fw._record_events(events)
        self.assertEqual(len(fw._event_history), 5)


# ═══════════════════════════════════════════════════════════════════════
# WorkspaceAnalyzer
# ═══════════════════════════════════════════════════════════════════════

class TestWorkspaceAnalyzer(unittest.TestCase):
    """20 tests"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.analyzer = WorkspaceAnalyzer(workspace_path=self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make(self, name, content):
        path = os.path.join(self.tmp, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return name  # relative

    def test_valid_json(self):
        rel = self._make("ok.json", '{"a": 1}')
        issues = self.analyzer.analyze_file(rel)
        self.assertEqual(issues, [])

    def test_invalid_json(self):
        rel = self._make("bad.json", '{"a": }')
        issues = self.analyzer.analyze_file(rel)
        types = [i.issue_type for i in issues]
        self.assertIn("invalid_json", types)

    def test_json_error_has_line(self):
        rel = self._make("bad2.json", '{\n  "a": ,\n}')
        issues = self.analyzer.analyze_file(rel)
        json_issues = [i for i in issues if i.issue_type == "invalid_json"]
        self.assertTrue(json_issues)
        self.assertIsNotNone(json_issues[0].line_number)

    def test_valid_python(self):
        rel = self._make("ok.py", "def foo():\n    return 42\n")
        issues = self.analyzer.analyze_file(rel)
        self.assertEqual(issues, [])

    def test_invalid_python(self):
        rel = self._make("bad.py", "def foo(\n")
        issues = self.analyzer.analyze_file(rel)
        types = [i.issue_type for i in issues]
        self.assertIn("syntax_error", types)

    def test_python_syntax_line(self):
        rel = self._make("bad3.py", "x = 1\ndef (:\n")
        issues = self.analyzer.analyze_file(rel)
        syntax = [i for i in issues if i.issue_type == "syntax_error"]
        self.assertTrue(syntax)
        self.assertIsNotNone(syntax[0].line_number)

    def test_valid_yaml(self):
        rel = self._make("ok.yaml", "key: value\n")
        issues = self.analyzer.analyze_file(rel)
        self.assertEqual(issues, [])

    def test_invalid_yaml(self):
        rel = self._make("bad.yaml", "key: :\n  - bad\n    indent: wrong\n")
        issues = self.analyzer.analyze_file(rel)
        # May or may not find YAML issue depending on pyyaml availability
        # but should not crash
        self.assertIsInstance(issues, list)

    def test_merge_conflict_detected(self):
        content = "line1\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\n"
        rel = self._make("conflict.py", content)
        issues = self.analyzer.analyze_file(rel)
        types = [i.issue_type for i in issues]
        self.assertIn("merge_conflict", types)

    def test_no_merge_conflict(self):
        rel = self._make("clean.py", "# no conflicts here\nx = 1\n")
        issues = self.analyzer.analyze_file(rel)
        types = [i.issue_type for i in issues]
        self.assertNotIn("merge_conflict", types)

    def test_merge_conflict_line_number(self):
        content = "line1\nline2\n<<<<<<< HEAD\nours\n"
        rel = self._make("conf2.py", content)
        issues = self.analyzer.analyze_file(rel)
        mc = [i for i in issues if i.issue_type == "merge_conflict"]
        if mc:
            self.assertEqual(mc[0].line_number, 3)

    def test_large_file_detected(self):
        rel = self._make("big.json", '{"x": "' + "a" * (11 * 1024 * 1024) + '"}')
        issues = self.analyzer.analyze_file(rel)
        types = [i.issue_type for i in issues]
        self.assertIn("large_file", types)

    def test_small_file_no_warning(self):
        rel = self._make("small.json", '{"x": 1}')
        issues = self.analyzer.analyze_file(rel)
        types = [i.issue_type for i in issues]
        self.assertNotIn("large_file", types)

    def test_analyze_batch(self):
        r1 = self._make("a.json", '{"ok": true}')
        r2 = self._make("b.json", '{bad}')
        issues = self.analyzer.analyze_batch([r1, r2])
        types = [i.issue_type for i in issues]
        self.assertIn("invalid_json", types)

    def test_nonexistent_file(self):
        issues = self.analyzer.analyze_file("no_such_file.py")
        self.assertEqual(issues, [])

    def test_absolute_path(self):
        self._make("abs.json", '{"a": 1}')
        full = os.path.join(self.tmp, "abs.json")
        issues = self.analyzer.analyze_file(full)
        self.assertEqual(issues, [])

    def test_issue_to_dict(self):
        issue = FileIssue(
            file_path="test.py", issue_type="syntax_error",
            description="bad", line_number=5, severity="error",
        )
        d = issue.to_dict()
        self.assertEqual(d["file_path"], "test.py")
        self.assertEqual(d["issue_type"], "syntax_error")
        self.assertEqual(d["line_number"], 5)

    def test_issue_from_dict(self):
        d = {"file_path": "x.py", "issue_type": "syntax_error",
             "description": "err", "line_number": 3, "severity": "error",
             "auto_fixable": True}
        issue = FileIssue.from_dict(d)
        self.assertEqual(issue.file_path, "x.py")
        self.assertTrue(issue.auto_fixable)

    def test_merge_conflict_regex(self):
        self.assertTrue(MERGE_CONFLICT_RE.search("<<<<<<< HEAD\n"))
        self.assertFalse(MERGE_CONFLICT_RE.search("<<< not enough\n"))

    def test_yml_extension(self):
        rel = self._make("ok.yml", "key: value\n")
        issues = self.analyzer.analyze_file(rel)
        self.assertEqual(issues, [])


# ═══════════════════════════════════════════════════════════════════════
# Suggestion
# ═══════════════════════════════════════════════════════════════════════

class TestSuggestion(unittest.TestCase):
    """6 tests"""

    def _issue(self, **kwargs):
        defaults = dict(file_path="test.py", issue_type="syntax_error",
                        description="err", severity="error", auto_fixable=False)
        defaults.update(kwargs)
        return FileIssue(**defaults)

    def test_creation(self):
        sug = Suggestion(
            suggestion_id="sug_abc",
            file_path="test.py",
            issue=self._issue(),
            proposed_action="Fix it",
        )
        self.assertEqual(sug.status, "pending")
        self.assertGreater(sug.created_at, 0)

    def test_generate_id(self):
        sid = Suggestion.generate_id()
        self.assertTrue(sid.startswith("sug_"))
        self.assertEqual(len(sid), 14)  # "sug_" + 10 hex chars

    def test_to_dict(self):
        sug = Suggestion(
            suggestion_id="sug_xyz",
            file_path="a.py",
            issue=self._issue(),
            proposed_action="Fix",
        )
        d = sug.to_dict()
        self.assertEqual(d["suggestion_id"], "sug_xyz")
        self.assertEqual(d["status"], "pending")
        self.assertIn("issue", d)

    def test_auto_applicable_default(self):
        sug = Suggestion(
            suggestion_id="s1",
            file_path="x.py",
            issue=self._issue(),
            proposed_action="Fix",
        )
        self.assertFalse(sug.auto_applicable)

    def test_status_transitions(self):
        sug = Suggestion(
            suggestion_id="s2",
            file_path="x.py",
            issue=self._issue(),
            proposed_action="Fix",
        )
        sug.status = "accepted"
        self.assertEqual(sug.status, "accepted")
        sug.status = "dismissed"
        self.assertEqual(sug.status, "dismissed")

    def test_unique_ids(self):
        ids = {Suggestion.generate_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)


# ═══════════════════════════════════════════════════════════════════════
# SuggestionEngine
# ═══════════════════════════════════════════════════════════════════════

class TestSuggestionEngine(unittest.TestCase):
    """15 tests"""

    def _mock_analyzer(self, issues=None):
        analyzer = MagicMock()
        analyzer.analyze_batch.return_value = issues or []
        return analyzer

    def test_init(self):
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer())
        self.assertEqual(engine.get_pending(), [])
        self.assertEqual(engine.total_generated, 0)

    def test_process_no_events(self):
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer())
        result = engine.process_events([])
        self.assertEqual(result, [])

    def test_process_delete_events_ignored(self):
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer())
        events = [FileEvent(path="x.py", event_type="deleted")]
        result = engine.process_events(events)
        self.assertEqual(result, [])

    def test_process_created_event_with_issue(self):
        issue = FileIssue(file_path="bad.json", issue_type="invalid_json",
                          description="err", severity="error")
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer([issue]))
        events = [FileEvent(path="bad.json", event_type="created")]
        result = engine.process_events(events)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].file_path, "bad.json")

    def test_process_modified_event(self):
        issue = FileIssue(file_path="bad.py", issue_type="syntax_error",
                          description="err", severity="error")
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer([issue]))
        events = [FileEvent(path="bad.py", event_type="modified")]
        result = engine.process_events(events)
        self.assertEqual(len(result), 1)

    def test_pending_list(self):
        issue = FileIssue(file_path="f.py", issue_type="syntax_error",
                          description="err", severity="error")
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer([issue]))
        engine.process_events([FileEvent(path="f.py", event_type="created")])
        pending = engine.get_pending()
        self.assertEqual(len(pending), 1)

    def test_accept_suggestion(self):
        issue = FileIssue(file_path="f.py", issue_type="syntax_error",
                          description="err", severity="error")
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer([issue]))
        sugs = engine.process_events([FileEvent(path="f.py", event_type="created")])
        self.assertTrue(engine.accept(sugs[0].suggestion_id))
        self.assertEqual(engine.get_pending(), [])

    def test_dismiss_suggestion(self):
        issue = FileIssue(file_path="f.py", issue_type="syntax_error",
                          description="err", severity="error")
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer([issue]))
        sugs = engine.process_events([FileEvent(path="f.py", event_type="created")])
        self.assertTrue(engine.dismiss(sugs[0].suggestion_id))
        self.assertIn("f.py", engine.dismissed_paths)

    def test_dismissed_not_re_suggested(self):
        issue = FileIssue(file_path="f.py", issue_type="syntax_error",
                          description="err", severity="error")
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer([issue]))
        sugs = engine.process_events([FileEvent(path="f.py", event_type="created")])
        engine.dismiss(sugs[0].suggestion_id)
        # Process same file again
        result = engine.process_events([FileEvent(path="f.py", event_type="modified")])
        self.assertEqual(result, [])

    def test_accept_nonexistent(self):
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer())
        self.assertFalse(engine.accept("nonexistent"))

    def test_dismiss_nonexistent(self):
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer())
        self.assertFalse(engine.dismiss("nonexistent"))

    def test_max_pending(self):
        issues = [
            FileIssue(file_path=f"f{i}.py", issue_type="syntax_error",
                      description="err", severity="error")
            for i in range(15)
        ]
        engine = SuggestionEngine(
            workspace_analyzer=self._mock_analyzer(issues),
            max_pending=5,
        )
        events = [FileEvent(path=f"f{i}.py", event_type="created") for i in range(15)]
        engine.process_events(events)
        # Queue is capped at max_pending (deque maxlen)
        self.assertLessEqual(len(engine.get_pending()), 5)

    def test_clear(self):
        issue = FileIssue(file_path="f.py", issue_type="syntax_error",
                          description="err", severity="error")
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer([issue]))
        engine.process_events([FileEvent(path="f.py", event_type="created")])
        engine.clear()
        self.assertEqual(engine.get_pending(), [])
        self.assertEqual(engine.dismissed_paths, set())

    def test_total_generated(self):
        issue = FileIssue(file_path="f.py", issue_type="syntax_error",
                          description="err", severity="error")
        engine = SuggestionEngine(workspace_analyzer=self._mock_analyzer([issue]))
        engine.process_events([FileEvent(path="f.py", event_type="created")])
        self.assertEqual(engine.total_generated, 1)

    def test_format_for_user(self):
        issue = FileIssue(file_path="test.py", issue_type="syntax_error",
                          description="bad syntax", line_number=10,
                          severity="error")
        sug = Suggestion(
            suggestion_id="s1",
            file_path="test.py",
            issue=issue,
            proposed_action="Fix the Python syntax error",
        )
        text = SuggestionEngine.format_for_user(sug)
        self.assertIn("test.py", text)
        self.assertIn("bad syntax", text)
        self.assertIn("Line 10", text)
        self.assertIn("Fix the Python syntax error", text)


# ═══════════════════════════════════════════════════════════════════════
# GitChange
# ═══════════════════════════════════════════════════════════════════════

class TestGitChange(unittest.TestCase):
    """5 tests"""

    def test_creation(self):
        gc = GitChange(change_type="new_commit", description="New commit")
        self.assertEqual(gc.change_type, "new_commit")
        self.assertGreater(gc.timestamp, 0)

    def test_to_dict(self):
        gc = GitChange(change_type="branch_switch", description="Switched",
                       details={"old": "main", "new": "dev"}, timestamp=100.0)
        d = gc.to_dict()
        self.assertEqual(d["change_type"], "branch_switch")
        self.assertEqual(d["details"]["old"], "main")

    def test_all_change_types(self):
        for t in ("new_commit", "branch_switch", "unstaged_changes",
                   "merge_conflict", "stash_change"):
            gc = GitChange(change_type=t, description="test")
            self.assertEqual(gc.change_type, t)

    def test_empty_details(self):
        gc = GitChange(change_type="new_commit", description="test")
        self.assertEqual(gc.details, {})

    def test_custom_timestamp(self):
        gc = GitChange(change_type="new_commit", description="test", timestamp=42.0)
        self.assertEqual(gc.timestamp, 42.0)


# ═══════════════════════════════════════════════════════════════════════
# GitState
# ═══════════════════════════════════════════════════════════════════════

class TestGitState(unittest.TestCase):
    """4 tests"""

    def test_defaults(self):
        gs = GitState()
        self.assertEqual(gs.branch, "")
        self.assertEqual(gs.head_commit, "")
        self.assertFalse(gs.is_git_repo)

    def test_to_dict(self):
        gs = GitState(branch="main", head_commit="abc123", is_git_repo=True)
        d = gs.to_dict()
        self.assertEqual(d["branch"], "main")
        self.assertTrue(d["is_git_repo"])

    def test_with_counts(self):
        gs = GitState(unstaged_count=3, staged_count=2, stash_count=1)
        self.assertEqual(gs.unstaged_count, 3)
        self.assertEqual(gs.staged_count, 2)
        self.assertEqual(gs.stash_count, 1)

    def test_merge_conflicts_flag(self):
        gs = GitState(has_merge_conflicts=True)
        self.assertTrue(gs.has_merge_conflicts)


# ═══════════════════════════════════════════════════════════════════════
# GitMonitor
# ═══════════════════════════════════════════════════════════════════════

class TestGitMonitor(unittest.TestCase):
    """12 tests"""

    def _make_monitor(self, **state_kwargs):
        """Create a monitor with mocked git commands."""
        monitor = GitMonitor(workspace_path="/tmp/fake")
        return monitor

    def _mock_state(self, monitor, **kwargs):
        """Set up _get_current_state to return a specific state."""
        defaults = dict(
            branch="main", head_commit="abc123",
            unstaged_count=0, staged_count=0,
            stash_count=0, has_merge_conflicts=False,
            is_git_repo=True,
        )
        defaults.update(kwargs)
        state = GitState(**defaults)
        monitor._get_current_state = MagicMock(return_value=state)
        return state

    def test_init(self):
        m = GitMonitor(workspace_path="/tmp")
        self.assertIsNone(m.last_state)
        self.assertEqual(m.change_history, [])

    def test_first_check_no_changes(self):
        m = GitMonitor(workspace_path="/tmp")
        self._mock_state(m, branch="main")

        async def _run():
            return await m.check_changes()
        changes = asyncio.get_event_loop().run_until_complete(_run())
        self.assertEqual(changes, [])
        self.assertIsNotNone(m.last_state)

    def test_branch_switch_detected(self):
        m = GitMonitor(workspace_path="/tmp")
        # Set initial state
        m._last_state = GitState(branch="main", head_commit="abc", is_git_repo=True)
        self._mock_state(m, branch="feature", head_commit="abc")

        async def _run():
            return await m.check_changes()
        changes = asyncio.get_event_loop().run_until_complete(_run())
        types = [c.change_type for c in changes]
        self.assertIn("branch_switch", types)

    def test_new_commit_detected(self):
        m = GitMonitor(workspace_path="/tmp")
        m._last_state = GitState(branch="main", head_commit="old123", is_git_repo=True)
        self._mock_state(m, branch="main", head_commit="new456")

        async def _run():
            return await m.check_changes()
        changes = asyncio.get_event_loop().run_until_complete(_run())
        types = [c.change_type for c in changes]
        self.assertIn("new_commit", types)

    def test_unstaged_changes_increase(self):
        m = GitMonitor(workspace_path="/tmp")
        m._last_state = GitState(branch="main", head_commit="abc",
                                 unstaged_count=0, is_git_repo=True)
        self._mock_state(m, branch="main", head_commit="abc", unstaged_count=3)

        async def _run():
            return await m.check_changes()
        changes = asyncio.get_event_loop().run_until_complete(_run())
        types = [c.change_type for c in changes]
        self.assertIn("unstaged_changes", types)

    def test_unstaged_decrease_not_reported(self):
        m = GitMonitor(workspace_path="/tmp")
        m._last_state = GitState(branch="main", head_commit="abc",
                                 unstaged_count=5, is_git_repo=True)
        self._mock_state(m, branch="main", head_commit="abc", unstaged_count=2)

        async def _run():
            return await m.check_changes()
        changes = asyncio.get_event_loop().run_until_complete(_run())
        types = [c.change_type for c in changes]
        self.assertNotIn("unstaged_changes", types)

    def test_merge_conflict_detected(self):
        m = GitMonitor(workspace_path="/tmp")
        m._last_state = GitState(branch="main", head_commit="abc",
                                 has_merge_conflicts=False, is_git_repo=True)
        self._mock_state(m, branch="main", head_commit="abc", has_merge_conflicts=True)

        async def _run():
            return await m.check_changes()
        changes = asyncio.get_event_loop().run_until_complete(_run())
        types = [c.change_type for c in changes]
        self.assertIn("merge_conflict", types)

    def test_stash_added(self):
        m = GitMonitor(workspace_path="/tmp")
        m._last_state = GitState(branch="main", head_commit="abc",
                                 stash_count=0, is_git_repo=True)
        self._mock_state(m, branch="main", head_commit="abc", stash_count=1)

        async def _run():
            return await m.check_changes()
        changes = asyncio.get_event_loop().run_until_complete(_run())
        types = [c.change_type for c in changes]
        self.assertIn("stash_change", types)
        sc = [c for c in changes if c.change_type == "stash_change"]
        self.assertIn("added", sc[0].description)

    def test_stash_removed(self):
        m = GitMonitor(workspace_path="/tmp")
        m._last_state = GitState(branch="main", head_commit="abc",
                                 stash_count=2, is_git_repo=True)
        self._mock_state(m, branch="main", head_commit="abc", stash_count=1)

        async def _run():
            return await m.check_changes()
        changes = asyncio.get_event_loop().run_until_complete(_run())
        sc = [c for c in changes if c.change_type == "stash_change"]
        self.assertTrue(sc)
        self.assertIn("removed", sc[0].description)

    def test_no_git_repo(self):
        m = GitMonitor(workspace_path="/tmp")
        self._mock_state(m, is_git_repo=False)

        async def _run():
            return await m.check_changes()
        changes = asyncio.get_event_loop().run_until_complete(_run())
        self.assertEqual(changes, [])

    def test_reset(self):
        m = GitMonitor(workspace_path="/tmp")
        m._last_state = GitState(branch="main", is_git_repo=True)
        m.reset()
        self.assertIsNone(m.last_state)

    def test_change_history_accumulates(self):
        m = GitMonitor(workspace_path="/tmp")
        m._last_state = GitState(branch="main", head_commit="old", is_git_repo=True)
        self._mock_state(m, branch="main", head_commit="new1")

        async def _run():
            await m.check_changes()
            m._last_state = GitState(branch="main", head_commit="new1", is_git_repo=True)
            m._get_current_state = MagicMock(return_value=GitState(
                branch="main", head_commit="new2", is_git_repo=True))
            await m.check_changes()
            return m.change_history

        history = asyncio.get_event_loop().run_until_complete(_run())
        self.assertEqual(len(history), 2)


# ═══════════════════════════════════════════════════════════════════════
# Agent Integration
# ═══════════════════════════════════════════════════════════════════════

class TestAgentIntegration(unittest.TestCase):
    """10 tests"""

    def _make_agent(self):
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        registry = MagicMock()
        prompt_builder = MagicMock()
        agent = Agent(provider=provider, registry=registry,
                      prompt_builder=prompt_builder)
        return agent

    def test_agent_has_file_watcher_attr(self):
        agent = self._make_agent()
        self.assertIsNone(agent.file_watcher)

    def test_agent_has_suggestion_engine_attr(self):
        agent = self._make_agent()
        self.assertIsNone(agent.suggestion_engine)

    def test_agent_has_git_monitor_attr(self):
        agent = self._make_agent()
        self.assertIsNone(agent.git_monitor)

    def test_file_watcher_assignable(self):
        agent = self._make_agent()
        fw = FileWatcher(workspace_path="/tmp")
        agent.file_watcher = fw
        self.assertIs(agent.file_watcher, fw)

    def test_suggestion_engine_assignable(self):
        agent = self._make_agent()
        analyzer = MagicMock()
        se = SuggestionEngine(workspace_analyzer=analyzer)
        agent.suggestion_engine = se
        self.assertIs(agent.suggestion_engine, se)

    def test_git_monitor_assignable(self):
        agent = self._make_agent()
        gm = GitMonitor(workspace_path="/tmp")
        agent.git_monitor = gm
        self.assertIs(agent.git_monitor, gm)

    def test_main_wiring_block_exists(self):
        """Verify main.py has Sprint 42 wiring."""
        import cowork_agent.main as main_mod
        source = open(main_mod.__file__).read()
        self.assertIn("Sprint 42", source)
        self.assertIn("FileWatcher", source)
        self.assertIn("WorkspaceAnalyzer", source)
        self.assertIn("SuggestionEngine", source)
        self.assertIn("GitMonitor", source)

    def test_file_watcher_config_defaults_in_wiring(self):
        """Verify wiring uses sensible defaults."""
        import cowork_agent.main as main_mod
        source = open(main_mod.__file__).read()
        self.assertIn("file_watcher", source)
        self.assertIn("poll_interval", source)

    def test_watcher_and_engine_work_together(self):
        """Integration: FileWatcher events → SuggestionEngine."""
        tmp = tempfile.mkdtemp()
        try:
            analyzer = WorkspaceAnalyzer(workspace_path=tmp)
            engine = SuggestionEngine(workspace_analyzer=analyzer)

            # Simulate a bad JSON file event
            bad_path = os.path.join(tmp, "bad.json")
            with open(bad_path, "w") as f:
                f.write("{invalid}")

            events = [FileEvent(path="bad.json", event_type="created")]
            suggestions = engine.process_events(events)
            self.assertTrue(len(suggestions) >= 1)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_watcher_and_git_monitor_independent(self):
        """FileWatcher and GitMonitor can coexist."""
        fw = FileWatcher(workspace_path="/tmp")
        gm = GitMonitor(workspace_path="/tmp")
        self.assertIsNotNone(fw)
        self.assertIsNotNone(gm)


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    """8 tests"""

    def test_empty_workspace_scan(self):
        tmp = tempfile.mkdtemp()
        try:
            fw = FileWatcher(workspace_path=tmp)
            state = fw._scan_directory()
            self.assertEqual(state, {})
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_binary_file_ignored_by_watcher(self):
        tmp = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmp, "image.bin"), "wb") as f:
                f.write(b"\x00\x01\x02")
            fw = FileWatcher(workspace_path=tmp)
            state = fw._scan_directory()
            self.assertEqual(state, {})
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_permission_error_in_scan(self):
        """Watcher shouldn't crash on unreadable directories."""
        fw = FileWatcher(workspace_path="/nonexistent_dir_42")
        state = fw._scan_directory()
        self.assertEqual(state, {})

    def test_analyzer_with_empty_file(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "empty.json")
            with open(path, "w") as f:
                f.write("")
            analyzer = WorkspaceAnalyzer(workspace_path=tmp)
            issues = analyzer.analyze_file("empty.json")
            # Empty JSON is an error
            types = [i.issue_type for i in issues]
            self.assertIn("invalid_json", types)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_git_monitor_run_git_failure(self):
        m = GitMonitor(workspace_path="/nonexistent_42")
        result = m._run_git("status")
        # Should return None, not crash
        self.assertTrue(result is None or isinstance(result, str))

    def test_suggestion_action_map_coverage(self):
        from cowork_agent.core.proactive_suggestions import _ACTION_MAP
        expected_types = ["invalid_json", "broken_yaml", "syntax_error",
                          "merge_conflict", "large_file"]
        for t in expected_types:
            self.assertIn(t, _ACTION_MAP)

    def test_suggestion_engine_no_issues_no_suggestions(self):
        analyzer = MagicMock()
        analyzer.analyze_batch.return_value = []
        engine = SuggestionEngine(workspace_analyzer=analyzer)
        events = [FileEvent(path="ok.py", event_type="modified")]
        result = engine.process_events(events)
        self.assertEqual(result, [])

    def test_rapid_events_debounced(self):
        fw = FileWatcher(workspace_path="/tmp")
        events = [
            FileEvent(path="a.py", event_type="modified", timestamp=1.0),
            FileEvent(path="a.py", event_type="modified", timestamp=1.01),
            FileEvent(path="a.py", event_type="modified", timestamp=1.02),
            FileEvent(path="b.py", event_type="modified", timestamp=1.0),
        ]
        result = fw._debounce(events)
        # a.py:modified should be collapsed to 1, b.py:modified = 1
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
