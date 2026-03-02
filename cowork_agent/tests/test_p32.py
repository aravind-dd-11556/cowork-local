"""
Sprint 32 Tests — File Management + Artifact System.

Tests:
  - RequestCoworkDirectoryTool: directory picker, callbacks, granted directories
  - AllowCoworkFileDeleteTool: delete permission, approval tracking
  - PresentFilesTool: file card generation, computer:// links, file types
  - ArtifactSystem: classification, link generation, renderable detection
  - Helper functions: file type mapping, size formatting
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from cowork_agent.tools.file_management_tools import (
    RequestCoworkDirectoryTool,
    AllowCoworkFileDeleteTool,
    PresentFilesTool,
    _get_file_type,
    _format_size,
)
from cowork_agent.core.artifact_system import (
    ArtifactSystem,
    ArtifactInfo,
    RENDERABLE_EXTENSIONS,
    REACT_AVAILABLE_LIBRARIES,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════
# TEST: RequestCoworkDirectoryTool
# ═══════════════════════════════════════════════════════════════════════

class TestRequestCoworkDirectoryTool(unittest.TestCase):

    def test_no_callback_returns_pending(self):
        tool = RequestCoworkDirectoryTool()
        result = run(tool.execute())
        self.assertTrue(result.success)
        self.assertTrue(result.metadata.get("pending"))

    def test_callback_returns_directory(self):
        tool = RequestCoworkDirectoryTool(
            on_directory_requested=lambda: "/home/user/docs"
        )
        result = run(tool.execute())
        self.assertTrue(result.success)
        self.assertEqual(result.metadata.get("directory"), "/home/user/docs")
        self.assertEqual(tool.workspace_dir, "/home/user/docs")

    def test_callback_returns_none_cancelled(self):
        tool = RequestCoworkDirectoryTool(
            on_directory_requested=lambda: None
        )
        result = run(tool.execute())
        self.assertTrue(result.success)
        self.assertTrue(result.metadata.get("cancelled"))

    def test_callback_raises_error(self):
        tool = RequestCoworkDirectoryTool(
            on_directory_requested=lambda: (_ for _ in ()).throw(RuntimeError("fail"))
        )
        result = run(tool.execute())
        self.assertFalse(result.success)

    def test_initial_workspace_in_granted(self):
        tool = RequestCoworkDirectoryTool(workspace_dir="/tmp/initial")
        self.assertIn("/tmp/initial", tool.granted_directories)

    def test_granted_directories_accumulate(self):
        dirs = ["/tmp/a", "/tmp/b", "/tmp/c"]
        idx = [0]
        def picker():
            d = dirs[idx[0]]
            idx[0] += 1
            return d
        tool = RequestCoworkDirectoryTool(on_directory_requested=picker)
        for _ in range(3):
            run(tool.execute())
        self.assertEqual(len(tool.granted_directories), 3)

    def test_workspace_dir_setter(self):
        tool = RequestCoworkDirectoryTool()
        tool.workspace_dir = "/new/path"
        self.assertEqual(tool.workspace_dir, "/new/path")
        self.assertIn("/new/path", tool.granted_directories)

    def test_no_duplicate_granted(self):
        tool = RequestCoworkDirectoryTool(workspace_dir="/tmp/x")
        tool.workspace_dir = "/tmp/x"  # Same path again
        self.assertEqual(tool.granted_directories.count("/tmp/x"), 1)

    def test_tool_name(self):
        self.assertEqual(RequestCoworkDirectoryTool().name, "request_cowork_directory")

    def test_schema(self):
        tool = RequestCoworkDirectoryTool()
        self.assertEqual(tool.input_schema["required"], [])


# ═══════════════════════════════════════════════════════════════════════
# TEST: AllowCoworkFileDeleteTool
# ═══════════════════════════════════════════════════════════════════════

class TestAllowCoworkFileDeleteTool(unittest.TestCase):

    def test_no_callback_returns_pending(self):
        tool = AllowCoworkFileDeleteTool()
        result = run(tool.execute(file_path="/tmp/test.txt"))
        self.assertTrue(result.success)
        self.assertTrue(result.metadata.get("pending"))

    def test_callback_approved(self):
        tool = AllowCoworkFileDeleteTool(on_delete_requested=lambda p: True)
        result = run(tool.execute(file_path="/tmp/test.txt"))
        self.assertTrue(result.success)
        self.assertTrue(result.metadata.get("approved"))

    def test_callback_denied(self):
        tool = AllowCoworkFileDeleteTool(on_delete_requested=lambda p: False)
        result = run(tool.execute(file_path="/tmp/test.txt"))
        self.assertTrue(result.success)
        self.assertFalse(result.metadata.get("approved"))

    def test_empty_path_error(self):
        tool = AllowCoworkFileDeleteTool()
        result = run(tool.execute(file_path=""))
        self.assertFalse(result.success)

    def test_approved_paths_tracked(self):
        tool = AllowCoworkFileDeleteTool(on_delete_requested=lambda p: True)
        run(tool.execute(file_path="/tmp/dir/file.txt"))
        self.assertTrue(tool.is_delete_approved("/tmp/dir/file.txt"))

    def test_already_approved_returns_immediately(self):
        tool = AllowCoworkFileDeleteTool(on_delete_requested=lambda p: True)
        run(tool.execute(file_path="/tmp/dir/file.txt"))
        result = run(tool.execute(file_path="/tmp/dir/other.txt"))
        self.assertTrue(result.success)
        self.assertTrue(result.metadata.get("already_approved"))

    def test_path_normalization(self):
        tool = AllowCoworkFileDeleteTool(on_delete_requested=lambda p: True)
        run(tool.execute(file_path="/tmp/dir/../dir/file.txt"))
        self.assertTrue(tool.is_delete_approved("/tmp/dir/file.txt"))

    def test_tool_name(self):
        self.assertEqual(AllowCoworkFileDeleteTool().name, "allow_cowork_file_delete")

    def test_schema_requires_file_path(self):
        tool = AllowCoworkFileDeleteTool()
        self.assertIn("file_path", tool.input_schema["required"])


# ═══════════════════════════════════════════════════════════════════════
# TEST: PresentFilesTool
# ═══════════════════════════════════════════════════════════════════════

class TestPresentFilesTool(unittest.TestCase):

    def setUp(self):
        self.tool = PresentFilesTool(workspace_dir="/tmp")
        # Create a real temp file for testing
        self.tmpfile = tempfile.NamedTemporaryFile(
            suffix=".md", delete=False, dir="/tmp"
        )
        self.tmpfile.write(b"# Test\nHello world")
        self.tmpfile.close()

    def tearDown(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def test_present_existing_file(self):
        result = run(self.tool.execute(files=[{"file_path": self.tmpfile.name}]))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["presented_count"], 1)
        self.assertIn("computer://", result.output)

    def test_present_nonexistent_file(self):
        result = run(self.tool.execute(
            files=[{"file_path": "/tmp/nonexistent_file_abc123.txt"}]
        ))
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)

    def test_present_empty_list_error(self):
        result = run(self.tool.execute(files=[]))
        self.assertFalse(result.success)

    def test_present_none_error(self):
        result = run(self.tool.execute(files=None))
        self.assertFalse(result.success)

    def test_present_invalid_entry(self):
        result = run(self.tool.execute(files=[{"bad_key": "value"}]))
        self.assertFalse(result.success)

    def test_present_relative_path_error(self):
        result = run(self.tool.execute(files=[{"file_path": "relative/path.txt"}]))
        self.assertFalse(result.success)

    def test_present_multiple_files(self):
        tmpfile2 = tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, dir="/tmp"
        )
        tmpfile2.write(b"print('hello')")
        tmpfile2.close()
        try:
            result = run(self.tool.execute(files=[
                {"file_path": self.tmpfile.name},
                {"file_path": tmpfile2.name},
            ]))
            self.assertTrue(result.success)
            self.assertEqual(result.metadata["presented_count"], 2)
        finally:
            os.unlink(tmpfile2.name)

    def test_present_mixed_valid_invalid(self):
        result = run(self.tool.execute(files=[
            {"file_path": self.tmpfile.name},
            {"file_path": "/tmp/nonexistent_abc.txt"},
        ]))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["presented_count"], 1)
        self.assertGreater(len(result.metadata["errors"]), 0)

    def test_file_type_detection(self):
        result = run(self.tool.execute(files=[{"file_path": self.tmpfile.name}]))
        self.assertIn("Markdown", result.output)

    def test_computer_link_format(self):
        result = run(self.tool.execute(files=[{"file_path": self.tmpfile.name}]))
        self.assertIn(f"computer://{self.tmpfile.name}", result.output)

    def test_tool_name(self):
        self.assertEqual(PresentFilesTool().name, "present_files")


# ═══════════════════════════════════════════════════════════════════════
# TEST: ArtifactSystem
# ═══════════════════════════════════════════════════════════════════════

class TestArtifactSystem(unittest.TestCase):

    def setUp(self):
        self.system = ArtifactSystem(
            workspace_dir="/tmp/workspace",
            session_dir="/tmp/session",
        )

    def test_classify_markdown(self):
        info = self.system.classify("/tmp/readme.md")
        self.assertEqual(info.artifact_type, "markdown")
        self.assertTrue(info.is_renderable)

    def test_classify_html(self):
        info = self.system.classify("/tmp/index.html")
        self.assertEqual(info.artifact_type, "html")
        self.assertTrue(info.is_renderable)

    def test_classify_react(self):
        info = self.system.classify("/tmp/App.jsx")
        self.assertEqual(info.artifact_type, "react")
        self.assertTrue(info.is_renderable)

    def test_classify_mermaid(self):
        info = self.system.classify("/tmp/diagram.mermaid")
        self.assertEqual(info.artifact_type, "mermaid")
        self.assertTrue(info.is_renderable)

    def test_classify_svg(self):
        info = self.system.classify("/tmp/icon.svg")
        self.assertEqual(info.artifact_type, "svg")
        self.assertTrue(info.is_renderable)

    def test_classify_pdf(self):
        info = self.system.classify("/tmp/doc.pdf")
        self.assertEqual(info.artifact_type, "pdf")
        self.assertTrue(info.is_renderable)

    def test_classify_python_as_code(self):
        info = self.system.classify("/tmp/script.py")
        self.assertEqual(info.artifact_type, "code")
        self.assertFalse(info.is_renderable)

    def test_classify_unknown_as_file(self):
        info = self.system.classify("/tmp/data.xyz")
        self.assertEqual(info.artifact_type, "file")
        self.assertFalse(info.is_renderable)

    def test_generate_link(self):
        link = self.system.generate_link("/tmp/file.txt")
        self.assertEqual(link, "computer:///tmp/file.txt")

    def test_generate_link_relative(self):
        link = self.system.generate_link("file.txt")
        self.assertTrue(link.startswith("computer://"))
        self.assertIn("file.txt", link)

    def test_get_share_link(self):
        link = self.system.get_share_link("/tmp/report.md")
        self.assertEqual(link, "[View report.md](computer:///tmp/report.md)")

    def test_is_renderable(self):
        self.assertTrue(self.system.is_renderable("test.md"))
        self.assertTrue(self.system.is_renderable("test.html"))
        self.assertTrue(self.system.is_renderable("test.jsx"))
        self.assertFalse(self.system.is_renderable("test.py"))
        self.assertFalse(self.system.is_renderable("test.txt"))

    def test_get_artifact_type(self):
        self.assertEqual(self.system.get_artifact_type("test.md"), "markdown")
        self.assertEqual(self.system.get_artifact_type("test.svg"), "svg")
        self.assertEqual(self.system.get_artifact_type("test.xyz"), "file")

    def test_suggest_output_path_workspace(self):
        path = self.system.suggest_output_path("report.md")
        self.assertEqual(path, "/tmp/workspace/report.md")

    def test_suggest_output_path_session_fallback(self):
        system = ArtifactSystem(session_dir="/tmp/session")
        path = system.suggest_output_path("report.md")
        self.assertEqual(path, "/tmp/session/report.md")

    def test_suggest_output_path_no_dir(self):
        system = ArtifactSystem()
        path = system.suggest_output_path("report.md")
        self.assertEqual(path, "report.md")

    def test_artifacts_tracking(self):
        self.system.classify("/tmp/a.md")
        self.system.classify("/tmp/b.py")
        self.assertEqual(len(self.system), 2)
        self.assertEqual(len(self.system.artifacts), 2)

    def test_renderable_artifacts_filter(self):
        self.system.classify("/tmp/a.md")
        self.system.classify("/tmp/b.py")
        self.system.classify("/tmp/c.html")
        renderables = self.system.renderable_artifacts
        self.assertEqual(len(renderables), 2)

    def test_clear(self):
        self.system.classify("/tmp/a.md")
        self.system.clear()
        self.assertEqual(len(self.system), 0)

    def test_workspace_dir_property(self):
        self.system.workspace_dir = "/new/workspace"
        self.assertEqual(self.system.workspace_dir, "/new/workspace")

    def test_static_renderable_extensions(self):
        exts = ArtifactSystem.get_renderable_extensions()
        self.assertIn(".md", exts)
        self.assertIn(".jsx", exts)

    def test_static_react_libraries(self):
        libs = ArtifactSystem.get_react_libraries()
        self.assertIn("recharts", libs)
        self.assertIn("d3", libs)
        self.assertIn("three", libs)


# ═══════════════════════════════════════════════════════════════════════
# TEST: ArtifactInfo Dataclass
# ═══════════════════════════════════════════════════════════════════════

class TestArtifactInfo(unittest.TestCase):

    def test_defaults(self):
        info = ArtifactInfo(
            file_path="/tmp/test.md",
            file_name="test.md",
            artifact_type="markdown",
            extension=".md",
        )
        self.assertEqual(info.size_bytes, 0)
        self.assertEqual(info.computer_link, "")
        self.assertFalse(info.is_renderable)
        self.assertEqual(info.title, "")

    def test_full_info(self):
        info = ArtifactInfo(
            file_path="/tmp/report.html",
            file_name="report.html",
            artifact_type="html",
            extension=".html",
            size_bytes=1024,
            computer_link="computer:///tmp/report.html",
            is_renderable=True,
            title="report",
        )
        self.assertEqual(info.size_bytes, 1024)
        self.assertTrue(info.is_renderable)


# ═══════════════════════════════════════════════════════════════════════
# TEST: Helper Functions
# ═══════════════════════════════════════════════════════════════════════

class TestHelperFunctions(unittest.TestCase):

    def test_get_file_type_known(self):
        self.assertEqual(_get_file_type(".md"), "Markdown")
        self.assertEqual(_get_file_type(".html"), "HTML")
        self.assertEqual(_get_file_type(".jsx"), "React")
        self.assertEqual(_get_file_type(".py"), "Python")
        self.assertEqual(_get_file_type(".pdf"), "PDF")
        self.assertEqual(_get_file_type(".docx"), "Word Document")
        self.assertEqual(_get_file_type(".pptx"), "PowerPoint")
        self.assertEqual(_get_file_type(".xlsx"), "Excel Spreadsheet")

    def test_get_file_type_unknown(self):
        self.assertEqual(_get_file_type(".xyz"), "File")

    def test_format_size_bytes(self):
        self.assertEqual(_format_size(500), "500 B")

    def test_format_size_kb(self):
        self.assertEqual(_format_size(2048), "2.0 KB")

    def test_format_size_mb(self):
        self.assertEqual(_format_size(5 * 1024 * 1024), "5.0 MB")

    def test_format_size_gb(self):
        self.assertEqual(_format_size(2 * 1024 * 1024 * 1024), "2.0 GB")

    def test_format_size_zero(self):
        self.assertEqual(_format_size(0), "0 B")


# ═══════════════════════════════════════════════════════════════════════
# TEST: Renderable Extensions
# ═══════════════════════════════════════════════════════════════════════

class TestRenderableExtensions(unittest.TestCase):

    def test_all_expected_extensions(self):
        expected = {".md", ".html", ".htm", ".jsx", ".tsx", ".mermaid", ".svg", ".pdf"}
        self.assertEqual(set(RENDERABLE_EXTENSIONS.keys()), expected)

    def test_types_are_correct(self):
        self.assertEqual(RENDERABLE_EXTENSIONS[".md"], "markdown")
        self.assertEqual(RENDERABLE_EXTENSIONS[".html"], "html")
        self.assertEqual(RENDERABLE_EXTENSIONS[".jsx"], "react")
        self.assertEqual(RENDERABLE_EXTENSIONS[".mermaid"], "mermaid")
        self.assertEqual(RENDERABLE_EXTENSIONS[".svg"], "svg")
        self.assertEqual(RENDERABLE_EXTENSIONS[".pdf"], "pdf")


# ═══════════════════════════════════════════════════════════════════════
# TEST: React Available Libraries
# ═══════════════════════════════════════════════════════════════════════

class TestReactLibraries(unittest.TestCase):

    def test_common_libraries_present(self):
        expected = {"recharts", "d3", "three", "lodash", "chart.js", "tone"}
        self.assertTrue(expected.issubset(set(REACT_AVAILABLE_LIBRARIES.keys())))

    def test_three_js_version(self):
        self.assertEqual(REACT_AVAILABLE_LIBRARIES["three"], "r128")

    def test_lucide_react_version(self):
        self.assertEqual(REACT_AVAILABLE_LIBRARIES["lucide-react"], "0.263.1")


# ═══════════════════════════════════════════════════════════════════════
# TEST: Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_classify_file_with_real_file(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(b"<html><body>hello</body></html>")
            fname = f.name
        try:
            system = ArtifactSystem()
            info = system.classify(fname)
            self.assertTrue(info.is_renderable)
            self.assertGreater(info.size_bytes, 0)
        finally:
            os.unlink(fname)

    def test_classify_nonexistent_file(self):
        system = ArtifactSystem()
        info = system.classify("/tmp/nonexistent_abc123.md")
        self.assertTrue(info.is_renderable)  # Type based on extension
        self.assertEqual(info.size_bytes, 0)  # File doesn't exist

    def test_present_files_not_a_list(self):
        tool = PresentFilesTool()
        result = run(tool.execute(files="not a list"))
        self.assertFalse(result.success)

    def test_present_files_string_entry(self):
        tool = PresentFilesTool()
        result = run(tool.execute(files=["not a dict"]))
        self.assertFalse(result.success)

    def test_artifact_system_classify_htm(self):
        system = ArtifactSystem()
        info = system.classify("/tmp/page.htm")
        self.assertEqual(info.artifact_type, "html")
        self.assertTrue(info.is_renderable)

    def test_artifact_system_classify_tsx(self):
        system = ArtifactSystem()
        info = system.classify("/tmp/component.tsx")
        self.assertEqual(info.artifact_type, "react")

    def test_artifact_system_classify_code_extensions(self):
        system = ArtifactSystem()
        for ext in [".py", ".js", ".ts", ".java", ".go", ".rs", ".rb"]:
            info = system.classify(f"/tmp/file{ext}")
            self.assertEqual(info.artifact_type, "code", f"Failed for {ext}")

    def test_delete_tool_callback_error(self):
        def bad_callback(p):
            raise RuntimeError("Permission system crashed")
        tool = AllowCoworkFileDeleteTool(on_delete_requested=bad_callback)
        result = run(tool.execute(file_path="/tmp/file.txt"))
        self.assertFalse(result.success)

    def test_share_link_special_chars(self):
        system = ArtifactSystem()
        link = system.get_share_link("/tmp/my file (1).md")
        self.assertIn("my file (1).md", link)
        self.assertIn("computer://", link)

    def test_artifact_title_from_filename(self):
        system = ArtifactSystem()
        info = system.classify("/tmp/my-report.pdf")
        self.assertEqual(info.title, "my-report")


if __name__ == "__main__":
    unittest.main()
