"""
Tool Generator — Dynamic tool creation at runtime.

Allows the agent to create new tools on-the-fly from Python code strings.
Tools are validated for safety, wrapped as BaseTool instances, registered
with the ToolRegistry, and persisted to disk for cross-session reuse.

Sprint 27: Tier 2 Differentiating Feature 3.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


# ── Dangerous imports / calls that are blocked ───────────────────

BLOCKED_IMPORTS = {
    "os.system", "subprocess", "socket", "ctypes", "shutil.rmtree",
    "multiprocessing", "threading", "signal", "sys.exit", "eval",
    "exec", "compile", "__import__", "importlib",
    "pickle", "shelve", "marshal",
}

BLOCKED_BUILTINS = {
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
}

# Allowed imports for generated tools
ALLOWED_IMPORTS = {
    "json", "re", "math", "datetime", "collections", "itertools",
    "functools", "string", "hashlib", "base64", "urllib.parse",
    "statistics", "decimal", "fractions", "textwrap", "difflib",
    "pathlib", "copy", "time", "uuid",
}

# H-12: Restricted builtin functions for sandbox execution
RESTRICTED_BUILTINS = {
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter',
    'float', 'frozenset', 'hasattr', 'int', 'isinstance', 'issubclass',
    'len', 'list', 'map', 'max', 'min', 'next', 'print', 'range',
    'repr', 'reversed', 'round', 'set', 'sorted', 'str', 'sum',
    'tuple', 'type', 'zip', 'format', 'hex', 'oct', 'bin', 'ord',
    'chr', 'slice', 'hash', 'id', 'iter', 'callable', 'divmod',
    'pow', '__build_class__',
}


# ── Dataclasses ──────────────────────────────────────────────────

@dataclass
class CodeValidationResult:
    """Result of validating generated tool code."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ── GeneratedTool ────────────────────────────────────────────────

class GeneratedTool:
    """
    A dynamically generated tool wrapping a Python code string.

    The code must define a function called `run(**kwargs)` that returns
    a string result. It's executed in a restricted namespace.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        code_string: str,
        created_at: float = 0.0,
    ):
        self.name = name
        self.description = f"[Generated] {description}"
        self.input_schema = input_schema
        self.code_string = code_string
        self.created_at = created_at or time.time()
        self.execution_count = 0
        self._compiled = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "code_string": self.code_string,
            "created_at": self.created_at,
            "execution_count": self.execution_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GeneratedTool:
        tool = cls(
            name=data["name"],
            description=data.get("description", "").replace("[Generated] ", ""),
            input_schema=data.get("input_schema", {}),
            code_string=data.get("code_string", ""),
            created_at=data.get("created_at", 0.0),
        )
        tool.execution_count = data.get("execution_count", 0)
        return tool

    async def execute(self, **kwargs) -> str:
        """
        Execute the generated code with the provided arguments.

        The code must define a `run(**kwargs)` function.
        Returns the string result or raises an exception.
        """
        self.execution_count += 1

        # Build restricted namespace
        namespace = self._build_namespace()

        try:
            # Execute the validated code in restricted namespace
            self._run_in_sandbox(self.code_string, namespace)

            # Call the run function
            run_fn = namespace.get("run")
            if not callable(run_fn):
                raise RuntimeError(
                    "Generated code must define a 'run(**kwargs)' function"
                )

            result = run_fn(**kwargs)

            # Ensure string output
            if not isinstance(result, str):
                result = str(result)

            return result

        except Exception as e:
            raise RuntimeError(f"Generated tool '{self.name}' failed: {e}") from e

    @staticmethod
    def validate_code(code: str) -> CodeValidationResult:
        """
        Validate generated code for safety and correctness.

        Checks:
          - Valid Python syntax
          - No dangerous imports or calls
          - Defines a `run` function
        """
        errors = []
        warnings = []

        # 1. Check syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return CodeValidationResult(
                valid=False,
                errors=[f"Syntax error: {e}"],
            )

        # 2. Check for dangerous patterns
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in BLOCKED_IMPORTS or any(
                        alias.name.startswith(b) for b in BLOCKED_IMPORTS
                    ):
                        errors.append(f"Blocked import: {alias.name}")
                    elif alias.name not in ALLOWED_IMPORTS:
                        warnings.append(f"Uncommon import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in BLOCKED_IMPORTS or any(
                    module.startswith(b) for b in BLOCKED_IMPORTS
                ):
                    errors.append(f"Blocked import: {module}")

            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_BUILTINS:
                        errors.append(f"Blocked builtin call: {node.func.id}")

                elif isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    if attr in {"system", "popen", "exec", "eval"}:
                        errors.append(f"Blocked method call: .{attr}()")

        # 3. Check that `run` function is defined
        has_run = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "run":
                    has_run = True
                    break

        if not has_run:
            errors.append("Code must define a 'run(**kwargs)' function")

        return CodeValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def _build_namespace() -> dict:
        """Build a restricted execution namespace.

        H-12: Uses a whitelist of safe builtins to prevent sandbox escapes.
        Dangerous builtins (eval, exec, __import__, etc.) are excluded.
        """
        import json as _json
        import re as _re
        import math as _math
        import datetime as _datetime
        import collections as _collections
        import hashlib as _hashlib
        import base64 as _base64
        import textwrap as _textwrap
        import copy as _copy
        import time as _time

        import builtins
        # H-12: Build restricted builtins from whitelist
        restricted_builtins = {}
        for name in RESTRICTED_BUILTINS:
            if hasattr(builtins, name):
                restricted_builtins[name] = getattr(builtins, name)

        return {
            "__builtins__": restricted_builtins,
            "json": _json,
            "re": _re,
            "math": _math,
            "datetime": _datetime,
            "collections": _collections,
            "hashlib": _hashlib,
            "base64": _base64,
            "textwrap": _textwrap,
            "copy": _copy,
            "time": _time,
        }

    @staticmethod
    def _run_in_sandbox(code_string: str, namespace: dict) -> None:
        """Execute validated code in a restricted namespace.

        This is intentional sandboxed execution: the code has passed
        validate_code() safety checks, and the namespace has restricted
        builtins (no eval/exec/import/etc). This method exists to
        encapsulate the necessary code execution mechanism.

        H-12: Uses the global 'exec' function with a restricted namespace.
        The namespace's __builtins__ dictionary ensures only safe functions
        are available to the executed code.
        """
        compiled = compile(code_string, "<generated_tool>", "exec")
        # Intentional sandboxed execution of validated code
        # Note: we use the unrestricted exec to run the compiled code,
        # but the namespace's __builtins__ restricts what the code can access
        exec(compiled, namespace)


# ── ToolGenerator ────────────────────────────────────────────────

class ToolGenerator:
    """
    Creates, validates, persists, and manages dynamically generated tools.

    Usage::

        generator = ToolGenerator(
            tool_registry=registry,
            workspace_dir="/path/to/workspace",
        )

        tool = generator.generate_tool(
            name="count_words",
            description="Count words in text",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            python_code='def run(text="", **kwargs):\\n    return str(len(text.split()))',
        )

    Storage layout::

        {workspace}/.cowork/generated_tools/
            {tool_name}/
                tool.json     — metadata
                code.py       — Python source code
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        workspace_dir: str = "",
        max_tools: int = 50,
    ):
        self._registry = tool_registry
        self._workspace_dir = workspace_dir
        self._max_tools = max_tools
        self._tools: Dict[str, GeneratedTool] = {}

        self._storage_dir = ""
        if workspace_dir:
            self._storage_dir = os.path.join(
                workspace_dir, ".cowork", "generated_tools"
            )
            os.makedirs(self._storage_dir, exist_ok=True)

    # ── Properties ───────────────────────────────────────────

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def tools(self) -> Dict[str, GeneratedTool]:
        return dict(self._tools)

    # ── Generate ─────────────────────────────────────────────

    def generate_tool(
        self,
        name: str,
        description: str,
        input_schema: dict,
        python_code: str,
    ) -> GeneratedTool:
        """
        Create and register a new dynamically generated tool.

        Args:
            name: Tool name (must be unique, alphanumeric + underscores).
            description: Human-readable description.
            input_schema: JSON Schema for tool input parameters.
            python_code: Python code defining a `run(**kwargs)` function.

        Returns:
            The created GeneratedTool.

        Raises:
            ValueError: If validation fails.
        """
        # Validate name
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Invalid tool name '{name}'. Must be alphanumeric with underscores/hyphens."
            )

        if name in self._tools:
            raise ValueError(f"Tool '{name}' already exists. Delete it first.")

        if self.tool_count >= self._max_tools:
            raise ValueError(
                f"Maximum tool limit ({self._max_tools}) reached. "
                f"Delete unused tools first."
            )

        # Validate code
        validation = GeneratedTool.validate_code(python_code)
        if not validation.valid:
            raise ValueError(
                f"Code validation failed: {'; '.join(validation.errors)}"
            )

        # Create tool
        tool = GeneratedTool(
            name=name,
            description=description,
            input_schema=input_schema,
            code_string=python_code,
        )

        # Register
        self._tools[name] = tool
        self._save_tool(tool)

        logger.info(f"Generated tool '{name}' created and registered")
        return tool

    # ── Delete ───────────────────────────────────────────────

    def delete_tool(self, name: str) -> bool:
        """Delete a generated tool from registry and disk."""
        if name not in self._tools:
            return False

        del self._tools[name]

        # Remove from disk
        if self._storage_dir:
            tool_dir = os.path.join(self._storage_dir, name)
            if os.path.isdir(tool_dir):
                import shutil
                try:
                    shutil.rmtree(tool_dir)
                except OSError:
                    pass

        logger.info(f"Generated tool '{name}' deleted")
        return True

    # ── Get ──────────────────────────────────────────────────

    def get_tool(self, name: str) -> Optional[GeneratedTool]:
        """Get a generated tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[dict]:
        """List all generated tools as dicts."""
        return [t.to_dict() for t in self._tools.values()]

    # ── Persistence ──────────────────────────────────────────

    def _save_tool(self, tool: GeneratedTool) -> None:
        """Save a tool to disk."""
        if not self._storage_dir:
            return

        tool_dir = os.path.join(self._storage_dir, tool.name)
        os.makedirs(tool_dir, exist_ok=True)

        # Save metadata
        meta_path = os.path.join(tool_dir, "tool.json")
        try:
            with open(meta_path, "w") as f:
                json.dump(tool.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save tool metadata for {tool.name}: {e}")

        # Save code separately for readability
        code_path = os.path.join(tool_dir, "code.py")
        try:
            with open(code_path, "w") as f:
                f.write(tool.code_string)
        except Exception as e:
            logger.warning(f"Failed to save tool code for {tool.name}: {e}")

    def load_tools(self) -> int:
        """
        Load all persisted generated tools from disk.

        Returns the number of tools loaded.
        """
        if not self._storage_dir or not os.path.isdir(self._storage_dir):
            return 0

        count = 0
        for dirname in sorted(os.listdir(self._storage_dir)):
            tool_dir = os.path.join(self._storage_dir, dirname)
            if not os.path.isdir(tool_dir):
                continue

            meta_path = os.path.join(tool_dir, "tool.json")
            if not os.path.isfile(meta_path):
                continue

            try:
                with open(meta_path, "r") as f:
                    data = json.load(f)

                # Prefer code from code.py if it exists
                code_path = os.path.join(tool_dir, "code.py")
                if os.path.isfile(code_path):
                    with open(code_path, "r") as f:
                        data["code_string"] = f.read()

                tool = GeneratedTool.from_dict(data)

                # Re-validate before loading
                validation = GeneratedTool.validate_code(tool.code_string)
                if validation.valid:
                    self._tools[tool.name] = tool
                    count += 1
                else:
                    logger.warning(
                        f"Skipping invalid generated tool '{tool.name}': "
                        f"{validation.errors}"
                    )

            except Exception as e:
                logger.warning(f"Failed to load generated tool from {dirname}: {e}")

        logger.info(f"Loaded {count} generated tools from disk")
        return count
