"""Execution environment abstraction."""

from __future__ import annotations

import fnmatch
import os
import platform as _platform
import re
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int


@dataclass
class DirEntry:
    name: str
    is_dir: bool
    size: int | None = None


class ExecutionEnvironment(ABC):
    """Abstract interface for tool execution backends."""

    @abstractmethod
    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str: ...

    @abstractmethod
    def write_file(self, path: str, content: str) -> None: ...

    @abstractmethod
    def file_exists(self, path: str) -> bool: ...

    @abstractmethod
    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]: ...

    @abstractmethod
    def exec_command(self, command: str, timeout_ms: int = 10_000, working_dir: str | None = None, env_vars: dict[str, str] | None = None) -> ExecResult: ...

    @abstractmethod
    def grep(self, pattern: str, path: str, options: dict[str, Any]) -> str: ...

    @abstractmethod
    def glob(self, pattern: str, path: str) -> list[str]: ...

    @abstractmethod
    def working_directory(self) -> str: ...

    @abstractmethod
    def platform(self) -> str: ...

    @abstractmethod
    def os_version(self) -> str: ...

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Sensitive environment variable patterns
# ---------------------------------------------------------------------------

_SENSITIVE_SUFFIXES = {"_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "_CREDENTIAL"}
_CORE_VARS = {"PATH", "HOME", "USER", "SHELL", "LANG", "TERM", "TMPDIR", "LOGNAME"}


class LocalExecutionEnvironment(ExecutionEnvironment):
    """Default implementation – runs on local machine."""

    def __init__(self, working_dir: str = ".") -> None:
        self._working_dir = os.path.abspath(working_dir)

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self._working_dir, path)

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        abs_path = self._resolve(path)
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if offset:
            lines = lines[offset - 1:]
        if limit:
            lines = lines[:limit]
        return "".join(lines)

    def write_file(self, path: str, content: str) -> None:
        abs_path = self._resolve(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

    def file_exists(self, path: str) -> bool:
        return os.path.exists(self._resolve(path))

    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]:
        abs_path = self._resolve(path)
        entries: list[DirEntry] = []
        for name in sorted(os.listdir(abs_path)):
            full = os.path.join(abs_path, name)
            entries.append(DirEntry(name=name, is_dir=os.path.isdir(full), size=os.path.getsize(full) if os.path.isfile(full) else None))
        return entries

    def exec_command(self, command: str, timeout_ms: int = 10_000, working_dir: str | None = None, env_vars: dict[str, str] | None = None) -> ExecResult:
        cwd = working_dir or self._working_dir
        env = self._filter_env(env_vars)
        start = time.monotonic()

        try:
            kwargs: dict[str, Any] = {
                "shell": True,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "cwd": cwd,
                "env": env,
            }
            if sys.platform != "win32":
                kwargs["preexec_fn"] = os.setsid

            proc = subprocess.Popen(command, **kwargs)
            try:
                stdout, stderr = proc.communicate(timeout=timeout_ms / 1000)
                timed_out = False
            except subprocess.TimeoutExpired:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
                time.sleep(1)
                if proc.poll() is None:
                    if sys.platform != "win32":
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                stdout, stderr = proc.communicate()
                timed_out = True

            return ExecResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=proc.returncode if proc.returncode is not None else -1,
                timed_out=timed_out,
                duration_ms=int((time.monotonic() - start) * 1000),
            )
        except Exception as exc:
            return ExecResult(stdout="", stderr=str(exc), exit_code=-1, timed_out=False, duration_ms=int((time.monotonic() - start) * 1000))

    def grep(self, pattern: str, path: str, options: dict[str, Any]) -> str:
        abs_path = self._resolve(path)
        glob_filter = options.get("glob_filter")
        case_insensitive = options.get("case_insensitive", False)
        max_results = options.get("max_results", 100)

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return f"Invalid regex: {exc}"

        results: list[str] = []
        target = Path(abs_path)

        if target.is_file():
            files = [target]
        else:
            files = sorted(target.rglob("*"))

        for file_path in files:
            if not file_path.is_file():
                continue
            if glob_filter and not fnmatch.fnmatch(file_path.name, glob_filter):
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{file_path}:{lineno}: {line}")
                    if len(results) >= max_results:
                        return "\n".join(results)
        return "\n".join(results) if results else "No matches found."

    def glob(self, pattern: str, path: str) -> list[str]:
        base = Path(self._resolve(path))
        matches = sorted(base.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
        return [str(m) for m in matches]

    def working_directory(self) -> str:
        return self._working_dir

    def platform(self) -> str:
        if sys.platform.startswith("darwin"):
            return "darwin"
        if sys.platform.startswith("linux"):
            return "linux"
        if sys.platform == "win32":
            return "windows"
        return "unknown"

    def os_version(self) -> str:
        return _platform.platform()

    @staticmethod
    def _filter_env(provided: dict[str, str] | None) -> dict[str, str]:
        base = dict(provided) if provided else dict(os.environ)
        filtered: dict[str, str] = {}
        for key, value in base.items():
            if key in _CORE_VARS:
                filtered[key] = value
                continue
            if any(key.upper().endswith(s) for s in _SENSITIVE_SUFFIXES):
                continue
            filtered[key] = value
        return filtered
