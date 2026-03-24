"""A utility to write to both a stream (stdout/stderr) and a log file simultaneously."""

from __future__ import annotations

import sys


class TeeWriter:
    """Write to both a stream (stdout/stderr) and a log file simultaneously.

    Usage:
        with open("log.txt", "w") as log_file:
            try:
                TeeWriter.patch_stdout_and_stderr(log_file)
                # Your code that prints to stdout/stderr here
            finally:
                TeeWriter.restore_original_stdout_and_stderr()
    """

    def __init__(self, stream: object, log_file: object) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def reconfigure(self, **kwargs: object) -> None:
        if hasattr(self._stream, "reconfigure"):
            self._stream.reconfigure(**kwargs)

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")

    @classmethod
    def patch_stdout_and_stderr(cls, log_file: object) -> None:
        """Patch sys.stdout and sys.stderr to write to both the original stream and the log file."""
        sys.stdout = cls(sys.stdout, log_file)
        sys.stderr = cls(sys.stderr, log_file)

    @staticmethod
    def restore_original_stdout_and_stderr() -> None:
        """Restore the original sys.stdout and sys.stderr."""
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
