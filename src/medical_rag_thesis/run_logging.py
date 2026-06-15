from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TextIO, TypeVar, Union


T = TypeVar("T")


class Tee:
    def __init__(self, console: TextIO, file_handle: TextIO):
        self.console = console
        self.file_handle = file_handle

    def write(self, text: str) -> int:
        self.console.write(text)
        self.file_handle.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.file_handle.flush()

    def isatty(self) -> bool:
        return self.console.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self.console, "encoding", "utf-8")


@contextmanager
def tee_std_streams(log_path: Union[str, Path], err_path: Union[str, Path]) -> Iterator[None]:
    log_path = Path(log_path)
    err_path = Path(err_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_handle, err_path.open(
        "w", encoding="utf-8"
    ) as err_handle:
        sys.stdout = Tee(original_stdout, log_handle)
        sys.stderr = Tee(original_stderr, err_handle)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_with_logs(log_path: Union[str, Path], err_path: Union[str, Path], func: Callable[[], T]) -> T:
    with tee_std_streams(log_path, err_path):
        try:
            return func()
        except Exception:
            traceback.print_exc(file=sys.stderr)
            raise
