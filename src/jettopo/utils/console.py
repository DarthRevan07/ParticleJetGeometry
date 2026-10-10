"""Rich console with log levels and a timed status spinner.

Ported from 2024 ``console/console.py`` and ``console/status.py`` (requires the
optional ``rich`` package). Change: ``LogStatus`` is defined here, fixing the
2024 ``from status import LogStatus`` import that only worked from that folder.
"""
from __future__ import annotations

import logging
import os
from time import time
from typing import Optional, Union

from rich._log_render import LogRender
from rich.console import Console as RichConsole
from rich.console import JustifyMethod, NewLine
from rich.live import Live
from rich.logging import RichHandler
from rich.scope import render_scope
from rich.segment import Segment
from rich.spinner import Spinner
from rich.status import Status
from rich.style import Style
from rich.styled import Styled
from rich.table import Table


class LogStatus(Status):
    def __init__(self, status, console: RichConsole, level: int = logging.INFO, enabled: bool = True,
                 speed: float = 1.0, refresh_per_second: float = 12.5):
        super().__init__(status, console=console, spinner="simpleDots", speed=speed, refresh_per_second=refresh_per_second)
        self.status, self.level, self.enabled = status, level, enabled
        spinner = Spinner("simpleDots", style="status.spinner", speed=speed)
        record = logging.LogRecord(name=None, level=level, pathname=None, lineno=None, msg=None, args=None, exc_info=None)
        handler = RichHandler(console=console)
        table = Table.grid()
        table.add_row(self.status, spinner)
        self._spinner = LogRender(show_level=True, time_format="[%X]")(
            console=console, level=handler.get_level_text(record), renderables=[table])
        self._live = Live(self.renderable, console=console, refresh_per_second=refresh_per_second, transient=True)

    def __enter__(self):
        if self.enabled:
            self._start_time = time()
            return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            super().__exit__(exc_type, exc_val, exc_tb)
            self.console.log(f"{self.status}...done in {time() - self._start_time:.2f} s", level=self.level)


class Console(RichConsole):
    CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET = (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO,
                                                     logging.DEBUG, logging.NOTSET)

    def __init__(self, *args, log_level: int = logging.INFO, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_level = log_level

    def status(self, status: str, level: int = logging.INFO) -> LogStatus:
        return LogStatus(status, console=self, level=level, enabled=level >= self.log_level)

    def log(self, *objects, level: int = logging.INFO, sep: str = " ", end: str = "\n",
            style: Union[str, Style, None] = None, justify: Optional[JustifyMethod] = None, emoji: Optional[bool] = None,
            markup: Optional[bool] = None, highlight: Optional[bool] = None, log_locals: bool = False,
            _stack_offset: int = 2) -> None:
        if level < self.log_level:
            return
        if not objects:
            objects = (NewLine(),)
        render_hooks = self._render_hooks[:]
        with self:
            renderables = self._collect_renderables(objects, sep, end, justify=justify, emoji=emoji, markup=markup,
                                                    highlight=highlight)
            if style is not None:
                renderables = [Styled(r, style) for r in renderables]
            filename, line_no, locals_ = self._caller_frame_info(_stack_offset)
            link_path = None if filename.startswith("<") else os.path.abspath(filename)
            path = filename.rpartition(os.sep)[-1]
            if log_locals:
                renderables.append(render_scope({k: v for k, v in locals_.items() if not k.startswith("__")},
                                                title="[i]locals"))
            record = logging.LogRecord(name=None, level=level, pathname=None, lineno=None, msg=None, args=None,
                                       exc_info=None)
            handler = RichHandler(console=self)
            self._log_render.show_level = True
            renderables = [self._log_render(self, renderables, level=handler.get_level_text(record),
                                            log_time=self.get_datetime(), path=path, line_no=line_no,
                                            link_path=link_path)]
            for hook in render_hooks:
                renderables = hook.process_renderables(renderables)
            new_segments = []
            for renderable in renderables:
                new_segments.extend(self.render(renderable, self.options))
            for line in Segment.split_and_crop_lines(new_segments, self.width, pad=False):
                self._buffer.extend(line)

    def debug(self, *args, **kwargs):
        self.log(*args, level=self.DEBUG, **kwargs)

    def info(self, *args, **kwargs):
        self.log(*args, level=self.INFO, **kwargs)

    def warning(self, *args, **kwargs):
        self.log(*args, level=self.WARNING, **kwargs)

    def error(self, *args, **kwargs):
        self.log(*args, level=self.ERROR, **kwargs)
