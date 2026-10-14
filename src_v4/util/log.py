# -*- coding: utf-8 -*-
"""Self-contained experiment logging without cross-repository dependencies.

入口脚本调用 setup，其他模块用 get；setup 的 level / log_dir 缺省时
自动读 src_v4/config/config.py（LOG_LEVEL / LOGS_ROOT）。
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Windows 管道/控制台统一 UTF-8，避免中文乱码（与 common/exp_logger 一致）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = {}                 # task_name -> logger，单例缓存，避免重复 FileHandler
_active_task: Optional[str] = None  # 当前活动任务名（setup 设置）


def _unique_path(log_dir: str, task_name: str, stamp: str = None) -> str:
    """生成不覆盖的日志文件路径：HHmmss_任务.log，已存在则 _1/_2 递增。"""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_name)
    stamp = stamp or datetime.now().strftime("%H%M%S")
    base = os.path.join(log_dir, f"{stamp}_{safe}.log")
    path, seq = base, 1
    while os.path.exists(path):
        path = f"{os.path.splitext(base)[0]}_{seq}{os.path.splitext(base)[1]}"
        seq += 1
    return path


def setup(task_name: str, log_dir: Optional[str] = None, console: bool = True,
          level: Optional[int] = None) -> logging.Logger:
    """配置任务级日志（等价 common.exp_logger.setup_exp_logging，自包含）。

    同任务只配置一次（返回已有 logger）；log_dir/level 缺省读 config.config。
    """
    if task_name in _configured:
        return _configured[task_name]
    if log_dir is None:
        try:
            from config.config import LOGS_ROOT
            log_dir = LOGS_ROOT or None
        except Exception:
            log_dir = None
    if level is None:
        level = logging.INFO
        try:
            from config.config import LOG_LEVEL
            level = getattr(logging, str(LOG_LEVEL).upper(), logging.INFO)
        except Exception:
            pass

    log_root = log_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "logs")
    day_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(day_dir, exist_ok=True)
    path = _unique_path(day_dir, task_name)

    logger = logging.getLogger(f"exp.{task_name}")
    logger.setLevel(level)
    logger.propagate = False  # 避免与 root 重复输出
    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    global _active_task
    _active_task = task_name
    logger.info("==== 日志系统启动: 任务=%s 文件=%s ====", task_name, os.path.basename(path))
    _configured[task_name] = logger
    return logger


def get(name: Optional[str] = None) -> logging.Logger:
    """等价 common.exp_logger.get_logger（有活动任务时命名到 exp.<任务>.<name>）。"""
    if _active_task:
        if name and not name.startswith(("exp.", f"exp.{_active_task}.")):
            return logging.getLogger(f"exp.{_active_task}.{name}")
        if name:
            return logging.getLogger(name)
        return logging.getLogger(f"exp.{_active_task}")
    return logging.getLogger(name) if name else logging.getLogger()
