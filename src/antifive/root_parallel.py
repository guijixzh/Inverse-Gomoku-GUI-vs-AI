"""根节点并行搜索池(进程级)。

背景:启发式引擎是纯 Python + numpy 小数组,单核约 1.5k 节点/秒;
alpha-beta 在根节点对每个候选着法都是"全窗口独立评估",天然可并行。
本模块提供一个**常驻**的 spawn 进程池(避免每次落子重新导入 numpy),
供各引擎模块把自己的"根着法评估"函数丢进来跑。

设计约束:
- 结果必须与串行**逐位一致**(并行只拆根,不改搜索):调用方按返回值排序,
  顺序取回后与串行同序,noise 由调用方在父进程加,避免各进程随机流不同;
- 每个 worker 各自持有置换表(worker 进程内全局缓存,跨深度迭代复用),
  不再与父进程共享 TT——这是并行的既定代价(实测影响很小);
- 时限:父进程把"剩余秒数"传给 worker,worker 用自己的 perf_counter 建 deadline;
- 子进程用 spawn 启动(Windows 语义一致),因此任务参数必须可 pickle。

典型用法(见 `tools/heuristic_v6.py:_root_iterative`):
    from ..root_parallel import map_tasks, shutdown_pools
    res = map_tasks("antifive.tools.heuristic_v6:_root_child_task", tasks, workers)
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import threading

_POOLS: dict[int, object] = {}
_LOCK = threading.Lock()

# 单个 worker 至少应该拿到这么多秒的预算,否则进程调度开销不值得
MIN_TASK_SECONDS = 0.02


def _run_task(payload):
    """在子进程里执行 (fn_path, args) —— fn_path = "module:function"。"""
    fn_path, args = payload
    mod_name, fn_name = fn_path.split(":", 1)
    fn = getattr(importlib.import_module(mod_name), fn_name)
    return fn(args)


def _allow_children() -> None:
    """允许在 multiprocessing.Pool 的 daemon 子进程里再开进程池。

    `match_web_py` 等对拍工具用 Pool 跑对局,Pool 子进程默认 daemon=True,
    而 daemon 进程按标准库断言不允许再有子进程。这里把当前进程标记改回
    非 daemon —— 本池的生命周期由本进程守护,父进程结束时随之退出,安全。"""
    try:
        proc = mp.current_process()
        if proc.daemon:
            proc.daemon = False
            cfg = getattr(proc, "_config", None)
            if isinstance(cfg, dict):
                cfg["daemon"] = False
    except Exception:
        pass


def _get_pool(workers: int):
    with _LOCK:
        pool = _POOLS.get(workers)
        if pool is None:
            _allow_children()
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("spawn"))
            _POOLS[workers] = pool
        return pool


def map_tasks(fn_path: str, tasks: list, workers: int) -> list:
    """把 tasks 分发给 workers 个进程执行(workers<=1 或任务太少时串行)。"""
    if not tasks:
        return []
    if workers <= 1 or len(tasks) == 1:
        return [_run_task((fn_path, t)) for t in tasks]
    workers = min(int(workers), len(tasks))
    pool = _get_pool(workers)
    return list(pool.map(_run_task, [(fn_path, t) for t in tasks]))


def default_workers() -> int:
    """环境变量 ANTIFIVE_ROOT_WORKERS(0/空=串行);上限为 CPU 核数。"""
    raw = os.environ.get("ANTIFIVE_ROOT_WORKERS", "").strip()
    if not raw:
        return 1
    try:
        n = int(raw)
    except ValueError:
        return 1
    if n <= 1:
        return 1
    return max(1, min(n, os.cpu_count() or 1))


def shutdown_pools() -> None:
    """关闭所有常驻池(测试/进程退出前调用)。"""
    with _LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


import atexit as _atexit

# 进程退出时收尾:否则在 mp.Pool 的子进程里再开的池会留下孤儿进程。
_atexit.register(shutdown_pools)
