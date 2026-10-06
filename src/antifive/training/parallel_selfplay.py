"""多进程并行自对弈(训练主进程 + N 个自对弈 worker 进程)

架构:
- 主进程(train.py):只做训练/评估/存档,从队列收集 worker 回传的样本;
- worker 进程:各自持有网络副本,持续运行 self_play_batch,产出样本
  经 multiprocessing.Queue 回传(队列有容量上限 → 天然背压);
- 权重同步:worker 每个自对弈周期检查 latest_model.pth 的 mtime,变化即重载,
  不中断自对弈(轻微滞后,符合 AlphaZero 并行训练惯例)。

Windows 使用 spawn 上下文;worker 内部异常会打印并退出,主进程靠
收集超时感知 worker 失活。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import time
import traceback

import numpy as np
import torch

from ..network import PolicyValueNet
from ..reversegomoku import GameConfig
from .selfplay import self_play_batch


def _worker_main(model_path, device_str, channels, res_blocks, config_dict, seed,
                 n_games, sims, k, c_puct, temperature_moves, alpha, noise_eps,
                 out_q, stop_ev):
    """worker 进程入口。"""
    err_log = os.path.join(os.path.dirname(os.path.abspath(model_path)), "worker_error.log")
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        device = torch.device(device_str)
        net = PolicyValueNet(channels=channels, res_blocks=res_blocks).to(device)
        config = GameConfig(**config_dict)
        last_mtime = -1.0
        while not stop_ev.is_set():
            # 刷新权重(模型文件被主进程原子替换)
            try:
                mtime = os.path.getmtime(model_path)
                if mtime != last_mtime and os.path.exists(model_path):
                    ckpt = torch.load(model_path, map_location=device)
                    net.load_state_dict(ckpt["state_dict"])
                    last_mtime = mtime
            except Exception:
                pass
            exs = self_play_batch(
                net, config=config, n_games=n_games, sims=sims, c_puct=c_puct,
                dirichlet_alpha=alpha, noise_eps=noise_eps,
                temperature_moves=temperature_moves, device=device, k=k)
            if not stop_ev.is_set():
                out_q.put(exs)
    except Exception:
        err = traceback.format_exc()
        try:
            with open(err_log, "a", encoding="utf-8") as f:
                f.write(f"[worker seed={seed}]\n{err}\n")
        except Exception:
            pass
        print(err, flush=True)
    finally:
        try:
            out_q.close()
        except Exception:
            pass


class ParallelSelfPlay:
    """自对弈 worker 池(主进程侧)。"""

    def __init__(self, model_path: str, device_str: str, channels: int, res_blocks: int,
                 config: GameConfig, workers: int, worker_games: int, sims: int,
                 k: int, c_puct: float, temperature_moves: int,
                 alpha: float, noise_eps: float, queue_size: int = 4,
                 seed: int = 0):
        self.model_path = model_path
        self.target_examples = workers * worker_games * 40   # 每迭代期望样本量(经验值)
        ctx = mp.get_context("spawn")
        self.out_q = ctx.Queue(maxsize=queue_size)
        self.stop_ev = ctx.Event()
        self.procs = []
        for w in range(workers):
            p = ctx.Process(
                target=_worker_main,
                args=(model_path, device_str, channels, res_blocks,
                      config.__dict__.copy(), seed + w * 100000,
                      worker_games, sims, k, c_puct, temperature_moves,
                      alpha, noise_eps, self.out_q, self.stop_ev),
                daemon=True)
            p.start()
            self.procs.append(p)
        print(f"已启动 {workers} 个自对弈 worker(每 worker {worker_games} 局 x {sims} 模拟, "
              f"k={k},目标样本/迭代 ~{self.target_examples})", flush=True)

    def collect(self, timeout: float = 300.0, max_wait: float = None):
        """收集样本:有产出就继续;全部 worker 失活或超过总期限(max_wait)才返回。
        自杀掩码补丁后对局显著变长,首轮产包可能需 30~60 分钟,不再因单次超时误判。"""
        if max_wait is None:
            max_wait = timeout * 12
        deadline = time.time() + max_wait
        examples = []
        last_status = time.time()
        while len(examples) < self.target_examples and time.time() < deadline:
            wait = min(60.0, max(0.0, deadline - time.time()))
            try:
                got = self.out_q.get(timeout=wait)
                examples.extend(got)
                last_status = time.time()
            except _queue.Empty:
                if not any(p.is_alive() for p in self.procs):
                    break
                if time.time() - last_status > 60:
                    alive = sum(1 for p in self.procs if p.is_alive())
                    print(f"    等待 worker 产包... 已收集 {len(examples)} 样本, "
                          f"存活 {alive}/{len(self.procs)} worker, 已等 "
                          f"{time.time() - (deadline - max_wait):.0f}s / {max_wait:.0f}s", flush=True)
                    last_status = time.time()
        return examples

    def close(self) -> None:
        self.stop_ev.set()
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
