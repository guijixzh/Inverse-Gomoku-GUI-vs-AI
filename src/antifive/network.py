"""策略-价值网络(PolicyValueNet,AlphaZero 架构)

注意:本自研网络的训练总体失败。受自对弈数据规模与训练预算限制,最终强度不足,
最多稍弱于启发式引擎(heuristic.py),不具备实用棋力。权重与训练代码仅供复现研究,
正式对局请使用启发式 AI 或 KataGo。

输入: 5 通道 (15,15) [己方子, 对方子, 己方危险, 对方危险, 移子待定标记]
输出:
  - 策略 logits (N, 225):单个落子平面(纯落子空间,掩码按阶段限定语义),
    展开为全局 softmax
  - 价值 (N,) ∈ [-1, 1],从当前行棋方视角
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .reversegomoku import ACTION_SIZE, BOARD_SIZE

IN_PLANES = 5
POLICY_PLANES = 1                        # 纯落子空间:单平面


class ResBlock(nn.Module):
    """残差块:解决深层网络梯度消失,AlphaZero 核心结构。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 128, res_blocks: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IN_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res = nn.Sequential(*[ResBlock(channels) for _ in range(res_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, POLICY_PLANES, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        """x: (N,5,15,15) → (policy_logits (N,A), value (N,))"""
        feat = self.res(self.conv(x))
        p = self.policy_head(feat).reshape(-1, ACTION_SIZE)
        v = self.value_head(feat).reshape(-1)
        return p, v

    @torch.no_grad()
    def policy_value(self, states: torch.Tensor, masks: torch.Tensor):
        """掩码 softmax:供 MCTS 叶子批量评估。masks: (N,A) bool。
        返回 (probs (N,A), value (N,)),probs 在合法动作上归一化。"""
        logits, v = self.forward(states)
        logits = logits.masked_fill(~masks, float("-inf"))
        probs = torch.softmax(logits, dim=1)
        return probs, v

    # ---- 存取 ----
    def save(self, path: str, step: int = 0) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "channels": self.conv[0].out_channels,
            "res_blocks": len(self.res),
            "step": step,
        }, path)

    @classmethod
    def load(cls, path: str, device=None) -> "PolicyValueNet":
        ckpt = torch.load(path, map_location=device)
        net = cls(channels=ckpt["channels"], res_blocks=ckpt["res_blocks"])
        net.load_state_dict(ckpt["state_dict"])
        if device is not None:
            net = net.to(device)     # 网络本体也要搬到目标设备
        net.eval()
        return net
