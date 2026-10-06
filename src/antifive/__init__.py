"""Antifive(逆五子棋):连五者输的 AlphaZero 风格 AI 与 pygame 图形界面。

核心模块:
- reversegomoku   规则引擎(纯 numpy)
- mcts            AlphaZero 风格蒙特卡洛树搜索(torch 可选)
- network         策略价值网络(torch 可选)
- heuristic       启发式 AI(纯 numpy)
- tactics         杀棋检测
- record          .afg 棋谱读写
- gui             pygame 图形界面
"""

__version__ = "0.9.0"
