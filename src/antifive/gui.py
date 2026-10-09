"""逆五子棋 pygame 图形界面(精致交互版)

操作:
- 左键:普通回合点空位=落子,点对方棋子=占领(移子第一步);
       移子待定时,绿色高亮为可达格,点击=安置手中棋子
- 按钮/快捷键: 1-4 切换模式, R 新局, U 悔棋, D 危险提示, N 棋子手数,
  A 移子箭头指示, H 辅助射线, S 保存棋谱, L 读取棋谱, E 设定, Esc 退出;
  「设定」按钮位于右侧按钮列表最顶端,随时更换 AI 与对局设置;
  右上角时钟显示 AI 读秒(分钟:秒.十分位),点击可开关限时,
  危险/手数/箭头/射线开关与全部快捷键提示统一在棋盘下方
- 读秒:AI 读秒倒计时固定在右上角时钟框内(暖木配色),不再遮挡棋盘;
  设定面板中思考时间/模拟次数/深度均可点击输入框直接输入数字
- AI 状态:右上角「AI 读秒」标题右侧为 AI 思考指示灯(思考中红灯+「AI思考中」,
  空闲为暗灯+「AI空闲」);信息集中在时钟下方状态框,右侧面板仅保留按钮与署名
- 状态框:时钟下方为引擎行(名称+权重/模型文件名)与两列信息格(限时/参数,
  手数/行棋,行棋带落子/移子动作后缀),各状态独立成格;底部整行状态条
  (落子/移子待定、复盘/试下提示、终局结果)按状态着色,随模式自动变化;
  其下为更新提示空隙(读秒开关/设定应用等临时消息),位于按钮区正上方;
  按钮区下方固定规则讲解,右下角固定署名 The gui made by 硅基飙尘葆光
- 底部提示:开关按钮行右侧为随状态变化的提示(终局/复盘操作),正下方为单行快捷键
- 辅助射线:鼠标悬停在棋子上方时,临时以射线+圆点标出该棋子可移动到的所有位置,
  棋盘下方「辅助射线」按钮或 H 键随时开关
- 音效:落子/移子播放合成短音效(无需外部资源),棋盘下方「音效」按钮或 M 键随时开关
- 移子箭头:占领后安置棋子时,棋子沿直线飞至落子点(0.5s 动画),
  随后蓝色箭头(实心三角头)保持显示,直到下一步完成;
  安置待定时悬停可达格显示淡色预览箭头
- 复盘:一局结束或读取棋谱后自动进入复盘模式,底部出现尺子式手数轴
  (隔 10 长刻度并标数、隔 5 中刻度、隔 1 短刻度),
  可点轴跳转、按钮/方向键逐手回退,未终局时可点击棋盘试下
  (分支变化,「原谱」按钮一键恢复主变化)

棋谱:.afg 纯文本格式(见 record.py),含规则配置与结果,可供 AI 训练解析。

用法:
    python -m antifive.gui                             # 或安装后直接执行 antifive
    python -m antifive.gui --model models/best_model.pth  # 指定自研神经网络(可选)
    python -m antifive.gui --engine heuristic          # 启发式 AI(默认深度 8 + 读秒 12s)
    python -m antifive.gui --engine heuristic --heuristic-depth 6  # 指定深度
    python -m antifive.gui --engine katago             # KataGo(路径取自 config/engines.json)
    python -m antifive.gui --model x.pth --sims 2000   # 调整神经网络模拟次数
    python -m antifive.gui --limit ai --ai-time 15     # 仅 AI 限时读秒,每步 15 秒(默认 ai/12)
    python -m antifive.gui --limit none                # 不限时(深度 8 单步可能较慢)

torch 为可选依赖:未安装时神经网络引擎自动禁用,其余功能不受影响。
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pygame

from . import paths, record
from .mcts import MCTS
from .reversegomoku import (
    BLACK, BOARD_SIZE, EMPTY, GameConfig, ReverseGomoku, WHITE, danger_map_for,
)
from .tactics import kill_captures

try:
    import torch
    from .network import PolicyValueNet

    NET_AVAILABLE = True
except ImportError:                     # torch 为可选依赖:无网络时其余功能可用
    torch = None
    PolicyValueNet = None
    NET_AVAILABLE = False

from dataclasses import replace as _dc_replace

CELL = 50                       # 每格像素
MARGIN = 48                     # 棋盘边距
BOARD_PX = CELL * (BOARD_SIZE - 1) + 2 * MARGIN
PANEL_W = 320
WIN_W = BOARD_PX + PANEL_W
WIN_H = BOARD_PX + 150          # 底部多留空间容纳全量快捷键提示

HEURISTIC_DEPTH_DEFAULT = 8     # 启发式 AI 默认搜索深度(三档版本通用,与网页困难档一致)
HEURISTIC_TIME_BUDGET = 50.0     # 启发式 AI 不限时模式外的兜底思考时间(秒)
# 引擎版本三档: beginner=初级(经典) / mid=中级(修复+开局库,无蒸馏) / advanced=高级(+蒸馏)
HEURISTIC_VERSION_DEFAULT = "advanced"
HEURISTIC_VCF_DEFAULT = True        # 中级/高级是否启用强制杀链搜索(VCF)
AI_TIME_BUDGET_DEFAULT = 12.0    # 默认读秒:AI 每步思考时间上限(秒)
LIMIT_MODE_DEFAULT = "ai"        # 默认仅 AI 限时读秒(不限时可被搜太慢,深度 8 尤甚)
NN_SIMS_DEFAULT = 1200           # 神经网络"决策默认阈值"(默认模拟次数)

MODE_NAMES = ("人执黑", "人执白", "人人对战", "机机观战")
MODE_HUMAN = {0: BLACK, 1: WHITE}          # 人类执子颜色,2/3 无人类

# 配色
COLOR_WOOD = (222, 178, 128)
COLOR_WOOD_DARK = (166, 114, 58)
COLOR_LINE = (78, 52, 20)
COLOR_TEXT = (40, 28, 16)
COLOR_PANEL = (247, 240, 226)
COLOR_BTN = (224, 208, 176)
COLOR_BTN_ACT = (255, 214, 130)
COLOR_BTN_HOVER = (238, 222, 192)
COLOR_DANGER_SELF = (220, 60, 60)
COLOR_DANGER_OPP = (228, 170, 40)
COLOR_TARGET = (60, 200, 90)
COLOR_LAST = (235, 70, 50)
COLOR_ARROW = (45, 110, 230)
COLOR_RAY = (140, 110, 230)          # 辅助射线(悬停棋子可移动位置)用色,区别于绿色落点/红色危险

ARROW_FLY_DURATION = 0.5         # 移子棋子飞行动画时长(秒),之后蓝色箭头保持显示
RESULT_OVERLAY_HOLD = 2.0        # 终局大字蒙版完整显示时长(秒)
RESULT_OVERLAY_FADE = 0.8        # 终局大字蒙版淡出时长(秒)

# 右上角固定位置读秒时钟框(不再悬浮于棋盘之上遮挡棋盘)
CLOCK_RECT = pygame.Rect(BOARD_PX + 20, 14, PANEL_W - 40, 66)
# 时钟下方状态框:引擎(名称/文件名) + 两列信息格(限时/参数, 手数/行棋) + 状态条
SETTINGS_RECT = pygame.Rect(BOARD_PX + 20, CLOCK_RECT.bottom + 10,
                            PANEL_W - 40, 97)
# 按钮区正上方的更新提示空隙(读秒开关/设定应用等临时消息)
NOTIFY_RECT = pygame.Rect(BOARD_PX + 20, SETTINGS_RECT.bottom + 6,
                          PANEL_W - 40, 26)
PANEL_BUTTON_TOP = NOTIFY_RECT.bottom + 10  # 右侧面板按钮起始 y(为时钟/状态框/提示条让位)


def _load_font(size: int):
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "msyh.ttf"):
        path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    return pygame.font.SysFont("microsoftyahei,simhei,simsun", size)


def _fmt_clock(seconds: float) -> str:
    """读秒时钟显示格式 M:SS.s(分钟:秒.十分位),如 0:12.0 / 12:34.5 / 10:00.0。"""
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m)}:{s:04.1f}"


def _make_sound(freq: float, dur: float, vol: float = 0.5,
                decay: float = 9.0):
    """numpy 合成短音效(正弦+泛音+指数衰减),mixer 未初始化返回 None。"""
    init = pygame.mixer.get_init()
    if init is None:
        return None
    sr, _fmt, channels = init
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    w = (np.sin(2 * np.pi * freq * t)
         + 0.4 * np.sin(2 * np.pi * freq * 2.0 * t)) * np.exp(-t * decay)
    w = np.clip(w * vol / 1.4, -1.0, 1.0)
    data = (w * 32767).astype(np.int16)
    if channels >= 2:
        data = np.repeat(data[:, None], 2, axis=1)   # 立体声混音器需双通道
    snd = pygame.sndarray.make_sound(data)
    snd.set_volume(0.85)
    return snd




def _stone_surface(radius: int, color: tuple, highlight: bool = False):
    """径向渐变棋子贴图。"""
    size = radius * 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    steps = 24
    for i in range(steps, 0, -1):
        r = int(radius * i / steps)
        if color[0] > 128:                       # 白子:边缘暗
            k = 0.75 + 0.25 * i / steps
            c = tuple(int(x * k) for x in color)
        else:                                    # 黑子:边缘亮
            k = 0.65 + 0.35 * (steps - i) / steps
            c = tuple(min(255, int(x + (255 - x) * (1 - k))) for x in color)
        pygame.draw.circle(surf, (*c, 255), (radius, radius), r)
    if highlight:
        pygame.draw.circle(surf, COLOR_LAST, (radius, radius), radius, 3)
    return surf


class Board:
    """棋盘静态层(木纹 + 网格 + 星位)+ 动态棋子渲染。"""

    def __init__(self):
        self.font_idx = _load_font(14)
        self.stone_r = CELL // 2 - 3
        self.bg = self._build_bg()
        self.stone_b = _stone_surface(self.stone_r, (30, 30, 34))
        self.stone_w = _stone_surface(self.stone_r, (242, 240, 235))
        self.stone_b_h = _stone_surface(self.stone_r, (30, 30, 34), True)
        self.stone_w_h = _stone_surface(self.stone_r, (242, 240, 235), True)

    def _build_bg(self):
        surf = pygame.Surface((BOARD_PX, BOARD_PX))
        surf.fill(COLOR_WOOD)
        # 木纹
        rng = np.random.default_rng(7)
        for _ in range(220):
            x = rng.uniform(0, BOARD_PX)
            y = rng.uniform(0, BOARD_PX)
            w = rng.uniform(30, 160)
            h = rng.uniform(1, 3)
            a = rng.uniform(0, 3.14)
            c = rng.integers(-14, 10)
            shade = tuple(max(0, min(255, x + c)) for x in COLOR_WOOD)
            s = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
            s.fill((*shade, 46))
            s = pygame.transform.rotate(s, np.degrees(a))
            surf.blit(s, (x - w / 2, y - h / 2))
        # 边框
        pygame.draw.rect(surf, COLOR_WOOD_DARK, (0, 0, BOARD_PX, BOARD_PX), 6)
        # 网格
        for i in range(BOARD_SIZE):
            p = MARGIN + i * CELL
            pygame.draw.line(surf, COLOR_LINE, (MARGIN, p), (BOARD_PX - MARGIN, p), 1)
            pygame.draw.line(surf, COLOR_LINE, (p, MARGIN), (p, BOARD_PX - MARGIN), 1)
        # 星位
        for r, c in ((3, 3), (3, 11), (11, 3), (11, 11), (7, 7)):
            pygame.draw.circle(surf, COLOR_LINE,
                               (MARGIN + c * CELL, MARGIN + r * CELL), 5)
        # 坐标
        for i in range(BOARD_SIZE):
            lbl = self.font_idx.render(chr(65 + i), True, (120, 80, 40))
            surf.blit(lbl, (MARGIN + i * CELL - lbl.get_width() // 2, 12))
            surf.blit(lbl, (MARGIN + i * CELL - lbl.get_width() // 2,
                            BOARD_PX - 24))
            num = self.font_idx.render(str(i), True, (120, 80, 40))
            surf.blit(num, (10, MARGIN + i * CELL - num.get_height() // 2))
            surf.blit(num, (BOARD_PX - 22, MARGIN + i * CELL - num.get_height() // 2))
        return surf

    def cell_pos(self, r: int, c: int):
        return (MARGIN + c * CELL, MARGIN + r * CELL)

    @staticmethod
    def _draw_arrow(surf, p1, p2, head_t: float, alpha: int) -> None:
        """在 surf 上画蓝色箭头:实心三角头位于 p1→p2 的 head_t 比例处,尾线止于三角底边。"""
        x1, y1 = p1
        x2, y2 = p2
        if head_t < 0.03:                          # 箭头尚未出发
            return
        hx, hy = x1 + (x2 - x1) * head_t, y1 + (y2 - y1) * head_t
        ang = math.atan2(y2 - y1, x2 - x1)
        hl, hw = 15, 9                              # 三角头长/半宽
        bx, by = hx - hl * math.cos(ang), hy - hl * math.sin(ang)
        px, py = -math.sin(ang), math.cos(ang)      # 垂直方向单位向量
        pygame.draw.polygon(surf, (*COLOR_ARROW, alpha),
                            ((hx, hy), (bx + hw * px, by + hw * py),
                             (bx - hw * px, by - hw * py)))
        pygame.draw.line(surf, (*COLOR_ARROW, alpha), p1, (bx, by), 4)

    def draw(self, screen, game, hover, held_pos, show_danger, legal_mask,
             show_numbers: bool = False, num: np.ndarray = None,
             arrow: tuple = None, hover_target: tuple = None,
             hover_rays: tuple = None):
        screen.blit(self.bg, (0, 0))
        # 危险提示
        if show_danger:
            d_self = danger_map_for(game.board, game.current_player)
            d_opp = danger_map_for(game.board, WHITE if game.current_player == BLACK else BLACK)
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if d_self[r, c]:
                        pygame.draw.circle(screen, (*COLOR_DANGER_SELF, 90),
                                           self.cell_pos(r, c), self.stone_r - 6)
                    elif d_opp[r, c]:
                        pygame.draw.circle(screen, (*COLOR_DANGER_OPP, 90),
                                           self.cell_pos(r, c), self.stone_r - 6)
        # 移子待定:绿色可达格 + 手中棋子随鼠标
        if game.pending >= 0:
            for i in np.nonzero(legal_mask)[0]:
                r, c = divmod(int(i), BOARD_SIZE)
                pos = self.cell_pos(r, c)
                pygame.draw.circle(screen, COLOR_TARGET, pos, self.stone_r - 4, 2)
                pygame.draw.circle(screen, (*COLOR_TARGET, 120), pos, 7)
            if held_pos:
                pos = self.cell_pos(*held_pos)
                # 手中棋子属于对方:持黑者拿白子,持白者拿黑子
                surf = self.stone_b if game.current_player == WHITE else self.stone_w
                surf.set_alpha(150)
                screen.blit(surf, (pos[0] - self.stone_r, pos[1] - self.stone_r))
                surf.set_alpha(255)
        # 辅助射线:悬停棋子临时标出可移动到的所有位置(画在棋子层下方)
        if hover_rays is not None and game.pending < 0:
            src, rm = hover_rays
            p1 = self.cell_pos(*divmod(int(src), BOARD_SIZE))
            overlay = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
            for i in np.nonzero(rm)[0]:
                p2 = self.cell_pos(*divmod(int(i), BOARD_SIZE))
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                dist = math.hypot(dx, dy)
                if dist > 1:
                    ux, uy = dx / dist, dy / dist
                    # 线从棋子边缘发出,终点收在圆点外圈,避免穿越大棋子产生偏移感
                    s0 = (p1[0] + ux * (self.stone_r + 2),
                          p1[1] + uy * (self.stone_r + 2))
                    e0 = (p2[0] - ux * 11, p2[1] - uy * 11)
                    pygame.draw.line(overlay, (*COLOR_RAY, 120), s0, e0, 2)
                pygame.draw.circle(overlay, (*COLOR_RAY, 205), p2, 7)
                pygame.draw.circle(overlay, (*COLOR_RAY, 80), p2, 11, 1)
            screen.blit(overlay, (0, 0))
        # 棋子(飞行动画期间仅隐藏落子位置棋子,占领位置的棋子正常显示)
        fly_skip = set()
        _fly_stone = None
        _fly_triangle = None
        if arrow is not None and arrow[2] is not None:
            fly_skip = {int(arrow[1])}             # 只隐藏落子位置(棋子在飞行中)
            f, t, prog = arrow
            p1 = self.cell_pos(*divmod(int(f), BOARD_SIZE))
            p2 = self.cell_pos(*divmod(int(t), BOARD_SIZE))
            sx = p1[0] + (p2[0] - p1[0]) * prog
            sy = p1[1] + (p2[1] - p1[1]) * prog
            dist = math.dist(p1, p2)
            if dist > 1:
                ux, uy = (p2[0] - p1[0]) / dist, (p2[1] - p1[1]) / dist
            else:
                ux, uy = 0, 0
            ox = ux * (self.stone_r + 3)
            oy = uy * (self.stone_r + 3)
            tip_x, tip_y = sx + ox, sy + oy
            base_x = sx - ux * (self.stone_r - 2)
            base_y = sy - uy * (self.stone_r - 2)
            px, py = -uy, ux
            _fly_triangle = ((tip_x, tip_y),
                             (base_x + 9 * px, base_y + 9 * py),
                             (base_x - 9 * px, base_y - 9 * py))
            tr, tc = divmod(int(t), BOARD_SIZE)
            _fly_stone = (sx, sy, game.board[tr, tc])
        last = game.record()[-1] if game.record() else None
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if r * BOARD_SIZE + c in fly_skip:
                    continue
                v = game.board[r, c]
                if v == EMPTY:
                    continue
                pos = self.cell_pos(r, c)
                is_last = (last == r * BOARD_SIZE + c)
                if v == BLACK:
                    surf = self.stone_b_h if is_last else self.stone_b
                else:
                    surf = self.stone_w_h if is_last else self.stone_w
                screen.blit(surf, (pos[0] - self.stone_r, pos[1] - self.stone_r))
                if show_numbers and num is not None and num[r, c] > 0:
                    txt = self.font_idx.render(
                        str(int(num[r, c])), True,
                        (255, 255, 255) if v == BLACK else COLOR_TEXT)
                    screen.blit(txt, (pos[0] - txt.get_width() // 2,
                                      pos[1] - txt.get_height() // 2))
        # 移子箭头:棋子飞行(0.5s)→ 蓝色箭头保持到下一步完成
        if arrow is not None:
            f, t, prog = arrow
            p1 = self.cell_pos(*divmod(int(f), BOARD_SIZE))
            p2 = self.cell_pos(*divmod(int(t), BOARD_SIZE))
            if prog is None:
                overlay = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
                self._draw_arrow(overlay, p1, p2, 1.0, 255)
                screen.blit(overlay, (0, 0))
            elif _fly_triangle is not None:
                # 三角头画在棋子下方(棋子在前方推着三角头走)
                pygame.draw.polygon(screen, COLOR_ARROW, _fly_triangle)
        # 飞行棋子(画在三角头上方)
        if _fly_stone is not None:
            fsx, fsy, fv = _fly_stone
            surf = self.stone_b if fv == BLACK else self.stone_w
            r0 = self.stone_r
            screen.blit(surf, (int(fsx) - r0, int(fsy) - r0))
            pygame.draw.circle(screen, COLOR_ARROW,
                               (int(fsx), int(fsy)), r0 + 3, 2)
        # 安置待定:悬停可达格的淡色预览箭头
        if hover_target is not None and game.pending >= 0:
            p1 = self.cell_pos(*divmod(int(game.pending), BOARD_SIZE))
            p2 = self.cell_pos(*hover_target)
            overlay = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
            self._draw_arrow(overlay, p1, p2, 1.0, 150)
            screen.blit(overlay, (0, 0))
        # 悬停
        if hover and game.pending < 0 and hover_rays is None:
            pos = self.cell_pos(*hover)
            pygame.draw.circle(screen, (*COLOR_TARGET, 110), pos, self.stone_r - 4, 2)
        # 辅助射线:被悬停棋子高亮环
        if hover_rays is not None:
            pos = self.cell_pos(*divmod(int(hover_rays[0]), BOARD_SIZE))
            pygame.draw.circle(screen, (*COLOR_RAY, 170), pos, self.stone_r + 2, 2)

    def screen_to_cell(self, x, y):
        c = round((x - MARGIN) / CELL)
        r = round((y - MARGIN) / CELL)
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            px, py = self.cell_pos(r, c)
            if abs(x - px) <= CELL * 0.45 and abs(y - py) <= CELL * 0.45:
                return (r, c)
        return None


class Panel:
    """右侧面板:模式/操作按钮 + 对局信息。"""

    def __init__(self, font, font_xs):
        self.font = font
        self.font_xs = font_xs
        self.buttons = {}                      # name -> (rect, label)
        y = PANEL_BUTTON_TOP
        for name in ("设定", "人执黑", "人执白", "人人对战", "机机观战", "新局", "悔棋",
                     "保存棋谱", "读取棋谱"):
            rect = pygame.Rect(BOARD_PX + 30, y, PANEL_W - 60, 44)
            self.buttons[name] = (rect, name)
            y += 56

    def hit(self, pos):
        for name, (rect, _) in self.buttons.items():
            if rect.collidepoint(pos):
                return name
        return None

    def draw(self, screen, mode, vmouse=None):
        pygame.draw.rect(screen, COLOR_PANEL,
                         (BOARD_PX, 0, PANEL_W, WIN_H))
        pygame.draw.line(screen, COLOR_WOOD_DARK, (BOARD_PX, 0), (BOARD_PX, WIN_H), 2)
        mouse = vmouse if vmouse is not None else pygame.mouse.get_pos()
        for name, (rect, _) in self.buttons.items():
            hover = rect.collidepoint(mouse)
            act = (name == MODE_NAMES[mode])
            color = COLOR_BTN_ACT if act else (COLOR_BTN_HOVER if hover else COLOR_BTN)
            pygame.draw.rect(screen, color, rect, border_radius=10)
            pygame.draw.rect(screen, COLOR_WOOD_DARK, rect, 2, border_radius=10)
            label = self.font.render(name, True, COLOR_TEXT)
            screen.blit(label, (rect.centerx - label.get_width() // 2,
                                rect.centery - label.get_height() // 2))
        # ---- 右下角署名(固定,快捷键提示已统一移至棋盘下方) ----
        made = self.font_xs.render("The GUI made by 硅基飙尘葆光.", True, (150, 128, 100))
        screen.blit(made, (BOARD_PX + PANEL_W - made.get_width() - 14,
                           WIN_H - made.get_height() - 10))


def _stone_numbers(game) -> np.ndarray:
    """每个格子上最后一次落子的手数(1 基),无子格为 0。返回 (15,15)。
    按历史逐手覆盖:占领/安置后该格显示新的手数。"""
    nums = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    for k, entry in enumerate(game.history, 1):
        nums[divmod(int(entry[0]), BOARD_SIZE)] = k
    return nums


def is_ai_turn(mode: int, game: ReverseGomoku) -> bool:
    if mode == 0:
        return game.current_player == WHITE
    if mode == 1:
        return game.current_player == BLACK
    if mode == 3:
        return True
    return False


class _MCTSEngine:
    """神经网络(或随机先验)MCTS 引擎。"""

    def __init__(self, mcts: MCTS):
        self.mcts = mcts

    @property
    def progress(self) -> int:
        """当前搜索已完成的模拟次数(实时计算量)。"""
        return self.mcts.progress

    @property
    def progress_total(self) -> int:
        return self.mcts.progress_total

    def choose(self, game) -> int | None:
        return self.mcts.search(game)


class _HeuristicEngine:
    """启发式 AI 引擎(heuristic.py:迭代加深极小化搜索 + 全局评估,无网络)。"""

    def __init__(self, depth: int = HEURISTIC_DEPTH_DEFAULT,
                 time_budget: float | None = HEURISTIC_TIME_BUDGET,
                 version: str = HEURISTIC_VERSION_DEFAULT,
                 use_vcf: bool = HEURISTIC_VCF_DEFAULT):
        from . import heuristic_versions as hv
        self.rng = np.random.default_rng()
        self.depth = depth
        self.time_budget = time_budget          # None = 不限时(按深度完整搜索)
        self.version = hv.normalize(version)
        self.use_vcf = bool(use_vcf) and hv.supports_vcf(self.version)
        self.progress = 0                       # 已搜索节点数(实时计算量)
        self._tt = None                         # 跨步复用置换表(中级/高级)
        if self.version == hv.VERSION_MID:
            from .tools import heuristic_v5
            self._tt = heuristic_v5._TT()
        elif self.version == hv.VERSION_ADVANCED:
            from .heuristic import _TT
            self._tt = _TT()

    def choose(self, game) -> int | None:
        from . import heuristic_versions as hv
        self.progress = 0
        return hv.choose(self.version, game.board, game.current_player,
                         game.pending, game.turn_count, game.white_turns,
                         game.config, self.rng, depth=self.depth,
                         time_budget=self.time_budget,
                         progress_cb=self._set_progress, tt=self._tt,
                         use_vcf=self.use_vcf)

    def _set_progress(self, nodes: int) -> None:
        self.progress = nodes


def _scan_models() -> list:
    """扫描可用神经网络模型(.pth):models/ 目录与当前目录,按路径去重。"""
    found = []
    for d in (paths.models_dir(), "."):
        try:
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".pth"):
                    p = os.path.normpath(os.path.join(str(d), f))
                    if p not in found:
                        found.append(p)
        except OSError:
            continue
    return found


# ---- GTP 坐标与 KataGo 引擎 ----
_GTP_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def _idx_to_gtp(idx: int) -> str:
    r, c = divmod(int(idx), 15)
    return f"{_GTP_LETTERS[c]}{15 - r}"


def _gtp_to_idx(coord: str) -> int:
    coord = coord.strip().upper()
    c = _GTP_LETTERS.index(coord[0])
    r = 15 - int(coord[1:])
    return r * 15 + c


def _load_engine_entries() -> list:
    """读取引擎配置列表:config/engines.local.json(本机覆盖)优先,
    其次 config/engines.json。返回 [] 表示读取失败。"""
    import json
    for name in ("engines.local.json", "engines.json"):
        p = paths.config_dir() / name
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [e for e in data if isinstance(e, dict)]
        except Exception:
            continue
    return []


def _path_for_config(path: str) -> str:
    """写回本地配置时,仓库内路径存为相对 app_base 的形式(可移植),
    仓库外路径(如 katago.exe)保持绝对路径。"""
    if not path:
        return path
    try:
        rel = Path(path).resolve().relative_to(paths.app_base().resolve())
        return rel.as_posix()
    except (ValueError, OSError):
        return path


def _probe_file(candidates: tuple) -> str:
    """在资源根目录(EXE 同级优先)下按候选相对路径探测存在的文件。"""
    for root in paths.roots():
        for rel in candidates:
            p = root / rel
            if p.is_file():
                return str(p)
    return ""


def _probe_weight() -> str:
    """在资源根目录下探测 KataGo 权重(*.bin,优先训练权重而非 init)。"""
    for root in paths.roots():
        for rel in ("models/katago", "katago/models", "katago"):
            d = root / rel
            if not d.is_dir():
                continue
            cands = sorted(d.glob("*.bin"), key=lambda f: f.stat().st_mtime,
                           reverse=True)
            trained = [f for f in cands if "init" not in f.name.lower()]
            picked = trained or cands
            if picked:
                return str(picked[0])
    return ""


def _kata_defaults() -> dict:
    """读取 KataGo 配置(engines.local.json 覆盖 engines.json)。
    auto_latest=true 时把权重自动解析到模型目录中最新的 antifive15-*.bin;
    相对路径按资源根目录探测解析;配置缺失/失效时在常见目录中自动探测。"""
    def _resolve(path: str) -> str:
        if path and not os.path.isabs(path):
            return paths.resolve(path)
        return path

    found = {"engine": "", "config": "", "weight": ""}
    for e in _load_engine_entries():
        if e.get("kind") == "heuristic":
            continue
        wp = _resolve(e.get("weight_path", ""))
        if e.get("auto_latest"):
            d = os.path.dirname(wp)
            try:
                # 排除随机初始化权重:解压工具不保留时间戳时 init 可能恰好
                # 比trained模型"更新"而被误选(见 experiments C1 实验)
                cands = sorted(
                    (os.path.join(d, f) for f in os.listdir(d)
                     if f.startswith("antifive15-") and f.endswith(".bin")
                     and not f.endswith("antifive15-init.bin")),
                    key=os.path.getmtime, reverse=True)
                if cands:
                    wp = cands[0]
            except OSError:
                pass
        found = {"engine": _resolve(e.get("engine_path", "")),
                 "config": _resolve(e.get("config_path", "")),
                 "weight": wp}
        break
    # 配置指向的文件不存在时,在 EXE 旁/仓库内常见位置自动探测
    if not (found["engine"] and os.path.exists(found["engine"])):
        found["engine"] = _probe_file(("katago/katago.exe", "katago.exe")) \
            or found["engine"]
    if not (found["config"] and os.path.exists(found["config"])):
        found["config"] = _probe_file(("config/gtp_test.cfg", "gtp_test.cfg")) \
            or found["config"]
    if not (found["weight"] and os.path.exists(found["weight"])):
        found["weight"] = _probe_weight() or found["weight"]
    return found


class _KataGoEngine:
    """KataGo 引擎(GTP 子进程,参考开源 Gomoku GUI 的 EngineProc 设计)。

    choose(game):增量同步历史到子进程后 genmove,返回着法 idx
    (认负/无子可走/引擎死亡返回 None)。历史缩短(undo)时全量重放。"""

    def __init__(self, engine_path: str, config_path: str, weight_path: str,
                 name: str = "KataGo", max_time: float | None = None,
                 max_visits: int | None = None, ai_temp: float = 0.0):
        self.name = name
        if not (engine_path and os.path.exists(engine_path)):
            raise ValueError(f"KataGo 可执行文件不存在: {engine_path}")
        if not (weight_path and os.path.exists(weight_path)):
            raise ValueError(f"KataGo 权重模型不存在: {weight_path}")
        if not (config_path and os.path.exists(config_path)):
            raise ValueError(f"KataGo 配置文件不存在: {config_path}")
        self.progress = 0                # 实时访问次数(由流式分析更新)
        self.progress_total = int(max_visits) if max_visits else 0
        self.progress_timeout = float(max_time) if max_time else 0.0
        self._analyze_ok = True          # 支持 kata-genmove_analyze 时启用实时分析
        self._awaiting_analyze_move = False
        cmd = [engine_path, "gtp", "-config", config_path, "-model", weight_path]
        overrides = []
        if max_time and max_time > 0:
            overrides.append(f"maxTime={max_time}")
        if max_visits and max_visits > 0:
            overrides.append(f"maxVisits={max_visits}")
        if ai_temp and ai_temp > 0:
            # AI 选点温度:按访问分布采样而非恒取最优,人机/机机对局均生效;
            # Early 同值保证全程恒温
            overrides.append(f"chosenMoveTemperature={ai_temp},"
                             f"chosenMoveTemperatureEarly={ai_temp}")
        if overrides:
            cmd += ["-override-config", ",".join(overrides)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            encoding="utf-8", errors="replace",
        )
        self._cmd_seq = 0
        self._resp_seq = 0
        self._resp_buf = {}
        self._resp_cond = threading.Condition()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self.send("boardsize 15 15")
        self.send("clear_board")
        self._synced_moves = 0
        self._synced_history = []
        self._need_clear = False
        self.last_error = None

    def _reader(self):
        while self.proc.poll() is None:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n")
            # 注意:kata-analyze 的应答是空结果(单独一行 "="),须一并计入,
            # 否则后续命令的响应序号会错位,genmove 等待超时
            if line.startswith("=") or line.startswith("?"):
                with self._resp_cond:
                    self._resp_seq += 1
                    self._resp_buf[self._resp_seq] = line
                    self._resp_cond.notify_all()
            elif line.startswith("info "):
                # 流式分析行:解析实时访问次数("info move D4 visits N ...")
                try:
                    parts = line.split()
                    j = parts.index("visits")
                    self.progress = int(parts[j + 1])
                except (ValueError, IndexError):
                    pass
            elif self._awaiting_analyze_move and line.startswith("play"):
                # kata-genmove_analyze 最终着法不带 "= " 前缀,而是裸行 "play D4"
                coord = line[len("play"):].strip()
                with self._resp_cond:
                    self._resp_seq += 1
                    self._resp_buf[self._resp_seq] = f"= {coord}" if coord else "= "
                    self._resp_cond.notify_all()
                self._awaiting_analyze_move = False

    def send(self, cmd: str):
        if self.proc.poll() is not None:
            return
        try:
            self._cmd_seq += 1
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass

    def sync_cmd(self, cmd: str, timeout: float = 120.0,
                 responses: int = 1) -> str:
        """发送命令并等待 responses 个应答(kata-genmove_analyze 有两个:
        立即应答 + 最终着法),返回最后一个应答。"""
        if self.proc.poll() is not None:
            return "? engine dead"
        self._cmd_seq += 1
        target = self._cmd_seq + responses - 1
        # 多应答命令(如 kata-genmove_analyze)把额外应答计入序号基线,
        # 保证后续命令的响应序号不漂移
        self._cmd_seq += responses - 1
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except Exception:
            return "? send failed"
        deadline = time.time() + timeout
        with self._resp_cond:
            while self._resp_seq < target:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return "? timeout"
                self._resp_cond.wait(min(remaining, 30.0))
            return self._resp_buf.get(target, "= ")

    def invalidate(self):
        """标记引擎历史失效:下次 choose 前强制 clear_board + 全量重放。"""
        self._need_clear = True

    def _sync_history(self, game) -> bool:
        """增量同步游戏历史到 KataGo 子进程。返回 True 表示同步成功。"""
        if self.proc.poll() is not None:
            return False
        hist = [(idx, player) for idx, player, *_ in game.history]
        n = len(hist)
        full = (self._need_clear
                or n < self._synced_moves
                or (self._synced_moves > 0
                    and hist[:self._synced_moves] != self._synced_history[:n]))
        self._need_clear = False
        if full:
            self.send("clear_board")
            for idx, player in hist:
                color = "B" if player == BLACK else "W"
                self.send(f"play {color} {_idx_to_gtp(idx)}")
        else:
            for idx, player in hist[self._synced_moves:]:
                color = "B" if player == BLACK else "W"
                self.send(f"play {color} {_idx_to_gtp(idx)}")
        self._synced_moves = n
        self._synced_history = list(hist)
        return True

    def choose(self, game) -> int | None:
        """返回 idx;None = 认负(pass/resign)。异常时 last_error 置为
        "timeout"/"engine"/"parse",供 GUI 区分"引擎故障"与"主动认负"。"""
        self.last_error = None
        if not self._sync_history(game):
            self.last_error = "engine"
            return None
        color = "B" if game.current_player == BLACK else "W"
        self.progress = 0
        # 带实时分析的 genmove(0.5s 间隔流式上报访问次数,搜索限制与普通
        # genmove 完全一致);引擎不支持时回退普通 genmove
        if self._analyze_ok:
            self._awaiting_analyze_move = True
            resp = self.sync_cmd(f"kata-genmove_analyze {color} 50",
                                 timeout=(self.progress_timeout or 0) + 120.0,
                                 responses=2)
            self._awaiting_analyze_move = False
            if resp.startswith("? "):
                self._analyze_ok = False
                resp = self.sync_cmd(f"genmove {color}")
        else:
            resp = self.sync_cmd(f"genmove {color}")
        if resp.startswith("? "):
            self.last_error = "timeout" if "timeout" in resp else "engine"
            return None
        coord = resp[2:].strip()
        if coord.lower() in ("resign", "pass"):
            return None
        try:
            idx = _gtp_to_idx(coord)
        except Exception:
            self.last_error = "parse"
            return None
        # 关键修复:fork 的 genmove 已把这手在引擎内部自落盘,必须记为已同步。
        # 否则下一回合增量重放会把它发一遍 play,而被 fork 忽略颜色参数、
        # 替对方执行成"占领手"接受 -> 引擎棋盘颜色反转/幻影子,
        # 从此在错误局面上行棋(experiments/exp_replay_capture_probe.py 实锤)。
        self._synced_moves += 1
        self._synced_history.append((idx, game.current_player))
        return idx

    def close(self):
        self._synced_moves = 0
        self._synced_history = []
        try:
            self.send("quit")
            time.sleep(0.05)
            self.proc.terminate()
        except Exception:
            pass


def _compute_text(eng) -> str:
    """AI 引擎实时计算量文本:MCTS 模拟数 / 启发式节点数 / KataGo 访问数。"""
    if eng is None:
        return ""
    p = int(getattr(eng, "progress", 0) or 0)
    total = int(getattr(eng, "progress_total", 0) or 0)
    if isinstance(eng, _KataGoEngine):
        return f"{p}/{total} 访问" if total else f"{p} 访问"
    if isinstance(eng, _HeuristicEngine):
        return f"{p / 1000:.1f}k 节点" if p >= 10000 else f"{p} 节点"
    return f"{p}/{total} 模拟" if total else f"{p} 模拟"


class _InputBox:
    """可聚焦文本输入框:支持光标移动、插入、退格、删除、Home/End。

    mode: "text"=任意可打印字符 / "int"=仅数字 / "float"=数字+一个小数点。
    """

    def __init__(self, rect, text="", mode="text", placeholder=""):
        self.rect = rect
        self.text = text
        self.caret = len(text)
        self.active = False
        self.mode = mode
        self.placeholder = placeholder

    def set_text(self, text):
        self.text = text
        self.caret = len(text)

    def handle_key(self, ev):
        k = ev.key
        if k == pygame.K_BACKSPACE:
            if self.caret > 0:
                self.text = self.text[:self.caret - 1] + self.text[self.caret:]
                self.caret -= 1
        elif k == pygame.K_DELETE:
            if self.caret < len(self.text):
                self.text = self.text[:self.caret] + self.text[self.caret + 1:]
        elif k == pygame.K_LEFT:
            self.caret = max(0, self.caret - 1)
        elif k == pygame.K_RIGHT:
            self.caret = min(len(self.text), self.caret + 1)
        elif k == pygame.K_HOME:
            self.caret = 0
        elif k == pygame.K_END:
            self.caret = len(self.text)
        else:
            ch = ev.unicode
            if not (ch and ch.isprintable()):
                return False
            if self.mode in ("int", "float"):
                ok = ch.isdigit() or (self.mode == "float" and ch == "."
                                      and "." not in self.text)
                if not ok:
                    return False
            self.text = self.text[:self.caret] + ch + self.text[self.caret:]
            self.caret += 1
        return True

    def draw(self, screen, font, vmouse=None):
        mouse = vmouse if vmouse is not None else pygame.mouse.get_pos()
        pygame.draw.rect(screen, (250, 246, 236), self.rect, border_radius=6)
        pygame.draw.rect(screen, COLOR_WOOD_DARK, self.rect, 2, border_radius=6)
        caret_px = font.size(self.text[:self.caret])[0] if self.text else 0
        clip = screen.get_clip()
        screen.set_clip(self.rect)
        px = self.rect.x + 8
        if self.text and caret_px + 10 > self.rect.w - 16:   # 长文本右滚,光标保持可见
            px -= (caret_px + 10) - (self.rect.w - 16)
        if self.text:
            t = font.render(self.text, True, COLOR_TEXT)
            screen.blit(t, (px, self.rect.centery - t.get_height() // 2))
            if self.active:
                pygame.draw.line(screen, COLOR_TEXT, (px + caret_px, self.rect.y + 7),
                                 (px + caret_px, self.rect.bottom - 7), 2)
        elif self.placeholder:
            t = font.render(self.placeholder, True, (160, 138, 110))
            screen.blit(t, (px, self.rect.centery - t.get_height() // 2))
        screen.set_clip(clip)
        if self.rect.collidepoint(mouse) and not self.active:   # 悬停描边提示可编辑
            pygame.draw.rect(screen, (200, 160, 110), self.rect, 2, border_radius=6)


class SettingsScreen:
    """UI 内设定面板(替代原 tkinter 启动对话框)。

    引擎选择(随机先验 MCTS / 神经网络模型 / 启发式 AI)、对局限时
    (不限时 / 仅 AI 读秒)、思考时间、模拟次数均在此界面内点击确定。
    """

    ROW_H = 24

    def __init__(self, app):
        self.app = app
        self.done = None               # "confirm" / "cancel"
        card_w, card_h = 780, 640
        self.card = pygame.Rect((WIN_W - card_w) // 2, (WIN_H - card_h) // 2,
                                card_w, card_h)
        # 工作副本(点「确定」后才写回 App)
        if isinstance(app.ai_black, _HeuristicEngine):
            self.kind = "heuristic"
        elif isinstance(app.ai_black, _KataGoEngine):
            self.kind = "kata"
        elif app.model_path:
            self.kind = "net"
        else:
            self.kind = "rand"
        self.model_path = app.model_path
        self.depth = app.heuristic_depth
        self.heuristic_version = getattr(app, "heuristic_version",
                                         HEURISTIC_VERSION_DEFAULT)
        self.heuristic_vcf = getattr(app, "heuristic_vcf",
                                     HEURISTIC_VCF_DEFAULT)
        self.limit_mode = app.limit_mode
        self.ai_time_budget = app.ai_time_budget
        self.mcts_sims = app.mcts_sims
        self.ai_temp = app.ai_temp
        self.kata_weight = app.kata_cfg.get("weight", "")
        self.kata_engine = app.kata_cfg.get("engine", "")
        self.kata_config = app.kata_cfg.get("config", "")
        self.models = _scan_models()
        self.model_scroll = 0
        self.focus_box = None          # 当前聚焦的 _InputBox 或 None
        self.path_box = self.depth_box = self.time_box = self.sims_box = None
        self.temp_box = None
        self.kata_weight_box = self.kata_engine_box = self.kata_config_box = None
        self._kata_weight_btn = None
        self._kata_weight_browse = self._kata_engine_browse = \
            self._kata_config_browse = None
        self._layout()

    # ---- 布局 ----
    def _layout(self):
        c = self.card
        pad = 26
        x0 = c.x + pad
        w = c.w - pad * 2
        self._x0, self._w = x0, w
        y = c.y + 22
        self._title_y = y
        y += 42
        self._engine_label_y = y
        y += 24
        eng = [("随机先验 MCTS", "rand")]
        if NET_AVAILABLE:
            eng.append(("神经网络模型", "net"))
        eng += [("启发式 AI", "heuristic"), ("KataGo", "kata")]
        bw = (w - 12 * (len(eng) - 1)) // len(eng)
        self._engine_radios = []
        for i, (label, val) in enumerate(eng):
            r = pygame.Rect(x0 + i * (bw + 12), y, bw, 36)
            self._engine_radios.append((r, label, val))
        y += 36 + 14
        self._model_list = self._model_input = self._model_load = None
        self._depth_rect = self._depth_minus = self._depth_plus = None
        self._kata_weight_btn = self._kata_engine_input = self._kata_config_input = None
        self._hver_radios: list = []
        self._hvcf_radios: list = []
        self._hver_label_y = self._hvcf_label_y = None
        if self.kind == "net":
            self._model_list = pygame.Rect(x0, y, w, 132)
            y += 132 + 10
            self._model_input = pygame.Rect(x0, y, w - 130, 32)
            self._model_load = pygame.Rect(x0 + w - 120, y, 120, 32)
            if self.path_box is None:
                self.path_box = _InputBox(self._model_input, self.model_path or "",
                                          mode="text", placeholder="输入 .pth 路径…")
            else:
                self.path_box.rect = self._model_input
            y += 32 + 14
        elif self.kind == "heuristic":
            self._depth_rect = pygame.Rect(x0, y, w, 34)
            self._depth_minus = pygame.Rect(x0 + w - 96, y, 32, 34)
            self._depth_plus = pygame.Rect(x0 + w - 34, y, 32, 34)
            self._depth_input = pygame.Rect(x0 + w - 96 - 10 - 56, y, 56, 34)
            if self.depth_box is None:
                self.depth_box = _InputBox(self._depth_input, str(self.depth), mode="int")
            else:
                self.depth_box.rect = self._depth_input
            y += 34 + 12
            # 引擎版本三档:初级(经典) / 中级(修复+开局库) / 高级(+蒸馏)
            self._hver_label_y = y
            y += 24
            ver = (("初级", "beginner"), ("中级", "mid"), ("高级", "advanced"))
            bwv3 = (w - 24) // 3
            self._hver_radios = [
                (pygame.Rect(x0 + i * (bwv3 + 12), y, bwv3, 34), lab, val)
                for i, (lab, val) in enumerate(ver)]
            y += 34 + 12
            # 强制杀链搜索(VCF):初级无此功能;默认开
            self._hvcf_label_y = y
            y += 24
            vcf = (("VCF 开(强制杀链)", True), ("VCF 关", False))
            bwv = (w - 12) // 2
            self._hvcf_radios = [
                (pygame.Rect(x0 + i * (bwv + 12), y, bwv, 34), lab, val)
                for i, (lab, val) in enumerate(vcf)]
            y += 34 + 16
        elif self.kind == "kata":
            # 权重模型(带「浏览」与「载入默认」按钮,后者从引擎配置解析最新模型)
            self._kata_weight_input = pygame.Rect(x0 + 118, y, w - 118 - 92 - 72, 32)
            self._kata_weight_browse = pygame.Rect(x0 + w - 82 - 72, y, 66, 32)
            self._kata_weight_btn = pygame.Rect(x0 + w - 82, y, 82, 32)
            self._kata_engine_input = pygame.Rect(x0 + 118, y + 40, w - 118 - 72, 32)
            self._kata_engine_browse = pygame.Rect(x0 + w - 66, y + 40, 66, 32)
            self._kata_config_input = pygame.Rect(x0 + 118, y + 80, w - 118 - 72, 32)
            self._kata_config_browse = pygame.Rect(x0 + w - 66, y + 80, 66, 32)
            if self.kata_weight_box is None:
                self.kata_weight_box = _InputBox(self._kata_weight_input,
                                                 self.kata_weight or "", mode="text",
                                                 placeholder="权重模型 .bin 路径…")
            else:
                self.kata_weight_box.rect = self._kata_weight_input
            if self.kata_engine_box is None:
                self.kata_engine_box = _InputBox(self._kata_engine_input,
                                                 self.kata_engine or "", mode="text",
                                                 placeholder="katago.exe 路径…")
            else:
                self.kata_engine_box.rect = self._kata_engine_input
            if self.kata_config_box is None:
                self.kata_config_box = _InputBox(self._kata_config_input,
                                                 self.kata_config or "", mode="text",
                                                 placeholder="GTP 配置 .cfg 路径…")
            else:
                self.kata_config_box.rect = self._kata_config_input
            y += 80 + 32 + 14
        self._limit_label_y = y
        y += 24
        lim = (("不限时", "none"), ("仅 AI 限时(读秒)", "ai"))
        bw2 = (w - 12) // 2
        self._limit_radios = []
        for i, (label, val) in enumerate(lim):
            r = pygame.Rect(x0 + i * (bw2 + 12), y, bw2, 36)
            self._limit_radios.append((r, label, val))
        y += 36 + 14
        self._time_rect = pygame.Rect(x0, y, w, 34)
        self._time_minus = pygame.Rect(x0 + w - 96, y, 32, 34)
        self._time_plus = pygame.Rect(x0 + w - 34, y, 32, 34)
        self._time_input = pygame.Rect(x0 + w - 96 - 10 - 140, y, 140, 34)
        if self.time_box is None:
            self.time_box = _InputBox(self._time_input, f"{self.ai_time_budget:.1f}",
                                      mode="float")
        else:
            self.time_box.rect = self._time_input
        y += 34 + 12
        self._sims_rect = pygame.Rect(x0, y, w, 34)
        self._sims_minus = pygame.Rect(x0 + w - 96, y, 32, 34)
        self._sims_plus = pygame.Rect(x0 + w - 34, y, 32, 34)
        self._sims_input = pygame.Rect(x0 + w - 96 - 10 - 140, y, 140, 34)
        if self.sims_box is None:
            self.sims_box = _InputBox(self._sims_input, str(self.mcts_sims), mode="int")
        else:
            self.sims_box.rect = self._sims_input
        y += 34 + 12
        # AI 温度(仅 KataGo 生效,人机/机机对局均有效):0=始终最优着,>0 按访问分布采样选点
        self._temp_rect = pygame.Rect(x0, y, w, 34)
        self._temp_minus = pygame.Rect(x0 + w - 96, y, 32, 34)
        self._temp_plus = pygame.Rect(x0 + w - 34, y, 32, 34)
        self._temp_input = pygame.Rect(x0 + w - 96 - 10 - 140, y, 140, 34)
        if self.temp_box is None:
            self.temp_box = _InputBox(self._temp_input, f"{self.ai_temp:g}",
                                      mode="float")
        else:
            self.temp_box.rect = self._temp_input
        y += 34 + 22
        bw3 = 180
        self._cancel_rect = pygame.Rect(c.centerx - bw3 - 10, y, bw3, 42)
        self._confirm_rect = pygame.Rect(c.centerx + 10, y, bw3, 42)

    # ---- 交互 ----
    def handle(self, ev):
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                if self.focus_box is not None:   # 聚焦输入框时 Esc 仅取消编辑
                    self._blur_box()
                    return None
                return "cancel"
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB):
                if self.focus_box is not None:   # 聚焦输入框时 Enter/Tab 仅提交失焦
                    self._blur_box()
                    return None
                if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return "confirm"
                return None
            if self.focus_box is not None:
                self.focus_box.handle_key(ev)
                return None
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1:
                return self._click(ev.pos)
            elif ev.button == 4:                       # 滚轮上
                if self._model_list and self._model_list.collidepoint(ev.pos):
                    self.model_scroll = max(0, self.model_scroll - 1)
            elif ev.button == 5:                       # 滚轮下
                if self._model_list and self._model_list.collidepoint(ev.pos):
                    rows = self._model_list.h // self.ROW_H
                    mx = max(0, len(self.models) - rows)
                    self.model_scroll = min(mx, self.model_scroll + 1)
        return None

    def _click(self, pos):
        # 先提交并失焦当前输入框
        self._blur_box()
        for b in (self.path_box, self.depth_box, self.time_box, self.sims_box,
                  self.temp_box,
                  self.kata_weight_box, self.kata_engine_box, self.kata_config_box):
            if b is not None and b.rect.collidepoint(pos):
                b.active = True
                self.focus_box = b
                return None
        for r, label, val in self._engine_radios:
            if r.collidepoint(pos):
                if val != self.kind:
                    self.kind = val
                    self._layout()
                    if val == "net" and self.model_path is None and self.models:
                        self.model_path = self.models[0]
                        self.path_box.set_text(self.model_path)
                return None
        for r, label, val in self._hver_radios:
            if r.collidepoint(pos):
                self.heuristic_version = val
                if val == "beginner":
                    self.heuristic_vcf = False   # 初级无 VCF
                return None
        if self.heuristic_version != "beginner":
            for r, label, val in self._hvcf_radios:
                if r.collidepoint(pos):
                    self.heuristic_vcf = bool(val)
                    return None
        for r, label, val in self._limit_radios:
            if r.collidepoint(pos):
                self.limit_mode = val
                return None
        if self._model_list and self._model_list.collidepoint(pos):
            idx = self.model_scroll + (pos[1] - (self._model_list.y + 4)) // self.ROW_H
            if 0 <= idx < len(self.models):
                self.model_path = self.models[idx]
                self.path_box.set_text(self.model_path)
            return None
        if self._model_load and self._model_load.collidepoint(pos):
            self.model_path = self.path_box.text or None
            return None
        if self._kata_weight_btn and self._kata_weight_btn.collidepoint(pos):
            d = _kata_defaults()
            self.kata_weight = d["weight"]
            self.kata_engine = d["engine"]
            self.kata_config = d["config"]
            self.kata_weight_box.set_text(self.kata_weight)
            self.kata_engine_box.set_text(self.kata_engine)
            self.kata_config_box.set_text(self.kata_config)
            return None
        if self._kata_weight_browse and self._kata_weight_browse.collidepoint(pos):
            p = self._browse_file("选择 KataGo 权重模型", self.kata_weight,
                                  [("KataGo 权重", "*.bin"), ("所有文件", "*.*")])
            if p:
                self.kata_weight = p
                self.kata_weight_box.set_text(p)
            return None
        if self._kata_engine_browse and self._kata_engine_browse.collidepoint(pos):
            p = self._browse_file("选择 katago 可执行文件", self.kata_engine,
                                  [("可执行文件", "*.exe"), ("所有文件", "*.*")])
            if p:
                self.kata_engine = p
                self.kata_engine_box.set_text(p)
            return None
        if self._kata_config_browse and self._kata_config_browse.collidepoint(pos):
            p = self._browse_file("选择 GTP 配置", self.kata_config,
                                  [("GTP 配置", "*.cfg"), ("所有文件", "*.*")])
            if p:
                self.kata_config = p
                self.kata_config_box.set_text(p)
            return None
        if self._depth_minus and self._depth_minus.collidepoint(pos):
            self.depth = max(1, self.depth - 1)
            self.depth_box.set_text(str(self.depth))
            return None
        if self._depth_plus and self._depth_plus.collidepoint(pos):
            self.depth = min(8, self.depth + 1)
            self.depth_box.set_text(str(self.depth))
            return None
        if self._time_minus.collidepoint(pos):
            self.ai_time_budget = max(1.0, round(self.ai_time_budget - 1.0, 1))
            self.time_box.set_text(f"{self.ai_time_budget:.1f}")
            return None
        if self._time_plus.collidepoint(pos):
            self.ai_time_budget = min(600.0, round(self.ai_time_budget + 1.0, 1))
            self.time_box.set_text(f"{self.ai_time_budget:.1f}")
            return None
        if self._sims_minus.collidepoint(pos):
            self.mcts_sims = max(100, self.mcts_sims - 100)
            self.sims_box.set_text(str(self.mcts_sims))
            return None
        if self._sims_plus.collidepoint(pos):
            self.mcts_sims = min(1000000, self.mcts_sims + 100)
            self.sims_box.set_text(str(self.mcts_sims))
            return None
        if self._temp_minus.collidepoint(pos):
            self.ai_temp = max(0.0, round(self.ai_temp - 0.1, 2))
            self.temp_box.set_text(f"{self.ai_temp:g}")
            return None
        if self._temp_plus.collidepoint(pos):
            self.ai_temp = min(2.0, round(self.ai_temp + 0.1, 2))
            self.temp_box.set_text(f"{self.ai_temp:g}")
            return None
        if self._confirm_rect.collidepoint(pos):
            return "confirm"
        if self._cancel_rect.collidepoint(pos):
            return "cancel"
        return None

    def _browse_file(self, title, current, filetypes):
        """弹出文件选择对话框(tkinter,与棋谱存取共用惰性根窗口)。"""
        try:
            fd = self.app._file_dialogs()
            init = os.path.dirname(current) if current else ""
            if not (init and os.path.isdir(init)):
                init = str(paths.app_base())
            return fd.askopenfilename(title=title, initialdir=init,
                                      filetypes=filetypes) or ""
        except Exception:
            return ""

    @staticmethod
    def _clamp_num(text, fallback, lo, hi, nd):
        """解析文本为数字并钳制到 [lo, hi];nd=0 取整,否则保留 nd 位小数。"""
        try:
            v = float(text)
        except (TypeError, ValueError):
            v = fallback
        v = min(hi, max(lo, v))
        return int(v) if nd == 0 else round(v, nd)

    def _blur_box(self):
        """提交并失焦当前输入框(把编辑结果解析回对应数值)。"""
        b = self.focus_box
        self.focus_box = None
        if b is None:
            return
        b.active = False
        if b is self.time_box:
            self.ai_time_budget = self._clamp_num(b.text, self.ai_time_budget,
                                                  1.0, 600.0, 1)
            b.set_text(f"{self.ai_time_budget:.1f}")
        elif b is self.sims_box:
            self.mcts_sims = self._clamp_num(b.text, float(self.mcts_sims),
                                             100, 1000000, 0)
            b.set_text(str(self.mcts_sims))
        elif b is self.depth_box:
            self.depth = self._clamp_num(b.text, float(self.depth), 1, 8, 0)
            b.set_text(str(self.depth))
        elif b is self.temp_box:
            self.ai_temp = self._clamp_num(b.text, self.ai_temp,
                                                 0.0, 2.0, 2)
            b.set_text(f"{self.ai_temp:g}")
        elif b is self.path_box:
            self.model_path = b.text or None
        elif b is self.kata_weight_box:
            self.kata_weight = b.text or ""
        elif b is self.kata_engine_box:
            self.kata_engine = b.text or ""
        elif b is self.kata_config_box:
            self.kata_config = b.text or ""

    # ---- 绘制 ----
    def draw(self, screen):
        font = self.app.font
        font_sm = self.app.font_sm
        font_xs = self.app.font_xs
        mouse = getattr(self, '_vmouse', None) or pygame.mouse.get_pos()
        c = self.card
        pygame.draw.rect(screen, COLOR_PANEL, c, border_radius=14)
        pygame.draw.rect(screen, COLOR_WOOD_DARK, c, 3, border_radius=14)
        title = self.app.font_big.render("设定", True, COLOR_TEXT)
        screen.blit(title, (c.centerx - title.get_width() // 2, self._title_y))
        # 引擎
        self._draw_section(screen, "引擎选择(黑白双方共用)", self._engine_label_y)
        for r, label, val in self._engine_radios:
            self._draw_radio(screen, font, r, label, val == self.kind)
        # 条件区域
        if self.kind == "net":
            self._draw_model_list(screen, font, font_xs)
        elif self.kind == "heuristic":
            self._draw_numrow(screen, font_sm, "启发式搜索深度", self.depth_box,
                              self._depth_rect, self._depth_minus, self._depth_plus)
            self._draw_section(screen, "引擎版本(初级经典 / 中级强化 / 高级+蒸馏)",
                               self._hver_label_y)
            for r, label, val in self._hver_radios:
                self._draw_radio(screen, font, r, label, val == self.heuristic_version)
            beginner = self.heuristic_version == "beginner"
            self._draw_section(screen, "强制杀链搜索 VCF(中级/高级生效)",
                               self._hvcf_label_y)
            for r, label, val in self._hvcf_radios:
                self._draw_radio(screen, font, r, label,
                                 (not beginner) and bool(val) == self.heuristic_vcf,
                                 disabled=beginner)
        elif self.kind == "kata":
            self._draw_pathrow(screen, font_sm, "权重模型(.bin)", self.kata_weight_box,
                               self._kata_weight_input,
                               btns=[(self._kata_weight_browse, "浏览"),
                                     (self._kata_weight_btn, "载入默认")])
            self._draw_pathrow(screen, font_sm, "KataGo 可执行", self.kata_engine_box,
                               self._kata_engine_input,
                               btns=[(self._kata_engine_browse, "浏览")])
            self._draw_pathrow(screen, font_sm, "GTP 配置", self.kata_config_box,
                               self._kata_config_input,
                               btns=[(self._kata_config_browse, "浏览")])
        # 限时
        self._draw_section(screen, "对局限时", self._limit_label_y)
        for r, label, val in self._limit_radios:
            self._draw_radio(screen, font, r, label, val == self.limit_mode)
        self._draw_numrow(screen, font_sm, "AI 每步思考时间(秒)", self.time_box,
                          self._time_rect, self._time_minus, self._time_plus)
        self._draw_numrow(screen, font_sm, "神经网络模拟次数(决策阈值)", self.sims_box,
                          self._sims_rect, self._sims_minus, self._sims_plus)
        self._draw_numrow(screen, font_sm, "AI 温度(KataGo,0=最优)",
                          self.temp_box,
                          self._temp_rect, self._temp_minus, self._temp_plus)
        # 按钮
        for r, label in ((self._cancel_rect, "取消"), (self._confirm_rect, "确定")):
            hover = r.collidepoint(mouse)
            color = COLOR_BTN_HOVER if hover else COLOR_BTN
            pygame.draw.rect(screen, color, r, border_radius=8)
            pygame.draw.rect(screen, COLOR_WOOD_DARK, r, 2, border_radius=8)
            txt = font.render(label, True, COLOR_TEXT)
            screen.blit(txt, (r.centerx - txt.get_width() // 2,
                              r.centery - txt.get_height() // 2))
        hint = "Enter 确定 · Esc 取消 · 右上角时钟点击可随时开关读秒"
        t = font_xs.render(hint, True, (140, 120, 90))
        screen.blit(t, (c.centerx - t.get_width() // 2, c.bottom - 26))

    def _draw_section(self, screen, label, y):
        t = self.app.font_sm.render(label, True, COLOR_WOOD_DARK)
        screen.blit(t, (self._x0, y))

    def _draw_radio(self, screen, font, r, label, selected, disabled=False):
        if disabled:
            color = (216, 206, 190)
        else:
            color = COLOR_BTN_ACT if selected else COLOR_BTN
        pygame.draw.rect(screen, color, r, border_radius=8)
        pygame.draw.rect(screen, COLOR_WOOD_DARK, r, 2, border_radius=8)
        t = font.render(label, True, (150, 140, 125) if disabled else COLOR_TEXT)
        screen.blit(t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2))

    def _draw_numrow(self, screen, font, label, box, rect, minus, plus):
        mouse = getattr(self, '_vmouse', None) or pygame.mouse.get_pos()
        t = font.render(label, True, COLOR_TEXT)
        screen.blit(t, (rect.x + 6, rect.centery - t.get_height() // 2))
        box.draw(screen, font)                    # 可编辑数字输入框
        for r, s in ((minus, "−"), (plus, "+")):
            hover = r.collidepoint(mouse)
            color = COLOR_BTN_HOVER if hover else COLOR_BTN
            pygame.draw.rect(screen, color, r, border_radius=6)
            pygame.draw.rect(screen, COLOR_WOOD_DARK, r, 2, border_radius=6)
            t = font.render(s, True, COLOR_TEXT)
            screen.blit(t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2))

    def _draw_pathrow(self, screen, font, label, box, rect, btns=()):
        """标签 + 文本输入框(可选右侧按钮组)的一行,供 KataGo 路径设定。"""
        mouse = getattr(self, '_vmouse', None) or pygame.mouse.get_pos()
        t = font.render(label, True, COLOR_TEXT)
        screen.blit(t, (rect.x - 116, rect.centery - t.get_height() // 2))
        box.draw(screen, font, mouse)
        for btn, btn_label in btns:
            if btn is None:
                continue
            hover = btn.collidepoint(mouse)
            color = COLOR_BTN_HOVER if hover else COLOR_BTN
            pygame.draw.rect(screen, color, btn, border_radius=6)
            pygame.draw.rect(screen, COLOR_WOOD_DARK, btn, 2, border_radius=6)
            t2 = font.render(btn_label, True, COLOR_TEXT)
            screen.blit(t2, (btn.centerx - t2.get_width() // 2,
                             btn.centery - t2.get_height() // 2))

    def _draw_model_list(self, screen, font, font_xs):
        r = self._model_list
        pygame.draw.rect(screen, (250, 246, 236), r, border_radius=6)
        pygame.draw.rect(screen, COLOR_WOOD_DARK, r, 2, border_radius=6)
        clip = screen.get_clip()
        screen.set_clip(r)
        if not self.models:
            t = font.render("未找到 .pth 模型,可在下方输入路径", True, (150, 120, 90))
            screen.blit(t, (r.x + 10, r.y + 6))
        else:
            mouse = getattr(self, '_vmouse', None) or pygame.mouse.get_pos()
            for i, p in enumerate(self.models):
                row = pygame.Rect(r.x + 2, r.y + 4 + (i - self.model_scroll) * self.ROW_H,
                                  r.w - 4, self.ROW_H)
                if row.bottom < r.y or row.top > r.bottom:
                    continue
                if p == self.model_path:
                    pygame.draw.rect(screen, COLOR_BTN_ACT, row, border_radius=4)
                elif row.collidepoint(mouse):
                    pygame.draw.rect(screen, COLOR_BTN_HOVER, row, border_radius=4)
                t = font_xs.render(os.path.basename(p), True, COLOR_TEXT)
                screen.blit(t, (row.x + 8, row.centery - t.get_height() // 2))
        screen.set_clip(clip)
        # 路径输入框(可编辑)
        self.path_box.draw(screen, font_xs)
        lr = self._model_load
        color = COLOR_BTN_HOVER if lr.collidepoint(getattr(self, '_vmouse', None) or pygame.mouse.get_pos()) else COLOR_BTN
        pygame.draw.rect(screen, color, lr, border_radius=6)
        pygame.draw.rect(screen, COLOR_WOOD_DARK, lr, 2, border_radius=6)
        t = font.render("载入路径", True, COLOR_TEXT)
        screen.blit(t, (lr.centerx - t.get_width() // 2, lr.centery - t.get_height() // 2))


class App:
    def __init__(self, device, args):
        # Windows 高 DPI 感知:让 display.Info() 返回物理分辨率,避免缩放屏上
        # 系统虚拟化分辨率导致自动缩放启发式误判(须在 SDL 视频初始化前设置)
        if os.name == "nt":
            os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitor")
        pygame.init()
        pygame.display.set_caption("逆五子棋 Antifive")
        # 按屏幕大小自动缩放,预留 80px 边距(任务栏+标题栏)
        info = pygame.display.Info()
        sw, sh = info.current_w, info.current_h
        self.scale = min((sw - 80) / WIN_W, (sh - 80) / WIN_H, 1.0)
        self.scale = max(self.scale, 0.3)
        aw, ah = int(WIN_W * self.scale), int(WIN_H * self.scale)
        self.screen = pygame.display.set_mode((aw, ah), pygame.RESIZABLE)
        self.virtual = pygame.Surface((WIN_W, WIN_H))
        # 用实际 surface 尺寸重算 scale(修正 DPI 缩放导致的偏差)
        actual_w, actual_h = self.screen.get_size()
        self.scale = min(actual_w / WIN_W, actual_h / WIN_H, 1.0)
        if os.name == "nt":                       # 启动即请求窗口焦点,避免键盘事件丢失
            try:
                import ctypes
                hwnd = pygame.display.get_wm_info().get("window")
                if hwnd:
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception:
                pass
        self.font = _load_font(22)
        self.font_sm = _load_font(17)
        self.font_big = _load_font(40)
        self.font_xs = _load_font(14)
        self.board = Board()
        self.panel = Panel(self.font_sm, self.font_xs)
        self.game = ReverseGomoku()
        self.mode = 0
        self.show_danger = False
        self.show_numbers = False        # 棋子上显示落子手数
        self.show_arrows = True          # 移子箭头指示(占领→安置)
        self.show_rays = True            # 辅助射线:悬停棋子临时显示其可移动位置
        self.arrow_anim = None           # (占领idx, 安置idx, 飞行起始时间|None=静态保持)
        self.toggle_btns: list = []      # 底部开关按钮 (rect, label, fn)
        self.ai_lock = threading.Lock()
        self.ai_result = None
        self.ai_thinking = False
        self._ai_retries = 0          # 引擎超时连续重试计数(防无限重试)
        self.epoch = 0                # 局面代数:新局/悔棋/落子后自增,丢弃过期 AI 结果
        self.hover = None
        self.result_shown_at = 0.0
        self.show_result_overlay = False  # 终局大字蒙版:对局结束显示并自动淡出,载入棋谱不显示
        self.status = ""              # 状态提示(保存/读取棋谱结果)
        self.status_until = 0.0
        self._shortcut_prev = None    # 键盘轮询兜底:上一帧按键状态
        self._handled_keys = set()    # 键盘轮询兜底:本帧已被 KEYDOWN/TEXTINPUT 处理过的键
        self.device = device
        self._smoke_frames = int(getattr(args, "smoke", 0) or 0)  # 自检帧数(0=正常)
        self.mcts_sims = args.sims              # 神经网络"决策默认阈值"(默认模拟次数)
        self.mcts_c_puct = args.c_puct
        self.ai_temp = getattr(args, "ai_temp", 0.0)  # KataGo AI 温度(人机/机机均生效)
        self.limit_mode = args.limit            # 对局限时模式: "none"=不限时 / "ai"=仅 AI 限时读秒
        self.ai_time_budget = args.ai_time      # 限时模式下 AI 每步思考时间上限(秒)
        self.ai_deadline = None                 # 当前 AI 思考截止时间(读秒显示用)
        self._tk_root = None          # tkinter 根窗口(仅保存/读取棋谱文件对话框,惰性创建)
        self._filedialog = None
        self.font_clock = _load_font(32)          # 读秒时钟数字字体
        # ---- 音效(合成,无外部资源;mixer 不可用则静音) ----
        self.sound_enabled = True
        self.snd_place = _make_sound(880.0, 0.08, vol=0.5, decay=12.0)   # 落子"嗒"
        self.snd_move = _make_sound(440.0, 0.12, vol=0.55, decay=9.0)    # 移子"咚"
        # ---- 引擎 ----
        self.ai_black = None          # 引擎对象(choose(game) -> 着法 idx)
        self.ai_white = None
        self.engine_desc = "随机先验 MCTS"
        self.model_path = None        # 当前神经网络模型路径(None=随机先验)
        self.heuristic_depth = HEURISTIC_DEPTH_DEFAULT  # 当前启发式 AI 搜索深度
        self.heuristic_version = getattr(args, "heuristic_version",
                                         HEURISTIC_VERSION_DEFAULT)
        self.heuristic_vcf = not getattr(args, "no_vcf", False)  # 中级/高级 VCF 开关
        self.kata_cfg = _kata_defaults()  # KataGo 引擎(可执行/配置/权重)路径
        self.settings = None          # 打开的设定面板(SettingsScreen 或 None)
        self._engine_from_cli = False
        if args.engine == "heuristic":
            self.set_engine("heuristic", depth=args.heuristic_depth,
                            version=self.heuristic_version,
                            use_vcf=self.heuristic_vcf)
            self._engine_from_cli = True
        elif args.engine == "katago":
            try:
                self.set_engine("kata")
                self._engine_from_cli = True
            except Exception as e:
                print(f"!! KataGo 启动失败: {e},回退到随机先验 MCTS")
                self.set_engine("mcts", None)
        elif args.model and os.path.exists(args.model):
            self.set_engine("mcts", args.model)
            self._engine_from_cli = True
        else:
            if args.model:
                print(f"!! 指定模型不存在: {args.model},可在「设定」中选择")
            self.set_engine("mcts", None)   # 默认随机先验,可在 UI 内「设定」更改
        # ---- 复盘状态 ----
        self.review = False           # 复盘模式(终局/载入棋谱后进入)
        self.record_moves: list = []  # 原谱主变化(完整着法序列)
        self.moves: list = []         # 当前变化线(试下会截断替换尾部)
        self.pos = 0                  # 当前位于 moves 的第几手(0=开局)
        self.review_divergence = None # 试下分支点(原谱回到该手之前)
        self.review_result = "未完"   # 原谱结果文本
        self.review_btns: list = []   # 底部按钮 (rect, label, fn)
        self.review_axis = None       # 手数轴命中区 (x0, x1, y0, y1),未绘制时为 None

    # ---- 引擎 ----
    def set_engine(self, kind: str, path: str = None, depth: int = None,
                   version: str = None, use_vcf: bool = None) -> None:
        """切换 AI 引擎:("mcts", path)/("mcts", None=随机先验)/
        ("heuristic", depth, version, use_vcf)/("kata", None=使用 self.kata_cfg 配置)。
        对局限时按 self.limit_mode / self.ai_time_budget / self.mcts_sims 配置。"""
        # 关闭旧 KataGo 子进程(黑白共用同一子进程,避免重复关闭)
        old = self.ai_black
        if old is not None and isinstance(old, _KataGoEngine):
            old.close()
        with self.ai_lock:
            self.ai_result = None
            self.ai_thinking = False
        self.epoch += 1
        self.ai_deadline = None
        ai_tb = self.ai_time_budget if self.limit_mode == "ai" else None
        if kind == "heuristic":
            from . import heuristic_versions as hv
            if depth is None:
                depth = HEURISTIC_DEPTH_DEFAULT
            if version is None:
                version = self.heuristic_version
            if use_vcf is None:
                use_vcf = self.heuristic_vcf
            self.heuristic_depth = depth
            self.heuristic_version = hv.normalize(version)
            self.heuristic_vcf = bool(use_vcf) and hv.supports_vcf(
                self.heuristic_version)
            self.model_path = None
            self.ai_black = _HeuristicEngine(depth, ai_tb,
                                             self.heuristic_version,
                                             self.heuristic_vcf)
            self.ai_white = _HeuristicEngine(depth, ai_tb,
                                             self.heuristic_version,
                                             self.heuristic_vcf)
            vtxt = hv.label(self.heuristic_version)
            if hv.supports_vcf(self.heuristic_version):
                vtxt += "·VCF开" if self.heuristic_vcf else "·VCF关"
            self.engine_desc = f"启发式 AI({vtxt}, 深度{depth})"
            return
        if kind == "kata":
            self.model_path = None
            self.heuristic_depth = HEURISTIC_DEPTH_DEFAULT
            k = self.kata_cfg
            weight_name = os.path.basename(k.get("weight", ""))
            self.engine_desc = f"KataGo({weight_name})"
            if "init" in weight_name.lower():
                self.set_status("警告:KataGo 加载的是随机初始化权重(init),"
                                "棋力无意义,请更换训练模型!", 30.0)
            eng = _KataGoEngine(k.get("engine", ""), k.get("config", ""),
                                k.get("weight", ""), max_time=ai_tb,
                                max_visits=self.mcts_sims,
                                ai_temp=self.ai_temp)
            self.ai_black = eng
            self.ai_white = eng          # 同一时刻仅一方在走,共用子进程
            return
        net = None
        if path and not NET_AVAILABLE:
            print("!! 未安装 torch,神经网络引擎不可用,回退到随机先验 MCTS")
            self.set_status("未安装 torch,神经网络引擎不可用")
            path = None
        if path:
            net = PolicyValueNet.load(path, self.device)
            self.engine_desc = os.path.basename(path)
        else:
            self.engine_desc = "随机先验 MCTS"
        self.model_path = path
        mcts_tb = ai_tb if ai_tb is not None else 0.0
        self.ai_black = _MCTSEngine(MCTS(net, self.device, sims=self.mcts_sims,
                                         c_puct=self.mcts_c_puct,
                                         time_budget=mcts_tb))
        self.ai_white = _MCTSEngine(MCTS(net, self.device, sims=self.mcts_sims,
                                         c_puct=self.mcts_c_puct,
                                         time_budget=mcts_tb))

    def _rebuild_engines(self, config) -> None:
        """载入不同规则配置的棋谱后,按新配置重建引擎(启发式/KataGo 无需重建)。"""
        if isinstance(self.ai_black, (_HeuristicEngine, _KataGoEngine)):
            return
        m = self.ai_black.mcts
        self.ai_black = _MCTSEngine(MCTS(m.network, self.device, sims=m.sims,
                                         c_puct=m.c_puct, config=config,
                                         time_budget=m.time_budget))
        self.ai_white = _MCTSEngine(MCTS(m.network, self.device, sims=m.sims,
                                         c_puct=m.c_puct, config=config,
                                         time_budget=m.time_budget))

    def _open_settings(self) -> None:
        """打开 UI 内设定面板(引擎选择 + 对局设置)。"""
        if self.settings is None:
            self.settings = SettingsScreen(self)

    def _apply_settings(self, s: SettingsScreen) -> None:
        """把设定面板的工作副本写回并重建引擎。"""
        try:
            self.limit_mode = s.limit_mode
            self.ai_time_budget = max(1.0, float(s.ai_time_budget))
            self.mcts_sims = max(100, int(s.mcts_sims))
            self.ai_temp = max(0.0, min(2.0, float(s.ai_temp)))
            if s.kind == "heuristic":
                self.set_engine("heuristic", depth=int(s.depth),
                                version=s.heuristic_version,
                                use_vcf=s.heuristic_vcf)
            elif s.kind == "net":
                self.set_engine("mcts", s.model_path)
            elif s.kind == "kata":
                self.kata_cfg = {
                    "engine": s.kata_engine_box.text.strip() if s.kata_engine_box else "",
                    "config": s.kata_config_box.text.strip() if s.kata_config_box else "",
                    "weight": s.kata_weight_box.text.strip() if s.kata_weight_box else "",
                }
                self.set_engine("kata")
                self._save_kata_config()
            else:
                self.set_engine("mcts", None)
            self.ai_deadline = None
            limit_txt = "仅 AI 限时读秒" if self.limit_mode == "ai" else "不限时"
            self.set_status(f"已应用: {self.engine_desc} · 限时: {limit_txt}")
        except Exception as e:
            self.set_status(f"设定失败: {e}")

    def _save_kata_config(self) -> None:
        """把当前 KataGo 路径持久化到 config/engines.local.json(本机覆盖,
        已被 gitignore;下次启动 _kata_defaults 优先读取)。"""
        import json
        entry = [{
            "name": "逆五KataGo",
            "engine_path": _path_for_config(self.kata_cfg.get("engine", "")),
            "weight_path": _path_for_config(self.kata_cfg.get("weight", "")),
            "config_path": _path_for_config(self.kata_cfg.get("config", "")),
            "auto_latest": False,
        }]
        try:
            p = paths.config_dir() / "engines.local.json"
            p.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
        except OSError:
            pass

    def _reapply_engine(self) -> None:
        """按当前引擎种类重建引擎(用于切换限时模式等仅改配置的场景)。"""
        if isinstance(self.ai_black, _HeuristicEngine):
            self.set_engine("heuristic", depth=self.heuristic_depth,
                            version=self.heuristic_version,
                            use_vcf=self.heuristic_vcf)
        elif isinstance(self.ai_black, _KataGoEngine):
            # 读秒切换对 KataGo 生效(经 -override-config maxTime),思考中不打断
            if not self.ai_thinking:
                self.set_engine("kata")
        else:
            self.set_engine("mcts", self.model_path)

    def _toggle_limit_mode(self) -> None:
        """点击右上角时钟:随时开关读秒(不限时 <-> 仅 AI 限时)。"""
        self.limit_mode = "none" if self.limit_mode == "ai" else "ai"
        self.ai_deadline = None
        self._reapply_engine()
        limit_txt = "仅 AI 限时读秒" if self.limit_mode == "ai" else "不限时"
        self.set_status(f"读秒已{'开启' if self.limit_mode == 'ai' else '关闭'} · {limit_txt}")

    # ---- 棋谱 ----
    def set_status(self, msg: str, dur: float = 6.0) -> None:
        self.status = msg
        self.status_until = time.time() + dur

    def _tk(self):
        if self._tk_root is None:
            import tkinter as tk
            from tkinter import filedialog
            self._tk_root = tk.Tk()
            self._tk_root.withdraw()
            self._filedialog = filedialog
        return self._tk_root

    def _file_dialogs(self):
        self._tk()
        return self._filedialog

    def save_game(self):
        if self.game.pending >= 0 and not self.game.game_over:
            self.set_status("移子待定:请先完成安置再保存")
            return
        fd = self._file_dialogs()
        path = fd.asksaveasfilename(
            title="保存棋谱", defaultextension=".afg",
            initialfile="对局.afg",
            filetypes=[("逆五子棋棋谱", "*.afg"), ("文本文件", "*.txt"),
                       ("所有文件", "*.*")])
        if not path:
            return
        try:
            moves = self.record_moves if self.record_moves else None
            result = record.save_game(path, self.game, moves=moves)
            self.set_status(f"棋谱已保存: {os.path.basename(path)} ({result})")
        except (OSError, ValueError) as e:
            self.set_status(f"保存失败: {e}")

    def load_game(self):
        fd = self._file_dialogs()
        path = fd.askopenfilename(
            title="读取棋谱",
            filetypes=[("逆五子棋棋谱", "*.afg"), ("文本文件", "*.txt"),
                       ("所有文件", "*.*")])
        if not path:
            return
        with self.ai_lock:
            self.ai_result = None
        try:
            rec = record.load_game(path)
            g = ReverseGomoku(rec.config)
            for m in rec.moves:
                g.make_move(m)
            if rec.config != self.game.config:
                # 规则配置不同:重建引擎,避免 AI 用错规则继续行棋
                self._rebuild_engines(rec.config)
            self.game = g
            self.epoch += 1
            self.show_result_overlay = False
            self.enter_review()
            self.set_status(f"已载入 {len(rec.moves)} 手 [{rec.result}] "
                            f"({os.path.basename(path)})")
        except Exception as e:
            self.set_status(f"读取失败: {e}")

    # ---- 交互 ----
    def _mark_move_arrow(self, animated: bool) -> None:
        """根据最新着法更新移子箭头:(占领位, 安置位, 飞行起始时间|None)。
        animated=True 播放 0.5s 棋子飞行(实时对局),None=静态保持(复盘)。
        箭头在下一步完成前保持显示:对方占领中(pending≥0)不清除,
        除非撤销的正是箭头本手(待定占领位 == 箭头起点)。"""
        if not self.show_arrows:
            self.arrow_anim = None
            return
        h = self.game.history
        if len(h) >= 2 and h[-1][2] >= 0:        # 最后一手为安置步,pending=占领位
            self.arrow_anim = (int(h[-1][2]), int(h[-1][0]),
                               time.time() if animated else None)
        elif (self.game.pending < 0
              or (self.arrow_anim is not None
                  and self.arrow_anim[0] == self.game.pending)):
            self.arrow_anim = None               # 新着法完成且非移子,或箭头本手被撤销

    def new_game(self):
        with self.ai_lock:
            self.ai_result = None
        self.epoch += 1
        self.review = False
        self.arrow_anim = None
        self.show_result_overlay = False
        self.game = ReverseGomoku()

    def undo(self):
        if self.review:
            self.review_prev()                     # 复盘模式下悔棋=回退一手
            return
        with self.ai_lock:
            self.ai_result = None
        self.epoch += 1
        if not self.game.history:
            return
        if self.game.pending >= 0:
            self.game.undo_move()                      # 取消占领,回到本回合普通步
            self._mark_move_arrow(False)
            return
        while self.game.history:                       # 连续悔棋,直到人类回合
            self.game.undo_move()
            if not is_ai_turn(self.mode, self.game):
                break
        self._mark_move_arrow(False)

    def _map_pos(self, pos):
        """实际窗口坐标 → 虚拟画布坐标"""
        if self.scale >= 1.0:
            return pos
        return (int(pos[0] / self.scale), int(pos[1] / self.scale))

    def click(self, pos):
        # pos 已是虚拟画布坐标(事件循环 _map_pos 一次即可,此处不可重复映射,
        # 否则 scale<1.0 的缩放屏上落子会向右下偏移)
        if self.review:
            cell = self.board.screen_to_cell(*pos)
            if cell is not None:
                self.review_try(cell[0] * BOARD_SIZE + cell[1])
            return
        if self.game.game_over:
            return
        if is_ai_turn(self.mode, self.game):
            return
        cell = self.board.screen_to_cell(*pos)
        if cell is None:
            return
        idx = cell[0] * BOARD_SIZE + cell[1]
        if self.game.legal_mask()[idx]:
            self.epoch += 1
            self._play_move_sound(self.game, idx)
            self.game.make_move(idx)
            self._mark_move_arrow(True)
            if self.game.game_over:
                self.enter_review(True)

    # ---- 复盘 ----
    def enter_review(self, overlay: bool = False):
        """进入复盘:以当前对局的完整着法序列为主变化。
        overlay=True 时显示终局大字蒙版(2 秒后自动淡出);载入棋谱不显示。"""
        self.review = True
        self.record_moves = list(self.game.record())
        self.moves = list(self.record_moves)
        self.pos = len(self.moves)
        self.review_divergence = None
        self.review_result = record.game_result(self.game)
        self.result_shown_at = 0.0
        self.show_result_overlay = overlay
        self._mark_move_arrow(False)

    def review_jump(self, k: int) -> None:
        """跳到第 k 手(0=开局),从零重建局面。"""
        self.pos = int(max(0, min(k, len(self.moves))))
        with self.ai_lock:
            self.ai_result = None
        self.epoch += 1
        g = ReverseGomoku(self.game.config)
        for m in self.moves[:self.pos]:
            g.make_move(m)
        self.game = g
        self.result_shown_at = 0.0
        self.arrow_anim = None                       # 跳转后按新局面重建箭头
        self._mark_move_arrow(False)

    def review_prev(self):
        self.review_jump(self.pos - 1)

    def review_next(self):
        self.review_jump(self.pos + 1)

    def review_home(self):
        self.review_jump(0)

    def review_end(self):
        self.review_jump(len(self.moves))

    def review_reset(self):
        """恢复原谱主变化,并回到"试下前那一手"(分支点);未试下则到终局。"""
        self.moves = list(self.record_moves)
        target = (self.review_divergence if self.review_divergence is not None
                  else len(self.moves))
        self.review_jump(target)

    def review_try(self, idx: int) -> None:
        """试下:在当前位置落子,替换其后所有变化。"""
        if self.game.game_over or not self.game.legal_mask()[idx]:
            return
        if self.review_divergence is None:
            self.review_divergence = self.pos     # 分支点 = 试下前那一手
        self.moves = self.moves[:self.pos] + [int(idx)]
        self.review_jump(self.pos + 1)

    def _bottom_click(self, pos) -> None:
        """底部工具条点击:先开关按钮,再复盘导航按钮,最后手数轴。"""
        for rect, label, fn in list(self.toggle_btns) + list(self.review_btns):
            if rect.collidepoint(pos):
                fn()
                return
        if self.review and self.review_axis is not None:
            x0, x1, y0, y1 = self.review_axis
            if y0 <= pos[1] <= y1:
                n = max(1, len(self.moves))
                t = (pos[0] - x0) / max(1, x1 - x0)
                self.review_jump(round(t * n))

    def _toggle_danger(self):
        self.show_danger = not self.show_danger

    def _toggle_numbers(self):
        self.show_numbers = not self.show_numbers

    def _toggle_arrows(self):
        self.show_arrows = not self.show_arrows
        if not self.show_arrows:
            self.arrow_anim = None
        else:
            self._mark_move_arrow(False)         # 重新开启时恢复当前局面箭头

    def _toggle_rays(self):
        self.show_rays = not self.show_rays

    def _toggle_sound(self):
        self.sound_enabled = not self.sound_enabled
        if not self.sound_enabled:
            try:
                pygame.mixer.stop()
            except pygame.error:
                pass

    def _play_move_sound(self, game, idx):
        """按动作类型播放音效:安置/占领=移子音,普通落子=落子音。"""
        if not self.sound_enabled:
            return
        if game.pending >= 0:
            snd = self.snd_move
        else:
            r, c = divmod(int(idx), BOARD_SIZE)
            opp = WHITE if game.current_player == BLACK else BLACK
            snd = self.snd_move if game.board[r, c] == opp else self.snd_place
        if snd is not None:
            snd.play()

    def _start_ai(self):
        snapshot = self.game.copy()
        epoch = self.epoch
        engine = self.ai_black if snapshot.current_player == BLACK else self.ai_white
        with self.ai_lock:
            self.ai_thinking = True
        # 读秒:仅 AI 限时模式下,记录本步思考截止时间
        if self.limit_mode == "ai":
            self.ai_deadline = time.time() + self.ai_time_budget
        else:
            self.ai_deadline = None

        def work():
            try:
                a = engine.choose(snapshot)
                with self.ai_lock:
                    if epoch == self.epoch:            # 设置时局面未变才登记
                        self.ai_result = (a, epoch)
            finally:
                with self.ai_lock:
                    self.ai_thinking = False
        threading.Thread(target=work, daemon=True).start()

    def _engine_legal(self, idx: int) -> bool:
        """按引擎(无自杀掩码)规则校验 AI 着法:自杀手允许走出,
        走出后完成己方五连自然判负,棋盘上出现真实五连终局。"""
        g = self.game
        cfg = _dc_replace(g.config, mask_suicide=False)
        return bool(ReverseGomoku.legal_mask_for(
            g.board, g.current_player, g.pending, g.white_turns,
            cfg, g.turn_count)[idx])

    def _finish_by_kill(self) -> bool:
        """行棋方必败终局前,替对手(胜方)走出一步绝杀(占领+安置,必成):
        让棋盘出现真实五连后自然终局。无一步绝杀返回 False(退回直接判负)。"""
        g = self.game
        if g.pending >= 0 or g.turn_count < g.config.loss_start_turns:
            return False
        winner = WHITE if g.current_player == BLACK else BLACK
        try:
            kills = kill_captures(g.board, winner, g.pending, g.turn_count,
                                  g.white_turns, g.config)
        except Exception:
            return False
        if len(kills) == 0:
            return False
        cap, place = int(kills[0][0]), int(kills[0][1])
        self._play_move_sound(g, cap)
        g.make_move(cap)            # 占领:进入安置
        if not g.game_over:
            self._play_move_sound(g, place)
            g.make_move(place)      # 安置:完成败方五连,自然终局
        self._mark_move_arrow(True)
        return g.game_over and not g.is_draw

    def _poll_shortcuts(self) -> None:
        """键盘状态轮询兜底:直接读取键盘状态,即使 KEYDOWN/TEXTINPUT 事件被
        输入法/远程环境吞掉,也能触发快捷键。仅处理"新按下"沿(按住不重复触发),
        本帧已被事件处理过的键跳过;设定面板打开时不生效。"""
        pressed = pygame.key.get_pressed()
        prev = self._shortcut_prev
        self._shortcut_prev = pressed
        if prev is None or self.settings is not None:
            return
        def edge(k: int) -> bool:
            return bool(pressed[k] and not prev[k] and k not in self._handled_keys)
        if edge(pygame.K_r):
            self.new_game()
        if edge(pygame.K_u):
            self.undo()
        if edge(pygame.K_d):
            self.show_danger = not self.show_danger
        if edge(pygame.K_n):
            self.show_numbers = not self.show_numbers
        if edge(pygame.K_a):
            self._toggle_arrows()
        if edge(pygame.K_h):
            self._toggle_rays()
        if edge(pygame.K_m):
            self._toggle_sound()
        if edge(pygame.K_s):
            self.save_game()
        if edge(pygame.K_l):
            self.load_game()
        if edge(pygame.K_e):
            self._open_settings()
        if edge(pygame.K_1):
            self.mode = 0
            self.new_game()
        if edge(pygame.K_2):
            self.mode = 1
            self.new_game()
        if edge(pygame.K_3):
            self.mode = 2
            self.new_game()
        if edge(pygame.K_4):
            self.mode = 3
            self.new_game()
        if self.review:
            if edge(pygame.K_LEFT):
                self.review_prev()
            if edge(pygame.K_RIGHT):
                self.review_next()

    # ---- 主循环 ----
    def run(self):
        clock = pygame.time.Clock()
        running = True
        frame = 0
        while running:
            self._handled_keys = set()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    s = min(ev.size[0] / WIN_W, ev.size[1] / WIN_H, 1.0)
                    s = max(s, 0.3)
                    aw, ah = int(WIN_W * s), int(WIN_H * s)
                    self.screen = pygame.display.set_mode(
                        (aw, ah), pygame.RESIZABLE)
                    actual_w, actual_h = self.screen.get_size()
                    self.scale = min(actual_w / WIN_W, actual_h / WIN_H, 1.0)
                elif self.settings is not None:
                    # 设定面板打开时,所有事件交给面板处理(暂停对局与 AI)
                    # 映射鼠标坐标到虚拟画布
                    if hasattr(ev, 'pos') and ev.pos is not None:
                        ev.pos = self._map_pos(ev.pos)
                    res = self.settings.handle(ev)
                    if res == "confirm":
                        self._apply_settings(self.settings)
                        self.settings = None
                    elif res == "cancel":
                        self.settings = None
                    continue
                elif ev.type == pygame.KEYDOWN:
                    self._handled_keys.add(ev.key)
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_r:
                        self.new_game()
                    elif ev.key == pygame.K_u:
                        self.undo()
                    elif ev.key == pygame.K_d:
                        self.show_danger = not self.show_danger
                    elif ev.key == pygame.K_n:
                        self.show_numbers = not self.show_numbers
                    elif ev.key == pygame.K_a:
                        self._toggle_arrows()
                    elif ev.key == pygame.K_h:
                        self._toggle_rays()
                    elif ev.key == pygame.K_m:
                        self._toggle_sound()
                    elif ev.key == pygame.K_s:
                        self.save_game()
                    elif ev.key == pygame.K_l:
                        self.load_game()
                    elif ev.key == pygame.K_e:
                        self._open_settings()
                    elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                        self.mode = ev.key - pygame.K_1
                        self.new_game()
                    elif self.review and ev.key == pygame.K_LEFT:
                        self.review_prev()
                    elif self.review and ev.key == pygame.K_RIGHT:
                        self.review_next()
                elif ev.type == pygame.TEXTINPUT:
                    # 输入法/远程桌面等环境下 KEYDOWN 可能被吞,以文本事件兜底触发快捷键
                    k = ev.text.lower()
                    kcode = {"r": pygame.K_r, "u": pygame.K_u, "d": pygame.K_d,
                             "n": pygame.K_n, "a": pygame.K_a, "h": pygame.K_h,
                             "m": pygame.K_m, "s": pygame.K_s, "l": pygame.K_l,
                             "e": pygame.K_e}.get(k)
                    if kcode is not None:
                        if kcode in self._handled_keys:
                            continue        # 本帧 KEYDOWN 已处理过,避免重复触发
                        self._handled_keys.add(kcode)
                    if k == "r":
                        self.new_game()
                    elif k == "u":
                        self.undo()
                    elif k == "d":
                        self.show_danger = not self.show_danger
                    elif k == "n":
                        self.show_numbers = not self.show_numbers
                    elif k == "a":
                        self._toggle_arrows()
                    elif k == "h":
                        self._toggle_rays()
                    elif k == "m":
                        self._toggle_sound()
                    elif k == "s":
                        self.save_game()
                    elif k == "l":
                        self.load_game()
                    elif k == "e":
                        self._open_settings()
                    elif k in ("1", "2", "3", "4"):
                        self.mode = int(k) - 1
                        self.new_game()
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    vpos = self._map_pos(ev.pos)
                    if CLOCK_RECT.collidepoint(vpos):
                        self._toggle_limit_mode()          # 点右上角时钟开关读秒
                        continue
                    if vpos[0] < BOARD_PX and vpos[1] > BOARD_PX:
                        self._bottom_click(vpos)           # 底部工具条
                        continue
                    name = self.panel.hit(vpos)
                    if name in MODE_NAMES:
                        self.mode = MODE_NAMES.index(name)
                        self.new_game()
                    elif name == "新局":
                        self.new_game()
                    elif name == "悔棋":
                        self.undo()
                    elif name == "设定":
                        self._open_settings()
                    elif name == "保存棋谱":
                        self.save_game()
                    elif name == "读取棋谱":
                        self.load_game()
                    else:
                        self.click(vpos)
                elif ev.type == pygame.MOUSEMOTION:
                    self.hover = self.board.screen_to_cell(*self._map_pos(ev.pos))

            # 键盘轮询兜底:输入法/远程环境吞掉按键事件时仍可触发快捷键(边缘触发)
            self._poll_shortcuts()

            # AI 回合:先应用已有结果,再判断是否启动新搜索
            # (顺序必须如此:先启动会导致旧局面的搜索结果落到已切换的回合上)
            if self.settings is None:
                if self.ai_result is not None:
                    with self.ai_lock:
                        a, ep = self.ai_result
                        self.ai_result = None
                    # 应用时校验代数:任何人类操作(落子/悔棋/新局)都会使代数变化
                    if ep == self.epoch:
                        if a is None:
                            mover = self.game.current_player
                            eng = self.ai_black if mover == BLACK else self.ai_white
                            if getattr(eng, "last_error", None) == "timeout" \
                                    and self._ai_retries < 1:
                                # 引擎搜索超时(并非认负):全量重同步后重试一次
                                self._ai_retries += 1
                                eng.invalidate()
                            else:
                                self._ai_retries = 0
                                # pass/认负/引擎故障:行棋方必败。终局前先替胜方
                                # 走出一步绝杀(若存在),让棋盘出现真实五连;
                                # 无一步绝杀才直接判负(棋盘无五连)
                                if not self._finish_by_kill():
                                    self.game.game_over = True
                                    self.game.loser = mover
                                self.enter_review(True)
                        elif self._engine_legal(a):
                            # 按引擎规则校验(含自杀手:走出即完成己方五连,
                            # 自然终局)。绝不静默丢弃 —— 引擎 genmove 已自落子,
                            # 丢弃会导致双方局面永久错位
                            self._ai_retries = 0
                            self._play_move_sound(self.game, a)
                            self.game.make_move(a, allow_suicide=True)
                            self._mark_move_arrow(True)
                            if self.game.game_over:
                                self.enter_review(True)
                if (not self.review and not self.game.game_over
                        and is_ai_turn(self.mode, self.game) and not self.ai_thinking):
                    self._start_ai()
                # 人类回合死局检测:所有落点均为自杀且未满盘 → 行棋方必败。
                # 终局前先替 AI 走出一步绝杀形成真实五连,无绝杀才直接判负
                if (not self.review and not self.game.game_over
                        and not is_ai_turn(self.mode, self.game)
                        and self.game.is_stuck()):
                    if not self._finish_by_kill():
                        self.game.game_over = True
                        self.game.loser = self.game.current_player
                    self.enter_review(True)

            # 绘制到虚拟画布,再缩放到实际窗口
            self.screen, self.virtual = self.virtual, self.screen
            self.draw()
            if self.settings is not None:
                self.settings._vmouse = self._vmouse
                overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 150))
                self.screen.blit(overlay, (0, 0))
                self.settings.draw(self.screen)
            self.screen, self.virtual = self.virtual, self.screen
            if self.scale < 1.0:
                scaled = pygame.transform.smoothscale(
                    self.virtual, self.screen.get_size())
                self.screen.blit(scaled, (0, 0))
            else:
                self.screen.blit(self.virtual, (0, 0))
            pygame.display.flip()
            clock.tick(60)
            frame += 1
            if self._smoke_frames and frame >= self._smoke_frames:
                running = False          # --smoke 自检:跑满指定帧数即退出
        pygame.quit()

    # ---- 绘制 ----
    def draw(self):
        self.screen.fill(COLOR_PANEL)
        # 鼠标坐标映射到虚拟画布(供 hover 检测)
        raw_mouse = pygame.mouse.get_pos()
        self._vmouse = (int(raw_mouse[0] / self.scale),
                        int(raw_mouse[1] / self.scale)) if self.scale < 1.0 else raw_mouse
        legal = self.game.legal_mask()
        held = None
        if self.game.pending >= 0:
            held = self.hover
        num = _stone_numbers(self.game) if self.show_numbers else None
        # 移子箭头:棋子飞行 0.5s,结束后箭头保持显示直到下一步完成
        arrow = None
        hover_target = None
        if self.show_arrows:
            if self.arrow_anim is not None:
                f, t, t0 = self.arrow_anim
                if t0 is None:
                    arrow = (f, t, None)             # 静态完整箭头(复盘/保持)
                else:
                    p = (time.time() - t0) / ARROW_FLY_DURATION
                    if p >= 1.0:
                        arrow = (f, t, None)         # 飞行结束,进入保持态
                    else:
                        arrow = (f, t, max(0.0, p))
            if self.game.pending >= 0 and self.hover is not None:
                hi = self.hover[0] * BOARD_SIZE + self.hover[1]
                if legal[hi]:
                    hover_target = self.hover
        # 辅助射线:悬停棋子时计算其可移动到的所有位置(安置步可达掩码)
        hover_rays = None
        if self.show_rays and self.hover is not None and self.game.pending < 0:
            hi = self.hover[0] * BOARD_SIZE + self.hover[1]
            if self.game.board.reshape(-1)[hi] != EMPTY:
                hover_rays = (hi, ReverseGomoku.legal_mask_for(
                    self.game.board, 0, hi, 0, self.game.config))
        self.board.draw(self.screen, self.game, self.hover, held,
                        self.show_danger, legal,
                        show_numbers=self.show_numbers, num=num,
                        arrow=arrow, hover_target=hover_target,
                        hover_rays=hover_rays)

        # 终局大字蒙版:仅实时对局结束显示,2 秒后自动淡出;载入棋谱不显示
        if self.game.game_over and self.show_result_overlay:
            if self.result_shown_at == 0:
                self.result_shown_at = time.time()
            k = 1.0 - (time.time() - self.result_shown_at
                       - RESULT_OVERLAY_HOLD) / RESULT_OVERLAY_FADE
            k = min(1.0, max(0.0, k))
            if k > 0.01:
                rect = pygame.Rect(0, 0, BOARD_PX, BOARD_PX)
                center = (rect.centerx, rect.centery - 40)
                overlay = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, int(120 * k)))
                self.screen.blit(overlay, (0, 0))
                text = self.font_big.render(
                    "和棋" if self.game.is_draw else
                    ("白棋获胜" if self.game.loser == BLACK else "黑棋获胜"),
                    True, (255, 235, 170))
                text.set_alpha(int(255 * k))
                self.screen.blit(text, (center[0] - text.get_width() // 2,
                                        center[1] - text.get_height() // 2))
        self.panel.draw(self.screen, self.mode, self._vmouse)
        self._draw_settings_summary()
        self._draw_notify()
        self._draw_clock()
        self._draw_rules()
        self._draw_bottom_bar()

    def _summary_lines(self) -> tuple:
        """状态框内容:引擎名/文件名/限时/参数/手数/行棋/状态文本/状态类型。
        各行状态独立成格,不再拼接:"行棋"带 落子/移子 动作后缀,
        状态条按类型着色(普通/待定/复盘/试下/结果)。"""
        limit = (f"读 {self.ai_time_budget:g} 秒" if self.limit_mode == "ai"
                 else "不限时")
        if isinstance(self.ai_black, _HeuristicEngine):
            from . import heuristic_versions as hv
            vtxt = hv.label(self.heuristic_version)
            if hv.supports_vcf(self.heuristic_version):
                vtxt += "VCF开" if self.heuristic_vcf else "VCF关"
            engine, fname, param = "启发式 AI", vtxt, f"深度 {self.heuristic_depth} 层"
        elif isinstance(self.ai_black, _KataGoEngine):
            name = os.path.basename(self.kata_cfg.get("weight", "")) or "KataGo"
            param = f"模拟 {self.mcts_sims}"
            if self.limit_mode == "ai":
                param += "(限时先到)"    # maxTime 与 maxVisits 并存,先到先停
            engine, fname = "KataGo", name
        elif self.model_path:
            engine, fname, param = ("神经网络", os.path.basename(self.model_path),
                                    f"模拟 {self.mcts_sims}")
        else:
            engine, fname, param = "随机先验 MCTS", "", f"模拟 {self.mcts_sims}"
        color = "黑" if self.game.current_player == BLACK else "白"
        action = "移子" if self.game.pending >= 0 else "落子"
        if self.review:
            step_lb, step_val = "复盘", f"{self.pos} / {len(self.moves)}"
            if self.game.game_over:
                mover, status, kind = ("已终局",
                                       f"结果: {record.game_result(self.game)}",
                                       "result")
            elif self.moves != self.record_moves:
                mover, status, kind = (f"{color}·{action}",
                                       f"试下中 · 原谱 {self.review_result}",
                                       "divergence")
            else:
                mover, status, kind = (f"{color}·{action}",
                                       "点击棋盘可试下 · 点轴回看", "review")
        else:
            who = "你" if not is_ai_turn(self.mode, self.game) else "AI"
            step_lb, step_val = ("手数",
                                 f"{self.game.turn_count + 1} / {self.game.move_count}")
            mover = f"{color}({who})·{action}"
            if self.game.pending >= 0:
                status, kind = "移子待定 · 点击目标位置安置", "pending"
            else:
                status, kind = "落子待定 · 可占敌移子", "normal"
        return engine, fname, limit, param, step_lb, step_val, mover, status, kind

    @staticmethod
    def _fit_text(font, text, max_w, color=COLOR_TEXT):
        """按宽度截断渲染文本,过长时尾部加省略号。"""
        t = font.render(text, True, color)
        if t.get_width() <= max_w:
            return t
        while len(text) > 1 and t.get_width() > max_w:
            text = text[:-1]
            t = font.render(text + "…", True, color)
        return t

    def _draw_settings_summary(self) -> None:
        """时钟下方状态框:引擎(名称/文件名) + 两列信息格(限时/参数, 手数/行棋) + 状态条。"""
        r = SETTINGS_RECT
        pygame.draw.rect(self.screen, (250, 246, 236), r, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_WOOD_DARK, r, 2, border_radius=8)
        engine, fname, limit, param, step_lb, step_val, mover, status, kind = \
            self._summary_lines()
        # 引擎行:名称 + 权重/模型文件名(仅文件类引擎,过长截断)
        t = self._fit_text(self.font_sm, engine, r.w - 20)
        self.screen.blit(t, (r.x + 10, r.y + 5))
        if fname:
            tf = self._fit_text(self.font_xs, fname, r.w - 20, (150, 120, 80))
            self.screen.blit(tf, (r.x + 10, r.y + 24))
        # 两列信息格:标签浅褐 + 值深褐,独立成格填满宽度(两行各两格)
        label_color = (150, 120, 80)
        cell_w = (r.w - 20 - 8) // 2
        left_x, right_x = r.x + 10, r.x + 10 + cell_w + 8
        for row, cells in enumerate(((("限时", limit), ("参数", param)),
                                     ((step_lb, step_val), ("行棋", mover)))):
            for col, (lb, val) in enumerate(cells):
                x = left_x if col == 0 else right_x
                y = r.y + 41 + row * 15
                tl = self.font_xs.render(lb, True, label_color)
                tv = self._fit_text(self.font_xs, val,
                                    cell_w - tl.get_width() - 4)
                self.screen.blit(tl, (x, y))
                self.screen.blit(tv, (x + tl.get_width() + 4, y))
        # 状态条:整行浅色底,按状态类型着色
        strip = pygame.Rect(r.x + 8, r.y + 74, r.w - 16, 18)
        style = {"normal": ((255, 243, 213), (90, 66, 36)),
                 "pending": ((255, 236, 190), (176, 118, 40)),
                 "review": ((255, 243, 213), (90, 66, 36)),
                 "divergence": ((255, 236, 190), (176, 118, 40)),
                 "result": ((252, 226, 216), (180, 60, 40))}
        bg, fg = style.get(kind, style["normal"])
        pygame.draw.rect(self.screen, bg, strip, border_radius=5)
        t = self._fit_text(self.font_xs, status, strip.w - 12, fg)
        self.screen.blit(t, (strip.x + 6, strip.centery - t.get_height() // 2))

    def _draw_notify(self) -> None:
        """按钮区正上方的更新提示空隙:set_status 消息在此显示,无消息时显示淡灰占位。"""
        r = NOTIFY_RECT
        pygame.draw.rect(self.screen, (250, 246, 236), r, border_radius=6)
        pygame.draw.rect(self.screen, (210, 186, 152), r, 1, border_radius=6)
        if self.status and time.time() < self.status_until:
            t = self._fit_text(self.font_xs, self.status, r.w - 12, (176, 118, 40))
        else:
            t = self._fit_text(self.font_xs, "暂无提示", r.w - 12, (200, 194, 182))
        self.screen.blit(t, (r.x + 6, r.centery - t.get_height() // 2))

    _RULES = ("1. 胜负：连成五子者输。",
              "2. 落子：黑先白后，在交叉点上落子。",
              "3. 推子：落在对方棋子上时，使对方棋子在八个方向上移动任意格，"
              "但无法越过其他棋子。",
              "4. 禁手：白前两手禁止推子，禁止自杀着，禁止无路线推子。")

    @staticmethod
    def _wrap_text(font, text, max_w):
        """按像素宽度把文本拆成多行(整段中文按字符切)。"""
        lines, cur = [], ""
        for ch in text:
            if cur and font.size(cur + ch)[0] > max_w:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        if cur:
            lines.append(cur)
        return lines

    def _draw_rules(self) -> None:
        """右侧面板原状态栏位置固定规则讲解(横排自动换行)。"""
        x, y = BOARD_PX + 24, PANEL_BUTTON_TOP + len(self.panel.buttons) * 56 + 14
        max_w = PANEL_W - 48
        for item in self._RULES:
            for line in self._wrap_text(self.font_xs, item, max_w):
                t = self.font_xs.render(line, True, (150, 128, 100))
                self.screen.blit(t, (x, y))
                y += 22

    def _draw_clock(self) -> None:
        """右上角读秒时钟(正常数字字体,分钟:秒.十分位,暖木配色,不遮挡棋盘)。

        不限时/复盘/终局显示静态;仅 AI 限时且轮到 AI 时显示倒计时,
        数字红色(<5s)提示超时风险。点击该时钟可随时开关读秒。
        """
        r = CLOCK_RECT
        mouse = getattr(self, '_vmouse', None) or pygame.mouse.get_pos()
        hover = r.collidepoint(mouse)
        # 暖木时钟框(与整体配色协调)
        pygame.draw.rect(self.screen, (88, 64, 44), r, border_radius=9)
        pygame.draw.rect(self.screen, (255, 200, 120) if hover else (146, 106, 66),
                         r, 2, border_radius=9)
        pygame.draw.rect(self.screen, (46, 33, 24), r.inflate(-8, -8), border_radius=6)
        # 标题 + AI 思考指示灯(原右侧面板"AI 思考中"状态移至时钟内)
        title = self.font_xs.render("AI 读秒", True, (240, 214, 178))
        self.screen.blit(title, (r.x + 14, r.y + 8))
        ind_x = r.x + 14 + title.get_width() + 10
        if self.ai_thinking:
            pygame.draw.circle(self.screen, (255, 120, 80), (ind_x + 5, r.y + 17), 4)
            ind = self.font_xs.render("AI思考中", True, (255, 190, 120))
        else:
            pygame.draw.circle(self.screen, (110, 92, 68), (ind_x + 5, r.y + 17), 4)
            ind = self.font_xs.render("AI空闲", True, (120, 100, 76))
        self.screen.blit(ind, (ind_x + 14, r.y + 8))
        # 计算显示内容与状态
        if self.limit_mode != "ai":
            txt, warn = "不限时", False
        elif self.review or self.game.game_over:
            txt, warn = "--:--", False
        elif is_ai_turn(self.mode, self.game):
            remain = self.ai_time_budget
            if self.ai_thinking and self.ai_deadline is not None:
                remain = self.ai_deadline - time.time()
            remain = max(0.0, remain)
            txt, warn = _fmt_clock(remain), remain < 5.0
        else:                                      # 人类回合,AI 未计时
            txt, warn = _fmt_clock(self.ai_time_budget), False
        # 正常数字字体:超时(<5s)转红;限时读数用时钟字体,不限时用常规字体
        on = COLOR_DANGER_SELF if warn else (255, 198, 96)
        f = self.font_clock if self.limit_mode == "ai" else self.font
        t = f.render(txt, True, on if self.limit_mode == "ai" else (235, 205, 160))
        self.screen.blit(t, (r.right - t.get_width() - 16,
                             r.centery - t.get_height() // 2))
        # 底部小字提示(贴内边框底部,避免与边框重合)
        tip = self.font_xs.render("点击开关读秒", True, (160, 132, 100))
        inner = r.inflate(-8, -8)
        self.screen.blit(tip, (r.x + 14, inner.bottom - tip.get_height() - 1))
        # AI 实时计算量(思考中显示在右下角):模拟数/节点数/访问数
        if self.ai_thinking:
            eng = (self.ai_black if self.game.current_player == BLACK
                   else self.ai_white)
            comp = _compute_text(eng)
            if comp:
                ct = self.font_xs.render(comp, True, (255, 214, 150))
                self.screen.blit(ct, (inner.right - ct.get_width() - 2,
                                      inner.bottom - ct.get_height() - 1))

    # ---- 底部工具条 ----
    def _draw_bottom_bar(self):
        """棋盘下方工具条:开关行右侧为随状态变化的变化提示,正下方为单行快捷键;
        复盘模式另含导航按钮与手数轴。"""
        y0 = BOARD_PX
        self.screen.fill(COLOR_PANEL, (0, y0, BOARD_PX, WIN_H - y0))
        pygame.draw.line(self.screen, COLOR_WOOD_DARK, (0, y0), (BOARD_PX, y0), 2)
        if self.review:
            self._draw_review_nav(y0)             # 行1:导航按钮
            self._draw_review_axis(y0)            # 行2:尺子式手数轴
            x = self._draw_toggle_btns(y0 + 78)   # 行3:开关 + 右侧变化提示
            self._draw_context_hint(x + 10, y0 + 86)
            self._draw_shortcut_hints(y0 + 112)   # 行4:单行快捷键
        else:
            x = self._draw_toggle_btns(y0 + 8)
            self._draw_context_hint(x + 10, y0 + 16)
            self._draw_shortcut_hints(y0 + 52)

    def _draw_context_hint(self, x: int, y: int) -> None:
        """横栏右侧变化提示(随状态切换,与右侧面板状态框信息互补)。"""
        if self.review:
            if self.game.game_over:
                hint = "终局 · 点击手数轴跳转回看"
            elif self.moves != self.record_moves:
                hint = "R 新局退出 · 原谱恢复主变化"
            else:
                hint = "点击手数轴跳转 · 棋盘可试下"
        else:
            hint = "终局或载入棋谱自动复盘"
        t = self.font_xs.render(hint, True, COLOR_TEXT)
        self.screen.blit(t, (x, y))

    def _draw_shortcut_hints(self, y: int) -> None:
        """底部单行快捷键提示(不换行)。"""
        line = ("R新局 U悔棋 D危险 N手数 A箭头 H射线 M音效 "
                "S保存 L读取 E设定 Esc退出 1-4模式")
        t = self.font_xs.render(line, True, COLOR_TEXT)
        self.screen.blit(t, (16, y))

    def _draw_toggle_btns(self, y: int) -> int:
        """危险/手数/箭头/辅助射线/音效开关,返回按钮行右侧 x。"""
        self.toggle_btns = []
        x = 16
        for label, on, fn in (("危险提示", self.show_danger, self._toggle_danger),
                              ("手数显示", self.show_numbers, self._toggle_numbers),
                              ("箭头指示", self.show_arrows, self._toggle_arrows),
                              ("辅助射线", self.show_rays, self._toggle_rays),
                              ("音效", self.sound_enabled, self._toggle_sound)):
            rect = pygame.Rect(x, y, 84, 30)
            self.toggle_btns.append((rect, label, fn))
            color = COLOR_BTN_ACT if on else COLOR_BTN
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, COLOR_WOOD_DARK, rect, 2, border_radius=8)
            txt = self.font_sm.render(label, True, COLOR_TEXT)
            self.screen.blit(txt, (rect.centerx - txt.get_width() // 2,
                                   rect.centery - txt.get_height() // 2))
            x += 94
        return x

    def _draw_review_nav(self, y0: int) -> None:
        self.review_btns = []
        x = 16
        for label, fn in (("开局", self.review_home), ("上一手", self.review_prev),
                          ("下一手", self.review_next), ("终局", self.review_end),
                          ("原谱", self.review_reset)):
            w = 60 if label in ("开局", "终局", "原谱") else 68
            rect = pygame.Rect(x, y0 + 8, w, 30)
            self.review_btns.append((rect, label, fn))
            pygame.draw.rect(self.screen, COLOR_BTN, rect, border_radius=8)
            pygame.draw.rect(self.screen, COLOR_WOOD_DARK, rect, 2, border_radius=8)
            txt = self.font_sm.render(label, True, COLOR_TEXT)
            self.screen.blit(txt, (rect.centerx - txt.get_width() // 2,
                                   rect.centery - txt.get_height() // 2))
            x += w + 10

    def _draw_review_axis(self, y0: int) -> None:
        """尺子式手数轴:隔 10 长刻度(并标数),隔 5 中刻度,隔 1 短刻度。
        手数超过 160 时按比例抽稀(保持 1/5/10 相对关系)。"""
        ax_y = y0 + 48
        x0, x1 = 20, BOARD_PX - 20
        n = len(self.moves)
        self.review_axis = (x0, x1, ax_y - 10, ax_y + 26)
        pygame.draw.line(self.screen, COLOR_WOOD_DARK, (x0, ax_y), (x1, ax_y), 2)
        thin = 1 if n <= 160 else math.ceil(n / 160)
        for k in range(0, n + 1, thin):
            tx = int(x0 + (x1 - x0) * k / max(1, n))
            length = 10 if k % 10 == 0 else (7 if k % 5 == 0 else 4)
            pygame.draw.line(self.screen, COLOR_WOOD_DARK,
                             (tx, ax_y), (tx, ax_y + length), 1)
            if k % 10 == 0:
                num = self.font_xs.render(str(k), True, COLOR_TEXT)
                self.screen.blit(num, (tx - num.get_width() // 2, ax_y + 12))
        t = int(x0 + (x1 - x0) * self.pos / max(1, n))
        pygame.draw.line(self.screen, COLOR_LAST, (t, ax_y - 9), (t, ax_y + 9), 3)


def _normalize_version(v: str) -> str:
    """CLI 版本参数规范化(兼容 old/new 旧值)。"""
    from . import heuristic_versions as hv
    return hv.normalize(v)


def main():
    import argparse
    p = argparse.ArgumentParser(description="逆五子棋 pygame 界面")
    p.add_argument("--model", default="", help="神经网络模型文件(省略则启动后选择)")
    p.add_argument("--engine", default="", choices=("", "mcts", "heuristic", "katago"),
                   help="引擎: mcts=神经网络/随机先验, heuristic=启发式 AI, "
                        "katago=KataGo(省略则启动后选择)")
    p.add_argument("--heuristic-depth", type=int, default=HEURISTIC_DEPTH_DEFAULT,
                   help=f"启发式 AI 搜索深度(默认 {HEURISTIC_DEPTH_DEFAULT},1-8)")
    p.add_argument("--heuristic-version", default=HEURISTIC_VERSION_DEFAULT,
                   type=_normalize_version,
                   choices=("beginner", "mid", "advanced"),
                   help="启发式引擎版本: beginner=初级(经典) / mid=中级(强化) / "
                        "advanced=高级(+蒸馏,默认)")
    p.add_argument("--no-vcf", action="store_true",
                   help="中级/高级关闭强制杀链搜索(VCF),默认开启")
    p.add_argument("--limit", default=LIMIT_MODE_DEFAULT, choices=("none", "ai"),
                   help="对局限时: ai=仅 AI 方限时读秒(默认), none=不限时")
    p.add_argument("--ai-time", type=float, default=AI_TIME_BUDGET_DEFAULT,
                   help=f"限时模式下 AI 每步思考时间上限秒数(默认 {AI_TIME_BUDGET_DEFAULT})")
    p.add_argument("--sims", type=int, default=NN_SIMS_DEFAULT,
                   help=f"神经网络决策默认阈值/模拟次数(默认 {NN_SIMS_DEFAULT},不限时时按此模拟)")
    p.add_argument("--c-puct", type=float, default=5.0, help="MCTS 探索系数")
    p.add_argument("--ai-temp", "--selfplay-temp", dest="ai_temp",
                   type=float, default=0.0,
                   help="KataGo AI 温度(如 0.5):选点按访问分布采样,"
                        "人机/机机对局均生效;0=始终最优着(默认)")
    p.add_argument("--device", default="", help="cuda/cpu,空=自动(torch 不可用时忽略)")
    p.add_argument("--smoke", type=int, default=0,
                   help="自检:启动后运行 N 帧即退出(0=正常,供无头冒烟测试)")
    args = p.parse_args()

    device = None
    if NET_AVAILABLE:
        device = torch.device(args.device if args.device else
                              ("cuda" if torch.cuda.is_available() else "cpu"))
    App(device, args).run()


if __name__ == "__main__":
    main()
