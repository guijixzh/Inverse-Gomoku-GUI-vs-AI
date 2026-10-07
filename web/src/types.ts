export const BLACK = 1;
export const WHITE = 2;
export const EMPTY = 0;
export const BOARD_SIZE = 15;
export const CELL = 50;
export const MARGIN = 48;
export const BOARD_PX = CELL * (BOARD_SIZE - 1) + 2 * MARGIN;

export type StepKind = "place" | "capture" | "place2";

export interface GameConfig {
  white_restrict_turns: number;
  loss_start_turns: number;
  mask_suicide: boolean;
}

export interface GameState {
  board: number[];
  player: number;
  pending: number;
  pending_color: number;
  turn_count: number;
  white_turns: number;
  move_count: number;
  game_over: boolean;
  loser: number;
  is_draw: boolean;
  winner: number;
  stuck: boolean;
  result: string;
  legal: number[];
  moves: number[];
  kinds: StepKind[];
  numbers: number[];
  danger_me: number[];
  danger_opp: number[];
  config: GameConfig;
}

export interface AiResult {
  state: GameState;
  move: number | null;
  nodes: number;
  elapsed: number;
}

export interface HintResult {
  pos_v: number;
  moves: { idx: number; v: number }[];
}

export interface EngineStatus {
  stage: string;
  detail?: string;
}
