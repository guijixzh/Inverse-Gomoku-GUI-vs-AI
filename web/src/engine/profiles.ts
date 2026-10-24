/** 五档引擎参数：与 Python heuristic_versions 的档位语义对齐。 */
export type EngineVersion = "beginner" | "mid" | "advanced" | "hell" | "legend";

const ALIASES: Record<string, EngineVersion> = {
  old: "beginner", beginner: "beginner",
  mid: "mid",
  new: "advanced", advanced: "advanced",
  hell: "hell", master: "hell", "地狱": "hell", "大师": "hell",
  legend: "legend", "传奇": "legend",
};

export function normalizeVersion(v: unknown): EngineVersion {
  return ALIASES[String(v ?? "")] ?? "advanced";
}

export interface Profile {
  version: EngineVersion;
  label: string;
  /** 初级没有 VCF / 置换表 / 开局库 / 蒸馏。 */
  supportsVcf: boolean;
  supportsBook: boolean;
  /** 0=无蒸馏，1=v1，2=v2；高级/大师 v1，传奇 v2（大师默认开 v1，见 docs）。 */
  distill: 0 | 1 | 2;
  vcfDepth: number;
  vcfNodes: number;
  defaultVcf: boolean;
  /** 大师/传奇档的搜索侧防守机制（移植自 Python heuristic.py）：
   * 防守预留时间片、证明必败降权、预防性封堵（先手礼防御）。 */
  defense: boolean;
}

const PROFILES: Record<EngineVersion, Profile> = {
  beginner: { version: "beginner", label: "初级", supportsVcf: false, supportsBook: false,
    distill: 0, vcfDepth: 0, vcfNodes: 0, defaultVcf: false, defense: false },
  mid: { version: "mid", label: "中级", supportsVcf: true, supportsBook: true,
    distill: 0, vcfDepth: 8, vcfNodes: 60_000, defaultVcf: true, defense: false },
  advanced: { version: "advanced", label: "高级", supportsVcf: true, supportsBook: true,
    distill: 1, vcfDepth: 8, vcfNodes: 60_000, defaultVcf: true, defense: false },
  hell: { version: "hell", label: "大师", supportsVcf: true, supportsBook: true,
    distill: 1, vcfDepth: 12, vcfNodes: 240_000, defaultVcf: true, defense: true },
  legend: { version: "legend", label: "传奇", supportsVcf: true, supportsBook: true,
    distill: 2, vcfDepth: 12, vcfNodes: 240_000, defaultVcf: true, defense: true },
};

export function profileFor(version: unknown): Profile {
  return PROFILES[normalizeVersion(version)];
}

/** UI 的“难度深度”按 Python 档位上限修正。 */
export function depthForProfile(version: EngineVersion, depth: number): number {
  const d = Math.max(1, Math.floor(depth));
  switch (version) {
    case "beginner": return Math.min(d, 2);
    case "mid": return Math.min(d, 4);
    case "advanced": return Math.min(d, 8);
    case "hell": return d + 1;
    case "legend": return d + 2;
  }
}

/** VCF 预算：有对局限时取 15%，夹在 [0.05, 1.8] 秒；不限时给 0.05s。 */
export function vcfSeconds(budgetSec: number): number {
  if (!budgetSec) return 0.05;
  return Math.min(1.8, Math.max(0.05, 0.15 * budgetSec));
}
