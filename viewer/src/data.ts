// Match data exported by `regista export-viewer`, and time helpers.

export interface MatchObject { id: string; team: "home" | "away" | "ball"; label: string }
export interface Period {
  period: number; first_index: number; n_frames: number;
  t_start: number; t_end: number; clock_start: string; clock_end: string;
}
export interface Manifest {
  id: string; source: string; match_id: string; split: string; fps: number;
  objects: MatchObject[]; periods: Period[]; absent: number; median_margin: number;
  credits: Record<string, string>; notes: string[];
}
export interface PhaseState { label: string; runner_up: string; margin: number; close_call: boolean }
export interface Card { period: number; t: number; team: "home" | "away"; usable: boolean; out: PhaseState | null; in: PhaseState | null }
export interface Alert {
  type: "back_line_change" | "press_change" | "line_height_shift"; team: "home" | "away";
  period: number; emit_t: number; start_t: number; emit_clock: string; start_clock: string;
  severity: number; before: number; after: number;
}
export interface PassOption { player: string; p: number; target: boolean }
export interface PassMoment {
  id: number; period: number; t: number; team: "home" | "away"; from: string; to: string;
  completed: boolean; options: PassOption[];
}
export interface Citation { match: string; period: number; clock_start: string; clock_end: string }
export interface AnswerItem {
  question: string; answer_text: string; status: "verified" | "unverified";
  citations: Citation[]; caveats: string[];
}
export interface Answers { model: string; generated: string; items: AnswerItem[] }
export interface IndexEntry { id: string; title: string; split: string }

export interface MatchData {
  manifest: Manifest; positions: Int16Array; cards: Card[]; alerts: Alert[];
  passes: PassMoment[]; answers: Answers | null; total: number;
}

const base = import.meta.env.BASE_URL;

async function json<T>(path: string): Promise<T> {
  const r = await fetch(`${base}${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json() as Promise<T>;
}

async function gunzip(path: string): Promise<ArrayBuffer> {
  const r = await fetch(`${base}${path}`);
  if (!r.ok || !r.body) throw new Error(`${path}: HTTP ${r.status}`);
  const stream = r.body.pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

export const loadIndex = () => json<IndexEntry[]>("data/index.json");

export async function loadMatch(id: string): Promise<MatchData> {
  const dir = `data/${id}`;
  const [manifest, buf, cards, alerts, passes] = await Promise.all([
    json<Manifest>(`${dir}/manifest.json`), gunzip(`${dir}/frames.i16z`),
    json<Card[]>(`${dir}/cards.json`), json<Alert[]>(`${dir}/alerts.json`),
    json<PassMoment[]>(`${dir}/pass_moments.json`),
  ]);
  let answers: Answers | null = null;
  const r = await fetch(`${base}${dir}/answers.json`);
  if (r.ok && (r.headers.get("content-type") ?? "").includes("json")) {
    answers = (await r.json()) as Answers;
  }
  const total = manifest.periods.reduce((n, p) => n + p.n_frames, 0);
  return { manifest, positions: new Int16Array(buf), cards, alerts, passes, answers, total };
}

// ---- time: a global frame index <-> (period, seconds into the period)

export interface Instant { period: number; t: number }

export function instantAt(m: Manifest, index: number): Instant {
  for (const p of m.periods) {
    if (index < p.first_index + p.n_frames) {
      return { period: p.period, t: p.t_start + Math.max(0, index - p.first_index) / m.fps };
    }
  }
  const last = m.periods[m.periods.length - 1];
  return { period: last.period, t: last.t_end };
}

export function indexAt(m: Manifest, at: Instant): number {
  const p = m.periods.find((q) => q.period === at.period) ?? m.periods[0];
  const i = Math.round((at.t - p.t_start) * m.fps);
  return p.first_index + Math.min(Math.max(i, 0), p.n_frames - 1);
}

const HALF = 45 * 60;

/** Same convention as regista.clock: second half starts at 45:00; stoppage as 45+m:ss. */
export function clock(period: number, t: number): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  if ((period === 1 && t >= HALF) || (period === 2 && t > HALF)) {
    const extra = Math.floor(t - HALF);
    return `${period === 1 ? 45 : 90}+${Math.floor(extra / 60)}:${pad(extra % 60)}`;
  }
  const total = Math.floor(t + (period === 2 ? HALF : 0));
  return `${pad(Math.floor(total / 60))}:${pad(total % 60)}`;
}

/** Parse a clock string back to an instant in the given period. */
export function parseClock(text: string, period: number): Instant {
  const stoppage = /^(45|90)\+(\d+)(?::(\d{2}))?$/.exec(text);
  if (stoppage) return { period, t: HALF + Number(stoppage[2]) * 60 + Number(stoppage[3] ?? 0) };
  const [mm, ss] = text.split(":").map(Number);
  const seconds = mm * 60 + (ss ?? 0);
  return { period, t: period === 2 ? seconds - HALF : seconds };
}

export const before = (a: Instant, b: Instant) =>
  a.period < b.period || (a.period === b.period && a.t <= b.t);
