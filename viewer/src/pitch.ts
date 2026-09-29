// Canvas pitch: markings, players (home filled, away ringed, both numbered), ball,
// optional back lines and passing-option arrows.

import type { MatchData, PassMoment } from "./data";

const L = 105, W = 68, M = 4; // pitch metres and margin
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export interface Overlay { backLines: Record<"home" | "away", number | null>; moment: PassMoment | null }

export class Pitch {
  private ctx: CanvasRenderingContext2D;
  private scale = 1;
  private dpr = 1;
  private colors = { pitch: "", stripe: "", line: "", home: "", away: "", paper: "", ink: "" };

  constructor(private canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("canvas 2D not supported");
    this.ctx = ctx;
    this.colors = {
      pitch: css("--pitch"), stripe: "oklch(0.62 0.075 145)", line: css("--pitch-line"),
      home: css("--home"), away: css("--away"), paper: css("--paper"), ink: css("--ink"),
    };
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  private resize() {
    const dpr = window.devicePixelRatio || 1;
    this.dpr = dpr;
    const w = this.canvas.clientWidth;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round((w * (W + 2 * M)) / (L + 2 * M) * dpr);
    this.scale = this.canvas.width / (L + 2 * M);
  }

  /** Metres (centre origin, +y up) to canvas pixels. */
  private px(x: number, y: number): [number, number] {
    return [(x + L / 2 + M) * this.scale, (W / 2 + M - y) * this.scale];
  }

  private markings() {
    const { ctx, colors } = this;
    ctx.fillStyle = colors.pitch;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.fillStyle = colors.stripe;
    for (let i = 0; i < 12; i += 2) {
      const [x0] = this.px(-L / 2 + (i * L) / 12, 0);
      ctx.fillRect(x0, this.px(0, W / 2)[1], (L / 12) * this.scale, W * this.scale);
    }
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = Math.max(1, 0.12 * this.scale);
    const rect = (x0: number, y0: number, x1: number, y1: number) => {
      const [a, b] = this.px(x0, y1); const [c, d] = this.px(x1, y0);
      ctx.strokeRect(a, b, c - a, d - b);
    };
    rect(-L / 2, -W / 2, L / 2, W / 2);
    ctx.beginPath(); ctx.moveTo(...this.px(0, W / 2)); ctx.lineTo(...this.px(0, -W / 2)); ctx.stroke();
    ctx.beginPath(); ctx.arc(...this.px(0, 0), 9.15 * this.scale, 0, 2 * Math.PI); ctx.stroke();
    for (const s of [-1, 1]) {
      rect(s * L / 2 - (s > 0 ? 16.5 : 0), -20.16, s * L / 2 + (s < 0 ? 16.5 : 0), 20.16);
      rect(s * L / 2 - (s > 0 ? 5.5 : 0), -9.16, s * L / 2 + (s < 0 ? 5.5 : 0), 9.16);
      ctx.beginPath(); ctx.arc(...this.px(s * (L / 2 - 11), 0), 0.25 * this.scale, 0, 2 * Math.PI);
      ctx.fillStyle = colors.line; ctx.fill();
    }
  }

  draw(data: MatchData, index: number, frac: number, overlay: Overlay) {
    const { ctx, colors } = this;
    const { manifest, positions } = data;
    const n = manifest.objects.length;
    const next = Math.min(index + 1, data.total - 1);
    const at = (i: number, obj: number): [number, number] | null => {
      const o = (i * n + obj) * 2;
      const x = positions[o], y = positions[o + 1];
      return x === manifest.absent ? null : [x / 10, y / 10];
    };
    const pos = (obj: number): [number, number] | null => {
      const a = at(index, obj);
      const b = at(next, obj);
      if (!a) return null;
      if (!b || frac === 0) return a;
      return [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac];
    };
    this.markings();

    const byTeam: Record<"home" | "away", { id: string; label: string; xy: [number, number] }[]> =
      { home: [], away: [] };
    let ball: [number, number] | null = null;
    manifest.objects.forEach((o, i) => {
      const xy = pos(i);
      if (!xy) return;
      if (o.team === "ball") ball = xy;
      else byTeam[o.team].push({ id: o.id, label: o.label, xy });
    });

    for (const team of ["home", "away"] as const) {
      const count = overlay.backLines[team];
      if (!count) continue;
      const sign = team === "home" ? 1 : -1; // home attacks +x
      const players = [...byTeam[team]].sort((a, b) => sign * (a.xy[0] - b.xy[0]));
      const line = players.slice(1, 1 + count).sort((a, b) => a.xy[1] - b.xy[1]);
      if (line.length < 2) continue;
      ctx.strokeStyle = team === "home" ? colors.home : colors.away;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 0.28 * this.scale;
      ctx.setLineDash([0.9 * this.scale, 0.6 * this.scale]);
      ctx.beginPath();
      line.forEach((p, k) => (k ? ctx.lineTo(...this.px(...p.xy)) : ctx.moveTo(...this.px(...p.xy))));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    if (overlay.moment) {
      this.options(overlay.moment, byTeam);
      ctx.font = `500 ${this.fontPx(1.2, 10)}px "Instrument Sans", sans-serif`;
      ctx.textAlign = "left";
      ctx.fillStyle = colors.line;
      const [cx, cy] = this.px(-L / 2, -W / 2 - 2.6);
      const who = overlay.moment.team === "home" ? "Home" : "Away";
      ctx.fillText(`${who} pass · % = model estimate (calibrated pitch control) · bold = played`, cx, cy);
    }

    const r = 1.15 * this.scale;
    ctx.font = `600 ${Math.round(1.25 * this.scale)}px "Instrument Sans", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const team of ["home", "away"] as const) {
      for (const p of byTeam[team]) {
        const [x, y] = this.px(...p.xy);
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        if (team === "home") {
          ctx.fillStyle = colors.home; ctx.fill();
          ctx.fillStyle = colors.paper;
        } else {
          ctx.fillStyle = colors.paper; ctx.fill();
          ctx.lineWidth = 0.32 * this.scale; ctx.strokeStyle = colors.away; ctx.stroke();
          ctx.fillStyle = colors.away;
        }
        ctx.fillText(p.label, x, y + 0.05 * this.scale);
      }
    }
    if (ball) {
      const [x, y] = this.px(...(ball as [number, number]));
      ctx.beginPath(); ctx.arc(x, y, 0.55 * this.scale, 0, 2 * Math.PI);
      ctx.fillStyle = colors.ink; ctx.fill();
      ctx.lineWidth = 0.18 * this.scale; ctx.strokeStyle = colors.paper; ctx.stroke();
    }
  }

  /** Font size in canvas pixels: ``metres`` on the pitch, never below ``minCss`` CSS pixels. */
  private fontPx(metres: number, minCss: number): number {
    return Math.round(Math.max(metres * this.scale, minCss * this.dpr));
  }

  private options(m: PassMoment, byTeam: Record<"home" | "away", { id: string; xy: [number, number] }[]>) {
    const { ctx, colors } = this;
    const team = byTeam[m.team];
    const passer = team.find((p) => p.id === m.from);
    if (!passer) return;
    const [x0, y0] = this.px(...passer.xy);
    // labelled: the pass played plus the three best-rated other options. Width and
    // opacity encode the estimate, so unlabelled arrows still read as weaker or stronger.
    const labelled = new Set([
      ...m.options.filter((o) => o.target).map((o) => o.player),
      ...m.options.filter((o) => !o.target).sort((a, b) => b.p - a.p).slice(0, 3).map((o) => o.player),
    ]);
    for (const o of m.options) {
      const target = team.find((p) => p.id === o.player);
      if (!target) continue;
      const [x1, y1] = this.px(...target.xy);
      ctx.strokeStyle = colors.ink;
      ctx.globalAlpha = o.target ? 0.95 : 0.2 + 0.55 * o.p;
      ctx.lineWidth = (0.08 + 0.26 * o.p + (o.target ? 0.08 : 0)) * this.scale;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      const ang = Math.atan2(y1 - y0, x1 - x0), h = 0.9 * this.scale, back = 1.3 * this.scale;
      const tx = x1 - Math.cos(ang) * back, ty = y1 - Math.sin(ang) * back;
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - Math.cos(ang - 0.45) * h, ty - Math.sin(ang - 0.45) * h);
      ctx.lineTo(tx - Math.cos(ang + 0.45) * h, ty - Math.sin(ang + 0.45) * h);
      ctx.closePath(); ctx.fillStyle = colors.ink; ctx.fill();
      ctx.globalAlpha = 1;
      if (!labelled.has(o.player)) continue;
      const label = percent(o.p);
      const lx = x0 + (x1 - x0) * 0.55, ly = y0 + (y1 - y0) * 0.55 - 0.9 * this.scale;
      ctx.font = `${o.target ? 700 : 500} ${this.fontPx(1.15, 11)}px "Instrument Sans", sans-serif`;
      ctx.lineWidth = Math.max(0.35 * this.scale, 3 * this.dpr); ctx.strokeStyle = colors.paper;
      ctx.strokeText(label, lx, ly); ctx.fillStyle = colors.ink; ctx.fillText(label, lx, ly);
    }
  }
}

/** A model estimate as a whole percentage, clamped to 1-99% (never shown as certain). */
export function percent(p: number): string {
  return `${Math.min(99, Math.max(1, Math.round(p * 100)))}%`;
}
