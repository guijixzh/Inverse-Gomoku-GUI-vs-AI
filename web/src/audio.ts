export class Sounds {
  enabled = true;
  private ctx: AudioContext | null = null;

  private ensure(): AudioContext | null {
    if (!this.ctx) {
      const Ctor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return null;
      this.ctx = new Ctor();
    }
    if (this.ctx.state === "suspended") void this.ctx.resume();
    return this.ctx;
  }

  private tone(freq: number, duration: number, volume: number) {
    if (!this.enabled) return;
    const ctx = this.ensure();
    if (!ctx) return;
    const t0 = ctx.currentTime;
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(volume, t0);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
    gain.connect(ctx.destination);
    for (const [mult, amp] of [[1, 1], [2, 0.4]] as const) {
      const osc = ctx.createOscillator();
      osc.type = "sine";
      osc.frequency.value = freq * mult;
      const partial = ctx.createGain();
      partial.gain.value = amp;
      osc.connect(partial);
      partial.connect(gain);
      osc.start(t0);
      osc.stop(t0 + duration);
    }
  }

  place() {
    this.tone(880, 0.08, 0.25);
  }

  move() {
    this.tone(440, 0.12, 0.28);
  }

  win() {
    this.tone(660, 0.18, 0.22);
    window.setTimeout(() => this.tone(990, 0.22, 0.2), 120);
  }
}
