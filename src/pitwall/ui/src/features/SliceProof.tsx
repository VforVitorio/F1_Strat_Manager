/**
 * What sprint 2 renders in both windows: proof the whole chain is live.
 *
 * There is no layout here on purpose. The DATA window's four bands and the
 * AGENTS window's 1:1 port of the Qt split are sprints 3 to 6; this exists to
 * show that the arcade's broadcast reaches two independent React roots
 * through one TCP client, sequenced, and that closing one does not blind the
 * other. It is deleted the moment the real panels land.
 */

import type { TickState } from "../lib/useTick";

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="slice-row">
      <span className="slice-label">{label}</span>
      <span className="slice-value">{value}</span>
    </div>
  );
}

export function SliceProof({ title, state }: { title: string; state: TickState }) {
  const { tick, discontinuity, live } = state;

  if (!live || !tick) {
    return (
      <main className="slice">
        <h1>{title}</h1>
        <p className="slice-waiting">
          Waiting for the arcade broadcast. Start a replay with <code>--strategy</code>.
        </p>
      </main>
    );
  }

  const { arcade, playback } = tick;
  const running = Object.values(arcade.drivers).filter((car) => car.active).length;

  return (
    <main className="slice">
      <h1>{title}</h1>
      <Row label="Session" value={`${arcade.location} ${arcade.year}`} />
      <Row label="Lap" value={`${arcade.lap} / ${arcade.total_laps}`} />
      <Row label="Playback" value={playback.paused ? "PAUSED" : `${playback.speed}x`} />
      <Row
        label="Frame"
        value={`${playback.frame_index} / ${playback.total_frames}`}
      />
      <Row label="Cars running" value={`${running} of ${Object.keys(arcade.drivers).length}`} />
      <Row label="Telemetry span" value={`${arcade.telemetry.main.length} samples`} />
      <Row label="Sequence" value={`#${tick.seq}`} />
      <Row label="Clock" value={discontinuity.kind} />
    </main>
  );
}
