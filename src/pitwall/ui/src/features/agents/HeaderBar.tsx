/**
 * The 44 px top strip: session, driver, neutralisation, connection, playback, lap.
 *
 * Every string is built host side; this places them.
 *
 * **The neutralisation chip is new and it is not decoration.** This window said
 * nothing at all about a safety car except as RCM prose inside the RADIO card,
 * which is the one fact that changes a strategy call the most. It reads the
 * tick's decoded `track_status_label`, the same field the DATA strip reads, and
 * wears it through the shared `trackStatusTreatment` so the two windows cannot
 * disagree about it on a desk where both are open.
 */

import type { HeaderView } from "../../lib/agents";
import { trackStatusTreatment } from "../../lib/trackStatus";

function TrackStatusChip({ header, frozen }: { header: HeaderView; frozen: boolean }) {
  const worn = trackStatusTreatment(header.track_status, header.track_status_colour, frozen);
  if (worn.kind === "unknown") return <span className="chip is-unknown">{worn.text}</span>;
  return (
    <span
      className={worn.kind === "filled" ? "chip is-filled" : "chip"}
      style={
        worn.kind === "filled"
          ? { background: worn.rgb, borderColor: worn.rgb }
          : { color: worn.rgb, borderColor: worn.rgb }
      }
    >
      {worn.text}
    </span>
  );
}

export function HeaderBar({ header, frozen = false }: { header: HeaderView; frozen?: boolean }) {
  return (
    <header className="header-bar">
      <span className="header-session">{header.session}</span>
      <span className="header-driver">{header.driver}</span>
      <span className="header-spacer" />
      {/* Left of the connection chip, because it is about the RACE and the ones
          after it are about this window's plumbing. */}
      <TrackStatusChip header={header} frozen={frozen} />
      <span className="chip header-conn" style={{ color: header.connection_colour }}>
        {header.connection}
      </span>
      {/* Where the chips already are, so a reader who only checks the strip gets
          it. The status bar says the same thing at the bottom. */}
      {frozen ? <span className="chip is-frozen">DATA FROZEN</span> : null}
      {/* A dash, because the last view's speed is not the replay's speed once the
          views stop arriving - `2.00x · PLAYING` is an assertion that it is still
          running. */}
      <span className="chip">{frozen ? "—" : header.playback}</span>
      <span className="chip">{header.lap}</span>
    </header>
  );
}
