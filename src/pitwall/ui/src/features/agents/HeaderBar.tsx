/**
 * The 44 px top strip: session, driver, connection, playback, lap counter.
 *
 * 1:1 with `window.py::HeaderBar` — same order, same three chips on the
 * right, same three-state connection colour. Every string is built host
 * side; this places them.
 */

import type { HeaderView } from "../../lib/agents";

export function HeaderBar({ header, frozen = false }: { header: HeaderView; frozen?: boolean }) {
  return (
    <header className="header-bar">
      <span className="header-session">{header.session}</span>
      <span className="header-driver">{header.driver}</span>
      <span className="header-spacer" />
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
