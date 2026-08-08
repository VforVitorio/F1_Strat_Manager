/**
 * The 44 px top strip: session, driver, connection, playback, lap counter.
 *
 * 1:1 with `window.py::HeaderBar` — same order, same three chips on the
 * right, same three-state connection colour. Every string is built host
 * side; this places them.
 */

import type { HeaderView } from "../../lib/agents";

export function HeaderBar({ header }: { header: HeaderView }) {
  return (
    <header className="header-bar">
      <span className="header-session">{header.session}</span>
      <span className="header-driver">{header.driver}</span>
      <span className="header-spacer" />
      <span className="chip header-conn" style={{ color: header.connection_colour }}>
        {header.connection}
      </span>
      <span className="chip">{header.playback}</span>
      <span className="chip">{header.lap}</span>
    </header>
  );
}
