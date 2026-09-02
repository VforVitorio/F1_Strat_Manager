/**
 * A Qt status bar's `showMessage(text, 1500)`, in React. Shared by both windows.
 *
 * `showMessage` with a timeout clears itself, so a Qt window whose producer
 * dies goes quiet within a second and a half. The AGENTS port typed a
 * `transient` flag, documented it, and read it nowhere - so a dead producer
 * kept saying "lap N · streaming" forever under a red Disconnected chip.
 *
 * **The timer is keyed on `seq`, not on the text.** Qt re-arms `showMessage`
 * on EVERY broadcast, ten times a second, so the message stays visible while
 * streaming and clears 1.5 s after the last tick. Keyed on the text it
 * re-arms once per LAP, because the string does not change in between - so
 * the bar went blank 1.5 s into every lap and stayed blank for the other
 * eighty-odd seconds of it. That is the same bug inverted, and the first
 * version of this hook shipped it.
 *
 * An error or a waiting message is NOT transient: Qt gives those no timeout,
 * because those are the ones that must still be readable.
 *
 * It lives in `lib/` rather than beside the AGENTS window because the DATA
 * window's status bar is the same widget with the same timeout, and a second
 * copy of a hook that has already been wrong twice is not a copy worth
 * having.
 */

import { useEffect, useState } from "react";

export interface StatusMessage {
  text: string;
  transient: boolean;
}

export function useStatusText(status: StatusMessage, seq: number | null): string {
  const [shown, setShown] = useState(status.text);

  useEffect(() => {
    setShown(status.text);
    if (!status.transient) return;
    const timer = window.setTimeout(() => setShown(""), 1500);
    return () => window.clearTimeout(timer);
  }, [seq, status.text, status.transient]);

  return shown;
}
