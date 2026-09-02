/**
 * What an empty window says, and it depends on whether the socket is up (#1004).
 *
 * Both windows paint in about 1.7 s and the arcade's first tick lands at about
 * 11 s, because the producer spends that long unpickling a 382 MB session. The
 * socket itself is accepted at about 8 s. So there are two different states
 * behind one blank window, measured on the real path:
 *
 *   0.4 -> 8.0 s   no producer at all. `get_connection()` says "Connecting...".
 *   8.0 -> 11.0 s  the arcade is there and loading. It says "Connected".
 *
 * The window used to print one sentence across both, and that sentence told the
 * reader to start a replay - an instruction that is right for the first three
 * seconds of the wait and wrong for the last three, when the thing it asks for
 * has been running since 8 s. Nine seconds of a window that looks hung is the
 * complaint; being told to fix it by doing what you already did is what turns a
 * slow start into a broken one.
 *
 * One module rather than a string in each window, because there are three copy
 * sites across the two of them and this repo's most expensive recurring defect
 * is the copy that got the fix and the twin that did not.
 */

/** True once the socket has been accepted, whatever the tick channel is doing. */
export function isSocketUp(connection: string | null): boolean {
  return connection === "Connected";
}

/**
 * The line under an empty window.
 *
 * Actionable only when there is something to act on. Once the socket is up the
 * honest answer is that there is nothing to do, so the sentence says what is
 * happening instead of what to type.
 */
export function waitingBody(connection: string | null): string {
  return isSocketUp(connection)
    ? "Connected to the arcade. It is loading the session; the first tick follows."
    : "Waiting for the arcade broadcast. Start a replay with --strategy.";
}

/** The status bar's version of the same two states, short enough for one line. */
export function waitingStatus(connection: string | null): string {
  return isSocketUp(connection)
    ? "Connected · the arcade is loading its session"
    : "Waiting for arcade stream…";
}
