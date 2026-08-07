/**
 * PITWALL · AGENTS - the 1:1 port of the Qt strategy window.
 *
 * Sprint 2 renders the vertical slice. The real layout is frozen and lands in
 * sprint 3: HeaderBar, a 540/740 horizontal split, and a 3x2 grid of agent
 * cards. It is a port, not a redesign.
 */

import { SliceProof } from "../SliceProof";
import { useTick } from "../../lib/useTick";

export function AgentsWindow() {
  return <SliceProof title="PITWALL · AGENTS" state={useTick()} />;
}
