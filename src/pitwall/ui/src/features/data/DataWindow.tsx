/**
 * PITWALL · DATA - the four-band shell.
 *
 * Sprint 2 renders the vertical slice. Bands 1 and 2 (status strip, timing
 * table, bests) land in sprint 4, band 3 (race pace) in sprint 5, band 4
 * (own-car traces and the ring) in sprint 6.
 */

import { SliceProof } from "../SliceProof";
import { useTick } from "../../lib/useTick";

export function DataWindow() {
  return <SliceProof title="PITWALL · DATA" state={useTick()} />;
}
