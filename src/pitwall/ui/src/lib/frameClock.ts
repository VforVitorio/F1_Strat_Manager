/**
 * The monotonic guard, and nothing else.
 *
 * Rewind in this product is the cheap kind - "so you do not miss something" -
 * not a study tool. Panels are therefore allowed to accumulate, and this is
 * what tells them when to throw the accumulation away.
 *
 * **Read before extending this.** The design originally specified
 * the guard in FRAMES, and a gate refuted that: the AGENTS window's two
 * accumulators (pace and tyre history) are keyed by LAP, and a
 * frame-indexed truncate cannot address a lap-keyed map. Worse, truncating
 * them destroys per-agent predictions that no channel can rebuild - the
 * history tail on the wire strips `per_agent`. So a lap-keyed store needs a
 * lap-keyed guard, and `discontinuity` below reports the lap alongside the
 * frame precisely so the caller can choose which one it evicts by.
 */

export interface ClockReading {
  frameIndex: number;
  lap: number;
}

export type Discontinuity =
  | { kind: "continuous" }
  /** The user seeked backwards, or the producer said it did. Evict the future. */
  | { kind: "rewound"; toFrame: number; toLap: number }
  /** A forward jump the wire could not carry. Whatever spans the hole is a lie. */
  | { kind: "gap"; droppedFrames: number };

export class FrameClock {
  private lastFrame = -1;
  private lastLap = -1;

  /**
   * Fold one tick in and say what the caller must do about it.
   *
   * `rewound` and `dropped` come from the producer, which knows things the
   * frame index alone does not: a backwards seek smaller than one frame is
   * invisible here, and a forward jump leaves the index moving forwards and
   * the sequence contiguous while the frames in between never existed.
   */
  advance(reading: ClockReading, rewound: boolean, dropped: number): Discontinuity {
    const wentBackwards = reading.frameIndex < this.lastFrame;
    this.lastFrame = reading.frameIndex;
    this.lastLap = reading.lap;

    if (rewound || wentBackwards) {
      return { kind: "rewound", toFrame: reading.frameIndex, toLap: reading.lap };
    }
    if (dropped > 0) {
      return { kind: "gap", droppedFrames: dropped };
    }
    return { kind: "continuous" };
  }

  get frame(): number {
    return this.lastFrame;
  }

  get lap(): number {
    return this.lastLap;
  }
}
