/**
 * The field's best sector and lap times, ranked, over what the clock revealed.
 *
 * One module because two panels need the identical answer: the bests panel
 * ranks it, and the tower colours a sector cell purple when it matches. Two
 * components each reducing over `bulk.drivers` is the twin this repo keeps
 * paying for - the day one of them starts excluding deleted laps and the
 * other does not, the tower paints a purple that the panel does not list.
 *
 * **Recomputed, never read off `IsPersonalBest`.** That flag is a running one
 * and it is safe under masking, but the two sequences are not identical:
 * measured on Melbourne 2025 they differ on 47 lap-flags across all twenty
 * drivers, in both directions, concentrated on the wet-start laps. They
 * converge only at the final frame, and mid-race is the only state a masked
 * panel ever renders. The exclusions the recompute needs - a deleted time
 * does not count, a generated row has no time at all - are already applied by
 * the reader that produced `best`.
 */

import type { Bulk } from "./bridge";

/** The fields a bests section can rank. `lap_time` is the fourth section. */
export type BestField = "s1" | "s2" | "s3" | "lap_time";

export interface BestEntry {
  code: string;
  value: number;
  /** The compound the fastest LAP was set on; null for a sector - see below. */
  compound: string | null;
}

export type SessionBests = Record<BestField, BestEntry[]>;

const FIELDS: BestField[] = ["s1", "s2", "s3", "lap_time"];

/**
 * Every driver's best per field, fastest first, drivers with none omitted.
 *
 * The compound is only carried on the lap ranking. The reader serves the
 * compound of a driver's fastest LAP, which is the right answer there and a
 * wrong one for a sector - his best S1 may well have been set on another set
 * entirely. Rather than attach a plausible-looking compound to a sector, the
 * sector rankings carry none, and the panel renders the column empty for them.
 */
export function sessionBests(bulk: Bulk | null): SessionBests {
  const ranked = { s1: [], s2: [], s3: [], lap_time: [] } as SessionBests;
  if (!bulk?.available) return ranked;

  for (const field of FIELDS) {
    const entries: BestEntry[] = [];
    for (const [code, driver] of Object.entries(bulk.drivers)) {
      const value = driver.best[field];
      if (value === null) continue;
      entries.push({
        code,
        value,
        compound: field === "lap_time" ? driver.best.compound : null,
      });
    }
    entries.sort((left, right) => left.value - right.value);
    ranked[field] = entries;
  }
  return ranked;
}

/**
 * The ideal lap: the field's best S1, S2 and S3 recombined.
 *
 * Null while any of the three is still unknown rather than a partial sum,
 * which would read as a lap somebody nearly drove. Lap 1 has no S1 for any
 * driver by construction, so this stays null until the field has completed
 * two laps - which is correct and not a bug to paper over.
 */
export function theoreticalLap(bests: SessionBests): number | null {
  const sectors = [bests.s1[0], bests.s2[0], bests.s3[0]];
  if (sectors.some((entry) => entry === undefined)) return null;
  return sectors.reduce((total, entry) => total + entry.value, 0);
}
