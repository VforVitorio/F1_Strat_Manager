/**
 * Band 4's corner: where the whole field is on the lap, right now.
 *
 * The traces answer what the own car is DOING; nothing else in this window
 * answers where everyone IS. `PITWALL_V2_ARCHITECTURE.md` 3.4 lists this as
 * one of only two DATA panels that are a function of the instant rather than
 * of the lap.
 *
 * **It is a schematic ring, not the circuit, and that is a constraint rather
 * than a preference.** `SessionData.ref_lap_xy`, `circuit_rotation_deg`,
 * `ref_lap_drs` and per-frame `x`/`y` all exist in the loader and NONE of
 * them crosses the wire - the pyglet map draws them in-process, next to this
 * window. An outline ring is a new host capability, not a tweak.
 *
 * **What the angle is honest about.** `rel_dist` is the fraction of the car's
 * OWN current-lap length, not of the circuit. Measured over 921 laps of
 * Melbourne 2025, a dot therefore sits a median of 1.3 degrees from its true
 * circuit position, and up to 24 degrees off on a pit lap (PIA's lap 44, 5565
 * driven metres against a 5220 m circuit). Correct for a schematic; do not
 * grow anything on top of this that claims geometric fidelity.
 *
 * Colours come from `driver_colors` on the wire, which the producer publishes
 * out of `src/arcade/palette.py` precisely so no consumer hardcodes a sixth
 * copy of the palette.
 */

import type { ArcadeState } from "../../lib/bridge";
import { driverStatus, type DriverStatus } from "../../lib/driverStatus";

/** SVG user units. The viewBox is square and the ring is centred in it. */
const SIZE = 200;
const CENTRE = SIZE / 2;
const RADIUS = 78;
const DOT_R = 5;
const MAIN_DOT_R = 7;

interface Dot {
  code: string;
  x: number;
  y: number;
  colour: string;
  status: DriverStatus;
  /** null for the eighteen unlabelled cars; otherwise which side its code goes. */
  label: "above" | "below" | null;
}

/**
 * Place a fraction of the lap on the circle.
 *
 * Start/finish sits at twelve o'clock and the lap runs clockwise, which is
 * the convention every timing graphic uses and costs one rotation: SVG's
 * angle zero points right, so the offset is a quarter turn back.
 */
function place(fraction: number): { x: number; y: number } {
  const radians = (fraction * 360 - 90) * (Math.PI / 180);
  return {
    x: CENTRE + RADIUS * Math.cos(radians),
    y: CENTRE + RADIUS * Math.sin(radians),
  };
}

function rgb(colour: [number, number, number] | undefined): string {
  return colour ? `rgb(${colour[0]}, ${colour[1]}, ${colour[2]})` : "var(--qt-fg-2)";
}

export function TrackRing({ arcade }: { arcade: ArcadeState }) {
  // The two labelled cars go on OPPOSITE sides of their dots. They are the
  // main driver and the car chosen to compare against, so they are routinely
  // seconds apart - on the session this was built against, NOR and PIA sit
  // 0.006 of a lap apart and both codes rendered above their dots, on top of
  // each other and of the dots. Fixed sides beat a collision test: it is
  // deterministic, so the same car is always in the same place.
  const labelSide = (code: string): "above" | "below" | null => {
    if (code === arcade.driver_main) return "above";
    if (code === arcade.driver_rival) return "below";
    return null;
  };

  // `race_order` rather than the drivers map, so the DOM order is the running
  // order and the leader's dot paints last - on top of whoever it is lapping.
  const ordered = [...arcade.race_order].reverse();
  const dots: Dot[] = [];
  const blind: string[] = [];

  for (const code of ordered) {
    const car = arcade.drivers[code];
    if (!car) continue;
    // Unknown is not zero. `rel_dist` is null when the telemetry never placed
    // the car, and a dot at fraction 0 would draw it exactly on the start
    // line - a position a real car can hold, which is the sentinel collision
    // this repo has already paid for once.
    if (car.rel_dist === null) {
      blind.push(code);
      continue;
    }
    dots.push({
      code,
      ...place(car.rel_dist),
      colour: rgb(arcade.driver_colors[code]),
      status: driverStatus(car),
      label: labelSide(code),
    });
  }

  return (
    <aside className="ring card">
      <header className="ring-header">
        <span className="ring-title">TRACK</span>
        <span className="ring-subtitle">schematic · lap fraction</span>
      </header>

      <svg className="ring-svg" viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="Field position around the lap">
        <circle className="ring-track" cx={CENTRE} cy={CENTRE} r={RADIUS} />
        {/* Start/finish, at twelve o'clock. */}
        <line
          className="ring-line"
          x1={CENTRE}
          y1={CENTRE - RADIUS - 6}
          x2={CENTRE}
          y2={CENTRE - RADIUS + 6}
        />
        <text className="ring-lap" x={CENTRE} y={CENTRE - 2} textAnchor="middle">
          {arcade.lap}
        </text>
        <text className="ring-lap-total" x={CENTRE} y={CENTRE + 14} textAnchor="middle">
          / {arcade.total_laps}
        </text>

        {dots.map((dot) => (
          <g key={dot.code} className={`ring-dot is-${dot.status}`}>
            <circle
              cx={dot.x}
              cy={dot.y}
              r={dot.label ? MAIN_DOT_R : DOT_R}
              // `out` is the only state drawn hollow, so the fill is the one
              // that has to change with it rather than an opacity on the group.
              fill={dot.status === "out" ? "none" : dot.colour}
              stroke={dot.colour}
              data-code={dot.code}
              data-status={dot.status}
            />
            {dot.label ? (
              <text
                className="ring-code"
                x={dot.x}
                y={dot.label === "above" ? dot.y - 11 : dot.y + 17}
                textAnchor="middle"
              >
                {dot.code}
              </text>
            ) : null}
          </g>
        ))}
      </svg>

      <footer className="ring-legend">
        <span className="ring-key is-running">● running</span>
        <span className="ring-key is-finished">● finished</span>
        <span className="ring-key is-out">○ out</span>
      </footer>
      {blind.length ? <p className="ring-blind">NO POSITION: {blind.join(", ")}</p> : null}
    </aside>
  );
}
