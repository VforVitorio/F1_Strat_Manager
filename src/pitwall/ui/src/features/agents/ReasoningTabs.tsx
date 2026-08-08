/**
 * The six reasoning tabs, with the Qt highlighter's colours.
 *
 * Both halves are decided host side: which tabs exist, what is in them —
 * including the decision-memory block that only appears on a lap where
 * the call changed — and where every coloured run starts and ends.
 *
 * The highlighter arrives as **typed segments**, not as HTML. Gate B
 * offered either; segments mean an LLM's free text never becomes markup
 * on the way to a webview, so this renders `<span>`s rather than trusting
 * `dangerouslySetInnerHTML` with a model's output.
 *
 * The selected tab is the only piece of state this window owns, because
 * it is the only thing that belongs to the person looking at it rather
 * than to the race.
 */

import { useState } from "react";
import type { ReasoningTab } from "../../lib/agents";

export function ReasoningTabs({ tabs }: { tabs: ReasoningTab[] }) {
  const [selected, setSelected] = useState(0);
  const active = tabs[selected] ?? tabs[0];

  return (
    <section className="reasoning">
      <div className="reasoning-tabbar" role="tablist">
        {tabs.map((tab, index) => (
          <button
            key={tab.key}
            role="tab"
            aria-selected={index === selected}
            className={index === selected ? "reasoning-tab is-selected" : "reasoning-tab"}
            onClick={() => setSelected(index)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="reasoning-body" role="tabpanel">
        {active?.segments.map((segment, index) => (
          <span
            key={index}
            style={{ color: segment.colour, fontWeight: segment.bold ? 700 : undefined }}
          >
            {segment.text}
          </span>
        ))}
      </div>
    </section>
  );
}
