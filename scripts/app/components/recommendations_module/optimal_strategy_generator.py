import plotly.figure_factory as ff
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def generate_optimal_strategy(recommendations):
    """
    Given a DataFrame of recommendations, select a compatible subset that forms an optimal strategy.
    Ensures at least one pit stop and at least two different compounds are used.
    Returns:
        - optimal_recs: List[dict] of selected recommendations
        - summary: str, narrative summary
        - stints: List[dict] with stint info (start_lap, end_lap, actions)
    """
    if recommendations is None or recommendations.empty:
        return [], "No recommendations available.", []

    # Sort by lap, then priority (high first), then confidence (high first)
    recs = recommendations.sort_values(
        ["LapNumber", "priority", "confidence"], ascending=[True, False, False]
    ).to_dict(orient="records")

    optimal_recs = []
    last_action_per_lap = {}
    incompatible_actions = {
        "pit_stop": ["extend_stint", "perform_overcut", "perform_undercut", "defensive_pit", "consider_pit"],
        "extend_stint": ["pit_stop", "perform_overcut", "perform_undercut", "defensive_pit", "consider_pit"],
        "perform_undercut": ["pit_stop", "extend_stint", "perform_overcut", "defensive_pit", "consider_pit"],
        "perform_overcut": ["pit_stop", "extend_stint", "perform_undercut", "defensive_pit", "consider_pit"],
        "defensive_pit": ["pit_stop", "extend_stint", "perform_overcut", "perform_undercut", "consider_pit"],
        "consider_pit": ["pit_stop", "extend_stint", "perform_overcut", "perform_undercut", "defensive_pit"],
    }

    # Step 1: Select optimal recommendations (no conflicts)
    for rec in recs:
        lap = rec.get("LapNumber")
        action = rec.get("action")
        if lap in last_action_per_lap:
            continue
        conflict = False
        for opt in optimal_recs:
            if abs(opt["LapNumber"] - lap) <= 2:
                if action in incompatible_actions.get(opt["action"], []):
                    conflict = True
                    break
        if not conflict:
            optimal_recs.append(rec)
            last_action_per_lap[lap] = action

    # Step 2: Ensure at least one pit stop
    has_pit = any(rec["action"] in ["pit_stop", "defensive_pit"]
                  for rec in optimal_recs)
    if not has_pit:
        pit_recs = [r for r in recs if r["action"]
                    in ["pit_stop", "defensive_pit"]]
        if pit_recs:
            best_pit = sorted(
                pit_recs, key=lambda x: (-x["priority"], -x["confidence"]))[0]
            optimal_recs.append(best_pit)
            optimal_recs = sorted(optimal_recs, key=lambda x: x["LapNumber"])

    # Step 3: Build stints and check compounds
    stints = []
    current_stint = {"start_lap": None, "end_lap": None, "actions": []}
    compounds_used = set()
    for i, rec in enumerate(optimal_recs):
        lap = rec["LapNumber"]
        action = rec["action"]
        compound = rec.get("CompoundName")
        if current_stint["start_lap"] is None:
            current_stint["start_lap"] = lap
        current_stint["end_lap"] = lap
        current_stint["actions"].append(rec)
        if compound:
            compounds_used.add(compound)
        if action in ["pit_stop", "defensive_pit"]:
            stints.append(current_stint)
            current_stint = {"start_lap": lap +
                             1, "end_lap": None, "actions": []}
    if current_stint["actions"]:
        stints.append(current_stint)

    # Step 4: Ensure at least two different compounds
    if len(compounds_used) < 2:
        all_compounds = set(r.get("CompoundName")
                            for r in recs if r.get("CompoundName"))
        missing = all_compounds - compounds_used
        if missing:
            for rec in recs:
                if rec.get("CompoundName") in missing and rec["action"] in ["pit_stop", "defensive_pit"]:
                    optimal_recs.append(rec)
                    break
            # Rebuild stints and compounds_used
            optimal_recs = sorted(optimal_recs, key=lambda x: x["LapNumber"])
            stints = []
            current_stint = {"start_lap": None, "end_lap": None, "actions": []}
            compounds_used = set()
            for i, rec in enumerate(optimal_recs):
                lap = rec["LapNumber"]
                action = rec["action"]
                compound = rec.get("CompoundName")
                if current_stint["start_lap"] is None:
                    current_stint["start_lap"] = lap
                current_stint["end_lap"] = lap
                current_stint["actions"].append(rec)
                if compound:
                    compounds_used.add(compound)
                if action in ["pit_stop", "defensive_pit"]:
                    stints.append(current_stint)
                    current_stint = {"start_lap": lap +
                                     1, "end_lap": None, "actions": []}
            if current_stint["actions"]:
                stints.append(current_stint)

    # Step 5: Generate narrative summary with stints and metrics
    summary = "Optimal race strategy:\n"
    for idx, stint in enumerate(stints):
        stint_compounds = set(rec.get("CompoundName")
                              for rec in stint["actions"] if rec.get("CompoundName"))
        compounds_str = ", ".join(
            sorted(stint_compounds)) if stint_compounds else "Unknown"
        summary += f"\n**Stint {idx+1}**: Laps {stint['start_lap']} - {stint['end_lap']} | Compound(s): {compounds_str}\n"
        for rec in stint["actions"]:
            metrics = []
            if "projected_lap_time" in rec:
                metrics.append(f"LapTime: {rec['projected_lap_time']:.2f}s")
            if "expected_position" in rec:
                metrics.append(f"Pos: {rec['expected_position']}")
            if "projected_degradation" in rec:
                metrics.append(
                    f"Degradation: {rec['projected_degradation']:.3f}/lap")
            metrics_str = f" [{' | '.join(metrics)}]" if metrics else ""
            summary += f"- Lap {rec['LapNumber']}: {rec['action'].replace('_', ' ').title()} (Priority {rec['priority']}, Confidence {rec['confidence']:.2f}){metrics_str}\n"
            if rec.get("explanation"):
                summary += f"    {rec['explanation']}\n"
        if idx < len(stints) - 1:
            summary += f"  → Pit Stop (Transition to Stint {idx+2})\n"

    return optimal_recs, summary, stints


def plot_optimal_strategy_gantt(stints):
    """
    Visualize stints and pit stops as a Gantt chart using Plotly.
    Each stint is a block; pit stops appear as annotations.
    Colors can be set by compound or action.
    """
    if not stints:
        st.info("No stints to display.")
        return

    # Example color map for compounds (customize as needed)
    compound_colors = {
        "Soft": "#FF3333",
        "Medium": "#FFD700",
        "Hard": "#AAAAAA",
        "Intermediate": "#33FF99",
        "Wet": "#3385FF"
    }

    bars = []
    y_labels = []
    annotations = []
    for idx, stint in enumerate(stints):
        start = stint['start_lap']
        end = stint['end_lap']
        # Default label
        desc = f"Stint {idx+1}"
        # Try to get compound name from first action in stint
        compound = None
        for rec in stint["actions"]:
            if "CompoundName" in rec:
                compound = rec["CompoundName"]
                break
        color = compound_colors.get(str(compound).capitalize(), "#888888")
        # Tooltip: show actions in this stint
        actions = ", ".join([a["action"].replace("_", " ").title()
                            for a in stint["actions"]])
        bars.append(
            dict(
                x=[start, end],
                y=[desc, desc],
                mode='lines',
                line=dict(color=color, width=20),
                showlegend=False,
                hoverinfo='text',
                text=f"{desc}<br>Compound: {compound or 'Unknown'}<br>Actions: {actions}"
            )
        )
        y_labels.append(desc)
        # Add pit stop annotation at end of stint (except last)
        if idx < len(stints) - 1:
            annotations.append(
                dict(
                    x=end,
                    y=desc,
                    text="Pit Stop",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                    font=dict(color="red", size=12),
                    bgcolor="rgba(255,255,255,0.7)"
                )
            )

    fig = go.Figure()
    for bar in bars:
        fig.add_trace(go.Scatter(**bar))

    fig.update_layout(
        title="Optimal Strategy Gantt Chart",
        xaxis=dict(
            title="Lap Number",
            type="linear",
            tickmode="linear",
            dtick=5  # adjust as needed
        ),
        yaxis=dict(
            title="Stint",
            tickvals=y_labels,
            ticktext=y_labels
        ),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="#22222A",
        paper_bgcolor="#22222A",
        font=dict(color="white"),
        annotations=annotations
    )
    st.plotly_chart(fig, use_container_width=True)
