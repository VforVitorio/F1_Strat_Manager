import streamlit as st
import plotly.graph_objects as go
import numpy as np
from collections import Counter


def select_strategies_ui(all_strategies):
    """
    UI component: Allow the user to select two strategies to compare.
    Returns:
        strategy_a_name, strategy_b_name, strategy_a, strategy_b
    """
    strategy_names = list(all_strategies.keys())
    if len(strategy_names) < 2:
        st.warning("At least two strategies are required for comparison.")
        return None, None, None, None

    col1, col2 = st.columns(2)
    with col1:
        strategy_a_name = st.selectbox(
            "Select Strategy A", strategy_names, key="strategy_a")
    with col2:
        strategy_b_name = st.selectbox("Select Strategy B", strategy_names, index=1 if len(
            strategy_names) > 1 else 0, key="strategy_b")

    if strategy_a_name == strategy_b_name:
        st.warning("Please select two different strategies.")
        return None, None, None, None

    strategy_a = all_strategies[strategy_a_name]
    strategy_b = all_strategies[strategy_b_name]
    return strategy_a_name, strategy_b_name, strategy_a, strategy_b


def extract_strategy_features(recommendations):
    """
    Extracts key features from a strategy for comparison.
    Returns a dict with:
        - First Pit Stop Lap
        - Number of Pit Stops
        - Shortest Stint (laps)
        - Longest Stint (laps)
        - Actions count by type
        - Sequence of actions (lap, action)
    """
    if not recommendations:
        return {}

    pit_laps = [rec["LapNumber"] for rec in recommendations if rec.get(
        "action") in ["pit_stop", "defensive_pit"]]
    first_pit = min(pit_laps) if pit_laps else None
    num_pits = len(pit_laps)

    # Build stints
    laps = [rec["LapNumber"] for rec in recommendations]
    laps = sorted(laps)
    if pit_laps:
        stints = []
        last_pit = 0
        for pit in sorted(pit_laps):
            stints.append(pit - last_pit)
            last_pit = pit
        stints.append(laps[-1] - last_pit + 1)
        shortest_stint = min(stints)
        longest_stint = max(stints)
    else:
        shortest_stint = longest_stint = laps[-1] - laps[0] + 1 if laps else 0

    # Count actions
    action_counts = Counter([rec["action"] for rec in recommendations])

    # Sequence of actions
    action_seq = [(rec["LapNumber"], rec["action"]) for rec in recommendations]

    return {
        "First Pit Stop Lap": first_pit,
        "Number of Pit Stops": num_pits,
        "Shortest Stint (laps)": shortest_stint,
        "Longest Stint (laps)": longest_stint,
        "Actions": dict(action_counts),
        "Action Sequence": action_seq,
        "Number of Actions": len(recommendations)
    }


def render_comparison_table(metrics_a, metrics_b, comparison, name_a="Strategy A", name_b="Strategy B"):
    """
    Show a table comparing key features of both strategies side by side.
    """
    rows = []
    keys = [
        "First Pit Stop Lap",
        "Number of Pit Stops",
        "Shortest Stint (laps)",
        "Longest Stint (laps)",
        "Number of Actions"
    ]
    for key in keys:
        val_a = metrics_a.get(key, "N/A")
        val_b = metrics_b.get(key, "N/A")
        diff = None
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            diff = val_a - val_b
        else:
            diff = "-"
        rows.append({
            "Metric": key,
            name_a: val_a,
            name_b: val_b,
            "Difference (A - B)": diff
        })
    # Add action counts
    all_actions = set(metrics_a.get("Actions", {}).keys()) | set(
        metrics_b.get("Actions", {}).keys())
    for action in sorted(all_actions):
        val_a = metrics_a.get("Actions", {}).get(action, 0)
        val_b = metrics_b.get("Actions", {}).get(action, 0)
        diff = val_a - val_b
        rows.append({
            "Metric": f"Actions: {action.replace('_',' ').title()}",
            name_a: val_a,
            name_b: val_b,
            "Difference (A - B)": diff
        })
    st.dataframe(rows, use_container_width=True)


def plot_strategy_action_timeline(metrics_a, metrics_b, name_a="Strategy A", name_b="Strategy B"):
    """
    Visualize both strategies' action sequences as a timeline.
    """
    fig = go.Figure()
    # Strategy A
    if metrics_a.get("Action Sequence"):
        laps_a = [x[0] for x in metrics_a["Action Sequence"]]
        actions_a = [x[1].replace('_', ' ').title()
                     for x in metrics_a["Action Sequence"]]
        fig.add_trace(go.Scatter(
            x=laps_a, y=actions_a, mode="markers+lines", name=name_a,
            marker=dict(size=12, symbol="circle", color="blue")
        ))
    # Strategy B
    if metrics_b.get("Action Sequence"):
        laps_b = [x[0] for x in metrics_b["Action Sequence"]]
        actions_b = [x[1].replace('_', ' ').title()
                     for x in metrics_b["Action Sequence"]]
        fig.add_trace(go.Scatter(
            x=laps_b, y=actions_b, mode="markers+lines", name=name_b,
            marker=dict(size=12, symbol="diamond", color="red")
        ))
    fig.update_layout(
        title="Strategy Action Timeline",
        xaxis_title="Lap",
        yaxis_title="Action",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_strategy_comparison(all_strategies, race_data):
    """
    Main function: Orchestrates the comparison workflow.
    """
    st.header("Strategy A/B Comparator")
    strategy_a_name, strategy_b_name, strategy_a, strategy_b = select_strategies_ui(
        all_strategies)
    if not strategy_a or not strategy_b:
        return

    st.subheader("Strategy Features Comparison")
    metrics_a = extract_strategy_features(strategy_a)
    metrics_b = extract_strategy_features(strategy_b)
    comparison = {k: (metrics_a.get(k, 0) - metrics_b.get(k, 0)) if isinstance(metrics_a.get(k, 0),
                                                                               (int, float)) and isinstance(metrics_b.get(k, 0), (int, float)) else "-" for k in metrics_a}

    st.subheader("Action Timeline Comparison")
    plot_strategy_action_timeline(
        metrics_a, metrics_b, name_a=strategy_a_name, name_b=strategy_b_name)

    st.subheader("Summary Table")
    render_comparison_table(metrics_a, metrics_b, comparison,
                            name_a=strategy_a_name, name_b=strategy_b_name)


def generate_aggressive_strategy(recommendations):
    """
    Selects an aggressive subset of recommendations: highest priority, earliest laps, resolves conflicts.
    Returns:
        - aggressive_recs: List[dict] of selected recommendations
        - summary: str, narrative summary
        - stints: List[dict] with stint info (start_lap, end_lap, actions)
    """
    if recommendations is None or recommendations.empty:
        return [], "No recommendations available.", []

    recs = recommendations.sort_values(
        ["priority", "LapNumber", "confidence"], ascending=[False, True, False]
    ).to_dict(orient="records")

    # Use similar conflict resolution as in generate_optimal_strategy
    aggressive_recs = []
    last_action_per_lap = {}
    incompatible_actions = {
        "pit_stop": ["extend_stint", "perform_overcut", "perform_undercut", "defensive_pit", "consider_pit"],
        "extend_stint": ["pit_stop", "perform_overcut", "perform_undercut", "defensive_pit", "consider_pit"],
        "perform_undercut": ["pit_stop", "extend_stint", "perform_overcut", "defensive_pit", "consider_pit"],
        "perform_overcut": ["pit_stop", "extend_stint", "perform_undercut", "defensive_pit", "consider_pit"],
        "defensive_pit": ["pit_stop", "extend_stint", "perform_overcut", "perform_undercut", "consider_pit"],
        "consider_pit": ["pit_stop", "extend_stint", "perform_overcut", "perform_undercut", "defensive_pit"],
    }

    for rec in recs:
        lap = rec.get("LapNumber")
        action = rec.get("action")
        if lap in last_action_per_lap:
            continue
        conflict = False
        for opt in aggressive_recs:
            if abs(opt["LapNumber"] - lap) <= 2:
                if action in incompatible_actions.get(opt["action"], []):
                    conflict = True
                    break
        if not conflict:
            aggressive_recs.append(rec)
            last_action_per_lap[lap] = action

    # Build stints (for swimlane, not for compound)
    stints = []
    current_stint = {"start_lap": None, "end_lap": None, "actions": []}
    for i, rec in enumerate(aggressive_recs):
        lap = rec["LapNumber"]
        action = rec["action"]
        if current_stint["start_lap"] is None:
            current_stint["start_lap"] = lap
        current_stint["end_lap"] = lap
        current_stint["actions"].append(rec)
        if action in ["pit_stop", "defensive_pit"]:
            stints.append(current_stint)
            current_stint = {"start_lap": lap +
                             1, "end_lap": None, "actions": []}
    if current_stint["actions"]:
        stints.append(current_stint)

    # Narrative summary
    summary = "Aggressive race strategy:\n"
    for idx, stint in enumerate(stints):
        summary += f"\n**Stint {idx+1}**: Laps {stint['start_lap']} - {stint['end_lap']}\n"
        for rec in stint["actions"]:
            metrics = []
            if "projected_lap_time" in rec:
                metrics.append(f"LapTime: {rec['projected_lap_time']:.2f}s")
            if "expected_position" in rec:
                metrics.append(f"Pos: {rec['expected_position']}")
            if "projected_degradation" in rec:
                metrics.append(
                    f"Degradation: {rec['projected_degradation']:.3f}/lap")
            rec_metrics_str = f" [{' | '.join(metrics)}]" if metrics else ""
            summary += f"- Lap {rec['LapNumber']}: {rec['action'].replace('_', ' ').title()} (Priority {rec['priority']}, Confidence {rec['confidence']:.2f}){rec_metrics_str}\n"
            if rec.get("explanation"):
                summary += f"    {rec['explanation']}\n"
        if idx < len(stints) - 1:
            summary += f"  → Pit Stop (Transition to Stint {idx+2})\n"

    return aggressive_recs, summary, stints


def generate_conservative_strategy(recommendations):
    """
    Selects a conservative subset of recommendations: lowest priority, latest laps, resolves conflicts.
    Returns:
        - conservative_recs: List[dict] of selected recommendations
        - summary: str, narrative summary
        - stints: List[dict] with stint info (start_lap, end_lap, actions)
    """
    if recommendations is None or recommendations.empty:
        return [], "No recommendations available.", []

    recs = recommendations.sort_values(
        ["priority", "LapNumber", "confidence"], ascending=[True, False, False]
    ).to_dict(orient="records")

    # Use similar conflict resolution as in generate_optimal_strategy
    conservative_recs = []
    last_action_per_lap = {}
    incompatible_actions = {
        "pit_stop": ["extend_stint", "perform_overcut", "perform_undercut", "defensive_pit", "consider_pit"],
        "extend_stint": ["pit_stop", "perform_overcut", "perform_undercut", "defensive_pit", "consider_pit"],
        "perform_undercut": ["pit_stop", "extend_stint", "perform_overcut", "defensive_pit", "consider_pit"],
        "perform_overcut": ["pit_stop", "extend_stint", "perform_undercut", "defensive_pit", "consider_pit"],
        "defensive_pit": ["pit_stop", "extend_stint", "perform_overcut", "perform_undercut", "consider_pit"],
        "consider_pit": ["pit_stop", "extend_stint", "perform_overcut", "perform_undercut", "defensive_pit"],
    }

    for rec in recs:
        lap = rec.get("LapNumber")
        action = rec.get("action")
        if lap in last_action_per_lap:
            continue
        conflict = False
        for opt in conservative_recs:
            if abs(opt["LapNumber"] - lap) <= 2:
                if action in incompatible_actions.get(opt["action"], []):
                    conflict = True
                    break
        if not conflict:
            conservative_recs.append(rec)
            last_action_per_lap[lap] = action

    # Build stints (for swimlane, not for compound)
    stints = []
    current_stint = {"start_lap": None, "end_lap": None, "actions": []}
    for i, rec in enumerate(conservative_recs):
        lap = rec["LapNumber"]
        action = rec["action"]
        if current_stint["start_lap"] is None:
            current_stint["start_lap"] = lap
        current_stint["end_lap"] = lap
        current_stint["actions"].append(rec)
        if action in ["pit_stop", "defensive_pit"]:
            stints.append(current_stint)
            current_stint = {"start_lap": lap +
                             1, "end_lap": None, "actions": []}
    if current_stint["actions"]:
        stints.append(current_stint)

    # Narrative summary
    summary = "Conservative race strategy:\n"
    for idx, stint in enumerate(stints):
        summary += f"\n**Stint {idx+1}**: Laps {stint['start_lap']} - {stint['end_lap']}\n"
        for rec in stint["actions"]:
            metrics = []
            if "projected_lap_time" in rec:
                metrics.append(f"LapTime: {rec['projected_lap_time']:.2f}s")
            if "expected_position" in rec:
                metrics.append(f"Pos: {rec['expected_position']}")
            if "projected_degradation" in rec:
                metrics.append(
                    f"Degradation: {rec['projected_degradation']:.3f}/lap")
            rec_metrics_str = f" [{' | '.join(metrics)}]" if metrics else ""
            summary += f"- Lap {rec['LapNumber']}: {rec['action'].replace('_', ' ').title()} (Priority {rec['priority']}, Confidence {rec['confidence']:.2f}){rec_metrics_str}\n"
            if rec.get("explanation"):
                summary += f"    {rec['explanation']}\n"
        if idx < len(stints) - 1:
            summary += f"  → Pit Stop (Transition to Stint {idx+2})\n"

    return conservative_recs, summary, stints
