import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def generate_optimal_strategy(recommendations):
    """
    Selects a compatible subset of recommendations for an optimal strategy.
    Returns:
        - optimal_recs: List[dict] of selected recommendations
        - summary: str, narrative summary
        - stints: List[dict] with stint info (start_lap, end_lap, actions)
    """
    if recommendations is None or recommendations.empty:
        return [], "No recommendations available.", []

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

    # Ensure at least one pit stop
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

    # Build stints (for swimlane, not for compound)
    stints = []
    current_stint = {"start_lap": None, "end_lap": None, "actions": []}
    for i, rec in enumerate(optimal_recs):
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
    summary = "Optimal race strategy:\n"
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

    return optimal_recs, summary, stints


def plot_optimal_strategy_step_chart(optimal_recs):
    """
    Step chart: X = LapNumber, Y = Action (as categorical), color by priority/confidence.
    """
    if not optimal_recs:
        st.info("No optimal strategy to display.")
        return

    df = pd.DataFrame(optimal_recs)
    df = df.sort_values("LapNumber")
    df["ActionLabel"] = df["action"].apply(
        lambda x: x.replace("_", " ").title())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["LapNumber"],
        y=df["ActionLabel"],
        mode="lines+markers",
        marker=dict(
            size=10,
            color=df["priority"],
            colorscale="Bluered",
            colorbar=dict(title="Priority"),
        ),
        line_shape="hv",
        text=[f"Lap {row.LapNumber}<br>Priority: {row.priority}<br>Confidence: {row.confidence:.2f}" for row in df.itertuples()],
        hoverinfo="text"
    ))
    fig.update_layout(
        title="Optimal Strategy Step Chart",
        xaxis_title="Lap Number",
        yaxis_title="Action",
        height=350,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_optimal_strategy_swimlane(optimal_recs):
    """
    Swimlane chart: Each action type is a lane, blocks show when each action is active.
    """
    if not optimal_recs:
        st.info("No optimal strategy to display.")
        return

    df = pd.DataFrame(optimal_recs)
    df = df.sort_values("LapNumber")
    action_types = df["action"].unique()
    fig = go.Figure()
    for i, action in enumerate(action_types):
        action_df = df[df["action"] == action]
        fig.add_trace(go.Bar(
            x=action_df["LapNumber"],
            y=[action.replace("_", " ").title()] * len(action_df),
            orientation="h",
            marker=dict(
                color=f"hsl({(i*60)%360},70%,50%)"
            ),
            name=action.replace("_", " ").title(),
            hovertext=[
                f"Lap {row.LapNumber}<br>Priority: {row.priority}<br>Confidence: {row.confidence:.2f}" for row in action_df.itertuples()],
            hoverinfo="text"
        ))
    fig.update_layout(
        title="Optimal Strategy Swimlane Chart",
        xaxis_title="Lap Number",
        yaxis_title="Action Type",
        barmode="stack",
        height=350,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_optimal_strategy_summary_table(optimal_recs):
    """
    Show a summary table of the optimal strategy.
    """
    if not optimal_recs:
        st.info("No optimal strategy to display.")
        return

    df = pd.DataFrame(optimal_recs)
    summary_cols = ["LapNumber", "action", "priority", "confidence",
                    "projected_lap_time", "expected_position", "projected_degradation", "explanation"]
    summary_cols = [col for col in summary_cols if col in df.columns]
    df = df[summary_cols]
    df = df.rename(columns={
        "LapNumber": "Lap",
        "action": "Action",
        "priority": "Priority",
        "confidence": "Confidence",
        "projected_lap_time": "LapTime",
        "expected_position": "Position",
        "projected_degradation": "Degradation",
        "explanation": "Explanation"
    })
    st.dataframe(df, use_container_width=True)
