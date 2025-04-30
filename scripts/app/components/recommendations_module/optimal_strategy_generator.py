def generate_optimal_strategy(recommendations):
    """
    Given a DataFrame of recommendations, select a compatible subset that forms an optimal strategy.
    Returns:
        - optimal_recs: List[dict] of selected recommendations
        - summary: str, narrative summary
    """
    if recommendations is None or recommendations.empty:
        return [], "No recommendations available."

    # Sort by lap, then priority (high first), then confidence (high first)
    recs = recommendations.sort_values(
        ["LapNumber", "priority", "confidence"], ascending=[True, False, False]
    ).to_dict(orient="records")

    optimal_recs = []
    used_laps = set()
    incompatible_actions = {
        "pit_stop": ["extend_stint", "perform_overcut", "perform_undercut", "defensive_pit", "consider_pit"],
        "extend_stint": ["pit_stop", "perform_overcut", "perform_undercut", "defensive_pit", "consider_pit"],
        "perform_undercut": ["pit_stop", "extend_stint", "perform_overcut", "defensive_pit", "consider_pit"],
        "perform_overcut": ["pit_stop", "extend_stint", "perform_undercut", "defensive_pit", "consider_pit"],
        "defensive_pit": ["pit_stop", "extend_stint", "perform_overcut", "perform_undercut", "consider_pit"],
        "consider_pit": ["pit_stop", "extend_stint", "perform_overcut", "perform_undercut", "defensive_pit"],
        # Add more incompatibilities as needed
    }

    last_action_per_lap = {}

    for rec in recs:
        lap = rec.get("LapNumber")
        action = rec.get("action")

        # Only one main action per lap (highest priority/confidence)
        if lap in last_action_per_lap:
            continue

        # Check for conflicts in a window of +/- 2 laps
        conflict = False
        for opt in optimal_recs:
            if abs(opt["LapNumber"] - lap) <= 2:
                if action in incompatible_actions.get(opt["action"], []):
                    conflict = True
                    break
        if not conflict:
            optimal_recs.append(rec)
            last_action_per_lap[lap] = action

    # Sort optimal_recs by lap number
    optimal_recs = sorted(optimal_recs, key=lambda x: x["LapNumber"])

    # Generate narrative summary
    summary = "Optimal race strategy:\n"
    for rec in optimal_recs:
        summary += f"- Lap {rec['LapNumber']}: {rec['action'].replace('_', ' ').title()} (Priority {rec['priority']}, Confidence {rec['confidence']:.2f})\n"
        if rec.get("explanation"):
            summary += f"    {rec['explanation']}\n"

    return optimal_recs, summary
