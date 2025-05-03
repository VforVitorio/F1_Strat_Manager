from components.strategy_chat import send_message_to_llm
import sys
import os
import io
import base64
import streamlit as st
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# F1 2023 driver mapping (English)
F1_2023_DRIVERS_BY_TEAM = {
    "Red Bull Racing": [
        {"number": 1, "name": "Max Verstappen"},
        {"number": 11, "name": "Sergio Pérez"}
    ],
    "Mercedes-AMG Petronas F1 Team": [
        {"number": 44, "name": "Lewis Hamilton"},
        {"number": 63, "name": "George Russell"}
    ],
    "Scuderia Ferrari": [
        {"number": 16, "name": "Charles Leclerc"},
        {"number": 55, "name": "Carlos Sainz"}
    ],
    "McLaren Racing": [
        {"number": 4, "name": "Lando Norris"},
        {"number": 81, "name": "Oscar Piastri"}
    ],
    "Alpine F1 Team": [
        {"number": 10, "name": "Pierre Gasly"},
        {"number": 31, "name": "Esteban Ocon"}
    ],
    "Aston Martin F1 Team": [
        {"number": 14, "name": "Fernando Alonso"},
        {"number": 18, "name": "Lance Stroll"}
    ],
    "Haas F1 Team": [
        {"number": 20, "name": "Kevin Magnussen"},
        {"number": 27, "name": "Nico Hülkenberg"}
    ],
    "Williams Racing": [
        {"number": 2, "name": "Logan Sargeant"},
        {"number": 23, "name": "Alexander Albon"}
    ],
    "Alfa Romeo F1 Team": [
        {"number": 24, "name": "Zhou Guanyu"},
        {"number": 77, "name": "Valtteri Bottas"}
    ],
    "Scuderia AlphaTauri": [
        {"number": 21, "name": "Nyck de Vries"},
        {"number": 22, "name": "Yuki Tsunoda"}
    ]
}


def get_driver_info(driver_number):
    for team, drivers in F1_2023_DRIVERS_BY_TEAM.items():
        for driver in drivers:
            if driver["number"] == driver_number:
                return {
                    "number": driver["number"],
                    "name": driver["name"],
                    "team": team
                }
    return {
        "number": driver_number,
        "name": "Unknown",
        "team": "Unknown"
    }


def dataframe_to_text(df, max_rows=20):
    if hasattr(df, "to_markdown"):
        return df.head(max_rows).to_markdown(index=False)
    elif hasattr(df, "to_string"):
        return df.head(max_rows).to_string(index=False)
    else:
        return str(df)


def image_to_base64(fig):
    try:
        import plotly.graph_objects as go
        if isinstance(fig, go.Figure):
            img_bytes = fig.to_image(format="png")
        else:
            import matplotlib.pyplot as plt
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_bytes = buf.read()
            buf.close()
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return None


def generate_llm_narrative(section, data, images=None, model="llama3.2-vision", temperature=0.2):
    """
    Streams the LLM response to capture all fragments.
    1) Serialize data (summary + head for DataFrames, truncated JSON for dicts).
    2) Send system message, one user-text message, then any image messages.
    3) Stream all response chunks via stream_llm_response and concatenate.
    """
    import json
    import pandas as pd
    from components.strategy_chat import stream_llm_response

    # 1) Prepare a compact data_str
    if isinstance(data, pd.DataFrame):
        summary_md = data.describe().head(5).to_markdown()
        head_md = data.head(5).to_markdown(index=False)
        data_block = (
            "### Summary (5 rows):\n" + summary_md + "\n\n"
            "### Sample (5 rows):\n" + head_md
        )
    elif isinstance(data, dict):
        raw = json.dumps(data, indent=2)
        data_block = raw[:2000] + \
            ("\n... (truncated)" if len(raw) > 2000 else "")
    else:
        s = str(data)
        data_block = s[:2000] + ("\n... (truncated)" if len(s) > 2000 else "")

    # 2) Build prompt_text
    prompt_text = (
        f"You are an expert Formula 1 strategy analyst.\n"
        f"Write a detailed, professional and concise narrative for the report section '{section}'.\n\n"
        f"Context data (compact):\n{data_block}\n\n"
        "Focus on insights, explanations and recommendations relevant to this section."
    )

    # 3) Construct messages
    messages = [
        {
            "role": "system",
            "type": "text",
            "content": (
                "You are an advanced Formula 1 strategy assistant. "
                "Answer only with professional, technical and concise explanations for F1 strategy reports."
            )
        },
        {"role": "user", "type": "text", "content": prompt_text}
    ]
    # Attach images separately
    if images:
        for idx, fig in enumerate(images):
            img_b64 = image_to_base64(fig)
            if img_b64:
                messages.append(
                    {"role": "user", "type": "image", "content": img_b64})

    # 4) Stream and concatenate
    assistant_text = ""
    for chunk in stream_llm_response(messages, model, temperature):
        if chunk:
            assistant_text += chunk
    return assistant_text


def export_fig_to_base64(fig):
    img_b64 = image_to_base64(fig)
    return img_b64 if img_b64 else ""


def collect_report_data(
    selected_driver,
    race_data,
    recommendations,
    gap_data,
    predictions,
    radio_data,
    competitive_data,
    gap_charts=None,
    prediction_charts=None,
    stint_charts=None,
    strategy_timeline_chart=None,
    competitive_charts=None,
    radio_charts=None,
    video_thumbnails=None,
    weather_data=None,
    additional_metadata=None,
    telemetry_charts=None,
    video_gap_charts=None,
    lap_time_charts=None,
    degradation_charts=None,
    radio_sentiment_charts=None,
    radio_intent_charts=None,
    nlp_entity_tables=None,
    extra_sections=None
):
    """
    Collects and organizes all relevant data and visualizations for the report of the selected driver.
    Returns a structured dictionary containing all the information needed for the report.
    """
    report_data = {}

    # 1. Metadata (fixed race/circuit)
    driver_info = get_driver_info(selected_driver)
    report_data['metadata'] = {
        'driver_number': driver_info['number'],
        'driver_name': driver_info['name'],
        'team': driver_info['team'],
        'race': "Spanish 2023 GP",
        'date': "2023-06-04",
        'circuit': "Circuit de Barcelona-Catalunya",
        'weather': weather_data if weather_data is not None else (race_data.get('weather', {}) if hasattr(race_data, 'get') else {})
    }
    if additional_metadata:
        report_data['metadata'].update(additional_metadata)

    report_data['recommendations'] = recommendations
    report_data['gap_data'] = gap_data
    report_data['gap_charts'] = gap_charts or []
    report_data['video_gap_charts'] = video_gap_charts or []
    report_data['predictions'] = predictions
    report_data['prediction_charts'] = prediction_charts or []
    report_data['lap_time_charts'] = lap_time_charts or []
    report_data['degradation_charts'] = degradation_charts or []
    report_data['stint_charts'] = stint_charts or []
    report_data['strategy_timeline_chart'] = strategy_timeline_chart
    report_data['telemetry_charts'] = telemetry_charts or []
    report_data['radio_data'] = radio_data
    report_data['radio_charts'] = radio_charts or []
    report_data['radio_sentiment_charts'] = radio_sentiment_charts or []
    report_data['radio_intent_charts'] = radio_intent_charts or []
    report_data['nlp_entity_tables'] = nlp_entity_tables or []
    report_data['competitive_data'] = competitive_data
    report_data['competitive_charts'] = competitive_charts or []
    report_data['video_thumbnails'] = video_thumbnails or []
    if extra_sections:
        report_data.update(extra_sections)
    report_data['raw'] = {
        'race_data': race_data,
        'gap_data': gap_data,
        'predictions': predictions,
        'radio_data': radio_data,
        'competitive_data': competitive_data,
        'recommendations': recommendations
    }
    return report_data


def build_html_report(report_data, selected_sections):
    """
    Builds the final HTML report using the collected data and LLM narratives.
    """
    import markdown
    html = """
    <html>
    <head>
    <meta charset='utf-8'>
    <title>F1 Strategy Report</title>
    <style>
    .markdown-body h1, .markdown-body h2, .markdown-body h3 { font-weight: bold; margin-top: 1.2em; }
    .markdown-body ul, .markdown-body ol { margin-left: 1.5em; }
    .markdown-body li { margin-bottom: 0.3em; }
    .markdown-body strong { font-weight: bold; }
    .markdown-body em { font-style: italic; }
    </style>
    </head>
    <body style='font-family:sans-serif;'>
    """

    # 1. Cover page with metadata
    if selected_sections.get("cover", True):
        meta = report_data.get("metadata", {})
        html += "<h1>F1 Strategy Report</h1>"
        html += f"<p><b>Driver:</b> {meta.get('driver_name','')} (#{meta.get('driver_number','')})</p>"
        html += f"<p><b>Team:</b> {meta.get('team','')}</p>"
        html += f"<p><b>Race:</b> {meta.get('race','')}</p>"
        html += f"<p><b>Date:</b> {meta.get('date','')}</p>"
        html += f"<p><b>Circuit:</b> {meta.get('circuit','')}</p>"
        if meta.get("weather"):
            html += f"<p><b>Weather:</b> {meta['weather']}</p>"
        html += "<hr>"

    # 2. Strategy summary and timeline
    if selected_sections.get("strategy_summary", True):
        html += "<h2>Optimal Strategy Summary</h2>"
        summary_text = report_data.get("strategy_summary_text", "")
        if summary_text:
            summary_html = markdown.markdown(summary_text)
            html += f"<div class='markdown-body'>{summary_html}</div>"
        timeline_fig = report_data.get("strategy_timeline_chart")
        if timeline_fig:
            img = export_fig_to_base64(timeline_fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"

    # 3. Gap analysis
    if selected_sections.get("gap_analysis", True):
        html += "<h2>Gap Analysis</h2>"
        gap_text = report_data.get("gap_analysis_text", "")
        if gap_text:
            html += f"<p>{gap_text}</p>"
        for fig in report_data.get("gap_charts", []):
            img = export_fig_to_base64(fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"
        for fig in report_data.get("video_gap_charts", []):
            img = export_fig_to_base64(fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"

    # 4. Predictions and tyre degradation
    if selected_sections.get("predictions", True):
        html += "<h2>Lap Time & Tyre Degradation Predictions</h2>"
        pred_text = report_data.get("prediction_text", "")
        if pred_text:
            html += f"<p>{pred_text}</p>"
        for fig in report_data.get("prediction_charts", []):
            img = export_fig_to_base64(fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"
        for fig in report_data.get("lap_time_charts", []):
            img = export_fig_to_base64(fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"
        for fig in report_data.get("degradation_charts", []):
            img = export_fig_to_base64(fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"

    # 5. Stint/strategy visualizations
    if selected_sections.get("stint_visualization", True):
        html += "<h2>Stint & Strategy Visualization</h2>"
        for fig in report_data.get("stint_charts", []):
            img = export_fig_to_base64(fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"

    # 6. Conflict analysis
    if selected_sections.get("conflict_analysis", True):
        html += "<h2>Conflict Analysis</h2>"
        conflict_text = report_data.get("conflict_analysis_text", "")
        if conflict_text:
            html += f"<p>{conflict_text}</p>"

    # 7. Telemetry and video analysis
    if selected_sections.get("telemetry", True):
        html += "<h2>Telemetry & Video Highlights</h2>"
        for fig in report_data.get("telemetry_charts", []):
            img = export_fig_to_base64(fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"
        for thumb in report_data.get("video_thumbnails", []):
            html += f"<img src='data:image/png;base64,{thumb}' style='max-width:350px; margin:10px;'>"

    # 8. Radio/NLP analysis
    if selected_sections.get("radio_nlp", True):
        html += "<h2>Radio & NLP Analysis</h2>"
        radio_text = report_data.get("radio_nlp_text", "")
        if radio_text:
            html += f"<p>{radio_text}</p>"
        for fig in report_data.get("radio_charts", []):
            img = export_fig_to_base64(fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"
        for fig in report_data.get("radio_sentiment_charts", []):
            img = export_fig_to_base64(fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"
        for fig in report_data.get("radio_intent_charts", []):
            img = export_fig_to_base64(fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"
        for table in report_data.get("nlp_entity_tables", []):
            if hasattr(table, "to_html"):
                html += table.to_html(index=False)

    # 9. Competitive analysis
    if selected_sections.get("competitive_analysis", True):
        html += "<h2>Competitive Analysis</h2>"
        comp_text = report_data.get("competitive_analysis_text", "")
        if comp_text:
            html += f"<p>{comp_text}</p>"
        for fig in report_data.get("competitive_charts", []):
            img = export_fig_to_base64(fig)
            html += f"<img src='data:image/png;base64,{img}' style='max-width:700px;'><br>"

    # 10. Any extra/custom sections
    if "extra_sections" in report_data:
        for section, content in report_data["extra_sections"].items():
            if selected_sections.get(section, True):
                html += f"<h2>{section.replace('_',' ').title()}</h2>"
                if isinstance(content, str):
                    html += f"<p>{content}</p>"
                elif hasattr(content, "to_html"):
                    html += content.to_html(index=False)

    # 11. Appendix with raw data
    if selected_sections.get("appendix", False):
        html += "<h2>Appendix: Raw Data</h2>"
        raw = report_data.get("raw", {})
        for key, df in raw.items():
            if hasattr(df, "to_html"):
                html += f"<h4>{key}</h4>"
                html += df.to_html(index=False)

    html += "</body></html>"
    return html


def render_report_export_ui(
    selected_driver,
    race_data,
    recommendations,
    gap_data,
    predictions,
    radio_data,
    competitive_data,
    gap_charts=None,
    prediction_charts=None,
    degradation_charts=None,
    competitive_charts=None
):
    """
    Streamlit interface to select sections and export the report.
    Gathers all relevant data and figures, then allows the user to generate and download the report.
    """

    st.header("Export Strategy Report")
    st.write(
        "Customize and export a professional strategy report for the selected driver.")

    # Section selection
    st.subheader("Select sections to include:")
    section_options = {
        "cover": "Cover page with metadata",
        "strategy_summary": "Optimal strategy summary and timeline",
        "gap_analysis": "Gap analysis",
        "predictions": "Lap time & tyre degradation predictions",
        "stint_visualization": "Stint & strategy visualization",
        "conflict_analysis": "Conflict analysis and resolutions",
        "telemetry": "Telemetry & video highlights",
        "radio_nlp": "Radio & NLP analysis",
        "competitive_analysis": "Competitive analysis",
        "appendix": "Appendix with raw data"
    }
    default_sections = {
        "cover": True,
        "strategy_summary": True,
        "gap_analysis": True,
        "predictions": True,
        "stint_visualization": True,
        "conflict_analysis": True,
        "telemetry": True,
        "radio_nlp": True,
        "competitive_analysis": True,
        "appendix": False
    }
    selected_sections = {}
    cols = st.columns(2)
    for i, (key, label) in enumerate(section_options.items()):
        with cols[i % 2]:
            selected_sections[key] = st.checkbox(
                label, value=default_sections[key], key=f"section_{key}")

    st.markdown("---")
    st.write("Click the button below to generate and download the report.")

    if st.button("Generate HTML Report"):
        with st.spinner("Generating report..."):
            # If charts are not provided, generate them here
            if gap_charts is None or degradation_charts is None or prediction_charts is None or competitive_charts is None:
                from utils.visualization import (
                    st_plot_gap_evolution,
                    st_plot_undercut_opportunities,
                    st_plot_gap_consistency,
                    st_plot_degradation_rate,
                    st_plot_regular_vs_adjusted_degradation,
                    st_plot_speed_vs_tire_age,
                    st_plot_fuel_adjusted_degradation
                )
                from components.competitive_analysis_view import get_competitive_analysis_figures

                # Gap analysis figures
                gap_figs = []
                try:
                    fig1 = st_plot_gap_evolution(gap_data, selected_driver)
                    if fig1:
                        gap_figs.append(fig1)
                    fig2 = st_plot_undercut_opportunities(
                        gap_data, selected_driver)
                    if fig2:
                        gap_figs.append(fig2)
                    fig3 = st_plot_gap_consistency(gap_data, selected_driver)
                    if fig3:
                        gap_figs.append(fig3)
                except Exception:
                    pass

                # Degradation analysis figures
                degradation_figs = []
                try:
                    fig1 = st_plot_degradation_rate(race_data, selected_driver)
                    if fig1:
                        degradation_figs.append(fig1)
                    fig2 = st_plot_regular_vs_adjusted_degradation(
                        race_data, selected_driver)
                    if fig2:
                        degradation_figs.append(fig2)
                    if 'CompoundID' in race_data.columns:
                        for compound_id in race_data['CompoundID'].unique():
                            fig3 = st_plot_speed_vs_tire_age(
                                race_data, selected_driver, compound_id)
                            if fig3:
                                degradation_figs.append(fig3)
                except Exception:
                    pass

                # Prediction figures
                prediction_figs = []
                try:
                    fig1 = st_plot_fuel_adjusted_degradation(
                        race_data, selected_driver)
                    if fig1:
                        prediction_figs.append(fig1)
                except Exception:
                    pass

                # Competitive analysis figures
                competitive_charts = get_competitive_analysis_figures(
                    race_data, selected_driver)

                gap_charts = gap_figs
                degradation_charts = degradation_figs
                prediction_charts = prediction_figs

            # Collect all data and charts
            report_data = collect_report_data(
                selected_driver,
                race_data,
                recommendations,
                gap_data,
                predictions,
                radio_data,
                competitive_data,
                gap_charts=gap_charts,
                prediction_charts=prediction_charts,
                degradation_charts=degradation_charts,
                competitive_charts=competitive_charts
            )

            # Progress bar for LLM narrative generation
            sections_llm = [
                "strategy_summary",
                "gap_analysis",
                "predictions",
                "conflict_analysis",
                "radio_nlp",
                "competitive_analysis"
            ]
            total_steps = sum([selected_sections.get(sec, False)
                              for sec in sections_llm])
            current_step = 0
            progress_bar = st.progress(0)

            # LLM narrative generation with multimodal support
            strategy_timeline_chart = report_data.get(
                "strategy_timeline_chart", None)
            radio_charts = report_data.get("radio_charts", None)

            if selected_sections.get("strategy_summary", True):
                st.info("Generating: Strategy Summary")
                report_data["strategy_summary_text"] = generate_llm_narrative(
                    "Strategy Summary", recommendations, images=[strategy_timeline_chart] if strategy_timeline_chart else None)
                current_step += 1
                progress_bar.progress(
                    current_step / total_steps if total_steps else 1)

            if selected_sections.get("gap_analysis", True):
                st.info("Generating: Gap Analysis")
                report_data["gap_analysis_text"] = generate_llm_narrative(
                    "Gap Analysis", gap_data, images=gap_charts)
                current_step += 1
                progress_bar.progress(
                    current_step / total_steps if total_steps else 1)

            if selected_sections.get("predictions", True):
                st.info("Generating: Lap Time & Tyre Degradation Predictions")
                report_data["prediction_text"] = generate_llm_narrative(
                    "Lap Time & Tyre Degradation Predictions", predictions, images=prediction_charts)
                current_step += 1
                progress_bar.progress(
                    current_step / total_steps if total_steps else 1)

            if selected_sections.get("conflict_analysis", True):
                st.info("Generating: Conflict Analysis")
                report_data["conflict_analysis_text"] = generate_llm_narrative(
                    "Conflict Analysis", recommendations)
                current_step += 1
                progress_bar.progress(
                    current_step / total_steps if total_steps else 1)

            if selected_sections.get("radio_nlp", True):
                st.info("Generating: Radio & NLP Analysis")
                report_data["radio_nlp_text"] = generate_llm_narrative(
                    "Radio & NLP Analysis", radio_data, images=radio_charts)
                current_step += 1
                progress_bar.progress(
                    current_step / total_steps if total_steps else 1)

            if selected_sections.get("competitive_analysis", True):
                st.info("Generating: Competitive Analysis")
                report_data["competitive_analysis_text"] = generate_llm_narrative(
                    "Competitive Analysis", competitive_data, images=competitive_charts)
                current_step += 1
                progress_bar.progress(
                    current_step / total_steps if total_steps else 1)

            # Build HTML report
            html = build_html_report(report_data, selected_sections)

            # Download button
            st.download_button(
                label="Download HTML Report",
                data=html,
                file_name="f1_strategy_report.html",
                mime="text/html"
            )
