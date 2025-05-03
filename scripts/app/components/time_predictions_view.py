import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_time_predictions_view(predictions_df, selected_driver):
    """
    Visualizes the real and predicted lap times for the selected driver.
    predictions_df must have: DriverNumber, LapNumber, LapTime, PredictedLapTime
    """
    st.markdown("---")

    st.header(
        f"Lap Time Prediction vs Real for Driver #{selected_driver}")

    # Filter data for the selected driver
    df = predictions_df[predictions_df['DriverNumber'].astype(
        int) == int(selected_driver)].copy()
    df = df.sort_values('LapNumber')
    if df.empty:
        st.info("No prediction data available for this driver.")
        return

    # Calculate mean absolute error (MAE)
    if 'LapTime' in df.columns and 'PredictedLapTime' in df.columns:
        valid = df[['LapNumber', 'LapTime', 'PredictedLapTime']].dropna()
        valid['AbsError'] = (valid['LapTime'] -
                             valid['PredictedLapTime']).abs()
        mae = valid['AbsError'].mean()

        st.metric("Mean Absolute Error (MAE)", f"{mae:.3f} s")

        # Show top 5 laps with the highest error
        st.markdown("---")
        st.markdown(
            "<h3 style='text-align: center;'>Top 5 Laps with Highest Prediction Error</h3>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 12px;'></div>",
                    unsafe_allow_html=True)
        top_errors = valid.sort_values('AbsError', ascending=False).head(5)
        st.table(top_errors[['LapNumber', 'LapTime', 'PredictedLapTime', 'AbsError']].rename(
            columns={'LapNumber': 'Lap', 'LapTime': 'Real Lap Time', 'PredictedLapTime': 'Predicted Lap Time', 'AbsError': 'Absolute Error'}))

    st.markdown("---")
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['LapNumber'],
        y=df['LapTime'],
        mode='lines+markers',
        name='Real Lap Time',
        line=dict(color='deepskyblue', width=2),
        marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=df['LapNumber'],
        y=df['PredictedLapTime'],
        mode='lines+markers',
        name='Predicted Lap Time',
        line=dict(color='orange', width=2, dash='dash'),
        marker=dict(size=4)
    ))

    fig.update_layout(
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (s)",
        title="Lap Time: Real vs Predicted",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    return fig
