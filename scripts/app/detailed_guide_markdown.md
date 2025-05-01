# Detailed Guide for Recommendations View and Strategic Assistant

> **Note:** All sections and features described here will later be enhanced and adapted with the integration of the S12_lab LLM, which will provide advanced language generation and contextual analysis capabilities.

---

## 1. Optimal Strategy Construction and Visualization

### 1.1. Automatic Selection of Compatible Recommendations

- Implement an algorithm to select a compatible subset of recommendations, resolving conflicts automatically (e.g., avoid 'extend_stint' and 'pit_stop' on close laps).
- Ensure FIA rules are respected (at least one pit stop and two different tyre compounds if applicable).
- Allow for future override by user selection.

### 1.2. Timeline and Step Chart Visualization

- Visualize the optimal strategy as a step chart or sequential timeline.
- Clearly show transitions between stints and pit stops.
- Use color and labels to distinguish action types and priorities.
- Display projected metrics (lap times, expected position, degradation) in tooltips or directly on the chart.

### 1.3. Narrative Summary

- Generate a natural language summary of the optimal strategy, including stint breakdown, key decisions, and projected outcomes.
- Highlight any FIA rule warnings (e.g., only one compound detected).

---

## 2. Strategic Conflict Detection and Resolution

### 2.1. Conflict Identification

- Automatically detect incompatible or overlapping recommendations.
- Highlight conflicts visually in the UI (e.g., warning icons, color overlays).

### 2.2. Conflict Explanation and User Resolution

- Provide clear explanations for why recommendations are considered conflicting.
- Suggest resolutions based on confidence, priority, or projected impact.
- Allow the user to manually resolve conflicts and re-calculate the optimal strategy.

---

---

## 4. Critical Decision Point Mapping

### 4.1. Identification of Key Moments

- Analyze the race timeline to identify laps where strategic decisions have the highest impact.
- Use heuristics or ML models to detect windows of opportunity or risk.

### 4.2. Heatmap or Timeline Visualization

- Display a heatmap or similar visualization showing the density and importance of decision points across the race.
- Allow users to click on a point to see detailed context and recommendations.

---

## 5. Competitive Analysis

### 5.1. Relative Position and Gap Mapping

- Visualize the car's position relative to competitors on track.
- Show gap evolution with key rivals and potential undercut/overcut windows.

### 5.2. Opponent Strategy Estimation

- Estimate likely strategies of other teams based on historical data and current race context.
- Alert the user to strategic threats or opportunities (e.g., "Rival X likely to pit soon").

### 5.3. Defensive and Offensive Recommendations

- Provide context-aware suggestions for defending or attacking based on real-time gaps and projected stints.

---

## 6. Video Analysis and gap_calculation.ipynb Integration

### 6.1. Video Upload and Processing Interface

- Add a section in the application where the user can upload a race video.
- Integrate the code logic from `gap_calculation.ipynb` to process the video directly from the app.

### 6.2. Interactive Controls for Video Analysis

- Provide UI controls to configure detection parameters (e.g., detection thresholds, frame range) from the app interface.
- Allow the user to start, pause, and step through the video analysis.

### 6.3. Real-Time Visualization and LLM Integration

- Display detection results (gaps, objects, events) overlaid on the video in real time or per frame.
- Add a window where a vision-capable LLM can analyze selected video segments and generate insights, explanations, or answer user queries about the video.

---

## 7. Strategy Report Export

### 7.1. Report Generation

- Allow the user to export the current strategy (including selected recommendations, visualizations, and narrative) as a professional HTML or PDF report.
- Include options to customize which sections and visualizations are included.

### 7.2. Report Structure

- Cover page with metadata (race, driver, date, etc.)
- Optimal strategy summary and timeline
- Conflict analysis and resolutions
- Telemetry and video analysis highlights
- Competitive analysis section
- Appendix with raw data if desired

---

## 8. (Future) S12_lab LLM Integration

- All sections above will be further enhanced with S12_lab LLM capabilities:
  - Natural language explanations, summaries, and justifications
  - Contextual Q&A and strategy suggestions
  - Automated video and image analysis
  - Dynamic report generation and customization
