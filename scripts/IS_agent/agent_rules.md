# F1 Strategy Engine: Strategic Rules

This document defines the rules that our expert system will implement to make strategic decisions in F1 races.

## A. Rules Based on Predicted Degradation

1. **High Degradation Rate Pit Stop**

   - IF (DegradationRate > 0.15 AND TyreAge > 10)
   - THEN recommend priority pit stop
   - CONFIDENCE: 0.85

2. **Stint Extension for Low Degradation**

   - IF (DegradationRate < 0.08 AND TyreAge > 12 AND Position < 5)
   - THEN recommend extending current stint
   - CONFIDENCE: 0.75

3. **Early Degradation Warning**

   - IF (DegradationRate increases by more than 0.03 in 3 consecutive laps)
   - THEN recommend pit stop preparation
   - CONFIDENCE: 0.7

## B. Rules Based on Lap Time Predictions

1. **Optimal Performance Window**

   - IF (predicted LapTime < current LapTime AND Position > 3 AND TyreAge < 8)
   - THEN recommend strategic push
   - CONFIDENCE: 0.75

2. **Performance Cliff Detection**

   - IF (predicted LapTime > current LapTime + 0.7 AND TyreAge > 15)
   - THEN recommend priority pit stop
   - CONFIDENCE: 0.85

3. **Post-Traffic Recovery**

   - IF (predicted LapTime < current LapTime - 0.5 AND Position changed negatively in last lap)
   - THEN recommend recovery stint
   - CONFIDENCE: 0.7

## C. Undercut/Overcut Rules (With Gaps)

1. **Undercut Opportunity**

   - IF (gap_ahead < 2.0s AND DegradationRate > 0.12 AND TyreAge > 8)
   - THEN recommend undercut
   - CONFIDENCE: 0.8

2. **Undercut Defense**

   - IF (gap_behind < 2.5s AND gap_behind decreasing AND DegradationRate > 0.1)
   - THEN recommend defensive pit stop
   - CONFIDENCE: 0.75

3. **Strategic Overcut**

   - IF (gap_ahead < 3.5s AND predicted LapTime < front_car_lap_time AND DegradationRate < 0.1)
   - THEN recommend overcut
   - CONFIDENCE: 0.75

## D. Rules Based on Communications (NLP)

1. **Response to Grip Issues**

   - IF (sentiment == "negative" AND "grip" in entities["SITUATION"] AND DegradationRate > 0.09)
   - THEN increase pit stop priority
   - CONFIDENCE: 0.85

2. **Weather Information Adjustment**

   - IF (intent == "WARNING" AND ("rain" in entities["SITUATION"] OR "wet" in entities["SITUATION"]))
   - THEN prepare for switch to rain tires
   - CONFIDENCE: 0.9

3. **Incident Reaction**

   - IF ("safety" in entities["INCIDENT"] OR "yellow" in entities["SITUATION"])
   - THEN reevaluate pit window taking advantage of neutralization
   - CONFIDENCE: 0.85

## E. High Priority Combined Rules

1. **Confirmed Critical Deterioration**

   - IF (DegradationRate > 0.18 AND predicted LapTime > previous LapTime + 0.5 AND sentiment == "negative")
   - THEN recommend urgent pit stop
   - CONFIDENCE: 0.95
   - PRIORITY: High

2. **Tactical Opportunity in Incident**

   - IF ("yellow" in entities["SITUATION"] AND Position > 10 AND gap_ahead > 5.0s)
   - THEN recommend pit stop taking advantage of yellow flag
   - CONFIDENCE: 0.85
   - PRIORITY: High
