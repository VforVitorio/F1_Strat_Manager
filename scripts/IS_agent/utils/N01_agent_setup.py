# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: f1_strat_manager
#     language: python
#     name: python3
# ---

# # Experta: Theory Fundamentals of Production Systems and RETE Algorithm

# ---

# ## Introduction
#
# Experta is a Python library based on Production Systems, also known as Expert Systems or Rule-Based Systems, wich is a fundamental paradigm in simbolic AI.

# ---

# ## What are Expert Systems?
#
# An expert system is formed by 3 key components:
#
# 1. **Facts base**: it stores the factual knowledge of the system. That is, what the system knows in a specific moment.
#
# 2. **Rule Base**: contains the procedure knowledge as "if-then" rules.
#
# 3. **Inference motor**: motor that determines whether apply one rule or another and when to apply it.
#
# The central idea is no model the thinking as a rule chaining process, simillar at how human experts would take a decission applying their knowledge to a specific subject.
#

# ---

# ## RETE Algorithm
#
# Experta´s core is RETE algorithm, designed in 1982 by Charles L.Forgy. This algorithm is crucal for knowing what Experta is doing and also knowing why it is efficient:
#
# - Its **function** is to optimize the coincidence of patterns between facts and rules.
#
# - It builds a node network that represent patterns. Then, it avoids reevaluating all the rules when the facts change.
#
# Therefore, RETE builds a "discrimination net" that acts as an efficient filter for determining which rules should be activated in response to changes made on the facts base.

# ---

# ## Experta´s execution cycle
#
# 1. **Match**: the motor identifies all the rules that can be activaded with the actual facts base.
# 2. **Conflict Resolution**: if multiple rules coincide, the motor decides which one is executed first (using conflic resolution strategies).
# 3. **Act**: it executes the action associated with the selected rule, that tipically modifies the facts base.
# 4. **Cycle**: the process is repeated until there are no more rules to be activated.
#
# This cycle is knwon as "cycle recognize-act" or "production cycle".

# ---

# ## Declarative vs Imperative Programming
#
# Experta is a **Declarative Programming paradigm**, in contrast with traditional programming. Instead of defining HOW to make something step by step, Declarative Programming specifies WHICH conditions need to be acomplished. In Experta, the developer defines rules declaratively and the motor is the one that says when and how are they going to be applied.

# ---

# ## Relevance for F1 Strategy
#
# Experta´s selection for the F1 strategy problem is theoretically justified due to this 5 points:
#
# 1. **Codificable expert knowledge**: F1 strategies can be expressed naturally as conditional rules based on expert knowledge.
# 2. **Incremental Reasoning**: during the race, information comes continously, such as radios, telemetry, track data or weather. RETE is efficient for updating conclussions based on new information.
# 3. **Knowledge Explanation**: rules can be read and modified by humans, allowing adjusting strategies based on feedback.
# 4. **Explainable Capacity**: unlike black box models like Neural Networks, a system based on rules can explain exactly which conditions made them make a decission.
# 5. **Multiple Information Integration**: key for my project, as it brings me the capacity to merge structured information as data with semi-structured information like NLP radio analysis or my prediction models in the same logical framework.
#
# The implementation through Experta allows to capture strategic reasoning of F1 Teams, creating a system that emulates how they would take real-time decissions based on the actual avaliable information.

# ---

# ## Basic Example of Decision Making with an Expert System
#
# I will illustrate a simple case: deciding whether an F1 car should pit based on tire degradation and weather conditions.
#
# ### 1. Problem Definition
# We have basic rules to decide on a pit stop:
#
# - If tire degradation is greater than 60%, recommend a pit stop.
# - If it is raining and the car has dry tires, recommend a pit stop.
# - If the degradation is moderate (30-60%) and the driver reports grip issues, recommend a pit stop.
#
# ### 2. Step-by-Step Process
# Here is what happens when we run Experta:
#
# #### 2.1 Initialization:
# - The rule engine `EstrategiaF1` is created.
# - The method `reset()` is called to prepare the engine.
#
# #### 2.2 Declaration of Facts:
# - We add that the tires have a degradation of 45% and are of the dry type.
# - We add that it is not raining.
# - We add that the driver reports grip issues.
#
# #### 2.3 Execution of the RETE Cycle:
# - The engine calls `run()`, starting the inference cycle.
# - RETE builds an activation network with the three defined rules.
#
# #### 2.4 Rule Evaluation:
# - **First rule (very_degraded_tires):** DOES NOT MATCH because degradation = 45% (less than 60%).
# - **Second rule (change_to_rain):** DOES NOT MATCH because raining = False.
# - **Third rule (moderate_grip_problems):** MATCHES because:
#   - degradation = 45% (is between 30% and 60%)
#   - the message contains "problems" and "grip"
#
# #### Rule Activation:
# - The rule `moderate_grip_problems` is activated.
# - Its action is executed, declaring a new fact: **Recommendation**.
#
# #### New Evaluation Cycle:
# - The engine evaluates whether there are new rules that match the newly added fact.
# - In this case, no additional rules are activated.
# - The engine terminates the execution.
#
# ### Final Result:
# We obtain a recommendation:
# - **Action recommended:** pit
# - **Reason:** Grip issues reported with moderate degradation
# - **Urgency:** medium
#
# This is the essence of how the expert system processes the rules: it continuously evaluates the available facts against the rule conditions and executes the corresponding actions when matches occur. The RETE algorithm makes this process efficient by avoiding the need to re-evaluate all rules for every fact.
#

# ---

# ## 1. Import the Libraries

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---

# ## 2. Definition of Fact Classes

# ---

from experta import Fact, Field, KnowledgeEngine


# Field object takes 5 possible arguments:
#
# 1. Datatype(mandatory) specifies the expected data type.
# 2. Default(optional) specifies a default value if none is given.
# 3. Mandatory(optional) is a boolean to put if the Field is mandatory.
# 4. Optional, contrary to Mandaroty.
# 5. Test (function) allows defininf a function to validate the value.

# ### 2.1 Telemetry Facts

# Telemetry Facts
class TelemetryFact(Fact):
    """
    Facts about car telemetry and performance
    """
    lap_time = Field(float, mandatory=False)           # Curremt lap time
    # Predicted lap time by the model
    predicted_lap_time = Field(float, mandatory=False)
    # Age of the current tire set in laps
    tire_age = Field(int, mandatory=False)
    compound_id = Field(int, mandatory=False)           # Tire type with ID
    position = Field(int, mandatory=False)             # Current race position
    driver_number = Field(int, mandatory=False)         # Driver number


# ### 2.2 Degradation Facts

class DegradationFact(Fact):
    """
    Facts about tire degradation
    """
    degradation_rate = Field(
        float, mandatory=False)           # Seconds lost per lap due to degradation
    # Last N degradation rates for trend analysis
    previous_rates = Field(list, mandatory=False)
    # Percentage degradation adjusted for fuel
    fuel_adjusted_deg_percent = Field(float, mandatory=False)


# ### 2.3 Gap Facts

class GapFact(Fact):
    """
    Facts about gaps to other cars
    """
    gap_ahead = Field(
        float, mandatory=False)          # Time to car ahead (seconds)
    # Time to car behind (seconds)
    gap_behind = Field(float, mandatory=False)
    # Change in gap ahead over last laps
    gap_ahead_trend = Field(float, mandatory=False)
    # Change in gap behind over last laps
    gap_behind_trend = Field(float, mandatory=False)


# ### 2.4 Radio Facts

class RadioFact(Fact):
    """
    Facts from radio communications analysis
    """
    sentiment = Field(str, mandatory=False)  # positive, negative, neutral
    intent = Field(str, mandatory=False)     # WARNING, QUESTION, etc.
    # Detected entities categorized (SITUATION, INCIDENT, PIT_CALL, etc)
    entities = Field(dict, mandatory=False)
    timestamp = Field(float, mandatory=False)  # When the message was received


# ### 2.5 Race Status

class RaceStatusFact(Fact):
    """
    Facts about current race status
    """
    lap = Field(int, mandatory=True)               # Current lap
    total_laps = Field(int, mandatory=True)        # Total race laps
    race_phase = Field(str, mandatory=False)       # start, mid, end
    # clear, yellow, safety car, red flag
    track_status = Field(str, mandatory=False)


# ### 2.6 Strategy Recomendation

class StrategyRecommendation(Fact):
    """
    Reccommendation produced by the Expert System
    """
    action = Field(
        str, mandatory=True)                        # Specific action to take
    # Confidende level (0-1)
    confidence = Field(float, mandatory=True)
    # Natural Language Explanation
    explanation = Field(str, mandatory=True)
    # Priority level (higher = more urgent)
    priority = Field(int, mandatory=False, default=0)
    # Lap when reccomendation was made
    lap_issued = Field(int, mandatory=True)


# ---

# ## 3. Engine Definition with Rule Documentation

class F1StrategyEngine(KnowledgeEngine):
    """
    Formula 1 strategy expert system engine
    """

    def __init__(self):
        super().__init__()
        self.rules_fired = []  # Tracking the rules that have been activated

    def get_recommendations(self):
        """
        Retrieve all current recommendations, sorted by priority and confidence
        """

        recommendations = []
        for fact in self.facts.values():
            if isinstance(fact, StrategyRecommendation):
                recommendations.append(
                    {
                        "action": fact["action"],
                        "confidence": fact["confidence"],
                        "explanation": fact["explanation"],
                        "priority": fact.get("priority", 0),
                        "lap_issued": fact["lap_issued"]
                    }
                )
        return sorted(
            recommendations,
            key=lambda x: (x["priority"], x["confidence"]),
            reverse=True
        )

    def record_rule_fired(self, rule_name):
        """
        Record when a rule is fired for explanation and debugging
        """

        current_lap = None
        for fact in self.facts.values():
            if isinstance(fact, RaceStatusFact):
                current_lap = fact.get("lap")
                break

        self.rules_fired.append(
            {
                "rule": rule_name,
                "lap": current_lap,
                "timestamp": pd.Timestamp.now()
            }
        )


# ---

# ## 4. Data Transformation Functions

def transform_degradation_prediction(prediction_data, driver_number):
    """
    Transform degradation prediction model output into facts.

    Args:
        prediction_data (dict): Output from the degradation prediction model
        driver_number (int): The driver number to extract data for

    Returns:
        dict: Dictionary with facts to declare
    """
    # Filter data for the specific driver
    driver_data = prediction_data[prediction_data['DriverNumber']
                                  == driver_number].iloc[-1]

    # Extract relevant fields
    degradation_fact = DegradationFact(
        degradation_rate=float(driver_data['DegradationRate']),
        fuel_adjusted_deg_percent=float(
            driver_data.get('FuelAdjustedDegPercent', 0))
    )

    # Get historical rates if available (last 3 laps)
    if len(prediction_data[prediction_data['DriverNumber'] == driver_number]) >= 3:
        historical = prediction_data[prediction_data['DriverNumber'] == driver_number].tail(
            3)
        degradation_fact['previous_rates'] = historical['DegradationRate'].tolist()

    telemetry_fact = TelemetryFact(
        tire_age=int(driver_data['TyreAge']),           # Convert to int
        compound_id=int(driver_data['CompoundID']),     # Convert to int
        position=int(driver_data['Position']),          # Convert to int
        driver_number=int(driver_number)                # Convert to int
    )

    return {
        'degradation': degradation_fact,
        'telemetry': telemetry_fact
    }


def transform_laptime_prediction(prediction_data, driver_number):
    """
    Transform lap time prediction model output into facts.

    Args:
        prediction_data (dict): Output from the lap time prediction model
        driver_number (int): The driver number to extract data for

    Returns:
        dict: Dictionary with facts to declare
    """
    # Filter data for the specific driver
    driver_data = prediction_data[prediction_data['DriverNumber']
                                  == driver_number].iloc[-1]

    # Extract current and predicted lap times
    current_lap_time = driver_data['LapTime']
    # Assuming this field exists
    predicted_lap_time = driver_data['PredictedLapTime']

    telemetry_fact = TelemetryFact(
        lap_time=current_lap_time,
        predicted_lap_time=predicted_lap_time,
        driver_number=driver_number
    )

    return {
        'telemetry': telemetry_fact
    }


def transform_radio_analysis(radio_json):
    """
    Transform NLP radio analysis into facts.

    Args:
        radio_json (dict): The output from radio NLP analysis

    Returns:
        RadioFact: Fact with radio analysis information
    """
    analysis = radio_json['analysis']

    return RadioFact(
        sentiment=analysis['sentiment'],
        intent=analysis['intent'],
        entities=analysis['entities'],
        timestamp=pd.Timestamp.now().timestamp()
    )


def calculate_race_phase(current_lap, total_laps):
    """Calculate the current phase of the race."""
    percentage = (current_lap / total_laps) * 100
    if percentage < 25:
        return "start"
    elif percentage > 75:
        return "end"
    else:
        return "mid"


# ---

# ## 6. Basic Engine Initialization Example

if __name__ == "main":
    # Create an engine instance

    engine = F1StrategyEngine()
    engine.reset()

    # Example declaring some initial facts

    engine.declare(RaceStatusFact(lap=1, total_laps=60,
                   race_phase="start", track_status="clear"))

    # Example of transforming model predictions into facts
    # (These would come from our actual models in practice)
    mock_degradation_data = pd.DataFrame({
        'DriverNumber': [44, 44],
        'DegradationRate': [0.05, 0.07],
        'FuelAdjustedDegPercent': [5.0, 7.0],
        'TyreAge': [3, 4],
        'CompoundID': [2, 2],  # Medium tire
        'Position': [1, 1]
    })

    facts = transform_degradation_prediction(mock_degradation_data, 44)
    engine.declare(facts['degradation'])
    engine.declare(facts['telemetry'])

    print("Facts initialized successfully. Rules will be implemented in subsequent notebooks.")
