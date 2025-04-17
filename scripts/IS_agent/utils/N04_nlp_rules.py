from utils.N01_agent_setup import (
    TelemetryFact,
    DegradationFact,
    RaceStatusFact,
    RadioFact,
    StrategyRecommendation,
    F1StrategyEngine,
    transform_radio_analysis,
    process_radio_message
)
import utils.N01_agent_setup as agent_setup
from experta import DefFacts, Fact, Field, KnowledgeEngine
from experta import Rule, TEST, MATCH
import pandas as pd              # For data manipulation and analysis
import numpy as np               # For numerical operations
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns            # For enhanced visualizations
import json                      # For handling JSON data from NLP analysis
from datetime import datetime    # For timestamp handling
import os                        # For operating system interactions
import sys                       # For system-specific parameters and functions

# Add parent directory to system path to make custom modules accessible
sys.path.append(os.path.abspath('../'))

# Import Experta components for building the rule engine

# Import custom fact classes and functions from previous modules

# Try importing the NLP radio processing module if available
try:
    from NLP_radio_processing.NLP_utils import N06_model_merging as radio_nlp
except ImportError:
    radio_nlp = None

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)


class F1RadioRules(F1StrategyEngine):
    """
    Rules for F1 strategy based on NLP analysis of radio communications.
    """
    @Rule(
        RadioFact(sentiment="negative"),
        RadioFact(entities=MATCH.entities),
        TEST(lambda entities:
             any('grip' in str(v).lower() or 'struggle' in str(v).lower() for lst in entities.values() for v in lst) or
             len(entities.get('TECHNICAL_ISSUE', [])) > 0),
        RaceStatusFact(lap=MATCH.lap)
    )
    def grip_issue_response(self, entities, lap):
        grip_related = []
        for cat, vals in entities.items():
            for val in vals:
                if 'grip' in str(val).lower() or 'struggle' in str(val).lower():
                    grip_related.append(f"{val} [{cat}]")
        tech_issues = entities.get('TECHNICAL_ISSUE', [])
        for issue in tech_issues:
            if issue not in grip_related:
                grip_related.append(f"{issue} [TECHNICAL_ISSUE]")
        issue_text = grip_related[0].split(
            ' [')[0] if grip_related else 'degradation issues'
        self.declare(
            StrategyRecommendation(
                action="prioritize_pit",
                confidence=0.85,
                explanation=(
                    f"Driver reports {issue_text}. Confirm with telemetry and prioritize pit stop if needed."),
                priority=2,
                lap_issued=lap
            )
        )
        self.record_rule_fired("grip_issue_response")

    @Rule(
        RadioFact(entities=MATCH.entities),
        TEST(lambda entities:
             len(entities.get('WEATHER', [])) > 0 or
             any('wet' in c.lower() for c in entities.get('TRACK_CONDITION', []))),
        RaceStatusFact(lap=MATCH.lap)
    )
    def weather_information_adjustment(self, entities, lap):
        weather = entities.get('WEATHER', [])
        wet = [c for c in entities.get(
            'TRACK_CONDITION', []) if 'wet' in c.lower()]
        weather_text = weather + wet or ['changing weather conditions']
        self.declare(
            StrategyRecommendation(
                action="prepare_rain_tires",
                confidence=0.9,
                explanation=(
                    f"Weather change detected: {', '.join(weather_text)}. Prepare for rain tires."),
                priority=3,
                lap_issued=lap
            )
        )
        self.record_rule_fired("weather_information_adjustment")

    @Rule(
        RadioFact(entities=MATCH.entities),
        TEST(lambda entities:
             any('safety' in i.lower() for i in entities.get('INCIDENT', [])) or
             any('yellow' in s.lower() for s in entities.get('SITUATION', []))),
        RaceStatusFact(lap=MATCH.lap)
    )
    def incident_reaction(self, entities, lap):
        safety = [i for i in entities.get(
            'INCIDENT', []) if 'safety' in i.lower()]
        yellow = [s for s in entities.get(
            'SITUATION', []) if 'yellow' in s.lower()]
        incident_type = safety[0] if safety else (
            yellow[0] if yellow else 'race neutralization')
        self.declare(
            StrategyRecommendation(
                action="reevaluate_pit_window",
                confidence=0.85,
                explanation=(
                    f"Incident detected: {incident_type}. Consider pitting under neutralization."),
                priority=3,
                lap_issued=lap
            )
        )
        self.record_rule_fired("incident_reaction")


def test_radio_rules(scenario, message, additional_facts=None):
    """
    Test radio rules with a message and optional extra facts.
    """
    if radio_nlp is None:
        raise ImportError("NLP module not available")
    json_path = radio_nlp.analyze_radio_message(message)
    with open(json_path, 'r') as f:
        analysis = json.load(f)
    radio_fact = transform_radio_analysis(json_path)
    engine = F1RadioRules()
    engine.reset()
    engine.declare(radio_fact)
    if additional_facts:
        for fact in additional_facts.values():
            engine.declare(fact)
    engine.run()
    return engine.get_recommendations()


if __name__ == "__main__":
    print("Libraries and fact classes loaded successfully.")

    # Sample pipeline demonstration
    sample = "Box this lap for softs, there's rain expected in 10 minutes"
    print(f"Processing sample message: '{sample}'")
    path = process_radio_message(sample)
    if path:
        print(f"Loaded analysis JSON at {path}")

    # Test scenarios
    from utils.N01_agent_setup import RaceStatusFact
    grip_recs = test_radio_rules(
        "Grip Issues",
        "I'm really struggling with grip, the rear feels terrible",
        {'status': RaceStatusFact(lap=20, total_laps=60)}
    )
    print("Grip Rules Recs:", grip_recs)

    weather_recs = test_radio_rules(
        "Weather Warning",
        "Warning: rain starting at turn 4, track is getting wet",
        {'status': RaceStatusFact(lap=25, total_laps=60)}
    )
    print("Weather Rules Recs:", weather_recs)

    incident_recs = test_radio_rules(
        "Safety Car",
        "Safety car has been deployed, box box box",
        {'status': RaceStatusFact(lap=30, total_laps=60)}
    )
    print("Incident Rules Recs:", incident_recs)
