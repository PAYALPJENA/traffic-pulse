"""
Module 5: Natural Language AI Layer
====================================
Translates structured data from the Decision Engine and Intelligence Engine
into clear, human-readable English narratives.

STRICT GUARDRAIL: The LLM (or template engine) must ONLY translate data
already computed by upstream modules. It must NEVER invent numbers,
make predictions, or generate recommendations independently.

Pipeline: Decision Engine dict → NL AI → Human-readable narrative

Supports:
  - Template-based generation (always available, no API key needed)
  - OpenAI-compatible API (optional, for richer narratives)
"""

import json
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# System Prompt for LLM guardrailing
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a traffic data translator for an AI Traffic Intelligence System.

STRICT RULES:
1. You will receive a structured data dictionary containing traffic metrics, health scores, predictions, and stakeholder-specific advisories.
2. Your ONLY job is to convert this structured data into EXACTLY ONE short, crisp, conversational sentence.
3. Sound like a helpful digital assistant (like Siri or Alexa) giving an instant insight.
4. Do NOT use markdown. Do NOT use bullet points. Do NOT write paragraphs. Maximum 20 words.
5. Example: "Traffic conditions are ideal for commuting right now."
6. Example: "Monitor Link 27, congestion is increasing."
7. You must NEVER invent data, make your own predictions, or generate recommendations that aren't already in the data.
"""


# ---------------------------------------------------------------------------
# Template-Based Narrative Generation (No API Key Needed)
# ---------------------------------------------------------------------------
def generate_narrative_template(profile: str, data: Dict) -> str:
    """
    Generate a natural language narrative from structured decision data
    using deterministic templates. Always available without an API key.

    Args:
        profile: Stakeholder profile name
        data: Structured output from the Decision Engine

    Returns:
        Human-readable narrative string
    """
    generators = {
        "Daily Commuter": _narrative_commuter,
        "Traffic Control Center": _narrative_control_center,
        "Emergency Services": _narrative_emergency,
        "Logistics & Fleet": _narrative_logistics,
        "City Planner": _narrative_city_planner,
        "City Bulletin": _narrative_city_bulletin,
        "City Commuter": _narrative_city_commuter,
        "City Traffic Police": _narrative_city_traffic_police,
        "City Logistics": _narrative_city_logistics,
    }

    generator = generators.get(profile, _narrative_generic)
    return generator(data)


def _narrative_commuter(data: Dict) -> str:
    return "Traffic conditions are currently manageable for your commute."

def _narrative_control_center(data: Dict) -> str:
    return "Monitor network for emerging congestion patterns and bottlenecks."

def _narrative_emergency(data: Dict) -> str:
    return "Fastest corridors have been identified for emergency routing."

def _narrative_logistics(data: Dict) -> str:
    return "Dispatch times are optimal for fuel efficiency and speed."

def _narrative_city_planner(data: Dict) -> str:
    return "Review recurring hotspots for potential capacity expansion."

def _narrative_city_bulletin(data: Dict) -> str:
    status = data.get("city_status", "Unknown")
    time_label = data.get("time_label", "right now")
    avg_speed = data.get("avg_speed", 0)
    crit_count = data.get("critical_count", 0)
    worst = data.get("worst_links", [])
    
    text = f"Welcome to the TrafficPulse City Bulletin for {time_label}. Overall city traffic is currently experiencing {status.lower()} conditions, with network speeds averaging {avg_speed:.0f} km/h. "
    
    if crit_count > 0:
        text += f"We are tracking {crit_count} critical bottlenecks across the network. "
        if worst:
            worst_id = worst[0].get("link_id", "Unknown")
            text += f"Drivers should expect significant delays on Link {worst_id} and consider alternate routes. "
    else:
        text += "The network is flowing smoothly with no major delays reported. "
        
    text += "Stay safe and enjoy your commute!"
    return text

def _narrative_city_commuter(data: Dict) -> str:
    return "City-wide conditions are generally clear for travel."

def _narrative_city_traffic_police(data: Dict) -> str:
    return "Critical links require monitoring and immediate deployment."

def _narrative_city_logistics(data: Dict) -> str:
    return "The overall network is flowing well for fleet dispatch."

def _narrative_generic(data: Dict) -> str:
    return "Data available for review."




# ---------------------------------------------------------------------------
# LLM-Based Narrative Generation (Optional — requires API key)
# ---------------------------------------------------------------------------
def generate_narrative_llm(
    profile: str,
    data: Dict,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> str:
    """
    Use an OpenAI-compatible API to generate richer natural language narratives.

    The LLM receives the structured data dict and system prompt constraining
    it to ONLY translate the provided data — never invent or alter numbers.

    Falls back to template generation on any error.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)

        user_message = (
            f"Stakeholder Profile: {profile}\n\n"
            f"Structured Traffic Data:\n"
            f"```json\n{json.dumps(data, indent=2, default=str)}\n```\n\n"
            f"Convert this data into a clear, well-organized narrative for a "
            f"{profile}. Use the exact numbers provided. Do not invent any data."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,  # Low temp for factual accuracy
            max_tokens=800,
        )

        return response.choices[0].message.content

    except Exception as e:
        # Graceful fallback to template-based generation
        return generate_narrative_template(profile, data)


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------
def generate_narrative(
    profile: str,
    data: Dict,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a narrative for the given stakeholder profile and data.

    Uses LLM if an API key is provided, otherwise falls back to
    template-based generation.
    """
    if api_key and api_key.strip():
        return generate_narrative_llm(profile, data, api_key, base_url, model)
    return generate_narrative_template(profile, data)
