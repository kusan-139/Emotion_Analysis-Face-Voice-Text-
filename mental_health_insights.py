"""Non-clinical wellbeing indicator based on emotion history.

IMPORTANT:
- This module DOES NOT perform any medical, psychological, or clinical diagnosis.
- It only provides simple, heuristic wellbeing indicators from observed emotions.
- It must NOT be used as a substitute for professional help.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Iterable, Optional
from collections import Counter
from datetime import datetime
import math


# -------------------------------------------------------------------
# Emotion groups
# -------------------------------------------------------------------
POSITIVE_EMOTIONS = {"happy", "surprise"}
NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}
NEUTRAL_EMOTIONS = {"neutral"}


# -------------------------------------------------------------------
# Basic helpers (used by both simple & multimodal paths)
# -------------------------------------------------------------------
def _normalize_distribution(counts: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(counts.values())) or 1.0
    return {k: v / total for k, v in counts.items()}


def _aggregate_groups(dist: Dict[str, float]) -> Dict[str, float]:
    pos = sum(dist.get(e, 0.0) for e in POSITIVE_EMOTIONS)
    neg = sum(dist.get(e, 0.0) for e in NEGATIVE_EMOTIONS)
    neu = sum(dist.get(e, 0.0) for e in NEUTRAL_EMOTIONS)
    total = pos + neg + neu or 1.0
    return {
        "positivity": pos / total,
        "negativity": neg / total,
        "neutrality": neu / total,
    }


def wellbeing_indicator(scores: Dict[str, float]) -> str:
    """Simple 3-level indicator based mainly on negativity."""
    pos = scores.get("positivity", 0.0)
    neg = scores.get("negativity", 0.0)

    # If both neg & pos are high -> "Mixed concern"
    if neg >= 0.5 and pos >= 0.3:
        return "Mixed emotions (monitor)"

    if neg >= 0.6:
        return "High concern"
    elif neg >= 0.3:
        return "Moderate concern"
    else:
        return "Low concern"


def _trend_label(past_neg: float, current_neg: float, eps: float = 0.05) -> str:
    """Very simple trend: is negativity going up, down, or similar?"""
    delta = current_neg - past_neg
    if delta > eps:
        return "increasing"
    elif delta < -eps:
        return "decreasing"
    return "stable"


# -------------------------------------------------------------------
# SIMPLE (backwards compatible) API
# -------------------------------------------------------------------
def compute_scores_simple(emotion_history: List[str]) -> Dict[str, float]:
    """Old behavior: unweighted counts over a list of labels."""
    counter = Counter(emotion_history)
    dist = _normalize_distribution(counter)
    return _aggregate_groups(dist)


def generate_insight_text(scores: Dict[str, float],
                          trend: Optional[str] = None) -> str:
    neg = scores["negativity"]
    pos = scores["positivity"]
    neu = scores["neutrality"]

    lines = []
    lines.append(
        f"Emotion distribution (approximate): "
        f"Positivity: {pos:.0%}, Negativity: {neg:.0%}, Neutrality: {neu:.0%}."
    )

    if trend:
        lines.append(f"Overall negative emotion trend appears to be {trend} recently.")

    if neg >= 0.6:
        lines.append(
            "High levels of sadness/anger/fear have been detected recently. "
            "This may indicate stress or a low mood in the recent interactions."
        )
    elif neg >= 0.3:
        lines.append(
            "Noticeable negative emotions are present alongside neutral or positive ones. "
            "This may reflect mixed moods, occasional stress, or frustration."
        )
    else:
        lines.append(
            "Negative emotions appear relatively low, with more neutral or positive expressions. "
            "This may reflect a generally stable or positive visible mood."
        )

    lines.append(
        "IMPORTANT: This is NOT a diagnosis or mental health assessment. "
        "It is only an informational wellbeing indicator based on observed emotions "
        "and should not be used as a substitute for professional help."
    )
    return "\n".join(lines)


def get_wellbeing_summary(emotion_history: List[str]) -> Dict[str, str]:
    """
    Backwards-compatible entry point.
    Use this if you just have a flat list of emotion labels.
    """
    if not emotion_history:
        scores = {"positivity": 0.0, "negativity": 0.0, "neutrality": 1.0}
        indicator = wellbeing_indicator(scores)
        text = generate_insight_text(scores)
        return {
            "positivity_score": f"{scores['positivity']:.3f}",
            "negativity_score": f"{scores['negativity']:.3f}",
            "neutrality_score": f"{scores['neutrality']:.3f}",
            "wellbeing_indicator": indicator,
            "trend": "unknown",
            "insight_text": text,
        }

    scores = compute_scores_simple(emotion_history)
    indicator = wellbeing_indicator(scores)
    text = generate_insight_text(scores)
    return {
        "positivity_score": f"{scores['positivity']:.3f}",
        "negativity_score": f"{scores['negativity']:.3f}",
        "neutrality_score": f"{scores['neutrality']:.3f}",
        "wellbeing_indicator": indicator,
        "trend": "n/a",
        "insight_text": text,
    }


# -------------------------------------------------------------------
# ADVANCED MULTIMODAL API (optional, for future use)
# -------------------------------------------------------------------
@dataclass
class EmotionEvent:
    """
    Represents a single emotion prediction.

    label:  one of {"angry","disgust","fear","happy","sad","surprise","neutral"}
    modality: "face", "text", or "audio" (or any string tag you like)
    timestamp: datetime of when this was observed
    weight: optional manual weight (e.g., give text more weight than face)
    """
    label: str
    modality: str
    timestamp: datetime
    weight: float = 1.0


def _time_decay_weight(now: datetime,
                       t: datetime,
                       half_life_minutes: float = 30.0) -> float:
    """
    Exponential decay: more recent events get higher weight.
    After 'half_life_minutes', the weight ~0.5 of original.
    """
    dt_min = max((now - t).total_seconds() / 60.0, 0.0)
    if half_life_minutes <= 0:
        return 1.0
    return 0.5 ** (dt_min / half_life_minutes)


def compute_scores_multimodal(
    events: Iterable[EmotionEvent],
    half_life_minutes: float = 30.0,
) -> Dict[str, float]:
    """
    Weighted scores over multimodal events.
    - More recent events are weighted more (time decay).
    - Each event can have its own 'weight' (e.g. trust text more).
    """
    events = list(events)
    if not events:
        return {"positivity": 0.0, "negativity": 0.0, "neutrality": 1.0}

    now = max(e.timestamp for e in events)
    weighted_counts: Dict[str, float] = {}

    for e in events:
        decay = _time_decay_weight(now, e.timestamp, half_life_minutes)
        w = e.weight * decay
        weighted_counts[e.label] = weighted_counts.get(e.label, 0.0) + w

    dist = _normalize_distribution(weighted_counts)
    return _aggregate_groups(dist)


def get_wellbeing_summary_multimodal(
    events: List[EmotionEvent],
    lookback_fraction: float = 0.5,
    half_life_minutes: float = 30.0,
) -> Dict[str, str]:
    """
    Advanced summary for multimodal events with trend.

    - 'events' should be sorted by time (oldest -> newest) if possible.
    - We compute:
        * current_scores: all events (time-decayed)
        * past_scores: older half of events (time-decayed)
        * trend: negativity increasing / decreasing / stable
    """
    if not events:
        return get_wellbeing_summary([])

    events_sorted = sorted(events, key=lambda e: e.timestamp)
    n = len(events_sorted)
    split_idx = max(1, int(n * lookback_fraction))

    past_events = events_sorted[:split_idx]
    recent_events = events_sorted[split_idx:]

    current_scores = compute_scores_multimodal(events_sorted, half_life_minutes)
    past_scores = compute_scores_multimodal(past_events, half_life_minutes)

    trend = _trend_label(
        past_scores.get("negativity", 0.0),
        current_scores.get("negativity", 0.0),
    )

    indicator = wellbeing_indicator(current_scores)
    text = generate_insight_text(current_scores, trend=trend)

    return {
        "positivity_score": f"{current_scores['positivity']:.3f}",
        "negativity_score": f"{current_scores['negativity']:.3f}",
        "neutrality_score": f"{current_scores['neutrality']:.3f}",
        "wellbeing_indicator": indicator,
        "trend": trend,
        "insight_text": text,
    }
