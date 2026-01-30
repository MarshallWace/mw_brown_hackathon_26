"""Stats analysis and sentiment comparison logic."""

from dataclasses import dataclass, field
from typing import Any

from mw_hackathon_brown.models import PlayerMention, PlayerPosition


# Position-relevant stats mapping
POSITION_STATS: dict[PlayerPosition, list[str]] = {
    PlayerPosition.QB: ["passingYards", "passingTouchdowns", "interceptions", "QBRating"],
    PlayerPosition.RB: ["rushingYards", "rushingTouchdowns", "yardsPerRushAttempt", "receptions"],
    PlayerPosition.WR: ["receptions", "receivingYards", "receivingTouchdowns", "yardsPerReception"],
    PlayerPosition.TE: ["receptions", "receivingYards", "receivingTouchdowns"],
    PlayerPosition.DL: ["sacks", "tacklesForLoss", "totalTackles"],
    PlayerPosition.LB: ["totalTackles", "sacks", "interceptions"],
    PlayerPosition.CB: ["interceptions", "passesDefended", "totalTackles"],
    PlayerPosition.S: ["interceptions", "totalTackles", "passesDefended"],
    PlayerPosition.K: ["fieldGoalPct", "fieldGoalsMade"],
    PlayerPosition.P: ["puntingAverage", "puntsInside20"],
    PlayerPosition.OL: [],  # Linemen don't have meaningful individual stats
    PlayerPosition.UNKNOWN: [],
}


@dataclass
class StatThreshold:
    """Thresholds for converting a stat to a 1-10 score."""

    elite: float      # 9-10 range
    good: float       # 7-8 range
    average: float    # 5-6 range
    below_avg: float  # 3-4 range
    # Below this is 1-2 range


# Hardcoded thresholds based on typical NFL season performance
STAT_THRESHOLDS: dict[str, StatThreshold] = {
    # QB stats
    "passingYards": StatThreshold(4500, 3500, 2500, 1500),
    "passingTouchdowns": StatThreshold(35, 25, 15, 8),
    "interceptions": StatThreshold(5, 8, 12, 16),  # Lower is better
    "QBRating": StatThreshold(100, 90, 80, 70),
    # RB stats
    "rushingYards": StatThreshold(1200, 800, 500, 250),
    "rushingTouchdowns": StatThreshold(12, 8, 5, 2),
    "yardsPerRushAttempt": StatThreshold(5.0, 4.5, 4.0, 3.5),
    # WR/TE stats
    "receptions": StatThreshold(100, 70, 40, 20),
    "receivingYards": StatThreshold(1200, 800, 500, 250),
    "receivingTouchdowns": StatThreshold(10, 7, 4, 2),
    "yardsPerReception": StatThreshold(15, 13, 11, 9),
    # Defensive stats
    "sacks": StatThreshold(12, 8, 5, 2),
    "totalTackles": StatThreshold(120, 90, 60, 30),
    "tacklesForLoss": StatThreshold(15, 10, 6, 3),
    "interceptions": StatThreshold(6, 4, 2, 1),
    "passesDefended": StatThreshold(15, 10, 6, 3),
    # K/P stats
    "fieldGoalPct": StatThreshold(90, 85, 80, 75),
    "fieldGoalsMade": StatThreshold(30, 25, 20, 15),
    "puntingAverage": StatThreshold(48, 46, 44, 42),
    "puntsInside20": StatThreshold(30, 25, 20, 15),
}


@dataclass
class StatsAnalysis:
    """Analysis comparing sentiment to actual stats."""

    player_name: str
    position: PlayerPosition
    sentiment_score: int
    raw_stats: dict[str, Any] = field(default_factory=dict)
    relevant_stats: dict[str, Any] = field(default_factory=dict)
    stat_scores: dict[str, int] = field(default_factory=dict)
    computed_score: int | None = None
    delta: int | None = None
    assessment: str = "insufficient_data"  # "justified", "overrated", "underrated", "insufficient_data"


def _parse_stat_value(value: Any) -> float | None:
    """Parse a stat value to float, handling comma-separated numbers."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            # Remove commas and convert
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


def _stat_to_score(stat_name: str, value: float) -> int | None:
    """Convert a raw stat value to a 1-10 score based on thresholds."""
    if stat_name not in STAT_THRESHOLDS:
        return None

    t = STAT_THRESHOLDS[stat_name]

    # Special handling for stats where lower is better (interceptions for QBs)
    if stat_name == "interceptions":
        if value <= t.elite:
            return 10
        elif value <= t.good:
            return 8
        elif value <= t.average:
            return 6
        elif value <= t.below_avg:
            return 4
        else:
            return 2

    # Normal stats where higher is better
    if value >= t.elite:
        return 10
    elif value >= t.good:
        return 8
    elif value >= t.average:
        return 6
    elif value >= t.below_avg:
        return 4
    else:
        return 2


def analyze_player_stats(mention: PlayerMention, stats: dict[str, Any]) -> StatsAnalysis:
    """Compare a player's sentiment score to their actual stats.

    Parameters
    ----------
    mention
        The player mention with sentiment score.
    stats
        Raw stats dict from ESPN API.

    Returns
    -------
    StatsAnalysis with computed score and assessment.
    """
    # TODO - implement
    pass
