"""Pydantic models for NFL player sentiment extraction."""

from enum import Enum

from pydantic import BaseModel, Field


class PlayerPosition(str, Enum):
    """NFL player position enum."""

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    OL = "OL"
    DL = "DL"
    LB = "LB"
    CB = "CB"
    S = "S"
    K = "K"
    P = "P"
    UNKNOWN = "UNKNOWN"


class PlayerMention(BaseModel):
    """A single player mention extracted from a document."""

    player_name: str = Field(description="Full name of the NFL player")
    team: str = Field(description="NFL team name or abbreviation")
    position: PlayerPosition = Field(description="Player's position")
    sentiment_score: int = Field(
        ge=1, le=10, description="Sentiment score from 1 (very negative) to 10 (very positive)"
    )
    context: str = Field(description="Brief excerpt or summary of the context where the player was mentioned")
    is_strength: bool = Field(default=False, description="Whether this mention is related to a team strength")
    is_concern: bool = Field(default=False, description="Whether this mention is related to a team concern")


class DocumentSentimentResult(BaseModel):
    """Result of sentiment extraction from a single document."""

    source_file: str = Field(description="Name of the source file")
    player_mentions: list[PlayerMention] = Field(default_factory=list, description="List of player mentions found")
    total_players_found: int = Field(default=0, description="Total number of unique players found")
