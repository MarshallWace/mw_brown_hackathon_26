"""Player name matching against ESPN player database."""

import csv
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from thefuzz import fuzz, process
from tqdm.asyncio import tqdm_asyncio

from mw_hackathon_brown.llm_client import BaseLLMClient
from mw_hackathon_brown.models import DocumentSentimentResult, PlayerMention


@dataclass
class MatchedPlayer:
    """Result of matching a player name to the ESPN database."""

    original_name: str
    matched_name: str | None
    player_id: str | None
    match_score: int
    match_type: str  # "llm" or "none"


class LLMMatchChoice(BaseModel):
    """LLM's choice for best matching player."""

    chosen_index: int = Field(description="Index (0-4) of the best matching player, or -1 if none match")
    confidence: str = Field(description="high, medium, or low")
    reasoning: str = Field(description="Brief explanation for the choice")


class PlayerMatcher:
    """Matches player names to ESPN player IDs using CSV lookup and LLM selection."""

    def __init__(
        self,
        csv_path: Path | str,
        llm_client: BaseLLMClient,
        fuzzy_threshold: int = 60,
    ) -> None:
        """Load player database from CSV.

        Parameters
        ----------
        csv_path
            Path to players.csv with columns: name, player_id
        llm_client
            LLM client for selecting best match from candidates.
        fuzzy_threshold
            Minimum fuzzy score to be considered a candidate.
        """
        self._llm_client = llm_client
        self._fuzzy_threshold = fuzzy_threshold
        self._exact_lookup: dict[str, tuple[str, str]] = {}  # normalized -> (original_name, id)
        self._all_names: list[tuple[str, str]] = []  # [(name, player_id), ...]
        self._name_to_id: dict[str, str] = {}

        self._load_csv(Path(csv_path))

    def _normalize(self, name: str) -> str:
        """Normalize name for exact matching."""
        return " ".join(name.lower().strip().split())

    def _load_csv(self, csv_path: Path) -> None:
        """Load player data from CSV."""
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["name"]
                player_id = row["player_id"]
                normalized = self._normalize(name)
                self._exact_lookup[normalized] = (name, player_id)
                self._all_names.append((name, player_id))
                self._name_to_id[name] = player_id

    def match(self, name: str, team: str = "", position: str = "") -> MatchedPlayer:
        """Match a player name to the database using LLM selection.

        Always uses LLM to pick from top 5 fuzzy matches, since multiple
        players can share the same name (e.g., Josh Allen QB vs Josh Allen LB).

        Parameters
        ----------
        name
            Player name to match.
        team
            Team abbreviation for context (e.g., "BAL").
        position
            Position for context (e.g., "QB").
        """
        # Get top 5 fuzzy matches - always use LLM to disambiguate
        name_list = [n for n, _ in self._all_names]
        candidates = process.extract(name, name_list, scorer=fuzz.token_sort_ratio, limit=5)

        # Filter by threshold
        candidates = [(n, score) for n, score in candidates if score >= self._fuzzy_threshold]

        if not candidates:
            return MatchedPlayer(
                original_name=name,
                matched_name=None,
                player_id=None,
                match_score=0,
                match_type="none",
            )

        # Use LLM to pick the best match
        choice = self._llm_select_match(name, team, position, candidates)

        if choice.chosen_index < 0 or choice.chosen_index >= len(candidates):
            return MatchedPlayer(
                original_name=name,
                matched_name=None,
                player_id=None,
                match_score=0,
                match_type="none",
            )

        matched_name, match_score = candidates[choice.chosen_index]
        player_id = self._name_to_id[matched_name]

        return MatchedPlayer(
            original_name=name,
            matched_name=matched_name,
            player_id=player_id,
            match_score=match_score,
            match_type="llm",
        )

    def _build_match_prompt(
        self,
        query_name: str,
        team: str,
        position: str,
        candidates: list[tuple[str, int]],
    ) -> list[dict]:
        """Build the prompt for LLM match selection."""
        candidates_text = "\n".join(
            f"  {i}: {name} (fuzzy score: {score})"
            for i, (name, score) in enumerate(candidates)
        )

        context_parts = []
        if team:
            context_parts.append(f"Team: {team}")
        if position:
            context_parts.append(f"Position: {position}")
        context_text = ", ".join(context_parts) if context_parts else "No additional context"

        prompt = f"""You are matching NFL player names from news articles to an official ESPN player database.

Player mentioned in article: "{query_name}"
Context: {context_text}

Candidate matches from database:
{candidates_text}

Pick the index (0-{len(candidates)-1}) of the player that best matches. Consider:
- Name similarity (typos, nicknames, abbreviated names)
- If the article mentions a team/position, prefer candidates that could match
- If none are a good match, return -1

Choose the best match."""

        return [{"role": "user", "content": prompt}]

    def _llm_select_match(
        self,
        query_name: str,
        team: str,
        position: str,
        candidates: list[tuple[str, int]],
    ) -> LLMMatchChoice:
        """Use LLM to select the best match from candidates."""
        messages = self._build_match_prompt(query_name, team, position, candidates)
        return self._llm_client.structured_completion(messages, LLMMatchChoice)

    async def _allm_select_match(
        self,
        query_name: str,
        team: str,
        position: str,
        candidates: list[tuple[str, int]],
    ) -> LLMMatchChoice:
        """Async version: Use LLM to select the best match from candidates."""
        messages = self._build_match_prompt(query_name, team, position, candidates)
        return await self._llm_client.astructured_completion(messages, LLMMatchChoice)

    def get_candidates(self, name: str) -> list[tuple[str, int]]:
        """Get fuzzy match candidates for a player name (no LLM call).

        Parameters
        ----------
        name
            Player name to match.

        Returns
        -------
            List of (candidate_name, fuzzy_score) tuples above threshold.
        """
        name_list = [n for n, _ in self._all_names]
        candidates = process.extract(name, name_list, scorer=fuzz.token_sort_ratio, limit=5)
        return [(n, score) for n, score in candidates if score >= self._fuzzy_threshold]

    async def match_async(self, name: str, team: str = "", position: str = "") -> MatchedPlayer:
        """Async version of match - uses async LLM call.

        Parameters
        ----------
        name
            Player name to match.
        team
            Team abbreviation for context (e.g., "BAL").
        position
            Position for context (e.g., "QB").
        """
        candidates = self.get_candidates(name)

        if not candidates:
            return MatchedPlayer(
                original_name=name,
                matched_name=None,
                player_id=None,
                match_score=0,
                match_type="none",
            )

        choice = await self._allm_select_match(name, team, position, candidates)

        if choice.chosen_index < 0 or choice.chosen_index >= len(candidates):
            return MatchedPlayer(
                original_name=name,
                matched_name=None,
                player_id=None,
                match_score=0,
                match_type="none",
            )

        matched_name, match_score = candidates[choice.chosen_index]
        player_id = self._name_to_id[matched_name]

        return MatchedPlayer(
            original_name=name,
            matched_name=matched_name,
            player_id=player_id,
            match_score=match_score,
            match_type="llm",
        )

    @staticmethod
    def get_unique_mentions(results: list[DocumentSentimentResult]) -> list[PlayerMention]:
        """Extract unique player mentions from extraction results.

        Deduplicates by (name, team, position) to handle same-name players.

        Parameters
        ----------
        results
            List of document extraction results.

        Returns
        -------
        List of unique PlayerMention objects.
        """
        seen: set[tuple[str, str, str]] = set()
        unique: list[PlayerMention] = []
        for result in results:
            for mention in result.player_mentions:
                key = (mention.player_name, mention.team, mention.position.value)
                if key not in seen:
                    seen.add(key)
                    unique.append(mention)
        return unique

    async def match_all_async(
        self,
        results: list[DocumentSentimentResult],
    ) -> tuple[list[PlayerMention], list[MatchedPlayer]]:
        """Match all unique players from extraction results in parallel.

        Parameters
        ----------
        results
            List of document extraction results.

        Returns
        -------
        Tuple of (unique_mentions, matches) in the same order.
        """
        unique_mentions = self.get_unique_mentions(results)
        print(f"\nAnalyzing {len(unique_mentions)} unique players...")

        async def match_one(mention: PlayerMention) -> MatchedPlayer:
            return await self.match_async(
                mention.player_name,
                team=mention.team,
                position=mention.position.value,
            )

        tasks = [match_one(m) for m in unique_mentions]
        matches = await tqdm_asyncio.gather(*tasks, desc="Matching players", unit="player")
        return unique_mentions, matches
