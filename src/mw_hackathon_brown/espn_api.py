"""ESPN NFL Player API integration.

Provides functionality to:
- Fetch all NFL players from ESPN's paginated API
- Save player name→ID mappings to CSV
- Fetch season stats for any player by ID
"""

import asyncio
import csv
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm_asyncio

BASE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
STATS_URL = "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes"

MAX_CONCURRENT_REQUESTS = 50


async def _fetch_json(session: aiohttp.ClientSession, url: str) -> dict | None:
    """Fetch JSON from a URL."""
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientError:
        return None


async def _fetch_player_details(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    ref_url: str,
) -> tuple[str, str, str, str] | None:
    """Fetch player details from a $ref URL, then fetch team and position.

    Args:
        session: aiohttp client session.
        semaphore: Semaphore to limit concurrent requests.
        ref_url: The ESPN API reference URL for the player.

    Returns:
        Tuple of (fullName, playerId, team, position) or None if fetch fails.
    """
    async with semaphore:
        data = await _fetch_json(session, ref_url)
        if not data:
            return None

        player_id = data.get("id")
        full_name = data.get("fullName")
        if not player_id or not full_name:
            return None

        # Fetch team and position from athlete endpoint
        athlete_url = f"{STATS_URL}/{player_id}"
        athlete_data = await _fetch_json(session, athlete_url)

        team = ""
        position = ""
        if athlete_data:
            athlete = athlete_data.get("athlete", {})
            team_info = athlete.get("team", {})
            position_info = athlete.get("position", {})
            team = team_info.get("displayName", "")
            position = position_info.get("displayName", "")

        return (full_name, player_id, team, position)
    return None


async def _fetch_all_players(page_limit: int | None = None) -> list[tuple[str, str, str, str]]:
    """Fetch all NFL players from ESPN's paginated API.

    Args:
        page_limit: Optional limit on number of pages to fetch (for testing).
                   If None, fetches all pages.

    Returns:
        List of (fullName, playerId, team, position) tuples.
    """
    async with aiohttp.ClientSession() as session:
        # First request to get page info
        url = f"{BASE_URL}/athletes?limit=1000&active=true&page=1"
        data = await _fetch_json(session, url)
        if not data:
            return []

        total_pages = data.get("pageCount", 7)
        if page_limit:
            total_pages = min(total_pages, page_limit)

        # Collect all refs from all pages
        all_refs = [item.get("$ref") for item in data.get("items", []) if item.get("$ref")]

        # Fetch remaining pages concurrently
        if total_pages > 1:
            page_urls = [
                f"{BASE_URL}/athletes?limit=1000&active=true&page={page}"
                for page in range(2, total_pages + 1)
            ]
            page_tasks = [_fetch_json(session, url) for url in page_urls]
            page_results = await asyncio.gather(*page_tasks)

            for page_data in page_results:
                if page_data:
                    all_refs.extend(
                        item.get("$ref") for item in page_data.get("items", []) if item.get("$ref")
                    )

        # Fetch all player details concurrently with semaphore
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [_fetch_player_details(session, semaphore, ref_url) for ref_url in all_refs]

        results = await tqdm_asyncio.gather(*tasks, desc="Fetching players", unit="player")

        return [player for player in results if player is not None]


def _save_players_to_csv(players: list[tuple[str, str, str, str]], filename: str = "players.csv") -> None:
    """Save player mappings to CSV file.

    Args:
        players: List of (fullName, playerId, team, position) tuples.
        filename: Output CSV filename.
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "player_id", "team", "position"])
        for name, player_id, team, position in players:
            writer.writerow([name, player_id, team, position])


async def get_player_stats(player_id: str | int | None) -> dict[str, Any]:
    """Fetch season stats for a player.

    Args:
        player_id: The ESPN player ID, or None.

    Returns:
        Dictionary mapping stat names to values (e.g., {"rushingYards": "1,585"}).
        Returns empty dict if player_id is None or stats not available.
    """
    if player_id is None:
        return {}

    url = f"{STATS_URL}/{player_id}/overview"
    async with aiohttp.ClientSession() as session:
        data = await _fetch_json(session, url)

    if not data:
        return {}

    stats = {}
    statistics = data.get("statistics", {})
    names = statistics.get("names", [])
    splits = statistics.get("splits", [])

    # Get Regular Season stats (first split)
    if splits and len(splits) > 0:
        regular_season = splits[0]
        stat_values = regular_season.get("stats", [])
        for name, value in zip(names, stat_values):
            stats[name] = value

    return stats


async def main() -> None:
    """Fetch all players and save to CSV, then demo stats lookup."""
    print("Fetching all NFL players from ESPN API...")
    players = await _fetch_all_players()
    print(f"\nFetched {len(players)} players")

    _save_players_to_csv(players)
    print("Saved players to players.csv")

    # Demo: fetch stats for player 4242335 (example from plan)
    print("\nFetching stats for player 4430027...")
    stats = await get_player_stats(4430027)
    for stat_name, value in stats.items():
        print(f"  {stat_name}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
