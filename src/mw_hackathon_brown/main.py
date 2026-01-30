"""Main entry point for NFL player sentiment extraction."""

import argparse
import asyncio

from mw_hackathon_brown.core import get_cache_path, get_players_csv_path, print_anaylysis_results, print_document_results, print_separator
from mw_hackathon_brown.document_loader import get_preseason_documents_path, load_markdown_documents
from mw_hackathon_brown.espn_api import get_player_stats
from mw_hackathon_brown.extractor import NFLSentimentExtractor
from mw_hackathon_brown.llm_client import create_llm_client
from mw_hackathon_brown.player_matcher import PlayerMatcher



async def main_async(refresh: bool = False) -> None:
    """Run the NFL player sentiment extraction and analysis.

    Parameters
    ----------
    refresh
        If True, ignore cache and re-extract from documents.
    """
    # LLM configuration
    backend = "litellm"
    model = "gemini-2.5-pro"

    print_separator("=")
    print("NFL Player Sentiment Extraction & Analysis")
    print_separator("=")

    # Load documents
    docs_path = get_preseason_documents_path()
    print(f"Loading documents from: {docs_path}")

    documents = load_markdown_documents(docs_path)
    print(f"Found {len(documents)} markdown files")

    # Initialize extractor and process with caching
    print(f"\nInitializing extractor (backend={backend}, model={model or 'default'})...")
    extractor = NFLSentimentExtractor(backend=backend, model=model)

    all_results = extractor.extract_with_cache(
        documents=documents,
        cache_path=get_cache_path(),
        refresh=refresh,
    )

    # Print results
    for result in all_results:
        print_document_results(result)

    # Create LLM client for player matching (use flash model for speed)
    print(f"\nInitializing LLM client for player matching (backend={backend}, model=gemini-2.5-flash)...")
    match_llm = create_llm_client(backend=backend, model="gemini-2.5-flash")

    # Load player matcher
    players_csv = get_players_csv_path()

    print(f"Loading player database from: {players_csv}")
    matcher = PlayerMatcher(players_csv, llm_client=match_llm)

    # Run sentiment vs stats analysis
    print_separator("=")
    print("SENTIMENT VS STATS ANALYSIS")
    print_separator("=")

    # Match all unique players in parallel
    unique_mentions, matches = await matcher.match_all_async(all_results)

    # Fetch all stats in parallel
    print("Fetching stats from ESPN API...")
    all_stats = await asyncio.gather(*[get_player_stats(m.player_id) for m in matches])

    await print_anaylysis_results(unique_mentions, matches, all_stats)

    print("\nDone!")


def main() -> None:
    """Entry point wrapper for async main."""
    parser = argparse.ArgumentParser(description="NFL Player Sentiment Extraction & Analysis")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cache and re-extract from documents",
    )
    args = parser.parse_args()

    asyncio.run(main_async(refresh=args.refresh))


if __name__ == "__main__":
    main()
