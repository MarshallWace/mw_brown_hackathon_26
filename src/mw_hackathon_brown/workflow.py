"""NFL player sentiment extraction using LlamaIndex Workflow framework."""

import argparse
import asyncio

from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

from mw_hackathon_brown.core import (
    get_cache_path,
    get_players_csv_path,
    print_anaylysis_results,
    print_document_results,
    print_separator,
)
from mw_hackathon_brown.document_loader import get_preseason_documents_path, load_markdown_documents
from mw_hackathon_brown.espn_api import get_player_stats
from mw_hackathon_brown.extractor import NFLSentimentExtractor
from mw_hackathon_brown.llm_client import create_llm_client
from mw_hackathon_brown.models import DocumentSentimentResult, PlayerMention
from mw_hackathon_brown.player_matcher import MatchedPlayer, PlayerMatcher

import os

# --- Events ---


class DocumentLoadedEvent(Event):
    """Event emitted when documents are loaded."""

    documents: dict[str, str]


class SentimentExtractionEvent(Event):
    """Event emitted after extraction completes."""

    results: list[DocumentSentimentResult]


class PlayerMatchingEvent(Event):
    """Event emitted after player matching completes."""

    unique_mentions: list[PlayerMention]
    matches: list[MatchedPlayer]


class StatsFetchedEvent(Event):
    """Event emitted after stats are fetched from ESPN API."""

    unique_mentions: list[PlayerMention]
    matches: list[MatchedPlayer]
    all_stats: list[dict]


# --- Workflow ---


class NFLSentimentWorkflow(Workflow):
    """LlamaIndex Workflow for NFL player sentiment extraction and analysis."""

    def __init__(
        self,
        backend: str = "litellm",
        extraction_model: str = "gemini-2.5-pro",
        matcher_model: str = "gemini-2.5-flash",
        refresh: bool = False,
        **kwargs,
    ):
        """Initialize the workflow.

        Parameters
        ----------
        backend
            Which LLM backend to use ("litellm").
        extraction_model
            Model to use for sentiment extraction.
        matcher_model
            Model to use for player matching (faster model preferred).
        refresh
            If True, ignore cache and re-extract from documents.
        **kwargs
            Additional arguments passed to Workflow.
        """
        super().__init__(**kwargs)
        self.backend = backend
        self.extraction_model = extraction_model
        self.matcher_model = matcher_model
        self.refresh = refresh

    @step
    async def load_documents(self, ctx: Context, ev: StartEvent) -> DocumentLoadedEvent:
        """Load markdown documents and initialize extractor and matcher."""
        docs_path = get_preseason_documents_path()
        print(f"Loading documents from: {docs_path}")

        documents = load_markdown_documents(docs_path)
        print(f"Found {len(documents)} markdown files")

        # Create extractor
        print(f"\nInitializing extractor (backend={self.backend}, model={self.extraction_model})...")
        extraction_client = create_llm_client(backend=self.backend, model=self.extraction_model)
        extractor = NFLSentimentExtractor(llm_client=extraction_client)

        # Create matcher with flash model
        print(f"Initializing LLM client for player matching (backend={self.backend}, model={self.matcher_model})...")
        matcher_client = create_llm_client(backend=self.backend, model=self.matcher_model)
        players_csv = get_players_csv_path()
        print(f"Loading player database from: {players_csv}")
        matcher = PlayerMatcher(players_csv, llm_client=matcher_client)

        # Store in context
        await ctx.store.set("extractor", extractor)
        await ctx.store.set("extraction_client", extraction_client)
        await ctx.store.set("matcher", matcher)
        await ctx.store.set("documents", documents)

        return DocumentLoadedEvent(documents=documents)

    @step
    async def process_documents(self, ctx: Context, ev: DocumentLoadedEvent) -> SentimentExtractionEvent:
        """Process all documents with caching support."""
        extractor: NFLSentimentExtractor = await ctx.store.get("extractor")
        cache_path = get_cache_path()

        # Use extract_with_cache (handles cache load/save, parallel processing)
        results = extractor.extract_with_cache(
            documents=ev.documents,
            cache_path=cache_path,
            refresh=self.refresh,
        )

        # Print document results
        for result in results:
            print_document_results(result)

        return SentimentExtractionEvent(results=results)

    @step
    async def match_players(self, ctx: Context, ev: SentimentExtractionEvent) -> PlayerMatchingEvent:
        """Match extracted players to ESPN database."""
        # TODO - implement
        pass

    @step
    async def fetch_stats(self, ctx: Context, ev: PlayerMatchingEvent) -> StatsFetchedEvent:
        """Fetch stats for all matched players from ESPN API."""
        print("Fetching stats from ESPN API...")

        all_stats = await asyncio.gather(*[get_player_stats(m.player_id) for m in ev.matches])

        return StatsFetchedEvent(
            unique_mentions=ev.unique_mentions,
            matches=ev.matches,
            all_stats=all_stats,
        )

    @step
    async def analyze_results(self, ctx: Context, ev: StatsFetchedEvent) -> StopEvent:
        """Analyze sentiment vs actual stats and print results."""
        await print_anaylysis_results(ev.unique_mentions, ev.matches, ev.all_stats)

        extraction_client = await ctx.store.get("extraction_client")
        if hasattr(extraction_client, "get_total_cost"):
            print(f"\nTotal API cost: ${extraction_client.get_total_cost():.4f}")

        return StopEvent(
            result={
                "unique_mentions": ev.unique_mentions,
                "matches": ev.matches,
                "all_stats": ev.all_stats,
            }
        )


async def run_workflow(
    backend: str = "litellm",
    extraction_model: str = "gemini-2.5-flash",
    matcher_model: str = "gemini-2.5-flash",
    refresh: bool = False,
) -> dict:
    """Run the NFL sentiment extraction workflow.

    Parameters
    ----------
    backend
        Which LLM backend to use ("litellm").
    extraction_model
        Model to use for sentiment extraction.
    matcher_model
        Model to use for player matching.
    refresh
        If True, ignore cache and re-extract from documents.

    Returns
    -------
    Dict with unique_mentions, matches, and all_stats.
    """
    print_separator("=")
    print(f"NFL Player Sentiment Extraction (LlamaIndex + {backend})")
    print_separator("=")

    workflow = NFLSentimentWorkflow(
        backend=backend,
        extraction_model=extraction_model,
        matcher_model=matcher_model,
        refresh=refresh,
        timeout=300,
        verbose=False,
    )

    result = await workflow.run()
    print("\nDone!")
    return result


def main() -> None:
    """Entry point for running the LlamaIndex workflow."""
    parser = argparse.ArgumentParser(description="NFL Player Sentiment Extraction & Analysis (LlamaIndex Workflow)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cache and re-extract from documents",
    )
    args = parser.parse_args()

    asyncio.run(
        run_workflow(
            backend="litellm",
            extraction_model="gemini-2.5-flash",
            matcher_model="gemini-2.5-flash",
            refresh=args.refresh,
        )
    )


if __name__ == "__main__":
    main()
