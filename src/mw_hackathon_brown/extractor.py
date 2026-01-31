"""NFL player sentiment extraction using LLM clients."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mw_hackathon_brown.llm_client import BaseLLMClient, create_llm_client
from mw_hackathon_brown.models import DocumentSentimentResult


EXTRACTION_PROMPT = """You are an NFL analyst extracting player mentions from preseason analysis articles.

Analyze the following document and extract ALL NFL player mentions with sentiment scores.

For each player mentioned:
1. Extract the player's full name
2. Identify their team (use standard abbreviations like BAL, PHI, KC, etc.)
3. Determine their position (QB, RB, WR, TE, OL, DL, LB, CB, S, K, P, or UNKNOWN if unclear)
4. Assign a sentiment score from 1-10:
   - 1-3: Very negative (major concerns, poor performance, injuries)
   - 4-5: Somewhat negative (minor concerns, questions)
   - 6-7: Neutral to positive (mentioned without strong opinion)
   - 8-9: Positive (praised, expected good performance)
   - 10: Extremely positive (MVP candidate, exceptional praise)
5. Extract a brief context (1-2 sentences) explaining why the player was mentioned
6. Mark is_strength=true if the player is mentioned as a team strength
7. Mark is_concern=true if the player is mentioned as a team concern

Source file: {source_file}

Document content:
{content}

Extract all player mentions with their sentiment scores."""


class NFLSentimentExtractor:
    """Extracts NFL player sentiment from documents using an LLM client."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        backend: str = "litellm",
        model: str | None = None,
    ):
        """Initialize the extractor with an LLM client.

        Args:
            llm_client: Pre-configured LLM client. If None, creates one using backend/model.
            backend: Which backend to use if llm_client is None ("litellm").
            model: Model name if llm_client is None.
        """
        if llm_client is not None:
            self._client = llm_client
        else:
            self._client = create_llm_client(backend=backend, model=model)

    def extract_from_text(self, content: str, source_name: str) -> DocumentSentimentResult:
        """Extract player mentions and sentiment from document text.

        Args:
            content: The document text content.
            source_name: The name of the source file.

        Returns:
            DocumentSentimentResult with extracted player mentions.
        """
        prompt = EXTRACTION_PROMPT.format(source_file=source_name, content=content)
        messages = [{"role": "user", "content": prompt}]

        result = self._client.structured_completion(messages, DocumentSentimentResult)

        # Update total_players_found based on unique player names
        unique_players = {mention.player_name for mention in result.player_mentions}
        result.total_players_found = len(unique_players)
        result.source_file = source_name

        return result

    def extract_from_documents(
        self,
        documents: dict[str, str],
        max_workers: int = 4,
    ) -> list[DocumentSentimentResult]:
        """Extract player mentions from multiple documents in parallel.

        Parameters
        ----------
        documents
            Dict mapping filename to content.
        max_workers
            Number of parallel threads for processing.

        Returns
        -------
        List of extraction results (failed documents are excluded).
        """
        def process_doc(item: tuple[str, str]) -> DocumentSentimentResult | None:
            filename, content = item
            try:
                return self.extract_from_text(content, filename)
            except Exception as e:
                print(f"ERROR processing {filename}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_doc, documents.items()))

        return [r for r in results if r is not None]

    @staticmethod
    def save_cache(results: list[DocumentSentimentResult], cache_path: Path) -> None:
        """Save extraction results to cache file."""
        data = [r.model_dump() for r in results]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved extraction cache to {cache_path}")

    @staticmethod
    def load_cache(cache_path: Path) -> list[DocumentSentimentResult] | None:
        """Load extraction results from cache file if it exists."""
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            results = [DocumentSentimentResult.model_validate(d) for d in data]
            print(f"Loaded {len(results)} documents from cache: {cache_path}")
            return results
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None

    def extract_with_cache(
        self,
        documents: dict[str, str],
        cache_path: Path,
        refresh: bool = False,
        max_workers: int = 4,
    ) -> list[DocumentSentimentResult]:
        """Extract from documents with caching support.

        Parameters
        ----------
        documents
            Dict mapping filename to content.
        cache_path
            Path to cache file.
        refresh
            If True, ignore cache and re-extract.
        max_workers
            Number of parallel threads for processing.

        Returns
        -------
        List of extraction results.
        """
        # Try cache first
        if not refresh:
            cached = self.load_cache(cache_path)
            if cached is not None:
                return cached

        # Extract from documents
        print(f"Processing {len(documents)} documents in parallel...")
        results = self.extract_from_documents(documents, max_workers=max_workers)

        # Save to cache
        if results:
            self.save_cache(results, cache_path)

        return results
