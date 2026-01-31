from collections import Counter
from pathlib import Path
import sys

from mw_hackathon_brown.models import DocumentSentimentResult, PlayerMention
from mw_hackathon_brown.player_matcher import MatchedPlayer
from mw_hackathon_brown.stats_analyzer import StatsAnalysis, analyze_player_stats


def print_separator(char: str = "=", length: int = 60) -> None:
    print(char * length)


def print_document_results(result: DocumentSentimentResult) -> None:
    print(f"\n[{result.source_file}] - {result.total_players_found} unique players found")
    print_separator("-", 40)

    for mention in result.player_mentions:
        strength_flag = " [STRENGTH]" if mention.is_strength else ""
        concern_flag = " [CONCERN]" if mention.is_concern else ""
        print(f"  {mention.player_name} ({mention.team}, {mention.position.value})")
        print(f"    Sentiment: {mention.sentiment_score}/10{strength_flag}{concern_flag}")
        print(f"    Context: {mention.context[:100]}..." if len(mention.context) > 100 else f"    Context: {mention.context}")
        print()


def print_summary(all_results: list[DocumentSentimentResult]) -> None:
    if not all_results:
        print("\nNo documents were successfully processed.")
        sys.exit(1)

    all_mentions: list[PlayerMention] = []
    for result in all_results:
        all_mentions.extend(result.player_mentions)

    print_separator("=")
    print("SUMMARY")
    print_separator("=")

    print(f"Total documents processed: {len(all_results)}")
    print(f"Total player mentions: {len(all_mentions)}")

    unique_players = {m.player_name for m in all_mentions}
    print(f"Unique players: {len(unique_players)}")

    print("\nSentiment Distribution:")
    sentiment_counts = Counter(m.sentiment_score for m in all_mentions)
    for score in sorted(sentiment_counts.keys()):
        bar = "#" * sentiment_counts[score]
        print(f"  {score:2d}: {bar} ({sentiment_counts[score]})")

    player_counts = Counter(m.player_name for m in all_mentions)
    print("\nTop 10 Most Mentioned Players:")
    for player, count in player_counts.most_common(10):
        avg_sentiment = sum(m.sentiment_score for m in all_mentions if m.player_name == player) / count
        print(f"  {player}: {count} mentions (avg sentiment: {avg_sentiment:.1f})")

    strengths = [m for m in all_mentions if m.is_strength]
    concerns = [m for m in all_mentions if m.is_concern]
    print(f"\nPlayers marked as strengths: {len(strengths)}")
    print(f"Players marked as concerns: {len(concerns)}")


def print_analysis_result(analysis: StatsAnalysis, matched: bool, match_score: int) -> None:
    print(f"\n  {analysis.player_name} ({analysis.position.value})")
    print(f"    Sentiment: {analysis.sentiment_score}/10")

    if not matched:
        print("    Match: NO MATCH FOUND")
        print("    Assessment: CANNOT ANALYZE - player not in ESPN database")
        return

    print(f"    Match: [score={match_score}%]")

    if analysis.relevant_stats:
        stats_str = ", ".join(f"{k}: {v}" for k, v in list(analysis.relevant_stats.items())[:4])
        print(f"    Stats: {stats_str}")

    if analysis.computed_score is not None:
        score_breakdown = ", ".join(f"{k}={v}" for k, v in analysis.stat_scores.items())
        print(f"    Computed Score: {analysis.computed_score}/10 ({score_breakdown})")
        delta_str = f"+{analysis.delta}" if analysis.delta and analysis.delta > 0 else str(analysis.delta)
        print(f"    Assessment: {analysis.assessment.upper()} (delta: {delta_str})")
    else:
        print("    Assessment: INSUFFICIENT DATA - no relevant stats found")


def get_players_csv_path() -> Path:
    return Path(__file__).parent.parent.parent / "players.csv"


def get_cache_path() -> Path:
    return Path(__file__).parent.parent.parent / "extraction_cache.json"


async def print_anaylysis_results(
        unique_mentions: list[PlayerMention], 
        matches: list[MatchedPlayer], 
        all_stats: list[dict]
    ) -> None:

    # Analyze and print results
    match_counts = {"llm": 0, "none": 0}
    assessment_counts = {"justified": 0, "overrated": 0, "underrated": 0, "insufficient_data": 0}

    for mention, match, stats in zip(unique_mentions, matches, all_stats):
        match_counts[match.match_type] += 1
        analysis = analyze_player_stats(mention, stats)
        assessment_counts[analysis.assessment] += 1
        print_analysis_result(analysis, matched=(match.player_id is not None), match_score=match.match_score)

    # Print analysis summary
    print_separator("-", 40)
    print("\nANALYSIS SUMMARY")
    print(f"  Matched: {match_counts['llm']} matched, {match_counts['none']} unmatched")
    print(f"  Assessments: {assessment_counts['justified']} justified, {assessment_counts['overrated']} overrated, {assessment_counts['underrated']} underrated, {assessment_counts['insufficient_data']} insufficient data")