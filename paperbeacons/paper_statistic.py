import json
import os
from pathlib import Path
from collections import Counter, defaultdict


def analyze_status_distribution(json_file):
    """
    Analyze status distribution for a conference

    Args:
        json_file (str): Path to the JSON file

    Returns:
        dict: Status distribution statistics
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Count status distribution
        status_counts = Counter()
        track_status = defaultdict(Counter)

        for paper in data:
            status = paper.get("status", "Unknown").strip()
            track = paper.get("track", "Unknown").strip()

            status_counts[status] += 1
            track_status[track][status] += 1

        total_papers = len(data)

        return {
            "total_papers": total_papers,
            "status_distribution": dict(status_counts),
            "track_status_distribution": dict(track_status),
            "status_percentages": {
                status: round(count / total_papers * 100, 2)
                for status, count in status_counts.items()
            },
        }

    except Exception as e:
        print(f"❌ Error analyzing {json_file}: {str(e)}")
        return None


def print_conference_analysis(conference_name, stats):
    """
    Pretty print conference analysis results

    Args:
        conference_name (str): Name of the conference
        stats (dict): Statistics dictionary
    """
    if not stats:
        return

    print(f"\n{'='*60}")
    print(f"📊 {conference_name.upper()} STATUS ANALYSIS")
    print(f"{'='*60}")

    print(f"📈 Total Papers: {stats['total_papers']:,}")

    print(f"\n🎯 Status Distribution:")
    print("-" * 40)
    for status, count in sorted(
        stats["status_distribution"].items(), key=lambda x: x[1], reverse=True
    ):
        percentage = stats["status_percentages"][status]
        print(f"  {status:<20}: {count:>6,} ({percentage:>5.1f}%)")

    print(f"\n📋 Track-wise Status Distribution:")
    print("-" * 50)
    for track, track_stats in stats["track_status_distribution"].items():
        if track == "Unknown" or not track:
            continue
        print(f"\n  🏷️  {track}:")
        for status, count in sorted(
            track_stats.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {status:<18}: {count:>4,}")


def generate_summary_report(all_stats):
    """
    Generate a summary report comparing all conferences

    Args:
        all_stats (dict): All conference statistics
    """
    print(f"\n{'='*70}")
    print(f"🌟 CONFERENCE STATUS SUMMARY")
    print(f"{'='*70}")

    # Total papers comparison
    print(f"\n📊 Paper Count Summary:")
    print("-" * 50)
    total_all = 0
    for conf_name, stats in all_stats.items():
        if stats:
            count = stats["total_papers"]
            total_all += count
            print(f"  {conf_name.upper():<12}: {count:>8,} papers")
    print(f"  {'TOTAL':<12}: {total_all:>8,} papers")

    # Conference-wise status breakdown
    print(f"\n🎯 Status Distribution by Conference:")
    print("=" * 70)

    for conf_name, stats in all_stats.items():
        if not stats:
            continue

        print(f"\n📋 {conf_name.upper()}:")
        print("-" * 40)

        # Sort statuses by count (descending)
        sorted_statuses = sorted(
            stats["status_distribution"].items(), key=lambda x: x[1], reverse=True
        )

        for status, count in sorted_statuses:
            percentage = stats["status_percentages"][status]
            print(f"  {status:<18}: {count:>6,} ({percentage:>5.1f}%)")


def main():
    """Main function to analyze all conferences"""
    print("🚀 Starting paper status analysis...")

    # Define file paths
    base_path = Path(__file__).parent / "selected_papers"
    conferences = {
        "icml2025": base_path / "icml2025.json",
        "nips2024": base_path / "nips2024.json",
        "iclr2025": base_path / "iclr2025.json",
        "cvpr2025": base_path / "cvpr2025.json",
        "www2025": base_path / "www2025.json",
    }

    # Check if files exist
    existing_conferences = {}
    for conf_name, file_path in conferences.items():
        if file_path.exists():
            existing_conferences[conf_name] = file_path
        else:
            print(f"⚠️  File not found: {file_path}")

    if not existing_conferences:
        print("❌ No conference files found! Please run paper_prepare.py first.")
        return

    # Analyze each conference
    all_stats = {}
    for conf_name, file_path in existing_conferences.items():
        print(f"\n🔍 Analyzing {conf_name}...")
        stats = analyze_status_distribution(str(file_path))
        all_stats[conf_name] = stats
        print_conference_analysis(conf_name, stats)

    # Generate summary report
    generate_summary_report(all_stats)

    print(f"\n{'='*70}")
    print("✅ Analysis completed successfully!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
