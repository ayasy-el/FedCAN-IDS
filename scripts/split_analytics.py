"""Analyze split artifacts without making assumptions about split strategy.

Examples:
    python scripts/split_analytics.py data/processed/current/*.parquet
    python scripts/split_analytics.py --json reports/split_analytics.json train.parquet val.parquet test.parquet
"""

import argparse
import json
from pathlib import Path

import polars as pl


def _counts(df, column):
    result = df.group_by(column, maintain_order=True).len().sort(column)
    return {str(row[column]): int(row["len"]) for row in result.to_dicts()}


def _percentages(counts):
    total = sum(counts.values())
    return {
        label: round(count / total * 100, 4) if total else 0.0
        for label, count in counts.items()
    }


def _list_length_counts(df, column):
    result = (
        df.select(pl.col(column).list.len().alias("length"))
        .group_by("length")
        .len()
        .sort("length")
    )
    return {str(row["length"]): int(row["len"]) for row in result.to_dicts()}


def _window_attack_summary(df):
    if "frame_labels" not in df.columns:
        return None
    attack_labels = (
        df.select(
            pl.col("frame_labels")
            .list.eval(
                pl.element().filter((pl.element() != 0) & (pl.element() != 5))
            )
            .list.unique()
            .alias("attack_labels")
        )["attack_labels"]
        .to_list()
    )
    multiple_attack = sum(len(labels) > 1 for labels in attack_labels)
    benign_mixed = 0
    for labels in df["frame_labels"].to_list():
        if 0 in labels and 5 in labels:
            benign_mixed += 1
    return {
        "windows_with_multiple_attack_labels": multiple_attack,
        "windows_with_both_benign_labels_0_and_5": benign_mixed,
    }


def analyze(path):
    path = Path(path)
    df = pl.read_parquet(path)
    label_column = "label" if "label" in df.columns else "Class" if "Class" in df.columns else None
    if label_column is None:
        raise ValueError(f"{path} has no 'label' or 'Class' column")

    unit = (
        "window"
        if {"window_id", "window_size", "label"}.issubset(df.columns)
        or "frames" in df.columns
        or "frame_labels" in df.columns
        else "frame"
    )
    labels = _counts(df, label_column)
    report = {
        "path": str(path),
        "unit": unit,
        "rows": len(df),
        "columns": df.columns,
        "label_column": label_column,
        "class_counts": labels,
        "class_percentages": _percentages(labels),
    }

    if "session_id" in df.columns:
        sessions = _counts(df, "session_id")
        report["session_count"] = len(sessions)
        report["session_row_counts"] = sessions
    else:
        report["session_count"] = None

    if unit == "window":
        if "frames" in df.columns:
            report["window_length_counts"] = _list_length_counts(df, "frames")
        elif "window_size" in df.columns:
            report["window_length_counts"] = _counts(df, "window_size")
        report["attack_summary"] = _window_attack_summary(df)
        identifier = "window_id" if "window_id" in df.columns else "start_row_id" if "start_row_id" in df.columns else None
    else:
        identifier = "row_id" if "row_id" in df.columns else None

    if identifier is not None:
        duplicate_count = len(df) - df.select(identifier).unique().height
        report["identifier"] = identifier
        report["duplicate_identifier_count"] = duplicate_count
    else:
        report["identifier"] = None
        report["duplicate_identifier_count"] = None
    return report


def _print_report(report):
    print(f"\n=== {report['path']} ===")
    print(f"unit: {report['unit']}")
    print(f"rows: {report['rows']:,}")
    print(f"label column: {report['label_column']}")
    print("class distribution:")
    for label, count in report["class_counts"].items():
        percentage = report["class_percentages"][label]
        print(f"  {label:>8}: {count:>12,} ({percentage:>8.4f}%)")
    if report["session_count"] is not None:
        print(f"sessions: {report['session_count']:,}")
    if "window_length_counts" in report:
        print(f"window lengths: {report['window_length_counts']}")
        print(f"attack summary: {report['attack_summary']}")
    if report["identifier"]:
        print(
            f"duplicate {report['identifier']}: "
            f"{report['duplicate_identifier_count']:,}"
        )


def _cross_split_report(reports):
    result = {}
    session_sets = {}
    identifier_sets = {}
    for report in reports:
        df = pl.read_parquet(report["path"])
        name = Path(report["path"]).stem
        if "session_id" in df.columns:
            session_sets[name] = set(df["session_id"].unique().to_list())
        identifier = report.get("identifier")
        if identifier:
            identifier_sets[name] = set(df[identifier].to_list())

    result["session_overlaps"] = {
        f"{left}__{right}": sorted(session_sets[left] & session_sets[right])
        for index, left in enumerate(session_sets)
        for right in list(session_sets)[index + 1:]
        if session_sets[left] & session_sets[right]
    }
    result["identifier_overlaps"] = {
        f"{left}__{right}": len(identifier_sets[left] & identifier_sets[right])
        for index, left in enumerate(identifier_sets)
        for right in list(identifier_sets)[index + 1:]
        if identifier_sets[left] & identifier_sets[right]
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Split Parquet files to analyze")
    parser.add_argument("--json", dest="json_path", help="Optional JSON report path")
    args = parser.parse_args()

    reports = [analyze(path) for path in args.paths]
    for report in reports:
        _print_report(report)
    if len(reports) > 1:
        cross_split = _cross_split_report(reports)
        print("\n=== cross-split checks ===")
        print(f"session overlaps: {cross_split['session_overlaps'] or 'none'}")
        print(f"identifier overlaps: {cross_split['identifier_overlaps'] or 'none'}")
        for report in reports:
            report["cross_split"] = cross_split

    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reports, indent=2))
        print(f"JSON report: {output}")


if __name__ == "__main__":
    main()
