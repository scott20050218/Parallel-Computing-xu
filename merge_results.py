#!/usr/bin/env python3
"""Merge all JSON reports in results/ into a single file"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = "results"
OUTPUT_FILE = "merged_report.json"
TOP_N = 50


def merge_reports(reports: list[dict]) -> dict:
    """Merge multiple thread reports into one"""
    total_requests = sum(r["summary"]["total_requests"] for r in reports)
    unique_visitors = sum(r["summary"]["unique_visitors"] for r in reports)  # upper bound
    error_count = sum(r["summary"]["error_count"] for r in reports)
    error_rate = round(error_count / total_requests, 4) if total_requests else 0

    ip_count = defaultdict(int)
    url_count = defaultdict(int)
    level_count = defaultdict(int)
    errors_sample = []

    for r in reports:
        for item in r["visitors"]["top_ips"]:
            ip_count[item["ip"]] += item["count"]
        for item in r["content"]["top_urls"]:
            url_count[item["url"]] += item["count"]
        for k, v in r["errors"]["by_level"].items():
            level_count[k] += v
        errors_sample.extend(r["errors"]["sample"])

    top_ips = sorted(ip_count.items(), key=lambda x: -x[1])[:TOP_N]
    top_urls = sorted(url_count.items(), key=lambda x: -x[1])[:TOP_N]

    return {
        "source_threads": [r["thread_id"] for r in reports],
        "summary": {
            "total_requests": total_requests,
            "unique_visitors": unique_visitors,
            "error_count": error_count,
            "error_rate": error_rate,
        },
        "visitors": {"top_ips": [{"ip": k, "count": v} for k, v in top_ips]},
        "content": {"top_urls": [{"url": k, "count": v} for k, v in top_urls]},
        "errors": {
            "by_level": dict(level_count),
            "sample": errors_sample[:100],
        },
    }


def main():
    results_path = Path(RESULTS_DIR)
    if not results_path.exists():
        print(f"Directory {RESULTS_DIR} not found")
        return

    json_files = sorted(results_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {RESULTS_DIR}")
        return

    reports = []
    for f in json_files:
        if f.name == OUTPUT_FILE:
            continue  # Skip already merged file
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if "thread_id" in data:
            reports.append(data)

    merged = merge_reports(reports)
    output_path = results_path / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(reports)} reports into {output_path}")


if __name__ == "__main__":
    main()
