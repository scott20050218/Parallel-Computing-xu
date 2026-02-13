#!/usr/bin/env python3
"""Simple multi-threaded Apache log analyzer"""

import re
import json
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
DATA_DIR = "data"
WORKER_THREADS = 4
CHUNK_SIZE = 5000
TOP_N = 20
OUTPUT_DIR = "results"


def parse_log_line(line: str) -> dict | None:
    """Parse Apache error log line"""
    # Format: [timestamp] [level] [client IP] message
    match = re.match(r'\[.*?\] \[(\w+)\] (?:\[client ([\d.]+)\])?(.*)', line.strip())
    if not match:
        return None
    level, ip, message = match.groups()
    ip = ip or "unknown"

    # Extract URL/path from message
    url = None
    for pattern in [
        r'\[uri "([^"]+)"\]',
        r'File does not exist: ([^\s]+)',
        r'Directory index forbidden by rule: ([^\s]+)',
        r'\[hostname "([^"]+)"\]',
    ]:
        m = re.search(pattern, message)
        if m:
            url = m.group(1).strip()
            break

    return {
        "ip": ip,
        "level": level,
        "url": url or message[:80] if message else "",
        "message": message or "",
    }


def process_chunk(lines: list[str]) -> dict:
    """Process a chunk of logs, return statistics"""
    ip_count = defaultdict(int)
    url_count = defaultdict(int)
    level_count = defaultdict(int)
    errors = []

    for line in lines:
        parsed = parse_log_line(line)
        if not parsed:
            continue
        ip_count[parsed["ip"]] += 1
        if parsed["url"]:
            url_count[parsed["url"]] += 1
        level_count[parsed["level"]] += 1
        if parsed["level"] == "error":
            errors.append({"ip": parsed["ip"], "url": parsed["url"], "msg": parsed["message"][:100]})

    return {
        "ip_count": dict(ip_count),
        "url_count": dict(url_count),
        "level_count": dict(level_count),
        "errors": errors[:100],
    }


def process_and_write(thread_id: int, chunks: list[list[str]], output_dir: Path, ts: str) -> str:
    """Each thread processes its assigned chunks and writes to a separate file"""
    merged = {"ip_count": defaultdict(int), "url_count": defaultdict(int), "level_count": defaultdict(int), "errors": []}
    for chunk in chunks:
        r = process_chunk(chunk)
        for k, v in r["ip_count"].items():
            merged["ip_count"][k] += v
        for k, v in r["url_count"].items():
            merged["url_count"][k] += v
        for k, v in r["level_count"].items():
            merged["level_count"][k] += v
        merged["errors"].extend(r["errors"])

    m = merged
    total = sum(m["ip_count"].values())
    top_ips = sorted(m["ip_count"].items(), key=lambda x: -x[1])[:TOP_N]
    top_urls = sorted(m["url_count"].items(), key=lambda x: -x[1])[:TOP_N]
    report = {
        "thread_id": thread_id,
        "summary": {"total_requests": total, "unique_visitors": len(m["ip_count"]), "error_count": m["level_count"].get("error", 0), "error_rate": round(m["level_count"].get("error", 0) / total, 4) if total else 0},
        "visitors": {"top_ips": [{"ip": k, "count": v} for k, v in top_ips]},
        "content": {"top_urls": [{"url": k, "count": v} for k, v in top_urls]},
        "errors": {"by_level": dict(m["level_count"]), "sample": m["errors"][:50]},
    }
    path = output_dir / f"report_{ts}_thread_{thread_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return str(path)


def main():
    # Read all .log files in data directory
    data_path = Path(DATA_DIR)
    log_files = list(data_path.glob("*.log")) if data_path.exists() else []

    if not log_files:
        print(f"No .log files found in {DATA_DIR}")
        return

    # Read all log lines
    all_lines = []
    for f in log_files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            all_lines.extend(fp.readlines())

    total_lines = len(all_lines)
    print(f"Read {total_lines} log lines, processing with {WORKER_THREADS} threads in parallel")

    # Split into chunks
    chunks = [all_lines[i : i + CHUNK_SIZE] for i in range(0, total_lines, CHUNK_SIZE)]

    # Assign chunks to threads: thread i processes chunks[i], chunks[i+N], chunks[i+2N], ...
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    thread_chunks = [[] for _ in range(WORKER_THREADS)]
    for i, chunk in enumerate(chunks):
        thread_chunks[i % WORKER_THREADS].append(chunk)

    # Each thread processes its assigned chunks and writes to a separate file
    output_paths = []
    with ThreadPoolExecutor(max_workers=WORKER_THREADS) as executor:
        futures = [
            executor.submit(process_and_write, tid, thread_chunks[tid], output_dir, ts)
            for tid in range(WORKER_THREADS)
        ]
        for future in as_completed(futures):
            output_paths.append(future.result())

    print(f"Done! {WORKER_THREADS} threads produced {WORKER_THREADS} report files:")
    for p in sorted(output_paths):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
