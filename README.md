# Apache Log Parallel Analyzer

## Project Overview

This project is a high-performance Apache web server log analysis tool built on **Python parallel computing technology**. It is designed to rapidly process massive log files (from MB to GB scale) and extract key business and operational insights.

### Core Analysis Objectives

The project focuses on three main analytical goals:

1.  ** Visitor Analysis**: Identify frequent visitors, analyze their geographical distribution, and detect crawler or proxy traffic.
2.  ** Popular Content Analysis**: Discover the most frequently accessed URLs, API endpoints, and the distribution of various file resources.
3.  ** Error Monitoring**: Monitor HTTP status code distributions, quickly pinpoint client and server errors, and track specific erroneous requests.

### Technical Core: Thread Pool-Based Parallel Architecture

The project employs a **Producer-Consumer model** with a **`concurrent.futures.ThreadPoolExecutor`** thread pool to create an efficient parallel processing architecture, significantly speeding up I/O-intensive log processing tasks.

- **Parallel Reading & Parsing**: Multiple worker threads process different chunks of log data simultaneously.
- **Efficient Resource Utilization**: Managed by a thread pool to avoid the overhead of frequent thread creation/destruction.
- **Result Aggregation**: Each thread performs preliminary counting independently, with the main thread efficiently merging results for a final global analysis.

## Features

- **High-Performance Parallel Processing**: Leverages multi-core CPUs, achieving processing speeds several times faster than single-threaded approaches.
- **Structured Outputs**: Analysis results are exported in **JSON** and **CSV** formats for easy integration and further analysis.

## Output Results Explained

After running `log_analyzer.py`, you will find the following structured files in the `results/` directory:

### 1. Per-Thread Reports (`report_YYYYMMDD_HHMMSS_thread_N.json`)

Each worker thread writes its own JSON report. The number of files equals the number of threads (e.g., 4 threads → 4 files). Each report has the same structure:

```json
{
  "thread_id": 0,
  "summary": {
    "total_requests": 14111,
    "unique_visitors": 1507,
    "error_count": 10335,
    "error_rate": 0.7324
  },
  "visitors": {
    "top_ips": [{"ip": "192.168.1.1", "count": 100}, ...]
  },
  "content": {
    "top_urls": [{"url": "/path", "count": 50}, ...]
  },
  "errors": {
    "by_level": {"error": 1000, "notice": 500},
    "sample": [{"ip": "...", "url": "...", "msg": "..."}, ...]
  }
}
```

### 2. Merged Report (`merged_report.json`)

Run `merge_results.py` to combine all per-thread reports into a single global report:

```json
{
  "source_threads": [0, 1, 2, 3],
  "summary": {
    "total_requests": 52004,
    "unique_visitors": 5124,
    "error_count": 38081,
    "error_rate": 0.7323
  },
  "visitors": { "top_ips": [...] },
  "content": { "top_urls": [...] },
  "errors": { "by_level": {...}, "sample": [...] }
}
```

| Field                         | Description                                  |
| :---------------------------- | :------------------------------------------- |
| **`summary.total_requests`**  | Total log entries processed                  |
| **`summary.unique_visitors`** | Count of distinct IP addresses               |
| **`summary.error_count`**     | Number of error-level log entries            |
| **`summary.error_rate`**      | Fraction of entries marked as error          |
| **`visitors.top_ips`**        | Top N most frequent visitor IPs              |
| **`content.top_urls`**        | Top N most referenced URLs/paths             |
| **`errors.by_level`**         | Log level distribution (error, notice, etc.) |
| **`errors.sample`**           | Sample error entries for inspection          |

## Configuration & Customization

You can adjust the analysis behavior by modifying the `config.py` file:

```python
# Parallel Processing Settings
WORKER_THREADS = 8  # Number of worker threads in the pool. Typically 1-2x CPU core count.
CHUNK_SIZE = 10000  # Number of log lines processed per thread chunk.

# Analysis Options
TOP_N = 50  # Number of items displayed in rankings (e.g., Top IPs, Top URLs)
CRAWLER_USER_AGENT_KEYWORDS = ['bot', 'crawler', 'spider']  # Keywords to identify crawlers
SLOW_REQUEST_THRESHOLD_MS = 5000  # Threshold (ms) to define a slow request

# Path Configuration
GEOIP_DATABASE_PATH = './geolite/GeoLite2-City.mmdb'
OUTPUT_DIR = './results'
```
