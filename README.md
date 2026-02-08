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

## Output Results Explained 📁

After running, you will find the following structured files in the `results/` directory:

### 1. Comprehensive Report (`report_YYYYMMDD_HHMMSS.json`)
A **JSON** format summary report containing all analysis dimensions, structured as follows:
```json
{
  "summary": {
    "total_requests": 100000,
    "unique_visitors": 1520,
    "crawler_traffic_ratio": 0.15,
    "error_rate": 0.02
  },
  "visitors": { ... },  // Detailed visitor analysis
  "content": { ... },   // Detailed content analysis
  "errors": { ... }     // Detailed error analysis
}
```

### 2. Detailed Data Files (`*.csv`)
For in-depth analysis using spreadsheet or database tools, the system generates several **CSV** files:

| Filename | Content Description | Example Columns |
| :--- | :--- | :--- |
| **`top_ips_<timestamp>.csv`** | **Top N Most Frequent Visitor IPs** | `IP, Requests, Country, Is_Crawler` |
| **`top_urls_<timestamp>.csv`** | **Top N Most Accessed URLs/APIs** | `URL, Request_Count, Type (e.g., API, Page, Image)` |
| **`errors_detail_<timestamp>.csv`** | **List of Specific Erroneous Requests** | `IP, URL, Status_Code, Time, User_Agent` |
| **`status_code_summary_<timestamp>.csv`** | **HTTP Status Code Distribution Statistics** | `Status_Group (e.g., 2xx), Count, Percentage` |

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
