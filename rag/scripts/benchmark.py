from __future__ import annotations

import argparse
import asyncio
import statistics
import time
import httpx


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int((len(values) - 1) * p)))]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/rag/query")
    parser.add_argument("--queries-file", required=True, help="UTF-8 text file: one query per line; at least 100 lines")
    args = parser.parse_args()
    queries = [line.strip() for line in open(args.queries_file, encoding="utf-8") if line.strip()]
    if len(queries) < 100: raise SystemExit("Provide at least 100 queries; benchmark values must be measured.")
    timings = []
    async with httpx.AsyncClient(timeout=30) as client:
        for query in queries[:100]:
            started = time.perf_counter(); response = await client.post(args.url, json={"query": query}); response.raise_for_status()
            timings.append((time.perf_counter()-started)*1000)
    print({"requests": len(timings), "p50_ms": round(percentile(timings,.50),2), "p70_ms": round(percentile(timings,.70),2), "p95_ms": round(percentile(timings,.95),2), "p100_ms": round(max(timings),2)})


if __name__ == "__main__": asyncio.run(main())
