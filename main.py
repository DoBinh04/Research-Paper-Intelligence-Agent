"""CLI entry point."""

import argparse
import json

from agent.graph import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Paper Intelligence Agent")
    parser.add_argument("query", nargs="?", default="Compare transformer attention mechanisms")
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    result = run_agent(args.query, session_id=args.session_id)
    print(json.dumps({
        "answer": result.get("final_answer"),
        "latency_ms": result.get("latency_ms"),
        "conflicts": result.get("conflicts"),
        "papers": [p.get("title") for p in result.get("papers_fetched", [])],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
