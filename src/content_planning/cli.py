from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .graph import ContentPlanningAgent


def main():
    parser = argparse.ArgumentParser(description="Content Planning Agent")
    parser.add_argument("input", help="JSON input file")
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    parser.add_argument("--runtime-dir", default="runtime")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inject-failure", metavar="TOOL:COUNT", help="test-only deterministic fault injection")
    args = parser.parse_args()
    load_dotenv()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    agent = ContentPlanningAgent(args.runtime_dir, args.mode)
    try:
        faults = None
        if args.inject_failure:
            name, count = args.inject_failure.split(":", 1)
            faults = {name: int(count)}
        print(json.dumps(agent.run(payload, resume=args.resume, fault_injection=faults), ensure_ascii=False, indent=2, default=str))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
