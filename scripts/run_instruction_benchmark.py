"""Evaluate instruction interpreters on data/instruction_benchmark.json.

Example:
    python scripts/run_instruction_benchmark.py            # deterministic parser
    python scripts/run_instruction_benchmark.py --with-llm # + LLM configured in .env
"""

from __future__ import annotations

import argparse

import pandas as pd

from energypilot.agent.interpretation import interpret_with_llm
from energypilot.agent.llm import try_create_llm_client
from energypilot.config import load_config
from energypilot.evaluation.instruction_benchmark import evaluate_interpreter, load_benchmark, summarize_benchmark
from energypilot.instructions import parse_instruction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--csv", default=None, help="optional path for per-item results")
    args = parser.parse_args()
    config = load_config()
    items = load_benchmark()
    frames = [evaluate_interpreter("deterministic_rules", parse_instruction, items, config)]
    if args.with_llm:
        llm, settings, error = try_create_llm_client(config)
        if llm is None:
            print(f"[warning] no LLM available ({error or 'LLM_PROVIDER=none'}); only the deterministic parser is evaluated")
        else:
            def interpreter(text, issued, cfg):
                return interpret_with_llm(llm, text, issued, cfg, max_tokens=settings.max_tokens, timeout_s=settings.timeout_s)[0]

            frames.append(evaluate_interpreter(f"llm:{llm.provider}/{llm.model}", interpreter, items, config))
    df = pd.concat(frames, ignore_index=True)
    pd.set_option("display.width", 200)
    print(summarize_benchmark(df).round(3).to_string(index=False))
    misses = df[~df.exact_correct][["interpreter", "id", "expected_status", "status", "unsafe_acceptance", "text"]]
    if len(misses):
        print("\nNot interpreted as labelled:")
        print(misses.to_string(index=False, max_colwidth=80))
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
