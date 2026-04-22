"""
CLI entry point.  Run with:  python -m toolkit <command> [options]

Commands
--------
setup    Walk through interactive setup and save a config JSON.
query    Query a model using a saved config.
analyze  Analyze results from a previous query run.
run      setup → query → analyze in one shot.
"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m toolkit",
        description="LLM QA + sentence-relevance evaluation toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- setup ----
    p_setup = sub.add_parser("setup", help="Interactively create a config file")
    p_setup.add_argument("--config", default="toolkit_config.json", help="Path to save config (default: toolkit_config.json)")
    p_setup.add_argument("--input", help="Input CSV path (can also be entered interactively)")

    # ---- query ----
    p_query = sub.add_parser("query", help="Query a model on your dataset")
    p_query.add_argument("--config", required=True, help="Config JSON path")
    p_query.add_argument("--input", help="Override input CSV path from config")
    p_query.add_argument("--output", help="Override output CSV path from config")

    # ---- analyze ----
    p_analyze = sub.add_parser("analyze", help="Analyze query results")
    p_analyze.add_argument("--config", required=True, help="Config JSON path")
    p_analyze.add_argument("--input", help="Input CSV (defaults to config output_path)")
    p_analyze.add_argument("--output-dir", default="toolkit_results", help="Directory for analysis output (default: toolkit_results)")

    # ---- run ----
    p_run = sub.add_parser("run", help="Full pipeline: setup → query → analyze")
    p_run.add_argument("--config", default="toolkit_config.json", help="Config path (created if absent)")
    p_run.add_argument("--output-dir", default="toolkit_results", help="Directory for analysis output")

    args = parser.parse_args()

    if args.command == "setup":
        from .config import interactive_setup
        interactive_setup(args.config, input_path=getattr(args, "input", None))

    elif args.command == "query":
        from .config import ToolkitConfig
        from .query import run_query
        config = ToolkitConfig.load(args.config)
        if getattr(args, "input", None):
            config.input_path = args.input
        if getattr(args, "output", None):
            config.output_path = args.output
        run_query(config)

    elif args.command == "analyze":
        from .config import ToolkitConfig
        from .analyze import run_analysis
        config = ToolkitConfig.load(args.config)
        input_path = getattr(args, "input", None) or config.output_path
        run_analysis(config, input_path, args.output_dir)

    elif args.command == "run":
        from .config import interactive_setup, ToolkitConfig
        from .query import run_query
        from .analyze import run_analysis
        config = interactive_setup(args.config)
        run_query(config)
        run_analysis(config, config.output_path, args.output_dir)


if __name__ == "__main__":
    main()
