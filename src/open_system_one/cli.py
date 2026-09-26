"""Command line: `open-system-one serve`."""

import argparse
import logging
import os


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="open-system-one")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the Jev-compatible API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--models", default="configs/models.yaml")
    serve.add_argument("--prompts", default="configs/prompts")
    serve.add_argument("--trace-dir", default="runs/traces")
    serve.add_argument("--no-trace", action="store_true", help="don't write request traces")

    ev = sub.add_parser("eval", help="run eval cases against a profile and the Jev baseline")
    ev.add_argument("--cases", default="evals/sanity-v0.jsonl")
    ev.add_argument("--target", required=True, help="profile or alias to evaluate")
    ev.add_argument(
        "--baseline", default="jev-1.13", help="Jev model via OpenRouter; 'none' to skip"
    )
    ev.add_argument("--split", default="tune", choices=["tune", "holdout", "all"])
    ev.add_argument("--models", default="configs/models.yaml")
    ev.add_argument("--prompts", default="configs/prompts")
    ev.add_argument("--out", default="runs/eval")
    ev.add_argument("--refresh-jev", action="store_true", help="ignore the Jev response cache")
    ev.add_argument(
        "--prompt", help="prompt version to use instead of the profile's (name@version)"
    )
    ev.add_argument(
        "--max-wait", type=float, default=600, help="seconds to wait for a cold backend"
    )

    args = parser.parse_args(argv)
    if args.command == "serve":
        _serve(args)
    elif args.command == "eval":
        _eval(args)


def _eval(args: argparse.Namespace) -> None:
    from open_system_one.eval.runner import run_eval

    logging.basicConfig(level=logging.WARNING)
    run_dir = run_eval(
        args.cases,
        args.target,
        split=args.split,
        baseline=None if args.baseline == "none" else args.baseline,
        models_path=args.models,
        prompts_dir=args.prompts,
        out_root=args.out,
        refresh_jev=args.refresh_jev,
        max_wait=args.max_wait,
        prompt_override=args.prompt,
    )
    print(f"\nreport: {run_dir / 'report.md'}")


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from open_system_one.config import load_registry
    from open_system_one.server import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    keys = [k.strip() for k in os.environ.get("OSO_API_KEY", "").split(",") if k.strip()]
    if not keys:
        raise SystemExit("OSO_API_KEY is not set; the server needs at least one bearer key.")
    app = create_app(
        load_registry(args.models),
        args.prompts,
        keys,
        trace_dir=None if args.no_trace else args.trace_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
