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

    args = parser.parse_args(argv)
    if args.command == "serve":
        _serve(args)


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
