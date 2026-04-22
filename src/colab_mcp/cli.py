import argparse
import asyncio
import sys

from colab_mcp import main_async


def parse_wrapper_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Colab MCP with HTTP-friendly defaults for MCP clients that connect to a persistent endpoint."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the HTTP MCP endpoint to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind the HTTP MCP endpoint to.",
    )
    parser.add_argument(
        "--transport",
        choices=["http", "sse", "streamable-http"],
        default="streamable-http",
        help="HTTP-based FastMCP transport to expose.",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Optional log directory to forward to the underlying server.",
    )
    parser.add_argument(
        "--client-oauth-config",
        default=None,
        help="Optional path to the Google OAuth client JSON that enables change_runtime.",
    )
    return parser.parse_args(argv)


def build_server_argv(args: argparse.Namespace) -> list[str]:
    server_argv = [
        "--transport",
        args.transport,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.log:
        server_argv.extend(["--log", args.log])
    if args.client_oauth_config:
        server_argv.extend(["--client-oauth-config", args.client_oauth_config])
    return server_argv


def main() -> None:
    args = parse_wrapper_args(sys.argv[1:])
    print(f"Colab MCP HTTP endpoint: http://{args.host}:{args.port}/mcp")
    asyncio.run(main_async(build_server_argv(args)))
