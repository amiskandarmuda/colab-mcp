from argparse import Namespace

from colab_mcp.cli import build_server_argv, parse_wrapper_args


def test_parse_wrapper_args_defaults():
    args = parse_wrapper_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.transport == "streamable-http"
    assert args.log is None
    assert args.client_oauth_config is None


def test_build_server_argv_with_oauth_and_log():
    args = Namespace(
        host="0.0.0.0",
        port=9000,
        transport="streamable-http",
        log="/tmp/colab-mcp-logs",
        client_oauth_config="/tmp/oauth.json",
    )

    assert build_server_argv(args) == [
        "--transport",
        "streamable-http",
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "--log",
        "/tmp/colab-mcp-logs",
        "--client-oauth-config",
        "/tmp/oauth.json",
    ]
