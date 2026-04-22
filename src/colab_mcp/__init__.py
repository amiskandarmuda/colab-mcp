# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import base64
import datetime
import json
import logging
from pathlib import Path
import tempfile
import sys
from urllib.parse import urlparse, urlunparse

from fastmcp import FastMCP
from fastmcp.utilities import logging as fastmcp_logger

from colab_mcp.session import ColabSessionProxy, NOT_CONNECTED_MSG
from colab_mcp.websocket_server import COLAB, SCRATCH_PATH


mcp = FastMCP(name="ColabMCP")

# These will be set during main_async() startup
_proxy_client = None
_session_mcp = None
_colab_client = None  # For runtime API (assign/unassign GPU)
_connection_base_url = f"{COLAB}{SCRATCH_PATH}"


def _normalize_connection_base(target_url: str) -> str:
    """Validate and normalize a target Colab notebook URL for MCP attachment."""
    parsed = urlparse(target_url)
    allowed_hosts = {"colab.research.google.com", "colab.google.com"}
    if parsed.scheme != "https" or parsed.netloc not in allowed_hosts:
        raise ValueError(
            "target_url must be an https://colab.research.google.com or https://colab.google.com notebook URL"
        )
    if not parsed.path:
        raise ValueError("target_url must include a Colab notebook path")
    # Preserve path and query, but replace any existing fragment with the MCP hash.
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", parsed.query, ""))


def _current_connection_url(target_url: str | None = None, persist: bool = False) -> str | None:
    """Return the current browser connection URL for the live proxy server."""
    global _connection_base_url
    if _proxy_client is None:
        return None
    if target_url:
        base_url = _normalize_connection_base(target_url)
        if persist:
            _connection_base_url = base_url
    else:
        base_url = _connection_base_url
    return (
        f"{base_url}"
        f"#mcpProxyToken={_proxy_client.wss.token}&mcpProxyPort={_proxy_client.wss.port}"
    )


async def _connected_tool_names() -> list[str]:
    """Return browser-side notebook tool names if connected."""
    if _proxy_client is None or not _proxy_client.is_connected():
        return []
    return await _proxy_client.await_tools_ready()


def _format_connection_status(connected: bool, url: str | None, tool_names: list[str]) -> str:
    """Render a compact, user-facing connection status message."""
    tools_text = ", ".join(tool_names) if tool_names else "none"
    url_text = url or "unavailable"
    if connected:
        return (
            "Connected to Google Colab via MCP.\n"
            f"Notebook tools: {tools_text}\n"
            f"Connection URL: {url_text}"
        )
    return (
        "Not connected to a Google Colab browser session.\n"
        f"Connection URL: {url_text}"
    )


async def _forward_or_stub(tool_name: str, arguments: dict) -> str:
    """Forward a tool call to the browser if connected, otherwise return stub message."""
    result = await _call_proxy_tool(tool_name, arguments)
    if isinstance(result, str):
        return result
    text_parts = _result_text_parts(result)
    if text_parts:
        return "\n".join(text_parts)
    return str(result)


async def _call_proxy_tool(tool_name: str, arguments: dict):
    """Forward a tool call to the browser if connected, otherwise return stub message."""
    if _proxy_client is not None and _proxy_client.is_connected():
        try:
            return await _proxy_client.proxy_mcp_client.call_tool(tool_name, arguments)
        except Exception as e:
            return f"Error calling {tool_name}: {e}. Try calling open_colab_browser_connection to reconnect."
    return NOT_CONNECTED_MSG


def _result_text_parts(result) -> list[str]:
    """Extract plain text parts from a tool result object."""
    if result is None:
        return []
    if isinstance(result, str):
        return [result]
    if hasattr(result, "content"):
        return [c.text for c in result.content if hasattr(c, "text")]
    return [str(result)]


def _result_json_payload(result):
    """Best-effort extraction of JSON payload from a tool result."""
    if result is None:
        return None
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    for text in _result_text_parts(result):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    return None


def _png_outputs_from_payload(payload: object) -> list[tuple[int, str]]:
    """Return (output_index, base64_png) tuples found in a run_code_cell-style payload."""
    if not isinstance(payload, dict):
        return []
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        return []
    png_outputs: list[tuple[int, str]] = []
    for index, output in enumerate(outputs):
        if not isinstance(output, dict):
            continue
        data = output.get("data")
        if not isinstance(data, dict):
            continue
        png_data = data.get("image/png")
        if isinstance(png_data, list):
            png_data = "".join(png_data)
        if isinstance(png_data, str) and png_data:
            png_outputs.append((index, png_data))
    return png_outputs


@mcp.tool()
async def get_colab_connection_url(target_url: str = "") -> str:
    """Return the exact Colab notebook URL for the current live proxy session. Optionally provide a specific Colab notebook URL (including Drive notebooks) to target manually."""
    url = _current_connection_url(target_url or None, persist=bool(target_url))
    if url is None:
        return "Server not initialized. Please wait and try again."
    return url


@mcp.tool()
async def get_colab_connection_status() -> str:
    """Return whether the browser is attached to the current Colab MCP proxy session, plus the current manual connection URL."""
    url = _current_connection_url()
    tool_names = await _connected_tool_names()
    connected = _proxy_client is not None and _proxy_client.is_connected()
    return _format_connection_status(connected, url, tool_names)


@mcp.tool()
async def wait_for_colab_browser_connection(timeout_seconds: int = 60) -> str:
    """Wait for a browser to attach to the current Colab MCP proxy session. Use after opening the manual URL yourself."""
    if _proxy_client is None:
        return "Server not initialized. Please wait and try again."

    deadline = asyncio.get_running_loop().time() + max(timeout_seconds, 0)
    while asyncio.get_running_loop().time() < deadline:
        if _proxy_client.is_connected():
            tool_names = await _connected_tool_names()
            return _format_connection_status(True, _current_connection_url(), tool_names)
        await asyncio.sleep(0.5)

    return _format_connection_status(False, _current_connection_url(), [])


@mcp.tool()
async def open_colab_browser_connection(target_url: str = "") -> str:
    """Prepare a connection to a Google Colab browser session and return the exact manual URL. Optionally provide a specific Colab notebook URL (including Drive notebooks). This tool does not open a browser for you."""
    if _proxy_client is not None and _proxy_client.is_connected():
        tool_names = await _connected_tool_names()
        return _format_connection_status(True, _current_connection_url(), tool_names)

    if _proxy_client is None:
        return "Server not initialized. Please wait and try again."

    url = _current_connection_url(target_url or None, persist=bool(target_url))
    if url is None:
        return "Server not initialized. Please wait and try again."

    # Wait for browser to connect
    await _proxy_client.await_proxy_connection()

    if _proxy_client.is_connected():
        tool_names = await _connected_tool_names()
        return _format_connection_status(True, url, tool_names)
    else:
        return (
            "Connection timed out. Please open the manual connection URL in your browser and try again.\n"
            f"Manual connection URL: {url}"
        )


@mcp.tool()
async def add_code_cell(code: str = "", cellIndex: int = 0, language: str = "python") -> str:
    """Add a new code cell to the Colab notebook. Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub("add_code_cell", {"code": code, "cellIndex": cellIndex, "language": language})


@mcp.tool()
async def add_text_cell(content: str = "", cellIndex: int = -1) -> str:
    """Add a new text/markdown cell to the Colab notebook. Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub("add_text_cell", {"content": content, "cellIndex": cellIndex})


@mcp.tool()
async def execute_cell(cellId: str = "", cellIndex: int = 0) -> str:
    """Execute a cell in the Colab notebook. Pass cellId (from add_code_cell result) or cellIndex. Requires an active browser connection via open_colab_browser_connection."""
    args = {}
    if cellId:
        args["cellId"] = cellId
    else:
        args["cellId"] = str(cellIndex)
    return await _forward_or_stub("run_code_cell", args)


@mcp.tool()
async def inspect_cell_images(
    cellId: str = "",
    cellIndex: int = 0,
    rerun: bool = True,
) -> str:
    """Run a notebook cell through MCP and export any image/png outputs to local files for inspection inside Codex."""
    if cellId:
        args = {"cellId": cellId}
    else:
        args = {"cellId": str(cellIndex)}

    tool_name = "run_code_cell" if rerun else "run_code_cell"
    result = await _call_proxy_tool(tool_name, args)
    if isinstance(result, str):
        return result

    payload = _result_json_payload(result)
    png_outputs = _png_outputs_from_payload(payload)
    if not png_outputs:
        text_parts = _result_text_parts(result)
        fallback = "\n".join(text_parts).strip()
        if fallback:
            return f"No image/png outputs found.\nTool output:\n{fallback}"
        return "No image/png outputs found."

    export_dir = Path(tempfile.mkdtemp(prefix="colab-mcp-images-"))
    exported_images = []
    for image_number, (output_index, png_base64) in enumerate(png_outputs):
        image_path = export_dir / f"cell-{cellId or cellIndex}-output-{output_index}-{image_number}.png"
        image_path.write_bytes(base64.b64decode(png_base64))
        exported_images.append(
            {
                "cellId": cellId or str(cellIndex),
                "outputIndex": output_index,
                "path": str(image_path),
            }
        )

    return json.dumps(
        {
            "cellId": cellId or str(cellIndex),
            "exportDir": str(export_dir),
            "images": exported_images,
        },
        indent=2,
    )


@mcp.tool()
async def update_cell(cellId: str = "", content: str = "") -> str:
    """Update the contents of an existing cell in the Colab notebook. Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub("update_cell", {"cellId": cellId, "content": content})


@mcp.tool()
async def get_cells() -> str:
    """Return the current notebook cells. Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub("get_cells", {})


@mcp.tool()
async def delete_cell(cellId: str = "", cellIndex: int = -1) -> str:
    """Delete a cell in the Colab notebook by ID or index. Requires an active browser connection via open_colab_browser_connection."""
    args = {}
    if cellId:
        args["cellId"] = cellId
    elif cellIndex >= 0:
        args["cellIndex"] = cellIndex
    else:
        return "Please provide either cellId or a non-negative cellIndex."
    return await _forward_or_stub("delete_cell", args)


@mcp.tool()
async def move_cell(cellId: str = "", cellIndex: int = -1, newIndex: int = 0) -> str:
    """Move a cell in the Colab notebook to a new index. Requires an active browser connection via open_colab_browser_connection."""
    args = {"newIndex": newIndex}
    if cellId:
        args["cellId"] = cellId
    elif cellIndex >= 0:
        args["cellIndex"] = cellIndex
    else:
        return "Please provide either cellId or a non-negative cellIndex."
    return await _forward_or_stub("move_cell", args)


@mcp.tool()
async def change_runtime(accelerator: str = "T4") -> str:
    """Change the Colab runtime to use a specific GPU accelerator. Valid values: NONE, T4, L4, A100. Requires OAuth setup (first time opens browser for consent)."""
    if _colab_client is None:
        return "Runtime API not initialized. Start with --client-oauth-config flag pointing to your OAuth client secrets JSON."
    try:
        from colab_mcp.client import Accelerator, Variant
        import uuid

        acc = Accelerator(accelerator)
        variant = Variant.GPU if acc != Accelerator.NONE else Variant.DEFAULT
        notebook_hash = str(uuid.uuid4())

        # Unassign current VM if any
        try:
            assignments = _colab_client.list_assignments()
            for a in assignments:
                _colab_client.unassign(a.endpoint)
        except Exception:
            pass

        # Assign new VM
        result = _colab_client.assign(notebook_hash, variant, acc)
        return f"Runtime changed to {accelerator}. Endpoint: {result.endpoint}. Use open_colab_browser_connection to connect to the new runtime."
    except Exception as e:
        return f"Failed to change runtime: {e}"


def init_logger(logdir):
    log_filename = datetime.datetime.now().strftime(
        f"{logdir}/colab-mcp.%Y-%m-%d_%H-%M-%S.log"
    )
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_filename,
        level=logging.INFO,
    )
    fastmcp_logger.get_logger("colab-mcp").info("logging to %s" % log_filename)


def parse_args(v):
    parser = argparse.ArgumentParser(
        description="ColabMCP is an MCP server that lets you interact with Colab."
    )
    parser.add_argument(
        "-l",
        "--log",
        help="if set, use this directory as a location for logfiles (if unset, will log to %s/colab-mcp-logs/)"
        % tempfile.gettempdir(),
        action="store",
        default=tempfile.mkdtemp(prefix="colab-mcp-logs-"),
    )
    parser.add_argument(
        "-p",
        "--enable-proxy",
        help="if set, enable the runtime proxy (enabled by default).",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--client-oauth-config",
        help="Path to OAuth client secrets JSON for Colab API access (enables change_runtime tool).",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--transport",
        help="FastMCP transport to use. Use streamable-http to expose one stable local MCP endpoint.",
        choices=["stdio", "http", "sse", "streamable-http"],
        default="stdio",
    )
    parser.add_argument(
        "--host",
        help="Host to bind when using an HTTP-based MCP transport.",
        action="store",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        help="Port to bind when using an HTTP-based MCP transport.",
        action="store",
        type=int,
        default=8765,
    )
    return parser.parse_args(v)


async def main_async(argv: list[str] | None = None):
    global _proxy_client, _session_mcp, _colab_client
    args = parse_args(sys.argv[1:] if argv is None else argv)
    init_logger(args.log)

    if args.enable_proxy:
        logging.info("enabling session proxy tools")
        _session_mcp = ColabSessionProxy()
        await _session_mcp.start_proxy_server()
        _proxy_client = _session_mcp.proxy_client

    if args.client_oauth_config:
        try:
            from colab_mcp.auth import get_credentials
            from colab_mcp.client import ColabClient, Prod
            logging.info("initializing Colab API client with OAuth")
            session = get_credentials(args.client_oauth_config)
            _colab_client = ColabClient(Prod(), session)
            logging.info("Colab API client ready")
        except Exception as e:
            logging.warning(f"Failed to initialize Colab API client: {e}")

    try:
        run_kwargs = {}
        if args.transport != "stdio":
            run_kwargs["host"] = args.host
            run_kwargs["port"] = args.port
        await mcp.run_async(transport=args.transport, **run_kwargs)

    finally:
        if args.enable_proxy and _session_mcp:
            await _session_mcp.cleanup()


def main() -> None:
    asyncio.run(main_async())
