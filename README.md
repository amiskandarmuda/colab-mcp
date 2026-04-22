# Colab MCP (Enhanced Fork)

An MCP server for controlling Google Colab from AI coding agents. This fork fixes the upstream tool-discovery problem, restores programmatic runtime changes, and adds a small HTTP wrapper so you can run it as a stable local MCP endpoint.

## What This Fork Changes

- Pre-registers the notebook tools so they show up immediately in MCP clients that do not handle dynamic tool refresh well.
- Restores `change_runtime` with Google OAuth so an agent can switch Colab runtimes programmatically.
- Uses a manual Colab connection URL instead of trying to open a browser from the server process.
- Adds extra notebook helpers: `get_cells`, `delete_cell`, `move_cell`, and `inspect_cell_images`.
- Adds `colab-mcp-http`, a wrapper CLI that exposes the server on `http://127.0.0.1:<port>/mcp` by default.

## Available Tools

Connection and status:

- `get_colab_connection_url`
- `get_colab_connection_status`
- `wait_for_colab_browser_connection`
- `open_colab_browser_connection`

Notebook control:

- `add_code_cell`
- `add_text_cell`
- `execute_cell`
- `inspect_cell_images`
- `update_cell`
- `get_cells`
- `delete_cell`
- `move_cell`

Runtime control:

- `change_runtime`

## Install

```bash
git clone https://github.com/SebastianGilPinzon/colab-mcp.git
cd colab-mcp
uv sync
```

## Launch Options

### Option 1: Stdio MCP

Use this when your MCP client starts local commands itself.

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/colab-mcp", "colab-mcp"],
      "timeout": 30000
    }
  }
}
```

If you want runtime switching too, add your OAuth JSON:

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/colab-mcp",
        "colab-mcp",
        "--client-oauth-config",
        "/path/to/colab-oauth.json"
      ],
      "timeout": 30000
    }
  }
}
```

### Option 2: Wrapped HTTP MCP

Use this when your MCP client connects to a persistent local URL instead of launching a stdio process.

Start the wrapper:

```bash
uv run colab-mcp-http
```

Or with OAuth enabled:

```bash
uv run colab-mcp-http --client-oauth-config /path/to/colab-oauth.json
```

Default endpoint:

```text
http://127.0.0.1:8765/mcp
```

You can change the bind address if needed:

```bash
uv run colab-mcp-http --host 0.0.0.0 --port 9000
```

## How To Use It

### Manual Colab Connection Flow

1. Call `get_colab_connection_url()` or `open_colab_browser_connection(target_url="https://colab.research.google.com/drive/<notebook-id>")`.
2. Open the returned URL in your browser.
3. Call `wait_for_colab_browser_connection()` if the browser has not attached yet.
4. Use notebook tools like `add_code_cell`, `execute_cell`, `get_cells`, or `update_cell`.

`open_colab_browser_connection` keeps the upstream tool name for compatibility, but in this fork it returns the manual URL instead of opening a browser for you.

### Example Agent Flow

```text
Agent: get_colab_connection_url(target_url="https://colab.research.google.com/drive/abc123")
> https://colab.research.google.com/drive/abc123#mcpProxyToken=...&mcpProxyPort=...

Agent: wait_for_colab_browser_connection(timeout_seconds=60)
> Connected to Google Colab via MCP.

Agent: add_code_cell(code="import torch; print(torch.cuda.is_available())")
Agent: execute_cell(cellIndex=0)
```

### Inspecting Image Outputs

If a Colab cell renders plots or other PNG output, call:

```text
inspect_cell_images(cellIndex=0)
```

The tool exports images to a local temp directory and returns the file paths so an agent can inspect them directly.

## OAuth Setup For `change_runtime`

You need a Google Cloud OAuth desktop client once per machine:

1. Create or choose a Google Cloud project.
2. Configure the OAuth consent screen.
3. Add your Google account as a test user.
4. Create an OAuth client ID of type `Desktop app`.
5. Download the client JSON and pass it with `--client-oauth-config`.

The token is cached at `~/.colab-mcp-auth-token.json`.

Example:

```text
Agent: change_runtime(accelerator="T4")
> Runtime changed to T4. Endpoint: gpu-t4-s-xxx.
```

Valid accelerators:

- `NONE`
- `T4`
- `L4`
- `A100`

## Development

Run tests:

```bash
uv run pytest
```

Run the wrapped HTTP endpoint locally:

```bash
uv run colab-mcp-http --port 8765
```

## Troubleshooting

- If notebook tools do not work, confirm the browser actually opened the returned Colab URL containing `mcpProxyToken` and `mcpProxyPort`.
- If `change_runtime` says `Runtime API not initialized`, start the server with `--client-oauth-config`.
- If you change your MCP config and nothing updates, restart the MCP client.
- If you expose the HTTP wrapper on a custom port, point your MCP client at `/mcp`, not `/`.
- To inspect logs, look in the latest `colab-mcp-logs-*` temp directory.

## Upstream Base

This fork is based on [`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp). Google does not accept external contributions there, so the fixes and wrapper live in this fork.

## License

Apache 2.0
