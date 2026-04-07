# MCP Integration — Claude Code

## Setup

Run the MCP server:

```bash
python mcp_server.py
```

Or add to Claude Code:

```bash
claude mcp add mempal -- python /path/to/mnemosyne/mcp_server.py
```

## Available Tools

- **mempal_status** — palace stats (wings, rooms, drawer counts)
- **mempal_search** — semantic search across all memories
- **mempal_list_wings** — list all projects in the palace

## Usage in Claude Code

Once configured, Claude Code can search your memories directly during conversations.
