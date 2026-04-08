#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Mnemosyne — Start Everything
# ═══════════════════════════════════════════════════════════
#
#  Usage:
#    ./start.sh              # Start all services
#    ./start.sh --no-ollama  # Skip Ollama (if running separately)
#    ./start.sh --stop       # Stop all services
#
#  Services started:
#    1. Ollama (local LLM server, port 11434)
#    2. LibreChat + MongoDB + MeiliSearch (port 3080)
#    3. Mnemosyne MCP Server (connected via stdio in LibreChat)
#
#  Open: http://localhost:3080
# ═══════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ─── Stop ────────────────────────────────────────────────
if [ "$1" = "--stop" ]; then
    echo -e "${CYAN}Stopping Mnemosyne...${NC}"
    docker compose -f docker-compose.librechat.yml down 2>/dev/null || true
    pkill -f "ollama serve" 2>/dev/null || true
    echo -e "${GREEN}All services stopped.${NC}"
    exit 0
fi

echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo -e "${CYAN}  Mnemosyne — Starting All Services${NC}"
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo ""

# ─── 1. Ollama ───────────────────────────────────────────
if [ "$1" != "--no-ollama" ]; then
    if command -v ollama &>/dev/null; then
        if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
            echo -e "${YELLOW}Starting Ollama...${NC}"
            ollama serve &>/tmp/ollama.log &
            sleep 2
        fi

        # Pull model if needed
        MODEL="${OLLAMA_MODEL:-phi3.5}"
        if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
            echo -e "${YELLOW}Pulling model: $MODEL (this may take a few minutes)...${NC}"
            ollama pull "$MODEL"
        fi
        echo -e "${GREEN}[OK] Ollama running (model: $MODEL)${NC}"
    else
        echo -e "${YELLOW}[SKIP] Ollama not installed. Install: curl -fsSL https://ollama.com/install.sh | sh${NC}"
    fi
fi

# ─── 2. Mnemosyne Python Package ────────────────────────
if ! python -c "import mempalace" 2>/dev/null; then
    echo -e "${YELLOW}Installing Mnemosyne...${NC}"
    pip install -e ".[dev]" -q
fi
echo -e "${GREEN}[OK] Mnemosyne Python package${NC}"

# ─── 3. LibreChat ───────────────────────────────────────
if command -v docker &>/dev/null; then
    echo -e "${YELLOW}Starting LibreChat...${NC}"
    docker compose -f docker-compose.librechat.yml up -d
    echo -e "${GREEN}[OK] LibreChat running${NC}"
else
    echo -e "${RED}[ERROR] Docker not installed. LibreChat requires Docker.${NC}"
    echo -e "  Install: https://docs.docker.com/get-docker/"
    exit 1
fi

# ─── 4. Status ──────────────────────────────────────────
echo ""
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo -e "${CYAN}  Mnemosyne is ready!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo ""
echo -e "  ${GREEN}LibreChat:${NC}    http://localhost:3080"
echo -e "  ${GREEN}Ollama:${NC}       http://localhost:11434"
echo ""
echo -e "  ${YELLOW}First time?${NC}  Create an account at http://localhost:3080"
echo -e "  ${YELLOW}MCP Tools:${NC}   25 tools auto-loaded (search, KG, graph, etc.)"
echo ""
echo -e "  Stop with: ${CYAN}./start.sh --stop${NC}"
echo ""
