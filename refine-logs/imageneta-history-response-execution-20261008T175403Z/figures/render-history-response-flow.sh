#!/usr/bin/env bash
set -euo pipefail
DIAGRAM_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
DIAGRAM_TOOLS=/tmp/history-response-mermaid-tools
if [[ -x "$DIAGRAM_TOOLS/package/node_modules/.bin/mmdc" ]]; then
    export PATH="$DIAGRAM_TOOLS/node-v22.14.0-linux-x64/bin:$PATH"
    export PUPPETEER_CACHE_DIR="$DIAGRAM_TOOLS/browser"
    DIAGRAM_CLI="$DIAGRAM_TOOLS/package/node_modules/.bin/mmdc"
else
    DIAGRAM_CLI=mmdc
fi
for DIAGRAM_FORMAT in png svg pdf; do
    "$DIAGRAM_CLI" -i "$DIAGRAM_DIR/history-response-flow.mmd" \
        -o "$DIAGRAM_DIR/history-response-flow.$DIAGRAM_FORMAT" \
        -p "$DIAGRAM_DIR/puppeteer-config.json" \
        -C "$DIAGRAM_DIR/flow-theme.css" -b white --size 2400 -s 1
done
