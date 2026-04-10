#!/bin/bash
# Launch MagScanner
cd "$(dirname "$0")"
.venv2/bin/python3 scanner.py "$@"
