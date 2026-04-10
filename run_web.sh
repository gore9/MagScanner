#!/bin/bash
# MagScanner — web dashboard
cd "$(dirname "$0")"
.venv2/bin/python3 app.py "$@"
