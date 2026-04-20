#!/bin/bash
echo "Starting frontend on http://localhost:3000"
conda activate ml2
source .venv/bin/activate
cd web/public
python3 -m http.server 3000
