#!/bin/bash
# Собирает presentation.pdf в корне репозитория из index.html: bash docs/presentation/build.sh
# Нужны Google Chrome и шрифты клиента (npm install --prefix client).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=4000 \
  --print-to-pdf="$HERE/../../presentation.pdf" "file://$HERE/index.html" 2>/dev/null
echo "Готово: presentation.pdf"
