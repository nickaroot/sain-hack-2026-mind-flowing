#!/bin/bash
# Собирает листы кадров для аватаров из трёх роликов с выражениями персонажей (по 12 кадров в каждом).
#
#   bash client/scripts/build-avatars.sh <клиницист.mp4> <нутрициолог.mp4> <биохакер.mp4>
#
# Нужны ffmpeg, ImageMagick (magick) и cwebp. Результат — client/src/avatars/{clinician,nutritionist,biohacker}.webp:
# сетка 4×3, кадр 480×480. Что делает с каждым кадром: выравнивает по неподвижному участку корпуса,
# растворяет края в прозрачность, кладёт на квадратный холст и увеличивает в полтора раза.
# Когда появятся исходники в большем разрешении, достаточно запустить сценарий на них:
# номера кадров и раскладка в sprites.ts останутся теми же, если порядок выражений не менялся.
set -euo pipefail
[ $# -eq 3 ] || { echo "Нужны три ролика: клиницист, нутрициолог, биохакер" >&2; exit 1; }

OUT="$(cd "$(dirname "$0")/../src/avatars" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# name video — участок корпуса для выравнивания: x y ширина высота (в пикселях исходного кадра)
build() {
  local name=$1 video=$2 tx=$3 ty=$4 tw=$5 th=$6 pad=10 d="$WORK/$1"
  mkdir -p "$d/out"
  ffmpeg -v error -y -i "$video" "$d/f%02d.png"
  local w h
  w=$(magick identify -format '%w' "$d/f01.png"); h=$(magick identify -format '%h' "$d/f01.png")
  magick "$d/f01.png" -crop "${tw}x${th}+${tx}+${ty}" +repage "$d/tpl.png"
  magick -size "${w}x${h}" xc:black -fill white -draw "rectangle 9,9 $((w-10)),$((h-10))" -blur 0x5 "$d/mask.png"
  local f base found dx dy
  for f in "$d"/f??.png; do
    base=$(basename "$f")
    magick "$f" -crop "$((tw+2*pad))x$((th+2*pad))+$((tx-pad))+$((ty-pad))" +repage "$d/win.png"
    found=$(magick compare -metric RMSE -subimage-search "$d/win.png" "$d/tpl.png" null: 2>&1 | sed -E 's/.*@ ([0-9]+),([0-9]+).*/\1 \2/' || true)
    dx=$(( ${found% *} - pad )); dy=$(( ${found#* } - pad ))
    magick "$f" -alpha off -virtual-pixel edge -distort SRT "0,0 1 0 $((-dx)),$((-dy))" "$d/shift.png"
    magick "$d/shift.png" "$d/mask.png" -alpha off -compose CopyOpacity -composite PNG32:"$d/cut.png"
    magick "$d/cut.png" -background none -gravity north -extent 320x320 -filter Lanczos -resize 480x480 -unsharp 0x0.7+0.6+0.01 PNG32:"$d/out/$base"
  done
  local o="$d/out"
  magick \( "$o"/f01.png "$o"/f02.png "$o"/f03.png "$o"/f04.png +append \) \
         \( "$o"/f05.png "$o"/f06.png "$o"/f07.png "$o"/f08.png +append \) \
         \( "$o"/f09.png "$o"/f10.png "$o"/f11.png "$o"/f12.png +append \) -background none -append PNG32:"$d/sheet.png"
  cwebp -quiet -q 88 -alpha_q 90 "$d/sheet.png" -o "$OUT/$name.webp"
  echo "$name: $(wc -c < "$OUT/$name.webp" | tr -d ' ') байт, фон #$(magick "$d/f01.png" -crop 12x12+3+3 +repage -scale 1x1 -format '%[hex:p{0,0}]' info:)"
}

build clinician    "$1"  70 190 100 80
build nutritionist "$2" 100 200 110 80
build biohacker    "$3" 120 200 120 80
