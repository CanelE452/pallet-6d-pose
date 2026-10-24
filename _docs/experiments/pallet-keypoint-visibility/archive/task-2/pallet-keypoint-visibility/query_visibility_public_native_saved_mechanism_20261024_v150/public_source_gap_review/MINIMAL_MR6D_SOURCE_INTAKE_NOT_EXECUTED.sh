#!/usr/bin/env bash
# NOT EXECUTED. Literal tiny example, not V151 prospective selection.
# Parent V151 policy/entry determines any actual acquisition.
set -euo pipefail
source_out="${1:?Pass a fresh dedicated public-source intake directory}"
mkdir -p "$source_out"
source_base='https://huggingface.co/datasets/anas-gouda/mr6d/resolve/3c4ed8453312655499c3d3a3cf38b4929fae77ee/'
while IFS= read -r rel; do
  mkdir -p "$source_out/$(dirname "$rel")"
  test ! -e "$source_out/$rel"
  curl --fail --location --max-time 45 --max-filesize 1000000 --output "$source_out/$rel" "$source_base$rel"
done <<'SOURCE_FILES'
README.md
models_names.json
models/models_info.json
models/obj_000001.ply
val/000000/scene_camera.json
val/000000/scene_gt.json
val/000000/scene_gt_info.json
val/000000/rgb/000000.jpg
val/000000/depth/000000.png
val/000000/mask/000000_000000.png
val/000000/mask/000000_000001.png
val/000000/mask_visib/000000_000000.png
val/000000/mask_visib/000000_000001.png
SOURCE_FILES
