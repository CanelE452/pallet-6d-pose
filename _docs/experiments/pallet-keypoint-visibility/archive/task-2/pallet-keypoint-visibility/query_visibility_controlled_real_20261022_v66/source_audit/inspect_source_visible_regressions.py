"""Show deterministic public-only examples of lost, mask-supported visibility."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "source_audit/SOURCE_VISIBLE_REGRESSION_CROPS_V2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {"path": str(path), "sha256": sha(path)}


def main():
    assert not OUTPUT.exists(), "Keep completed diagnostic galleries immutable."
    source_rows = ROOT / "source_audit/SOURCE_BOUNDARY_FAILURE_DIAGNOSIS_QUERY_ROWS.csv"
    by_id = {}
    with source_rows.open() as handle:
        for row in csv.DictReader(handle):
            if row["source"] == "ORIGINAL_YCB":
                by_id.setdefault(row["sample_id"], {})[row["model"]] = row
    manifest_path = ROOT / "controlled/COMPOSITES.jsonl"
    originals = {}
    with manifest_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] == "VALIDATION":
                original = record["target_original"]
                originals[original["id"]] = original
    qualified = []
    for sample_id, models in by_id.items():
        new, old = models["CONTROLLED_PLUS_PALLOX"], models["V65_BASELINE"]
        if (new["strict25_V"] == "True" and new["rawSAM_exact_positive"] == "True"
                and new["head_exact_visible"] == "False" and old["head_exact_visible"] == "True"):
            assert new["allowed"] == "1" and new["exact_GT"] == "0"
            qualified.append((hashlib.sha256(sample_id.encode()).hexdigest(), sample_id, new, old))
    qualified.sort()
    # First one per confident output mechanism, then hash-ranked distinct targets.
    chosen, used = [], set()
    for state in ["1", "2", "3"]:
        candidates = [r for r in qualified if r[2]["exact_predicted"] == state and r[2]["global_id"] not in used]
        if candidates:
            chosen.append(candidates[0])
            used.add(candidates[0][2]["global_id"])
    for candidate in qualified:
        if len(chosen) >= 4:
            break
        if candidate[2]["global_id"] not in used:
            chosen.append(candidate)
            used.add(candidate[2]["global_id"])
    assert len(chosen) == 4
    OUTPUT.mkdir()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 15)
    canvas = Image.new("RGB", (1160, 130 + 340 * len(chosen)), "#f4f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 14), "Public source: visible interiors lost by V66 (saved predictions only)", font=font, fill="#152c3b")
    draw.text((22, 45), "Same native query; 5x5 BOP visible-mask positive. BOP render/depth mask is not human pixel gold.", font=small, fill="#334b5b")
    draw.text((22, 70), "Source RGB + target prompt / native 96px crop x3 / separate native mask crop x3", font=small, fill="#334b5b")
    selections = []
    for index, (_, sample_id, new, old) in enumerate(chosen):
        record = originals[new["global_id"]]
        image_path, mask_path = Path(record["image"]["path"]), Path(record["visible"]["path"])
        assert sha(image_path) == record["image"]["sha256"] == new["image_sha256"]
        assert sha(mask_path) == record["visible"]["sha256"]
        rgb = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        x, y = int(new["pixel_x"]), int(new["pixel_y"])
        pixels = np.asarray(mask)
        assert (pixels[y-2:y+3, x-2:x+3] > 0).all()
        y0 = 115 + index * 340
        draw.text((22, y0), f"{sample_id}  object={record['object_id']}  inside distance={float(new['visible_inside_distance_px']):.2f}px", font=font, fill="#152c3b")
        full = rgb.resize((384, 288), Image.Resampling.BILINEAR)
        fd = ImageDraw.Draw(full)
        fd.rectangle(tuple(float(v) * .6 for v in record["box"]), outline="#00c9d8", width=2)
        qx, qy = x * .6, y * .6
        fd.line((qx-6, qy, qx+6, qy), fill="#ff1838", width=2)
        fd.line((qx, qy-6, qx, qy+6), fill="#ff1838", width=2)
        canvas.paste(full, (22, y0 + 28))
        crop_box = (x-48, y-48, x+48, y+48)
        crop = rgb.crop(crop_box).resize((288, 288), Image.Resampling.NEAREST)
        md = mask.crop(crop_box).resize((288, 288), Image.Resampling.NEAREST).convert("RGB")
        # Mark the sampled pixel with a ring, leaving its RGB value unchanged.
        for panel in [crop, md]:
            pd = ImageDraw.Draw(panel)
            pd.rectangle((144-8, 144-8, 144+10, 144+10), outline="#ff1838", width=1)
        canvas.paste(crop, (426, y0 + 28))
        canvas.paste(md, (734, y0 + 28))
        state = {"0": "V", "1": "E", "2": "O", "3": "U"}[new["exact_predicted"]]
        text_y = y0 + 294
        draw.text((426, text_y), f"pV v65={float(old['exact_pV']):.3f} -> v66={float(new['exact_pV']):.3f}", font=small, fill="#a32b2b")
        draw.text((734, text_y), f"SAM={float(new['rawSAM_exact_pV']):.3f}; v66={state}", font=small, fill="#152c3b")
        selections.append({"sample_id": sample_id, "source_record": record,
                           "v65": old, "v66": new, "RGB": bind(image_path), "visible_mask": bind(mask_path),
                           "crop_box_native_pixels": crop_box, "strict25_V_verified": True})
    image_path = OUTPUT / "PUBLIC_VISIBLE_REGRESSIONS.png"
    canvas.save(image_path)
    result = {"schema": "PUBLIC_SOURCE_VISIBLE_REGRESSION_FIXED_CROPS", "completed": True,
              "selection": "ORIGINAL_YCB VAL; strict25V; rawSAM>0; v65 pV>=.5; V66 pV<.5; SHA256(sample_id) sorted first distinct target per E/O/U then fill to4.",
              "qualified": len(qualified), "chosen": selections,
              "inputs": [bind(source_rows), bind(manifest_path), bind(__file__)],
              "figure": bind(image_path), "new_model_forward_training_PnP": 0,
              "current223_data_reads": 0,
              "interpretation": "Qualitative RGB/mask review only. Strict25 truth is BOP render/depth-derived, not independent human pixel annotation. No relabeling or training exclusion follows from this gallery."}
    (OUTPUT / "SELECTION_AND_PIXEL_CHECK.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": True, "qualified": len(qualified), "figure": str(image_path)}))


if __name__ == "__main__":
    main()
