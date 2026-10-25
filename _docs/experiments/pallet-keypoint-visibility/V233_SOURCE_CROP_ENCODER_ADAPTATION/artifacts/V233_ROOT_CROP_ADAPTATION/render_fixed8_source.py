"""Render prospectively fixed SOURCE DEV buckets; never runs a model."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

H = Path(__file__).parent
POLICY_SHA = "dcd313caf192764124e92a794aee1a7de2c5d1bdfa2eb5fec923315a7bc5c11f"
FIT_SHA = "dd662d0a466e38972a386104f9a26f235f955b060c20da4a3b678283f70ed114"
CROP_SHA = "45b294ea7e89ff1a9e7b1b4765a24c88b2b0b402a79aaa3e5e77b65f4ac80535"


def guard(p):
    p = Path(p)
    if "v112" in str(p).lower():
        raise ValueError("forbidden lexical path")
    if p.is_symlink():
        raise ValueError("no symlinks")
    return p


def digest(p):
    s = hashlib.sha256()
    with guard(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            s.update(b)
    return s.hexdigest()


def bound(p, sha):
    p = guard(p)
    assert digest(p) == sha, str(p)
    return p


def dump(p, obj):
    guard(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    out = H / "fixed8_source_dev_gallery_v2"
    out.mkdir(exist_ok=False)
    policy = json.loads(bound(H / "FIXED8_DEV_CASE_POLICY_BEFORE_PREDICTION_READ.json", POLICY_SHA).read_text())
    fit = json.loads(bound(H / "common_local_crop_pair_v2/COMPLETE.json", FIT_SHA).read_text())
    crop = json.loads(bound(H / "crop_samples_v1/COMPLETE.json", CROP_SHA).read_text())
    arms = ["CONTROL_LOCAL_CROP", "TREAT_LAST_QKV_LORA4"]
    rows = {}
    for a in arms:
        b = fit["outputs"][a]
        with bound(b["path"], b["sha256"]).open() as f:
            rows[a] = {int(r["sample_index"]): r for r in csv.DictReader(f)}
        assert len(rows[a]) == 7617
    assert rows[arms[0]].keys() == rows[arms[1]].keys()
    selected = []
    for k, bucket in enumerate(policy["buckets"]):
        ids = []
        for i, r in rows[arms[1]].items():
            if r["role"] != "DEV" or int(r["family"]) != bucket["family"] or int(r["target"]) != bucket["target"]:
                continue
            if (r["prediction"] == ("D" if bucket["target"] == 0 else "E")) == bucket["treatment_correct"]:
                ids.append(i)
        selected.append(dict(bucket, bucket_index=k, population=len(ids), sample_index=min(ids) if ids else None))
    # Written before raw crop array is opened or hashed.
    dump(out / "CASE_IDS_BEFORE_PIXEL_READ.json", {"policy_sha256": POLICY_SHA, "selection": selected})
    bindings = {Path(b["path"]).name: b for b in crop["outputs"]}
    db = bindings["SAMPLE_DIAGNOSTICS.jsonl"]
    diagnostics = {}
    wanted = {r["sample_index"] for r in selected if r["sample_index"] is not None}
    with bound(db["path"], db["sha256"]).open() as f:
        for line in f:
            d = json.loads(line)
            if d["sample_index"] in wanted:
                diagnostics[d["sample_index"]] = d
    assert set(diagnostics) == wanted
    rb = bindings["raw_rgb96.npy"]
    rgb = np.load(bound(rb["path"], rb["sha256"]), mmap_mode="r", allow_pickle=False)
    assert rgb.shape == (7617, 3, 96, 96) and rgb.dtype == np.uint8
    fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(fontpath, 18)
    small = ImageFont.truetype(fontpath, 15)
    title = ImageFont.truetype(fontpath, 25)
    board = Image.new("RGB", (1600, 1140), "#101827")
    draw = ImageDraw.Draw(board)
    draw.text((24, 15), "V233 SOURCE DEV: same q/L crop, frozen encoder vs last-qkv LoRA", fill="white", font=title)
    draw.text((24, 53), "D=visible, E=external occlusion. Cyan=query; magenta=SYN reference (diagnostic only).", fill="#d6e2f1", font=font)
    draw.text((24, 83), "First sample in each fixed bucket; counts show its population. SOURCE internal DEV, not a blind/current test.", fill="#d6e2f1", font=small)
    cases = []
    for s in selected:
        k = s["bucket_index"]
        x, y = 20 + (k % 4) * 395, 125 + (k // 4) * 470
        family = "SYN native witness" if s["family"] == 0 else "YCB associated-r"
        truth = "D" if s["target"] == 0 else "E"
        status = "correct" if s["treatment_correct"] else "error"
        draw.rounded_rectangle((x, y, x + 380, y + 454), radius=8, fill="#1f2a3c")
        draw.text((x + 12, y + 10), f"{family} | true {truth}", fill="white", font=font)
        draw.text((x + 12, y + 37), f"Treatment {status} | bucket N={s['population']}", fill="#9cd6ff", font=small)
        i = s["sample_index"]
        if i is None:
            draw.text((x + 25, y + 170), "EMPTY BUCKET", fill="#ffc27a", font=title)
            cases.append(s)
            continue
        d = diagnostics[i]
        c, t = rows[arms[0]][i], rows[arms[1]][i]
        assert c["row_id"] == t["row_id"] == d["row_id"]
        assert int(c["target"]) == int(t["target"]) == d["target"]
        # The assembler's diagnostic hash is CHW, while PNG expects HWC.
        assert hashlib.sha256(rgb[i].tobytes()).hexdigest() == d["pixel_sha256"]
        pixels = np.transpose(rgb[i], (1, 2, 0)).copy()
        raw = Image.fromarray(pixels)
        raw.save(out / f"CASE_{k:02d}_RAW96.png")
        view = raw.resize((288, 288), Image.Resampling.NEAREST)
        vd = ImageDraw.Draw(view)
        qx = qy = 143.5
        vd.line((qx - 12, qy, qx + 12, qy), fill="#00ffff", width=2)
        vd.line((qx, qy - 12, qx, qy + 12), fill="#00ffff", width=2)
        local = d.get("reference_locality_diagnostic_only") or {}
        ref = local.get("raw96_reference_index_xy")
        if ref and all(math.isfinite(float(z)) for z in ref):
            rx, ry = (float(z) * 3 + 1 for z in ref)
            vd.ellipse((rx - 7, ry - 7, rx + 7, ry + 7), outline="#ff42df", width=2)
        view.save(out / f"CASE_{k:02d}_OVERLAY.png")
        board.paste(view, (x + 46, y + 65))
        draw.text((x + 12, y + 362), f"sample {i} | CONTROL {c['prediction']} | TREAT {t['prediction']}", fill="white", font=small)
        draw.text((x + 12, y + 386), f"logit: {float(c['selected_family_logit']):.3f} / {float(t['selected_family_logit']):.3f}", fill="#d6e2f1", font=small)
        dist = d.get("q_reference_distance_px_diagnostic_only")
        disttext = f"q-reference {dist:.3f}px" if dist is not None else "native UNKNOWN; reference not exported"
        draw.text((x + 12, y + 410), disttext, fill="#d6e2f1", font=small)
        cases.append(dict(s, diagnostic=d, control=c, treatment=t, raw_png=f"CASE_{k:02d}_RAW96.png", overlay_png=f"CASE_{k:02d}_OVERLAY.png"))
    draw.text((24, 1083), "Masked S/U rows are not model abstentions. No retraining, threshold change, coordinate correction or PnP in this gallery.", fill="#d6e2f1", font=small)
    draw.text((24, 1110), "Cross-family ancestry UNKNOWN. SYN and YCB targets differ. These crops alone do not establish the cause of transfer failure.", fill="#ffc27a", font=small)
    board.save(out / "SOURCE_DEV_FIXED8.png")
    dump(out / "CASE_DATA.json", {"cases": cases, "policy_sha256": POLICY_SHA, "fit_complete_sha256": FIT_SHA, "crop_complete_sha256": CROP_SHA})
    cards = []
    for c in cases:
        heading = f"{'SYN native witness' if c['family']==0 else 'YCB associated-r'} / {'D' if c['target']==0 else 'E'} / {'정답' if c['treatment_correct'] else '오답'} / 모집단 {c['population']}"
        if c["sample_index"] is None:
            body = "이 칸에 해당하는 사례가 없습니다. YCB E 정답 0개도 그대로 표시합니다."
        else:
            d = c["diagnostic"]
            body = f'<img src="{c["overlay_png"]}" alt="고정 원 query crop"><p>sample {c["sample_index"]}: <code>{html.escape(d["row_id"])}</code></p><p>CONTROL {c["control"]["prediction"]}, TREAT {c["treatment"]["prediction"]}. <a href="{c["raw_png"]}">원본96 crop</a></p><details><summary>좌표·라벨·점수</summary><pre>{html.escape(json.dumps(c,ensure_ascii=False,indent=2))}</pre></details>'
        cards.append(f"<article><h2>{heading}</h2>{body}</article>")
    text = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>V233 SOURCE 실제 crop</title><style>body{font:16px system-ui;background:#101827;color:#e5edf5;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}article{background:#1f2a3c;padding:16px;border-radius:10px}h2{font-size:18px}a{color:#8ddcff}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%;image-rendering:pixelated}</style><h1>V233: 잘된 경우와 실패한 경우</h1><p>D=보임, E=외부 가림. cyan 십자가는 원 query, magenta 원은 SYN teacher reference(진단 전용)입니다. 실제 q/L crop을 그대로 사용했습니다.</p><p>점수 확인 전 고정한 각 칸의 최소 sample index를 표시합니다. 공개 SOURCE의 내부 DEV이며 새로운 비공개 평가가 아닙니다. SYN native witness와 YCB associated-r의 의미는 다릅니다. YCB native는 UNKNOWN입니다. 교차 계보 독립성도 UNKNOWN이므로 현재 실사 성공으로 해석할 수 없습니다.</p><p><a href="SOURCE_DEV_FIXED8.png">전체 이미지</a> · <a href="CASE_DATA.json">모든 사례 값</a> · <a href="CASE_IDS_BEFORE_PIXEL_READ.json">픽셀 열기 전 선택 명세</a></p><main>' + ''.join(cards) + '</main></html>'
    (out / "ROOT_READOUT_KO.html").write_text(text)
    outputs = []
    for p in sorted(out.iterdir()):
        outputs.append({"path": p.name, "sha256": digest(p), "size_bytes": p.stat().st_size})
    dump(out / "COMPLETE.json", {"PASS": True, "model_forward": 0, "new_fit": 0, "current_or_held_input": 0, "actual_crop_rows": len(wanted), "empty_buckets": sum(s['sample_index'] is None for s in selected), "code_sha256": digest(Path(__file__)), "policy_sha256": POLICY_SHA, "outputs": outputs})
    print(json.dumps({"PASS": True, "gallery": str(out), "samples": sorted(wanted), "populations": [s['population'] for s in selected]}))


if __name__ == "__main__":
    main()
