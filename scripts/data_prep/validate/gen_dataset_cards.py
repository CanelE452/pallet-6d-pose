"""데이터셋 폴더마다 DATASET.md(데이터셋 카드)를 만든다.

이미지는 .gitignore 로 GitHub 에 올라가지 않는다. 그래서 저장소만 봐서는 어떤
데이터가 어디에 얼마나 있는지 알 수 없다. 이 스크립트가 각 폴더를 실측해
카드를 남기고, .gitignore 는 `*.md` 만 통과시키므로 카드는 함께 올라간다.

    python scripts/data_prep/validate/gen_dataset_cards.py            # 전체
    python scripts/data_prep/validate/gen_dataset_cards.py --root challenge/data
    python scripts/data_prep/validate/gen_dataset_cards.py --dry-run

프레임 수·용량은 전수로 센다. split·gt_source·해상도처럼 JSON 을 열어야 하는
항목은 폴더가 크면 표본을 쓰고, 카드에 표본 크기를 명시한다 — 표본에서 나온
비율을 전수처럼 읽으면 안 되기 때문이다.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
SAMPLE_LIMIT = 400          # JSON 을 여는 항목의 표본 상한
IMG_EXT = (".png", ".jpg", ".jpeg")


def human(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def png_size(path: pathlib.Path) -> tuple[int, int] | None:
    """PNG 헤더 24바이트만 읽어 해상도를 얻는다 (디코딩 없이)."""
    try:
        with path.open("rb") as fh:
            head = fh.read(24)
        if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return struct.unpack(">II", head[16:24])
    except OSError:
        return None


def scan(folder: pathlib.Path) -> dict:
    files = [p for p in folder.rglob("*") if p.is_file()]
    imgs = [p for p in files if p.suffix.lower() in IMG_EXT]
    jsons = [p for p in files if p.suffix.lower() == ".json"]
    total_bytes = sum(p.stat().st_size for p in files)

    # 표본으로 여는 항목
    sample = jsons[:SAMPLE_LIMIT]
    splits, sources, classes = collections.Counter(), collections.Counter(), collections.Counter()
    for p in sample:
        try:
            payload = json.loads(p.read_text("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):   # 라벨이 아닌 사이드카(list 형태 등)
            continue
        objects = payload.get("objects")
        if not isinstance(objects, list) or not objects:
            continue
        obj = objects[0]
        splits[obj.get("split", "(none)")] += 1
        sources[obj.get("gt_source", "(none)")] += 1
        classes[obj.get("class", "(none)")] += 1

    res = collections.Counter()
    for p in imgs[:SAMPLE_LIMIT]:
        wh = png_size(p)
        if wh:
            res[f"{wh[0]}x{wh[1]}"] += 1

    return {
        "files": len(files), "images": len(imgs), "jsons": len(jsons),
        "bytes": total_bytes, "sample": len(sample),
        "splits": splits, "sources": sources, "classes": classes,
        "res": res, "img_sample": min(len(imgs), SAMPLE_LIMIT),
        "examples": sorted(p.name for p in files[:3]),
    }


def card(folder: pathlib.Path, info: dict) -> str:
    rel = folder.relative_to(ROOT)
    out = [f"# {folder.name}", "",
           f"`{rel}`", "",
           "이미지는 저장소에 올리지 않는다. 이 카드는 폴더를 실측한 요약이며,",
           "`scripts/data_prep/validate/gen_dataset_cards.py` 로 다시 만들 수 있다.",
           "", "## 규모", "", "```",
           f"파일     {info['files']:>8,}",
           f"이미지   {info['images']:>8,}",
           f"JSON     {info['jsons']:>8,}",
           f"용량     {human(info['bytes']):>8}", "```", ""]

    if info["res"]:
        out += ["## 해상도", "",
                f"이미지 {info['img_sample']:,}장 표본." if info["img_sample"] < info["images"]
                else f"이미지 {info['images']:,}장 전수.", "", "```"]
        for k, v in info["res"].most_common(5):
            out.append(f"{k:<12} {v:>6,}")
        out += ["```", ""]

    if info["sample"]:
        note = (f"JSON {info['sample']:,}개 표본 (전체 {info['jsons']:,}개)."
                if info["sample"] < info["jsons"] else f"JSON {info['jsons']:,}개 전수.")
        out += ["## 라벨", "", note, "", "```"]
        for title, counter in (("split", info["splits"]),
                               ("gt_source", info["sources"]),
                               ("class", info["classes"])):
            if not counter:
                continue
            out.append(f"[{title}]")
            for k, v in counter.most_common(6):
                out.append(f"  {k:<28} {v:>6,}")
        out += ["```", ""]

    if info["examples"]:
        out += ["## 파일명 예", "", "```"] + list(info["examples"]) + ["```", ""]
    return "\n".join(out)


# 카드 한 장이 덮는 단위. 하위 폴더(overlays/, images/train/ …)는 여기에 합산한다.
# 이미지를 담은 폴더마다 카드를 두면 146장이 나와 오히려 안 읽힌다.
DATASET_GLOBS = [
    "challenge/data/01_real/*/*",
    "challenge/data/02_synthetic/training/*",
    "challenge/data/03_derived/*",
    "challenge/data/04_results/*",
    "data/pallet/training_data/*",
    "data/pallet/training_data/paper_release/*",
    "data/pallet/real_unlabeled_ralph*",
    "data/pallet/raw_data/*",
]


def summary(rows: list[tuple[pathlib.Path, dict]]) -> str:
    """전 데이터셋 한 장 요약. 저장소 방문자가 처음 보는 문서."""
    total_img = sum(i["images"] for _, i in rows)
    total_b = sum(i["bytes"] for _, i in rows)
    out = ["# 데이터셋 목록", "",
           "이미지는 `.gitignore` 로 저장소에 올리지 않는다. 이 표는 실제 디스크를",
           "실측한 것이고, 폴더마다 `DATASET.md` 에 더 자세한 카드가 있다.",
           "",
           f"총 {len(rows)}개 데이터셋 · 이미지 {total_img:,}장 · {human(total_b)}",
           "",
           "재생성: `python scripts/data_prep/validate/gen_dataset_cards.py`",
           ""]

    groups: dict[str, list] = collections.OrderedDict()
    for rel, info in rows:
        parts = rel.parts
        if parts[:2] == ("challenge", "data"):
            key = f"challenge/data/{parts[2]}" if len(parts) > 2 else "challenge/data"
        else:
            key = "/".join(parts[:3])
        groups.setdefault(key, []).append((rel, info))

    for key, items in groups.items():
        gi = sum(i["images"] for _, i in items)
        gb = sum(i["bytes"] for _, i in items)
        out += [f"## {key}", "",
                f"{len(items)}개 · 이미지 {gi:,}장 · {human(gb)}", "", "```",
                f"{'dataset':<44} {'images':>8} {'json':>8} {'size':>8}",
                "─" * 72]
        for rel, info in sorted(items, key=lambda x: -x[1]["images"]):
            out.append(f"{rel.name:<44} {info['images']:>8,} "
                       f"{info['jsons']:>8,} {human(info['bytes']):>8}")
        out += ["```", ""]
    return "\n".join(out)


def dataset_folders(patterns: list[str]) -> list[pathlib.Path]:
    """데이터셋 단위 폴더. 이미지가 하위 어딘가에 있으면 대상이다."""
    seen, out = set(), []
    for pat in patterns:
        for d in sorted(ROOT.glob(pat)):
            if not d.is_dir() or d in seen:
                continue
            if any(part.startswith(".") for part in d.parts):
                continue
            if any(p.suffix.lower() in IMG_EXT for p in d.rglob("*") if p.is_file()):
                seen.add(d)
                out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glob", action="append", default=[],
                    help="데이터셋 단위 glob (ROOT 상대). 기본: DATASET_GLOBS")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", default="_docs/DATASETS.md",
                    help="전체 요약표를 쓸 경로. 빈 문자열이면 안 쓴다")
    args = ap.parse_args()

    folders = dataset_folders(args.glob or DATASET_GLOBS)
    rows, written = [], 0
    for folder in folders:
        info = scan(folder)
        if not info["images"]:
            continue
        target = folder / "DATASET.md"
        print(f"{'[dry] ' if args.dry_run else ''}{target.relative_to(ROOT)}  "
              f"img {info['images']:,}  {human(info['bytes'])}")
        if not args.dry_run:
            target.write_text(card(folder, info), encoding="utf-8")
        rows.append((folder.relative_to(ROOT), info))
        written += 1

    if args.summary and not args.dry_run and rows:
        path = ROOT / args.summary
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary(rows), encoding="utf-8")
        print(f"summary -> {args.summary}")
    print(f"{'would write' if args.dry_run else 'wrote'} {written} cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
