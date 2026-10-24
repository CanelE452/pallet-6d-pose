"""Render sealed public source recipes/crops only. No automatic execution/import I/O."""
import argparse
import ast
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

H = Path(__file__).resolve().parent
V = H.parent
META = V / 'input_preparation'
OUT = V / 'fixed_corpus_visual_review'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
STAGES = ('BEFORE', 'OPAQUE_COVER', 'CUE_PRESERVING_SHAM')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(binding):
    if sha(binding['path']) != binding['sha256']:
        raise ValueError('Binding mismatch: ' + binding['path'])
    return Path(binding['path'])


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def select(cues, attempts):
    """All32 + all8 ranks; first feasible original index, never quality selection."""
    bykey = {(r['cue_index'], r['donor_rank']): r for r in attempts}
    if sorted(bykey) != [(ci, rank) for ci in range(32) for rank in range(8)]:
        raise ValueError('Exact256 cue/rank attempts required')
    cuecases, rankcases = [], []
    for ci in range(32):
        rr = [bykey[(ci, rank)] for rank in range(8)]
        ok = [r['donor_rank'] for r in rr if r['pair_ready'] is True]
        cuecases.append(dict(cue_index=ci, donor_rank=ok[0] if ok else None, jitter_index=4,
                             role='ALL32_CUES_CENTRAL_FIRST_FEASIBLE_RANK',
                             feasible_ranks=ok, failed_rank_count=8-len(ok),
                             failed_reasons=dict(Counter(r['reason'] for r in rr if not r['pair_ready']))))
    for rank in range(8):
        ok = [ci for ci in range(32) if bykey[(ci, rank)]['pair_ready'] is True]
        rankcases.append(dict(donor_rank=rank, cue_index=ok[0] if ok else None, jitter_index=4,
                              role='ALL8_RANKS_FIRST_FEASIBLE_CUE',
                              no_feasible_pair=not bool(ok)))
    return cuecases, rankcases, bykey


def reconstruct_function(code):
    tree = ast.parse(Path(code).read_text())
    wanted = {'require', 'patch_slices', 'reconstruct_full_RGB'}
    ns = dict(np=np)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(nodes) != 3:
        raise ValueError('Executed producer reconstruction API changed')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(code), 'exec'), ns)
    return ns['reconstruct_full_RGB']


def font(size):
    return ImageFont.truetype(FONT, size)


def wrapped(draw, xy, message, size=18, width=1400, maxlines=2, color='#152236'):
    x, y = xy
    f = font(size)
    lines, line = [], ''
    for ch in str(message):
        if ch == '\n' or draw.textlength(line + ch, font=f) > width:
            lines.append(line)
            line = '' if ch == '\n' else ch
        else:
            line += ch
    if line:
        lines.append(line)
    for i, s in enumerate(lines[:maxlines]):
        draw.text((x, y + i * (size + 6)), s, font=f, fill=color)


def paste_fit(canvas, image, xy, wh, nearest=False):
    image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    scale = min(wh[0] / image.width, wh[1] / image.height)
    size = tuple(max(1, round(x * scale)) for x in image.size)
    resized = image.resize(size, Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)
    canvas.paste(resized, (xy[0] + (wh[0]-size[0])//2, xy[1] + (wh[1]-size[1])//2))


def marked_full(rgb, cue, evidence):
    im = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(im)
    thick = max(2, round(max(im.size) / 550))
    d.rectangle(cue['target_bbox_xyxy'], outline='#15a6dc', width=thick)
    for r in evidence['original_regions']:
        d.rectangle(r['bbox'], outline='#ffd400', width=thick)
    x, y = evidence['reference_xy']
    radius = max(7, round(max(im.size) / 130))
    d.line((x-radius, y, x+radius, y), fill='#f62246', width=thick)
    d.line((x, y-radius, x, y+radius), fill='#f62246', width=thick)
    return im


def make_card(case, cue, evidence, attempt, occurrences, crops, native_image, provenance):
    ci, rank = case['cue_index'], case['donor_rank']
    qi = 9*ci+4
    indices = [17*qi] if rank is None else [17*qi, 17*qi+1+2*rank, 17*qi+2+2*rank]
    card = Image.new('RGB', (1440, 860), '#ffffff')
    d = ImageDraw.Draw(card)
    ready = rank is not None
    title = f"CUE {ci:02d} · {cue['source']} · {cue['split']} · " + (f"rank {rank}" if ready else 'NO FEASIBLE PAIR')
    wrapped(d, (18, 8), title, 25)
    wrapped(d, (18, 46), cue['cue_id'] + ' | ' + cue['development_group'], 17)
    q = occurrences[indices[0]]
    wrapped(d, (18, 74), f"central offset [0,0], q={q['query_xy']}, L={q['L']:.4f}px · 원 cue/unc 유지 · 실패 rank {case.get('failed_rank_count', 'see all-cue gallery')}/8", 17)
    if ready:
        mass = attempt['local_alpha_mass96_at_r_L0']
        wrapped(d, (18, 102), f"donor={attempt['donor_id']} · scale={attempt['selected_factor']:.2f}×L0 · local alpha mass COVER={mass[0]:.3f}, SHAM={mass[1]:.3f}, Δ={abs(mass[0]-mass[1]):.3f} / 9216 (동일 mass 주장 없음)", 16)
    else:
        wrapped(d, (18, 102), '모든 donor rank 실패: ' + json.dumps(case['failed_reasons'], ensure_ascii=False), 16)
    for stage_no, stage in enumerate(STAGES):
        x = 16 + 480*stage_no
        wrapped(d, (x, 140), stage, 21, width=450, maxlines=1)
        if stage_no >= len(indices):
            wrapped(d, (x+40, 230), 'NO FEASIBLE PAIR\n다른 donor로 대체하지 않음\n해당 after 입력은 U로 보존', 21, width=400, maxlines=4)
            continue
        idx = indices[stage_no]
        r = occurrences[idx]
        if (r['cue_index'], r['jitter_index'], r['variant']) != (ci, 4, stage):
            raise ValueError('Central offset/variant identity differs')
        if r['query_xy'] != q['query_xy'] or r['L'] != q['L'] or r['initial_points'] != q['initial_points'] or r['context_box_xyxy'] != q['context_box_xyxy']:
            raise ValueError('Matched query geometry changed across variants')
        rgb = native_image(r['full_RGB_recipe'])
        raw96 = crops['rgb96'][idx]
        valid96 = crops['valid96'][idx]
        if digest(raw96) != r['rgb96_sha256'] or digest(valid96) != r['valid96_sha256']:
            raise ValueError('Saved RGB96/valid96 differs')
        wrapped(d, (x, 174), 'full native context · 표시 복사본', 16, width=455)
        paste_fit(card, marked_full(rgb, cue, evidence), (x, 202), (452, 286))
        wrapped(d, (x, 500), 'actual RGB96 ×3 · 무표시 입력', 18, width=455)
        paste_fit(card, raw96.transpose(1, 2, 0), (x+80, 535), (288, 288), nearest=True)
        wrapped(d, (x, 827), f"row {idx} · valid={valid96.mean():.4f} · available={r['query_available']}", 15, width=455)
        provenance.append(dict(occurrence_index=idx, occurrence_id=r['occurrence_id'], cue_index=ci,
                               donor_rank=rank, variant=stage, q=r['query_xy'], L=r['L'],
                               recipe=r['full_RGB_recipe'], RGB96_sha256=r['rgb96_sha256'],
                               valid96_sha256=r['valid96_sha256']))
    return card


def placeholder_rank(rank):
    im = Image.new('RGB', (1440, 860), '#ffffff')
    d = ImageDraw.Draw(im)
    wrapped(d, (40, 70), f'DONOR RANK {rank}: NO FEASIBLE EXAMPLE', 30)
    wrapped(d, (40, 150), '32개 cue 모두 geometry-feasible pair 없음. 다른 rank/cue로 교체하지 않습니다.', 24)
    return im


def contact(images, title):
    out = Image.new('RGB', (1440, 84 + 430*((len(images)+1)//2)), '#ffffff')
    d = ImageDraw.Draw(out)
    wrapped(d, (16, 8), title, 24)
    wrapped(d, (16, 43), 'full context는 cue/unc 표시 복사본, RGB96는 저장된 무표시 모델 입력. 개별 card에서 확대 확인.', 16)
    for n, im in enumerate(images):
        paste_fit(out, im, (720*(n%2), 84+430*(n//2)), (720, 430))
    return out


def run(seal_path):
    seal = read(seal_path)
    if seal.get('completed') is not True or seal.get('PASS') is not True:
        raise ValueError('Completed corpus seal required; no partial-source rendering')
    if OUT.exists():
        raise FileExistsError('Preserve existing visual review; no automatic rerender')
    # Only source metadata is read while fixing the complete all32/rank8 selection.
    protocol = read(checked(seal['protocol']))
    executed_code = checked(seal['code'])
    outputs = seal['outputs']
    attempts = rows(checked(outputs['FEASIBILITY256_GT_ONLY']))
    occurrences = rows(checked(outputs['OCCURRENCES4896_INPUT_ONLY']))
    cues = rows(META / 'CUES32_INPUT_ONLY.jsonl')
    evidence = rows(META / 'CUES32_WEAK_EVIDENCE_GT_ONLY.jsonl')
    if len(cues) != 32 or len(evidence) != 32 or len(occurrences) != 4896 or [r['occurrence_index'] for r in occurrences] != list(range(4896)):
        raise ValueError('All32/4896 retention/order required')
    # Check actual executed producer metadata bindings when present; no parameter file is opened.
    for r in cues:
        if evidence[r['cue_index']]['cue_id'] != r['cue_id']:
            raise ValueError('Cue/evidence identity differs')
    cuecases, rankcases, bykey = select(cues, attempts)
    OUT.mkdir()
    write(OUT / 'SELECTION_BEFORE_VISUAL_PAYLOADS.json', dict(
        completed=True, corpus_seal=bind(seal_path), all32_central_first_feasible_rank=cuecases,
        all8_rank_first_feasible_cue=rankcases, no_quality_substitution=True,
        rule='All32 cue central jitter4, smallest feasible rank0..7; all8 ranks smallest feasible cue_index; absent groups explicit.',
        selection_precedes_any_RGB_patch_crop_payload_read=True))
    with np.load(checked(outputs['INPUT_CROPS4896_INPUT_ONLY']), allow_pickle=False) as f:
        crops = {key: np.array(f[key], copy=True) for key in ('rgb96', 'valid96', 'occurrence_index')}
    if crops['rgb96'].shape != (4896, 3, 96, 96) or crops['rgb96'].dtype != np.uint8 or crops['valid96'].shape != (4896, 1, 96, 96) or crops['valid96'].dtype != bool or not np.array_equal(crops['occurrence_index'], np.arange(4896)):
        raise ValueError('Actual saved RGB96/valid schema differs')
    reconstruct = reconstruct_function(executed_code)
    originals, patches = {}, {}

    def native_image(recipe):
        b = recipe['original_full_RGB']
        if b['path'] not in originals:
            path = checked(b)
            if recipe['original_format'] == 'NPY':
                a = np.load(path, allow_pickle=False)
            else:
                with Image.open(path) as im:
                    a = np.array(im.convert('RGB'), copy=True)
            if a.dtype != np.uint8 or a.shape != tuple(recipe['raw_hw'])+(3,):
                raise ValueError('Native source schema')
            originals[b['path']] = a
        a = originals[b['path']]
        if recipe['kind'] == 'OPAQUE_PATCH':
            pb = recipe['patch']
            if pb['path'] not in patches:
                with np.load(checked(pb), allow_pickle=False) as f:
                    patches[pb['path']] = (np.array(f['rgb'], copy=True), np.array(f['alpha'], copy=True))
            pr, pa = patches[pb['path']]
            a = reconstruct(a, pr, pa, recipe['placement_xy'])
        elif recipe['kind'] != 'ORIGINAL':
            raise ValueError('Unknown native reconstruction recipe')
        if digest(a) != recipe['decoded_RGB_sha256']:
            raise ValueError('Native reconstructed RGB differs from actual corpus')
        return a

    provenance, cuefiles, rankfiles, contactfiles = [], [], [], []
    thumbs = []
    for c in cuecases:
        ci, rank = c['cue_index'], c['donor_rank']
        card = make_card(c, cues[ci], evidence[ci], None if rank is None else bykey[(ci, rank)], occurrences, crops, native_image, provenance)
        name = f"CUE_{ci:02d}_CENTRAL_" + ('NO_FEASIBLE_PAIR' if rank is None else f'RANK_{rank:02d}') + '.png'
        path = OUT / name
        card.save(path)
        cuefiles.append(bind(path))
        thumbs.append(card.resize((720, 430), Image.Resampling.BILINEAR))
        if len(thumbs) == 8:
            first = ci-7
            path = OUT / f'CONTACT_CUES_{first:02d}_{ci:02d}.png'
            contact(thumbs, f'ALL32 중앙 offset · CUE {first:02d}–{ci:02d} (첫 feasible rank)').save(path)
            contactfiles.append(bind(path))
            thumbs = []
    rankthumbs = []
    for case in rankcases:
        rank, ci = case['donor_rank'], case['cue_index']
        if ci is None:
            card = placeholder_rank(rank)
        else:
            cc = dict(cuecases[ci], donor_rank=rank)
            card = make_card(cc, cues[ci], evidence[ci], bykey[(ci, rank)], occurrences, crops, native_image, provenance)
        path = OUT / f'RANK_{rank:02d}_FIRST_FEASIBLE.png'
        card.save(path)
        rankfiles.append(bind(path))
        rankthumbs.append(card.resize((720, 430), Image.Resampling.BILINEAR))
    path = OUT / 'CONTACT_ALL8_DONOR_RANKS.png'
    contact(rankthumbs, 'Donor rank 0–7 · 원 cue 순서 첫 feasible example / 없으면 명시').save(path)
    contactfiles.append(bind(path))
    cards_html = ''.join(f'<article><a href="{Path(b["path"]).name}"><img src="{Path(b["path"]).name}" alt="cue{n:02d} centraloffset"></a><p>cue{n:02d} | failed ranks {cuecases[n]["failed_rank_count"]}/8</p></article>' for n, b in enumerate(cuefiles))
    ranks_html = ''.join(f'<article><a href="{Path(b["path"]).name}"><img src="{Path(b["path"]).name}" alt="donor rank{n}"></a></article>' for n, b in enumerate(rankfiles))
    gallery = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V122 actual source corpus</title><style>body{{font:17px/1.6 sans-serif;max-width:1480px;margin:24px auto;padding:0 18px;color:#152236}}img{{width:100%;height:auto}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}article{{min-width:0}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}a{{color:#0756a6}}</style><h1>V122 실제 합성 source · all32 cue</h1><p>모든 cue의 중앙 offset에서 첫 geometry-feasible donor rank를 표시합니다. 없으면 no feasible pair로 유지합니다. 추가8 rank는 원래 cue 순서의 첫 feasible example입니다. 모델 출력·점수는 읽지 않았습니다.</p><p>청록: target bbox, 노랑: 원 cue uncertainty, 빨강: 원 reference. Full context는 원 native composite에 표시한 복사본이며 RGB96는 실제 저장된 무표시 입력을 nearest ×3 확대한 그림입니다. source-emulated q/L/context와 uncertainty를 바꾸지 않았습니다.</p><p>AI weak region/component cue visibility이며 native8 equality·pixel-owner·human gold가 아닙니다. TOMIE는 모두 하나의 development-stress episode, LOCO는 기존 fold를 유지합니다. cue·jitter·donor derivatives는 독립 데이터가 아닙니다. local alpha mass는96 grid의 bilinear weighted alpha sum(/9216)이고 cover/sham 일치를 주장하지 않습니다. 학습·성능·실제 배포 성공을 뜻하지 않습니다.</p><h2>All32 cues</h2><div class="grid">{cards_html}</div><h2>All8 donor ranks</h2><div class="grid">{ranks_html}</div><p><a href="SELECTION_BEFORE_VISUAL_PAYLOADS.json">선정 기록</a> · <a href="FIGURE_PROVENANCE.json">provenance</a></p></html>'''
    htmlpath = OUT / 'GALLERY_KO.html'
    with htmlpath.open('x') as f:
        f.write(gallery)
    write(OUT / 'FIGURE_PROVENANCE.json', dict(completed=True, corpus_seal=bind(seal_path),
          code=bind(Path(__file__)), executed_reconstruction_code=seal['code'], crops=outputs['INPUT_CROPS4896_INPUT_ONLY'],
          occurrences=outputs['OCCURRENCES4896_INPUT_ONLY'], attempts=outputs['FEASIBILITY256_GT_ONLY'],
          cue_metadata=bind(META / 'CUES32_INPUT_ONLY.jsonl'), cue_evidence=bind(META / 'CUES32_WEAK_EVIDENCE_GT_ONLY.jsonl'),
          selection=bind(OUT / 'SELECTION_BEFORE_VISUAL_PAYLOADS.json'), case_rows=provenance,
          marked_full_context_is_display_only=True, RGB96_saved_unmarked_bytes_exact=True,
          actual_model_score_weights_encoder_fit_current_V112_PnP_reads=0))
    write(OUT / 'RENDERING_RECEIPT.json', dict(completed=True, corpus_seal=bind(seal_path),
          code=bind(Path(__file__)), cue_cards=cuefiles, rank_cards=rankfiles, contacts=contactfiles,
          gallery=bind(htmlpath), provenance=bind(OUT / 'FIGURE_PROVENANCE.json'),
          all32_cues_kept=True, all8_donor_ranks_kept=True,
          no_feasible_cues=[c['cue_index'] for c in cuecases if c['donor_rank'] is None],
          no_feasible_donor_ranks=[c['donor_rank'] for c in rankcases if c['no_feasible_pair']],
          actual_unique_native_sources_read=len(originals), actual_unique_patch_files_read=len(patches),
          actual_visual_review_status='RENDERED_NOT_YET_ACTUALLY_VIEWED_BY_THIS_SCRIPT',
          model_encoder_head_fit_current_V112_PnP=0))
    print(json.dumps(dict(completed=True, receipt=bind(OUT / 'RENDERING_RECEIPT.json')), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-seal', type=Path, required=True)
    args = ap.parse_args()
    run(args.corpus_seal)
