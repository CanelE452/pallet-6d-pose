"""Concise saved PUBLIC C14 diagnostics. No encoder/head/fit/raw-image access."""
import argparse
import csv
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

H = Path(__file__).resolve().parent
V = H.parent
OUT = V / 'source_prediction_visual_review'
RESULTS = V / 'frozen_C14_source_baseline/saved_weak_evaluation/SOURCE_RESULTS.json'
HELPER = V / 'visual_review_preparation/render_corpus_fixed.py'
ROLE_NAMES = dict(ALL3='첫 중앙 known triplet 전체 정답', COVER_MISS='첫 중앙 COVER E→D 누락',
                  SHAM_FALSE_E='첫 중앙 SHAM D→E', BEFORE_FALSE_E='첫 중앙 BEFORE D→E')
STAGES = ('before', 'cover', 'sham')
STATE = {-1: 'U 라벨 미정', 0: 'D 보임', 1: 'E 외부 가림'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bind(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s]


def checked(b):
    if sha(b['path']) != b['sha256']:
        raise ValueError('Binding differs: ' + b['path'])
    return Path(b['path'])


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def helper():
    s = importlib.util.spec_from_file_location('V122_inert_visual_helpers', HELPER)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def typed_pairs(path):
    with Path(path).open() as f:
        rr = list(csv.DictReader(f))
    for r in rr:
        for k in ('query_index', 'cue_index', 'jitter_index', 'donor_rank'):
            r[k] = int(r[k])
        for s in STAGES:
            r[s+'_label'] = int(r[s+'_label'])
            r[s+'_prediction'] = int(r[s+'_prediction'])
            r[s+'_z'] = float(r[s+'_z'])
        for k in ('teacher_triplet_known_D_E_D', 'all3_correct'):
            r[k] = r[k] == 'True'
    if [(r['query_index'], r['donor_rank']) for r in rr] != [(q, k) for q in range(288) for k in range(8)]:
        raise ValueError('Original2304 pair order required')
    return rr


def selection(pairrows):
    centre = [r for r in pairrows if r['jitter_index'] == 4]
    tests = dict(ALL3=lambda r: r['teacher_triplet_known_D_E_D'] and r['all3_correct'],
                 COVER_MISS=lambda r: r['cover_label'] == 1 and r['cover_prediction'] == 0,
                 SHAM_FALSE_E=lambda r: r['sham_label'] == 0 and r['sham_prediction'] == 1,
                 BEFORE_FALSE_E=lambda r: r['before_label'] == 0 and r['before_prediction'] == 1)
    selected = {name: next((r for r in centre if test(r)), None) for name, test in tests.items()}
    anchor = next((selected[n] for n in ('COVER_MISS', 'SHAM_FALSE_E', 'BEFORE_FALSE_E', 'ALL3') if selected[n] is not None), centre[0])
    scales = [next(r for r in centre if r['cue_index'] == anchor['cue_index'] and r['donor_rank'] == k) for k in range(4)]
    wood = next((r for r in centre if r['donor_rank'] == 7 and r['cover_label'] == 1 and r['cover_prediction'] == 0), None)
    wood_cone = None if wood is None else next(r for r in centre if r['cue_index'] == wood['cue_index'] and r['donor_rank'] == 1)
    return selected, scales, wood, wood_cone


def indices(r):
    base = 17*r['query_index']; rank = r['donor_rank']
    return [base, base+1+2*rank, base+2+2*rank]


def card(r, title, occ, crops, vis):
    canvas = Image.new('RGB', (1140, 700), '#ffffff')
    d = ImageDraw.Draw(canvas)
    vis.wrapped(d, (16, 8), title, 23, width=1100)
    if r is None:
        vis.wrapped(d, (35, 190), '해당 사례 없음\n다른 offset·기준으로 대체하지 않음', 27, width=1060, maxlines=3)
        return canvas
    ids = indices(r)
    q = occ[ids[0]]
    vis.wrapped(d, (16, 47), f"cue{r['cue_index']:02d} · rank{r['donor_rank']} · {r['source']} / {r['raw_split']} · 중앙 offset[0,0]", 17)
    vis.wrapped(d, (16, 73), f"q={np.round(q['query_xy'], 3).tolist()}, L={q['L']:.3f}px · 같은 q8/context/L · raw logit0: D<0, E≥0", 16)
    for n, (stage, idx) in enumerate(zip(STAGES, ids)):
        x = 16+380*n
        row = occ[idx]
        if row['jitter_index'] != 4 or row['query_xy'] != q['query_xy'] or row['L'] != q['L'] or row['context_box_xyxy'] != q['context_box_xyxy']:
            raise ValueError('Central matched input changed')
        pred = r[stage+'_prediction']
        predtext = 'U 입력 보류' if pred < 0 else STATE[pred]
        vis.wrapped(d, (x, 105), stage.upper(), 20, width=356)
        vis.wrapped(d, (x, 137), f"weak {STATE[r[stage+'_label']]} → {predtext}", 16, width=355)
        vis.wrapped(d, (x, 165), f"zE={r[stage+'_z']:+.5f} · row{idx}", 16, width=355)
        raw = crops['rgb96'][idx].transpose(1, 2, 0)
        vis.paste_fit(canvas, raw, (x+28, 204), (288, 288), nearest=True)
        vis.wrapped(d, (x+28, 496), '무표시 실제 RGB96 ×3', 16, width=326)
        # Mark q only on a SEPARATE display copy, never on saved model input.
        marked = Image.fromarray(raw.copy()).resize((144, 144), Image.Resampling.NEAREST)
        md = ImageDraw.Draw(marked);c=(47.5+.5)*1.5-.5
        md.line((c-8,c,c+8,c), fill='#ff1542', width=2)
        md.line((c,c-8,c,c+8), fill='#ff1542', width=2)
        canvas.paste(marked, (x+100, 535))
        vis.wrapped(d, (x+16, 680), f"별도 q 표시 · valid {crops['valid96'][idx].mean():.3f}", 14, width=354)
    return canvas


def board(cards, title, vis):
    canvas = Image.new('RGB', (2280, 90+700*((len(cards)+1)//2)), '#f2f5f9')
    d=ImageDraw.Draw(canvas)
    vis.wrapped(d, (20, 8), title, 29, width=2220)
    vis.wrapped(d, (20, 51), '공개 conditional AUX source 진단 · native/human/pixel gold 및 현재 성능 아님 · 모델/학습 선택 금지', 20, width=2220)
    for i, im in enumerate(cards):
        canvas.paste(im, (1140*(i%2),90+700*(i//2)))
    return canvas


def percent(x): return '—' if x is None else f'{100*x:.1f}%'


def metric_graphic(results, vis):
    canvas=Image.new('RGB',(1640,1120),'#ffffff');d=ImageDraw.Draw(canvas)
    vis.wrapped(d,(20,10),'Frozen C14 · 공개 flat-pallet 합성 source baseline',28)
    vis.wrapped(d,(20,55),'D=보임, E=외부 가림, teacher U=라벨 미정. 중앙 offset과9개 correlated jitter를 분리합니다.',18)
    columns=(20,360,545,735,925,1100,1285,1450)
    for x, name in zip(columns,('source / offset','D 정답/분모','D recall','E 정답/분모','E recall','E precision','known 예측U','teacher U')):
        vis.wrapped(d,(x,108),name,16,width=175)
    y=145
    for group, allm in results['metrics'].items():
        label={'ALL::ALL':'ALL(개발 노출 source)','LOCO::TRAIN':'LOCO 기존 TRAIN','LOCO::VALIDATION':'LOCO 기존 VAL','TOMIE::TOMIE_DEVELOPMENT_STRESS':'TOMIE 별도 development stress'}[group]
        for offset in ('centre_offset0','all9_correlated'):
            m=allm[offset];cm=np.array(m['CM_D_E_rows_D_E_U_columns'])
            d.rectangle((12,y-2,1628,y+65),fill='#f3f6fa' if offset=='centre_offset0'else'#ffffff')
            values=(label+'\n'+('중앙[0,0]'if offset=='centre_offset0'else'all9 correlated'),f'{cm[0,0]}/{cm[0].sum()}',percent(m['D_recall']),f'{cm[1,1]}/{cm[1].sum()}',percent(m['E_recall']),percent(m['E_precision']),str(m['model_U_on_known']),str(m['teacher_U']))
            for x,value in zip(columns,values):vis.wrapped(d,(x,y),value,17,width=320 if x==20 else 170,maxlines=2)
            y+=70
    vis.wrapped(d,(20,728),'중앙 offset: donor rank별 known D/E/D triplet 전체 정답 (분모=geometry-feasible cue/rank)',21)
    for rank in range(8):
        rr=results['paired_by_rank'][f'ALL::ALL::R{rank}'];a,b=rr['centre_all3_correct'],rr['centre_known_triplets']
        yy=776+36*rank
        name=(f'cone {(.6,1.,1.4,2.)[rank]:.1f}×L0'if rank<4 else f'YCB object{(4,5,15,16)[rank-4]}')
        vis.wrapped(d,(22,yy),f'R{rank} {name}',17,width=260,maxlines=1)
        d.rectangle((310,yy+5,1210,yy+23),fill='#e5e9ef')
        if b:d.rectangle((310,yy+5,310+900*a/b,yy+23),fill='#2563ab'if rank<7 else'#b74738')
        vis.wrapped(d,(1240,yy),f'{a}/{b} '+(''if not b else percent(a/b)),17,width=350,maxlines=1)
    vis.wrapped(d,(20,1074),'BEFORE는 원래 visible cue, E는 controlled cover. 자연 외부가림·현재223 성능 또는 독립 heldout 검증으로 일반화하지 않습니다.',16,width=1590)
    return canvas


def run(results_path):
    if OUT.exists():raise FileExistsError('Preserve completed visualization; no automatic rerender')
    result=read(results_path)
    if not result.get('completed'):raise ValueError('Completed SOURCE_RESULTS required')
    evpath=Path(results_path).parent/'EVALUATION_SEAL.json';ev=read(evpath)
    if not ev.get('completed')or not ev.get('PASS'):raise ValueError('Saved evaluation incomplete')
    sealpath=checked(result['prediction_seal']);seal=read(sealpath)
    if not seal.get('completed')or not seal.get('PASS')or not seal['model_state_unchanged']:raise ValueError('Prediction incomplete')
    if seal['source_model_freeze']['sha256']!='4b2a8fdbd2037dd566aec44282aa84c9b3e408469d399b1a052cd571cf317b3f':raise ValueError('Frozen C14 expected, never old C13/current')
    corpus=read(checked(seal['corpus_seal']));occ=rows(checked(corpus['outputs']['OCCURRENCES4896_INPUT_ONLY']))
    pairrows=typed_pairs(checked(result['paired_table']))
    selected,scales,wood,cone=selection(pairrows)
    OUT.mkdir()
    write(OUT/'SELECTION_AFTER_SAVED_PREDICTIONS.json',dict(completed=True,source_results=bind(results_path),prediction_seal=result['prediction_seal'],
          rule='First original query_index/rank order at jitter4: known all3correct; COVER E→D; SHAM D→E; BEFORE D→E. Absent stays absent. Cone0..3 share first cover-miss cue, else next error/success/fixed0. Explicit woodrank7miss vs samecue cone1.',
          selected_roles=selected,cone_scale_counterparts=scales,first_woodrank7_cover_miss=wood,samecue_cone_rank1=cone,
          posthoc_diagnostic_only=True,model_threshold_training_selection=False,selection_precedes_crop_payload_reads=True))
    with np.load(checked(seal['predictions']),allow_pickle=False)as f:
        pred=np.array(f['prediction']);z=np.array(f['query_logit']);available=np.array(f['query_available']);ix=np.array(f['global_occurrence_index'])
    if not np.array_equal(ix,np.arange(4896))or not np.array_equal(pred,np.where(available,(z>=0).astype(np.int8),-1)):raise ValueError('Prediction sign/order mismatch')
    for r in pairrows:
        for stage,i in zip(STAGES,indices(r)):
            if pred[i]!=r[stage+'_prediction']or float(z[i])!=r[stage+'_z']:raise ValueError('Saved paired score mismatch')
    cb=corpus['outputs']['INPUT_CROPS4896_INPUT_ONLY']
    with np.load(checked(cb),allow_pickle=False)as f:crops={k:np.array(f[k],copy=True)for k in('rgb96','valid96','occurrence_index')}
    if crops['rgb96'].shape!=(4896,3,96,96)or not np.array_equal(crops['occurrence_index'],np.arange(4896)):raise ValueError('Exact RGB96/order required')
    shown=[r for r in selected.values()if r is not None]+scales+[r for r in(wood,cone)if r is not None]
    for i in {i for r in shown for i in indices(r)}:
        if hashlib.sha256(crops['rgb96'][i].tobytes()).hexdigest()!=occ[i]['rgb96_sha256']or hashlib.sha256(crops['valid96'][i].tobytes()).hexdigest()!=occ[i]['valid96_sha256']:raise ValueError('Saved crop bytes differ')
    vis=helper();files=[]
    def save(name,im):
        path=OUT/name;im.save(path);files.append(bind(path));return path
    maincards=[card(selected[k],ROLE_NAMES[k],occ,crops,vis)for k in ROLE_NAMES]
    save('SOURCE_SUCCESS_FAILURE_TRIPLETS.png',board(maincards,'고정 C14 · 첫 중앙 source 성공/실패 triplet',vis))
    save('CONE_SCALE_COUNTERPARTS.png',board([card(r,f"동일 cue cone rank{r['donor_rank']} / {(.6,1.,1.4,2.)[r['donor_rank']]}×L0",occ,crops,vis)for r in scales],'같은 cue · cone 크기0.6 / 1.0 / 1.4 / 2.0×L0 (실패U도 유지)',vis))
    save('SAME_CUE_CONE_VS_WOODBLOCK.png',board([card(wood,'첫 woodblock rank7 COVER 누락',occ,crops,vis),card(cone,'같은 cue · cone rank1 비교',occ,crops,vis)],'동일 cue / q / L: cone rank1 vs woodblock rank7 · 차이는 관찰이며 원인 증명 아님',vis))
    save('SOURCE_METRICS.png',metric_graphic(result,vis))
    cue_meta=rows(V/'input_preparation/CUES32_INPUT_ONLY.jsonl')
    existing=V/'fixed_corpus_visual_review';existing_receipt=read(existing/'RENDERING_RECEIPT.json')
    cuecards=existing_receipt['cue_cards'];links=[]
    for ci,b in enumerate(cuecards):
        rel=os.path.relpath(b['path'],OUT)
        links.append(f'<li><a href="{html.escape(rel)}">cue{ci:02d} {html.escape(cue_meta[ci]["source"])} / {html.escape(cue_meta[ci]["split"])} · 기존 context/unc card</a></li>')
    missing=', '.join(ROLE_NAMES[k]for k,r in selected.items()if r is None)or'없음'
    allm=result['metrics']['ALL::ALL'];centre=allm['centre_offset0'];cm=centre['CM_D_E_rows_D_E_U_columns']
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V122 source frozen C14</title><style>body{{font:18px/1.65 sans-serif;color:#152236;max-width:1640px;margin:28px auto;padding:0 20px}}img{{width:100%;height:auto}}a{{color:#0756a6}}nav{{position:sticky;top:0;background:white;padding:8px;border-bottom:1px solid #ddd}}li{{overflow-wrap:anywhere}}.links{{columns:2}}@media(max-width:700px){{.links{{columns:1}}}}</style><h1>고정 C14 · 공개 flat-pallet source 진단</h1><p>새 학습 없이 저장된 C14(logit0)를 평가했습니다. 중앙 offset 보임 D {cm[0][0]}/{sum(cm[0])}, 외부 가림 E {cm[1][1]}/{sum(cm[1])}; teacher U {centre['teacher_U']}, known 예측U {centre['model_U_on_known']}. 없는 유형: {html.escape(missing)}.</p><p>D=보임, E=외부 가림, U=라벨 미정입니다. U 라벨과 입력 보류는 구분합니다. Conditional AI weak cue/component visibility와 controlled native-RGB cutout을 사용하는 source 진단이며 native/human/pixel gold·현재 성능이 아닙니다. 32cue는9 native images에서 왔고9 jitters·8donor/rank derivatives는 독립 데이터가 아닙니다. LOCO는 기존 개발 노출 source, TOMIE는 하나의 development-stress episode입니다.</p><nav><a href="#metrics">수치</a> · <a href="#examples">성공/실패</a> · <a href="#wood">동일 cue cone/wood</a> · <a href="#scales">cone scale</a> · <a href="#all32">기존 all32 context</a></nav><h2 id="metrics">저장 source 수치</h2><img src="SOURCE_METRICS.png" alt="source metrics"><h2 id="examples">첫 성공·실패 사례</h2><p>예측 저장 이후 원래 query/rank 순서의 첫 해당 사례만 표시했습니다. 없는 유형은 대체하지 않습니다. 무표시 RGB96는 실제 저장 입력, q crosshair는 별도 표시용 복사본입니다. 원본 q/L/labels는 바꾸지 않았습니다.</p><img src="SOURCE_SUCCESS_FAILURE_TRIPLETS.png" alt="source success failure"><h2 id="wood">같은 cue: cone과 woodblock</h2><p>서로 다른 donor에 대한 응답 차이를 보여주며 단일 인과 기전이나 전체 transfer 성공을 증명하지 않습니다. 간단한 cone 합성의 성공을 자연 외부가림이나 기존 current 결과로 일반화하지 않습니다.</p><img src="SAME_CUE_CONE_VS_WOODBLOCK.png" alt="same cue cone and wood"><h2 id="scales">같은 cue cone rank0–3</h2><p>크기는 prospective 고정0.6/1.0/1.4/2.0×L0입니다. 기하 실패는 teacherU 그대로 표시하며 다른 사례로 교체하지 않습니다.</p><img src="CONE_SCALE_COUNTERPARTS.png" alt="cone scale counterparts"><h2 id="all32">기존 all32 full context·uncertainty 카드</h2><ul class="links">{''.join(links)}</ul><p>이 시각화는 사후 진단용입니다. 모델·임계값·학습 row 선택, fit, 새로운 current/V112/PnP 실행을 하지 않았습니다. <a href="SELECTION_AFTER_SAVED_PREDICTIONS.json">투명한 선정 기록</a></p></html>'''
    hp=OUT/'READOUT_KO.html'
    with hp.open('x')as f:f.write(page)
    write(OUT/'VISUALIZATION_RECEIPT.json',dict(completed=True,code=bind(Path(__file__)),source_results=bind(results_path),evaluation_seal=bind(evpath),prediction_seal=result['prediction_seal'],
          actual_input_crops=cb,selection=bind(OUT/'SELECTION_AFTER_SAVED_PREDICTIONS.json'),figures=files,HTML=bind(hp),all32_existing_context_cards=cuecards,
          actual_rendered_not_yet_reviewed=True,posthoc_diagnostic_only=True,raw_RGB_model_weights_encoder_head_fit_current_V112_PnP_reads=0))
    print(json.dumps(dict(completed=True,receipt=bind(OUT/'VISUALIZATION_RECEIPT.json')),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-results',type=Path,default=RESULTS);a=ap.parse_args();run(a.source_results)
