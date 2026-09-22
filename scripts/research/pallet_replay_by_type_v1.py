"""Separate real-data adaptation per material; identical synthetic replay control."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from scripts.research import pallet_replay_clean19_v1 as P

ROOT=P.ROOT
DOC=ROOT/'_docs/experiments/pallet_replay_by_type_v1'
RAW=ROOT/'data/pallet/results/pallet_replay_by_type_v1'
OUT=ROOT/'outputs/pallet_replay_by_type_v1'
OLD_DOC=P.DOC;OLD_RAW=P.RAW


def configure(material):
    P.DOC=DOC/material;P.RAW=RAW/material;P.OUT=OUT/material


def prepare():
    assert not DOC.exists() and not RAW.exists() and not OUT.exists()
    original=P.read(OLD_DOC/'SPLIT.json');inputs=P.read(OLD_DOC/'INPUT_LOCK.json')
    for b in inputs['files']:P.N.F.verify(b)
    for material,ntrain,neval in [('plastic',10,184),('wood',9,116)]:
        configure(material)
        for path in (P.DOC,P.RAW,P.OUT):path.mkdir(parents=True)
        split=copy.deepcopy(original)
        split['train']=[r for r in original['train'] if r['object_type']==material]
        split['evaluation']=[r for r in original['evaluation'] if r['object_type']==material]
        assert len(split['train'])==ntrain and len(split['evaluation'])==neval
        P.save(P.DOC/'SPLIT.json',split)
        P.save(P.RAW/'BASELINE_PREDICTIONS.json',P.read(OLD_RAW/'BASELINE_PREDICTIONS.json'))
        protocol=copy.deepcopy(P.read(OLD_DOC/'PROTOCOL.json'))
        protocol.update(train=ntrain,evaluation=neval,material=material,
            purpose='Separate real supervision by material; compare on exact previous material evaluation subset',
            synthetic_replay='unchanged historical synthetic replay order for both arms; not material-filtered',
            real_exposures=2400,synthetic_exposures=2400,steps=300,
            comparison_caveat='same optimizer steps/real exposures per model, so per-image exposure exceeds mixed training; not exposure-matched per type',
            primary_metric=f'{material} existing-reference PCK10',old_mixed_checkpoint_used_as_init=False)
        P.save(P.DOC/'PROTOCOL.json',protocol)
        files=inputs['files']+[P.bind(Path(__file__)),P.bind(P.DOC/'SPLIT.json'),P.bind(P.DOC/'PROTOCOL.json'),P.bind(P.RAW/'BASELINE_PREDICTIONS.json')]
        P.save(P.DOC/'INPUT_LOCK.json',dict(files=files))
    print('PREPARED plastic10/184 wood9/116; fresh synthetic PRIOR1 each; same synthetic replay',flush=True)


def run(material):
    configure(material)
    print('START_MATERIAL',material,flush=True)
    P.train()
    P.infer()
    score(material)


def score(material):
    configure(material);P.verify()
    lock=P.read(P.DOC/'PREDICTION_LOCK.json');P.N.F.verify(lock['predictions'])
    assert lock['frozen_before_scoring']
    records=P.read(P.DOC/'SPLIT.json')['evaluation']
    predictions=P.read(P.RAW/'PREDICTIONS.json')['predictions']
    truth=P.read(OLD_RAW/'TRUTH_FOR_DISPLAY_ONLY.json')
    prior=P.read(OLD_RAW/'FRAME_METRICS.json')
    metrics={a:{r['id']:prior[a][r['id']] for r in records} for a in ('R0','N2_DIM_ONLY','N3_DIM_SYM','PRIOR1')}
    metrics['MIXED19_REPLAY']={r['id']:prior['CLEAN19_REPLAY'][r['id']] for r in records}
    for arm,name in [('CLEAN19_REPLAY','TYPE_REPLAY'),('CLEAN19_REPLAY_CAP8','TYPE_REPLAY_CAP8')]:
        metrics[name]={}
        for r in records:
            fid=r['id'];t=truth[fid];base=prior['R0'][fid];p=predictions[arm][fid]
            candidate=P.C.selected(p);q=np.full((9,2),np.nan) if candidate is None else candidate['keypoints_xy']
            P.assert_preserved(P.read(P.RAW/'BASELINE_PREDICTIONS.json')['R0'][fid],p)
            m=P.M.measure(q,t['gt'],t['valid'],t['permutations'],t['hw'],base['matched'],base['detected'])
            assert m['canonical_valid']==base['canonical_valid']
            metrics[name][fid]=dict(id=fid,**m)
    summary={}
    groups={'ALL':[r['id'] for r in records],**{c:[r['id'] for r in records if r['severity']==c] for c in P.SEVERITIES}}
    for group,ids in groups.items():
        summary[group]={}
        for arm,mm in metrics.items():
            rows=[mm[i] for i in ids];s=P.M.summary(rows)
            s.update(correct10=sum(e<=10 for r in rows for e in r.get('errors',[])),gross20_count=sum(e>20 for r in rows for e in r.get('errors',[])))
            summary[group][arm]=s
        assert len({(s['corners'],s['matched'],s['total_frames']) for s in summary[group].values()})==1
    P.save(P.RAW/'FRAME_METRICS.json',metrics);P.save(P.DOC/'RESULTS.json',summary)
    P.save(P.DOC/'AUDIT.json',dict(complete=True,material=material,
        real_materials=sorted({r['object_type'] for r in P.read(P.DOC/'SPLIT.json')['train']}),
        same_evaluation_as_mixed=True,checkpoint='step300',synthetic_only_initialization=True,
        mixed_checkpoint_not_used=True,steps=300,same_session_adaptation=True))
    print('MATERIAL_RESULT',material,{a:s['PCK']['10'] for a,s in summary['ALL'].items()},flush=True)


def report():
    configure('plastic')
    result={m:P.read(DOC/m/'RESULTS.json') for m in ('plastic','wood')}
    lines=['# 종류별 Clean 실사 + 합성 replay 결과','',
        '플라스틱10장과 목재9장을 섞지 않고 보정기2개를 각각 합성전용PRIOR1에서 시작했다. seed1/300step, 실사8+합성8, BN통계고정. 실사 감독은 해당 종류만이며 합성 replay 순서는 혼합 실험과 동일하게 유지했다.', '',
        '평가목록도 혼합학습과 동일: 플라스틱184장·목재116장. 학습이미지 중복0, 같은 촬영 세션은 사용자 요청으로 허용했다. 새환경 일반화 주장이 아니다.', '',
        '모델당300step이므로 종류별 모델은 해당 종류의 실사 노출이 혼합모델보다 많다. 따라서 재료분리만의 순수 인과효과로 해석하지 않는다.', '']
    for material,groups in result.items():
        lines+=['## '+material,'','| 난도 | 모델 | PCK10% | PCK20% | observed med px | P90 px | >20 코너 |','|---|---|---:|---:|---:|---:|---:|']
        for group,arms in groups.items():
            for arm,s in arms.items():
                fmt=lambda x:'—' if x is None else f'{x:.2f}'
                lines.append(f'| {group} | {arm} | {fmt(None if s["PCK"]["10"] is None else 100*s["PCK"]["10"])} | {fmt(None if s["PCK"]["20"] is None else 100*s["PCK"]["20"])} | {fmt(s["matched_pooled_corner8_median_px"])} | {fmt(s["matched_pooled_corner8_P90_px"])} | {s["gross20_count"]} |')
        lines.append('')
    lines+=['cap8은 사전등록 보조출력이다. 모델을 평가 결과로 자동 선택·교체하지 않았다. 기존 모델/GT/319장 수치 보존. 이번 작업은 수동지도 보정기 적응이며 self-training이 아니다.']
    # P.save is scoped to material, so use explicit exclusive-create for root report.
    with (DOC/'RESULTS_KO.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print('REPORT',DOC/'RESULTS_KO.md',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','plastic','wood','report']);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='report':report()
    else:run(a.stage)
