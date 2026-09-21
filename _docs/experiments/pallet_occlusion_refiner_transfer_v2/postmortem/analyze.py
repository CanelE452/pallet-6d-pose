"""Frozen-output analysis only. No optimizer, inference, label edits, or git writes."""
import argparse
import copy
import csv
import html
import io
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_occlusion_refiner_transfer_v2 import run as C

OUT = Path(__file__).resolve().parent
ARMS = ('R0', 'N2', 'N3', 'POSEFIX_SYNTH', 'REPLAY', 'A10', 'A11')
SHOW = ('R0', 'N2', 'REPLAY', 'A10', 'A11')
POP = 'PRIMARY_OCC96'
REVIEW = ROOT/'data/evaluation/pallet_eval_v1/review'


def save(name, obj):
    p = OUT/name
    p.parent.mkdir(parents=True, exist_ok=True)
    value = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if p.exists():
        assert p.read_text() == value, f'Completed artifact cannot change: {p}'
    else:
        with p.open('x') as f: f.write(value)


def csvsave(name, rows):
    assert rows
    keys = list(dict.fromkeys(k for r in rows for k in r))
    s = io.StringIO(); w = csv.DictWriter(s, fieldnames=keys); w.writeheader()
    for row in rows:
        w.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in row.items()})
    save(name, s.getvalue())


def transition(a, b, threshold=10):
    return ('GOOD' if a <= threshold else 'BAD')+'->'+('GOOD' if b <= threshold else 'BAD')


def band(error, matched):
    if not matched: return 'B5_DETECTION_MATCH_FAILURE'
    for i, upper in enumerate((5,10,20,40)):
        if error <= upper:return f'B{i}'
    return 'B4'


def stats(values):
    a=np.asarray([v for v in values if v is not None],float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
                p10=float(np.quantile(a,.1)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None)


def metadata_sources():
    human={}; auto={}; paths=[]
    for prefix in ('DAYTIME_VISIBILITY','COVERAGE_GAP'):
        path=REVIEW/(prefix+'_AMENDMENTS.json');d=C.read(path);paths.append(path)
        assert 'prediction-blinded' in d['protocol']
        for fid, r in d['frames'].items():
            for j,state in r.items():
                if not j.isdigit():continue
                key=(fid.replace('__',':',1),int(j))
                assert key not in human or human[key]['state']==state
                human[key]=dict(state=state,path=str(path.relative_to(ROOT)),protocol=d['protocol'])
    for name in ('DAYTIME_OCCLUSION_REVIEW_QUEUE.csv','COVERAGE_GAP_OCCLUSION_QUEUE.csv'):
        path=REVIEW/name; paths.append(path)
        for r in csv.DictReader(path.open()):
            key=(r['frame_id'].replace('__',':',1),int(r['kp_index']))
            assert key not in auto or auto[key]['row']==r
            auto[key]=dict(row=r,path=str(path.relative_to(ROOT)))
    return human,auto,paths


def visibility(entry, human, auto):
    """Only explicitly reviewed subtype is verified; proxy evidence is separate."""
    state='UNKNOWN'; source='UNVERIFIED_LEGACY'; generic='UNKNOWN'; verified=False
    if human and human['state'] in ('v','o','t'):
        expected={'v':(2,'visible'),'o':(1,'occluded'),'t':(0,'truncated')}[human['state']]
        assert (entry['visibility'],entry['reason'])==expected, 'Human amendment/current annotation conflict'
        generic={'v':'VISIBLE','o':'OCCLUDED_UNSPECIFIED','t':'TRUNCATED/OFFSCREEN'}[human['state']]
        state=generic if generic!='OCCLUDED_UNSPECIFIED' else 'UNKNOWN'
        source='HUMAN_VISIBILITY_REVIEW';verified=True
    elif entry.get('in_frame') is False:
        state='TRUNCATED/OFFSCREEN';source='STORED_OFFSCREEN';generic=state;verified=True
    proxy=auto['row']['final_auto_status'] if auto else 'MISSING'
    # Even human 'o' + proxy self-visible does not constitute manual external subtype GT.
    return dict(verified_visibility_class=state,visibility_source=source,manual_visibility=generic,
                manual_reviewed=bool(human and human['state'] in ('v','o','t')),generic_verified=verified,
                stored_visibility=entry.get('visibility'),stored_reason=entry.get('reason'),
                stored_in_frame=entry.get('in_frame'),auto_proxy_state=proxy,
                auto_self_hyp_A=auto['row'].get('self_visibility_hyp_A') if auto else None,
                auto_self_hyp_B=auto['row'].get('self_visibility_hyp_B') if auto else None,
                external_subtype_verified=False,human_record=human,auto_record=auto)


def prepare():
    assert not (OUT/'INPUTS.json').exists(), 'Analysis already initialized'
    protocol=C.read(C.DOC/'E2_PROTOCOL.json');ids=protocol['populations'][POP]
    assert len(ids)==93 and len(set(ids))==93
    old=C.read(C.V.RAW/'FROZEN_PREDICTIONS.json');pred=old['predictions']
    lock=C.read(C.DOC/'E2_PREDICTIONS_LOCK.json');C.verify(lock['baseline'])
    for a in C.ARMS:
        C.verify(lock['predictions'][a]);C.verify(lock['checkpoints'][a])
        if a in ARMS: pred[a]=C.read(C.ROOT/lock['predictions'][a]['path'])['predictions']
    bindings=C.read(C.DOC/'DATA_ROLE_MANIFEST.json')['checkpoint_bindings']
    for b in bindings.values():C.verify(b)
    metrics=C.read(C.RAW/'E2_FRAME_METRICS.json');truth=C.read(C.V.RAW/'E1_FRAME_METRICS.json')['truth_for_display_only']
    objects=C.N.C.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json'
    perms={r['object_type']:r['permutations'] for r in C.read(objects)['objects']}
    human,auto,vispaths=metadata_sources();records={r['id']:r for r in protocol['eval_records']}
    _,conditions=C.V.condition_map(); rows=[];ann_paths=[];checks=0
    for fid in ids:
        r=records[fid];C.verify(r['annotation']);annpath=ROOT/r['annotation']['path'];ann_paths.append(annpath)
        ann=C.read(annpath); obj=ann['objects'][0]; t=truth[fid];entries=obj['keypoint_annotations']
        np.testing.assert_allclose(np.asarray([q['xy'] for q in entries])[:8][t['valid'][:8]],np.asarray(t['gt'])[:8][t['valid'][:8]],atol=1e-6)
        b=C.P.top(pred['R0'][fid]);bb=np.array(b['box_xyxy']) if b else None
        indices={}
        for a in ARMS:
            assert fid in pred[a] and fid in metrics[a]
            C.assert_preserved(pred['R0'][fid],pred[a][fid]);m=metrics[a][fid];p=C.P.top(pred[a][fid])
            assert m['canonical_valid']==t['valid'][:8]
            calc=C.V.M.measure(p['keypoints_xy'] if p else np.full((9,2),np.nan),t['gt'],t['valid'],perms[r['object_type']],[480,640],m['matched'],m['detected'])
            assert calc['branch']==m['branch']
            np.testing.assert_allclose([x for x in calc['canonical_errors'] if x is not None],[x for x in m['canonical_errors'] if x is not None],atol=1e-7,rtol=0)
            indices[a]=np.argsort(perms[r['object_type']][m['branch']]); checks+=1
        for j,v in enumerate(t['valid'][:8]):
            if not v:continue
            row=dict(frame_id=fid,session_id=r['session'],object_type=r['object_type'],corner_id=j,
                     gt_xy=t['gt'][j],target_provenance=entries[j].get('source'),gt_object_source=obj.get('gt_source'),
                     migration_status=obj.get('migration_status'),manual_review_reasons=obj.get('manual_review_reasons'),
                     image=r['image'],annotation=r['annotation'],
                     image_conditions={k:conditions.get(r['image']['path'],{}).get(k) for k in ('occlusion','truncation','usage_role')},
                     **visibility(entries[j],human.get((fid,j)),auto.get((fid,j))))
            for a in ARMS:
                m=metrics[a][fid];p=C.P.top(pred[a][fid]);native=int(indices[a][j]);xy=p['keypoints_xy'][native] if p else None
                error=m['canonical_errors'][j]
                geometric=float(np.linalg.norm(np.array(xy)-t['gt'][j])) if xy is not None else None
                if m['matched']:assert abs(geometric-error)<1e-6
                else:assert error==800.
                row[a]=dict(xy=xy,native_index=native,error_px=error,geometric_distance_px=geometric,branch=m['branch'],
                            detected=m['detected'],matched=m['matched'],PCK10=error<=10,PCK20=error<=20,
                            raw_confidence_same_native=b['keypoints_conf'][native] if b else None,
                            native_correction_px=float(np.linalg.norm(np.array(xy)-np.array(b['keypoints_xy'][native]))) if b and xy else None)
            x,y=row['A10']['error_px'],row['A11']['error_px'];initial=row['R0']['error_px']
            row.update(transition10=transition(x,y),transition20=transition(x,y,20),band=band(initial,row['R0']['matched']),
                       hard_recovery=initial>20 and y<=10,good_damage=initial<5 and y>10,
                       medium_damage=5<=initial<=20 and x<=10 and y>10,
                       bbox_area=float(np.prod(bb[2:]-bb[:2])) if b else None,
                       bbox_aspect=float((bb[2]-bb[0])/(bb[3]-bb[1])) if b else None,
                       bbox_center=((bb[:2]+bb[2:])/2).tolist() if b else None,
                       bbox_xyxy=bb.tolist() if b else None,
                       R0_kp_confidence=row['R0']['raw_confidence_same_native'],
                       R0_reliable_point_count=sum(c>=.5 and np.isfinite(p).all() and 0<=p[0]<640 and 0<=p[1]<480 for c,p in zip(b['keypoints_conf'][:8],b['keypoints_xy'][:8])) if b else 0,
                       verified_visible_point_count=sum(visibility(entries[k],human.get((fid,k)),auto.get((fid,k)))['verified_visibility_class']=='VISIBLE' for k in range(8)),
                       self_occlusion_PnP_available=None,
                       canonical_disagreement_px=float(np.linalg.norm(np.array(row['A10']['xy'])-row['A11']['xy'])) if row['A10']['xy'] and row['A11']['xy'] else None,
                       branch_switch_A10_A11=row['A10']['branch']!=row['A11']['branch'])
            if b:
                pa=C.P.top(pred['A10'][fid]);pb=C.P.top(pred['A11'][fid]);row['native_disagreement_px']=float(np.linalg.norm(np.array(pa['keypoints_xy'][indices['R0'][j]])-pb['keypoints_xy'][indices['R0'][j]]))
            else:row['native_disagreement_px']=None
            rows.append(row)
    assert len(rows)==713 and len({(r['frame_id'],r['corner_id']) for r in rows})==713
    table={str(t):dict(Counter(r['transition'+str(t)] for r in rows)) for t in (10,20)}
    totals={a:sum(r[a]['PCK10'] for r in rows) for a in ARMS}
    assert totals['A10']==356 and totals['A11']==350
    assert table['10'].get('GOOD->BAD',0)-table['10'].get('BAD->GOOD',0)==6
    original=C.read(C.DOC/'E2_REAL_RESULTS.json')['summary'][POP]
    for a in ARMS:
        for t in (10,20):assert abs(sum(r[a]['error_px']<=t for r in rows)/len(rows)-original[a]['PCK'][str(t)])<1e-12
    protected=[p for d in (C.DOC,C.V.DOC) for p in d.iterdir() if p.is_file()]
    protected += [C.RAW/'E2_FRAME_METRICS.json',C.V.RAW/'E1_FRAME_METRICS.json',C.V.RAW/'FROZEN_PREDICTIONS.json',objects,
                  ROOT/'scripts/annotate/apply_visibility_amendments.py',ROOT/'scripts/annotate/migrate_real_gt_v2.py',
                  ROOT/'data/evaluation/pallet_eval_v1/manifests/frames.csv']+vispaths+ann_paths
    protected += [C.ROOT/b['path'] for b in lock['predictions'].values()]+[C.ROOT/b['path'] for b in lock['checkpoints'].values()]+[C.ROOT/b['path'] for b in bindings.values()]
    protected += [p for d in (ROOT/'scripts/research/pallet_occlusion_refiner_transfer_v1',ROOT/'scripts/research/pallet_occlusion_refiner_transfer_v2') for p in d.glob('*.py')]
    # Parse/read required reports and all machine-readable artifacts, including source probes.
    for p in protected:
        if p.suffix=='.json':C.read(p)
        elif p.suffix in ('.md','.py'):p.read_text()
    save('INPUTS.json',dict(status='[확인]',HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        bindings=[C.bind(p) for p in sorted(set(protected))],frames=ids,scoring_parity_checks=checks,
        physical_identity='Native point i maps to canonical GT permutation[branch][i]; inverse permutation used for canonical join.',
        checkpoint_provenance={**bindings,**{a:lock['checkpoints'][a] for a in ('A10','A11')}},
        prediction_provenance=dict(baselines=lock['baseline'],A10=lock['predictions']['A10'],A11=lock['predictions']['A11'])))
    save('D1_TRANSITIONS.json',dict(status='[확인]',rows=rows,frames=93,corners=713,transition=table,correct10=totals))
    flat=[]
    for r in rows:
        f={k:v for k,v in r.items() if k not in ARMS}
        for a in ARMS:
            for k,v in r[a].items():f[a+'_'+k]=v
        flat.append(f)
    csvsave('D1_TRANSITIONS.csv',flat)
    save('PRECHECK.md',f'''# 입력 감사

[확인] HEAD `{subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}`, main. 새 학습/추론 없이 기존 frozen prediction만 사용했다. 전체 입력 SHA와 model checkpoint provenance는 [INPUTS](INPUTS.json)에 있다.

[확인] 모든 7개 후보의 동일 93-frame 집합, 동일 713 canonical GT 코너를 join했다. {checks}개 frame×candidate를 기존 metric과 다시 계산해 대조했다. whole-object 승인 대칭 한 분기만 사용했다. A10/A11 10px 정답 {totals['A10']}/{totals['A11']}, 차이 {totals['A10']-totals['A11']}개.

[확인] E2_FRAME_METRICS는 문서 폴더가 아닌 `data/pallet/results/pallet_occlusion_refiner_transfer_v2/E2_FRAME_METRICS.json`에 있다. A10/A11 freeze lock, 원본 model SHA, 현재 annotation SHA를 검사했다.

[확인] 매칭 실패 프레임 {sum(not metrics['R0'][f]['matched'] for f in ids)}개, 그 유효 코너 {sum(not r['R0']['matched'] for r in rows)}개를 삭제하지 않고 원래 800px 벌점으로 유지했다. 저장 좌표의 기하 거리와 채점 벌점은 CSV에서 별도 필드다.

[확인] 좌표 출처는 `{dict(Counter(r['target_provenance'] for r in rows))}`이다. GT object의 manual 표기를 코너별 manual 좌표 증명으로 사용하지 않았다. 비초록 legacy reference이며 기존 DEV 재사용이다.

[확인] 가시성은 기존 human amendment와 auto queue를 추적했다. annotation reason만으로 external/self를 구분하지 않는다. 사람의 generic occluded는 UNKNOWN subtype으로 남기며 auto self/visible은 proxy로 별도 집계한다. 수동 검토된 가시성과 좌표 provenance는 서로 다른 속성이다.

[확인] 기존 E2 decision/report 및 checkpoint는 수정하지 않았다. source clean/stress 결과도 기존 파일에서 읽었으며 새 평가를 선택해 과거 PARTIAL을 변경하지 않는다.
''')
    print('PRECHECK_OK',table,flush=True)


def subset_summary(rows):
    return dict(count=len(rows),frames=len({r['frame_id'] for r in rows}),
                models={a:dict(PCK10=sum(r[a]['PCK10'] for r in rows)/len(rows) if rows else None,
                    PCK20=sum(r[a]['PCK20'] for r in rows)/len(rows) if rows else None,
                    error=stats([r[a]['error_px'] for r in rows])) for a in SHOW},
                gained=sum(r['transition10']=='BAD->GOOD' for r in rows),lost=sum(r['transition10']=='GOOD->BAD' for r in rows),
                hard_recovery=sum(r['hard_recovery'] for r in rows),good_damage=sum(r['good_damage'] for r in rows),medium_damage=sum(r['medium_damage'] for r in rows))


def analyze():
    d=C.read(OUT/'D1_TRANSITIONS.json');rows=d['rows'];ids=C.read(OUT/'INPUTS.json')['frames']
    bands={}
    for name in ('B0','B1','B2','B3','B4','B5_DETECTION_MATCH_FAILURE'):
        rr=[r for r in rows if r['band']==name];delta=np.array([r['A11']['error_px']-r['A10']['error_px'] for r in rr])
        bands[name]=dict(**subset_summary(rr),improved=int((delta<0).sum()),worsened=int((delta>0).sum()),
            improved5=int((delta<=-5).sum()),worsened5=int((delta>=5).sum()),improved10=int((delta<=-10).sum()),worsened10=int((delta>=10).sum()))
    save('D2_ERROR_BANDS.json',bands)
    csvsave('D2_ERROR_BANDS.csv',[dict(band=k,**v) for k,v in bands.items()])
    visibility_stats={key:{str(v):subset_summary([r for r in rows if r[key]==v]) for v in sorted({r[key] for r in rows},key=str)}
        for key in ('verified_visibility_class','manual_visibility','auto_proxy_state','stored_reason')}
    visibility_stats.update(status='EXTERNAL_OCCLUSION_CORNER_RECOVERY_UNVERIFIED',
        verified_external_hard_recovery=sum(r['hard_recovery'] and r['external_subtype_verified'] for r in rows),
        external_hard_recovery_unknown=sum(r['hard_recovery'] and not r['external_subtype_verified'] for r in rows),
        manual_reviewed_corners=sum(r['manual_reviewed'] for r in rows),
        note='0 verified external is absence of subtype evidence, not evidence of no external recovery. Stored/proxy groups are NOT manual external GT.')
    save('D3_VISIBILITY_STATS.json',visibility_stats)
    metrics=C.read(C.RAW/'E2_FRAME_METRICS.json')
    selected={f:min(ARMS,key=lambda a:metrics[a][f]['frame_mean_px']) for f in ids}
    paired={f:min(('A10','A11'),key=lambda a:metrics[a][f]['frame_mean_px']) for f in ids}
    ties={f:[a for a in ARMS if abs(metrics[a][f]['frame_mean_px']-metrics[selected[f]][f]['frame_mean_px'])<1e-9] for f in ids}
    frame_rows=[metrics[selected[f]][f] for f in ids]
    oracle=C.V.M.summary(frame_rows);pair_oracle=C.V.M.summary([metrics[paired[f]][f] for f in ids])
    corner_rows=[];chosen_corners={}
    for f in ids:
        rr=[r for r in rows if r['frame_id']==f];choose=[min(ARMS,key=lambda a:r[a]['error_px']) for r in rr]
        e=[r[a]['error_px'] for r,a in zip(rr,choose)];cr=copy.deepcopy(metrics['R0'][f]);cr['errors']=e
        cr['observed_errors']=e if cr['matched'] else [];cr['frame_mean_px']=float(np.mean(e));cr['E_sym']=float(np.mean(e)/800)
        corner_rows.append(cr);chosen_corners[f]=[dict(corner_id=r['corner_id'],arm=a) for r,a in zip(rr,choose)]
    secondary=C.V.M.summary(corner_rows)
    baselines={a:C.V.M.summary([metrics[a][f] for f in ids]) for a in ARMS}
    all_fail=sum(all(not r[a]['PCK10'] for a in ARMS) for r in rows)
    d4=dict(status='[확인] DIAGNOSTIC_ONLY',rule='Minimum whole-object frame mean error. Unlike old E1 PCK-first oracle; no historical artifact changed.',
        arms=list(ARMS),baselines=baselines,ORACLE_FRAME=oracle,ORACLE_CORNER=secondary,A10_A11_FRAME_ORACLE=pair_oracle,
        headroom={a:100*(oracle['PCK']['10']-baselines[a]['PCK']['10']) for a in ('N2','A10','A11')},
        secondary_headroom={a:100*(secondary['PCK']['10']-baselines[a]['PCK']['10']) for a in ('N2','A10','A11')},
        chosen_frames=selected,secondary_chosen_corners=chosen_corners,
        deterministic_best_frequency=dict(Counter(selected.values())),tie_candidates=ties,
        unique_best_frequency=dict(Counter(t[0] for t in ties.values() if len(t)==1)),
        tied_best_frequency={a:sum(a in t for t in ties.values()) for a in ARMS},
        A10_A11_best_frequency=dict(Counter(paired.values())),
        A11_lower_frame_error=sum(metrics['A11'][f]['frame_mean_px']<metrics['A10'][f]['frame_mean_px'] for f in ids),
        A10_lower_frame_error=sum(metrics['A10'][f]['frame_mean_px']<metrics['A11'][f]['frame_mean_px'] for f in ids),
        both_equal_frame_error=sum(metrics['A10'][f]['frame_mean_px']==metrics['A11'][f]['frame_mean_px'] for f in ids),
        all_candidates_fail_PCK10=all_fail,
        A11_unique_best_frames={f:dict(R0_hard_corners=sum(r['R0']['error_px']>20 for r in rows if r['frame_id']==f),A11_hard_recovery=sum(r['hard_recovery'] for r in rows if r['frame_id']==f)) for f,t in ties.items() if t==['A11']},
        secondary_not_deployable=True,mean_error_not_PCK_optimal=True)
    save('D4_ORACLE_HEADROOM.json',d4)
    features=('bbox_area','bbox_aspect','R0_kp_confidence','R0_reliable_point_count','verified_visible_point_count','canonical_disagreement_px','native_disagreement_px')
    clusters={}
    for group in ('GOOD->GOOD','GOOD->BAD','BAD->GOOD','BAD->BAD'):
        rr=[r for r in rows if r['transition10']==group]
        clusters[group]=dict(**subset_summary(rr),features={k:stats([r[k] for r in rr]) for k in features},
            A11_native_correction=stats([r['A11']['native_correction_px'] for r in rr]),
            corners=dict(Counter(str(r['corner_id']) for r in rr)),sessions=dict(Counter(r['session_id'] for r in rr)),
            object_types=dict(Counter(r['object_type'] for r in rr)),verified_visibility=dict(Counter(r['verified_visibility_class'] for r in rr)),
            proxy_visibility=dict(Counter(r['auto_proxy_state'] for r in rr)),
            conditions=dict(Counter(json.dumps(r['image_conditions'],sort_keys=True) for r in rr)),
            bbox_centers=dict(x=stats([r['bbox_center'][0] if r['bbox_center'] else None for r in rr]),y=stats([r['bbox_center'][1] if r['bbox_center'] else None for r in rr])))
    # Fixed descriptive bins, not thresholds for deployment or tuning.
    bins={}
    for field in ('native_disagreement_px','A11_native_correction'):
        bins[field]={}
        for low,hi in ((0,5),(5,10),(10,20),(20,float('inf'))):
            def val(r):return r['A11']['native_correction_px'] if field=='A11_native_correction' else r[field]
            rr=[r for r in rows if val(r) is not None and low<=val(r)<hi]
            s=subset_summary(rr);good=sum(r['R0']['error_px']<5 for r in rr)
            s['good_denominator']=good;s['good_damage_rate']=s['good_damage']/good if good else None
            bins[field][f'[{low},{hi})']=s
    frame_features=[]
    for f in ids:
        rr=[r for r in rows if r['frame_id']==f]
        frame_features.append(dict(frame_id=f,mean_native_disagreement=stats([r['native_disagreement_px'] for r in rr])['mean'],
            A11_minus_A10_mean_error=metrics['A11'][f]['frame_mean_px']-metrics['A10'][f]['frame_mean_px'],
            A11_minus_A10_correct10=sum(r['A11']['PCK10']-r['A10']['PCK10'] for r in rr),
            matched=metrics['R0'][f]['matched']))
    goodframes=[f for f in frame_features if f['matched'] and f['mean_native_disagreement'] is not None]
    x=np.array([f['mean_native_disagreement'] for f in goodframes]);y=np.array([f['A11_minus_A10_mean_error'] for f in goodframes])
    correlation=dict(n=len(x),pearson_signed_delta=float(np.corrcoef(x,y)[0,1]),pearson_absolute_delta=float(np.corrcoef(x,abs(y))[0,1]),
        note='Descriptive association on reused GT; magnitude can flag disagreement but does not determine which candidate is correct. No selector trained/validated.')
    save('D5_FAILURE_CLUSTERS.json',dict(groups=clusters,bins=bins,frame_features=frame_features,correlation=correlation,
        missing_features=['independent manual external/self subtype','actual distance','verified self-occlusion PnP availability on primary93'],
        classifier_trained=False,selector_signal_validated=False))
    csvsave('D5_FAILURE_CLUSTERS.csv',[dict(frame_id=r['frame_id'],corner_id=r['corner_id'],group=r['transition10'],
        **{k:r[k] for k in features},A11_native_correction=r['A11']['native_correction_px'],session=r['session_id'],visibility=r['verified_visibility_class']) for r in rows])
    decision=C.read(C.DOC/'E2_DECISION.json');delta=100*(baselines['A11']['PCK']['10']-baselines['A10']['PCK']['10'])
    source=C.read(C.DOC/'E2_SOURCE_RESULTS.json')['models']
    save('DECISION_REINTERPRETATION.json',dict(status='[확인]',HISTORICAL_GATE_RESULT=decision['decision'],
        historical_recovery_requires_A11_greater_A10=False,CAUSAL_INTERPRETATION='NOT_SUPPORTED_IN_THIS_RUN' if delta<=0 else 'POSITIVE_DIRECT_CONTRAST_PILOT_ONLY',
        A11_minus_A10_PCK10_pp=delta,A11_minus_A10_PCK20_pp=100*(baselines['A11']['PCK']['20']-baselines['A10']['PCK']['20']),
        A11_minus_A10_matched_median_px=baselines['A11']['matched_pooled_corner8_median_px']-baselines['A10']['matched_pooled_corner8_median_px'],
        A11_minus_A10_matched_P90_px=baselines['A11']['matched_pooled_corner8_P90_px']-baselines['A10']['matched_pooled_corner8_P90_px'],
        A11_minus_A10_full_frame_mean_px=float(np.mean([f['A11_minus_A10_mean_error'] for f in frame_features])),
        source_stress_PCK10={a:source[a]['stress']['PCK10'] for a in ('INITIAL_SYNTH','A10','A11')},
        source_clean_PCK10={a:source[a]['clean']['PCK10'] for a in ('INITIAL_SYNTH','A10','A11')}))
    print('ANALYZED',d['transition'],'bands', {k:(v['gained'],v['lost']) for k,v in bands.items()},'oracle',d4['headroom'],flush=True)


def gallery():
    rows=C.read(OUT/'D1_TRANSITIONS.json')['rows'];lookup={(r['frame_id'],r['corner_id']):r for r in rows}
    chosen={}
    for r in rows:
        tags=[]
        if r['transition10']=='BAD->GOOD':tags.append('A11_ONLY_SUCCESS')
        if r['transition10']=='GOOD->BAD':tags.append('A10_ONLY_SUCCESS')
        if r['hard_recovery']:tags.append('HARD_RECOVERY')
        if r['good_damage']:tags.append('GOOD_POINT_DAMAGE')
        if tags:chosen[(r['frame_id'],r['corner_id'])]=tags
    ordered=sorted(lookup)
    for i in np.random.default_rng(1).choice(len(ordered),20,replace=False):chosen.setdefault(ordered[int(i)],[]).append('RANDOM_CONTROL')
    imgdir=OUT/'review_images';imgdir.mkdir(exist_ok=True);cards=[];imgs={};selection=[]
    colors=dict(R0='#ffe266',N2='#35caff',REPLAY='#ff71e8',A10='#ff9f43',A11='#ff5656')
    for (fid,j),tags in chosen.items():
        r=lookup[(fid,j)]
        if fid not in imgs:
            name=fid.replace(':','__')+'.jpg';path=imgdir/name
            assert not path.exists();C.verify(r['image'])
            Image.open(ROOT/r['image']['path']).convert('RGB').save(path,quality=88)
            imgs[fid]='review_images/'+name
        content=f'<image href="{imgs[fid]}" width="640" height="480"/>'
        gx,gy=r['gt_xy'];content+=f'<path d="M{gx-6},{gy}h12 M{gx},{gy-6}v12" stroke="#66ff66" stroke-width="2"/><text x="{gx+7}" y="{gy-7}" fill="#66ff66" stroke="black" stroke-width=".3" font-size="13">GT G{j}</text>'
        for a in SHOW:
            p=r[a]['xy']
            if p is None:continue
            x,y=p;content+=f'<circle cx="{x}" cy="{y}" r="4" fill="none" stroke="{colors[a]}" stroke-width="2"/><line x1="{gx}" y1="{gy}" x2="{x}" y2="{y}" stroke="{colors[a]}" stroke-width="1"/>'
        box=r['bbox_xyxy'] or [0,0,640,480]
        x0=max(0,box[0]-25);y0=max(0,box[1]-25);x1=min(640,box[2]+25);y1=min(480,box[3]+25)
        if x1<=x0 or y1<=y0:x0,y0,x1,y1=0,0,640,480
        svg=f'<svg viewBox="0 0 640 480">{content}</svg>';crop=f'<svg viewBox="{x0} {y0} {x1-x0} {y1-y0}">{content}</svg>'
        table='<table><tr><th>model</th><th>native corner</th><th>error px</th><th>matched</th></tr>'+''.join(f'<tr><td style="color:{colors[a]}">{a}</td><td>P{r[a]["native_index"]}</td><td>{r[a]["error_px"]:.3f}</td><td>{r[a]["matched"]}</td></tr>' for a in SHOW)+'</table>'
        cards.append(f'<section><h2>{html.escape(fid)} / G{j}</h2><p>{" | ".join(tags)}</p><p>Stored: {r["stored_reason"]}; human: {r["manual_visibility"]}; proxy: {r["auto_proxy_state"]}; verified subtype: {r["verified_visibility_class"]}</p><div class="pair"><div>Full RGB{svg}</div><div>R0 bbox crop (25px margin){crop}</div></div>{table}<p>[확인] GT green cross. Circles/lines show only this physical corner, mapped through each approved whole-object branch. No new human visibility decision has been entered.</p></section>')
        selection.append(dict(frame_id=fid,corner_id=j,reasons=tags))
    save('D3_REVIEW_SELECTION.json',dict(status='[확인]',corners=len(chosen),frames=len(imgs),random_seed=1,random_count=20,rows=selection,review_completed=False))
    save('D3_REVIEW_GALLERY.html','<!doctype html><meta charset="utf-8"><title>D3 corner visibility review</title><style>body{background:#11212a;color:#edf6f8;font:16px system-ui;margin:24px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:15px}svg{width:100%;max-height:480px;background:#172a32}section{border-top:2px solid #789;margin:24px 0;padding-top:10px}td,th{padding:4px 20px;text-align:left}a{color:#70cfff}</style><h1>D3 — Review needed, not new visibility GT</h1><p>[확인] 모든 A11-only/A10-only/hard-recovery/good-damage + seed1 random20. 중복은 합침. external/self subtype 자동 확정 없음. 예측을 표시한 사후 검토이므로 독립 blinded visibility GT가 아님.</p>'+''.join(cards))
    print('GALLERY',len(chosen),'corners',len(imgs),'frames',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','analyze','gallery'));args=parser.parse_args()
    globals()[args.stage]()
