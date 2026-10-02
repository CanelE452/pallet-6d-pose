"""Read-only provenance audits and static label reuse; no automatic human labels."""
from collections import Counter
import csv,html,hashlib
from pathlib import Path
import numpy as np
from .compute import C,R,M,ROOT,DOC,RAW,SOURCE,write,bind

def resnet():
    from scripts.research.pallet_n3_completion_v3 import resnet_constant_adapter as A
    import torch
    torch.set_num_threads(4)
    contract,checkpoint=A.load_constant_contract(load_checkpoint=True)
    network=A.folded_rgb_network(checkpoint['model_state_dict'])
    protocol=C.read(A.PROTOCOL);frozen=C.read(C.DOC/'PROTOCOL.json')
    old=frozen['backbones']['resnet18']['adapter']['input']
    correction={
        'checkpoint_selection':{'stale':old['checkpoint_selection'],'actual':'fixed final epoch10; step34990; synthetic calibration only'},
        'decoder':{'stale':old['decoder'],'actual':protocol['decoder']},
        'mse':{'stale':old['mse'],'actual':protocol['objective']},
        'confidence_threshold':{'stale':old['confidence_threshold'],'actual':'not used in DSNT decoder; nine finite expectations; validity all nine'},
    }
    effective={'model_name':'ResNet-18 RGB (10-epoch CONSTANT-fold)','verified_contract':contract,'recipe':A.recipe(),
        'actual_training':protocol,'corrections':correction,'original_N3_protocol':bind(C.DOC/'PROTOCOL.json'),
        'decoder_source':bind(ROOT/'scripts/research/pallet_resnet18_dim_refiner_20261002_v1/full_adapter.py'),
        'strict_fold_load_pass':True,'folded_parameters':sum(p.numel() for p in network.parameters()),
        'new_training':0,'new_optimizer_updates':0,'weights_or_predictions_changed':False,
        'interpretation':'CONSTANT fixes conditioning at zero, not image output. RGB base plus image-feature/WDH-conditioned N3. Backbone total training budgets differ.'}
    write(DOC/'RESNET_EFFECTIVE_PROTOCOL.json',effective)
    text='# ResNet 명세 정정\n\n원본 protocol/receipt와 가중치는 수정하지 않았다. 실제 사용 모델은 10-epoch CONSTANT-fold RGB이다.\n\n'
    for k,v in correction.items():text+=f'- `{k}`: 과거 `{v["stale"]}` → 실제 `{v["actual"]}`.\n'
    text+='\nCONSTANT는 치수 조건 z=0을 뜻한다. 고정 FiLM을 마지막 1×1 convolution에 접어 RGB만 받는다. N3는 이미지 특징과 등록 W,D,H를 입력받으며 각 seed 6,000 update 완료 기록을 재사용한다. 기본 추정기까지 같은 총학습량을 통제한 백본 인과 비교가 아니다. 실제 checkpoint 헤더 epoch10/step34990과 strict folded load를 재확인했다. 상세 원본 해시는 RESNET_EFFECTIVE_PROTOCOL.json에 있다.\n'
    (DOC/'PROTOCOL_CORRECTIONS.md').write_text(text)

def labels():
    rows,_=R.load_dev_context(SOURCE,include_pose=False);by={r['id']:r for r in rows}
    curp=SOURCE/R.OCCLUSION_REL;oldp=SOURCE/R.SPLIT_REL
    cur={r['frame_id']:r for r in C.read(curp)['rows'] if r['frame_id'] in by}
    old=C.read(oldp)['heldout'];changes=[]
    for r in old:
        if r['severity']!=cur[r['id']]['severity']:changes.append({'id':r['id'],'old':r['severity'],'current':cur[r['id']]['severity'],'current_source':cur[r['id']]['label_source'],'old_source':'frozen SPLIT_LOCK copied severity; original review edit reason/timestamp not recorded here'})
    # Approved manual status is separate from coordinates. Never replace locked DEV coordinates.
    vp=SOURCE/'data/pallet/results/pallet_verified_anchor_v1/metadata_conflict_qa/VERIFIED_LABELS_FINAL_PRIVATE.json'
    v=C.read(vp);assert v['reference_version']=='VERIFIED_VISIBLE_ANCHOR_FINAL_V2'
    vis={};audited=[]
    for fi,ci in v['review_queue']:
        f=v['frames'][fi];c=f['corners'][ci];fid=f['frame_id']
        if fid not in by:continue
        image=SOURCE/by[fid]['_image_path'];assert C.sha256(image)==f['image_sha256']
        assert c['id']==ci and ci<8
        # Reuse reviewed native identity's visibility only, not newer reference coordinates.
        if c['status'] in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED'):vis[(fid,ci)]=c['status']
        audited.append({'id':fid,'corner_id':ci,'status':c['status'],'image_sha256':f['image_sha256'],
            'coordinate_source':c['coordinate_source'],'locked_DEV_xy':by[fid]['gt'][ci],'review_xy':c['xy'],
            'coordinate_equal':bool(np.array_equal(c['xy'],by[fid]['gt'][ci])),'coordinate_replaced':False})
    scores=C.read(RAW/'YOLO_SCORES.json');summaries={}
    for method,p in scores.items():
        groups={x:[] for x in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED','UNKNOWN')}
        for r in p['corner_scores']:
            for ci,ok in enumerate(r['canonical_valid']):
                if ok:groups[vis.get((r['id'],ci),'UNKNOWN')].append({'error':r['canonical_errors'][ci],'observed':r['matched'],'id':r['id']})
        summaries[method]={}
        for group,points in groups.items():
            allv=np.array([p['error'] for p in points]);obs=np.array([p['error'] for p in points if p['observed']])
            summaries[method][group]={'corners':len(points),'observed_corners':len(obs),'frames':len({p['id'] for p in points}),
                'median_px':float(np.median(obs)) if len(obs) else None,'P90_px':float(np.quantile(obs,.9)) if len(obs) else None,
                'PCK10_fraction':float(np.mean(allv<=10)) if len(allv) else None}
        assert sum(g['corners'] for g in summaries[method].values())==2499
    unknown=[r for r in rows if r['id'] not in cur];assert len(unknown)==191
    # The old split has extra reviewed/copied classes for 66 fitting frames. Do not
    # silently promote them: direct-review source doesn't cover those IDs.
    copied_train=[r for r in C.read(oldp)['train'] if r['id'] in {x['id'] for x in unknown}]
    audit={'current_counts':dict(Counter(r['occlusion'] for r in rows)),'old_heldout_counts':dict(Counter(r['severity'] for r in old)),
        'changes':changes,'membership_same':set(cur)=={r['id'] for r in old},'sources':[bind(curp),bind(oldp),bind(vp)],
        'visibility_statuses_reused':dict(Counter(vis.values())),'visibility_reviewed_points':audited,
        'visibility_reference_note':'Reviewed native corner statuses only; frozen DEV coordinates and full-object symmetry unchanged. 66 direct-visible manual-click coordinates differ from locked DEV, so this is visibility stratification of existing reference error, not accuracy against updated clicks.',
        'visibility_results':summaries,'unclassified_ids':[r['id'] for r in unknown],
        'additional_split_copied_labels':copied_train,'copied_labels_status':'BLOCKED_REVIEW: 66 split fitting rows have severity values but no direct human-class provenance in locked direct-review responses; do not promote',
        'prior_prediction_exposure':'UNKNOWN; anchor review was PnP-assisted before manual status',
        'cross_counts':dict(Counter(r['material']+'::'+r['occlusion'] for r in rows)),
        'sessions_by_group':{g:len({r['session'] for r in rows if r['occlusion']==g}) for g in {r['occlusion'] for r in rows}}}
    write(DOC/'STATIC_LABEL_AUDIT.json',audit)
    review=DOC/'review';review.mkdir(exist_ok=True)
    cards=[];template=[]
    for r in rows:
        # Local symlink keeps original image unchanged and avoids duplicating a large dataset.
        name=hashlib.sha256(r['id'].encode()).hexdigest()[:20]+'.png';p=review/name
        if not p.exists():p.symlink_to(SOURCE/r['_image_path'])
        template.append({'id':r['id'],'session':r['session'],'external_occlusion':'UNREVIEWED' if r['id'] not in cur else 'ALREADY_REVIEWED',
            'self_occlusion':'UNREVIEWED','out_of_frame':'UNREVIEWED','corner_visibility':[vis.get((r['id'],i),'UNREVIEWED') for i in range(8)],'reviewer':'','prior_prediction_exposure':'UNKNOWN'})
        cards.append(f'<article><h2>{html.escape(r["id"])}</h2><p>{html.escape(r["session"])}</p><img loading="lazy" width="640" src="{name}" /></article>')
    (review/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Static image review</title><h1>Static image review — pending human labels</h1><p>Only original RGB, frame ID and session. Separate external occlusion, self-occlusion and truncation; do not infer visibility from presence of coordinates. No model outputs are shown. Record uncertain cases as unknown.</p>'+''.join(cards))
    write(review/'REVIEW_TEMPLATE.json',template)
    text='# 정지 이미지 주석 감사\n\n'+f'현재 DEV319: {audit["current_counts"]}. 과거 128장과 ID 집합은 같다.\n\n'
    for d in changes:text+=f'- `{d["id"]}`: {d["old"]} → {d["current"]}. 현재 직접 검수 manifest/잠금 응답과 이전 SPLIT_LOCK의 복사본이 다르다. 변경 시각·사유는 미확인. 개수를 맞추는 변경은 하지 않았다.\n'
    text+=f'\n승인된 FINAL_V2 원자료에서 직접 가시 {Counter(vis.values())["DIRECT_VISIBLE"]}점과 외부 가림 {Counter(vis.values())["EXTERNAL_OCCLUDED"]}점의 상태를 회수했다. 기존 DEV 좌표는 수정하지 않았다. 수동 재클릭 좌표와 기존 참조 좌표는 다르므로 이 표는 기존 참조 오차의 가시성별 분석이다. PnP 보조 검수 이력과 사전 예측 노출 미확인을 명시한다. 나머지 점은 UNKNOWN이며 유효 좌표를 가시성으로 간주하지 않는다.\n\n미분류191장의 직접 검수 근거는 찾지 못했다. 그중 66장은 SPLIT_LOCK에 등급이 복사되어 있지만 원 직접 검수 manifest에 없는 ID여서 확정 레이블로 승격하지 않았다. review/index.html과 REVIEW_TEMPLATE.json은 검수 대기 자료이며, 모델·오차·순위가 없다.\n'
    (DOC/'STATIC_LABEL_AUDIT.md').write_text(text)

def square():
    from scripts.research.pallet_n3_completion_v3 import square_yolo as Y,evaluation as E
    from scripts.evaluation.green_saved_labels_v1 import annotation_arrays
    snapshot=C.read(Y.DATASET);group=next(r for r in C.read(Y.SYMMETRY)['objects'] if r['object_type']==Y.OBJECT_TYPE)
    modes={};outside=[];bindings=[]
    for mode in ['manual_declared','manual_in_frame']:
        truth=[]
        for r in snapshot['records']:
            p=SOURCE/r['annotation']['path'];assert C.sha256(p)==r['annotation']['sha256'];bindings.append(bind(p)) if mode=='manual_declared' else None
            gt,known=annotation_arrays(C.read(p));_,manual=annotation_arrays(C.read(p),True);h,w=r['original_hw']
            inside=np.isfinite(gt).all(-1)&(gt[:,0]>=0)&(gt[:,0]<w)&(gt[:,1]>=0)&(gt[:,1]<h)
            if mode=='manual_declared':
                for ci in np.flatnonzero(manual[:8]&~inside[:8]):outside.append({'id':r['id'],'corner_id':int(ci),'xy':gt[ci].tolist(),'hw':[h,w]})
            match=known&inside
            truth.append({'id':r['id'],'session':r['session'],'hw':[h,w],'gt':gt.tolist(),'valid':(manual if mode=='manual_declared' else manual&inside).tolist(),
                'box':np.r_[gt[match].min(0),gt[match].max(0)].tolist(),'permutations':group['permutations'],'material':'plastic','occlusion':'unclassified'})
        modes[mode]=truth
    yraw=C.read(C.RAW/'SQUARE_YOLO_PREDICTIONS.json');new=Y.evaluate_predictions(yraw['predictions'],modes);old=C.read(C.DOC/'SQUARE_YOLO_RESULTS.json')
    checks={};output={'yolo':{}}
    for mode in modes:
        checks['yolo:'+mode]=new['modes'][mode]['summary']==old['modes'][mode]['summary'];output['yolo'][mode]=new['modes'][mode]['families']
    for b in ['dope','resnet18']:
        d=C.read(C.DOC/f'SQUARE_{b.upper()}_RESULTS.json');p=C.read(C.RAW/f'predictions/{b}_GREEN0918_119.json')
        normalized=E.normalize_prediction_payload(p)['methods']
        # Normalize via the same adapter; no bootstrap recomputation for one-session audit.
        output[b]={}
        for mode,truth in modes.items():
            rows=[]
            for t in truth:
                row=dict(t);row.update(predictions={},matched={},detected={})
                for method,records in normalized.items():
                    pred=records[t['id']];row['predictions'][method]=pred['points'];row['detected'][method]=pred['detected'];row['matched'][method]=bool(pred['detected'] and E._iou(pred['bbox'],t['box'])>=.5)
                rows.append(row)
            output[b][mode]={}
            for method in E.METHODS:
                result=M.evaluate_method(rows,method)
                checks[b+':'+mode+':'+method]=result['corner']==d['modes'][mode]['methods'][method]['result']['corner']
                output[b][mode][method]=E._headline(result)
    assert all(checks.values()),checks
    square_sha={r['image']['sha256'] for r in snapshot['records']}
    dev=C.read(C.DEV)['items'];dev_sha={C.sha256(SOURCE/r['image_path']) for r in dev}
    write(DOC/'SQUARE_AUDIT.json',{'checks':checks,'results':output,'outside_declared_corners':outside,'annotation_bindings':bindings,
        'manual_declared':602,'manual_in_frame':600,'sessions':len({r['session'] for r in snapshot['records']}),'dimensions':[1.1,1.1,.15],
        'unique_image_hashes':len(square_sha),'exact_RGB_overlap_with_DEV319':len(square_sha&dev_sha),
        'exposure_limit':'Exact hash overlap against DEV319 only; not proof of all historical upstream training non-exposure. Existing synthetic-only fitting receipts reused.',
        'mode_selection':'Both modes reported; earlier in-frame audit and later declared table retained without retrospective primary-mode selection.',
        'pose':'x: no independent canonical 6D reference; no new reconstructed GT','CI':'NA: single session; no generalization inference'})

if __name__=='__main__':
    resnet();print('ResNet verified',flush=True)
    labels();print('Labels audited',flush=True)
    square();print('Square verified',flush=True)
