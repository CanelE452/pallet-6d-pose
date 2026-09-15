"""Human provenance gates precede any frozen-panel confirmation inference."""
import copy
import numpy as np,torch
from env import *
def handoff():
    previous=import_old('confirmation');previous.history();previous.test_gate()
    schema=read(DOC/'CONFIRMATION_SCHEMA.json')
    schema['required_panel_fields']+=['model_freeze_sha256','lineage_review_author','lineage_review_time','annotation1_file','annotation2_file','adjudication_file','double_annotation_frame_ids','reference_QA_completed']
    schema['required_frame_fields']+=['timestamp','camera_id','pallet_identity','landmark_types','measurement_provenance']
    schema['predictions_schema']='methods R0,P1,P2,P3,D1,D2,D3,L1,L2,L3,PRIOR1,PRIOR2,PRIOR3 -> frame_id -> candidate list'
    write(DOC/'CONFIRMATION_SCHEMA.json',schema)
    text='''독립 확인 자료 인계 — 연구자/촬영 담당자/어노테이터가 제공해야 합니다.

현재: 기존 FINAL manifest는 membership UNAVAILABLE. 새 독립 확인 평가 0장.
권장 출발안: 별도12세션×15장=180장(검정력 보장 아님), 실내/외×낮/밤.
예측 전 frame 선택 규칙, 원영상/source video/capture group/촬영시각/카메라·장착,
팔레트 identity/material/실측 치수·카메라 intrinsics·왜곡 설정과 측정 출처를 보존하세요.
과거 source/train/cal/selection/DEV/AL/replay/QA에서 본 장면은 재촬영·재인코딩
파일명 변경만으로 독립이 되지 않습니다. lineage 검토 담당자가 연결 이력을 확인해야 합니다.

원자료 위치(공개 Git 제외): data/pallet/results/pallet_sensors_submission_v1/confirmation/incoming/
panel.json: CONFIRMATION_SCHEMA.json의 필드를 따르며 실제 file path/hash를 적습니다.
annotation1/annotation2/adjudication은 별도 파일로 보존합니다.
모델 예측을 보지 않은 상태로 최소30장 이중 어노테이션 및 blind adjudication,
corner0..7/center8 의미, visible/amodal/unavailable 구분을 기록합니다.
누락점은 예측으로 채우지 않습니다. 모델 성능을 보고 GT나 membership을 수정하지 않습니다.
검토 서명/시각/실측 여부를 CLI가 대신 true로 채우지 않습니다.

모든 prior 비교군 완료/선택 고정 이후 아래 명령으로 schema/hash/history/QA를 검증하고
고정 전체 panel의 실제 영상 추론 및2D 분석을 재개합니다:
  python scripts/research/pallet_sensors_submission_v1/run.py confirmation --panel data/pallet/results/pallet_sensors_submission_v1/confirmation/incoming/panel.json
  python scripts/research/pallet_sensors_submission_v1/run.py manuscript

새 frame의 geometry-reference6D는 카메라/축/실측 치수/수동 reference 연결 감사 후 별도로
binding해야 합니다.2D 개선을6D 실측 결과로 환산하지 않습니다.
Jetson은 미측정. 저자/ORCID/소속/연구비/COI/선행발표/AI 사용/투고동의는 AUTHOR_REVIEW.md.

GPU 비용 측정 재개: 현재 FINAL_STATUS의 COST가 보류이면 GPU 작업 담당자가 다른 학습의
종료를 확인한 뒤 run.py evaluate를 실행하세요. 저장된 DEV_COMPLETE로 비용 단계부터
재개하며 본 학습/영상 추론/통계를 다시 하지 않습니다. 이어 confirmation → manuscript →
새 PDF 모든 페이지 검사 및 visual receipt → audit → publish입니다. 다른 작업을 kill하지 않습니다.

Synthetic toy 레코드(실제 확인자료가 아님): frame_id=toy_000, session_id=toy_session,
image=toy.png, keypoints_xy=[[0,0],...총9개], supervision=[false,...총9개].
가짜 image hash/실측 값/검토 서명으로 이 예시를 평가 입력으로 사용하지 마세요.
'''
    (DOC/'HUMAN_INPUTS_REQUIRED.txt').write_text(text)
def panel_lock():
    paths={'R0':R0}
    for s in (1,2,3):paths.update({f'P{s}':BRAW/f'runs/seed{s}/last.pt',f'D{s}':OLD_RAW/f'runs/D{s}/last.pt',f'L{s}':LINE/f'runs/image_line_only_seed{s}/last.pt',f'PRIOR{s}':RAW/f'runs/PRIOR{s}/last.pt'})
    value=dict(all_comparators_complete=True,checkpoints={k:bound(p) for k,p in paths.items()},rules={k:bound(p) for k,p in {'P':B/'P_SELECTION.json','D':OLD_DOC/'D_SELECTION.json','L':LINE/'SELECTION.json','PRIOR':DOC/'PRIOR_SELECTION.json'}.items()},code={n:sha(HERE/n) for n in ('env.py','prior_model.py','prior_data.py','train_prior.py','prior_inference.py','posefix_contract_math.py','evaluate_prior.py','prior_analysis.py','confirmation.py')},historical_code_and_metrics=bound(DOC/'SOURCE_BINDING.json'),original_panel_lock=bound(DOC/'MODEL_PANEL_LOCK.json'))
    value['canonical_scorer_compatibility']=dict(code=bound(HERE/'evaluate_compatible.py'),selection=bound(DOC/'PRIOR_SELECTION_CANONICAL_METADATA.json'),amendment=bound(DOC/'SCORER_METADATA_AMENDMENT.json'))
    freeze(DOC/'CONFIRMATION_MODEL_FREEZE.json',value)
def run(panel_path=None):
    start=now();verify();handoff()
    if complete('DEV_COMPLETE'):
        panel_lock()
        ready=read(DOC/'CONFIRMATION_READINESS.json');ready.update(all_comparators_frozen=True,prior_baseline_pending=False,model_freeze=bound(DOC/'CONFIRMATION_MODEL_FREEZE.json'));write(DOC/'CONFIRMATION_READINESS.json',ready)
    incoming=RAW/'confirmation/incoming/panel.json'
    if panel_path is None and incoming.exists():panel_path=incoming
    if panel_path is None:
        write(DOC/'CONFIRMATION_COMPLETE.json',dict(complete=False,status='AWAITING_INDEPENDENT_DATA',checked_at=now(),incoming_path=str(incoming),available_confirmed_frames=0,only_confirmation_stage_waits=True));print('AWAITING_INDEPENDENT_DATA');return
    assert complete('DEV_COMPLETE'),'All comparators must finish accuracy evaluation before opening confirmation'
    panel=read(panel_path);schema=read(DOC/'CONFIRMATION_SCHEMA.json')
    for k in schema['required_panel_fields']:assert k in panel,k
    assert panel['model_freeze_sha256']==sha(DOC/'CONFIRMATION_MODEL_FREEZE.json')
    assert panel['reference_QA_completed'] is True and panel['lineage_review_author'] and panel['lineage_review_time']
    assert len(set(panel['double_annotation_frame_ids']))>=30
    assert set(panel['double_annotation_frame_ids']) <= {f['frame_id'] for f in panel['frames']}
    for k in ('annotation1_file','annotation2_file','adjudication_file'):assert Path(panel[k]).is_file()
    assert len({Path(panel[k]).resolve() for k in ('annotation1_file','annotation2_file','adjudication_file')})==3,'Independent annotations and adjudication must be separately preserved files'
    for f in panel['frames']:
        for k in schema['required_frame_fields']:assert k in f,(f.get('frame_id'),k)
    previous=import_old('confirmation');previous.gate(panel,read(RAW/'confirmation/HISTORY_IMAGES.json'))
    frozen=DOC/'CONFIRMATION_MEMBERSHIP_LOCK.json';freeze(frozen,dict(panel=bound(panel_path),QA_files={k:bound(panel[k]) for k in ('annotation1_file','annotation2_file','adjudication_file')},members=[{k:f[k] for k in ('frame_id','session_id','image_sha256','source_video_id','capture_group_id')} for f in panel['frames']]))
    from point_inference import PointInference
    from prior_inference import PriorInference
    dm=import_old('direct_inference');dm.DOC=OLD_DOC;dm.RAW=OLD_RAW
    er=old('evaluate_real');models=['R0']+[f'{a}{s}' for a in ('P','D','L','PRIOR') for s in (1,2,3)]
    import cv2,gc
    from paired_stats_helper import paired_pooled_median
    rows={};metrics={};output={};E=old('paper_evaluation').E
    for name in models:
        if name=='R0':model=er.PlainBaseline(R0)
        elif name.startswith('PRIOR'):model=PriorInference(int(name[5:]))
        elif name.startswith('P'):model=PointInference(int(name[1:]))
        elif name.startswith('D'):model=dm.DirectInference(int(name[1:]))
        else:model=old('inference').PalletLinePoseInference(LINE/f'runs/image_line_only_seed{name[1:]}/last.pt',LINE/'SELECTION.json')
        rr=[];den=0;hits={t:0 for t in (5,10,20)};preds={}
        try:
            for f in panel['frames']:
                image=cv2.imread(f['image']);assert image is not None
                pred=model.predict(image);candidates=pred if name=='R0' else pred['candidates'];preds[f['frame_id']]=old('inference').jsonable(candidates)
                if name!='R0':er.check_prediction(pred,output['R0'][f['frame_id']],frame_key=f['frame_id'])
                mask=np.asarray(f['supervision'],bool);gt=np.asarray(f['keypoints_xy'],float);assert np.isfinite(gt[mask]).all();den+=int(mask.sum())
                top=max(candidates,key=lambda c:c['score']) if candidates else None;e=np.empty(0)
                if top is not None and E._box_iou(np.asarray(top['box_xyxy']),np.asarray(f['box_xyxy']))>=.5 and top['keypoints_xy'] is not None:
                    e=np.linalg.norm(np.asarray(top['keypoints_xy'],float)[mask]-gt[mask],axis=-1)
                    assert np.isfinite(e).all(),'Keep failed outputs; conditional precision must be reviewed, no silent subset'
                    for t in hits:hits[t]+=int((e<=t).sum())
                rr.append(e)
        finally:
            if hasattr(model,'close'):model.close()
        rows[name]=rr;output[name]=preds;full=np.concatenate(rr);assert len(full)>0
        metrics[name]=dict(median_px=float(np.median(full)),p90_px=float(np.quantile(full,.9)),frame_mean_px=float(np.mean([x.mean() for x in rr if len(x)])),gross20=float((full>20).mean()),ALL_GT_PCK={str(t):hits[t]/den for t in hits},gt_denominator=den,supervised_points=len(full),nonfinite_points=int((~np.isfinite(full)).sum()),matched_frames=sum(bool(len(x)) for x in rr),total_frames=len(rr),coverage=sum(bool(len(x)) for x in rr)/len(rr))
        del model;gc.collect();torch.cuda.empty_cache()
    paired=paired_pooled_median(rows['R0'],[rows[f'P{s}'] for s in (1,2,3)],[f['session_id'] for f in panel['frames']])
    write(RAW/'confirmation/PREDICTIONS.json',output)
    stores={name:[dict(errors=e,session_id=f['session_id']) for e,f in zip(rr,panel['frames'])] for name,rr in rows.items()}
    contrast=import_old('analysis_panel').contrast
    secondary={}
    for name in ('D','PRIOR'):
        same=all([len(e) for e in rows[f'{name}{s}']]==[len(e) for e in rows[f'P{s}']] for s in (1,2,3))
        secondary[name]=contrast(stores,'P',name) if same else dict(status='SUPPORT_MISMATCH_NO_PRIMARY_PRECISION_CONTRAST')
    write(DOC/'CONFIRMATION_RESULTS.json',dict(complete_2D=True,metrics=metrics,P_R0=paired,P_minus_secondary=secondary,secondary_multiplicity='Unadjusted exploratory intervals; not independent multiple superiority claims',pose_status='AWAITING_NEW_FRAME_REFERENCE_BINDING',independent_physical_6D=False,membership=bound(frozen)))
    receipt('CONFIRMATION_COMPLETE',[frozen,DOC/'CONFIRMATION_MODEL_FREEZE.json'],[DOC/'CONFIRMATION_RESULTS.json',RAW/'confirmation/PREDICTIONS.json'],start,status='2D_COMPLETE_POSE_REFERENCE_PENDING')
