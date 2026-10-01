"""Completed ResNet18 results only: historical YOLO context, no invented outputs.

Requires the final60 baseline, six completed P/D fits, source-only selection,
DEV319 evaluation and all390 runtime samples. Source baseline learning curves
are epoch-wide means from the completion receipt, not sparse progress logging.
"""
import csv
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from common import *

def fmt(v,d=3):return 'NA' if v is None or not np.isfinite(v) else f'{v:.{d}f}'
def percent(v):return None if v is None else 100*v
def strict_mean(values):
    return None if any(v is None or not np.isfinite(v) for v in values) else float(np.mean(values))
def mean_method(rows,family):
    vals=[rows[f'{family}{s}'] for s in SEEDS]
    assert len({v['matched_frames'] for v in vals})==1
    return dict(median_px=strict_mean([r['median_px'] for r in vals]),p90_px=strict_mean([r['p90_px'] for r in vals]),
       ALL_GT_PCK={'10':strict_mean([r['ALL_GT_PCK']['10'] for r in vals])},matched_frames=vals[0]['matched_frames'],
       pose={k:strict_mean([r['pose'].get(k) for r in vals]) for k in ('translation_median_cm','rotation_median_deg',
           'translation_p90_cm','rotation_p90_deg','yaw_p90_deg','coverage')})

def tables(result):
    historical={k:v['metrics'] for k,v in result['historical_yolo_results']['methods'].items()}
    rows=[]
    for backbone,methods in [('YOLO',historical),('RESNET18',result['methods'])]:
        for arm in ('baseline','D','P'):
            value=methods['R0' if backbone=='YOLO' else 'RESNET18'] if arm=='baseline' else mean_method(methods,arm)
            rows.append(dict(backbone=backbone,refiner=arm,**value))
    return rows

def figures(rows):
    directory=DOC/'figures';directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors=['#52687a','#d6a24a','#258878']
    for panel,field,title in [(0,'median_px','Conditional 2D median (px)'),(1,'p90_px','Conditional 2D P90 (px)'),(2,'PCK','Full-GT PCK10 (%)')]:
        for group,backbone in enumerate(('YOLO','RESNET18')):
            rr=[r for r in rows if r['backbone']==backbone]
            vals=[percent(r['ALL_GT_PCK']['10']) if field=='PCK' else r[field] for r in rr]
            vals=[np.nan if v is None else v for v in vals]
            xx=np.arange(3)+4*group
            axs[panel].bar(xx,vals,color=colors)
            for x,y in zip(xx,vals):
                if np.isfinite(y):axs[panel].text(x,y,f'{y:.2f}',ha='center',va='bottom',fontsize=8)
                else:axs[panel].text(x,0,'NA',ha='center',va='bottom',fontsize=8)
        axs[panel].set_xticks([0,1,2,4,5,6],['Base','D','P','Base','D','P']);axs[panel].set_title(title)
        axs[panel].set_xlabel('Historical YOLO                 Current ResNet18');axs[panel].grid(axis='y',alpha=.2)
    fig.suptitle('Same DEV319 / 2,818 labeled points; D/P = mean of 3 seed statistics\nConditional support can differ between estimators; missing cases remain in full-GT PCK')
    fig.savefig(directory/'cross_estimator_2d.png',dpi=160);fig.savefig(directory/'cross_estimator_2d.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key,title,mult in zip(axs,('translation_median_cm','rotation_median_deg','coverage'),
                               ('Translation median (cm)','Rotation median (deg)','Pose coverage (%)'),(1,1,100)):
        vals=[np.nan if r['pose'][key] is None else r['pose'][key]*mult for r in rows];ax.bar(np.arange(6),vals,color=colors*2)
        for i,v in enumerate(vals):ax.text(i,v if np.isfinite(v) else 0,fmt(v,2) if np.isfinite(v) else 'NA',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(range(6),['Y base','Y+D','Y+P','R18 base','R18+D','R18+P'],rotation=25);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Canonical geometry-derived pose reference; conditional errors and full-population coverage\nThese measurements do not establish independently measured physical accuracy')
    fig.savefig(directory/'cross_estimator_pose.png',dpi=160);fig.savefig(directory/'cross_estimator_pose.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for seed in SEEDS:
        data=read(RAW/'runs'/f'seed{seed}'/'STEP_METRICS.json')
        for ax,arm in zip(axs,('P','D')):
            # Show all fixed 100-update block means, not selected best epochs.
            x=[r['step'] for r in data];v=[r['arms'][arm]['loss'] for r in data]
            xx=np.array(x).reshape(-1,100).mean(1);vv=np.array(v).reshape(-1,100).mean(1)
            ax.plot(xx,vv,label=f'seed{seed}')
            ax.set_title(f'{arm}: '+('soft-target CE' if arm=='P' else 'normalized residual L1'));ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Fixed 6,000-update training; 100-update block means; losses have different units')
    fig.savefig(directory/'training.png',dpi=160);plt.close(fig)

def verify_binding(value):
    path=ROOT/value['path'];assert sha(path)==value['sha256'],path
    if 'bytes' in value:assert path.stat().st_size==value['bytes'],path
    return path

def baseline_curves(completion):
    rows=completion['curves'];assert len(rows)==60
    assert [row['epoch'] for row in rows]==list(range(1,61))
    for row in rows:
        assert row['training_frames']==55980 and row['calibration']['frames']==1004
        assert row['step']==math.ceil(55980/16)*row['epoch']
        assert np.isfinite([row['training_loss'],row['calibration']['loss'],row['lr']]).all()
        assert row['training_loss']>=0 and row['calibration']['loss']>=0
    fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
    ax.plot([r['epoch'] for r in rows],[r['training_loss'] for r in rows],label='TRAIN mean loss (55,980 frames / epoch)',color='#258878')
    ax.plot([r['epoch'] for r in rows],[r['calibration']['loss'] for r in rows],label='Calibration loss (1,004 frames)',color='#bc6b39')
    ax.set(xlabel='Completed epoch',ylabel='Masked heatmap MSE (fixed 9-channel denominator)',
           title='ResNet18 baseline: all 60 epochs; fixed final checkpoint, no best-epoch selection')
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.savefig(DOC/'figures/baseline_learning_curve.png',dpi=160);plt.close(fig)
    path=DOC/'data/BASELINE_CURVE.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','step','lr','training_frames','training_loss','calibration_frames','calibration_loss','calibration_supervised_channels'],lineterminator='\n')
        writer.writeheader()
        for r in rows:writer.writerow({k:r[k] for k in ('epoch','step','lr','training_frames','training_loss')}|
            dict(calibration_frames=r['calibration']['frames'],calibration_loss=r['calibration']['loss'],
                 calibration_supervised_channels=r['calibration']['supervised_channels']))
    return dict(source=bound(BASELINE_COMPLETE),rows=rows,figure=bound(DOC/'figures/baseline_learning_curve.png'),csv=bound(path))

def gallery():
    from evaluation import canonical_modules,canonical_key
    E,*_=canonical_modules()
    pair=E.validate_evaluation_request(positive_manifest=DEV,negative_manifest=DEV.with_name('DEV_NEG2689.json'),population_role=E.PopulationRole.DEV,allow_unavailable_final=False)
    targets={i.frame_id:E._legacy_forbidden_target(i) for i in pair.positive.items}
    pred=read(RAW/'DEV_PREDICTIONS.json')['frames'];byid={r['frame_id']:r for r in pred}
    errors=read(RAW/'DEV_FULL_PRECISION_ERRORS.json');base={r['frame_id']:r for r in errors['RESNET18']}
    refined={r['frame_id']:r for r in errors['P1']}
    matched=[fid for fid in sorted(base) if base[fid]['canonical_matched'] and base[fid]['errors'] and refined[fid]['errors']]
    delta={fid:float(np.median(refined[fid]['errors'])-np.median(base[fid]['errors'])) for fid in matched}
    ranked=sorted(matched,key=lambda fid:(delta[fid],fid));choices=[]
    for label,candidates in [('smallest_delta',ranked[:2]),('largest_delta',list(reversed(ranked[-2:]))),('missing_or_unmatched',[fid for fid in sorted(base) if not base[fid]['canonical_matched']][:2])]:
        for fid in candidates:
            if fid not in [p['frame_id'] for p in choices]:choices.append(dict(frame_id=fid,selection=label,delta_frame_median_px=delta.get(fid)))
    mapping={r['frame_id']:r for r in read(DOC/'DEV_EVALUATION_LOCK.json')['frame_mapping']}
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for j,c in enumerate(choices,1):
        row=byid[c['frame_id']];meta=mapping[c['frame_id']];path=ROOT/row['image_key']
        assert sha(path)==row['image_sha256'];im=np.array(Image.open(path).convert('RGB'))
        target=targets[c['frame_id']];gt=target.keypoints_xy;gv=target.keypoint_supervision_mask
        fig,axs=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
        for ax,arm in zip(axs,('RESNET18','D1','P1')):
            q=np.asarray(row['base_points'] if arm=='RESNET18' else row['refined'][arm],float)
            ax.imshow(im);ax.scatter(gt[gv,0],gt[gv,1],s=24,facecolors='none',edgecolors='#00d687',linewidths=1.4,label='2D reference')
            finite=np.isfinite(q).all(-1)
            for a,b in edges:
                if finite[a] and finite[b]:ax.plot(q[[a,b],0],q[[a,b],1],color='#ff5555',lw=1)
            ax.scatter(q[finite,0],q[finite,1],s=12,color='#ff5555',label='Prediction')
            for k in np.flatnonzero(finite):ax.text(q[k,0]+2,q[k,1],str(k),color='#fff14d',fontsize=7)
            ax.set_xlim(0,im.shape[1]);ax.set_ylim(im.shape[0],0);ax.set_title(arm);ax.axis('off')
        dims=meta['dimensions_m'];K=np.asarray(meta['camera_intrinsics'])
        fig.suptitle(f"{c['frame_id']} | {c['selection']} | native {im.shape[1]} x {im.shape[0]} px\n"
                     f"Pallet long x short x height = {dims['long']*1000:g} x {dims['short']*1000:g} x {dims['height']*1000:g} mm; "
                     f"K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}",fontsize=10)
        axs[0].legend(loc='lower left',fontsize=7)
        name=f'case_{j:02d}.png';fig.savefig(DOC/'figures'/name,dpi=150);plt.close(fig)
        c.update(figure=f'figures/{name}',image_sha256=row['image_sha256'],image_key=row['image_key'],
          original_hw=row['original_hw'],dimensions_m=dims,camera_intrinsics=K.tolist(),annotation=meta['annotation'],
          original_coordinates_used=True,no_manual_coordinate_edit=True)
    write(DOC/'GALLERY_MANIFEST.json',dict(complete=True,rule='Seed1 min2/max2 frame-median delta among canonical matched frames with nonempty errors, first2 missing/unmatched; lexical ties; deduplicate. At most six available cases. Outcome-selected illustrations, not aggregate evidence.',
        reference='Existing annotated2D points, visibility>0; not independent measured pose',cases=choices,
        predictions=bound(RAW/'DEV_PREDICTIONS.json'),errors=bound(RAW/'DEV_FULL_PRECISION_ERRORS.json'),
        evaluation_mapping=bound(DOC/'DEV_EVALUATION_LOCK.json')))
    return choices

def report():
    assert not (DOC/'REPORT_COMPLETE.json').exists(),'Completed report is frozen'
    protocol=verify_lock();baseline=baseline_contract();baseline_complete=read(BASELINE_COMPLETE)
    result=read(DOC/'DEV_RESULTS.json');paired=read(DOC/'DEV_PAIRED_RESULTS.json')
    training=read(DOC/'TRAINING_COMPLETE.json');cache=read(DOC/'SOURCE_CACHE_COMPLETE.json');selection=read(DOC/'SELECTION.json')
    runtime=read(DOC/'RUNTIME.json')
    assert all(x['complete'] for x in (result,paired,training,cache,selection,runtime))
    assert result['schema']=='resnet18_refiner_dev_results_v1' and result['role']=='REUSED_DEV'
    assert result['positive_frames']==319 and result['sessions']==13 and result['gt_denominator']==2818
    assert set(result['methods'])=={'RESNET18','D1','D2','D3','P1','P2','P3'}
    for key in ('baseline','baseline_protocol','baseline_training_complete'):
        assert result[key]==protocol[key];verify_binding(result[key])
    assert training['fits']==6 and training['steps_per_fit']==6000 and training['total_head_updates']==36000
    assert training['baseline_sha256']==baseline['checkpoint']['sha256']
    for seed in training['seeds']:
        assert seed['complete'] and seed['updates_per_arm']==6000 and seed['exposures_per_arm']==96000
        verify_binding(seed['checkpoint']);verify_binding(seed['order'])
        steps=read(RAW/'runs'/f"seed{seed['seed']}"/'STEP_METRICS.json')
        assert len(steps)==6000 and [r['step'] for r in steps]==list(range(1,6001))
    assert selection['real_selection'] is False and selection['protocol']==bound(DOC/'PROTOCOL.json')
    assert read(DOC/'SYNTHETIC_HELDOUT.json')['selection']==bound(DOC/'SELECTION.json')
    for value in result['source_bindings']:verify_binding(value)
    for key in ('code','evaluation_lock','predictions','paired_results','frame_results','full_precision_errors'):verify_binding(result[key])
    assert paired['predictions']['sha256']==result['predictions']['sha256']
    assert runtime['PASS'] and runtime['measured_calls']==390 and runtime['all_cache_replays_PASS']
    assert runtime['schema']=='resnet18_refiner_runtime_v1' and runtime['warmup_calls']==60
    assert runtime['frames']==26 and runtime['sessions']==13 and len(runtime['measurements'])==390
    verify_binding(runtime['raw_rows']);verify_binding(runtime['code'])
    plan=read(DOC/'RUNTIME_PLAN.json');assert runtime['plan']==bound(DOC/'RUNTIME_PLAN.json')
    assert plan['predictions']==bound(RAW/'DEV_PREDICTIONS.json')
    assert plan['protocol']==bound(DOC/'PROTOCOL.json')
    for value in plan['codes']:verify_binding(value)
    rows=tables(result);figures(rows);cases=gallery()
    public=DOC/'data';public.mkdir(exist_ok=True)
    curves=baseline_curves(baseline_complete)
    for name in ('DEV_PREDICTIONS.json','DEV_FULL_PRECISION_ERRORS.json','DEV_FRAME_RESULTS.csv'):
        shutil.copyfile(RAW/name,public/name)
    with (public/'CROSS_ESTIMATOR_TABLE.csv').open('w',newline='') as f:
        keys=['backbone','refiner','median_px','p90_px','PCK10_all_GT','matched_frames','T_median_cm','R_median_deg','T_P90_cm','R_P90_deg','yaw_P90_deg','pose_coverage']
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader()
        for r in rows:writer.writerow(dict(backbone=r['backbone'],refiner=r['refiner'],median_px=r['median_px'],p90_px=r['p90_px'],PCK10_all_GT=r['ALL_GT_PCK']['10'],
            matched_frames=r['matched_frames'],T_median_cm=r['pose']['translation_median_cm'],R_median_deg=r['pose']['rotation_median_deg'],
            T_P90_cm=r['pose'].get('translation_p90_cm'),R_P90_deg=r['pose'].get('rotation_p90_deg'),yaw_P90_deg=r['pose'].get('yaw_p90_deg'),pose_coverage=r['pose']['coverage']))
    contrast=paired['results']['P_minus_RESNET18']['conditional_keypoint_median']['session']
    if contrast.get('status')=='COMPLETE' and contrast['high']<0:
        finding='RESNET18에서도 P의 조건부 2D 중앙값 감소를 관찰했고, 재사용 DEV의 세션 단위 구간도 감소 방향을 지지한다.'
    elif contrast.get('status')=='COMPLETE' and contrast['delta']<0:
        finding='RESNET18에서 P의 조건부 2D 중앙값은 낮아졌지만, 세션 단위 구간이 일관된 개선을 뒷받침하지 않는다.'
    else:finding='RESNET18에 동일 보정 원리를 적용하는 실험을 완료했지만, 조건부 2D 중앙값의 일관된 개선 근거는 확보하지 못했다.'
    text=[f'# ResNet18에서의 P 보정기 검증 — IEEE Sensors 원고용\n\n검토일: {now()[:10]}. {finding}',
      '\n## 이 실험이 답하는 질문\n\n세 기반 추정기 검증 중 새 ResNet18 부분이다. 표의 YOLO는 이전에 측정한 결과이며, RESNET18은 이번 실행 결과다. 같은 DEV319장·13세션·주석2,818점을 사용하지만 추정기 자체의 순위를 보정 효과로 해석하지 않는다. DOPE 결과는 별도 실험 보고서에서 다루며 이 두 기반 표를 세 모델 통합 결과로 표시하지 않는다. 각 기반을 고정한 뒤 자신의 예측 오류와 내부 특징에 대해 P/D head를 각각 학습한다. YOLO용 head 가중치의 무학습 전이나 모든 backbone 일반화를 뜻하지 않는다.',
      '\n## 독립 ResNet18 본체 학습\n\nImageNet ResNet18 초기화에 256채널 deconvolution3단과9채널 heatmap head를 결합한 SimpleBaseline 계열 전체 영상 추정기다. 공식 사람 pose benchmark를 그대로 재현한 모델이나 YOLO/DOPE 가중치를 바꿔 부른 모델이 아니다. 합성 TRAIN55,980장을 예측 매칭 필터 없이 전부 사용하여 고정60epoch, seed42로 본체를 학습했다. calibration1,004장의 epoch별 손실은 기록만 하며 최저 손실 checkpoint를 선택하지 않았다.',
      f"\n실제 본체 실행: **{baseline_complete['epochs']} epochs / {baseline_complete['updates']:,} updates / {baseline_complete['source_exposures']:,} 영상 노출**, trainable parameters **{baseline_complete['parameters']:,}**. epoch60만 고정했다. 중단 후 정확한 상태를 재개한 기록은 {len(baseline_complete['resumes'])}회이며 [본체 완료 영수증](BASELINE_TRAINING_COMPLETE.json)에 남긴다.",
      '\n입력은 source의 기존 reflect100 영상에 추가 padding 없이 aspect resize 후384×512 중앙 letterbox(fill128)를 적용하고 RGB ImageNet FP32로 정규화한다. 실사는 native RGB에 reflect100을 한 번 적용한다. 실제 정수 resize 폭·높이와 letterbox 이동을 기록하여 원영상 좌표로 되돌린다. GT crop·박스 입력·flip·기하 증강은 없고 본체 학습에만 brightness/contrast/saturation0.8–1.2를 쓴다. camera K와 물리 치수는 이후 동일 pose 계산에서만 사용한다.',
      '\n타깃은 stride4의96×128 heatmap9개, 연속 중심(net xy/4), sigma2,3sigma 유한 범위다. visibility>0이며 유한하고 grid 안인 타깃만 손실에 참여하지만 분모는 항상9채널 및 전체 batch다. 손실은0.5×pixel-mean MSE다. argmax+내부 quarter-pixel 보정과 고정 peak≥0.1을 사용하며 결측을 채우지 않는다. 코너3개 이상과 원영상 hull 두 변>1px 조건의 예측 box를 사용한다.',
      f"\n60epoch 누적 supervision 기록(고유 영상 수 아님): 감독 채널 **{baseline_complete['training_counts']['supervised_channels']:,}**, grid 밖 제외 **{baseline_complete['training_counts']['outside_support_channels']:,}**, 전체 채널 분모 **{baseline_complete['training_counts']['denominator_channels']:,}**. [본체 사전 조건](BASELINE_PROTOCOL.json).",
      '\n![본체 TRAIN·calibration 전체 epoch 곡선](figures/baseline_learning_curve.png)\n\nTRAIN은 각 epoch의 전55,980장에 대한 영상 가중 평균 손실이고 sparse 로그 평균이 아니다. TRAIN은 색 증강 및 epoch 중 갱신되는 가중치를 사용하고 calibration은 epoch 말 고정 가중치·무증강으로 계산하므로 두 곡선의 절대 높이는 동일 입력 조건 비교가 아니다. [60행 곡선 CSV](data/BASELINE_CURVE.csv).',
      '\n## 보정기 실제 실행과 통제\n\n합성 TRAIN55,980 / calibration1,004 / selection1,031 / heldout1,985 분할을 재사용했다. 본체 학습을 끝낸 뒤, 보정기 학습만 실제 RESNET18 예측과 GT의 box IoU≥0.5 및 공동 유효 코너로 행을 고정했고 전체6만 장의 실패 기록은 유지했다. 새 실사 학습은0장이다. P와 직접회귀 D는 각각3seed, seed당6,000updates×batch16이며 같은 seed의 두 head는 같은 행 순서·동결 특징을 썼다. 보정기 시험용2updates씩은 폐기한 별도 head로 기록했다.',
      f"\n보정기에 실제 사용한 TRAIN 행: **{cache['partition_counts']['train']['usable']:,}장**. P/D trainable parameters는 각각 **{training['seeds'][0]['parameters']['P']:,} / {training['seeds'][0]['parameters']['D']:,}**. YOLO P와 입력 채널 adapter가 달라 파라미터 수는 같지 않다.",
      '\n[사전 고정 조건](PROTOCOL.json) · [입력 완료](SOURCE_CACHE_COMPLETE.json) · [GPU 검산](GPU_SMOKE.json) · [학습 완료](TRAINING_COMPLETE.json) · [합성 선택](SELECTION.json) · [합성 heldout](SYNTHETIC_HELDOUT.json)',
      '\n## 주 결과\n\n| 기반 | 보정 | 2D 중앙값 px↓ | P90 px↓ | 전체 GT PCK10 %↑ | 조건부 성공 장수 | T cm↓ | R deg↓ | pose coverage %↑ |\n|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"| {r['backbone']} | {r['refiner']} | {fmt(r['median_px'])} | {fmt(r['p90_px'])} | {fmt(percent(r['ALL_GT_PCK']['10']))} | {r['matched_frames']:.0f}/319 | {fmt(r['pose']['translation_median_cm'])} | {fmt(r['pose']['rotation_median_deg'])} | {fmt(percent(r['pose']['coverage']))} |")
    text += ['\n| 기반 | 보정 | T P90 cm↓ | R P90 deg↓ | yaw P90 deg↓ |\n|---|---|---:|---:|---:|']
    for r in rows:text.append(f"| {r['backbone']} | {r['refiner']} | {fmt(r['pose'].get('translation_p90_cm'))} | {fmt(r['pose'].get('rotation_p90_deg'))} | {fmt(r['pose'].get('yaw_p90_deg'))} |")
    text += ['\nNA는 해당 영수증에 통계가 없거나 유효 pose가 없는 경우다. 없는 값을0이나 성공으로 치환하지 않았다.']
    text += ['\nD/P는 각 seed 통계의 평균이며 앙상블이 아니다. 조건부 2D는 고정 box IoU≥0.5 및9점 모두 유한한 경우의 주석점만 집계한다. RESNET18의 결측 때문에 두 기반의 조건부 분모가 다를 수 있다. 전체 GT PCK는 제외된 점도 실패로 포함한다. 일부 점만 있는 프레임의 별도 pointwise PCK는 원본 JSON에 SECONDARY로 남긴다. T/R은 독립 측정기 정답이 아니라 주석·K·치수로 생성한 기존 평가 참조에 대한 값이며, 유효 pose에 조건부인 오차와 전체319 coverage를 함께 읽어야 한다.',
      '\n![2D와 전체 GT 성공률](figures/cross_estimator_2d.png)\n\n![Pose 오차와 분모](figures/cross_estimator_pose.png)',
      '\n[전체 지표](DEV_RESULTS.json) · [paired 통계](DEV_PAIRED_RESULTS.json) · [전 프레임 CSV](data/DEV_FRAME_RESULTS.csv) · [전후 좌표](data/DEV_PREDICTIONS.json) · [논문 표 CSV](data/CROSS_ESTIMATOR_TABLE.csv)',
      '\n## 세션 단위 paired 비교\n\n| RESNET18 비교 | 2D 중앙값 차이 px | 세션95% 구간 |\n|---|---:|---|']
    for key in ('P_minus_RESNET18','D_minus_RESNET18','P_minus_D'):
        c=paired['results'][key]['conditional_keypoint_median']['session']
        text.append(f"| {key} | {fmt(c.get('delta'),6)} | [{fmt(c.get('low'),6)}, {fmt(c.get('high'),6)}] / {c.get('status')} |")
    text += ['\n## 같은 RESNET18 패널에서 측정한 지연시간\n\n| 방법 | 2D 평균 ms | Pose 평균 ms | 전체 평균 ms | 전체 중앙값 ms | 전체 P90 ms |\n|---|---:|---:|---:|---:|---:|']
    for arm in ('RESNET18','D1','P1'):
        r=runtime['results'][arm]
        text.append(f"| {arm} | {fmt(r['keypoints_ms']['mean'])} | {fmt(r['pose_ms']['mean'])} | {fmt(r['full_ms']['mean'])} | {fmt(r['full_ms']['median'])} | {fmt(r['full_ms']['p90'])} |")
    text += ['\n13세션에서 순서상 첫2장씩26장, batch1, 각 방법20회 warmup 후5반복으로390회를 측정했다. 디스크 읽기·모델 초기화는 제외하고 decoded BGR 입력의 전처리부터 실제 2D 추론·보정·같은 PnP까지 포함했다. 실패 사례도 남겼으며, 모든 timed 출력이 평가 좌표 캐시와 일치함을 확인했다. 정확도는3seed 평균, 속도는 대표 seed1이므로 서로 구별한다. [속도 전체 기록](RUNTIME.json) · [고정 순서](RUNTIME_PLAN.json).']
    text += ['\n음수는 앞 방법의 오차가 작다는 뜻이다. 13세션 단위10,000 bootstrap, seed20260914의 탐색적 구간이며 다중 비교 보정은 없다. 개발 데이터의 반복 사용을 독립 확인으로 표현하지 않는다.',
       '\n## 실제 이미지·치수·실패 사례\n\n녹색 원은 기존2D 참조, 빨간 점과 선은 실제 예측이다. 참조는 표시용이며 추론에 넣지 않았다. P1 기준 프레임 중앙값 변화의 양끝과 결측/미매칭 사례를 사전 명시한 규칙으로 골랐다. 좋은 사례만 모은 대표 성능 표가 아니다. 번호0–7은 코너,8은 보존된 중심점이다. 치수는 long×short×height 순서이며 K는 원영상 좌표계다.']
    for c in cases:text.append(f"\n![{c['frame_id']} — {c['selection']}]({c['figure']})\n\n`{c['frame_id']}`: P1−RESNET18 프레임 중앙값 변화 {fmt(c['delta_frame_median_px'],6)}px. [이미지·치수·K 출처](GALLERY_MANIFEST.json).")
    text += ['\n## 학습 및 원고에서의 해석\n\n![고정 예산 학습](figures/training.png)',
      '\nP의 후보222개, local stencil, soft-target CE와 D의 직접 residual L1 원리는 유지했다. ResNet18의 실제 layer2/layer3 특징(stride8/16,128/256채널)을 사용한다. 본체와 BN 통계를 동결하고 head 학습 때는 layer3까지의 prefix만 실행하며 특징은 기존 P처럼 FP16을 거친다. 실제 좌표의 resize 크기를 축별로 되돌리고 원영상 단위 변위에 cap을 적용한다. box·score·center·결측은 보존하지만 이것만으로2D/pose 오차 악화가 금지되는 것은 아니다.',
      '\n이 본체는 전영상9채널 heatmap을 출력하며 사람 검출기 crop을 입력받는 원 논문 설정과 다르다. 새로운60epoch 본체 학습과 역사적 YOLO/DOPE의 초기화·checkpoint 선택·전체 계산량은 동일하지 않다. 따라서 통제되는 것은 각 기반 내 보정 전후와 동일 기반의 P/D 추가 예산이다. 세 backbone 통합 결론은 별도 통합 보고서에서 다룬다.',
      '\n기존 YOLO P와 더 큰 PoseFix-derived PRIOR의 정밀도–비용 결과는 별도 측정 근거다. [기존 전체 보고서](../pallet_sensors_submission_v1/FINAL_REPORT_KO.md). 새 RESNET18 latency는 같은 시점에 측정한 RESNET18/D1/P1 간 비교만 직접 해석한다. 과거 YOLO 지연시간과 현재 RESNET18 지연시간을 하나의 동시 측정 순위로 만들지 않는다.',
      '\n본체 구성의 출처는 [Simple Baselines, ECCV2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html) 및 [공식 구현](https://github.com/microsoft/human-pose-estimation.pytorch)이다. local feature를 사용하는 P를 [PoseFix](https://openaccess.thecvf.com/content_CVPR_2019/html/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.html)의 모델 내부 특징을 사용하지 않는 입력 방식과 동일하다고 주장하지 않는다.',
      '\n실사 T/R의 안정적 공동 개선이라는 원래 목표는 별도 고정 판정 조건이 있으며 이번 표로 소급 통과시키지 않는다. IEEE Sensors 투고·채택이나 새 세션 일반화는 완료했다고 표시하지 않는다.']
    (DOC/'REPORT_KO.md').write_text('\n'.join(text)+'\n')
    write(DOC/'REPORT_COMPLETE.json',dict(complete=True,code=bound(__file__),report=bound(DOC/'REPORT_KO.md'),
       sources=[bound(DOC/name) for name in ('BASELINE_PROTOCOL.json','BASELINE_TRAINING_COMPLETE.json',
           'PROTOCOL.json','SOURCE_CACHE_COMPLETE.json','DEV_RESULTS.json','DEV_PAIRED_RESULTS.json',
           'DEV_EVALUATION_LOCK.json','TRAINING_COMPLETE.json','SELECTION.json','SYNTHETIC_HELDOUT.json',
           'GALLERY_MANIFEST.json','RUNTIME.json','RUNTIME_PLAN.json')],
       baseline_curve=curves,paper_rows=rows,historical_backbone='YOLO',actual_backbone='RESNET18',
       DOPE_result_included=False,three_backbone_integrated_claim=False,
       stable_joint_TR_goal_passed=False,goal_complete=False,
       figures=[bound(p) for p in sorted((DOC/'figures').glob('*'))],case_count=len(cases),data_files=[bound(p) for p in sorted(public.glob('*'))]))

if __name__=='__main__':report()
