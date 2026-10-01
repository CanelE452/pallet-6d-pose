"""Export completed input audits, never selector performance or supervision.

The image gallery is fixed TRAIN input evidence. It reprojects cached R0 poses
without PnP/GT/new inference; arrows are explicitly enlarged twentyfold.
"""
from . import common as C
from .direction_features import project_direction
import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import numpy as np

EXPERTS=('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')
HYP=('long-face-front','short-face-front')
READS=None
ALLOWED_IMAGES=set()
ALLOWED_PREDICTIONS=set()


def install_guard():
    global READS
    if READS is not None:return
    READS=set();sys.dont_write_bytecode=True
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();name=str(path);mode,flags=args[1:3]
        writing=((isinstance(mode,str) and any(c in mode for c in 'wax+')) or
            (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))))
        if writing:
            if path.is_relative_to(C.ROOT):assert path.is_relative_to(C.DOC),('REPORT_WRITE_SCOPE',name)
            return
        assert not any(token in name for token in ('SOURCE_TRAIN_LABELS','REPRESENTATION_ARRAYS.npz',
            'GEOMETRY_SIDETABLE','GEOMETRY_RESOLVED_POSE_GT','SYNTH_LABELS','SYNTH_RECORDS',
            'SOURCE_MANIFEST','SOURCE_VAL_','/data/evaluation/','/real_gt_v2/','/annotations/',
            '/fits/','/model_parameters/','POSE_METRICS','REAL_CHOICES','REAL_FEATURES')),('REPORT_LABEL_QUALITY_WEIGHT_DENIED',name)
        assert path.suffix.lower() not in ('.pt','.pth','.onnx'),('REPORT_WEIGHT_DENIED',name)
        if path.suffix=='.npz':assert path==C.RAW/'TRAIN_DIRECTIONS.npz',('REPORT_ARRAY_DENIED',name)
        if '/source_predictions/' in name:assert path in ALLOWED_PREDICTIONS,('REPORT_NON_GALLERY_PREDICTION_DENIED',name)
        if path.suffix.lower() in ('.jpg','.jpeg','.png') and not path.is_relative_to(C.DOC):
            assert path in ALLOWED_IMAGES,('REPORT_NON_GALLERY_IMAGE_DENIED',name)
        if path.is_relative_to(C.ROOT):READS.add(str(path.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def publish(path,value):
    data=value if isinstance(value,bytes) else (value if isinstance(value,str) else
        json.dumps(C.clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==data,('EXISTING_PUBLICATION_DIFFERS',str(path))
    else:
        with path.open('xb') as handle:handle.write(data)


def csv_export(name,rows):
    assert rows
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);publish(C.DOC/name,out.getvalue())
    return dict(binding=C.bind(C.DOC/name),rows=len(rows),columns=list(rows[0]))


def bound_read(binding):
    C.verify(binding);return C.read(C.ROOT/binding['path'])


def load():
    install_guard()
    # Authenticate the protocol and its operator lineage without re-opening the
    # complete historical feature NPZ. Relevant gallery inputs are verified below.
    C.verify(C.read(C.DOC/'AUDIT_PROTOCOL_SHA.json'))
    protocol=C.read(C.DOC/'AUDIT_PROTOCOL.json')
    for binding in protocol['codes']:C.verify(binding)
    # Verify the representation protocol bytes, not its TRAIN-label binding:
    # report generation never opens that supervision array even for hashing.
    C.verify(C.read(C.DOC/'REPRESENTATION_PROTOCOL_SHA.json'))
    representation_protocol=C.read(C.DOC/'REPRESENTATION_PROTOCOL.json')
    assert representation_protocol['feature_protocol']==C.bind(C.DOC/'AUDIT_PROTOCOL.json')
    feature=C.read(C.DOC/'FEATURE_AUDIT.json');representation=C.read(C.DOC/'REPRESENTATION_AUDIT.json')
    verification=C.read(C.DOC/'VERIFICATION.json')
    for value in (feature,representation,verification):assert value['complete'] and value['PASS']
    assert verification['audited_feature_receipt']==C.bind(C.DOC/'FEATURE_AUDIT.json')
    assert verification['audited_representation']==C.bind(C.DOC/'REPRESENTATION_AUDIT.json')
    assert verification['protocol']==C.bind(C.DOC/'REPRESENTATION_PROTOCOL.json')
    assert feature['protocol']==C.bind(C.DOC/'AUDIT_PROTOCOL.json')
    assert representation['protocol']==C.bind(C.DOC/'REPRESENTATION_PROTOCOL.json')
    assert representation['inputs']['direction_receipt']==C.bind(C.DOC/'FEATURE_AUDIT.json')
    assert representation['inputs']['direction_features']==feature['directions']
    assert feature['source_TRAIN_only'] and representation['source_TRAIN_only']
    for key in ('new_fits','actual_data_weight_probes','actual_data_objective_probes','new_policy_argmin','image_forwards','new_PnP_calls'):
        assert representation[key]==0
    assert not representation['VAL_quality_read'] and not representation['real_targets_read']
    C.verify(feature['directions'])
    with np.load(C.ROOT/feature['directions']['path'],allow_pickle=False) as z:
        ids=z['ids'].copy();source_index=z['source_index'].copy();anchor_index=z['anchor_index'].copy()
        masks={m:z[m+'_valid'].copy() for m in EXPERTS}
    assert len(ids)==2598 and len(set(ids.tolist()))==2598 and (anchor_index<0).sum()==1
    metadata=bound_read(protocol['inputs']['metadata'])
    rows=[metadata[int(i)] for i in source_index]
    assert [r['id'] for r in rows]==ids.tolist() and all(r['split']=='TRAIN' for r in rows)
    return dict(protocol=protocol,feature=feature,representation=representation,verification=verification,
        ids=ids,source_index=source_index,anchor_index=anchor_index,masks=masks,metadata=metadata,rows=rows)


def exports(data):
    feature,representation=data['feature'],data['representation'];rows=[]
    for model in EXPERTS:
        r=feature['models'][model]
        rows.append(dict(model=model,frames=r['frames'],valid_candidates=r['valid_candidates'],allinvalid_rows=r['allinvalid_rows'],
            raw94_px_max_absolute=r['parity_max_absolute']['raw94_px'],
            raw94_bboxnorm_max_absolute=r['parity_max_absolute']['raw94_bboxnorm'],
            frozen_corner8_px_max_absolute=r['parity_max_absolute']['frozen_corner8_px'],
            raw94_px_max_fraction_of_tolerance=r['parity_max_fraction_of_tolerance']['raw94_px'],
            raw94_bboxnorm_max_fraction_of_tolerance=r['parity_max_fraction_of_tolerance']['raw94_bboxnorm'],
            frozen_corner8_px_max_fraction_of_tolerance=r['parity_max_fraction_of_tolerance']['frozen_corner8_px']))
    out={'FEATURE_PARITY.csv':csv_export('FEATURE_PARITY.csv',rows)};summary=[];columns=[]
    for model in C.MODEL_NAMES:
        row=representation['models'][model];s=row['input_span'];c=row['collisions'];old,new=c['old_all_valid'],c['extended_all_valid']
        summary.append(dict(model=model,nonanchor_rows=s['rows'],old_rank=s['old']['rank'],extended_rank=s['extended']['rank'],
            rank_gain=s['rank_gain'],frobenius_residual_ratio=s['frobenius_residual_ratio'],ratio_le1e_4=s['extra_information_below_fixed_ratio'],
            old_repeated_groups=old['repeated_groups'],old_anchor_nonanchor_groups=old['anchor_nonanchor_groups'],
            old_T_sign_conflict_groups=old['axes']['T']['sign_conflict']['groups'],new_T_sign_conflict_groups=new['axes']['T']['sign_conflict']['groups'],
            old_R_sign_conflict_groups=old['axes']['R']['sign_conflict']['groups'],new_R_sign_conflict_groups=new['axes']['R']['sign_conflict']['groups'],
            old_T_strict_opposite_groups=old['axes']['T']['strict_opposite']['groups'],new_T_strict_opposite_groups=new['axes']['T']['strict_opposite']['groups'],
            old_R_strict_opposite_groups=old['axes']['R']['strict_opposite']['groups'],new_R_strict_opposite_groups=new['axes']['R']['strict_opposite']['groups'],
            forced_anchor_rows=c['forced_anchor_rows'],failed_rows_retained=c['all_invalid_frame_rows']))
        for k in range(18):
            columns.append(dict(model=model,column=k,name=f'point{k//2}_{"x" if k%2==0 else "y"}',
                extra_column_norm=s['extra_column_norm'][k],residual_column_norm=s['residual_column_norm'][k],
                residual_ratio=s['per_column_residual_ratio'][k]))
    out['REP_SUMMARY.csv']=csv_export('REP_SUMMARY.csv',summary)
    out['EXTRA18_COLUMN_RESIDUAL.csv']=csv_export('EXTRA18_COLUMN_RESIDUAL.csv',columns)
    membership=[]
    for n,row in enumerate(data['rows']):
        value=dict(id=row['id'],source_index=int(data['source_index'][n]),split=row['split'],
            source_family=row['id'].split('__')[0],anchor_index=int(data['anchor_index'][n]),
            anchor_available=bool(data['anchor_index'][n]>=0))
        value.update({m+'_valid_candidates':int(data['masks'][m][n].sum()) for m in EXPERTS})
        membership.append(value)
    out['TRAIN_ROW_MEMBERSHIP.csv']=csv_export('TRAIN_ROW_MEMBERSHIP.csv',membership)
    assert {k:v['rows'] for k,v in out.items()}=={'FEATURE_PARITY.csv':4,'REP_SUMMARY.csv':4,'EXTRA18_COLUMN_RESIDUAL.csv':72,'TRAIN_ROW_MEMBERSHIP.csv':2598}
    return out


def plots(representation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=C.DOC/'figures';path.mkdir(parents=True,exist_ok=True)
    labels=list(C.MODEL_NAMES);old=[representation['models'][m]['input_span']['old']['rank'] for m in labels]
    new=[representation['models'][m]['input_span']['extended']['rank'] for m in labels]
    ratio=[representation['models'][m]['input_span']['frobenius_residual_ratio'] for m in labels]
    values=dict(rank_and_residual=dict(models=labels,old_rank=old,extended_rank=new,frobenius_residual_ratio=ratio,threshold=1e-4))
    artifacts=[];x=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    axes[0].bar(x-.2,old,.4,label='Existing 253 input columns',color='#64748b')
    axes[0].bar(x+.2,new,.4,label='Extended 271 input columns',color='#0d9488')
    for j,(a,b) in enumerate(zip(old,new)):
        axes[0].text(j-.2,a,str(a),ha='center',va='bottom');axes[0].text(j+.2,b,str(b),ha='center',va='bottom')
    axes[0].set_xticks(x,labels);axes[0].set_ylabel('Numerical input-matrix rank');axes[0].set_ylim(0,max(new+old)*1.15);axes[0].legend(fontsize=8)
    axes[1].bar(x,ratio,color='#7c3aed');axes[1].axhline(1e-4,color='#dc2626',ls='--',label='Predeclared input-only threshold 1e-4')
    for j,value in enumerate(ratio):axes[1].text(j,value,f'{value:.6f}',ha='center',va='bottom')
    axes[1].set_xticks(x,labels);axes[1].set_ylabel('||E - projection_on_old_X(E)||F / ||E||F')
    axes[1].set_ylim(0,max(max(ratio),1e-4)*1.2);axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('TRAIN input representation audit ONLY | no new model fit or T/R performance evaluation\nOriginal valid nonanchor rows; targets are not passed into the input-span calculation.')
    file=path/'input_rank_and_residual.png';fig.savefig(file,dpi=150);plt.close(fig);artifacts.append(C.bind(file))
    matrix=[representation['models'][m]['input_span']['per_column_residual_ratio'] for m in labels]
    columns=[f'P{k//2} {"x" if k%2==0 else "y"}' for k in range(18)]
    values['column_residual_ratio']=dict(models=labels,columns=columns,values=matrix)
    fig,ax=plt.subplots(figsize=(16,5),constrained_layout=True)
    shown=np.asarray(matrix);im=ax.imshow(shown,cmap='viridis',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(np.arange(18),columns,rotation=45,ha='right');ax.set_yticks(np.arange(4),labels)
    for i in range(4):
        for j in range(18):ax.text(j,i,f'{shown[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if shown[i,j]<.5 else 'black')
    fig.colorbar(im,ax=ax,label='Input-column residual norm / original extra-column norm')
    ax.set_title('18 direction columns after projection onto existing253 | not uncertainty or improvement probability\nZero-denominator column has ratio0; anchors and failed candidates excluded from SVD.')
    file=path/'extra18_column_residual.png';fig.savefig(file,dpi=150);plt.close(fig);artifacts.append(C.bind(file))
    return artifacts,values


def gallery(data):
    import matplotlib.pyplot as plt
    from PIL import Image
    protocol=data['protocol'];ids=protocol['gallery']['ids'];assert len(ids)==len(set(ids))==6
    scale=float(protocol['gallery']['display_arrow_scale']);assert scale==20.
    byid={r['id']:r for r in data['rows']};positions={str(fid):int(i) for fid,i in zip(data['ids'],data['source_index'])}
    anchor_indices={str(fid):int(i) for fid,i in zip(data['ids'],data['anchor_index'])}
    poses=bound_read(protocol['inputs']['poses']);predlock=bound_read(protocol['inputs']['source_predictions_lock'])
    receipt=bound_read(predlock['receipts']['R0']);assert receipt['complete'] and receipt['model']=='R0'
    source_protocol=bound_read(predlock['protocol'])
    details=[]
    for fid in ids:
        row=byid[fid];index=positions[fid];binding=receipt['files'][index]
        expected=C.PARENT_RAW/'source_predictions'/'R0'/f'{index:05d}.json'
        assert binding['path']==str(expected.relative_to(C.ROOT));ALLOWED_PREDICTIONS.add(expected.resolve())
        saved=bound_read(binding)
        assert saved['id']==fid and saved['model']=='R0' and saved['protocol_sha']==predlock['protocol']['sha256']
        assert saved['checkpoint_sha']==source_protocol['checkpoints']['R0']['sha256']
        record=poses['records']['R0'][fid];pose=record['GEO_pose'];prediction=saved['prediction']
        selected=prediction['selected_index']
        if selected is not None:assert 0<=selected<len(prediction['candidates'])
        candidate=prediction['candidates'][selected] if selected is not None else None
        image_path=(C.ROOT/row['image']['path']).resolve();ALLOWED_IMAGES.add(image_path);C.verify(row['image'])
        with Image.open(image_path) as image:rgb=np.asarray(image.convert('RGB'))
        assert list(rgb.shape[:2])==row['hw'],'PREPARED_IMAGE_SHAPE_NO_ADDITIONAL_PADDING'
        q=np.asarray(candidate['keypoints_xy'],np.float64) if candidate is not None else None
        box=np.asarray(candidate['box_xyxy'],np.float64) if candidate is not None else None
        available=bool(anchor_indices[fid]>=0 and pose['available'] and candidate is not None)
        if available:assert HYP[anchor_indices[fid]]==record['GEO_name']
        projected=residual=arrow_end=None;diagonal=None
        if available:
            result=project_direction(pose,q,box,row['K']);projected=result['projected'];residual=projected-q
            arrow_end=q+scale*residual;diagonal=result['diagonal']
        details.append(dict(id=fid,source_index=index,split='TRAIN',source_family=fid.split('__')[0],
            image=row['image'],image_hw=row['hw'],additional_padding=0,K=row['K'],dimensions_m=row['dims'],
            original_physical_dimension_order='Width=x,Height=y,Depth=z',source_prediction=binding,
            source_prediction_receipt=predlock['receipts']['R0'],prediction_selected_index=selected,
            observed_q9=q,bbox_xyxy=box,anchor_index=anchor_indices[fid],anchor_hypothesis=record['GEO_name'],anchor_pose=pose,
            anchor_available=available,projected_uv=projected,residual_px=residual,
            display_arrow_scale=scale,display_arrow_end_uv=arrow_end,bbox_diagonal=diagonal,
            GT_outline_shown=False,physical_pose_errors_shown=False,new_pose_estimate=False))
    artifacts=[]
    for page in range(3):
        fig,axes=plt.subplots(2,1,figsize=(13,14))
        fig.subplots_adjust(top=.86,bottom=.03,left=.03,right=.97,hspace=.4)
        for ax,row in zip(axes,details[2*page:2*page+2]):
            with Image.open(C.ROOT/row['image']['path']) as image:rgb=np.asarray(image.convert('RGB'))
            ax.imshow(rgb);q=row['observed_q9']
            if q is not None:
                q=np.asarray(q);ax.scatter(q[:,0],q[:,1],s=38,facecolors='none',edgecolors='#00d6ff',linewidths=1.6,label='Observed R0 keypoints (q)')
            if row['anchor_available']:
                uv=np.asarray(row['projected_uv']);end=np.asarray(row['display_arrow_end_uv'])
                ax.scatter(uv[:,0],uv[:,1],s=38,c='#ff9d00',marker='x',linewidths=1.8,label='Frozen pose projection (actual)')
                for k,(start,stop) in enumerate(zip(q,end)):
                    ax.annotate('',xy=stop,xytext=start,arrowprops=dict(arrowstyle='->',color='#ef44ff',lw=1.1,alpha=.8))
                    ax.text(start[0]+3,start[1]-3,str(k),color='white',fontsize=8,bbox=dict(facecolor='black',alpha=.5,pad=.5))
                ax.plot([],[],color='#ef44ff',label='Residual arrows: 20x DISPLAY ONLY')
            else:ax.text(.5,.5,'FROZEN R0 ANCHOR UNAVAILABLE\nCase retained; no replacement',transform=ax.transAxes,ha='center',va='center',color='red',bbox=dict(facecolor='white',alpha=.8))
            dims=row['dimensions_m'];title=f"{row['id']} | TRAIN INPUT ONLY\nPhysical dimensions: Width {100*dims[0]:g} / Height {100*dims[1]:g} / Depth {100*dims[2]:g} cm"
            title+='\nPrepared source RGB: additional padding 0 | '+str(row['anchor_hypothesis'])
            ax.set_title(title,fontsize=10);ax.set_xlim(-.5,rgb.shape[1]-.5);ax.set_ylim(rgb.shape[0]-.5,-.5);ax.axis('off')
            if q is not None:ax.legend(loc='upper left',fontsize=8,framealpha=.85)
        fig.suptitle('TRAIN INPUTS ONLY — NO NEW FIT / NO REAL EVALUATION / NO T-R CLAIM\nCyan: observed q9 | Orange: actual frozen pose projection\nMAGENTA ARROWS ARE ENLARGED 20x FOR DISPLAY (not actual point displacement)',fontsize=13,y=.985)
        file=C.DOC/'figures'/f'train_input_residual_directions_{page+1}.jpg'
        fig.savefig(file,dpi=120);plt.close(fig);artifacts.append(C.bind(file))
    selection=dict(complete=True,status='FIXED_TRAIN_INPUT_ILLUSTRATIONS_ONLY',protocol=C.bind(C.DOC/'AUDIT_PROTOCOL.json'),
        selected_gallery_ids=ids,selection=protocol['gallery']['selection'],metadata=protocol['inputs']['metadata'],
        poses=protocol['inputs']['poses'],predictions_lock=protocol['inputs']['source_predictions_lock'],
        display_arrow_scale=scale,additional_padding=0,actual_RGB_images=6,illustrations=details,
        new_fits=0,image_forwards=0,new_PnP_calls=0,new_pose_error_calculations=0,real_images_or_reference_reads=0)
    publish(C.DOC/'GALLERY_SELECTION.json',selection)
    return artifacts,details


def report_text(data,illustrations):
    feature=data['feature'];rep=data['representation'];novelty=rep['novelty']
    line=['# 잔차 방향 18개 입력의 TRAIN 표현 감사','',
        '**새 학습 0회 · 새 VAL 성능 평가 0회 · 실사 평가 0회. T/R 개선 효과는 측정하지 않았다.** 입력 감사와 독립 검산의 PASS는 성능 성공을 뜻하지 않는다. 원래 목표는 아직 달성되지 않았다.','',
        '이번 단계는 고정된 모델이 보는 입력에 어떤 정보가 빠져 있는지 검사했다. 기존 253차원 입력을 그대로 재현하고, 9개 점의 `투영 위치 − 관측 위치`를 x/y 방향으로 분리한 18개 값을 추가했을 때 기존 입력의 선형 조합으로 설명되는지 확인했다. 최종 모델 가중치를 읽거나 적용하지 않았고, 후보를 새로 선택하지 않았다.','',
        f"사전 적격 TRAIN **{rep['frames']:,}행**을 전부 유지했다. 유효 anchor {rep['available_anchor_rows']:,}행과 전체 후보 실패 {rep['failed_rows_retained']}행을 포함한다. 4개 expert의 유효 후보는 합계 {sum(feature['models'][m]['valid_candidates'] for m in EXPERTS):,}개이며, 추가 정규화는 R0의 유효 TRAIN 후보 {feature['normalization']['valid_candidate_count']:,}개만 사용했다.",'',
        '## 실제로 확인한 것','',
        '기존 94개 특징에는 각 점의 잔차 크기가 들어 있지만 x/y 부호가 별도 열로 들어 있지 않다. 다만 다른 PnP 기반 특징에도 정보가 들어 있으므로 이 사실만으로 전체 표현의 부족을 증명할 수는 없다. 이번에는 기존 전체 253차원과 새 방향 18차원을 직접 대조했다.','',
        '추가 입력은 `d18 = flatten((project(R_cf, centroid, cf_extents, K) − q9) / bbox_diagonal)`이다. 코너 0–7과 중심 8을 사용한다. 원래 q9·bbox·K·동결 pose를 재사용했으며 새 PnP, 이미지 추론, 참조 pose 계산을 하지 않았다. 방향 부호는 투영값에서 관측값을 뺀 값으로 고정했다.','',
        'R0 TRAIN 유효 후보의 float32 평균·표준편차를 구하고 표준편차를 최소 1e−6으로 제한했다. 후보를 float32로 정규화한 뒤 float64로 변환하고 원래 anchor의 18차원을 뺐다. 기존 253차원과 기존 타깃의 6개 해시는 직전 단계와 정확히 같다. anchor 입력은 정확히 0이며 invalid 후보와 실패 행도 그대로 유지했다.','',
        '| Expert | TRAIN 행 | 유효 후보 | 실패 행 | 기존 9점 잔차 크기와 최대 차이(px) | bbox 정규화 잔차 최대 차이 | 기존 코너 8개 캐시 최대 차이(px) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model in EXPERTS:
        r=feature['models'][model];p=r['parity_max_absolute']
        line.append(f"| {model} | {r['frames']} | {r['valid_candidates']} | {r['allinvalid_rows']} | {p['raw94_px']:.9g} | {p['raw94_bboxnorm']:.9g} | {p['frozen_corner8_px']:.9g} |")
    line+=['','위 차이는 이미 저장된 특징과 투영 재구성의 수치 차이이며 팔레트의 위치·회전 정답 오차가 아니다. 허용오차와 전체 4행은 [FEATURE_PARITY.csv](FEATURE_PARITY.csv)와 [FEATURE_AUDIT.json](FEATURE_AUDIT.json)에 있다.','',
        '## 입력 열의 비중복성','',
        'SVD는 유효 nonanchor 후보만 사용했다. 기존 차분 입력을 X, 새 정규화 방향 차분을 E라고 하면 `E − U_r U_rᵀ E`를 계산한다. r은 `max(X.shape) × float64 eps × 최대 singular value`보다 큰 singular value의 개수다. 데이터 중심화, 절편, 프레임 가중치가 없으며 이 계산에는 타깃을 전달하지 않았다. 전체 singular spectrum과 허용오차도 JSON에 남겼다.','',
        '| 모델 | nonanchor 후보 | 기존 253열 rank | 확장 271열 rank | rank 증가 | 추가 입력 잔차 Frobenius 비율 | ≤ 1e−4 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for model in C.MODEL_NAMES:
        s=rep['models'][model]['input_span']
        line.append(f"| {model} | {s['rows']} | {s['old']['rank']} | {s['extended']['rank']} | {s['rank_gain']} | {s['frobenius_residual_ratio']:.9g} | {s['extra_information_below_fixed_ratio']} |")
    line+=['','![입력 rank와 새 방향 열의 투영 잔차](figures/input_rank_and_residual.png)','',
        '![4개 모델 × 18개 추가 입력의 잔차 비율](figures/extra18_column_residual.png)','',
        '두 그림은 입력 배열의 구조를 나타낸다. 신뢰도, 개선 확률, T/R 오차 또는 성능 그래프가 아니다. 분모가 0인 추가 열의 비율은 사전 규칙대로 0이다. [REP_SUMMARY.csv](REP_SUMMARY.csv)의 4행과 [EXTRA18_COLUMN_RESIDUAL.csv](EXTRA18_COLUMN_RESIDUAL.csv)의 72행이 그림 원자료다.','',
        f"사전 판정: `{novelty['status']}`. 4개 모델의 비율이 모두 1e−4 이하인지: **{novelty['all_models_ratio_le1e_4']}**. 이 기준은 입력 추가의 비중복성 근거만 판단하며 성능 gate가 아니다.",'',
        '## 동일 입력의 타깃 충돌','',
        'float64 벡터의 +0/−0만 +0으로 통일한 뒤 byte가 정확히 같은 후보를 묶었다. 반올림하거나 유사한 입력을 같은 것으로 간주하지 않았다. 동결 TRAIN 캐시의 타깃은 이 그룹의 값 범위·부호 충돌을 설명하는 데만 사용했다. 부호 충돌에는 0과 양수/음수의 차이도 포함하며, strict opposite는 같은 그룹에 음수와 양수가 모두 있는 경우다.','',
        '| 모델 | 기존 반복 그룹 | anchor/nonanchor 혼합 그룹 | T 부호 충돌 기존→271 | R 부호 충돌 기존→271 | T 양·음 충돌 기존→271 | R 양·음 충돌 기존→271 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        c=rep['models'][model]['collisions'];a,b=c['old_all_valid'],c['extended_all_valid']
        cells=[]
        for kind in ('sign_conflict','strict_opposite'):
            for axis in ('T','R'):cells.append(f"{a['axes'][axis][kind]['groups']}→{b['axes'][axis][kind]['groups']}")
        line.append(f"| {model} | {a['repeated_groups']} | {a['anchor_nonanchor_groups']} | "+' | '.join(cells)+' |')
    line+=['','anchor의 차분 입력과 타깃은 정의상 모두 0이다. 이 반복은 nonanchor 충돌과 따로 집계했으며, anchor/nonanchor 혼합 그룹도 별도로 기록했다. 확장 입력이 나눈 그룹과 남은 충돌의 전체 목록은 [REPRESENTATION_AUDIT.json](REPRESENTATION_AUDIT.json)에 있다. 정확한 충돌이 없더라도 표현이 충분하거나 선형 모델이 올바른 순서를 배울 수 있다는 결론은 나오지 않는다.','',
        '## 고정 TRAIN 입력 사진과 물리 치수','',
        '**다음 6장은 실사 성능 결과가 아닌 source TRAIN 입력 예시다.** G38/P0/TEX별 사전 적격 ID의 사전순 첫 2장을 프로토콜에서 추출 전에 고정했다. 오류 크기나 결과를 보고 고르거나 실패 사례를 교체하지 않았다. 이미 준비된 이미지 크기와 metadata가 일치하며 **추가 padding은 0픽셀**이다.','',
        '파란 원은 R0에서 관측한 q9, 주황 X는 동결된 R0 운영 pose의 실제 투영 위치다. **자홍 화살표는 보기 쉽게 20배 확대한 방향 표시이며 실제 점 간 거리와 다르다.** 참조 정답 윤곽, GT pose, T/R 개선 수치를 표시하지 않았다. 박스에 사용한 camera-facing extents와 원래 물리 Width/Height/Depth는 좌표계가 다를 수 있어 각각 metadata에 보존했다.','',
        '| TRAIN ID | 이미지 H×W(px) | Width(cm) | Height(cm) | Depth(cm) | R0 anchor 사용 가능 |',
        '|---|---:|---:|---:|---:|---|']
    for row in illustrations:
        h,w=row['image_hw'];d=row['dimensions_m']
        line.append(f"| {row['id']} | {h}×{w} | {100*d[0]:g} | {100*d[1]:g} | {100*d[2]:g} | {row['anchor_available']} |")
    for page in range(1,4):line+=['',f'![고정 TRAIN 입력과 20배 방향 표시 {page}](figures/train_input_residual_directions_{page}.jpg)']
    line+=['','[GALLERY_SELECTION.json](GALLERY_SELECTION.json)에 원본 이미지·예측 SHA, K, 물리 치수, cf_extents/R_cf/centroid, 관측 q9, bbox, 실제 투영점과 확대 표시점을 모두 남겼다. [TRAIN_ROW_MEMBERSHIP.csv](TRAIN_ROW_MEMBERSHIP.csv)는 그림에 보이지 않는 행까지 포함한 전체 TRAIN 2,598행과 실패 1행을 공개한다.','',
        '## 읽은 데이터와 검산 범위','',
        '기존 source 컨테이너에는 TRAIN/VAL 합계 5,120행이 들어 있지만 실제 방향 재구성·표현 진단 대상은 사전에 적격으로 고정한 TRAIN 2,598행이다. VAL 품질, 실사 참조, 기존 모델 가중치, 새 후보 선택은 읽거나 실행하지 않았다. 원래 TRAIN 정답 캐시는 표현 감사의 해시·충돌 설명에만 사용했으며 입력 열공간 투영에는 전달하지 않았다. 이 보고서 생성 과정은 해당 정답 NPZ와 진단 배열 NPZ를 다시 열지 않고 동결된 JSON 집계와 방향 NPZ의 행·유효성 metadata만 사용한다.','',
        '독립 검산은 좌표별 투영, float32 정규화, 기존 253차원과 6개 해시, 모든 동일 입력 그룹의 범위·부호, 확장 그룹 분할을 재구성했다. 입력 열공간은 pivoted QR와 작은 행렬의 GESVD로 별도 계산해 생산자의 직접 SVD와 비교했다. 검산 PASS는 이 재현성 검사에 대한 판정이다.','',
        '실행 중 두 번의 파일 접근 중단도 보존했다. 첫 시도는 threadpool 런타임 초기화가 `/dev/null`을 쓰기 모드로 여는 데서 계산 전에 멈췄다. 두 번째는 4개 모델 계산과 진단 NPZ 저장 뒤, 자체 출력 파일을 해시하려는 읽기가 guard에 막혔다. 생산자 코드를 수정하지 않은 초기화·출력 확정 복구 경로, 남아 있던 NPZ와 실제 실행 내역은 [EXECUTION_KO.md](EXECUTION_KO.md)에 기록한다. 실패 시도를 삭제하거나 단일 성공 시도처럼 취급하지 않았다.','',
        '## 해석과 다음 단계의 범위','']
    if novelty['all_models_ratio_le1e_4']:
        line+=['사전 기준에 따르면 이번 18개 열을 추가할 비중복성 근거가 부족하다. 이 결과만으로 후속 271차원 학습을 진행할 근거로 사용하지 않는다.']
    else:
        line+=['이번 감사는 18개 방향 열이 기존 입력의 선형 열공간과 완전히 중복된다는 설명을 기각할 근거를 제공한다. 이 정보가 물리 오차의 부호를 예측하거나 새 데이터로 전이된다는 증거는 아직 없다. 후속 후보는 기존 253차원에 이 고정 18차원만 붙인 271차원 입력으로 4개 모델을 학습하는 한 가지 실험이다. 이 보고서에서는 그 학습을 실행하지 않았다.']
    line+=['','후속 실험을 한다면 원래 TRAIN 행·valid mask·anchor·signed T/R 타깃·정규화·RBF basis·Huber+sign 목적식·λ·zero 초기화·Newton/Armijo 예산과 수렴 인증을 유지한다. R0_ONLY 1개와 UNION 3개를 모두 남기고 최고 seed를 고르지 않는다. source 45/45 조건과 원래 및 matched 실사 5개 조건도 유지하며, source 실패 시 실사 learned route를 실행하지 않는다. 입력 18개 추가 외의 가중치·임계값·추가 탐색을 이 감사에서 선택하지 않았다.','',
        '재사용 source VAL과 과거 실사 관측, 원래 teacher 및 수동 참조의 계보 제약은 없어지지 않는다. 이번 TRAIN 내부 감사는 독립 시험집합을 새로 만들거나 이전의 불안정한 T/R 성과를 개선 결과로 바꾸지 않는다. 이전 실험 결과는 [직전 Huber+sign 보고서](../pallet_pose_signed_axes_sign_20261001_v1/REPORT_KO.md), 근거는 [설계 문서](DESIGN_KO.md)에서 확인할 수 있다.','',
        '## 파일과 재현 근거','',
        '- [FEATURE_AUDIT.json](FEATURE_AUDIT.json): 방향 재구성과 기존 특징 수치 일치.','- [REPRESENTATION_AUDIT_KO.md](REPRESENTATION_AUDIT_KO.md) / [JSON](REPRESENTATION_AUDIT.json): 전체 입력 진단.','- [VERIFICATION_KO.md](VERIFICATION_KO.md) / [JSON](VERIFICATION.json): 독립 수치 검산.','- [AUDIT_PROTOCOL.json](AUDIT_PROTOCOL.json) / [REPRESENTATION_PROTOCOL.json](REPRESENTATION_PROTOCOL.json): 추출·진단 전 고정 계약.','- [REPORT_DATA.json](REPORT_DATA.json): 표·그림 값과 원본 SHA 연결.','- [PUBLIC_REVIEW_KO.md](PUBLIC_REVIEW_KO.md) / [PUBLIC_REVIEW.json](PUBLIC_REVIEW.json): 공개물 검산.','- [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json): 최종 공개 파일 목록과 SHA.','- [report.py](../../../scripts/research/pallet_pose_residual_direction_audit_20261001_v1/report.py), [direction_features.py](../../../scripts/research/pallet_pose_residual_direction_audit_20261001_v1/direction_features.py), [representation_audit.py](../../../scripts/research/pallet_pose_residual_direction_audit_20261001_v1/representation_audit.py), [verify_audit.py](../../../scripts/research/pallet_pose_residual_direction_audit_20261001_v1/verify_audit.py).','',
        '새 모델 checkpoint나 성능 CSV는 없다. 공개 CSV 4개는 특징 일치·입력 rank/잔차·행 소속을 담으며 T/R 성능표가 아니다. 원본 캐시·이미지의 로컬 가용성 및 재현 한계는 실행 기록을 따른다.','']
    return '\n'.join(line)


def main():
    data=load()
    assert (C.DOC/'EXECUTION_KO.md').exists(),'WAIT_FOR_FINAL_EXECUTION_RECORD'
    tables=exports(data);plot_files,figure_values=plots(data['representation']);gallery_files,illustrations=gallery(data)
    figures=plot_files+gallery_files;assert len(figures)==5
    publish(C.DOC/'REPORT_KO.md',report_text(data,illustrations))
    publish(C.DOC/'README.md','# 잔차 방향 입력 감사\n\n새 학습·VAL 성능 평가·실사 평가는 모두 0이다. 입력 감사 PASS와 T/R 개선 성공을 구분한다.\n\n[상세 보고서](REPORT_KO.md) · [원자료와 SHA](REPORT_DATA.json) · [독립 검산](VERIFICATION_KO.md) · [실행 기록](EXECUTION_KO.md)\n')
    report=dict(complete=True,status='TRAIN_INPUT_REPRESENTATION_AUDIT_COMPLETE_PERFORMANCE_UNMEASURED',
        diagnostic_only=True,method_success=False,goal_complete=False,
        code=C.bind(__file__),protocol=C.bind(C.DOC/'AUDIT_PROTOCOL.json'),
        representation_protocol=C.bind(C.DOC/'REPRESENTATION_PROTOCOL.json'),
        feature_audit=C.bind(C.DOC/'FEATURE_AUDIT.json'),representation_audit=C.bind(C.DOC/'REPRESENTATION_AUDIT.json'),
        independent_verification=C.bind(C.DOC/'VERIFICATION.json'),execution=C.bind(C.DOC/'EXECUTION_KO.md'),
        gallery=C.bind(C.DOC/'GALLERY_SELECTION.json'),exports=tables,csv_rows={k:v['rows'] for k,v in tables.items()},
        figures=figures,figure_values=figure_values,illustrations=illustrations,
        report=C.bind(C.DOC/'REPORT_KO.md'),readme=C.bind(C.DOC/'README.md'),
        frames=2598,available_anchor_rows=2597,failed_rows_retained=1,actual_RGB_images=6,
        source_TRAIN_only=True,
        additional_padding=0,display_arrow_scale=20.,current_learned_real_panels=0,
        new_fits=0,new_optimizer_steps=0,new_image_forwards=0,new_PnP_calls=0,new_pose_error_calculations=0,
        VAL_quality_reads=0,real_reference_reads=0,real_routes=0,new_policy_argmin=0,prior_weight_reads=0,
        report_cached_TRAIN_label_reads=0,report_representation_array_reads=0,
        learned_real_evaluated=False,performance_measured=False,
        novelty=data['representation']['novelty'],
        proposed_followup_executed=False,
        artifacts=[*[v['binding'] for v in tables.values()],*figures,C.bind(C.DOC/'GALLERY_SELECTION.json'),
            C.bind(C.DOC/'REPORT_KO.md'),C.bind(C.DOC/'README.md')],
        read_paths=sorted(READS))
    publish(C.DOC/'REPORT_DATA.json',report)
    print(json.dumps(dict(PASS=True,report=C.bind(C.DOC/'REPORT_KO.md'),report_data=C.bind(C.DOC/'REPORT_DATA.json'),
        csv_rows=report['csv_rows'],figures=5,new_fits=0,performance_measured=False),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();main()
