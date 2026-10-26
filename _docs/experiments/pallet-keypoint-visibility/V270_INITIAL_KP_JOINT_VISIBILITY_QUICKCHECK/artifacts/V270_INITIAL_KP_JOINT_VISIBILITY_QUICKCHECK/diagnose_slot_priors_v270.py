"""Post-hoc saved SYN label/channel-prior diagnostic; never fit/relabel.

Motivated by the frozen current evaluation's channel collapse. The new prior
baseline is a diagnostic only, not a pre-registered adoption policy. It uses
SYN TRAIN counts alone and never reads real labels or images.
"""
from pathlib import Path
import collections
import csv
import hashlib
import io
import json
import math
import shutil

import numpy as np

HERE=Path(__file__).absolute().parent


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112','cache','.pyc','..')):raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def checked(bound):
    assert binding(bound['path'])==bound,bound
    return safe(bound['path']).read_bytes()


def savej(path,x):
    p=safe(path);assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def csvwrite(path,rows):
    with safe(path).open('x')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def npz(path):
    with np.load(safe(path),allow_pickle=False)as z:return {k:z[k] for k in z.files}


def summarize(y,p):
    pred=p>=.5
    dd=int(((y==0)&~pred).sum());de=int(((y==0)&pred).sum())
    ed=int(((y==1)&~pred).sum());ee=int(((y==1)&pred).sum())
    dr=dd/max(1,dd+de);er=ee/max(1,ed+ee)
    prob=np.clip(p,1e-12,1-1e-12)
    return {'N':len(y),'D_to_D':dd,'D_to_E':de,'E_to_D':ed,'E_to_E':ee,
        'D_recall':dr,'E_recall':er,'E_precision':ee/(ee+de) if ee+de else None,
        'balanced_accuracy':(dr+er)/2,'accuracy':(dd+ee)/len(y),
        'E_F1':2*ee/(2*ee+de+ed) if 2*ee+de+ed else None,
        'BCE':float(-(y*np.log(prob)+(1-y)*np.log(1-prob)).mean()),
        'Brier':float(np.mean((p-y)**2))}


def main():
    out=safe(HERE/'slot_prior_diagnosis_v1');assert not out.exists(),out
    nc=json.loads(safe(HERE/'syn_joint_targets_v1/COMPLETE.json').read_text());assert nc['PASS']
    frames=[json.loads(x) for x in checked(nc['outputs']['FRAME_TARGETS1003.jsonl']).decode().splitlines()]
    manifest_path=HERE.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK/data_audit/FRAME_MANIFEST.jsonl'
    manifest=[json.loads(x) for x in safe(manifest_path).read_text().splitlines()]
    a=npz(HERE/'syn_joint_targets_v1/TARGETS.npz')
    dev=np.flatnonzero(~a['fit']);train=np.flatnonzero(a['fit'])
    assert len(dev)==235 and len(train)==768
    preds={arm:npz(HERE/f'syn_training_v1/{arm}_SYNDEV235.npz') for arm in ('VIS_ONLY','JOINT')}
    out.mkdir()
    savej(out/'POSTHOC_DIAGNOSIS_PROTOCOL_BEFORE_GEOMETRY_READS.json',{
        'completed':True,'code':binding(Path(__file__).absolute()),'source_targets_complete':binding(HERE/'syn_joint_targets_v1/COMPLETE.json'),
        'motivation':'Frozen current evaluation showed near-constant per-raw-slot E/D. This checks SYN teacher selection and a TRAIN-only static slot prior.',
        'posthoc_baseline_not_pre_registered':True,'no_method_adoption_or_threshold_change':True,
        'predictions_and_targets_unchanged':True,'RGB_reads':0,'current223_reads':0,'model_forwards':0,'optimizer_updates':0})
    census=[];fractions=[];geometry=[];geoby=collections.defaultdict(list)
    support_owners=collections.Counter();no_global_occlusion=collections.Counter();binding_list=[]
    for fold,ix in [('train',train),('validation',dev)]:
        for k in range(8):
            loc=a['loc_valid'][ix,k];sv=a['source_convex_self_visible'][ix,k];q=a['external_ray_fraction'][ix,k];y=a['visibility_y'][ix,k]
            D=int((y==0).sum());E=int((y==1).sum());valid=loc&sv
            census.append({'split':fold,'raw_model_slot':k,'N':len(ix),'D_strict_fraction0':D,'E_fraction_GE_half':E,
                'partial_0_LT_fraction_LT_half_excluded':int((valid&(q>0)&(q<.5)).sum()),
                'self_inside_excluded':int((loc&~sv).sum()),'outside_excluded':int((~loc).sum()),
                'strict_E_prevalence':E/(D+E) if D+E else None,
                'source_LT_half_negative_count_diagnostic_only':int((valid&(q<.5)).sum()),
                'source_LT_half_count_is_not_relabeling':True})
            for scope,mask in [('ALL',np.ones(len(ix),bool)),('SOURCE_SELF_VISIBLE_INFRAME',valid)]:
                values=collections.Counter(f'{x:.4f}' for x in q[mask])
                for fraction,n in sorted(values.items()):fractions.append({'split':fold,'raw_model_slot':k,'scope':scope,'fraction':fraction,'N':n})
    # Exact source geometry metadata only; no RGB or masks are opened.
    for r,m in zip(frames,manifest):
        assert r['id']==m['input']['id']
        ann=json.loads(checked(r['annotation']));obj=ann['objects'][0]
        pts=np.asarray(obj['cuboid'],np.float64);z=pts[:,2];zmin=float(z.min());zmax=float(z.max())
        span=zmax-zmin;assert span>0
        sp=obj.get('scene_placement_v2',{});ground=sp.get('support_ground_z');owner=sp.get('support_surface_name')
        support_owners[str(owner)]+=1;binding_list.append(r['annotation'])
        if ground is not None:ground=float(ground)
        for k in range(8):
            relative=(float(z[k])-zmin)/span
            role='LOWER_Z' if abs(relative)<1e-4 else 'UPPER_Z' if abs(relative-1)<1e-4 else 'OTHER_Z'
            geoby[k].append({'z':float(z[k]),'z_above_frame_min':float(z[k]-zmin),'relative_height':relative,
                            'ground_offset':float(z[k]-ground) if ground is not None else None,'role':role})
            if m['target']['M0_only_area']==0 and r['loc_valid8'][k] and r['source_convex_self_visible8'][k]:
                no_global_occlusion[(r['split'],k,f"{r['external_ray_fraction8'][k]:.4f}")]+=1
    for k,values in sorted(geoby.items()):
        offset=[v['ground_offset'] for v in values if v['ground_offset'] is not None]
        geometry.append({'raw_model_slot':k,'N':len(values),
            'LOWER_Z':sum(v['role']=='LOWER_Z' for v in values),'UPPER_Z':sum(v['role']=='UPPER_Z' for v in values),
            'OTHER_Z':sum(v['role']=='OTHER_Z' for v in values),
            'relative_height_min':min(v['relative_height'] for v in values),'relative_height_max':max(v['relative_height'] for v in values),
            'support_ground_z_available':len(offset),'support_ground_offset_min_m':float(min(offset)) if offset else None,
            'support_ground_offset_median_m':float(np.median(offset)) if offset else None,'support_ground_offset_max_m':float(max(offset)) if offset else None})
    slot_rows=[]
    y=a['visibility_y'][dev];known=y>=0
    priors=[]
    for k in range(8):
        tt=a['visibility_y'][train,k];p=float((tt==1).sum()/max(1,(tt>=0).sum()));priors.append(p)
    prior=np.tile(np.asarray(priors),(235,1));prior_cm=summarize(y[known],prior[known])
    baselines=[{'method':'POSTHOC_STATIC_SLOT_FIT_PREVALENCE',**prior_cm}]
    for arm,pred in preds.items():
        assert np.array_equal(pred['frame_index'],dev)
        prob=1/(1+np.exp(-np.clip(pred['E_logit'][:,:8].astype(np.float64),-60,60)))
        baselines.append({'method':arm,**summarize(y[known],prob[known])})
        binary=prob>=.5
        for k in range(8):
            slot_rows.append({'method':arm,'raw_model_slot':k,'N_frames':235,
                'pred_D_all_frames':int((~binary[:,k]).sum()),'pred_E_all_frames':int(binary[:,k].sum()),
                'strict_D':int((y[:,k]==0).sum()),'strict_E':int((y[:,k]==1).sum()),
                'D_to_E':int(((y[:,k]==0)&binary[:,k]).sum()),'E_to_D':int(((y[:,k]==1)&~binary[:,k]).sum()),
                'E_logit_min':float(pred['E_logit'][:,k].min()),'E_logit_median':float(np.median(pred['E_logit'][:,k])),
                'E_logit_max':float(pred['E_logit'][:,k].max()),'slot_FIT_E_prevalence':priors[k]})
    csvwrite(out/'SYN_SLOT_LABEL_CENSUS.csv',census);csvwrite(out/'SYNDEV_SLOT_PREDICTIONS.csv',slot_rows)
    csvwrite(out/'SOURCE_FRACTION_COUNTS.csv',fractions);csvwrite(out/'WORLD_Z_SLOT_GEOMETRY.csv',geometry)
    csvwrite(out/'POSTHOC_SLOT_PRIOR_BASELINES.csv',baselines)
    noocc=[{'split':key[0],'raw_model_slot':key[1],'fraction':key[2],'N':n} for key,n in sorted(no_global_occlusion.items())]
    csvwrite(out/'NO_GLOBAL_M0_M4_DIFFERENCE_FRACTIONS.csv',noocc)
    savej(out/'ANNOTATION_BINDINGS.json',{'annotations':binding_list})
    result={'completed':True,'PASS':True,'schema':'V270_POSTHOC_SYN_SLOT_PRIOR_DIAGNOSIS_V1',
        'TRAIN_slot_E_prevalences':priors,'baselines':baselines,'slot_census':census,'geometry':geometry,
        'support_surface_owner_names':dict(support_owners),'no_global_target_silhouette_occlusion_fraction_rows':noocc,
        'conclusion':'Strict clear==0 removes every eligible lower-slot negative. Slots2/3/6/7 receive E-only supervised labels; a TRAIN-only static slot prior nearly explains DEV classification. Global constant prior is insufficient evidence of visual cue learning.',
        'selection_effect':'Under original source <0.5 negative definition, eligible lower slots have many negative candidates. This is a retrospective diagnostic of the frozen strict-clear selection, not a target change or a new fit.',
        'floor_cause_status':'World lower corners coincide with stored support ground height in this corpus, but per-ray first-hit owners/offsets and byte-identical original generator are unavailable. Floor-induced offset-ray blockage is a plausible inference, not a proven cause.',
        'not_a_channel_transport_bug':'raw coordinates/fraction/self use the same dynamic channel; source perm was not reapplied. Asset canonical permutation is provenance only, current-native mapping identity.',
        'posthoc_diagnostic_only':True,'real_data_or_prediction_reads':0,'RGB_or_mask_reads':0,
        'model_forwards':0,'optimizer_updates':0,'targets_modified':0}
    savej(out/'RESULTS.json',result)
    # Before changing explanatory prose, preserve the first interpretation.
    old=HERE/'SYNDEV_READOUT_KO.html'
    if old.exists():
        shutil.copyfile(safe(old),safe(out/'PRE_SLOT_DIAGNOSIS_SYNDEV_READOUT_KO.html'))
    pages=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V270 슬롯 사전값 진단</title>',
      '<style>body{font:17px sans-serif;line-height:1.65;max-width:1400px;margin:32px auto;padding:12px;background:#111820;color:#eee}table{border-collapse:collapse}td,th{border:1px solid #566473;padding:8px}a{color:#8ed0ff}strong{color:#ffd18a}</style>',
      '<h1>V270 실패 원인 추가 진단: 가림 교사와 raw 슬롯의 강한 결합</h1>',
      '<p><strong>합성 strict 교사에서 아래쪽 슬롯2/3/6/7은 보임(D) 감독이 한 번도 없었다. 모델의 높은 일부 분류값을 이미지 단서 학습으로 해석할 수 없다.</strong> 실제 결과의 슬롯별 고정 예측을 확인한 후 수행한 사후 진단이며 새 학습이나 라벨 수정은 하지 않았다.</p>',
      '<h2>슬롯별 고정 교사</h2><table><tr><th>분할</th><th>raw 슬롯</th><th>D0</th><th>E≥0.5</th><th>partial 제외</th><th>self 제외</th><th>밖 제외</th></tr>']
    for r in census:pages.append('<tr>'+''.join('<td>'+str(r[k])+'</td>' for k in ['split','raw_model_slot','D_strict_fraction0','E_fraction_GE_half','partial_0_LT_fraction_LT_half_excluded','self_inside_excluded','outside_excluded'])+'</tr>')
    pages+=['</table><p>partial=0&lt;fraction&lt;0.5를 모두 제외한 strict-clear 선택 때문에 아래쪽 슬롯은 E-only가 되었다. 원 source의 &lt;0.5를 D로 사용하는 규약이라면 같은 eligible 하단 슬롯에 TRAIN582/595/218/252개의 음성 후보가 있다. 이것은 사후 원인 진단이며 기존 라벨·지표·학습을 바꾸지 않았다.</p>',
      '<h2>이미지를 보지 않는 TRAIN 슬롯 사전값과 비교</h2><table><tr><th>방법</th><th>CM [[D→D,D→E],[E→D,E→E]]</th><th>D recall</th><th>E recall</th><th>균형정확도</th><th>정확도</th></tr>']
    for r in baselines:pages.append('<tr>'+''.join('<td>'+str(x)+'</td>' for x in [r['method'],f"[[{r['D_to_D']},{r['D_to_E']}],[{r['E_to_D']},{r['E_to_E']}]]",f"{100*r['D_recall']:.2f}%",f"{100*r['E_recall']:.2f}%",f"{100*r['balanced_accuracy']:.2f}%",f"{100*r['accuracy']:.2f}%"] )+'</tr>')
    pages+=['</table><p>정적 사전값은 위쪽0/1/4/5를 D, 아래쪽2/3/6/7을 E로 예측한다. JOINT는 이 기준보다 위쪽 E10점을 추가로 맞추는 대신 D28점을 틀린다. 전역 상수 예측보다 낫다는 이전 설명만으로 실제 이미지 가림 단서를 배웠다고 결론내리면 안 된다.</p>',
      '<h2>아래쪽 높이와 바닥 원인 여부</h2><p>1003장 모두2/3/6/7은 world Z 최저면이고0/1/4/5는 최고면이다. 저장 support_ground_z와 높이 차이는 원표로 제공한다. 바닥 높이와 일치하고 주변 ray 비율이 반복되는 점은 확인했지만, 실제 각 ray의 hit owner/offset과 원래 생성 코드 전체는 없으므로 바닥 first-hit가 직접 원인이라고 확정하지 않았다.</p>',
      '<p>순서 오류와도 구별된다. 원 좌표·fraction·self는 같은 camera-dynamic 채널로 대응하며 permutation을 재적용하지 않았다. canonical asset 번호와 현재 native_slot은 별도다.</p>',
      '<h2>해결 방향에 대한 제한된 결론</h2><p>가림을 배우게 하려면 같은 raw 슬롯·비슷한 자세에서 보임과 외부가림이 모두 나타나는 감독이 필요하다. 현재처럼 하단의 보임을 모두 제외하면 슬롯만으로 loss를 줄일 수 있다. 이후 실험을 하더라도 표면/native 의미와 ray hit-owner를 먼저 맞추고, 같은 장면에서 외부 물체만 바꾸는 D/E 쌍으로 정적 슬롯 사전값을 넘는지 확인해야 한다. 이번3회 예산을 늘리거나 현재 실사에 맞춰 문턱·라벨을 바꾸지 않았다.</p>',
      '<p><a href="SYN_SLOT_LABEL_CENSUS.csv">슬롯 교사표</a> · <a href="SYNDEV_SLOT_PREDICTIONS.csv">DEV 슬롯 예측</a> · <a href="WORLD_Z_SLOT_GEOMETRY.csv">world 높이</a> · <a href="SOURCE_FRACTION_COUNTS.csv">분수 분포</a> · <a href="../SYNDEV_READOUT_KO.html">원 SYNDEV 수치</a> · <a href="../LABEL_AUDIT_READOUT_KO.html">실제 합성 crop</a></p></html>']
    safe(out/'READOUT_KO.html').write_text('\n'.join(pages)+'\n')
    names=['SYN_SLOT_LABEL_CENSUS.csv','SYNDEV_SLOT_PREDICTIONS.csv','SOURCE_FRACTION_COUNTS.csv',
      'WORLD_Z_SLOT_GEOMETRY.csv','POSTHOC_SLOT_PRIOR_BASELINES.csv','NO_GLOBAL_M0_M4_DIFFERENCE_FRACTIONS.csv',
      'ANNOTATION_BINDINGS.json','RESULTS.json','READOUT_KO.html','POSTHOC_DIAGNOSIS_PROTOCOL_BEFORE_GEOMETRY_READS.json']
    savej(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':binding(Path(__file__).absolute()),
        'outputs':{n:binding(out/n) for n in names},'targets_modified':0,'model_forwards':0,'optimizer_updates':0,'current223_reads':0})
    print(json.dumps({'COMPLETE':binding(out/'COMPLETE.json'),'baselines':baselines,'geometry':geometry},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
