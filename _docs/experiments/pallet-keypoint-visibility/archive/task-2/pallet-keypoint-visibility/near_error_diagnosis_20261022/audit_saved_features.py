"""Descriptive frozen-cache comparison. No forward, fitting or model calibration."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RUNS=HERE.parent.parent
V60=RUNS/'external_ray_proxy_head_20261022_v60'
V61=RUNS/'target_presence_classifier_20261022_v61'
MANIFEST=HERE.parent/'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv'


def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else path.open()) as f:return json.load(f)


def csvread(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def writecsv(name,rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


allrows=[]
for scope in ('SYNTH','REAL'):
    inputs=read(V60/('PREPARED_INPUTS_SYNTH.json.gz'if scope=='SYNTH'else'PREPARED_INPUTS_REAL_VERIFIED_V2.json.gz'))
    metadata=read(V60/f'{scope}_FEATURE_METADATA.json')
    z=np.load(V60/f'{scope}_ROI_CACHE.npz')
    x=z['x'].astype(np.float32);valid=x[:,:,64]>0.5
    features=x[:,:,:64]
    mean=np.divide((features*valid[:,:,None]).sum((-1,-2)),valid.sum((-1,-2))[:,:,None],out=np.zeros(features.shape[:3],np.float32),where=valid.sum((-1,-2))[:,:,None]>0)
    center_norm=np.linalg.norm(features[:,:,:,4,4],axis=-1)
    if scope=='SYNTH':
        labels={(r['id'],int(r['native_corner'])):r for r in csvread(V61/'PRESENCE_QUERY_LABELS.csv')}
        pred={(r['id'],int(r['native_corner'])):r for r in csvread(V61/'SYNTH_QUERY_PREDICTIONS.csv')}
    else: labels={(r['id'],int(r['native_corner'])):r for r in csvread(MANIFEST)}
    for i,u in enumerate(inputs):
        assert metadata[i]['id']==u['id']
        for k in range(8):
            r=labels[(u['id'],k)]
            allrows.append(dict(scope=scope,id=u['id'],native_corner=k,group=u['group'] if scope=='REAL' else 'SYNTH_'+u['split'].upper(),roi_side_raw_px=metadata[i]['side'],roi_span_P3_cells=metadata[i]['side']*metadata[i]['gain']/8,valid_ROI_samples=int(valid[i,k].sum()),center_feature_L2=float(center_norm[i,k]),mean_feature_L2=float(np.linalg.norm(mean[i,k])),feature_mean=mean[i,k],label=r,pred=pred[(u['id'],k)] if scope=='SYNTH'else r))

# TRAIN-only descriptive reference, never applied to inference or decisions.
train=[r for r in allrows if r['scope']=='SYNTH'and r['group']=='SYNTH_TRAIN'and r['valid_ROI_samples']>0 and r['label']['valid']=='True']
vec=np.stack([r['feature_mean']for r in train]);mu=vec.mean(0);sd=vec.std(0)
sd=np.maximum(sd,1e-6)
for r in allrows:r['channel_mean_synthetic_TRAIN_z_RMS']=float(np.sqrt(np.mean(((r['feature_mean']-mu)/sd)**2)))

def subgroup(r):
    if r['scope']=='SYNTH':
        if r['group']!='SYNTH_VALIDATION':return None
        if r['label']['valid']!='True':return None
        return 'SYNTH_VAL_ALL_VALID'
    lab=r['label']
    if lab['human_state']!='DIRECT_VISIBLE' or lab['query_valid']!='True' or float(lab['initial_reference_error_px'])>3:return None
    return r['group']+'_NEAR_D_'+('V61_POS'if lab['present_at_fixed_0_5']=='True'else'V61_NEG')

summary=[];query_rows=[]
for r in allrows:
    group=subgroup(r)
    if not group or r['valid_ROI_samples']==0:continue
    query_rows.append({k:v for k,v in r.items()if k not in ('label','pred','feature_mean')}|{'diagnostic_group':group})
for group in sorted({r['diagnostic_group']for r in query_rows}):
    rs=[r for r in query_rows if r['diagnostic_group']==group]
    out={'group':group,'N':len(rs),'frames':len({r['id']for r in rs})}
    for key in ('roi_side_raw_px','roi_span_P3_cells','center_feature_L2','mean_feature_L2','channel_mean_synthetic_TRAIN_z_RMS'):
        a=np.array([r[key]for r in rs]);out.update({key+'_'+q:float(np.quantile(a,p))for q,p in [('q05',.05),('median',.5),('q95',.95)]})
    summary.append(out)
writecsv('FROZEN_FEATURE_QUERY_DESCRIPTORS.csv',query_rows)
writecsv('FROZEN_FEATURE_DISTRIBUTION.csv',summary)
notes={'new_fits':0,'new_forward_calls':0,'new_PnP_calls':0,'model_statistics_or_decisions_changed':False,'diagnostic_reference':'64 spatially averaged channel descriptors standardized with SYNTH TRAIN summary mean/std for descriptive comparison only; no classifier or normalizer attached to model.','limits':'Descriptive feature/scale differences do not certify domain failure without real exact-query pixel labels. Outcome-based groups are descriptive, not independent causal comparisons.','ROI':'v61 samples a globally computed P3 feature map into 9x9; RGB is not cropped/resized and re-encoded. ROI resampling does not restore fine image details.','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [MANIFEST,V60/'head_and_roi.py',V61/'MODEL_FREEZE.json']},'summary':summary}
(HERE/'FROZEN_FEATURE_AUDIT.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
