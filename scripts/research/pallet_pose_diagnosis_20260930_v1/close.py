"""Final reusable tables and independent contract audit; no model forward."""
import csv
import io
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
import numpy as np
from . import run as C

def writecsv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row));f=io.StringIO();w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader()
    for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    C.save(path,f.getvalue())

def scalar_check(pose,metric,truth):
    assert pose['available']==metric['available']
    if not pose['available']:return
    t=np.asarray(pose['centroid']);r=np.asarray(pose['R_physical']);g=np.asarray(truth['R']);gt=np.asarray(truth['t'])
    assert truth['order']==2
    angles=[]
    for q in (np.eye(3),np.diag([-1.,1.,-1.])):
        angles.append(np.degrees(np.arccos(np.clip((np.trace((g@q).T@r)-1)/2,-1,1))))
    np.testing.assert_allclose([np.linalg.norm(t-gt)*100,min(angles)], [metric['translation_cm'],metric['rotation_deg']],rtol=1e-7,atol=1e-7)

def main():
    start,cpu=time.monotonic(),time.process_time();rows=C.metadata();meta={r['id']:r for r in rows};clean=[r for r in rows if r['severity']=='CLEAN'];_,gt=C.O.D.Pose.metadata('REAL_DEV')
    frame=[];candidates=[];checked=0
    e1=C.read(C.RAW/'E1_POSE_METRICS.json');two=C.read(C.RAW/'E1_2D_METRICS.json');e1c=C.read(C.RAW/'E1_CANDIDATES.json')
    def add(stage,arm,cond,mm,tt,baseline,cc=None):
        nonlocal checked
        for i,m in mm.items():
            r=meta[i];t=tt.get(i,{}) if tt else {};b=baseline[i]
            x=dict(stage=stage,id=i,recording=r['recording'],population='CLEAN29' if r['severity']=='CLEAN' else 'NATURAL99',severity=r['severity'],model=arm,condition=cond,
                valid=m['available'],detected=t.get('detected'),matched=t.get('matched'),translation_cm=m.get('translation_cm'),rotation_deg=m.get('rotation_deg'),
                translation_status='FINITE' if m['available'] else 'POSITIVE_INFINITY',rotation_status='FINITE' if m['available'] else 'POSITIVE_INFINITY',
                twoD_frame_mean_px=t.get('frame_mean_px'),twoD_corner_count=t.get('corners'),twoD_canonical_errors=t.get('canonical_errors'),
                delta_T_cm=m['translation_cm']-b['translation_cm'] if m['available'] and b['available'] else None,
                delta_R_deg=m['rotation_deg']-b['rotation_deg'] if m['available'] and b['available'] else None)
            if cc is not None and i in cc:
                rec=cc[i];x['GEO_name']=rec['GEO_name'];x['D9_name']=rec['selected_name'];x['GEO_fallback']=rec['GEO_fallback']
                pose=rec['GEO_pose']
                x['GEO_free_name']=rec['GEO_name']
                if cond=='held_identity' or cond.endswith('_held'):
                    x['GEO_name']=e1c['identity'][i]['GEO_name']
                    pose=next((h['pose'] for h in rec['hypotheses'] if h['name']==x['GEO_name']),dict(available=False))
                x['selected_hypothesis_for_pose']=x['GEO_name']
                scalar_check(pose,m,gt[i]);checked+=1
                if cond!='held_identity' and not cond.endswith('_held'):
                    for h in rec['hypotheses']:
                        metric=C.metric(i,h['pose'],gt[i]);candidates.append(dict(stage=stage,id=i,recording=r['recording'],model=arm,condition=cond,
                            hypothesis=h['name'],GEO_selected=h['name']==rec['GEO_name'],valid=metric['available'],T_cm=metric.get('translation_cm'),R_deg=metric.get('rotation_deg')))
            x['twoD_value_scope']='condition_output'
            x['twoD_NA_reason']='UNMATCHED_OBJECT_OR_NO_VALID_SUPERVISED_CORNERS' if x['twoD_frame_mean_px'] is None else ''
            frame.append(x)
    for a,mm in e1.items():
        arm=a.removesuffix('_held_identity');cond='held_identity' if a.endswith('_held_identity') else 'GEO'
        add('E1',arm,cond,mm,two[arm],e1['identity'],e1c[arm])
    e3=C.read(C.RAW/'E3_POSE_METRICS.json');t3=C.read(C.RAW/'E3_2D_METRICS.json');c3=C.read(C.RAW/'E3_CANDIDATES.json')
    for kind,arms in e3.items():
        for a,conds in arms.items():
            for cond,mm in conds.items():add('E3',a,kind+'_'+cond,mm,t3[kind][a][cond],e3[kind]['identity'][cond],c3[kind][a][cond])
    e4=C.read(C.RAW/'E4_METRICS.json');c4=C.read(C.RAW/'E4_REFERENCE_INTERVENTION_CANDIDATES.json');t4=C.read(C.RAW/'E4_REVIEW_2D_METRICS.json')
    for a,mm in e4.items():
        kind,mode=a.rsplit('_',1);add('E4','identity',kind+'_'+mode,mm,t4[kind],e4['baseline_'+mode],c4[kind])
    e5=C.read(C.RAW/'E5_STRESS_POSE_METRICS.json');t5=C.read(C.RAW/'E5_STRESS_2D_METRICS.json');c5=C.read(C.RAW/'E5_STRESS_CANDIDATES.json')
    for a,kinds in e5.items():
        for kind,mm in kinds.items():add('E5stress',a,kind,mm,t5[a][kind],e5['identity'][kind],c5[a][kind])
    e6=C.read(C.RAW/'E6_POSE_METRICS.json');t6=C.read(C.RAW/'E6_2D_METRICS.json');c6=C.read(C.RAW/'E6_CANDIDATES.json')
    add('E6','REALFT_A','GEO',e6,t6,e1['identity'],c6)
    writecsv(C.DOC/'FRAME_RESULTS.csv',frame);writecsv(C.DOC/'CANDIDATE_RESULTS.csv',candidates)
    # Explicit population×recording tables, keeping mixed clean/natural recordings separate.
    byrecord={}
    for pop,poprows in [('NATURAL99',[r for r in rows if r['severity']!='CLEAN']),('CLEAN29',clean)]:
        byrecord[pop]={}
        for rec in sorted({r['recording'] for r in poprows}):
            rr=[r for r in poprows if r['recording']==rec];ids=[r['id'] for r in rr]
            models=dict(e1,REALFT_A=e6)
            byrecord[pop][rec]=dict(N=len(ids),models={a:C.summarize(v[i] for i in ids) for a,v in models.items()},
                paired={a:C.paired(e1['identity'],models[a],rr,False) for a in ('PRIOR1','FULL125','REALFT_A')})
    C.save(C.DOC/'BY_RECORDING.json',byrecord)
    # All supplied original bindings and imported production code remain unchanged.
    manifest=C.read(C.DOC/'RUN_MANIFEST.json')
    for b in manifest['inputs']+manifest['codes']:C.verify(b)
    for row in C.read(C.DOC/'E7_REFERENCE_LIMITS.json')['source256']['verified'].items():assert row==('VERIFIED',768)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip()==manifest['HEAD']
    oldstatus=(C.RAW/'INITIAL_GIT_STATUS.txt').read_text().splitlines();newstatus=subprocess.check_output(['git','status','--porcelain'],cwd=C.ROOT,text=True).splitlines()
    preexisting=[line for line in oldstatus if C.NAME not in line]
    assert all(line in newstatus for line in preexisting),'Pre-existing user change status changed'
    for pop,count,recordings in [('NATURAL99',99,6),('CLEAN29',29,3)]:
        assert sum(x['N'] for x in byrecord[pop].values())==count and len(byrecord[pop])==recordings
    assert len(frame)==len({(r['stage'],r['id'],r['model'],r['condition']) for r in frame})
    assert checked==len(frame)
    costs={p.stem.removeprefix('COST_'):C.read(p) for p in sorted(C.RAW.glob('COST_*.json'))}
    imageforwards=sum(x['image_forwards'] for x in costs.values());assert imageforwards==1027
    C.save(C.DOC/'VALIDATION.json',dict(passed=True,scalar_T_R_rechecks=checked,frame_rows=len(frame),candidate_rows=len(candidates),
        legacy_same_GEO_parity_frames=256,source_bindings_verified=768,old_input_and_code_hashes_unchanged=True,preexisting_git_status_preserved=True,
        image_forwards=imageforwards,new_fits=0,optimizer_steps=0,tests='Independent scalar T/R checks for every frame row; original R0 and OLD_REF same-GEO parity; operational center/invalid/detector preservation; affine roundtrip; exact direction marginals; frozen model-state equality; all original input hashes.',
        caveats=['CPU/GPU cache reuse smoke failed at .025px; resolved by consistently recomputing E3 CPU clean and masked inputs, preserving failed smoke.',
            'Oracle/reference substitutions are diagnostic only. Reused DEV, no physical independent GT.']))
    C.ledger('close_audit',start,cpu)
    costs['close_audit']=C.read(C.RAW/'COST_close_audit.json')
    C.save(C.DOC/'RESOURCE_LEDGER.json',dict(events=costs,totals=dict(image_forwards=imageforwards,new_fits=0,optimizer_updates=0,GPU_seconds=0,
        CPU_seconds=sum(x['CPU_seconds'] for x in costs.values()),summed_stage_wall_seconds=sum(x['wall_seconds'] for x in costs.values()),
        peak_process_RSS_KiB=max(x['peak_process_RSS_KiB'] for x in costs.values())),
        accounting='Stage wall times overlap and are not elapsed wall time. CPU seconds sum measured Python stage processes; shell/browser/authoring overhead excluded. RSS peak per process, not concurrent total.',
        reuse='E1 uses512 cached image predictions; same-GEO R0/OLD_REF independently reproduced256; source768 file bindings checked; historical TRAIN and source scalar summaries reused. GEO feature-vector forward and PnP CPU work included in CPU time, not image-network forward count.'))
    C.save(DOC_PATH:=C.DOC/'EXECUTION_COMMANDS.md', '# 실행 명령\n\n환경: `MPLCONFIGDIR=/tmp/pallet-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_diagnosis_20260930_v1.` 뒤에 아래 모듈·인자를 붙였다. 기존 결과를 덮어쓰지 않으므로 재실행은 새 namespace를 사용한다.\n\n```text\nrun prepare\nrun e1\ninference prepare\ninference r0\ninference r0_cpu\ninference PRIOR1\ninference FULL125\ninference e6\nscoring detection\nscoring visibility\nscoring realft\nscoring e3\ntransfer\nstress\nscoring stress\nprovenance\nclose\n```\n\n`inference r0`는 CPU/GPU 캐시 재사용 smoke 불일치로 종료했다. `r0_cpu`가 동일 CPU 조건으로 clean/가림을 구성했고 최초 smoke 출력은 재사용했다. 초기 `pallet-pose` 환경의 import는 오래된 ultralytics 때문에 실패했고, 설치된 `pallet-yolo26` 환경으로 옮겼다. 학습/패키지 설치/드라이버 변경은 없었다.\n')
    print('AUDIT_COMPLETE',len(frame),imageforwards,flush=True)

if __name__=='__main__':main()
