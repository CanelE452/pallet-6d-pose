"""Post-observation CAL-only fit for each F policy's actual full points.

No PRIMARY values are numerically decoded; fitting is separate from RGB/pose
production. The same pooled recipe is fixed for both policies by S2 LOGO.
"""
from pathlib import Path
import argparse
from copy import deepcopy
from collections import Counter
import time
import numpy as np
from .s1_union import rows,read,sha,write,save,payload_sha
from . import calibration as C
from . import qcov as Q

def solver_selection(selection):
    """Remove only unused policy labels; raw selections remain fully archived.

    LOCAL._prepare/owner_model consume states/full_points/partial_lines.
    This preserves coordinates, noise, native targets, physical-source IDs,
    actual line records and strict revocation archives verbatim.
    """
    selected={k:deepcopy(selection[k])for k in ('states','full_points','partial_lines')}
    for row in selected['full_points']:
        row.get('provenance',{}).pop('policy',None)
    return selected

def run(d1,campaign,observations,reference,output):
    d1,campaign,observations,reference,output=map(Path,(d1,campaign,observations,reference,output))
    seal=read(observations/'OBSERVATION_MANIFEST.json')
    if seal.get('complete')is not True or seal['rows']!=1241:raise ValueError('All F observations must be sealed')
    start=time.monotonic();output.mkdir(parents=True,exist_ok=True)
    frames={domain:list(rows(d1/f'd1_upstream_v1/{name}_UPSTREAM.jsonl.gz'))for domain,name in [('real','REAL238'),('synthetic','SYNTHETIC1003')]}
    sealed={(r['domain'],r['index']):r for r in seal['frames']}
    expected={(domain,i)for domain,ff in frames.items()for i in range(len(ff))}
    if set(sealed)!=expected or len(seal['frames'])!=1241:raise ValueError('Observation manifest population changed')
    for domain,ff in frames.items():
        for i,frame in enumerate(ff):
            folder=observations/domain/f'{i:04d}';entry=sealed[(domain,i)]
            if entry['id']!=frame['id']or sha(folder/'FRAME_SEAL.json')!=entry['seal_sha256']:raise ValueError('Observation frame seal ancestry changed')
    canonical_seal=read(reference.parent/'SCORE_SEAL.json')
    if sha(reference)!=canonical_seal['artifacts'][reference.name]:raise ValueError('Canonical reference bytes changed')
    calframes=[f for f in frames['real']if f['partition']=='CAL'];ids={f['id']for f in calframes}
    if len(ids)!=46:raise ValueError('Canonical CAL46 only')
    # Reuse the audited lexical extractor; record the F-specific purpose and
    # scope explicitly instead of copying an old permission receipt.
    text=reference.read_text();first,last=C._member(text,'references');lo,hi=C._member(text,'CAL',first)
    if hi>last:raise ValueError('CAL outside reference object')
    import json
    refs=json.loads(text[lo:hi])
    if set(refs)!=ids or any(r['order']!=2 for r in refs.values()):raise ValueError('Exact CAL C2 population required')
    write(output/'CAL_REFERENCE_ACCESS.json',dict(purpose='F own-observation pooled covariance recipe, same across policies',
        input_observation_manifest_sha256=sha(observations/'OBSERVATION_MANIFEST.json'),
        policy_lock_sha256=sha(campaign/'F_OBSERVER_POLICY_LOCK_V2.json'),reference_sha256=sha(reference),
        numeric_reference_decodes={'CAL':46,'PRIMARY':0},reference_objects=list(sorted(ids)),
        no_bias_fit=True,no_PRIMARY_threshold_or_weight_search=True))
    models={};diagnostics={};summary=[]
    for arm in ('COMMON','BI'):
        points=[];cases=[]
        for i,frame in enumerate(frames['real']):
            if frame['partition']!='CAL':continue
            folder=observations/'real'/f'{i:04d}';fs=read(folder/'FRAME_SEAL.json')
            if sha(folder/'SELECTIONS.jsonl.gz')!=fs['artifacts']['SELECTIONS.jsonl.gz']:raise ValueError('F selection changed')
            selected=next(rows(folder/'SELECTIONS.jsonl.gz'))['selections'][arm]
            effective,partial,dedup=Q.effective(selected);scale=C.nonself_scale(frame,selected)
            aligned,phase=C.project_reference(frame,refs[frame['id']]);group=frame['cluster_id']
            cases.append(dict(id=frame['id'],partition='CAL',cluster_id=group,nonself_scale_px=scale))
            for corner,r in sorted(effective.items()):
                valid=aligned is not None and bool(aligned[1][corner])
                points.append(dict(deepcopy(r),id=frame['id'],partition='CAL',cluster_id=group,nonself_scale_px=scale,
                    reference_error_xy=(np.asarray(r['xy'])-aligned[0][corner]).tolist()if valid else None))
        model=C.fit(points,cases);model['recipe']=dict(model['recipe'],coordinate_source='F_'+arm+'_ACTUAL_NEW_POLICY',selected_calibration='CP_POOLED_FIXED_BY_S2_LOGO')
        model['F_observation_manifest_sha256']=sha(observations/'OBSERVATION_MANIFEST.json');models[arm]=model
        details=C.evaluate(points,model);heldout=[]
        for group in sorted({c['cluster_id']for c in cases}):
            train=[p for p in points if p['cluster_id']!=group];traincases=[c for c in cases if c['cluster_id']!=group]
            fm=C.fit(train,traincases);heldout+=C.evaluate([p for p in points if p['cluster_id']==group],fm)
        diagnostics[arm]=dict(fit=C.summarize(details),LOGO=C.summarize(heldout),counts=dict(Counter(p['state']for p in points)),
                              points=points,model=model)
    for domain,ff in frames.items():
        for i,frame in enumerate(ff):
            folder=observations/domain/f'{i:04d}';fs=read(folder/'FRAME_SEAL.json')
            if sha(folder/'SELECTIONS.jsonl.gz')!=fs['artifacts']['SELECTIONS.jsonl.gz']:raise ValueError('F selection changed')
            raw=next(rows(folder/'SELECTIONS.jsonl.gz'));selections={}
            if raw['id']!=frame['id']:raise ValueError('Canonical F selection identity changed')
            for arm in ('COMMON','BI'):
                effective,dedup=C.common_effective_selection(solver_selection(raw['selections'][arm]));scale=C.nonself_scale(frame,effective)
                selections['F_'+arm]=C.apply_selection(effective,models[arm],'CP',scale)
            Q.write(output/'packets'/domain/f'{i:04d}.json.gz',dict(id=frame['id'],partition=frame['partition'],split=frame['split'],
                selections=selections,source_seal_sha256=sha(folder/'FRAME_SEAL.json')))
            summary.append(dict(domain=domain,index=i,id=frame['id'],packet_sha256=sha(output/'packets'/domain/f'{i:04d}.json.gz')))
    write(output/'MODELS.json',models);Q.write(output/'CAL_DIAGNOSTICS.json.gz',diagnostics)
    write(output/'PACKET_MANIFEST.json',dict(complete=True,rows=1241,arms=['F_COMMON','F_BI'],frames=summary,
        observations_sha256=sha(observations/'OBSERVATION_MANIFEST.json'),models_sha256=sha(output/'MODELS.json'),
        numeric_reference_scope='CAL46 only; no PRIMARY decoded or used for fitting',wall_seconds=time.monotonic()-start,
        calibration_policy='same pooled zero-centered floor recipe, each arm fitted to its own q; no changed-observation covariance transplantation',
        unused_metadata_removed_only=['selection policy','selected_candidates','cue_pool_sha256','full_points.provenance.policy'],
        complete_policy_provenance_retained_in='OBSERVATIONS_RAW/*/*/SELECTIONS.jsonl.gz'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('d1','campaign','observations','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.d1,a.campaign,a.observations,a.reference,a.output)
