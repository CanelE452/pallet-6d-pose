"""Model-blind, renderer-group-disjoint split; exact metadata inventory."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from . import common as C

MANIFEST=C.ROOT/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
TABLE=C.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'

def main():
    assert not (C.sdoc(2)/'SYNTHETIC_SPLIT_LOCK.json').exists()
    source=C.read(MANIFEST)['records']
    with np.load(TABLE) as loaded:table={k:loaded[k] for k in loaded.files}
    ix={str(s):i for i,s in enumerate(table['stems'])}
    plans=[r for r in map(json.loads,(C.STRUCT/'AUGMENTATION_PLAN.jsonl').read_text().splitlines()) if r['material']=='PLASTIC' and not r['real']]
    replay_paths=sorted({r['image'] for r in plans});assert len(replay_paths)==512
    excluded_sha={C.sha(p) for p in replay_paths};excluded_scenarios={r['scenario_id'] for r in source if r['image_sha256'] in excluded_sha}
    groups={};seen=set();excluded=Counter();inventory=[]
    for r in source:
        if r['image_sha256'] in excluded_sha or r['scenario_id'] in excluded_scenarios:excluded['replay_or_derivative']+=1;continue
        if r['image_sha256'] in seen:excluded['duplicate_sha']+=1;continue
        if not Path(r['image']).is_file():excluded['missing_image']+=1;continue
        if r['id'] not in ix:excluded['no_exact_geometry']+=1;continue
        i=ix[r['id']]
        if table['match_err'][i]>.05:excluded['geometry_mismatch']+=1;continue
        # P0 and texture variants of the same shard stay together; never split derivatives.
        group='G38_merged_archive' if r['source']=='G38' else 'P0_TEX_'+r['scenario_id'].split(':')[1].rsplit('_f',1)[0]
        row=dict(id=r['id'],source=r['source'],scenario=r['scenario_id'],group=group,image=dict(path=str(Path(r['image']).relative_to(C.ROOT)),sha256=r['image_sha256']),
            renderer_locator=r['renderer_annotation_locator_provenance_only'],table_index=i,hw=r['prepared_shape_hw'],pad=r['reflect_pad_px'])
        groups.setdefault(group,[]).append(row);seen.add(r['image_sha256']);inventory.append(row)
    largest=max(groups,key=lambda g:len(groups[g]));remaining=sorted([g for g in groups if g!=largest],key=lambda g:C.key(g))
    assigned={'TRAIN':[largest],'VAL':[],'TEST':[]}
    for part in ('VAL','TEST'):
        while sum(len(groups[g]) for g in assigned[part])<1024 and remaining:assigned[part].append(remaining.pop(0))
    assigned['TRAIN']+=remaining
    requested={'TRAIN':4096,'VAL':1024,'TEST':1024};minimum={'TRAIN':1024,'VAL':256,'TEST':256}
    selected={p:sorted([r for g in gg for r in groups[g]],key=lambda r:C.key(r['image']['sha256']))[:requested[p]] for p,gg in assigned.items()}
    assert all(len(selected[p])>=minimum[p] for p in selected),'SYNTHETIC_SELECTOR_DATA_INSUFFICIENT'
    # Each exact renderer frame is assigned by its source group before any output is read.
    framegroups={};shas={};rows=[];inputs=[]
    for p,rr in selected.items():
        for r in rr:
            C.verify(r['image']);assert r['image']['sha256'] not in shas;shas[r['image']['sha256']]=p
            assert r['scenario'] not in framegroups or framegroups[r['scenario']]==p;framegroups[r['scenario']]=p
            i=r['table_index'];fx,fy,cx,cy=table['K'][i]
            rr0=dict(r,split=p);rows.append(rr0)
            inputs.append(dict(id=r['id'],split=p,image=r['image'],hw=r['hw'],pad=r['pad'],
                K=[[fx,0,cx],[0,fy,cy],[0,0,1.]],dims=table['dims'][i].tolist()))
    C.freeze(C.sraw(2)/'SYNTH_INPUTS.json',inputs)
    C.freeze(C.sraw(2)/'SYNTH_RECORDS.json',rows)
    C.freeze(C.sdoc(2)/'SYNTHETIC_SPLIT_LOCK.json',dict(created_at=C.now(),seed=20260925,counts={p:len(r) for p,r in selected.items()},
        groups=assigned,ids={p:[r['id'] for r in rr] for p,rr in selected.items()},records=C.bind(C.sraw(2)/'SYNTH_RECORDS.json'),
        inputs=C.bind(C.sraw(2)/'SYNTH_INPUTS.json'),manifest=C.bind(MANIFEST),geometry=C.bind(TABLE),
        replay512_sha256=sorted(excluded_sha),excluded_derivative_scenarios=sorted(excluded_scenarios),
        frame_level=True,group_disjoint=True,image_sha_disjoint=True,renderer_derivative_disjoint=True,
        note='Group-disjoint: G38 merged archive assigned TRAIN; P0/TEX same shard kept together. VAL/TEST have low-elevation source shift. Base R0 previously fitted this broader synthetic population; scorer TEST is not base-model-unseen.'))
    C.freeze(C.sdoc(2)/'SYNTHETIC_POOL_AUDIT.json',dict(total_manifest=len(source),exact_table=len(table['stems']),available_after_exclusion=len(inventory),
        excluded=dict(excluded),group_sizes={g:len(r) for g,r in groups.items()},selected_sources={p:dict(Counter(r['source'] for r in rr)) for p,rr in selected.items()},
        geometry_origin='Existing exact renderer R,t,K,physical dimensions and stored camera-facing 3D corners Xcf; no historical image-area heuristic.',
        class_and_elevation_statistics='Appended only after prediction lock and exact synthetic label linkage.',status='SYNTHETIC_POOL_SUFFICIENT'))
    print('SYNTH_SPLIT',{p:len(r) for p,r in selected.items()},assigned,flush=True)

if __name__=='__main__':main()
