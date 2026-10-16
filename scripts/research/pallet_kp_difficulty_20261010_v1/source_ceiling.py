"""Read retained source supervision; optionally probe unchanged CPU heads.

No RGB synthesis, detector forward, pose solver, optimizer or renderer runs.
An ideal soft-target class choice tests the existing line/corner assembly.
Source annotation geometry provides a separately labeled oracle cuboid mask;
the original source predicted-pose mask was not cached and is unavailable.
"""
import argparse
import gzip
import json
from collections import Counter,defaultdict
from pathlib import Path
import time

import numpy as np
from scripts.research.pallet_observation_refiner_20261009_v1 import common as C
from scripts.research.pallet_observation_refiner_20261009_v1 import source_audit as S

DOC=C.WORKTREE/'_docs/experiments/pallet_kp_difficulty_20261010_v1'
CACHE=C.SCRATCH/'learned_cache'
EDGES=S.EDGES


def stats(a):
    a=np.asarray(a,float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),P90=float(np.quantile(a,.9)),P99=float(np.quantile(a,.99)),max=float(a.max())) if len(a) else dict(n=0)


def query_geometry(points):
    points=np.asarray(points,float);center=[];normal=[];valid=[]
    for a,b in EDGES:
        d=points[b]-points[a];length=np.linalg.norm(d);tangent=d/max(length,1e-6)
        for u in np.arange(1,8)/8:
            center.append((1-u)*points[a]+u*points[b]);normal.append([-tangent[1],tangent[0]])
            valid.append(bool(np.isfinite(d).all() and length>1e-6 and not np.any(np.all(points[[a,b]]==-1,axis=1))))
    return np.asarray(center),np.asarray(normal),np.asarray(valid)


def decode_choices(points,choices,query_valid=None):
    """NumPy copy of the frozen decoder's TLS and strongest incident pair."""
    center,normal,valid=query_geometry(points)
    if query_valid is not None:valid&=query_valid
    selected=valid&(choices<65);lines=[];corners=[]
    for edge,(a,b) in enumerate(EDGES):
        ids=np.flatnonzero(selected[edge*7:(edge+1)*7])+edge*7
        if len(ids)<2:continue
        xy=center[ids]+(choices[ids]-32)[:,None]*normal[ids];mid=xy.mean(0)
        _,singular,v=np.linalg.svd(xy-mid,full_matrices=False)
        if singular[0]<1e-6:continue
        n=np.array([-v[0,1],v[0,0]]);off=float(n@mid);support=(xy-mid)@v[0]
        lines.append(dict(edge=edge,normal=n.tolist(),offset=off,queries=ids.tolist(),support_length_px=float(support.max()-support.min()),
                          residual_rms_px=float(np.sqrt(np.mean((xy@n-off)**2)))))
    lookup={r['edge']:r for r in lines}
    for corner in range(8):
        incident=[e for e,pair in enumerate(EDGES) if corner in pair and e in lookup];pairs=[]
        for i,a in enumerate(incident):
            for b in incident[i+1:]:
                matrix=np.asarray([lookup[a]['normal'],lookup[b]['normal']]);det=float(np.linalg.det(matrix))
                if abs(det)>1e-6:pairs.append((-abs(det),a,b,matrix))
        if not pairs:continue
        det,a,b,matrix=min(pairs,key=lambda r:r[:3]);xy=np.linalg.solve(matrix,[lookup[a]['offset'],lookup[b]['offset']])
        if np.isfinite(xy).all():corners.append(dict(id=corner,xy=xy.tolist(),edges=[a,b],absolute_normal_determinant=-det))
    return dict(selected_queries=int(selected.sum()),selected_by_edge=selected.reshape(12,7).sum(1).tolist(),
                line_count=len(lines),corner_count=len(corners),lines=lines,corners=corners)


def oracle_hidden(g):
    camera=-g['R'].T@g['t'];rays=camera[None]-g['X']
    cosine=np.sign(g['X'])*rays/np.linalg.norm(rays,axis=1)[:,None]
    return np.flatnonzero(cosine.max(1)<-np.sin(np.deg2rad(2))).tolist()


def summarize(records):
    result={}
    for partition in ['all','train','calibration','source_test']:
        rr=records if partition=='all' else [r for r in records if r['partition']==partition]
        result[partition]=dict(families=len(rr),target_counts={k:sum(r['target_counts'][k] for r in rr) for k in ['positive','no_match','ignore']},
            before_mask_corners_histogram=dict(Counter(r['ideal']['corner_count'] for r in rr)),
            before_mask_in_frame_corners_histogram=dict(Counter(len(r['before_mask_in_frame_corner_ids']) for r in rr)),
            ideal_before_mask_ge4=sum(r['ideal']['corner_count']>=4 for r in rr),
            ideal_before_mask_in_frame_ge4=sum(len(r['before_mask_in_frame_corner_ids'])>=4 for r in rr),
            after_source_annotation_oracle_H_histogram=dict(Counter(len(r['after_source_annotation_oracle_H_corner_ids']) for r in rr)),
            after_source_annotation_oracle_H_in_frame_ge4=sum(len(r['after_source_annotation_oracle_H_in_frame_corner_ids'])>=4 for r in rr),
            target_counts_by_predicted_role={role:{k:sum(r['target_counts_by_predicted_role'][role][k] for r in rr) for k in ['positive','no_match','ignore']} for role in ['BOUNDARY','INTERNAL','UNAVAILABLE']},
            positive_offset_abs_px=stats([v for r in rr for v in r['positive_absolute_offsets_px']]),
            source_initial_corner_error_px=stats([v for r in rr for v in r['source_initial_corner_error_px']]),
            ideal_selected_corner_error_px=stats([v for r in rr for v in r['ideal_selected_corner_error_px']]),
            stored_predicted_initial_H='NA: source feature pose and hidden IDs not retained; no new PnP run')
    return result


def ceiling():
    begin=time.monotonic();manifest=C.read(CACHE/'CACHE_MANIFEST.json');families=S.selected_families()
    arrays={k:np.load(CACHE/(k+'.npy'),mmap_mode='r') for k in ['features','lo','hi','weight','valid']};records=[]
    assert len(manifest['records'])==len(families)==1024
    for cached in manifest['records']:
        i=cached['index'];lo=arrays['lo'][i];hi=arrays['hi'][i];weight=arrays['weight'][i];valid=arrays['valid'][i]
        points=np.asarray(cached['selected_points'],float);paths=S.locate(families[i]);ann=C.read(paths['label']);g=S.geometry(families[i],ann)
        targetxy=np.asarray(ann['objects'][0]['projected_cuboid'],float)
        role=np.asarray(arrays['features'][i,:,25:28,32]).argmax(1);positive=valid&(lo<65);none=valid&(lo==65);ignored=~valid
        choices=np.where(valid,np.where(weight<=.5,lo,hi),65).astype(int);ideal=decode_choices(points,choices)
        h,w=g['hw'];inside=[c['id'] for c in ideal['corners'] if 0<=c['xy'][0]<w and 0<=c['xy'][1]<h]
        hidden=oracle_hidden(g);after=[c['id'] for c in ideal['corners'] if c['id'] not in hidden]
        physical_edges=[e for e,(a,b) in enumerate(EDGES) if np.isclose(g['X'][a,1],g['X'][b,1])]
        counts={k:int(v.sum()) for k,v in [('positive',positive),('no_match',none),('ignore',ignored)]}
        byrole={name:{k:int((v&(role==j)).sum()) for k,v in [('positive',positive),('no_match',none),('ignore',ignored)]} for j,name in enumerate(['BOUNDARY','INTERNAL','UNAVAILABLE'])}
        records.append(dict(index=i,id=cached['id'],family=cached['family'],partition=cached['partition'],
            cache_record_sha256=C.digest(cached),source_annotation=C.binding(paths['label']),source_RGB_binding=cached['original_rgb'],
            frozen_selected_points=points.tolist(),raw_hw=g['hw'],physical_edge_ids=physical_edges,
            unsupported_vertical_edge_ids=sorted(set(range(12))-set(physical_edges)),
            targets=dict(lo=lo.tolist(),hi=hi.tolist(),weight=weight.tolist(),valid=valid.tolist()),
            target_counts=counts,target_counts_by_predicted_role=byrole,predicted_role_query_ids=role.tolist(),
            ideal_choices=choices.tolist(),ideal=ideal,before_mask_in_frame_corner_ids=inside,
            source_annotation_oracle_H=hidden,after_source_annotation_oracle_H_corner_ids=after,
            after_source_annotation_oracle_H_in_frame_corner_ids=sorted(set(after)&set(inside)),
            predicted_initial_H=None,predicted_initial_H_status='NOT_RETAINED; no new fit permitted',
            positive_absolute_offsets_px=np.abs(lo[positive]+weight[positive]-32).tolist(),
            source_initial_corner_error_px=np.linalg.norm(points[:8]-targetxy,axis=1).tolist(),
            ideal_selected_corner_error_px=[float(np.linalg.norm(np.asarray(c['xy'])-targetxy[c['id']])) for c in ideal['corners']],
            ground_truth_R_t_used_only_for_source_proxy_H=True,source_pose_fitted=False))
    DOC.mkdir(parents=True,exist_ok=True);C.save_rows(DOC/'SOURCE_CEILING_ROWS.jsonl.gz',records)
    output=dict(schema='unchanged_source_target_observation_ceiling_v1',families=1024,source='existing P0 only',summary=summarize(records),
        interpretation='Perfect valid target classes still require >=2 selected queries per edge and two nonparallel incident edges per corner. Counts are an observation ceiling, not pose availability.',
        ideal_selection='argmax of stored soft target distribution: lower bin at weight<=0.5; upper bin otherwise; NONE for target none and IGNORE',
        after_mask='source annotation cuboid geometric proxy ONLY, margin2deg; supplied source pose, not the uncached predicted initial mask; not actual mesh visibility',
        source_predicted_mask_recoverable_without_new_fit=False,pose_metrics_computed=False,
        new_detector_forwards=0,new_head_forwards=0,new_optimizer_updates=0,new_PnP_calls=0,new_auxiliary_rays=0,new_RGB=0,
        bindings=[C.binding(CACHE/'CACHE_MANIFEST.json')]+[C.binding(CACHE/(k+'.npy')) for k in arrays]+[C.binding(C.DOC/'SOURCE_FAMILY_SPLIT.json'),C.binding(Path(__file__))],
        rows=C.binding(DOC/'SOURCE_CEILING_ROWS.jsonl.gz'),wall_seconds=time.monotonic()-begin)
    C.write(DOC/'SOURCE_CEILING.json',output);print('SOURCE_CEILING_COMPLETE',json.dumps(output['summary']),flush=True)


def probe_frozen_heads():
    """24 CPU batch forwards of unchanged final heads, source-test128 only."""
    import torch
    from scripts.research.pallet_observation_refiner_20261009_v1 import model as M
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    begin=time.monotonic();manifest=C.read(CACHE/'CACHE_MANIFEST.json');arrays={k:np.load(CACHE/(k+'.npy'),mmap_mode='r') for k in ['features','lo','hi','weight','valid']}
    published=C.read(C.DOC/'TRAINING_COMPLETION.json');records=[];summary={};forwards=0;checkpointbindings=[]
    for arm in M.ARMS:
        path=C.SCRATCH/'learned_fits'/(arm+'.pt');binding=C.binding(path);expected=next(r['checkpoint'] for r in published['checkpoints'] if r['arm']==arm);assert binding['sha256']==expected['sha256']
        checkpoint=torch.load(path,map_location='cpu',weights_only=False);assert checkpoint['arm']==arm and checkpoint['steps']==3000
        assert checkpoint['protocol_sha256']==C.sha(C.DOC/'LEARNING_PROTOCOL.json');head=M.CorrespondenceHead().eval();head.requires_grad_(False);head.load_state_dict(checkpoint['model']);checkpointbindings.append(binding)
        rows=[]
        with torch.no_grad():
            for start in range(896,1024,16):
                ids=np.arange(start,start+16);x=torch.tensor(np.asarray(arrays['features'][ids]),dtype=torch.float32)
                logits=head(x,arm).numpy();forwards+=1
                for offset,i in enumerate(ids):
                    cached=manifest['records'][int(i)];scores=logits[offset];choice=scores.argmax(1);lo=arrays['lo'][i];weight=arrays['weight'][i];valid=arrays['valid'][i];pos=valid&(lo<65);none=valid&(lo==65);ignored=~valid
                    center,normal,queryvalid=query_geometry(cached['selected_points']);accept=(choice<65)&queryvalid;acceptedpositive=pos&accept
                    target=lo+weight-32;delta=choice-32;role=np.asarray(arrays['features'][i,:,25:28,32]).argmax(1);decoded=decode_choices(cached['selected_points'],choice)
                    row=dict(id=cached['id'],family=cached['family'],index=int(i),partition='source_test',arm=arm,
                        cache_record_sha256=C.digest(cached),choices=choice.tolist(),max_candidate_minus_none_margin=(scores[:,:65].max(1)-scores[:,65]).tolist(),
                        counts=dict(positive=int(pos.sum()),positive_accepted=int(acceptedpositive.sum()),no_match=int(none.sum()),no_match_false_accepted=int((none&accept).sum()),
                                    ignored=int(ignored.sum()),ignored_accepted_without_ground_truth=int((ignored&accept).sum()),selected_queries=int(accept.sum())),
                        positive_adopted_query_ids=np.flatnonzero(acceptedpositive).tolist(),
                        positive_target_offsets_px=target[pos].tolist(),accepted_positive_target_offsets_px=target[acceptedpositive].tolist(),
                        zero_offset_error_all_positive_px=np.abs(target[pos]).tolist(),
                        zero_offset_error_on_same_accepted_positive_set_px=np.abs(target[acceptedpositive]).tolist(),
                        learned_error_on_same_accepted_positive_set_px=np.abs(delta[acceptedpositive]-target[acceptedpositive]).tolist(),
                        selected_by_predicted_role=np.bincount(role[accept],minlength=3).tolist(),
                        accepted_positive_by_predicted_role=np.bincount(role[acceptedpositive],minlength=3).tolist(),
                        target_positive_by_predicted_role=np.bincount(role[pos],minlength=3).tolist(),
                        selected_by_edge=accept.reshape(12,7).sum(1).tolist(),decoded_corner_ids=[c['id'] for c in decoded['corners']],decoded_line_edges=[r['edge'] for r in decoded['lines']],
                        decoder_can_accept_source_ignored_queries_but_they_are_not_ground_truth_valid=True)
                    rows.append(row)
        positive=sum(r['counts']['positive'] for r in rows);accepted=sum(r['counts']['positive_accepted'] for r in rows);none=sum(r['counts']['no_match'] for r in rows)
        baseline=[v for r in rows for v in r['zero_offset_error_on_same_accepted_positive_set_px']];learned=[v for r in rows for v in r['learned_error_on_same_accepted_positive_set_px']]
        summary[arm]=dict(source_test_families=128,counts={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']},positive_adoption_rate=accepted/positive,
            no_match_false_acceptance=sum(r['counts']['no_match_false_accepted'] for r in rows)/none,
            zero_offset_all_positive_error_px=stats([v for r in rows for v in r['zero_offset_error_all_positive_px']]),
            zero_offset_same_accepted_set_error_px=stats(baseline),learned_same_accepted_set_error_px=stats(learned),
            learned_minus_zero_same_accepted_set_px=stats(np.asarray(learned)-baseline),
            learned_better_same_accepted_queries=int((np.asarray(learned)<baseline).sum()),learned_worse_same_accepted_queries=int((np.asarray(learned)>baseline).sum()),
            accepted_positive_by_role=np.sum([r['accepted_positive_by_predicted_role'] for r in rows],0).tolist(),
            target_positive_by_role=np.sum([r['target_positive_by_predicted_role'] for r in rows],0).tolist(),
            selected_by_role=np.sum([r['selected_by_predicted_role'] for r in rows],0).tolist(),
            decoded_corners_histogram=dict(Counter(len(r['decoded_corner_ids']) for r in rows)),
            decoded_ge4_corner_frames=sum(len(r['decoded_corner_ids'])>=4 for r in rows))
        records.extend(rows)
    C.save_rows(DOC/'SOURCE_FROZEN_HEAD_DIAGNOSTICS_ROWS.jsonl.gz',records)
    C.write(DOC/'SOURCE_FROZEN_HEAD_DIAGNOSTICS.json',dict(schema='unchanged_heads_source_test_CPU_probe_v1',summary=summary,
        forward_calls=forwards,batch=16,image_head_exposures=384,source_only_test_families=128,device='CPU float32',source_features='unchanged FP16 cache → float32',
        new_detector_forwards=0,new_optimizer_updates=0,new_PnP_calls=0,new_auxiliary_rays=0,new_RGB=0,
        original9000updates_and_original_scores_unchanged=True,no_model_reselection=True,CPU_vs_original_GPU_rounding_may_change_near_ties=True,
        checkpoints=checkpointbindings,cache_manifest=C.binding(CACHE/'CACHE_MANIFEST.json'),code=C.binding(Path(__file__)),rows=C.binding(DOC/'SOURCE_FROZEN_HEAD_DIAGNOSTICS_ROWS.jsonl.gz'),wall_seconds=time.monotonic()-begin))
    print('SOURCE_FROZEN_HEAD_DIAGNOSTICS_COMPLETE',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe-frozen-heads',action='store_true');args=p.parse_args()
    if args.probe_frozen_heads:probe_frozen_heads()
    else:ceiling()


if __name__=='__main__':main()
