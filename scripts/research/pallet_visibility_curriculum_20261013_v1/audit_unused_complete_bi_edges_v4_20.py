"""Read-only actual-chain coverage at still-complete B/I likelihoods.

Counts physical incident edge hypotheses, never statistically independent
observations. Uses frozen17c pixel/support validators and archived B topology;
I intersections are geometric diagnostics, not an adopted point/ownership
certificate. No NN, fresh NMS/line fit, model calibration, pose or GT reader.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import html

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from . import common as C
from . import observe_grazing_native_edge_chains_v4_18b as OBS
from . import image_chain_mixture_problem_v4_18 as MIX
from . import native_edge_topology_junctions_v4_17b as TOP
from . import render_refined_grazing_mixture_process_v4_18c as DRAW


OUTPUT = OBS.OUTPUT.parent/'unused_complete_bi_edge_evidence_v4_20'
CFG = OBS.NMS.DEFAULT_CONFIGURATION


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _pair(k, mate):
    return tuple(sorted((int(k), int(mate))))


def _validated_chain(ray, chain, k, mate, hw):
    _check(ray['measurement_source']=='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT'
        and ray.get('response_is_not_window_clipped') is True, 'Actual unclipped NMS source required')
    observed, mask, normal, offset, tangent, interval = TOP._validate_chain(chain,k,mate,CFG)
    actual = observed[mask]
    _check(chain.get('invented_bridge_points')==0 and (actual>=0).all()
        and (actual[:,0]<hw[1]).all() and (actual[:,1]<hw[0]).all(), 'Actual canvas pixels, no bridge')
    mp, mm = np.asarray(chain['mask_plus']), np.asarray(chain['mask_minus'])
    _check(mp.shape==mask.shape and mm.shape==mask.shape and np.isfinite(mp).all()
        and np.isfinite(mm).all(), 'Bound occupancy arrays required')
    if chain['mask_context_kind']=='TARGET_TARGET_INTERNAL':
        context=(mp>=CFG['minimum_inside'])&(mm>=CFG['minimum_inside'])
    else:
        context=(np.maximum(mp,mm)>=CFG['minimum_inside'])&(np.minimum(mp,mm)<=CFG['maximum_outside'])
        context&=np.abs(mp-mm)>=CFG['minimum_transition']
    _check(context[mask].all(), 'Positive archived target/context support required')
    ordered=np.asarray(sorted(map(tuple,np.round(actual,6))),dtype=float)
    return dict(normal=normal,offset=offset,tangent=tangent,interval=interval,actual=actual,
        token=hashlib.sha256(ordered.tobytes()).hexdigest(),
        lattice=np.asarray(chain['actual_inlier_image_lattice_xy'],int))


def _pairs(chains,xy,hw):
    """Existing .25 sine /11px support /12px shift; no fitted new line."""
    result=[];margin=float(CFG['along_start_px']+CFG['boundary_context_px'])
    for a,b in combinations(chains,2):
        if a['mate']==b['mate']:
            continue
        sine=abs(float(np.linalg.det(np.asarray([a['normal'],b['normal']]))))
        alias=a['token']==b['token']
        record=dict(refs=[a['ref'],b['ref']],native_neighbors=[a['mate'],b['mate']],
            intersection_sine=sine,same_actual_pixel_set=alias,
            both_physical_edges_unused=a['unused']and b['unused'],
            at_least_one_physical_edge_unused=a['unused']or b['unused'])
        if alias or sine<CFG['minimum_intersection_sine']:
            record['nonparallel_distinct_support']=False;result.append(record);continue
        point=np.linalg.solve(np.asarray([a['normal'],b['normal']]),-np.asarray([a['offset'],b['offset']]))
        gaps=[]
        for component in (a,b):
            v=float(component['tangent']@point);lo,hi=component['interval']
            gaps.append(max(float(lo-v),float(v-hi),0.))
        inframe=bool(0<=point[0]<hw[1]and 0<=point[1]<hw[0])
        shift=float(np.linalg.norm(point-xy))
        record.update(nonparallel_distinct_support=True,intersection_xy=point.tolist(),
            actual_finite_extrapolation_px=gaps,finite11_and_canvas_pass=bool(max(gaps)<=margin and inframe),
            shift_from_archived_complete_xy_px=shift,
            existing_finite_shift_canvas_pass=bool(max(gaps)<=margin and inframe and shift<=CFG['maximum_corner_shift_px']))
        result.append(record)
    return result


def _summary(corners):
    result={}
    for partition in ('CAL','TEST','REMAINDER'):
        result[partition]={}
        for state in ('B','I'):
            rows=[r for r in corners if r['partition']==partition and r['state']==state]
            result[partition][state]=dict(complete_corners=len(rows),frames=len({r['id']for r in rows}),
                proper_incident_edge_count_histogram=dict(sorted(Counter(r['proper_edge_count']for r in rows).items())),
                unused_physical_incident_edge_count_histogram=dict(sorted(Counter(r['unused_edge_count']for r in rows).items())),
                at_least_one_proper_edge=sum(r['proper_edge_count']>=1 for r in rows),
                at_least_two_proper_edges=sum(r['proper_edge_count']>=2 for r in rows),
                any_nonparallel_distinct_pair=sum(r['nonparallel_pair_count']>0 for r in rows),
                some_edge_pair_all_alternatives_nonparallel=sum(bool(r['robust_nonparallel_native_edge_pairs'])for r in rows),
                any_finite11_canvas_pair=sum(r['finite11_canvas_pair_count']>0 for r in rows),
                any_existing_finite_shift_canvas_pair=sum(r['finite_shift_canvas_pair_count']>0 for r in rows),
                pair_with_any_unused_physical_edge=sum(r['nonparallel_pair_involving_unused_edge_count']>0 for r in rows),
                pair_with_two_unused_physical_edges=sum(r['nonparallel_pair_both_unused_count']>0 for r in rows),
                existing_B_topology_candidate_corners=sum(r['B_topology_candidate_count']>0 for r in rows),
                existing_B_topology_with_unused_edge=sum(r['B_topology_candidates_with_unused_edge']>0 for r in rows),
                existing_B_topology_candidates_within_old3px=sum(r['B_topology_candidate_count']>0 and r['B_topology_within_old3px']for r in rows),
                after_any_exact_pixel_overlap_group_abstention_at_least_one=sum(r['after_exact_overlap_block_edge_count']>=1 for r in rows),
                after_any_lattice_overlap_group_abstention_at_least_one=sum(r['after_lattice_overlap_block_edge_count']>=1 for r in rows),
                after_any_exact_pixel_overlap_group_abstention_nonparallel=sum(r['after_exact_overlap_block_nonparallel']for r in rows),
                after_any_lattice_overlap_group_abstention_nonparallel=sum(r['after_lattice_overlap_block_nonparallel']for r in rows),
                after_any_exact_overlap_B_topology=sum(r['after_exact_overlap_B_topology']for r in rows),
                after_any_lattice_overlap_B_topology=sum(r['after_lattice_overlap_B_topology']for r in rows),
                distinct_chain_alternatives=sum(r['proper_chain_count']for r in rows))
    return result


def _reuse(all_refs, topology):
    """Cross-*physical*-edge support reuse; alternative multiplicity retained."""
    groups={}
    for value in all_refs.values():
        pair=value['physical_edge']
        groups.setdefault(pair,{})[value['token']]=value
    exact_union={pair:set().union(*(set(map(tuple,np.round(v['actual'],6)))for v in values.values()))for pair,values in groups.items()}
    lattice_union={pair:set().union(*(set(map(tuple,v['lattice']))for v in values.values()))for pair,values in groups.items()}
    blocked_exact=set();blocked_lattice=set();records=[]
    for a,b in combinations(sorted(groups),2):
        exact=exact_union[a]&exact_union[b];lattice=lattice_union[a]&lattice_union[b]
        identical=bool(set(groups[a])&set(groups[b]))
        if exact:blocked_exact.update((a,b))
        if lattice:blocked_lattice.update((a,b))
        sines=[abs(float(np.linalg.det(np.asarray([x['normal'],y['normal']]))))for x in groups[a].values()for y in groups[b].values()]
        records.append(dict(physical_edges=[list(a),list(b)],incident=bool(set(a)&set(b)),
            identical_actual_pixel_set_component_present=identical,
            any_actual_subpixel_overlap=bool(exact),shared_actual_subpixels=len(exact),
            any_actual_lattice_overlap=bool(lattice),shared_actual_lattice_pixels=len(lattice),
            any_nonparallel_component_pair=max(sines)>=CFG['minimum_intersection_sine'],
            all_component_pairs_nonparallel=min(sines)>=CFG['minimum_intersection_sine']))
    junctions=[];margin=float(CFG['along_start_px']+CFG['boundary_context_px'])
    for record in topology.values():
        for candidate in record.get('candidates',[]):
            refs=candidate['chain_refs'];a,b=[all_refs[(r['ray_index'],r['chain_index'])]for r in refs]
            exact=set(map(tuple,np.round(a['actual'],6)))&set(map(tuple,np.round(b['actual'],6)))
            lattice=set(map(tuple,a['lattice']))&set(map(tuple,b['lattice']))
            q=np.asarray(candidate['intersection_xy']);shared=np.asarray(sorted(lattice),float).reshape(-1,2)
            within=all(np.max(np.abs((shared-q)@x['tangent']))<=margin+1e-9 for x in(a,b))if len(shared)else None
            junctions.append(dict(native_corner=record['native_corner'],chain_refs=refs,
                actual_subpixel_overlap=bool(exact),actual_lattice_overlap=bool(lattice),
                shared_lattice_count=len(lattice),all_shared_lattice_within_existing11_tangent_context=within,
                removed_by_any_exact_overlap_group_abstention=a['physical_edge']in blocked_exact or b['physical_edge']in blocked_exact,
                removed_by_any_lattice_overlap_group_abstention=a['physical_edge']in blocked_lattice or b['physical_edge']in blocked_lattice))
    return dict(physical_group_count=len(groups),cross_native_group_pairs=records,
        identical_physical_edge_counted_once=True,
        groups_blocked_by_any_exact_overlap=[list(x)for x in sorted(blocked_exact)],
        groups_blocked_by_any_lattice_overlap=[list(x)for x in sorted(blocked_lattice)],
        B_topology_junction_support_reuse=junctions),blocked_exact,blocked_lattice


def audit():
    if OUTPUT.exists():
        raise FileExistsError('Preserve prior evidence')
    op=C.read(OBS.OUTPUT/'PROTOCOL.json');os=C.read(OBS.OUTPUT/'OBSERVATION_SEAL.json')
    gp=C.read(OBS.GRAZING/'PROTOCOL.json');gs=C.read(OBS.GRAZING/'INPUT_SEAL.json')
    _check(os['rows']==238 and gs['rows']==238 and os['ordered_ids']==gs['ordered_ids'], 'Uniform238 join')
    _check(C.sha(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz')==os['observations_sha256'], 'Observation seal')
    _check(C.sha(OBS.GRAZING/'INPUTS.jsonl.gz')==gs['input_sha256'], 'Complete/partial profile seal')
    for p in (op,gp):
        for key in ('source_sha256','input_sha256','protected_user_sha256'):
            OBS.OLD._verify_map(p[key])
    paths=[Path(__file__),Path(OBS.__file__),Path(OBS.NMS.__file__),Path(TOP.__file__),Path(MIX.__file__),Path(DRAW.__file__)]
    paths += [OBS.OUTPUT/n for n in ('PROTOCOL.json','PREPARATION_SEAL.json','INPUTS.jsonl.gz','OBSERVATION_SEAL.json','OBSERVATIONS.jsonl.gz')]
    paths += [OBS.GRAZING/n for n in ('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz')]
    bindings={str(p.resolve().relative_to(C.ROOT)):C.sha(p)for p in paths}
    OUTPUT.mkdir(parents=True)
    C.write(OUTPUT/'PREPARATION_SEAL.json',dict(source_sha256=bindings,rows=238,
        before_chain_inventory=True,reference_files_opened=False,no_new_thresholds_or_fits=True))
    profile={}
    for t in C.rows(OBS.GRAZING/'INPUTS.jsonl.gz'):
        boundary=t['boundary_merged'];refs=set()
        for b in boundary:
            k=b['native_corner']
            _check(t['states'][k]in('B','I')and not t['initial_self_hidden'][k], 'Partial state first')
            for line in b['lines']:
                mate=line['provenance'].get('native_neighbor')
                if isinstance(mate,int)and mate in TOP.NEIGHBORS[k]and not t['initial_self_hidden'][mate]:
                    refs.add(_pair(k,mate))
        profile[t['id']]=dict(states=t['states'],visible=t['spec']['visible'],partial_ids={b['native_corner']for b in boundary},
            refs=refs,boundary_digest=OBS.OLD._digest(boundary),K=t['spec']['K'])
    _check(list(profile)==os['ordered_ids'],'Profile exact order')
    rows=[];corners=[];validation=Counter();selected_source={};reuse_rows=[]
    for row in C.rows(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz'):
        fid=row['id'];p=profile[fid];states=row['states'];hidden=row['initial_geometry']['initial_self_hidden']
        _check(states==p['states']and row['visible_measurements']==p['visible'],'Same frozen visible states/q/Sigma')
        _check(all(bool(hidden[k])==(states[k]=='S')for k in range(8)),'Permanent S')
        obs=row['chain_observation'];topology={r['native_corner']:r for r in row['junction_proposals']['records']}
        _check(obs['configuration']==CFG,'Frozen17c recipe')
        _check(obs['S_local_queries']==0 and obs['U_corner_queries']==0 and obs['U_coordinate_reads']==0,'S/U contract')
        hw=row['initial_geometry']['image_hw'];eligible={};ray_summary=[]
        for measurement in p['visible']:
            k=measurement['native_corner']
            _check(not hidden[k]and states[k]in('B','I'),'Visible state before coordinate')
            if k not in p['partial_ids']:
                eligible[k]=measurement
        per_corner={k:[] for k in eligible};all_refs={}
        for ri,ray in enumerate(obs['edge_records']):
            k,mate=ray['native_corner'],ray['native_neighbor']
            if hidden[k]or states[k]not in('B','I')or hidden[mate]or states[mate]=='S':
                _check(not ray.get('pixel_queried',False),'S gate before queries');continue
            _check(mate in TOP.NEIGHBORS[k],'Incident native edge')
            ray_summary.append(dict(native_corner=k,native_neighbor=mate,supported=ray.get('supported',False),reason=ray.get('reason'),
                complete_anchor=k in eligible,overflow=ray.get('component_linking_audit',{}).get('combinatorial_overflow_abstained',False)))
            if not ray.get('supported'):continue
            for ci,chain in enumerate(ray['chains']):
                v=_validated_chain(ray,chain,k,mate,hw);validation['validated_chains']+=1
                expected='TARGET_BACKGROUND_OUTLINE'if states[k]=='B'else'TARGET_TARGET_INTERNAL'
                role='OUTLINE_VISIBLE_HIDDEN_FACES'if states[k]=='B'else'INTERNAL_TWO_VISIBLE_FACES'
                proper=chain['mask_context_kind']==expected and ray['face_association']['role']==role
                unknown=states[mate]in('U','O','T')
                if unknown:
                    _check(ray['neighbor_image_coordinate_used']is False,'No unknown image coordinate')
                    lattice=np.asarray(chain['actual_inlier_image_lattice_xy'])
                    guide=np.asarray(row['initial_geometry']['initial_projection_xy'][mate])
                    _check((np.linalg.norm(lattice-guide,axis=1)>=2*CFG['boundary_context_px']-1e-9).all(),'Existing14px unknown-mate lattice protection')
                    validation['unknown_mate_chains_protected']+=1
                item={**v,'ref':dict(ray_index=ri,chain_index=ci),'mate':mate,'proper':proper,
                    'unused':_pair(k,mate)not in p['refs'],'context':chain['mask_context_kind'],
                    'physical_edge':_pair(k,mate)}
                all_refs[(ri,ci)]=item
                if k in eligible:per_corner[k].append(item)
        reuse,blocked_exact,blocked_lattice=_reuse(all_refs,topology)
        reuse_rows.append(dict(id=fid,partition=row['partition'],**reuse))
        frame_corners=[]
        for k,measurement in eligible.items():
            values=per_corner[k];proper=[v for v in values if v['proper']]
            # Same physical edge + identical actual-pixel set counts once.
            distinct={}
            for v in proper:distinct.setdefault((v['mate'],v['token']),v)
            proper=list(distinct.values());pairs=_pairs(proper,np.asarray(measurement['xy']),hw)
            nonparallel=[v for v in pairs if v['nonparallel_distinct_support']]
            without_exact=[v for v in proper if v['physical_edge']not in blocked_exact]
            without_lattice=[v for v in proper if v['physical_edge']not in blocked_lattice]
            after_exact_pairs=_pairs(without_exact,np.asarray(measurement['xy']),hw)
            after_lattice_pairs=_pairs(without_lattice,np.asarray(measurement['xy']),hw)
            robust=[]
            for a,b in combinations(sorted({v['mate']for v in proper}),2):
                group=[v for v in pairs if set(v['native_neighbors'])=={a,b}]
                if group and all(v['nonparallel_distinct_support']for v in group):robust.append([a,b])
            top=topology[k];candidates=top.get('candidates',[])
            for candidate in candidates:
                _check(all((r['ray_index'],r['chain_index'])in all_refs for r in candidate['chain_refs']),'B candidate actual-chain references')
            new_candidates=sum(any(_pair(k,r['native_neighbor'])not in p['refs']for r in c['chain_refs'])for c in candidates)
            item=dict(id=fid,partition=row['partition'],native_corner=k,state=states[k],source=measurement['source'],
                complete_archived_xy=measurement['xy'],proper_edge_count=len({v['mate']for v in proper}),
                unused_edge_count=len({v['mate']for v in proper if v['unused']}),
                proper_chain_count=len(proper),other_context_chain_count=sum(not v['proper']for v in values),
                chain_refs=[dict(**v['ref'],native_neighbor=v['mate'],actual_pixel_sha256=v['token'],
                    oldpartial_physical_edge_already_used=not v['unused'],mask_context_kind=v['context'])for v in proper],
                nonparallel_pair_count=len(nonparallel),robust_nonparallel_native_edge_pairs=robust,
                finite11_canvas_pair_count=sum(v.get('finite11_and_canvas_pass',False)for v in nonparallel),
                finite_shift_canvas_pair_count=sum(v.get('existing_finite_shift_canvas_pass',False)for v in nonparallel),
                nonparallel_pair_involving_unused_edge_count=sum(v['at_least_one_physical_edge_unused']for v in nonparallel),
                nonparallel_pair_both_unused_count=sum(v['both_physical_edges_unused']for v in nonparallel),
                B_topology_candidate_count=len(candidates),B_topology_candidates_with_unused_edge=new_candidates,
                B_topology_within_old3px=top.get('geometrically_unique_within_existing_3px_rule',False),
                after_exact_overlap_block_edge_count=len({v['mate']for v in without_exact}),
                after_lattice_overlap_block_edge_count=len({v['mate']for v in without_lattice}),
                after_exact_overlap_block_nonparallel=any(v['nonparallel_distinct_support']for v in after_exact_pairs),
                after_lattice_overlap_block_nonparallel=any(v['nonparallel_distinct_support']for v in after_lattice_pairs),
                after_exact_overlap_B_topology=any(all(_pair(k,r['native_neighbor'])not in blocked_exact for r in c['chain_refs'])for c in candidates),
                after_lattice_overlap_B_topology=any(all(_pair(k,r['native_neighbor'])not in blocked_lattice for r in c['chain_refs'])for c in candidates),
                B_topology_record=top if states[k]=='B'else None,pair_diagnostics=pairs,
                I_pair_is_geometric_potential_not_topology_or_ownership_certificate=states[k]=='I')
            corners.append(item);frame_corners.append(item)
        rows.append(dict(id=fid,partition=row['partition'],states=states,visible_BI_count=len(p['visible']),
            complete_BI_count=len(eligible),old_partial_count=len(p['partial_ids']),oldpartial_physical_edges=[list(v)for v in sorted(p['refs'])],
            ray_inventory=ray_summary,corners=frame_corners))
    _check([r['id']for r in rows]==os['ordered_ids'],'All238, including zero support')
    summary=_summary(corners)
    reuse_summary={}
    for part in('CAL','TEST','REMAINDER'):
        frames=[r for r in reuse_rows if r['partition']==part]
        value=dict(frames=len(frames),physical_groups=sum(r['physical_group_count']for r in frames),
            groups_blocked_any_exact_overlap=sum(len(r['groups_blocked_by_any_exact_overlap'])for r in frames),
            groups_blocked_any_lattice_overlap=sum(len(r['groups_blocked_by_any_lattice_overlap'])for r in frames))
        for incident in(True,False):
            values=[v for r in frames for v in r['cross_native_group_pairs']if v['incident']==incident]
            value['incident'if incident else'nonincident']=dict(pairs=len(values),
                identical_actual_sets=sum(v['identical_actual_pixel_set_component_present']for v in values),
                any_exact_overlap=sum(v['any_actual_subpixel_overlap']for v in values),
                partial_exact_overlap_without_identical_sets=sum(v['any_actual_subpixel_overlap']and not v['identical_actual_pixel_set_component_present']for v in values),
                any_lattice_overlap=sum(v['any_actual_lattice_overlap']for v in values),
                overlapping_and_any_nonparallel=sum(v['any_actual_lattice_overlap']and v['any_nonparallel_component_pair']for v in values))
        junctions=[v for r in frames for v in r['B_topology_junction_support_reuse']]
        value['B_topology_candidates']=dict(count=len(junctions),
            exact_overlap=sum(v['actual_subpixel_overlap']for v in junctions),
            lattice_overlap=sum(v['actual_lattice_overlap']for v in junctions),
            overlap_all_within_existing11=sum(v['all_shared_lattice_within_existing11_tangent_context']is True for v in junctions),
            removed_by_any_exact_overlap_group_rule=sum(v['removed_by_any_exact_overlap_group_abstention']for v in junctions),
            removed_by_any_lattice_overlap_group_rule=sum(v['removed_by_any_lattice_overlap_group_abstention']for v in junctions))
        reuse_summary[part]=value
    C.write(OUTPUT/'CROSS_NATIVE_SUPPORT_REUSE.json',dict(summary=reuse_summary,rows=reuse_rows,
        exact_subpixel_rounding_decimals=6,original_integer_lattice_overlap_reported_separately=True,
        policy_is_diagnostic_only_no_groups_actually_removed=True,
        overlap_uses_any_component_in_any_supported_group_before_coverage_exclusion=True,
        shared_native_endpoint_overlap_is_not_automatically_wrong_association=True,
        statistical_independence_claim=False,new_thresholds_or_fits=0))
    C.write(OUTPUT/'COVERAGE.json',dict(schema='UNUSED_COMPLETE_BI_ACTUAL_EDGE_COVERAGE_V4_20',rows=rows,summary=summary,
        validation=dict(validation),all238_kept=True,source_bindings=bindings,
        definitions=dict(complete='Frozen visible B/I with no boundary_merged record; full2D likelihood remains',
            unused='Physical edge not referenced by any old partial-normal factor, even at the opposite endpoint',
            nonparallel='Two distinct incident native edges and distinct actual-pixel sets; existing |sin|>=.25',
            robust='There exists an edge pair for which every retained cross-alternative pair meets that nonparallel criterion',
            B='Two external lines plus archived same-exterior-mask topology/finite11/shift12 checks',
            I='Internal-only line pairs; finite11/shift12 diagnostics are not a new I topology certificate'),
        no_statistical_independence_native_ownership_or_precise_corner_claim=True,
        old_partial_only_CAL_noise_not_transferred_or_refit=True,new_queries_fits_NN_GT_reads=0))
    # Representative choices use only coverage categories and known frame IDs, no metric/GT file.
    selections=[]
    categories=[('CAL B topology support control',lambda r:r['partition']=='CAL'and r['state']=='B'and r['B_topology_candidate_count']>0),
        ('CAL I internal-pair support control',lambda r:r['partition']=='CAL'and r['state']=='I'and r['finite_shift_canvas_pair_count']>0),
        ('TEST unused nonparallel edges',lambda r:r['partition']=='TEST'and r['nonparallel_pair_involving_unused_edge_count']>0),
        ('REMAINDER unused nonparallel edges',lambda r:r['partition']=='REMAINDER'and r['nonparallel_pair_both_unused_count']>0)]
    for label,predicate in categories:
        chosen=next((r for r in corners if predicate(r)),None)
        if chosen:selections.append((label,chosen))
    for fid in ('wood_night_01:031426','wood_night_01:030743'):
        group=[r for r in corners if r['id']==fid]
        if group:selections.append(('Known diagnostic frame; no metric selection',max(group,key=lambda r:(r['proper_edge_count'],r['proper_chain_count']))))
    seen=set();selections=[(a,b)for a,b in selections if not((b['id'],b['native_corner'])in seen or seen.add((b['id'],b['native_corner'])))]
    targets={r['id']for _,r in selections}
    for row in C.rows(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz'):
        if row['id']in targets:selected_source[row['id']]=row
    DRAW.DRAW._font();assets=OUTPUT/'assets';assets.mkdir();panels=[]
    for index,(label,item)in enumerate(selections):
        row=selected_source[item['id']];path=C.ROOT/row['image'];_check(C.sha(path)==row['image_sha256'],'Representative actual RGB')
        bindings[row['image']]=row['image_sha256'];image=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        fig,ax=DRAW.DRAW._base(image,item['id']+' · complete '+item['state']+str(item['native_corner']))
        k=item['native_corner'];q=np.asarray(item['complete_archived_xy'])
        DRAW.DRAW._wire(ax,np.asarray(row['initial_geometry']['initial_projection_xy']),'#929aa2',ls='--')
        DRAW.DRAW._points(ax,{k:q},'#d423cb','x',prefix='고정q '+item['state'],offset=(7,-15))
        palette=('#00afc3','#ee9e00','#875ad5')
        mates=sorted({r['native_neighbor']for r in item['chain_refs']})
        for ref in item['chain_refs']:
            ray=row['chain_observation']['edge_records'][ref['ray_index']];chain=ray['chains'][ref['chain_index']]
            actual=np.asarray(chain['observed_points'])[np.asarray(chain['inlier_mask'],bool)]
            color=palette[mates.index(ref['native_neighbor'])%3];ax.scatter(actual[:,0],actual[:,1],s=9,c=color,alpha=.6)
            n=np.asarray(chain['normal']);t=np.array([-n[1],n[0]]);lo,hi=chain['actual_inlier_tangent_interval_px']
            segment=-float(chain['offset'])*n+np.asarray([lo,hi])[:,None]*t
            ax.plot(segment[:,0],segment[:,1],c=color,lw=1.1,alpha=.6)
        candidates=item['B_topology_record']['candidates']if item['state']=='B'else[v for v in item['pair_diagnostics']if v.get('existing_finite_shift_canvas_pass')]
        for candidate in candidates:
            point=candidate['intersection_xy'];ax.plot(*point,'o',ms=5,mfc='none',mec='#cf6900',alpha=.65)
        h,w=image.shape[:2];initial=np.asarray(row['initial_geometry']['initial_projection_xy'])
        lo=np.maximum(initial.min(0)-22,0);hi=np.minimum(initial.max(0)+22,[w,h])
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
        caption=('서로다른 native edge '+str(item['proper_edge_count'])+'개 / 기존partial에 미사용 '+str(item['unused_edge_count'])+
            '개. 색점/선=모든 실제 인라이어/유한구간; 원=미채택 교점 후보; 자홍×=유지된 q; 회색=초기 모델. '+
            ('B 외곽 topology 검사'if item['state']=='B'else'I 내부선 기하 진단, ownership 인증 아님'))
        dest=assets/f'example{index}.png';DRAW.DRAW._save(fig,dest,caption)
        panels.append(dict(label=label,id=item['id'],native_corner=k,asset=str(dest.relative_to(OUTPUT)),caption=caption))
    known={fid:[r for r in corners if r['id']==fid]for fid in('wood_night_01:031426','wood_night_01:030743','wood_night_01:031344','wood_night_01:032645')}
    C.write(OUTPUT/'REPRESENTATIVES.json',dict(selection_uses_only_coverage_and_preexisting_frame_IDs=True,
        no_pose_error_or_GT_files_read=True,examples=panels,known_case_complete_corner_diagnostics=known))
    table=[]
    for partition in ('CAL','TEST','REMAINDER'):
        for state in ('B','I'):
            x=summary[partition][state]
            table.append('<tr>'+''.join('<td>'+str(v)+'</td>'for v in(partition,state,x['complete_corners'],x['at_least_one_proper_edge'],
                x['any_nonparallel_distinct_pair'],x['some_edge_pair_all_alternatives_nonparallel'],x['pair_with_two_unused_physical_edges'],
                x['existing_B_topology_candidate_corners']if state=='B'else x['any_existing_finite_shift_canvas_pair']))+'</tr>')
    intro='완전 B/I 비용으로 남아 있는 점 중 실제 선 지지를 살핀 읽기 전용 감사입니다. B 외곽과 I 내부선을 섞지 않았습니다. edge 두 개는 통계적 독립을 뜻하지 않으며, 여러 parallel/subset 대안을 새 정보로 세지 않았습니다. I의 두 선은 기하 가능성이지 실제 native 코너·ownership 인증이 아닙니다. 모든 교점은 미채택이며 공분산/상태/q/기존 방법을 바꾸지 않았습니다. 기존 oldpartial CAL 모델을 이 새 모집단에 전이하지 않았습니다.'
    sections=''.join('<h2>'+html.escape(p['label'])+'</h2><p>'+html.escape(p['id'])+'</p><img src="'+p['asset']+'"><p>'+p['caption']+'</p>'for p in panels)
    (OUTPUT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>완전 B/I의 미사용 실제 선 지지</title><style>body{font-family:system-ui;color:#173044;max-width:1100px;margin:auto;padding:16px}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #bbc;padding:7px}.wrap{overflow-x:auto}p{line-height:1.6}</style><h1>완전 B/I 기록의 미사용 실제 선 지지</h1><p>'+intro+'</p><div class="wrap"><table><tr><th>집합</th><th>역할</th><th>완전 점</th><th>≥1 edge</th><th>어떤 대안쌍 비평행</th><th>모든 대안쌍 비평행인 edge쌍</th><th>두 edge 모두 미사용</th><th>B topology / I 유한·shift</th></tr>'+''.join(table)+'</table></div><p><a href="COVERAGE.json">전238 프레임·모든 코너/대안 검산</a> · <a href="REPRESENTATIVES.json">표시 선택·기존 실패사례 기록</a></p>'+sections+'</html>')
    for path,digest in bindings.items():_check(C.sha(C.ROOT/path)==digest,'Bound source/input unchanged')
    C.write(OUTPUT/'EVIDENCE_SEAL.json',dict(source_sha256=C.sha(Path(__file__)),input_sha256=bindings,
        assets={str(p.relative_to(OUTPUT)):C.sha(p)for p in OUTPUT.rglob('*')if p.is_file()},rows=238,
        new_NN_line_pose_fits_or_reference_reads=0,production_changes=0))
    print(dict(output=str(OUTPUT),summary=summary,validation=dict(validation)),flush=True)


if __name__=='__main__':audit()
