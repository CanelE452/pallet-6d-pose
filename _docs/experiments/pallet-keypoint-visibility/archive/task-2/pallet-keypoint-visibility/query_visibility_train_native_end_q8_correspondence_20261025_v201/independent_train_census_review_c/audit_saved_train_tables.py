"""Independent scalar check of bounded TRAIN metadata and cross-artifact tables.

Never import/run A/B producers, numpy, model code, or referenced binary packets.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
import stat

H=Path(__file__).absolute().parent
V=H.parent
B=V/'target_correspondence_b'
A=V/'q8_provenance_a'
ACCESSES=[]


def raw(p):
    p=Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert all('v112' not in x.casefold() for x in p.parts)
    assert p.suffix in ('.json','.csv','.py') or p in [V/'scientific_readout_v1/TRAIN_QUERY_AND_WEAK_END.png',V/'scientific_readout_v1/TRAIN_QUERY_AND_WEAK_END.svg']
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        final=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(final).st_mode)
    with os.fdopen(final,'rb')as f:data=f.read()
    b=dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),size_bytes=len(data))
    if b not in ACCESSES:ACCESSES.append(b)
    return data


def read_bound(b):
    data=raw(b['path']);assert hashlib.sha256(data).hexdigest()==b['sha256']
    return list(csv.DictReader(io.StringIO(data.decode())))if Path(b['path']).suffix=='.csv'else json.loads(data)


def median(values):
    x=sorted(values);n=len(x)
    return (x[(n-1)//2]+x[n//2])/2


def stats(values):
    return dict(N=len(values),min=min(values),median=median(values),max=max(values))


def geometry(q,L,box):
    """Invert first sample center and pitch, not B's direct projection expression."""
    assert L>0 and all(math.isfinite(x)for x in [*q,L,*box])
    x0,y0,x1,y1=box;assert x0<=x1 and y0<=y1
    pitch=L/96
    first=(q[0]-47.5*pitch,q[1]-47.5*pitch)
    projected=[(box[k]-first[k%2])/pitch for k in range(4)]
    # Separate raw interval clamp for nearest region distance.
    closest=(min(max(q[0],x0),x1),min(max(q[1],y0),y1))
    d2=(q[0]-closest[0])**2+(q[1]-closest[1])**2
    dx=((x0+x1)/2-q[0])/pitch;dy=((y0+y1)/2-q[1])/pitch
    return dict(box=projected,inside=x0<=q[0]<=x1 and y0<=q[1]<=y1,
                region=math.sqrt(d2)/pitch,center=math.sqrt(dx*dx+dy*dy),
                dxy=[dx,dy],wh=[(x1-x0)/pitch,(y1-y0)/pitch],extent=all(0<=x<=95 for x in projected))


def run():
    checks=0;maxdelta=0.
    def exact(a,b):
        nonlocal checks
        assert a==b,(a,b);checks+=1
    def close(a,b):
        nonlocal checks,maxdelta
        d=abs(a-b);maxdelta=max(maxdelta,d);assert d<1e-10,(a,b,d);checks+=1
    def vector(a,b):
        exact(len(a),len(b))
        for x,y in zip(a,b):
            if isinstance(x,(list,tuple)):vector(x,y)
            else:close(x,y)
    exact(geometry((0.,0.),192.,[6.,8.,10.,12.])['region'],5.)
    exact(geometry((100.,200.),96.,[99.,199.,101.,201.])['box'],[46.5,46.5,48.5,48.5])
    summary=json.loads(raw(B/'SUMMARY.json'))
    original=read_bound(summary['inputs']['all109'])['triplets109']
    mapping=read_bound(summary['inputs']['mapping'])
    selection=read_bound(summary['inputs']['selection'])
    saved=read_bound(summary['outputs']['per109'])
    groupscsv=read_bound(summary['outputs']['source_groups'])
    canonicalcsv=read_bound(summary['outputs']['native_canonical'])
    raw(B/'analyze_train_target_correspondence.py')
    raw(V/'build_train_correspondence_figure.py')
    fig=json.loads(raw(V/'scientific_readout_v1/COMPLETE.json'))
    rootview=json.loads(raw(V/'ROOT_ACTUAL_TRAIN_METADATA_FIGURE_QA.json'))
    bview=json.loads(raw(B/'ACTUAL_FIXED_TRAIN_VISUAL_OBSERVATIONS.json'))
    exact(next(x for x in ACCESSES if x['path']==str(B/'ACTUAL_FIXED_TRAIN_VISUAL_OBSERVATIONS.json'))['sha256'],'b95a19e326eb81a699727ca40537a26377a15da678e09b64c6a20426fadb611d')
    exact(len(original),109);exact(len(saved),109)
    bykey={(x['domain'],x['attempt_id']):x for x in original};exact(len(bykey),109)
    csvby={(x['domain'],x['attempt_id']):x for x in saved};exact(set(csvby),set(bykey))
    groups=Counter();canonical=Counter();native=Counter();recomputed={};rows_by_domain=defaultdict(list)
    for key,source in bykey.items():
        a=source['association'];r=csvby[key];d=source['domain'];n=source['selected_native_slot'];native[n]+=1
        exact(n,a['selected_native_slot']);exact(source['selected_query_available'],True)
        exact(a['all_other_native_targets_UNKNOWN'],True);exact(a['association_only_not_nativephysicalgold'],True)
        exact(a['training_admission'],False);exact(a['proof_status'],'CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED')
        group='BASE'+str(a['source_base_index']).zfill(3)if a['source_base_index']is not None else'FRAME'+str(a['source_frame_index']).zfill(4)
        groups[(d,group)]+=1;canonical[(d,n,str(a['canonical_index_metadata']))]+=1
        exact(r['source_group'],group);exact(r['component_id'],source['component_id']);exact(int(r['selected_native_slot']),n)
        for name in ['source_base_index','source_frame_index','source_case_id','source_donor_id','canonical_index_metadata']:
            exact(r[name],''if a[name]is None else str(a[name]))
        for name in ['canonical_authority','weak_authority','end_locality_id']:exact(r[name],a[name])
        q=a['original_selected_query_xy'];L=a['original_selected_L']
        vector(q,[float(r['original_q_x']),float(r['original_q_y'])]);close(L,float(r['original_L']))
        boxes=a['candidate_enclosures_fullRGB'];exact(json.loads(r['candidate_boxes_fullRGB_json']),boxes)
        gs=[geometry(q,L,b)for b in boxes];exact(int(r['candidate_count']),len(gs))
        vector([g['box']for g in gs],a['candidate_enclosures_original96_diagnostic'])
        vector([g['box']for g in gs],json.loads(r['candidate_boxes_crop96_json']))
        vector([g['dxy']for g in gs],json.loads(r['candidate_center_dxy_crop_px_json']))
        vector([g['wh']for g in gs],json.loads(r['candidate_width_height_crop_px_json']))
        inside=any(g['inside']for g in gs);near=min(g['region']for g in gs);center=min(g['center']for g in gs)
        exact(r['query_inside_any_candidate'],str(inside));exact(r['all_candidates_in_original_sample_coordinate_extent'],str(all(g['extent']for g in gs)))
        close(float(r['query_crop_x']),47.5);close(float(r['query_crop_y']),47.5)
        close(float(r['min_query_distance_to_candidate_region_crop_px']),near)
        close(float(r['min_candidate_center_distance_crop_px']),center)
        close(float(r['max_candidate_center_distance_crop_px']),max(g['center']for g in gs))
        exact(r['stored_proof_status'],a['proof_status']);exact(r['physical_native_certified'],'False');exact(r['training_admission'],'False')
        rows_by_domain[d].append(dict(source=source,inside=inside,near=near,center=center,widths=[g['wh'][0]for g in gs],heights=[g['wh'][1]for g in gs],group=group))
    for d,rr in rows_by_domain.items():
        population=dict(triplets=len(rr),declared_known_weak_labels=3*len(rr),all8_variant_rows=24*len(rr),other7_variant_targets_UNKNOWN=21*len(rr),native_known_triplets={str(k):sum(x['source']['selected_native_slot']==k for x in rr)for k in range(8)},unique_source_groups=len({x['group']for x in rr}),unique_source_components=len({x['source']['component_id']for x in rr}),unique_cases=len({x['source']['association']['source_case_id']for x in rr if x['source']['association']['source_case_id']is not None}),query_inside_any_candidate=sum(x['inside']for x in rr),query_outside_all_candidates=sum(not x['inside']for x in rr),canonical_unknown=sum(x['source']['association']['canonical_index_metadata']is None for x in rr))
        for k,values in [('region_distance_crop_px',[x['near']for x in rr]),('center_distance_crop_px',[x['center']for x in rr]),('candidate_width_crop_px',[z for x in rr for z in x['widths']]),('candidate_height_crop_px',[z for x in rr for z in x['heights']])]:population[k]=stats(values)
        for k,x in population.items():
            y=summary['domain_summary'][d][k]
            if k.endswith('_px'):
                for kk in x:close(x[kk],y[kk])
            else:exact(x,y)
        recomputed[d]=population
        f=fig['stats'][d];exact(f['triplets'],population['triplets']);exact(f['query_inside_weak_enclosure'],population['query_inside_any_candidate']);exact(f['unique_source_groups'],population['unique_source_groups']);close(f['region_distance_median'],population['region_distance_crop_px']['median']);close(f['region_distance_max'],population['region_distance_crop_px']['max'])
    exact(groups,Counter({(x['domain'],x['source_group']):int(x['triplets'])for x in groupscsv}))
    exact(canonical,Counter({(x['domain'],int(x['native_slot']),x['canonical_index']):int(x['triplets'])for x in canonicalcsv}))
    exact({str(k):native[k]for k in range(8)},summary['selected_native_counts'])
    exact(sum(native.values()),109);exact(summary['declared_known_weak_labels'],109*3);exact(summary['all8_variant_rows'],109*24);exact(summary['other7_variant_UNKNOWN_rows'],109*21)
    # Selected mapping is actual saved TRAIN-only metadata, not target NPZ.
    selected={(x['domain'],x['attempt_id'])for x in selection['selected']};mg=defaultdict(list)
    for r in mapping:exact(r['role'],'TRAIN');mg[(r['domain'],r['attempt_id'])].append(r)
    exact(set(mg),selected);exact(len(mapping),288);exact(len(mg),12)
    for key,rr in mg.items():
        source=bykey[key];n=source['selected_native_slot'];exact({(r['variant'],int(r['native_slot']))for r in rr},{(v,k)for v in ['BEFORE','COVER','SHAM']for k in range(8)})
        exact({int(r['cache_index'])for r in rr},set(source['cache_indices24']))
        for k in range(8):
            r8=[r for r in rr if int(r['native_slot'])==k]
            exact(len({(r['original_query_xy'],r['original_L'],r['original_query_available'],r['original_global_index'])for r in r8}),1)
            for r in r8:
                exact(int(r['direct_weak_target']),({'BEFORE':0,'COVER':1,'SHAM':0}[r['variant']]if k==n else -1))
                if r['original_query_available']=='False':exact(int(r['direct_weak_target']),-1)
        first=next(r for r in rr if int(r['native_slot'])==n and r['variant']=='BEFORE')
        exact(json.loads(first['original_query_xy']),source['association']['original_selected_query_xy']);close(float(first['original_L']),source['association']['original_selected_L'])
    exact(sum(int(r['direct_weak_target'])>=0 for r in mapping),36);exact(sum(r['original_query_available']=='False'for r in mapping),24)
    for k in [6,7]:
        caveat=summary['native6_7_caveat'][str(k)]
        exact(caveat['selected_native_known_in109'],native[k]);exact(native[k],0)
        exact(caveat['selected12_all8_rows'],sum(int(r['native_slot'])==k for r in mapping))
        exact(caveat['selected12_all8_direct_targets_UNKNOWN'],sum(int(r['native_slot'])==k and int(r['direct_weak_target'])==-1 for r in mapping))
    canonical6=sum(n for(d,k,c),n in canonical.items()if c=='6');canonical7=sum(n for(d,k,c),n in canonical.items()if c=='7')
    exact(canonical6,9);exact(canonical7,4)
    # B's four prior PNG views and zero raw NPZ reads are distinct facts.
    exact(bview['actual_images_viewed'],4);exact(bview['actual_contacts'],2);exact(bview['actual_BEFORE_full_contexts'],2);exact(bview['all109_or288_images_viewed'],False)
    exact(fig['input'],summary['outputs']['per109']);exact(fig['rows_recomputed'],109)
    for b in fig['outputs']:read_bound(b)if Path(b['path']).suffix=='.json'else None;exact(hashlib.sha256(raw(b['path'])).hexdigest(),b['sha256'])
    exact(rootview['image']['sha256'],fig['outputs'][0]['sha256']);exact(rootview['actual_tool_views'],1)
    # A bounded code/readout arithmetic review. No65536-row source rescan.
    ac=json.loads(raw(A/'CENSUS_RESULT.json'));raw(A/'census_q8_provenance.py')
    ap=read_bound(ac['exports']['populations']);an=read_bound(ac['exports']['native']);aq=read_bound(ac['exports']['selected12'])
    exact(len(ap),5);exact(len(an),40)
    for x in ac['populations']:
        row=next(z for z in ap if z['population']==x['population']);exact({k:(z if k=='population'else int(z))for k,z in row.items()},x)
        nn=[z for z in ac['native']if z['population']==x['population']];exact({z['native_slot']for z in nn},set(range(8)))
        for nk,pk in [('TRAIN_rows','TRAIN_rows'),('inputU','TRAIN_inputU'),('finite_q','TRAIN_finite_q'),('null_L','TRAIN_null_L')]:exact(sum(z[nk]for z in nn),x[pk])
        exact(x['TRAIN_rows']+x['HOLDOUT_role_metadata_rows'],x['physical_rows'])
        exact(x['TRAIN_finite_q']+x['TRAIN_null_q'],x['TRAIN_rows']);exact(x['TRAIN_null_L'],0)
    exact([{k:(v if k=='population'else int(v))for k,v in x.items()}for x in an],ac['native'])
    exact(sum(x['physical_rows']for x in ac['populations']),73976);exact(sum(x['TRAIN_rows']for x in ac['populations']),64160)
    exact(sum(x['TRAIN_inputU']for x in ac['populations']),1145)
    exact(sum(ac['checks']['SELF_state_counts'].values()),61440);exact(len(ac['checks']['SELF_state_counts']),15)
    exact(ac['checks']['TRAIN_SELF_active_q_or_L_differs_from_original_anchor_rows'],(15-1)*440*8)
    exact(ac['checks']['TRAIN_SELF_finite_anchor_transform_recoverable_rows'],15*440*8)
    exact(sum(ac['checks']['direct_group_origins'].values()),393)
    exact(aq['labels_targets_poses_end_enclosures_or_predictions_included'],False);exact(aq['new_selection'],False)
    exact([(x['domain'],x['attempt_id'])for x in aq['rows']],[(x['domain'],x['attempt_id'])for x in selection['selected']])
    mindex={(x['domain'],x['attempt_id'],x['variant'],int(x['native_slot'])):x for x in mapping};qjoined=0
    for x in aq['rows']:
        exact(len(x['variants']),3);exact(x['selected_native_slot'],bykey[(x['domain'],x['attempt_id'])]['selected_native_slot'])
        for g in x['variants']:
            exact(g['native_slot'],list(range(8)))
            if g['source_variant']=='ADJACENT_SHAM':exact(g['variant'],'SHAM')
            for k in range(8):
                r=mindex[(x['domain'],x['attempt_id'],g['variant'],k)]
                exact(g['query_xy8'][k],json.loads(r['original_query_xy']));close(g['L8'][k],float(r['original_L']));exact(g['original_query_available8'][k],r['original_query_available']=='True');exact(g['original_global_indices8'][k],int(r['original_global_index']));exact(g['original_row_ids8'][k],r['row_id']);qjoined+=1
    exact(qjoined,288)
    result=dict(completed=True,PASS=True,code=dict(path=str(Path(__file__).absolute()),sha256=hashlib.sha256(raw(Path(__file__).absolute())).hexdigest()),scope=summary['scope'],checks=checks,max_abs_floating_delta=maxdelta,tolerance=1e-10,
                B_independent_original_TRAIN_metadata_recompute=dict(rows109=109,selected_mapping288=288,domains=recomputed,native_triplets={str(k):native[k]for k in range(8)},canonical6=canonical6,canonical7=canonical7,source_group_count53=len(groups),target_counts_declared_only=dict(known327=327,other7_UNKNOWN2289=2289),projection='Own inverse-grid sample-origin/pitch scalar arithmetic; raw interval clamp distance, no B helper import/run'),
                A_bounded_arithmetic_and_crossartifact_review=dict(populations5=True,native_rows40=True,physical_rows_sum73976=73976,TRAIN_rows_sum64160=64160,TRAIN_inputU_sum1145=1145,selected12_qLU_ID_native_join_rows288=qjoined,source65536_metadata_rescan=False,TRAINprefix4096_initializer_claim='A code+result authority checked; C did not independently reread prefix or source rows'),
                actual_PNG_toolview_C=dict(count=1,path=str(V/'scientific_readout_v1/TRAIN_QUERY_AND_WEAK_END.png'),original_size='2250x780',tool_display='2048x710',observations=['Three panels and82/27 triplet and41/12 group counts legible.','Original query marker47.5, weak boxes and region-distance ECDF/median labels/footer legible.','Small synthetic boxes vs larger public weak boxes are different definitions; no comparable physical area/GT-coordinate-error claim.'],HTML_browser_rendered=False),
                B_separate_actual_PNG_toolview=dict(receipt=next(x for x in ACCESSES if x['path']==str(B/'ACTUAL_FIXED_TRAIN_VISUAL_OBSERVATIONS.json')),images4=True,contacts2=True,BEFOREcontexts2=True,all109_or288_claim=False),
                access_accounting=dict(actual_source_NPZ_headers_arrays_weights_models_predictions_scores_HOLDOUT_targets_pixels=0,referenced_binary_bindings_followed=0,producer_reruns=0,C_source_RGB_PNG_views=0,C_generated_metadata_plot_view=1),
                limitations=['B109 figures are donor/attempt-weighted metadata enclosures, not109 independent frames or physical point gold.','C independently recalculated B109 from original allowed TRAIN109 and288 mapping/selection, not raw payload bytes.','A population arithmetic and selected12 q8 cross-artifact join checked; not an independent full65536 census or4096 initializer-prefix byte audit.','Input finite coordinates do not imply available/visible queries. Native6/7 known0 differs from canonical6/7 present9/4 at other slots; no unsupported SELF/D/E/OOF cause inference.','B4 source PNG views are separate from phase rawNPZ0. C viewed one generated metadata plot only; no original RGB/crop image.','No new label, admission, model/input/threshold/architecture selection or causal/performance conclusion.'],accesses=ACCESSES,final_A_narrative_receipt_pending=True)
    with(H/'COMPUTED_CHECKS.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,checks=checks,max_abs_delta=maxdelta,output=str(H/'COMPUTED_CHECKS.json'),sha256=hashlib.sha256((H/'COMPUTED_CHECKS.json').read_bytes()).hexdigest())))


if __name__=='__main__':run()
