"""Independent scalar saved-JSON/CSV census; no producer imports or arrays.

No NPZ/NPY/image/weight/model/evaluation loader is permitted by this script.
The optional scientific figure uses independently recounted table metadata only.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,math,os,stat

H=Path(__file__).absolute().parent;V=H.parent
ROOT=V/'root_descriptive_source_tables'

def body(path):
    p=Path(path)
    assert p.is_absolute() and '..' not in p.parts and all('v112' not in s.casefold() for s in p.parts)
    assert p.suffix in {'.py','.json','.jsonl','.csv'},'Only explicit textmetadata'
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nf=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nf
        final=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(final,'rb') as f:
        assert stat.S_ISREG(os.fstat(f.fileno()).st_mode)
        return f.read()

def bind(path):return dict(path=str(path),sha256=hashlib.sha256(body(path)).hexdigest())
def read(path):return json.loads(body(path))
def lines(path):return [json.loads(s) for s in body(path).splitlines() if s]
def checked(b):
    assert bind(Path(b['path']))==b,'Bound textmetadata SHA drift'
    return Path(b['path'])
def write(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rows_csv(path):return list(csv.DictReader(body(path).decode().splitlines()))

def one_axis(q,L,n):
    """Independent separability oracle: actual 96 linear support, not its bbox."""
    pixels=set();coordinates=[]
    for i in range(96):
        x=q+((i+.5)/96-.5)*L;coordinates.append(x)
        if not (-.5<=x<n-.5):continue
        left=math.floor(x);fraction=x-left
        if 1-fraction>0 and 0<=left<n:pixels.add(left)
        if fraction>0 and 0<=left+1<n:pixels.add(left+1)
    return pixels,(min(coordinates),max(coordinates))

def scalar_taps(q,L,hw):
    found=set();h,w=hw
    for j in range(96):
        for i in range(96):
            x=q[0]+((i+.5)/96-.5)*L;y=q[1]+((j+.5)/96-.5)*L
            if not (-.5<=x<w-.5 and -.5<=y<h-.5):continue
            x0,y0=math.floor(x),math.floor(y);fx,fy=x-x0,y-y0
            for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
                weight=(fx if dx else 1-fx)*(fy if dy else 1-fy)
                p=(x0+dx,y0+dy)
                if weight>0 and 0<=p[0]<w and 0<=p[1]<h:found.add(p)
    return found

def rectangle_union(boxes):
    return {(x,y) for b in boxes for x in range(math.floor(b[0]),math.ceil(b[2])+1)
            for y in range(math.floor(b[1]),math.ceil(b[3])+1)}

def audit():
    checks={}
    def record(name,value):
        assert value,name;checks[name]=True
    sc=read(V/'synthetic_direct_end_controls_v1/COMPLETE.json');rc=read(V/'real_end_local_pairs_v1/COMPLETE.json')
    syn=lines(checked(sc['attempts']));variants=lines(checked(sc['variants']));real=lines(checked(rc['manifest']));observations=lines(checked(rc['observations']))
    original=lines(checked(sc['population']));original_by_id={r['attempt_id']:r for r in original};am={r['attempt_id']:r for r in syn}
    record('all1024_3072_66_22_unique_id_census',len(syn)==len(am)==len(original_by_id)==1024 and len(variants)==3072 and len(real)==len({r['pair_id']for r in real})==66 and len(observations)==len({r['case_id']for r in observations})==22)
    record('synthetic_dense_attempt_index_and_all128base_native8', [r['attempt_index']for r in syn]==list(range(1024)) and {(r['base_index'],r['native_slot'])for r in syn}=={(b,k)for b in range(128)for k in range(8)})
    keep=('base_index','native_slot','canonical_index','original_global_index','original_occurrence_id','query_xy','L','original_input_flags','raw_hw','factors','geometry_key')
    record('synthetic_original_metadata_qL_flags_role_exact',all(all(r[k]==original_by_id[r['attempt_id']][k]for k in keep)for r in syn))
    record('all_synthetic_original_native_quality_training_unchanged',all(r['direct_end_target']==r['new_native_visibility_target']==r['new_point_quality_target']==-1 and r['original_targets_not_copied_or_rewritten'] and not r['training_admission'] for r in syn))
    variant_map=defaultdict(list)
    for r in variants:variant_map[r['attempt_id']].append(r)
    for r in syn:
        vv=variant_map[r['attempt_id']]
        assert [x['variant']for x in vv]==['BEFORE','COVER','SHAM']
        assert [x['direct_end_external_target']for x in vv]==r['direct_end_states']
        assert r['direct_end_states'] in ([-1,-1,-1],[0,1,0])
        for x in vv:
            assert all(x[k]==r[k]for k in ('base_index','native_slot','original_global_index','query_xy','L','original_input_flags','status','reasons'))
            assert x['new_native_visibility_target']==x['new_quality_target']==x['new_owner_target']==-1 and not x['training_admission']
            if not r['original_input_flags']['query_available']:assert x['direct_end_external_target']==-1
    record('all3072_variant_states_joins_U_and_other_targets_unknown',True)
    known=[r for r in syn if r['direct_end_states']==[0,1,0]];unknown=[r for r in syn if r['direct_end_states']==[-1,-1,-1]]
    record('synthetic_known_closed_proofs_and_no_hidden_admission',all(r['status']=='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED' and r['outputs'] and r['closed_proof']['new_native_visibility_target']==r['closed_proof']['new_quality_target']==-1 and r['closed_proof']['training_admission'] is False for r in known))
    group_real=defaultdict(list)
    for r in real:group_real[r['case_id']].append(r)
    for o in observations:
        rr=group_real[o['case_id']];assert len(rr)==3 and {r['pair_id']for r in rr}==set(o['fixed_original_attempt_ids'])
        for r in rr:
            i=r['original_input']
            assert (i['original_global_index'],i['original_row_id'],i['native_slot'],i['query_xy'],i['L'],i['original_query_available'])==(o['original_global_index'],o['query_id'],o['native_slot'],o['original_query_xy'],o['original_L'],o['original_query_available'])
            assert i['development_role']=='SOURCE_DEVELOPMENT' and r['endpoint_observation']==o['endpoint_observation']
            assert r['training_admission'] is False and r['new_native_quality_owner_all8']==-1
            expected=dict(BEFORE='VISIBLE_WEAK',COVER='EXTERNALLY_BLOCKED_WEAK',SHAM='VISIBLE_WEAK') if r['PASS'] else dict(BEFORE='UNKNOWN',COVER='UNKNOWN',SHAM='UNKNOWN')
            assert r['direct_end_weak_states']==expected
            if r['PASS']:assert r['constructed'] and all(r['pixel_checks'].values()) and all(r['before_parity'].values())
    record('real66_original_identity_qL_U_end_observation_and_target_declarations',True)
    cells=[]
    for cell in sorted({tuple(r['factors'][k]for k in ('material','aspect','geometry_family'))for r in syn}):
        subset=[r for r in syn if tuple(r['factors'][k]for k in ('material','aspect','geometry_family'))==cell]
        cells.append(dict(zip(('material','aspect','geometry_family'),cell),attempts=len(subset),supported=sum(r['direct_end_states']==[0,1,0]for r in subset),UNKNOWN=sum(r['direct_end_states']!=[0,1,0]for r in subset),input_U=sum(not r['original_input_flags']['query_available']for r in subset)))
    record('all8_independent_material_aspect_geometry_cells_128_each',len(cells)==8 and all(r['attempts']==128 for r in cells))
    native=[]
    for k in range(8):
        ss=[r for r in syn if r['native_slot']==k];rr=[r for r in real if r['original_input']['native_slot']==k]
        native.append(dict(native_slot=k,synthetic_attempts=len(ss),synthetic_supported=sum(r['direct_end_states']==[0,1,0]for r in ss),synthetic_UNKNOWN=sum(r['direct_end_states']!=[0,1,0]for r in ss),real_attempts=len(rr),real_supported=sum(r['PASS']for r in rr),real_UNKNOWN_or_failure=sum(not r['PASS']for r in rr)))
    record('native0_to7_full_denominators_including_zero_support',len(native)==8 and all(r['synthetic_attempts']==128 for r in native))
    locality=[];separable_count=0;nonlocal_detail=None
    for o in observations:
        rr=group_real[o['case_id']];i=rr[0]['original_input'];boxes=o['endpoint_observation']['candidate_enclosures_xyxy']
        rec=dict(case_id=o['case_id'],native_slot=o['native_slot'],L=o['original_L'],observation=o['endpoint_observation_status'],candidate_pixels=0,raw_support_missing_pixels=0,missing_outside_support_bounding_box=0,missing_inside_support_bounding_box_holes=0,local_class='DIRECT_END_OBSERVATION_UNKNOWN',attempts=len(rr),supported=sum(r['PASS']for r in rr),status_counts=json.dumps(dict(Counter(r['status']for r in rr)),ensure_ascii=False),reasons=json.dumps([r.get('reason',[])for r in rr],ensure_ascii=False))
        if boxes:
            xs,xi=one_axis(i['query_xy'][0],i['L'],i['raw_hw'][1]);ys,yi=one_axis(i['query_xy'][1],i['L'],i['raw_hw'][0]);support={(x,y)for x in xs for y in ys}
            assert support==scalar_taps(i['query_xy'],i['L'],i['raw_hw']);separable_count+=1
            candidates=rectangle_union(boxes);missing=candidates-support
            internal={p for p in missing if min(xs)<=p[0]<=max(xs) and min(ys)<=p[1]<=max(ys)}
            category='ALL_CANDIDATE_RAW_PIXELS_IN_SUPPORT' if not missing else 'INTERNAL_DOWNSAMPLING_SUPPORT_HOLES' if missing==internal else 'CANDIDATE_CROSSES_SUPPORT_EXTENT'
            rec.update(candidate_pixels=len(candidates),raw_support_missing_pixels=len(missing),missing_outside_support_bounding_box=len(missing-internal),missing_inside_support_bounding_box_holes=len(internal),local_class=category)
        elif o['no_local_opportunity_due_to_outside_original_crop']:
            rec['local_class']='OBSERVED_INTENDED_END_OUTSIDE_ORIGINAL_CROP_UNKNOWN_NOT_E'
            boxes=o['nonlocal_visual_candidate_enclosures_raw_xyxy'];xs,xi=one_axis(i['query_xy'][0],i['L'],i['raw_hw'][1]);ys,yi=one_axis(i['query_xy'][1],i['L'],i['raw_hw'][0]);candidate=rectangle_union(boxes)
            overlap=sum(x in xs and y in ys for x,y in candidate)
            assert o['case_id']=='Q0057' and overlap==0 and all(b[3]<min(ys) for b in boxes)
            nonlocal_detail=dict(case_id=o['case_id'],authority='Assistant weak observed nonlocal RGB candidate; interval arithmetic only',nonlocal_enclosures=boxes,original_q=i['query_xy'],original_L=i['L'],raw_positive_support_bounds=[min(xs),min(ys),max(xs),max(ys)],raw96_sample_coordinate_intervals=[xi,yi],candidate_pixels=len(candidate),candidate_positive_support_overlap=overlap,all_candidates_above_minimum_sampled_raw_y=True,not_physical_coordinate_error=True)
        locality.append(rec)
    record('16local_candidate_cases_separable_and_scalar_four_taps_exact',separable_count==16)
    record('11_internal_rawtap_holes_not_outside_crop_extent',sum(r['local_class']=='INTERNAL_DOWNSAMPLING_SUPPORT_HOLES'for r in locality)==11 and all(r['L']>192 and r['missing_outside_support_bounding_box']==0 for r in locality if r['local_class']=='INTERNAL_DOWNSAMPLING_SUPPORT_HOLES'))
    record('Q0057_single_explicit_nonlocal_candidate_interval_no_overlap',nonlocal_detail is not None and sum(r['local_class']=='OBSERVED_INTENDED_END_OUTSIDE_ORIGINAL_CROP_UNKNOWN_NOT_E'for r in locality)==1)
    payloads=[v for r in real for v in r['variants'].values() if isinstance(v,dict) and 'path'in v]
    synthetic=dict(attempts=len(syn),variants=len(variants),supported_triplets=len(known),unknown_triplets=len(unknown),status=dict(Counter(r['status']for r in syn)),reason_occurrences=dict(Counter(x for r in syn for x in r['reasons'])),known_selected_variant_labels=dict(Counter(str(r['direct_end_external_target'])for r in variants)),all8_attempted_packet_slots=len(syn)*3*8,original_before_U=sum(not r['original_input_flags']['query_available']for r in syn),all_original1024_before_crop_parity=sc['counts']['before_RGB96_valid96_exactparity'])
    real_summary=dict(cases=len(group_real),attempts=len(real),variant_statuses=len(real)*3,constructed=sum(r['constructed']for r in real),supported=sum(r['PASS']for r in real),failed_or_UNKNOWN=sum(not r['PASS']for r in real),status=dict(Counter(r['status']for r in real)),actual_payloads=len(payloads),missing_payloads=len(real)*3-len(payloads),actual_payload_native_slots=len(payloads)*8,all_attempted_variant_native_slots=len(real)*3*8,reason_occurrences=dict(Counter(str(r.get('reason',''))for r in real if not r['PASS'])),case_locality_classes=dict(Counter(r['local_class']for r in locality)))
    # Only now compare independent calculations to saved tables/summary.
    expected=read(ROOT/'SUMMARY.json');done=read(ROOT/'COMPLETE.json')
    record('root_summary_counts_status_reasons_full_exact',expected['synthetic']==synthetic and expected['real']==real_summary)
    record('root_summary_native8_factor8_locality22_exact',expected['native8']==native and expected['factor_cells']==cells and expected['all22_locality']==locality)
    for name,ours in [('SYNTHETIC_FACTOR_CELLS.csv',cells),('FULL_NATIVE8_DENOMINATORS.csv',native),('ALL22_REAL_LOCALITY_DIAGNOSTIC.csv',locality)]:
        saved=rows_csv(ROOT/name);record('CSV_exact_'+name,saved==[{k:str(v)for k,v in row.items()}for row in ours])
    for b in [done['summary']]+done['tables']:checked(b)
    record('root_completion_bindings_exact',done['completed'] and done['PASS'])
    result=dict(completed=True,PASS=True,review_type='INDEPENDENT_SAVED_TEXT_CENSUS_NOT_PIXEL_OR_MODEL_AUDIT',code=bind(Path(__file__).absolute()),inputs={name:bind(path)for name,path in [('synthetic_complete',V/'synthetic_direct_end_controls_v1/COMPLETE.json'),('synthetic_attempts',Path(sc['attempts']['path'])),('synthetic_variants',Path(sc['variants']['path'])),('real_complete',V/'real_end_local_pairs_v1/COMPLETE.json'),('real_attempts',Path(rc['manifest']['path'])),('all22observations',Path(rc['observations']['path'])),('root_tables_complete',ROOT/'COMPLETE.json'),('root_summary',ROOT/'SUMMARY.json')]},checks=checks,number_checks=len(checks),synthetic=synthetic,real=real_summary,factor_cells=cells,native8=native,all22_locality=locality,explicit_nonlocal_Q0057=nonlocal_detail,
        limitations=['Metadata arithmetic and recorded proof declarations only; source/NPZ pixels not independently verified here.','11 all-enclosure raw-support holes are stricter than recognizable end presence and are not physical end outside crop.','Q0057 is a separate assistantweak nonlocal observed endpoint with zero interval overlap; no new coordinate gold.','97 synthetic and5real controls are protocol-supported data, not model accuracy; native6/7 synthetic known0 and realknown onlynative0/twoframes do not qualify full goal.','No sourceUNKNOWN completion, target relabel, fit admission, model/threshold/data selection or performance claim.'],model_forward_fit_or_eval=0,NPZ_NPY_image_weight_or_feature_body_reads=0,selected_model=None,fully_solved=False,final_root_report_QA='PENDING_ROOT_REPORT_READY')
    write(H/'REVIEW.json',result)
    with (H/'INDEPENDENT_ALL22_LOCALITY.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(locality[0]));w.writeheader();w.writerows(locality)
    return result


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig=plt.figure(figsize=(15,8.4),dpi=150);gs=fig.add_gridspec(2,2,height_ratios=[1,1.12])
    ax=fig.add_subplot(gs[0,0]);r=result['native8'];x=np.arange(8)
    known=[a['synthetic_supported']for a in r];u=[a['synthetic_UNKNOWN']for a in r]
    ax.bar(x,known,color='#286aa6',label='Protocol supported');ax.bar(x,u,bottom=known,color='#d5d8dc',label='UNKNOWN/failure retained')
    for i,n in enumerate(known):ax.text(i,n+3,str(n),ha='center',fontsize=10)
    ax.set(xticks=x,xlabel='Requested native slot',ylabel='Attempts',ylim=(0,143),title='Synthetic TRAIN: 97 / 1024 supported triplets (128 attempts per slot)');ax.legend(loc='upper right',fontsize=9)
    ax=fig.add_subplot(gs[0,1]);known=[a['real_supported']for a in r];u=[a['real_UNKNOWN_or_failure']for a in r]
    ax.bar(x,known,color='#c15b1b',label='Weak controlled pairs');ax.bar(x,u,bottom=known,color='#d5d8dc',label='UNKNOWN/failure retained')
    for i,a in enumerate(r):ax.text(i,a['real_attempts']+.35,str(known[i])+'/'+str(a['real_attempts']),ha='center',fontsize=10)
    ax.set(xticks=x,xlabel='Requested native slot',ylabel='Attempts',ylim=(0,18),title='Public real TRAIN: 5 / 66 supported (native0, two frames)');ax.legend(loc='upper right',fontsize=9)
    ax=fig.add_subplot(gs[1,:]);cells=result['factor_cells'];x=np.arange(8);known=[a['supported']for a in cells];u=[a['UNKNOWN']for a in cells]
    labels=[('Polymer'if a['material']=='DARK_POLYMER'else'Wood')+' / '+('Rect'if a['aspect']=='RECTANGULAR'else'Square')+'\n'+('Board-block'if a['geometry_family']=='BOARD_BLOCK_RUNNER'else'Rib-leg')for a in cells]
    ax.bar(x,known,color='#286aa6');ax.bar(x,u,bottom=known,color='#d5d8dc')
    for i,n in enumerate(known):ax.text(i,n+3,str(n)+'/128',ha='center',fontsize=10)
    ax.set(xticks=x,xticklabels=labels,ylabel='Attempts',ylim=(0,145),title='All eight material × aspect × geometry cells retained; geometry style is separate from material/aspect')
    for a in fig.axes:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.16);a.set_axisbelow(True)
    fig.suptitle('V188 source-control coverage — data proof counts, not model performance',fontsize=17,y=.985)
    fig.text(.5,.015,'Fixed q/L and native roles. UNKNOWN and input-U retained. 11 real cases fail conservative raw-tap coverage inside the crop extent; Q0057 is separately nonlocal.\nNo training admission, new fit/forward/evaluation, model adoption or full-goal qualification.',ha='center',va='bottom',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.955));path=H/'SOURCE_CONTROL_COVERAGE.png';fig.savefig(path);plt.close(fig)
    # PNG hash is our own newly created metadata-only scientific figure.
    write(H/'PLOT_COMPLETE.json',dict(completed=True,PASS=True,plot=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),source_review=bind(H/'REVIEW.json'),generated_from_table_metadata_only=True,actual_tool_visual_QA='PENDING_VIEW_IMAGE',no_source_pixels_read=True))


if __name__=='__main__':
    result=audit();plot(result);print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),checks=result['number_checks'],synthetic=[result['synthetic']['supported_triplets'],result['synthetic']['unknown_triplets']],real=[result['real']['supported'],result['real']['failed_or_UNKNOWN']])) )
