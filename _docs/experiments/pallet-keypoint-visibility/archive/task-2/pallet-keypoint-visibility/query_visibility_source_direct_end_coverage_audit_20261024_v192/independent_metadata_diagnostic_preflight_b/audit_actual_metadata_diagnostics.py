"""Independent scalar V192 metadata QA. Never imports the producer or arrays."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import io
import json
import math
import os
import stat
import traceback

H=Path(__file__).absolute().parent
V=H.parent
OUT=V/'independent_metadata_diagnostic_review_b'
SCOPE=V/'ROOT_INDEPENDENT_SOURCE_METADATA_DIAGNOSTIC_QA_SCOPE.json'
SCOPE_SHA='fa6493c02bd7a11cfb5cdad27bd86259527795232157982a2486e1f9792236e6'
PROTOCOL=H/'ACTUAL_QA_PROTOCOL_BEFORE_SOURCE_JSONL_RECALCULATION.json'
CHECKS=H/'ACTUAL_QA_INVENTED_CHECKS.json'
BINS=('0','1','2~4','5~8','9~16','17~32','33~64','65이상','missing')
QUANT=('end_related96_samples','COVER_changed_RGB96_samples','SHAM_changed_RGB96_samples')
BOOL=('end_support_actual_original96_bytes_changed_COVER','end_support_actual_original96_bytes_unchanged_SHAM')


def need(x,m):
    if not x:raise AssertionError(m)


def safe(p):
    p=Path(p)
    need(p.is_absolute() and '..' not in p.parts and not any('v112' in s.casefold() for s in p.parts),'Lexical allowed path')
    return p


def opened(p):
    p=safe(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nextfd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nextfd
        last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(last).st_mode),'Regular file only')
    return os.fdopen(last,'rb')


def data(p):
    with opened(p) as f:return f.read()


def bind(p):return dict(path=str(p),sha256=hashlib.sha256(data(p)).hexdigest())
def read(p):return json.loads(data(p))
def checked(b):
    need(bind(b['path'])==b,'Exact bound text '+str(b['path']));return b['path']
def rows(b):return [json.loads(s) for s in data(checked(b)).splitlines() if s.strip()]
def write(p,d):
    safe(p)
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def source_samples(q,L,hw):
    """Separate separable linear-basis oracle; no producer support helper."""
    h,w=hw;output=[];support=set()
    xs=[q[0]+((i+.5)/96-.5)*L for i in range(96)]
    ys=[q[1]+((i+.5)/96-.5)*L for i in range(96)]
    for y in ys:
        for x in xs:
            points=set()
            if -.5<=x<w-.5 and -.5<=y<h-.5:
                for py in (math.floor(y),math.floor(y)+1):
                    wy=1-abs(y-py)
                    for px in (math.floor(x),math.floor(x)+1):
                        wx=1-abs(x-px)
                        if wx>0 and wy>0 and 0<=px<w and 0<=py<h:points.add((px,py))
            support.update(points);output.append(points)
    return support,output


def candidate(boxes,hw):
    h,w=hw;out=set();outside=False
    for x0,y0,x1,y1 in boxes:
        need(x0<=x1 and y0<=y1,'Unchanged ordered closed enclosure')
        outside|=not(0<=x0<=x1<w and 0<=y0<=y1<h)
        for y in range(math.floor(y0),math.ceil(y1)+1):
            for x in range(math.floor(x0),math.ceil(x1)+1):
                if 0<=x<w and 0<=y<h:out.add((x,y))
                else:outside=True
    return out,outside


def rectangle(b):return {(x,y)for y in range(b[1],b[3])for x in range(b[0],b[2])}
def in_frame(b,hw):return 0<=b[0]<b[2]<=hw[1] and 0<=b[1]<b[3]<=hw[0]


def geometry(r,cache):
    o=r['original_input'];e=r['endpoint_observation'];p=r.get('placement_geometry')
    localized=(e.get('authority')=='SOURCE_RGB_SEMANTIC_END_WEAK' and e.get('native_slot')==o['native_slot'] and
        bool(e.get('evidence_binding')) and bool(e.get('end_locality_id')) and all(e.get(k) is True for k in
        ('selected_instance_supported','intended_part_supported','queried_end_localization_supported','all_plausible_candidates_enclosed','before_end_visible_supported')))
    z=dict(weak_semantic_localization_supported=localized,placement_metadata_available=p is not None,
        original_input_available=o['original_query_available'],old_geometry_checks=None,opposite_geometry_checks=None,bbox_only_possible=None,
        proposal='EXACTLY_ONE: existing integer dx sign flipped, original dy asserted0',actual_alpha_RGB_identity_front_depth_verified=False,new_direct_label=None,training_admission=False)
    if not localized:
        z['diagnostic_reason']='FROZEN_WEAK_OBSERVATION_UNKNOWN_UNCHANGED';return z
    q,L,hw=o['query_xy'],o['L'],o['raw_hw'];cache_key=json.dumps([q,L,hw])
    if cache_key not in cache:cache[cache_key]=source_samples(q,L,hw)
    support,samples=cache[cache_key];pixels,outside=candidate(e['candidate_enclosures_xyxy'],hw)
    extent=[q[0]+(.5/96-.5)*L,q[1]+(.5/96-.5)*L,q[0]+(95.5/96-.5)*L,q[1]+(95.5/96-.5)*L]
    cr=[]
    for box in e['candidate_enclosures_xyxy']:
        pp,oo=candidate([box],hw)
        cr.append(dict(enclosure_xyxy=box,in_original_sample_extent=extent[0]<=box[0]<=box[2]<=extent[2] and extent[1]<=box[1]<=box[3]<=extent[3],
            raw_footprint_in_frame=not oo,positive_original96_samples=sum(bool(pp&s) for s in samples)))
    witnesses=[tuple(w['raw_pixel_xy'])for w in e.get('surrounding_material_witnesses',[])if w.get('material_observed') is True and w.get('same_selected_part_supported') is True and w.get('evidence_binding')]
    z.update(original96_sample_extent=extent,candidates=cr,chosen_raw_witnesses=[dict(raw_pixel_xy=list(w),positive_original96_tap=w in support,outside_candidate=w not in pixels) for w in witnesses],all_chosen_witnesses_positive96=bool(witnesses) and all(w in support for w in witnesses))
    if p is None:
        z['diagnostic_reason']='NO_STORED_PLACEMENT_NO_ALTERNATIVE_FABRICATED';return z
    cover=p['cover_bbox_xyxy'];dx,dy=p['sham_translation_xy'];need(type(dx) is int and dx!=0 and dy==0,'Only original dx sign flip')
    old=[cover[0]+dx,cover[1],cover[2]+dx,cover[3]];op=[cover[0]-dx,cover[1],cover[2]-dx,cover[3]]
    cover_pixels=rectangle(cover)
    def flags(sham):
        sp=rectangle(sham)
        return dict(cover_bbox_contains_all_end_support=bool(pixels) and pixels<=cover_pixels,
            sham_bbox_disjoint_all_end_support=not bool(pixels&sp),cover_and_sham_in_frame=in_frame(cover,hw) and in_frame(sham,hw),
            cover_bbox_in_frame=in_frame(cover,hw),sham_bbox_in_frame=in_frame(sham,hw),same_bbox_and_positive_mass_translation=True,
            sham_has_some_original_rgb96_support=bool(sp&support),
            certified_surrounding_part_retained_by_both_bbox_transforms=bool(witnesses) and all(w in support and w not in pixels and w not in cover_pixels and w not in sp for w in witnesses),
            all_chosen_witnesses_outside_cover=bool(witnesses) and not any(w in cover_pixels for w in witnesses),
            all_chosen_witnesses_outside_sham=bool(witnesses) and not any(w in sp for w in witnesses),
            cover_bbox_intersects_end_candidate=bool(cover_pixels&pixels),sham_bbox_intersects_end_candidate=bool(sp&pixels))
    f0,f1=flags(old),flags(op)
    saved=r.get('finite_end_geometry',{}).get('geometry_controls')
    if saved is not None:need(all(f0[k]==v for k,v in saved.items()),'Original saved geometry exact independently')
    prerequisite=bool(o['original_query_available']) and bool(o.get('feature_constructed')) and not outside and bool(cr) and all(c['in_original_sample_extent'] and c['raw_footprint_in_frame'] and c['positive_original96_samples']>0 for c in cr)
    required=('cover_bbox_contains_all_end_support','sham_bbox_disjoint_all_end_support','cover_and_sham_in_frame','sham_has_some_original_rgb96_support','certified_surrounding_part_retained_by_both_bbox_transforms')
    z.update(cover_bbox_xyxy=cover,existing_sham_bbox_xyxy=old,opposite_sham_bbox_xyxy=op,existing_translation_xy=[dx,dy],opposite_translation_xy=[-dx,0],
        old_geometry_checks=f0,opposite_geometry_checks=f1,same_frozen_end_and_input_opportunity=prerequisite,bbox_only_possible=prerequisite and all(f1[k] for k in required),diagnostic_reason='BBOX_ONLY_NOT_ALPHA_RGB_OR_NEW_TRIPLET_PROOF')
    return z


def get_quant(domain,r):
    if domain=='SYNTHETIC':
        e=r.get('closed_proof',{}).get('actual_proof_metadata',{}).get('evidence_bindings',{})
        q=dict(zip(QUANT,[e.get(k) for k in ('end_related96_samples','end_related96_COVER_changed_RGB_samples','end_related96_SHAM_changed_RGB_samples')]))
        return q,{}
    return dict(zip(QUANT,[r.get('pixel_detail',{}).get('end_related_RGB96_sample_count'),None,None])),{k:r.get('pixel_checks',{}).get(k)for k in BOOL}


def compact(domain,r,index,cache):
    q,b=get_quant(domain,r);known=r['status']==('CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED' if domain=='SYNTHETIC' else 'SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED') and (domain=='SYNTHETIC' or r['PASS'] is True)
    if domain=='SYNTHETIC':
        identity={k:r.get(k)for k in ('attempt_id','base_index','original_global_index','original_occurrence_id','query_xy','L','original_input_flags')}
        native=r['native_slot'];factors=r['factors'];available=r['original_input_flags']['query_available'];need(r['split']=='TRAIN','Source TRAIN only')
    else:
        identity={k:r.get(k)for k in ('pair_id','case_id','donor_id','original_input')};native=r['original_input']['native_slot'];factors={};available=r['original_input']['original_query_available'];need(r['original_input']['development_role']=='SOURCE_DEVELOPMENT','Public source TRAIN only')
    reasons=r.get('reasons',r.get('reason',[]));reasons=[reasons]if isinstance(reasons,str)else reasons
    z=dict(domain=domain,source_census_row_index=index,identity=identity,native_slot=native,factors=factors,original_query_available=available,stored_status=r['status'],stored_supported=known,stored_reasons=reasons,stored_quantities=q,stored_change_booleans=b,new_label_generated=False,training_admission=False)
    if domain=='REAL':z['geometry_bbox_only']=geometry(r,cache)
    return z


def bucket(n):
    if n is None:return 'missing'
    need(type(n) is int and 0<=n<=9216,'Integer stored count, never bool')
    for label,lo,hi in [('0',0,0),('1',1,1),('2~4',2,4),('5~8',5,8),('9~16',9,16),('17~32',17,32),('33~64',33,64),('65이상',65,9216)]:
        if lo<=n<=hi:return label
    raise AssertionError('Unassigned count')


def expected_tables(rr):
    populations=[];distributions=[];booleans=[];geom=[]
    for domain in ('SYNTHETIC','REAL'):
        dd=[r for r in rr if r['domain']==domain];groups=[('ALL','ALL',dd)]
        groups.extend(('native',str(k),[r for r in dd if r['native_slot']==k])for k in range(8))
        def fv(r,k):return str(r['factors'].get(k,'NOT_DECLARED_IN_REAL_MANIFEST'))
        for k in ('material','aspect','geometry_family','foreign_family'):
            for value in sorted({fv(r,k)for r in dd}):groups.append((k,value,[r for r in dd if fv(r,k)==value]))
        for cell in sorted({tuple(fv(r,k)for k in ('material','aspect','geometry_family'))for r in dd}):
            subset=[r for r in dd if tuple(fv(r,k)for k in ('material','aspect','geometry_family'))==cell]
            groups.append(('material_aspect_geometry','|'.join(cell),subset))
            for k in range(8):groups.append(('material_aspect_geometry_native','|'.join(cell)+'|native'+str(k),[r for r in subset if r['native_slot']==k]))
        for kind,k,subset in groups:
            populations.append(dict(domain=domain,stratum=kind,key=k,attempts=len(subset),original_inputU_selected_native=sum(not r['original_query_available']for r in subset),stored_supported=sum(r['stored_supported']for r in subset),stored_nonsupported=sum(not r['stored_supported']for r in subset),stored_status_counts=dict(Counter(r['stored_status']for r in subset)),overlapping_stored_reason_counts=dict(Counter(s for r in subset for s in set(r['stored_reasons'])))))
        for cohort in ('ALL_ATTEMPTS','STORED_SUPPORTED_ONLY'):
            subset=[r for r in dd if cohort=='ALL_ATTEMPTS' or r['stored_supported']]
            for field in QUANT:
                counts=Counter(bucket(r['stored_quantities'][field])for r in subset)
                for label in BINS:distributions.append(dict(domain=domain,cohort=cohort,quantity=field,fixed_bin=label,count=counts[label],cohort_denominator=len(subset)))
            if domain=='REAL':
                for field in BOOL:
                    values=[r['stored_change_booleans'][field]for r in subset];need(all(v is None or type(v) is bool for v in values),'Preserve booleans')
                    booleans.append(dict(cohort=cohort,stored_boolean=field,true_count=sum(v is True for v in values),false_count=sum(v is False for v in values),missing_count=sum(v is None for v in values),denominator=len(subset)))
    gs=[r['geometry_bbox_only']for r in rr if r['domain']=='REAL']
    def gcount(scope,k,values):
        return dict(scope=scope,check=k,true_count=sum(v is True for v in values),false_count=sum(v is False for v in values),missing_count=sum(v is None for v in values),denominator=66,counts_can_overlap=True)
    for k in ('weak_semantic_localization_supported','placement_metadata_available','all_chosen_witnesses_positive96','bbox_only_possible'):geom.append(gcount('ALL66',k,[r.get(k)for r in gs]))
    for side in ('old_geometry_checks','opposite_geometry_checks'):
        for k in sorted({k for r in gs for k in (r.get(side)or{})}):geom.append(gcount(side,k,[(r.get(side)or{}).get(k)for r in gs]))
    return populations,distributions,geom,booleans


def compare_table(actual,expected,keys,json_fields=()):
    need(len(actual)==len(expected),'Table full denominator/strata count')
    aa={tuple(r[k]for k in keys):r for r in actual};ee={tuple(r[k]for k in keys):r for r in expected}
    need(len(aa)==len(actual) and aa.keys()==ee.keys(),'Every unique exact table stratum')
    for k,er in ee.items():
        ar=aa[k]
        for field,value in er.items():
            got=ar[field]
            if field in json_fields and isinstance(got,str):got=json.loads(got)
            elif type(value) is int and isinstance(got,str):got=int(got)
            elif type(value) is bool and isinstance(got,str):got={'True':True,'False':False}[got]
            need(got==value,'Independent table cell '+str(k)+'/'+field)


def invented():
    tests={}
    support,samples=source_samples([48.,48.],48.,[96,96]);tests['grid96_all_output_positions_preserved']=len(samples)==9216 and bool(support)
    sparse,ss=source_samples([256.,256.],384.,[512,512]);end,_=candidate([[248.,248.,265.,265.]],[512,512]);tests['sampling_holes_not_no_opportunity']=bool(end-sparse) and sum(bool(s&end)for s in ss)>0
    zero,zsamples=source_samples([1000.,1000.],1.,[96,96]);tests['fully_invalid_positions_retained_empty_taps']=not zero and len(zsamples)==9216
    pp,oo=candidate([[2.25,3.25,3.25,4.25]],[8,8]);tests['closed_conservative_fractional_hull']=pp=={(x,y)for x in (2,3,4)for y in (3,4,5)} and not oo
    tests['halfopen_paste_boundary']=(6,5)not in rectangle([4,4,6,6]) and (4,4)in rectangle([4,4,6,6])
    tests['fixed_count_bins']=tuple(bucket(x)for x in [0,1,2,5,9,17,33,65,None])==BINS
    try:bucket(True)
    except AssertionError:tests['bool_not_sample_count']=True
    r=dict(original_input=dict(native_slot=0,query_xy=[48.,48.],L=48.,raw_hw=[96,96],original_query_available=True,feature_constructed=True),endpoint_observation=dict(authority='SOURCE_RGB_SEMANTIC_END_WEAK',native_slot=0,evidence_binding={'invented':1},end_locality_id='INVENTED',candidate_enclosures_xyxy=[[45.,45.,48.,48.]],surrounding_material_witnesses=[dict(raw_pixel_xy=[56,46],material_observed=True,same_selected_part_supported=True,evidence_binding={'invented':1})],**{k:True for k in ('selected_instance_supported','intended_part_supported','queried_end_localization_supported','all_plausible_candidates_enclosed','before_end_visible_supported')}),placement_geometry=dict(cover_bbox_xyxy=[44,44,50,50],sham_translation_xy=[10,0]))
    g=geometry(r,{});tests['opposite_dx_exact_single_candidate']=g['opposite_sham_bbox_xyxy']==[34,44,40,50] and g['bbox_only_possible'] and not g['old_geometry_checks']['all_chosen_witnesses_outside_sham']
    r['endpoint_observation']['surrounding_material_witnesses'][0]['raw_pixel_xy']=[46,46];tests['COVER_witness_cannot_be_repaired_by_SHAM']=not geometry(r,{})['bbox_only_possible']
    r['endpoint_observation']['queried_end_localization_supported']=False;tests['semantic_UNKNOWN_not_promoted']=geometry(r,{})['bbox_only_possible'] is None
    need(all(tests.values()),'Independent invented fixtures')
    return dict(completed=True,PASS=True,invented_only=True,checks=tests,source_or_saved_diagnostic_body_reads=0)


def prepare():
    need(bind(SCOPE)['sha256']==SCOPE_SHA,'Root source-only QA scope')
    scope=read(SCOPE);write(CHECKS,invented())
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),checks=bind(CHECKS),scope=bind(SCOPE),complete=scope['actual_completion'],producer_entry=scope['producer_entry'],addendum=scope['root_addendum'],lineage=scope['source_lineage'],output=str(OUT),rules=['No producer/evaluator imports; separate separable positive-four-tap scalar basis and explicit closed-candidate/half-open-paste set algebra.','All1024/66 compact row META/status/quantities/booleans and original/opposite geometry equality. All table strata/native8/material/factor/U/missing/bins independently recomputed; every CSV cell and summary table checked.','Exactly one dx sign flip, dy0 unchanged; bbox-only feasibility not opacity/RGB/newlabel/trainingadmission. Source quantities are prior saved evidence, not new pixel measures.','All source/published report text only; no array/header/pixel/alpha/model/GT/evaluation reads. Future root report text QA is separately recorded without rerunning source.'],automatic_retry=False)
    write(PROTOCOL,protocol);write(H/'ACTUAL_QA_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_independent_source_metadata_QA=False,code=protocol['code'],protocol=bind(PROTOCOL),scope=protocol['scope'],complete=protocol['complete'],output=str(OUT),automatic_retry=False))
    write(H/'ACTUAL_QA_FINAL_READY.json',dict(completed=True,PASS=True,prepared_only=True,code=protocol['code'],protocol=bind(PROTOCOL),checks=protocol['checks'],template=bind(H/'ACTUAL_QA_ENTRY_TEMPLATE_APPROVED_FALSE.json'),root_authorization_scope=protocol['scope'],actual_source_diagnostic_body_reads=0))
    print(json.dumps(dict(prepared=True,code=protocol['code'],protocol=bind(PROTOCOL))))


def run(entry):
    e=read(entry);need(e['approved_independent_source_metadata_QA'] is True and e['automatic_retry'] is False,'Root-authorized one source metadata run')
    p=read(checked(e['protocol']));need(e['code']==p['code']==bind(__file__) and e['scope']==p['scope']==bind(SCOPE) and e['complete']==p['complete'],'Frozen own code/protocol/scope/complete')
    need(e['output']==p['output']==str(OUT) and read(checked(p['checks']))['PASS'],'Output/checks closure')
    scope=read(checked(p['scope']));need(scope['approved_independent_metadata_diagnostic_QA'] is True and scope['actual_completion']==e['complete'],'Root actual complete authority')
    done=read(checked(e['complete']));need(done['completed'] is True and done['PASS'] is True and done['source_only'] is True and done['entry']==p['producer_entry'],'Completed source diagnostic')
    add=read(checked(p['addendum']));need(done['inputs']==add['exact_JSONL_inputs'],'Exact two source JSONLs')
    OUT.mkdir(exist_ok=False);write(OUT/'STARTED.json',dict(entry=bind(entry),automatic_retry=False))
    try:
        src={d:rows(done['inputs'][k])for d,k in [('SYNTHETIC','synthetic'),('REAL','real')]}
        need(len(src['SYNTHETIC'])==1024 and len(src['REAL'])==66,'Full population')
        need({(r['base_index'],r['native_slot'])for r in src['SYNTHETIC']}=={(b,k)for b in range(128)for k in range(8)},'All128/native8 original cube')
        cache={};expected=[compact(d,r,i,cache)for d in ('SYNTHETIC','REAL')for i,r in enumerate(src[d])]
        outputs=done['outputs']
        observed=rows(outputs['ALL1024_SYNTHETIC_ATTEMPT_DIAGNOSTICS.jsonl'])+rows(outputs['ALL66_REAL_BBOX_ONLY_DIAGNOSTICS.jsonl'])
        need(observed==expected,'Every1090 diagnostic row independently exact')
        pt,dt,gt,bt=expected_tables(expected)
        csv_specs=[('POPULATION_AND_STORED_STATUS_BY_STRATUM.csv',pt,('domain','stratum','key'),('stored_status_counts','overlapping_stored_reason_counts')),('STORED_END96_FIXED_BIN_DISTRIBUTIONS.csv',dt,('domain','cohort','quantity','fixed_bin'),()),('REAL_OVERLAPPING_GEOMETRY_COUNTS.csv',gt,('scope','check'),()),('STORED_REAL_CHANGE_BOOLEAN_COUNTS.csv',bt,('cohort','stored_boolean'),())]
        for name,wanted,keys,jf in csv_specs:
            actual=list(csv.DictReader(io.StringIO(data(checked(outputs[name])).decode())))
            compare_table(actual,wanted,keys,jf)
        summary=read(checked(outputs['SUMMARY.json']))
        for field,table,keys,jf in [('full_population_tables',pt,('domain','stratum','key'),('stored_status_counts','overlapping_stored_reason_counts')),('fixed_bin_distributions',dt,('domain','cohort','quantity','fixed_bin'),()),('real_geometry_overlapping_counts',gt,('scope','check'),()),('stored_real_change_boolean_counts',bt,('cohort','stored_boolean'),())]:compare_table(summary[field],table,keys,jf)
        statuses={d:dict(Counter(r['stored_status']for r in expected if r['domain']==d))for d in ('SYNTHETIC','REAL')}
        supported={d:sum(r['stored_supported']for r in expected if r['domain']==d)for d in ('SYNTHETIC','REAL')}
        rr=[r for r in expected if r['domain']=='REAL'];possible=[r for r in rr if r['geometry_bbox_only']['bbox_only_possible'] is True]
        need(summary['stored_status']==statuses and summary['stored_supported']==supported and summary['opposite_bbox_only_possible_count']==len(possible),'Independent headline counts')
        need(summary['new_labels_generated'] is False and summary['training_admission'] is False and summary['fully_solved'] is False,'No authority or goal promotion')
        lineage=read(checked(p['lineage']));need(lineage['direct']['known_rows']==306 and lineage['direct']['synthetic_accepted_source_bases']==47 and lineage['direct']['real_accepted_source_frames']==2,'Frozen lineage source scope matches')
        result=dict(completed=True,PASS=True,review_type='INDEPENDENT_SOURCE_METADATA_NUMERICAL_QA',entry=bind(entry),code=bind(__file__),protocol=bind(PROTOCOL),actual_completion=e['complete'],producer_entry=p['producer_entry'],source_lineage=p['lineage'],all1090_rows_and_every_csv_summary_table_cell_exact=True,coverage=dict(synthetic1024=1024,real66=66,original_native_slots=list(range(8)),source_selected_inputU={d:sum(not r['original_query_available']for r in expected if r['domain']==d)for d in ('SYNTHETIC','REAL')},stored_supported=supported,stored_status=statuses,population_strata=len(pt),fixed_bin_cells=len(dt),geometry_cells=len(gt),boolean_cells=len(bt),opposite_bbox_only_possible=len(possible),opposite_possible_by_original_status=dict(Counter(r['stored_status']for r in possible)),old_supported_fails_opposite=sum(r['stored_supported'] and r['geometry_bbox_only']['bbox_only_possible'] is False for r in rr),geometry_unknown_null=sum(r['geometry_bbox_only']['bbox_only_possible'] is None for r in rr)),opposite_possible_pair_ids=[r['identity']['pair_id']for r in possible],no_new_labels_or_admission=True,original_statuses_and_source_outputs_immutable=True,source_arrays_pixels_alpha_model_evaluation_reads=0,automatic_retry=False,limits=['All checks are source text/metadata arithmetic, not new alpha/RGB/frontdepth/instance/end semantic certification.','Stored sample-count histograms summarize existing evidence; one dx reflection is the new hypothetical bbox question. No26 new triplet/admission claim.','All306 direct rows were training data;47synthetic bases+2real frames provide no independent holdout for these frozen heads.','Root report text QA is not yet part of this numerical receipt; it will receive a separate receipt once available.'],adopted=False,fully_solved=False)
        write(OUT/'REVIEW.json',result)
        print(json.dumps(dict(completed=True,PASS=True,review=bind(OUT/'REVIEW.json'),coverage=result['coverage'])))
    except BaseException:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',dict(completed=False,PASS=False,entry=bind(entry),automatic_retry=False,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare()if a.prepare else run(a.entry)
