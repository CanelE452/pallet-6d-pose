"""V192 exact source JSONL diagnostics; no arrays, images, models or labels made.

Preparation uses root JSON/code text and invented records only. Actual source
JSONL hashes/bodies require a separate frozen root entry. Opposite SHAM is one
bbox-only proposal, never an opacity/RGB proof, new label, or admission.
"""
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, io, json, math, os, stat

H=Path(__file__).absolute().parent; V=H.parent; P=V.parent
SCOPE=V/'ROOT_SOURCE_ONLY_AUDIT_SCOPE_BEFORE_SOURCE_BODY_COMPUTATIONS.json'
ADDENDUM=V/'ROOT_PROSPECTIVE_SOURCE_METADATA_DIAGNOSTIC_ADDENDUM.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_JSONL_DIAGNOSTICS.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS.json'; OUT=V/'source_metadata_diagnostic_v1'
BINS=('0','1','2~4','5~8','9~16','17~32','33~64','65이상','missing')
SYNTH_SUPPORTED='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED'
REAL_SUPPORTED='SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED'
QUANT_KEYS=('end_related96_samples','COVER_changed_RGB96_samples','SHAM_changed_RGB96_samples')

def require(x,m):
    if not bool(x): raise AssertionError(m)

def safe(value):
    p=Path(value).absolute()
    require(P in p.parents and '..' not in p.parts and not any('v112' in s.casefold() for s in p.parts),'Explicit source path before any access')
    return p

def opened(value,create=False):
    p=safe(value); fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd); os.close(fd); fd=nxt
        last=os.open(p.name,(os.O_WRONLY|os.O_CREAT|os.O_EXCL if create else os.O_RDONLY)|os.O_NOFOLLOW,0o644,dir_fd=fd)
    finally: os.close(fd)
    require(stat.S_ISREG(os.fstat(last).st_mode),'Ordinary file, no symlink')
    return os.fdopen(last,'wb' if create else 'rb')

def sha(path):
    h=hashlib.sha256()
    with opened(path) as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def bind(p): return dict(path=str(safe(p)),sha256=sha(p))
def read(p):
    with opened(p) as f:return json.load(f)
def checked(b): require(bind(b['path'])==b,'Exact frozen binding'); return safe(b['path'])
def write(p,d):
    with opened(p,True) as f:f.write((json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def jlines(p,rows):
    with opened(p,True) as f:
        for r in rows:f.write((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode())
def csvwrite(p,rows):
    s=io.StringIO(); w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with opened(p,True) as f:f.write(s.getvalue().encode())

def bin_count(value):
    if value is None:return 'missing'
    require(type(value) is int and 0<=value<=9216,'Stored sample count, not a boolean or invented RGB magnitude')
    if value<=1:return str(value)
    for lo,hi in ((2,4),(5,8),(9,16),(17,32),(33,64)):
        if value<=hi:return '%d~%d'%(lo,hi)
    return '65이상'

def finite_q(q):
    return isinstance(q,list) and len(q)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in q)

def rgb_taps(x,y,hw):
    """Independent scalar original F64 grid/positive bilinear source support."""
    h,w=hw
    if not (-.5<=x<w-.5 and -.5<=y<h-.5):return []
    ix,iy=math.floor(x),math.floor(y);dx,dy=x-ix,y-iy
    return [(ix+a,iy+b,z) for a,b,z in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)) if z>0 and 0<=ix+a<w and 0<=iy+b<h]

def grid_support(original):
    q,L,hw=original.get('query_xy'),original.get('L'),original.get('raw_hw')
    require(finite_q(q) and type(L) in (int,float) and math.isfinite(L) and L>0 and len(hw)==2 and all(type(z) is int and z>0 for z in hw),'Original q/L/HW')
    support=set(); samples=[]
    for j in range(96):
        for i in range(96):
            taps=rgb_taps(q[0]+((i+.5)/96-.5)*L,q[1]+((j+.5)/96-.5)*L,hw)
            points=tuple((x,y) for x,y,weight in taps);support.update(points);samples.append(points)
    return support,samples

def candidate_pixels(boxes,hw):
    require(bool(boxes),'All frozen candidate alternatives, never shrink/delete')
    h,w=hw;pixels=set();outside=False
    for box in boxes:
        require(len(box)==4 and all(type(x) in (int,float) and math.isfinite(x) for x in box),'Closed continuous candidate')
        x0,y0,x1,y1=box;require(x0<=x1 and y0<=y1,'Ordered closed enclosure')
        outside|=not (0<=x0<=x1<w and 0<=y0<=y1<h)
        for y in range(math.floor(y0),math.ceil(y1)+1):
            for x in range(math.floor(x0),math.ceil(x1)+1):
                if 0<=x<w and 0<=y<h:pixels.add((x,y))
                else:outside=True
    return pixels,outside

def in_box(p,b):return b[0]<=p[0]<b[2] and b[1]<=p[1]<b[3]
def translated(b,dx,dy):return [b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy]
def in_frame(b,hw):return 0<=b[0]<b[2]<=hw[1] and 0<=b[1]<b[3]<=hw[0]
def bbox_intersects_candidate(b,pixels):return any(in_box(p,b) for p in pixels)
def weak_localized(e,o):
    return e.get('authority')=='SOURCE_RGB_SEMANTIC_END_WEAK' and e.get('native_slot')==o['native_slot'] and bool(e.get('evidence_binding')) and bool(e.get('end_locality_id')) and all(e.get(k) is True for k in ('selected_instance_supported','intended_part_supported','queried_end_localization_supported','all_plausible_candidates_enclosed','before_end_visible_supported'))

def real_geometry(row,cache):
    o=row['original_input'];e=row['endpoint_observation'];placement=row.get('placement_geometry')
    d=dict(weak_semantic_localization_supported=bool(weak_localized(e,o)),placement_metadata_available=placement is not None,
        original_input_available=o['original_query_available'],old_geometry_checks=None,opposite_geometry_checks=None,bbox_only_possible=None,
        proposal='EXACTLY_ONE: existing integer dx sign flipped, original dy asserted0',actual_alpha_RGB_identity_front_depth_verified=False,new_direct_label=None,training_admission=False)
    if not d['weak_semantic_localization_supported']:
        d['diagnostic_reason']='FROZEN_WEAK_OBSERVATION_UNKNOWN_UNCHANGED';return d
    key=json.dumps([o['query_xy'],o['L'],o['raw_hw']],separators=(',',':'))
    if key not in cache:cache[key]=grid_support(o)
    support,samples=cache[key];boxes=e['candidate_enclosures_xyxy'];pixels,outside=candidate_pixels(boxes,o['raw_hw'])
    q,L=o['query_xy'],o['L'];extent=[q[0]+(.5/96-.5)*L,q[1]+(.5/96-.5)*L,q[0]+(95.5/96-.5)*L,q[1]+(95.5/96-.5)*L]
    candidate_rows=[]
    for b in boxes:
        pp,oo=candidate_pixels([b],o['raw_hw']);n=sum(any(p in pp for p in ss) for ss in samples)
        candidate_rows.append(dict(enclosure_xyxy=b,in_original_sample_extent=extent[0]<=b[0]<=b[2]<=extent[2] and extent[1]<=b[1]<=b[3]<=extent[3],raw_footprint_in_frame=not oo,positive_original96_samples=n))
    witnesses=[tuple(w['raw_pixel_xy']) for w in e.get('surrounding_material_witnesses',[]) if w.get('material_observed') is True and w.get('same_selected_part_supported') is True and w.get('evidence_binding')]
    require(all(len(w)==2 and all(type(z) is int for z in w) for w in witnesses),'Frozen integer witnesses, no replacement')
    d.update(original96_sample_extent=extent,candidates=candidate_rows,chosen_raw_witnesses=[dict(raw_pixel_xy=list(w),positive_original96_tap=w in support,outside_candidate=w not in pixels) for w in witnesses],all_chosen_witnesses_positive96=bool(witnesses) and all(w in support for w in witnesses))
    if placement is None:
        d['diagnostic_reason']='NO_STORED_PLACEMENT_NO_ALTERNATIVE_FABRICATED';return d
    cover=placement['cover_bbox_xyxy'];require(len(cover)==4 and all(type(z) is int for z in cover),'Original half-open integer paste bbox')
    shift=placement['sham_translation_xy'];require(len(shift)==2 and all(type(z) is int for z in shift) and shift[1]==0 and shift[0]!=0,'Original declared horizontal dx, dy0')
    old=translated(cover,*shift);opposite=translated(cover,-shift[0],shift[1])
    def checks(sham):
        return dict(cover_bbox_contains_all_end_support=bool(pixels) and all(in_box(p,cover) for p in pixels),
            sham_bbox_disjoint_all_end_support=not bbox_intersects_candidate(sham,pixels),
            cover_and_sham_in_frame=in_frame(cover,o['raw_hw']) and in_frame(sham,o['raw_hw']),
            same_bbox_and_positive_mass_translation=True,
            sham_has_some_original_rgb96_support=any(in_box(p,sham) for p in support),
            certified_surrounding_part_retained_by_both_bbox_transforms=bool(witnesses) and all(w in support and w not in pixels and not in_box(w,cover) and not in_box(w,sham) for w in witnesses),
            all_chosen_witnesses_outside_cover=bool(witnesses) and all(not in_box(w,cover) for w in witnesses),
            all_chosen_witnesses_outside_sham=bool(witnesses) and all(not in_box(w,sham) for w in witnesses),
            cover_bbox_intersects_end_candidate=bbox_intersects_candidate(cover,pixels),
            sham_bbox_intersects_end_candidate=bbox_intersects_candidate(sham,pixels))
    oldchecks=checks(old);opchecks=checks(opposite);saved=row.get('finite_end_geometry',{}).get('geometry_controls')
    if saved is not None:require(all(oldchecks[k]==v for k,v in saved.items()),'Independent old bbox checks equal saved geometry flags')
    prerequisite=bool(o['original_query_available']) and bool(o.get('feature_constructed')) and not outside and bool(candidate_rows) and all(r['in_original_sample_extent'] and r['raw_footprint_in_frame'] and r['positive_original96_samples']>0 for r in candidate_rows)
    necessary=('cover_bbox_contains_all_end_support','sham_bbox_disjoint_all_end_support','cover_and_sham_in_frame','sham_has_some_original_rgb96_support','certified_surrounding_part_retained_by_both_bbox_transforms')
    d.update(cover_bbox_xyxy=cover,existing_sham_bbox_xyxy=old,opposite_sham_bbox_xyxy=opposite,existing_translation_xy=shift,opposite_translation_xy=[-shift[0],0],old_geometry_checks=oldchecks,opposite_geometry_checks=opchecks,same_frozen_end_and_input_opportunity=prerequisite,bbox_only_possible=prerequisite and all(opchecks[k] for k in necessary),diagnostic_reason='BBOX_ONLY_NOT_ALPHA_RGB_OR_NEW_TRIPLET_PROOF')
    return d

def quantities(domain,row):
    if domain=='SYNTHETIC':
        ev=row.get('closed_proof',{}).get('actual_proof_metadata',{}).get('evidence_bindings',{})
        values=dict(end_related96_samples=ev.get('end_related96_samples'),COVER_changed_RGB96_samples=ev.get('end_related96_COVER_changed_RGB_samples'),SHAM_changed_RGB96_samples=ev.get('end_related96_SHAM_changed_RGB_samples'))
        bools={}
    else:
        ev=row.get('pixel_detail',{});proof=row.get('pixel_checks',{})
        values=dict(end_related96_samples=ev.get('end_related_RGB96_sample_count'),COVER_changed_RGB96_samples=None,SHAM_changed_RGB96_samples=None)
        bools={k:proof.get(k) for k in ('end_support_actual_original96_bytes_changed_COVER','end_support_actual_original96_bytes_unchanged_SHAM')}
    for value in values.values():bin_count(value)
    if values['end_related96_samples'] is not None:
        require(all(v is None or v<=values['end_related96_samples'] for k,v in values.items() if k!='end_related96_samples'),'Changed count bounded by stored sample mask')
    return values,bools

def supported(domain,r):return r['status']==(SYNTH_SUPPORTED if domain=='SYNTHETIC' else REAL_SUPPORTED) and (domain=='SYNTHETIC' or r['PASS'] is True)
def reasons(row):
    z=row.get('reasons',row.get('reason',[]));return [z] if isinstance(z,str) else z

def compact(domain,row,index,cache):
    values,bools=quantities(domain,row);known=supported(domain,row)
    if domain=='SYNTHETIC':
        native=row['native_slot'];factors=row['factors'];identity={k:row.get(k) for k in ('attempt_id','base_index','original_global_index','original_occurrence_id','query_xy','L','original_input_flags')}
        require(row['split']=='TRAIN','Original synthetic TRAIN only')
    else:
        native=row['original_input']['native_slot'];factors={};identity={k:row.get(k) for k in ('pair_id','case_id','donor_id','original_input')}
        require(row['original_input']['development_role']=='SOURCE_DEVELOPMENT','Original public source TRAIN role')
    d=dict(domain=domain,source_census_row_index=index,identity=identity,native_slot=native,factors=factors,stored_status=row['status'],stored_supported=known,stored_reasons=reasons(row),stored_quantities=values,stored_change_booleans=bools,new_label_generated=False,training_admission=False)
    if domain=='REAL':d['geometry_bbox_only']=real_geometry(row,cache)
    return d

def population_tables(records):
    tables=[]
    for domain in ('SYNTHETIC','REAL'):
        rr=[r for r in records if r['domain']==domain];groups=[('ALL','ALL',rr)]
        groups += [('native',str(k),[r for r in rr if r['native_slot']==k]) for k in range(8)]
        for factor in ('material','aspect','geometry_family','foreign_family'):
            names=sorted({str(r['factors'].get(factor,'NOT_DECLARED_IN_REAL_MANIFEST')) for r in rr})
            groups += [(factor,name,[r for r in rr if str(r['factors'].get(factor,'NOT_DECLARED_IN_REAL_MANIFEST'))==name]) for name in names]
        cells=sorted({tuple(str(r['factors'].get(k,'NOT_DECLARED_IN_REAL_MANIFEST')) for k in ('material','aspect','geometry_family')) for r in rr})
        for cell in cells:
            subset=[r for r in rr if tuple(str(r['factors'].get(k,'NOT_DECLARED_IN_REAL_MANIFEST')) for k in ('material','aspect','geometry_family'))==cell]
            groups.append(('material_aspect_geometry','|'.join(cell),subset))
            groups += [('material_aspect_geometry_native','|'.join(cell)+'|native'+str(k),[r for r in subset if r['native_slot']==k]) for k in range(8)]
        for kind,key,rows in groups:
            cc=Counter(r['stored_status'] for r in rows);rs=Counter(z for r in rows for z in set(r['stored_reasons']))
            tables.append(dict(domain=domain,stratum=kind,key=key,attempts=len(rows),stored_supported=sum(r['stored_supported'] for r in rows),stored_nonsupported=sum(not r['stored_supported'] for r in rows),stored_status_counts=json.dumps(dict(cc),sort_keys=True),overlapping_stored_reason_counts=json.dumps(dict(rs),sort_keys=True)))
    return tables

def distributions(records):
    out=[]
    for domain in ('SYNTHETIC','REAL'):
        for cohort in ('ALL_ATTEMPTS','STORED_SUPPORTED_ONLY'):
            rr=[r for r in records if r['domain']==domain and (cohort=='ALL_ATTEMPTS' or r['stored_supported'])]
            for field in QUANT_KEYS:
                cc=Counter(bin_count(r['stored_quantities'][field]) for r in rr)
                for b in BINS:out.append(dict(domain=domain,cohort=cohort,quantity=field,fixed_bin=b,count=cc[b],cohort_denominator=len(rr),definition='Stored integer sample count only; bool is never a changed-sample count'))
    return out

def geometry_counts(records):
    rows=[r['geometry_bbox_only'] for r in records if r['domain']=='REAL'];out=[]
    names=('weak_semantic_localization_supported','placement_metadata_available','all_chosen_witnesses_positive96','bbox_only_possible')
    for name in names:
        cc=Counter('missing' if r.get(name) is None else str(r[name]) for r in rows);out.append(dict(scope='ALL66',check=name,true_count=cc['True'],false_count=cc['False'],missing_count=cc['missing'],denominator=66,counts_can_overlap=True))
    for side in ('old_geometry_checks','opposite_geometry_checks'):
        keys=sorted({k for r in rows if r.get(side) for k in r[side]})
        for key in keys:
            cc=Counter('missing' if not r.get(side) or key not in r[side] else str(r[side][key]) for r in rows)
            out.append(dict(scope=side,check=key,true_count=cc['True'],false_count=cc['False'],missing_count=cc['missing'],denominator=66,counts_can_overlap=True))
    return out

def invented():
    tests={}
    taps=rgb_taps(.25,.75,[10,10]);tests['independent_scalar_four_tap_weights']=taps==[(0,0,.1875),(1,0,.0625),(0,1,.5625),(1,1,.1875)]
    tests['whole_invalid_sample_and_partial_zero_padding_distinct']=rgb_taps(9.5,0,[10,10])==[] and math.fsum(z for x,y,z in rgb_taps(-.25,0,[10,10]))==.75
    tests['fixed_bins_missing_not_zero_and_bool_rejected']=tuple(bin_count(x) for x in [0,1,2,5,9,17,33,65,None])==BINS
    try:bin_count(True)
    except AssertionError:tests['boolean_cannot_be_converted_to_changed_count']=True
    o=dict(development_role='SOURCE_DEVELOPMENT',native_slot=0,query_xy=[48.,48.],L=48.,raw_hw=[96,96],original_query_available=True,feature_constructed=True)
    e=dict(authority='SOURCE_RGB_SEMANTIC_END_WEAK',native_slot=0,end_locality_id='INVENTED',evidence_binding={'invented':True},candidate_enclosures_xyxy=[[45.,45.,48.,48.]],surrounding_material_witnesses=[dict(raw_pixel_xy=[56,46],material_observed=True,same_selected_part_supported=True,evidence_binding={'invented':True})],**{k:True for k in ('selected_instance_supported','intended_part_supported','queried_end_localization_supported','all_plausible_candidates_enclosed','before_end_visible_supported')})
    row=dict(original_input=o,endpoint_observation=e,placement_geometry=dict(cover_bbox_xyxy=[44,44,50,50],sham_translation_xy=[10,0]))
    a=real_geometry(row,{});tests['one_opposite_proposal_preserves_cover_and_avoids_old_sham_witness']=a['bbox_only_possible'] and not a['old_geometry_checks']['all_chosen_witnesses_outside_sham'] and a['opposite_sham_bbox_xyxy']==[34,44,40,50]
    bad=json.loads(json.dumps(row));bad['endpoint_observation']['surrounding_material_witnesses'][0]['raw_pixel_xy']=[46,46]
    tests['cover_witness_failure_not_repaired_by_opposite_sham']=real_geometry(bad,{})['bbox_only_possible'] is False
    bad=json.loads(json.dumps(row));bad['endpoint_observation']['queried_end_localization_supported']=False
    tests['weak_UNKNOWN_never_promoted_by_bbox_geometry']=real_geometry(bad,{})['bbox_only_possible'] is None
    bad=json.loads(json.dumps(row));bad['placement_geometry']['sham_translation_xy']=[10,1]
    try:real_geometry(bad,{})
    except AssertionError:tests['original_dy0_is_asserted_not_modified']=True
    bad=json.loads(json.dumps(row));bad['placement_geometry']['sham_translation_xy']=[60,0]
    tests['opposite_OOF_not_clipped_or_searched']=real_geometry(bad,{})['bbox_only_possible'] is False
    sparse=dict(o,query_xy=[256.,256.],L=384.,raw_hw=[512,512]);support,samples=grid_support(sparse);pixels,_=candidate_pixels([[248.,248.,265.,265.]],[512,512])
    tests['L384_internal_raw_holes_positive_sample_opportunity_distinct']=bool(pixels-support) and any(any(p in pixels for p in ss) for ss in samples)
    val,boo=quantities('REAL',dict(pixel_detail={'end_related_RGB96_sample_count':4},pixel_checks={'end_support_actual_original96_bytes_changed_COVER':True}))
    tests['real_stored_mask_count_but_changed_count_missing_not_one']=val['end_related96_samples']==4 and val['COVER_changed_RGB96_samples'] is None and boo['end_support_actual_original96_bytes_changed_COVER'] is True
    require(all(tests.values()),'Meaningful invented source metadata fixtures')
    return dict(completed=True,PASS=True,invented_only=True,tests=tests,number_checks=len(tests),source_JSONL_hash_body_or_array_pixel_model_eval_reads=0)

def prepare():
    scope=read(SCOPE);add=read(ADDENDUM);require(add['approved_preparation_only'] and add['root_scope']==bind(SCOPE),'Prospective root-only metadata scope')
    write(CHECKS,invented());p=dict(completed=True,prepared_only=True,code=bind(__file__),root_scope=bind(SCOPE),addendum=bind(ADDENDUM),inputs=add['exact_JSONL_inputs'],checks=bind(CHECKS),fixed_bins=list(BINS),counts={'synthetic_attempts':1024,'real_attempts':66,'native_slots':list(range(8))},source_inputs_opened_or_hashed_in_preparation=False,opposite_SHAM='Single exact dx sign flip; dy asserted0; bbox-only feasibility, alpha/RGB/identity/frontdepth not verified; no new labels/admission',output_directory=str(OUT),no_model_array_pixel_eval_or_training_reads=True)
    write(PROTOCOL,p);template=H/'ROOT_ACTUAL_METADATA_DIAGNOSTIC_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_source_metadata_diagnostic=False,arrays_pixels_models_eval_or_training=False,code=p['code'],protocol=bind(PROTOCOL),root_scope=p['root_scope'],addendum=p['addendum'],inputs=p['inputs'],output_directory=str(OUT),automatic_retry=False))
    done=dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=bind(CHECKS),false_entry=bind(template),actual_source_JSONL_run_or_hash_before_entry=0)
    write(H/'FINAL_READY.json',done);print(json.dumps(done))

def run(entry):
    e=read(entry);require(e['approved_source_metadata_diagnostic'] is True and e['arrays_pixels_models_eval_or_training'] is False and e['automatic_retry'] is False,'Root actual source JSONL only entry')
    p=read(checked(e['protocol']));require(e['code']==bind(__file__)==p['code'] and e['root_scope']==p['root_scope']==bind(SCOPE) and e['addendum']==p['addendum']==bind(ADDENDUM),'Frozen code/scope closure')
    add=read(checked(p['addendum']));require(e['inputs']==p['inputs']==add['exact_JSONL_inputs'] and e['output_directory']==str(OUT)==p['output_directory'],'Only exact two source JSONLs')
    require(read(checked(p['checks']))['PASS'],'Invented checks closure');require(not OUT.exists(),'Exclusive fresh diagnostic output')
    sources={}
    for domain,key,n in [('SYNTHETIC','synthetic',1024),('REAL','real',66)]:
        with opened(checked(p['inputs'][key])) as f:rows=[json.loads(x) for x in f if x.strip()]
        require(len(rows)==n,'Full original attempt population');sources[domain]=rows
    require(len({r['attempt_id'] for r in sources['SYNTHETIC']})==1024 and {(r['base_index'],r['native_slot']) for r in sources['SYNTHETIC']}=={(b,k) for b in range(128) for k in range(8)},'Original all128/native8')
    require(len({r['pair_id'] for r in sources['REAL']})==66 and len({r['case_id'] for r in sources['REAL']})==22,'Original all22/66, no replacement')
    cache={};records=[compact(domain,r,i,cache) for domain in ('SYNTHETIC','REAL') for i,r in enumerate(sources[domain])]
    OUT.mkdir();jlines(OUT/'ALL1024_SYNTHETIC_ATTEMPT_DIAGNOSTICS.jsonl',[r for r in records if r['domain']=='SYNTHETIC']);jlines(OUT/'ALL66_REAL_BBOX_ONLY_DIAGNOSTICS.jsonl',[r for r in records if r['domain']=='REAL'])
    tables=population_tables(records);bins=distributions(records);geometry=geometry_counts(records)
    csvwrite(OUT/'POPULATION_AND_STORED_STATUS_BY_STRATUM.csv',tables);csvwrite(OUT/'STORED_END96_FIXED_BIN_DISTRIBUTIONS.csv',bins);csvwrite(OUT/'REAL_OVERLAPPING_GEOMETRY_COUNTS.csv',geometry)
    summary=dict(completed=True,PASS=True,source_only=True,attempts={'synthetic':1024,'real':66},stored_status={d:dict(Counter(r['stored_status'] for r in records if r['domain']==d)) for d in ('SYNTHETIC','REAL')},stored_supported={d:sum(r['stored_supported'] for r in records if r['domain']==d) for d in ('SYNTHETIC','REAL')},opposite_bbox_only_possible_count=sum(r.get('geometry_bbox_only',{}).get('bbox_only_possible') is True for r in records),real_changed_sample_count_not_stored=True,source_quantities_not_recomputed_from_pixels=True,full_population_tables=tables,fixed_bin_distributions=bins,real_geometry_overlapping_counts=geometry,limitations=['Stored synthetic end-mask/changed-sample counts are quantities; real change proof is boolean only, never interpreted as RGB magnitude or sample count.','Opposite dx is a single bbox-only diagnostic. No alpha, actual RGB change, target identity or front-depth proof is added; no new supported pair or labels/admission.','Existing6 weak-UNKNOWN cases remain UNKNOWN. All failures/U and native0..7 denominators retained; missing real material/aspect/factors not guessed.'],all_existing_source_outputs_immutable=True,new_labels_generated=False,training_admission=False,goal_status='active_unsolved',fully_solved=False)
    write(OUT/'SUMMARY.json',summary)
    for b in p['inputs'].values():checked(b)
    outputs={name:bind(OUT/name) for name in ['ALL1024_SYNTHETIC_ATTEMPT_DIAGNOSTICS.jsonl','ALL66_REAL_BBOX_ONLY_DIAGNOSTICS.jsonl','POPULATION_AND_STORED_STATUS_BY_STRATUM.csv','STORED_END96_FIXED_BIN_DISTRIBUTIONS.csv','REAL_OVERLAPPING_GEOMETRY_COUNTS.csv','SUMMARY.json']}
    write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,entry=bind(entry),code=p['code'],protocol=bind(PROTOCOL),inputs=p['inputs'],outputs=outputs,all1024_and66_attempts_retained=True,arrays_pixels_models_evaluation_or_training_read=0,new_labels_generated=False,training_admission=False,automatic_retry=False))
    print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'),stored_status=summary['stored_status'],opposite_bbox_only_possible_count=summary['opposite_bbox_only_possible_count'])))

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.entry)
