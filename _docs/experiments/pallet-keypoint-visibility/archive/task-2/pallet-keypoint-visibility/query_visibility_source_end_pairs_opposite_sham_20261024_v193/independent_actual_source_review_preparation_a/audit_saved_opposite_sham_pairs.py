"""V193 independent saved source audit. Preparation is text/invented-only.

The reused numerical kernels are our prior independent audit, not a producer.
No producer imports, model/evaluation calls, new labels or new composites exist.
"""
from pathlib import Path
from collections import Counter
import argparse,ast,hashlib,itertools,json,math,os,stat,traceback
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
C=V/'real_end_opposite_sham_preparation_c';GEN=V/'real_end_opposite_sham_pairs_v1'
SCOPE=V/'ROOT_PROSPECTIVE_SOURCE_OPPOSITE_SHAM_POLICY_BEFORE_ASSET_READS.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_SOURCE_ARRAY_AUDIT.json';CHECKS=H/'GENERATED_INVENTED_AUDIT_CHECKS.json'
OLD_KERNEL=P/'query_visibility_sampled_domain_direct_end_20261024_v189/independent_actual_real_review_a/audit_real_end_pairs.py'
OUT=V/'independent_actual_source_review_a/actual_review_v1'
NAMES=('BEFORE','COVER','ADJACENT_SHAM')
KERNELS=('eq','ash','taps','footprint','enclosure','sampled_relation','local_sample','context_sample','raw_mask','resized_donor','invented')

def need(x,m):
    if not bool(x):raise AssertionError(m)
def safe(value):
    p=Path(value);need(p.is_absolute() and '..' not in p.parts and P in p.parents and all('v112' not in z.casefold() for z in p.parts),'Exact lexical source path before access');return p
def opened(value):
    p=safe(value);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        end=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(end).st_mode),'Ordinary file/no symlinks');return os.fdopen(end,'rb')
def sha(path):
    h=hashlib.sha256()
    with opened(path)as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(safe(p)),sha256=sha(p))
def read(p):
    with opened(p)as f:return json.load(f)
def checked(b):need(bind(b['path'])==b,'Frozen text/metadata binding');return safe(b['path'])
def bound_json(b):return read(checked(b))
def bound_lines(b):
    with opened(checked(b))as f:return [json.loads(x) for x in f if x.strip()]
def write(p,d):
    safe(p)
    with p.open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pure():
    with opened(OLD_KERNEL)as f:tree=ast.parse(f.read().decode())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in KERNELS]
    need({n.name for n in nodes}==set(KERNELS),'Exact prior independent pure numeric kernels')
    ns=dict(np=np,math=math,itertools=itertools,Image=Image,distance_transform_edt=distance_transform_edt,need=need,hashlib=hashlib)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'prior_A_independent_audit::numeric_only','exec'),ns)
    return ns
def inside(pt,b):return b[0]<=pt[0]<b[2] and b[1]<=pt[1]<b[3]

def geometry(k,original,endpoint,cover,dx,points,support):
    sham=[cover[0]+dx,cover[1],cover[2]+dx,cover[3]];hw=original['raw_hw']
    witnesses=[w['raw_pixel_xy'] for w in endpoint['surrounding_material_witnesses'] if w.get('material_observed') is True and w.get('same_selected_part_supported') is True and w.get('evidence_binding')]
    result=dict(cover_bbox_contains_all_end_support=bool(points) and all(inside(p,cover) for p in points),sham_bbox_disjoint_all_end_support=all(not inside(p,sham) for p in points),cover_and_sham_in_frame=all(0<=b[0]<b[2]<=hw[1] and 0<=b[1]<b[3]<=hw[0] for b in (cover,sham)),same_bbox_and_positive_mass_translation=dx!=0,sham_has_some_original_rgb96_support=any(inside(p,sham) for p in support),certified_surrounding_part_retained_by_both_bbox_transforms=bool(witnesses) and all(tuple(w) in support and tuple(w) not in points and not inside(w,cover) and not inside(w,sham) for w in witnesses))
    return {a:bool(b) for a,b in result.items()}

def semantic_locality(endpoint,original,k):
    localized=endpoint.get('authority')=='SOURCE_RGB_SEMANTIC_END_WEAK' and endpoint.get('native_slot')==original['native_slot'] and endpoint.get('end_locality_id') and endpoint.get('evidence_binding') and all(endpoint.get(n) is True for n in ('selected_instance_supported','intended_part_supported','queried_end_localization_supported','all_plausible_candidates_enclosed','before_end_visible_supported'))
    if not original['original_query_available'] or not original['feature_constructed'] or not localized:return False,None,None,None
    q,L=original['query_xy'],original['L'];need(q is not None and L is not None and np.isfinite(q).all() and math.isfinite(L) and L>0,'Frozen available original q/L')
    support,by=k['footprint'](q,L,original['raw_hw']);points=k['enclosure'](endpoint['candidate_enclosures_xyxy'])
    extent,rows,eligible=k['sampled_relation'](q,L,original['raw_hw'],endpoint['candidate_enclosures_xyxy'],by)
    return eligible,points,support,by

def policy_for(placement,old_checks,opposite_checks):
    need(placement['sham_translation_xy'][1]==0 and type(placement['sham_translation_xy'][0])is int and placement['sham_translation_xy'][0]!=0,'Original integer dx/dy0')
    if all(old_checks.values()):return 'ORIGINAL_RETAINED',placement['sham_translation_xy'][0],False
    return ('OPPOSITE_ONE_ATTEMPT_READY' if all(opposite_checks.values()) else 'OPPOSITE_ONE_ATTEMPT_FAILED_RETAINED'),-placement['sham_translation_xy'][0],True

def proof14(pixel,before_verified,before,cover,donor):
    return dict(original_q_L_native_flags_and_all8_inputs_unchanged=True,before_original_RGB96_valid96_target_context_exact=bool(before_verified),every_candidate_positive_RGB_tap_cover_alpha_one=pixel['end_every_candidate_raw_tap_COVER_alpha1'],every_candidate_positive_RGB_tap_sham_alpha_zero=pixel['end_every_candidate_raw_tap_SHAM_alpha0'],distinct_foreign_material_supported=donor['PASS'] is True and donor['split']=='TRAIN' and donor['donor_id'].startswith('YCB:'),foreign_in_front_paste_composite_or_source_order=pixel['same_transformed_RGB_alpha_pasted_at_both_locations'] and pixel['outside_paste_fullRGB_exact'],same_transformed_foreign_RGB_alpha_cover_and_sham=pixel['same_transformed_RGB_alpha_pasted_at_both_locations'],equal_positive_alpha_pasted_mass=pixel['equal_positive_pasted_alpha_mass'],background_outside_paste_exact=pixel['outside_paste_fullRGB_exact'],original_crop_cover_RGB_bytes_changed=bool(np.any(before['rgb96']!=cover['rgb96'])),original_crop_sham_has_donor_RGB_support=pixel['SHAM_donor_positive_original96_RGB_support'],end_related_original96_COVER_bytes_changed=pixel['end_support_actual_original96_bytes_changed_COVER'],end_related_original96_SHAM_bytes_unchanged=pixel['end_support_actual_original96_bytes_unchanged_SHAM'],certified_surrounding_material_RGB_unchanged_both=pixel['chosen_surrounding_samepart_RGB_unchanged_in_both'])

def invented():
    k=pure();old=k['invented']();tests={a:bool(b) for a,b in old.items()}
    p=dict(sham_translation_xy=[10,0]);yes=dict(a=True,b=True);no=dict(a=False,b=True)
    tests['old_geometry_ready_no_opposite_attempt']=policy_for(p,yes,yes)==('ORIGINAL_RETAINED',10,False)
    tests['old_failed_one_reflection_only']=policy_for(p,no,yes)==('OPPOSITE_ONE_ATTEMPT_READY',-10,True)
    tests['opposite_failure_has_no_search']=policy_for(p,no,no)==('OPPOSITE_ONE_ATTEMPT_FAILED_RETAINED',-10,True)
    original=dict(original_query_available=False,feature_constructed=True,native_slot=0)
    tests['original_U_does_not_enter_source_asset_path']=semantic_locality({},original,k)[0] is False
    a=np.arange(24,dtype=np.uint8).reshape(2,4,3);b=a.copy();b[0,0,0]^=1
    tests['old_decoded_array_byte_drift_detected']=not k['eq'](a,b) and not k['eq'](a,a.astype(np.float32))
    need(all(tests.values()),'Invented independent audit fixtures');return dict(completed=True,PASS=True,invented_only=True,tests=tests,count=len(tests),source_binary_JSONL_header_alpha_pixel_model_eval_reads=0)

def prepare():
    final=read(C/'FINAL_READY.json');p=bound_json(final['protocol']);need(final['code']==bind(C/'build_real_end_opposite_sham_pairs.py') and p['root_scope']==bind(SCOPE),'Exact frozen producer contract')
    inherited=bound_json(p['inherited_asset_bindings']);need(inherited['metadata_only'] and len(inherited['actual_binary_whitelist'])==94,'Text-only inherited79+old15 binary whitelist schema')
    write(CHECKS,invented())
    contract=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),producer_final_ready=bind(C/'FINAL_READY.json'),producer_code=final['code'],producer_protocol=final['protocol'],inherited_asset_bindings=p['inherited_asset_bindings'],prior_independent_numeric_code=bind(OLD_KERNEL),checks=bind(CHECKS),planned_producer_complete=str(GEN/'COMPLETE.json'),planned_output=str(OUT),actual_binding_fields_required=['producer_complete','producer_entry','actual_root_source_scope','independent_static_review'],source_BINARY_or_actual_output_JSONL_header_hash_reads_in_preparation=0,coverage='All66 identities/all8 source qL/U, original-geometry-first/one-opposite policy, source/donor transform and alpha, all emitted96/context arrays, selected-end changed/unchanged samples, closed14 proof, old5 all3 every decoded array exact, source bytes unchanged. No new semantic observation or model claim.')
    write(PROTOCOL,contract);template=H/'ROOT_ACTUAL_SAVED_SOURCE_AUDIT_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_actual_saved_source_audit=False,source_only=True,no_model_fit_eval_or_new_labels=True,automatic_retry=False,code=contract['code'],protocol=bind(PROTOCOL),producer_complete={'path':str(GEN/'COMPLETE.json'),'sha256':'ROOT_FILL_AFTER_TERMINAL'},producer_entry={'path':'ROOT_FILL_EXACT_PRODUCER_ENTRY','sha256':'ROOT_FILL'},actual_root_source_scope={'path':'ROOT_FILL_EXACT_ACTUAL_SOURCE_SCOPE','sha256':'ROOT_FILL'},independent_static_review={'path':str(V/'independent_source_producer_preflight_a/REVIEW.json'),'sha256':'ROOT_FILL'},output_directory=str(OUT)))
    ready=dict(completed=True,PASS=True,prepared_only=True,code=contract['code'],protocol=bind(PROTOCOL),checks=bind(CHECKS),false_entry=bind(template),actual_source_or_generated_body_hash_header_reads=0,actual_run_requires_root_entry=True);write(H/'FINAL_READY.json',ready);print(json.dumps(ready))

def run(entry):
    e=read(entry);need(e['approved_actual_saved_source_audit'] is True and e['source_only'] is True and e['no_model_fit_eval_or_new_labels'] is True and e['automatic_retry'] is False,'Separate root actual audit entry')
    p=bound_json(e['protocol']);need(e['code']==p['code']==bind(__file__) and e['output_directory']==p['planned_output']==str(OUT),'Frozen auditor/output closure')
    need(bind(OLD_KERNEL)==p['prior_independent_numeric_code'] and bound_json(p['checks'])['PASS'],'Frozen independent numerical code/checks');bound_json(e['actual_root_source_scope'])
    producer_entry=bound_json(e['producer_entry']);gp=bound_json(p['producer_protocol']);done=bound_json(e['producer_complete']);need(e['producer_complete']['path']==p['planned_producer_complete'] and done['completed'] and done['PASS'] and done['entry']==e['producer_entry'],'Terminal producer closure')
    review=bound_json(e['independent_static_review']);need(review['completed'] and review['PASS'] and review['code']==p['producer_code'] and review['protocol']==p['producer_protocol'],'Exact preflight code/protocol')
    need(producer_entry['code']==p['producer_code'] and producer_entry['protocol']==p['producer_protocol'],'Producer runtime matches reviewed source')
    inherited=bound_json(p['inherited_asset_bindings']);need(producer_entry['exact_binary_asset_whitelist']==inherited['actual_binary_whitelist'],'Root exact source/old5 whitelist')
    whitelist={(b['path'],b['sha256']) for b in inherited['actual_binary_whitelist']};opened_sources={};k=pure()
    def array(b,source=False,keys=None):
        safe(b['path'])
        if source:need((b['path'],b['sha256']) in whitelist,'Only inherited exact source/old5 assets before binary access');opened_sources[b['path']]=b['sha256']
        else:need(Path(b['path']).is_relative_to(GEN),'Only generated packets in exact output')
        with opened(b['path'])as f:
            initial=os.fstat(f.fileno());hh=hashlib.sha256()
            for z in iter(lambda:f.read(1<<20),b''):hh.update(z)
            need(hh.hexdigest()==b['sha256'],'Same-descriptor array SHA');f.seek(0);z=np.load(f,allow_pickle=False)
            if isinstance(z,np.lib.npyio.NpzFile):
                with z:a={name:z[name].copy() for name in (keys or z.files)}
            else:need(keys is None,'NPY schema');a=z.copy()
            after=os.fstat(f.fileno());need((initial.st_ino,initial.st_size,initial.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Asset stable during read')
        return a
    sourceplan=bound_json(gp['source_plan']);frames={f['frame_index']:f for f in sourceplan['frames']};attempts=sourceplan['attempts'];oldrows=bound_lines(gp['predecessor_manifest']);old={r['pair_id']:r for r in oldrows};obs=bound_lines(producer_entry['direct_end_observations']);em={r['case_id']:r['endpoint_observation'] for r in obs}
    donor_proof=bound_json(gp['donor_proof_complete']);donor_rows=bound_lines(gp['donor_records']);dm={r['donor_id']:r for r in donor_rows}
    need(donor_proof['completed'] and donor_proof['PASS'] and donor_proof['records']==gp['donor_records'] and len(dm)==59 and all(r['PASS'] and r['split']=='TRAIN' for r in donor_rows),'Original59 TRAIN donor proof authority, no imputation')
    need(len(attempts)==66 and len(frames)==21 and len(em)==22 and len(old)==66,'Original full source population')
    rows=bound_lines(done['manifest']);need([r['pair_id'] for r in rows]==[r['pair_id'] for r in attempts],'All66 original order');need(not OUT.exists(),'Fresh exclusive audit');OUT.mkdir(parents=True)
    source={};donors={};aud=[];statuses=[];packet_count=0;slotU=0;old5=0
    for origin,r in zip(attempts,rows):
        pid=r['pair_id'];fr=frames[origin['frame_index']];slot=origin['native_slot'];ep=em[r['case_id']];prior=old[pid];record=dict(pair_id=pid,case_id=r['case_id'],native_slot=slot,stored_status=r['status'],checks={})
        def check(name,value):need(value,pid+': '+name);record['checks'][name]=True
        check('perattempt_proof_equals_manifest',read(GEN/pid/'PROOF.json')==r)
        check('original_identity_donor_and_weak_endpoint_fixed',r['case_id']==origin['case_id'] and r['donor_id']==origin['donor_id'] and r['old_donor_lineage']==origin['donor'] and r['endpoint_observation']==ep and r['original_input']==prior['original_input'] and ep==prior['endpoint_observation'])
        o=r['original_input'];check('source_qL_native_originalflags_exact',o['native_slot']==slot and o['query_xy']==fr['query_xy'][slot] and o['L']==fr['L'][slot] and o['original_row_id']==fr['query_id'][slot] and o['original_global_index']==fr['global_index'][slot] and o['original_query_available']==fr['native_continuous_valid'][slot] and o['original_flags']==fr['original_query_input_metadata'][slot])
        check('source_only_newtargets_unknown_no_admission',r['training_admission'] is False and r['new_native_quality_owner_all8']==-1 and fr['split']=='SOURCE_DEVELOPMENT')
        available={name:b for name,b in r['variants'].items() if 'path' in b}
        for name in NAMES:statuses.append(dict(pair_id=pid,variant=name,constructed_packet=name in available,native_rows=8,original_input_U=sum(not x for x in fr['native_continuous_valid']),weak_state=r['direct_end_weak_states']['SHAM' if name=='ADJACENT_SHAM' else name]))
        eligible,points,support,by=semantic_locality(ep,o,k)
        if not available:
            check('no_unsupported_semantic_input_promotion',not r['PASS'] and not r['constructed'] and all(v=='UNKNOWN' for v in r['direct_end_weak_states'].values()) and (not eligible or r['status']=='ATTEMPT_FAILURE_UNKNOWN_RETAINED'))
            check('no_policy_or_binary_for_original_UNKNOWN',not eligible and 'opposite_sham_policy' not in r or r['status']=='ATTEMPT_FAILURE_UNKNOWN_RETAINED');aud.append(record);continue
        check('frozen_sampled_local_end_opportunity',eligible)
        fi=fr['frame_index']
        if fi not in source:
            rgb=array(fr['canonical_RGB_NPY'],True);facts=array(fr['existing_factual96_packet'],True,('rgb96','valid96','crop_constructed','global_index'));need(rgb.dtype==np.uint8 and list(rgb.shape)==fr['raw_hw']+[3] and k['ash'](rgb)==fr['decoded_RGB_sha256'],'Canonical RGB definition');source[fi]=(rgb,facts)
        rgb,facts=source[fi];pack={}
        for name,b in available.items():
            need(Path(b['path']).is_relative_to(GEN/pid),'Packet remains its source attempt');a=array(b);pack[name]=a;packet_count+=1;slotU+=int((~a['query_available']).sum())
            expected=dict(query_xy=np.array(fr['query_xy'],np.float64),L=np.array(fr['L'],np.float64),native_slot=np.arange(8,dtype=np.int8),query_id=np.array(fr['query_id']),original_global_index=np.array(fr['global_index'],np.int32),query_available=np.array(fr['native_continuous_valid'],bool),native_continuous_valid=np.array(fr['native_continuous_valid'],bool),nearest_valid=np.array(fr['nearest_valid'],bool),crop_constructed=np.array(fr['feature_constructed'],bool))
            for flag in ('no_detection','finite_non_sentinel','query_RGB96_centre_domain_valid','native_grid_any_valid'):expected['original_'+flag]=np.array([z[flag] for z in fr['original_query_input_metadata']],bool)
            check(name+'_all8_F64_qL_ID_native_availability_flags_exact',all(k['eq'](a[key],value) for key,value in expected.items()))
            check(name+'_all8_native_quality_owner_minus1',all(a[t].dtype==np.int8 and a[t].shape==(8,) and (a[t]==-1).all() for t in ('native_visibility_target','point_quality_target','owner_target')))
            raw=np.zeros((8,96,96,3),np.uint8);valid=np.zeros((8,96,96),bool)
            for n in range(8):
                if fr['feature_constructed'][n]:raw[n],valid[n]=k['local_sample'](a['full_RGB'],fr['query_xy'][n],fr['L'][n])
            check(name+'_all8_independent_F64_RGB96_valid_replay',k['eq'](raw,a['rgb96']) and k['eq'](valid,a['valid96']))
            check(name+'_central32_slice',k['eq'](a['central32'],a['rgb96'][:,32:64,32:64]))
            ctx,cv=k['context_sample'](a['full_RGB'],fr['context_box_xyxy']);fra=np.array([[np.float32(cv.mean()),np.float32(valid[n].mean()),np.float32(valid[n,32:64,32:64].mean())] for n in range(8)],np.float32)
            check(name+'_independent_context224_valid_fractions',k['eq'](ctx,a['target224']) and k['eq'](cv,a['target_valid']) and k['eq'](fra,a['fractions']))
            check(name+'_center4_context_flags',k['eq'](a['central4_available'],valid[:,47:49,47:49].all((1,2))) and k['eq'](a['context_available'],np.full(8,cv.any(),bool)))
        before=pack['BEFORE'];check('original_fullRGB_and_factual8_before_exact',k['eq'](before['full_RGB'],rgb) and k['eq'](before['rgb96'],facts['rgb96']) and k['eq'](before['valid96'],facts['valid96']) and k['eq'](before['crop_constructed'],facts['crop_constructed']) and np.array_equal(facts['global_index'],fr['global_index']) and not before['pasted_alpha'].any())
        check('original_RGB96_valid_context_fact_hashes',all(k['ash'](before['rgb96'][n])==fr['facts'][n]['RGB96_HWC_sha256'] and k['ash'](before['valid96'][n])==fr['facts'][n]['valid96_sha256'] for n in range(8)) and k['ash'](before['target224'])==fr['facts'][0]['target224_F32_HWC_sha256'])
        did=r['donor_id']
        if did not in donors:donors[did]=array(origin['donor']['compact_RGBA_cutout'],True,('rgb','alpha'))
        donor=donors[did];check('assigned_original_TRAIN_donor_pixels',origin['donor']['split']=='TRAIN' and origin['donor']['compact_RGBA_cutout']==dm[did]['compact_RGBA_cutout'] and origin['donor']['array_proof']==dm[did]['array_proof'] and k['ash'](donor['rgb'])==origin['donor']['array_proof']['rgb']['array_sha256'] and k['ash'](donor['alpha'])==origin['donor']['array_proof']['alpha']['array_sha256'])
        xs,ys=zip(*points);box=[min(xs),min(ys),max(xs)+1,max(ys)+1];radius=float(distance_transform_edt(np.pad(donor['alpha'],1))[1:-1,1:-1].max());required=.5*math.hypot(box[2]-box[0]-1,box[3]-box[1]-1)+4
        need(radius>1,'Original opaque interior, no fallback');size=max(1,math.ceil(max(donor['alpha'].shape)*required/(radius-1)));dr,da=k['resized_donor'](donor['rgb'],donor['alpha'],size);td=array(r['transformed_donor'])
        check('single_original_EDT_resize_and_eroded_alpha',k['eq'](dr,td['rgb']) and k['eq'](da,td['alpha']) and size==r['scale_geometry']['longside_pixels'] and r['scale_geometry']==prior['scale_geometry'])
        dd=distance_transform_edt(np.pad(da,1))[1:-1,1:-1];ay,ax=np.unravel_index(int(dd.argmax()),dd.shape);center=[(box[0]+box[2]-1)/2,(box[1]+box[3]-1)/2];co=np.rint(np.array(center)-[ax,ay]).astype(int);ch,cw=da.shape;direction=1 if center[0]<=(fr['raw_hw'][1]-1)/2 else -1;dx=int(box[2]+4-co[0]) if direction==1 else int(box[0]-4-(co[0]+cw));cb=[int(co[0]),int(co[1]),int(co[0]+cw),int(co[1]+ch)]
        oldchecks=geometry(k,o,ep,cb,dx,points,support);oppchecks=geometry(k,o,ep,cb,-dx,points,support);selection,chosen_dx,second=policy_for(prior['placement_geometry'],oldchecks,oppchecks);so=co+[chosen_dx,0];pg=r['placement_geometry'];policy=r['opposite_sham_policy'];gc=oppchecks if second else oldchecks
        check('original_V189_COVER_origin_scale_proposal_exact',prior['placement_geometry']['cover_origin_xy']==co.tolist() and prior['placement_geometry']['cover_bbox_xyxy']==cb and prior['placement_geometry']['sham_translation_xy']==[dx,0])
        check('original_first_only_failed_one_opposite_policy',policy['selection']==selection and policy['opposite_attempted']==second and pg['sham_translation_xy']==[chosen_dx,0] and pg['cover_origin_xy']==co.tolist() and pg['sham_origin_xy']==so.tolist() and pg['cover_bbox_xyxy']==cb and pg['transformed_hw']==[ch,cw])
        check('independent_geometry_controls_equal',gc==r['finite_end_geometry']['geometry_controls'] and oldchecks==policy['original_geometry']['geometry_controls'] and (not second or oppchecks==policy['opposite_geometry']['geometry_controls']))
        record.update(original_geometry_checks=oldchecks,selected_geometry_checks=gc,selected_policy=selection,opposite_attempted=second)
        if not r['constructed']:
            check('failed_geometry_BEFORE_only_unknown',not all(gc.values()) and set(pack)=={'BEFORE'} and not r['PASS'] and all(v=='UNKNOWN' for v in r['direct_end_weak_states'].values()));aud.append(record);continue
        check('constructed_only_after_valid_geometry_all3',all(gc.values()) and set(pack)==set(NAMES));cover=pack['COVER'];sham=pack['ADJACENT_SHAM'];ca=k['raw_mask'](fr['raw_hw'],da,co);sa=k['raw_mask'](fr['raw_hw'],da,so)
        pixel=dict(end_every_candidate_raw_tap_COVER_alpha1=all(ca[y,x] for x,y in points),end_every_candidate_raw_tap_SHAM_alpha0=all(not sa[y,x] for x,y in points),equal_positive_pasted_alpha_mass=int(ca.sum())==int(sa.sum())>0,outside_paste_fullRGB_exact=k['eq'](cover['full_RGB'][~ca],rgb[~ca]) and k['eq'](sham['full_RGB'][~sa],rgb[~sa]),same_transformed_RGB_alpha_pasted_at_both_locations=True)
        for a,mask,originxy in ((cover,ca,co),(sham,sa,so)):
            pixel['same_transformed_RGB_alpha_pasted_at_both_locations'] &= k['eq'](a['pasted_alpha'],mask) and k['eq'](a['full_RGB'][originxy[1]:originxy[1]+ch,originxy[0]:originxy[0]+cw][da],dr[da])
        sample=np.array([bool(z&points) for z in by],bool).reshape(96,96)
        pixel['end_support_actual_original96_bytes_changed_COVER']=bool(sample.any() and np.any(cover['rgb96'][slot][sample]!=before['rgb96'][slot][sample]));pixel['end_support_actual_original96_bytes_unchanged_SHAM']=k['eq'](sham['rgb96'][slot][sample],before['rgb96'][slot][sample]);pixel['SHAM_donor_positive_original96_RGB_support']=any(sa[y,x] for x,y in support)
        ww=[w['raw_pixel_xy'] for w in ep['surrounding_material_witnesses'] if w.get('material_observed') is True and w.get('same_selected_part_supported') is True and w.get('evidence_binding')]
        pixel['chosen_surrounding_samepart_RGB_unchanged_in_both']=bool(ww) and all(not ca[y,x] and not sa[y,x] and k['eq'](rgb[y,x],cover['full_RGB'][y,x]) and k['eq'](rgb[y,x],sham['full_RGB'][y,x]) for x,y in ww)
        for key,value in pixel.items():check('independent_pixel_'+key,bool(value)==r['pixel_checks'][key])
        check('stored_all8_invariant_proofs_independently_true',all(r['pixel_checks'][name] is True for name in ('all8_input_q_L_flags_validity_and_ID_unchanged','all8_original_initializer_flags_unchanged','all8_new_native_quality_owner_unknown','all8_central32_exact_RGB96_slice')))
        detail=r['pixel_detail'];check('stored_end96_sample_and_alpha_counts_exact',int(sample.sum())==detail['end_related_RGB96_sample_count'] and int(ca.sum())==detail['cover_positive_alpha_pixels'] and int(sa.sum())==detail['sham_positive_alpha_pixels'])
        p14=proof14(pixel,True,before,cover,dm[did]);check('closed14_all_bool_mapping_exact',len(p14)==14 and all(bool(value)==r['closed14_actual_proof'][key] for key,value in p14.items()) and bool(r['closed14_actual_proof']['evidence_bindings']))
        check('closed_contract_no_native_quality_or_training_admission',r['closed_end_contract']['new_native_visibility_target']==r['closed_end_contract']['new_quality_target']==-1 and r['closed_end_contract']['training_admission'] is False)
        expected_PASS=all(pixel.values()) and all(p14.values());check('weak_PASS_only_actual14proof',r['PASS']==expected_PASS and (r['closed_end_contract']['status']=='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED')==expected_PASS)
        check('weak_D_E_D_or_unknown_no_native_gold',r['direct_end_weak_states']==({'BEFORE':'VISIBLE_WEAK','COVER':'EXTERNALLY_BLOCKED_WEAK','SHAM':'VISIBLE_WEAK'} if expected_PASS else {'BEFORE':'UNKNOWN','COVER':'UNKNOWN','SHAM':'UNKNOWN'}))
        if prior['PASS']:
            check('old5_no_opposite_and_still_supported',not second and expected_PASS)
            for name in NAMES:
                saved=array(prior['variants'][name],True);check('old5_'+name+'_every_decoded_array_exact',set(saved)==set(pack[name]) and all(k['eq'](saved[key],pack[name][key]) for key in saved))
            old5+=1
        for name,binding in r['clean_selected96'].items():
            need(Path(binding['path']).is_relative_to(GEN/pid),'Generated clean PNG path');checked({key:binding[key] for key in ('path','sha256')})
            with opened(binding['path'])as f:
                with Image.open(f)as im:png=np.array(im)
            check(name+'_saved_clean96_PNG_exact',k['eq'](png,pack[name]['rgb96'][slot]) and k['ash'](png)==binding['decoded_array_sha256'])
        record['independent_pixel_checks']={key:bool(value) for key,value in pixel.items()};record['independent_closed14']=p14;aud.append(record)
        print(json.dumps(dict(audited=len(aud),total=66,pair_id=pid,constructed=True)),flush=True)
    need(len(aud)==66 and len(statuses)==198 and old5==5,'Full66/all198/mandatoryold5')
    for path,digest in opened_sources.items():need(sha(path)==digest,'Original source/donor/old5 bytes unchanged after audit')
    count=dict(attempts66=len(aud),cases22=len({r['case_id'] for r in rows}),attempted_variant_statuses198=len(statuses),attempted_allnative_rows1584=8*len(statuses),actual_variant_packets=packet_count,actual_allnative_input_rows=8*packet_count,actual_original_inputU_rows=slotU,constructed_pairs=sum(r['constructed'] for r in rows),supported_weak_pairs=sum(r['PASS'] for r in rows),original_supported5_all3_allarrays_exact=old5,actually_opened_source_frames=len(source),actually_opened_original_assigned_donors=len(donors),assigned_donor_whitelist_unique=len({r['donor_id'] for r in attempts}))
    need(count['constructed_pairs']==done['constructed_pairs'] and count['supported_weak_pairs']==done['supported_weak_end_pairs'],'Completion counts independently agree')
    result=dict(completed=True,PASS=True,source_only=True,code=p['code'],protocol=bind(PROTOCOL),audit_entry=bind(entry),actual_source_complete=e['producer_complete'],producer_entry=e['producer_entry'],counts=count,stored_status_counts=dict(Counter(r['status'] for r in rows)),policy_counts=dict(Counter(r.get('selected_policy','NO_POLICY_UNKNOWN') for r in aud)),independent_all66_checks=aud,all198_variant_statuses=statuses,all_original_opened_assets_unchanged=[dict(path=path,sha256=digest) for path,digest in opened_sources.items()],new_native_quality_owner_targets_all_minus1=True,training_admission=False,new_labels_or_models_or_forward_fit_evaluation=0,limits=['All66 metadata; every actually emitted packet and all8 inputs independently replayed. Alpha/RGB/closed14 checks apply to constructed triplets, not missing variants.','Original source inputU and weak UNKNOWN/failures retained. Pixel proof is a source2D composite fact, not natural-occlusion/native gold, classifier improvement or independent holdout.','Source selected-instance/material-end semantics remain frozen assistant weak observations; this audit does not create or replace them.','Source/context96/224 samplers reuse prior independent audit numerical kernels by exact source binding, never generator/helper imports.'],fully_solved=False,selected_model=None)
    write(OUT/'REVIEW.json',result);print(json.dumps(dict(completed=True,PASS=True,review=bind(OUT/'REVIEW.json'),counts=count,status_counts=result['stored_status_counts'])))

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare:prepare()
    else:
        try:run(a.entry)
        except BaseException:
            dest=H/'ACTUAL_AUDIT_FAILURE_NO_AUTOMATIC_RETRY.json'
            if not dest.exists():write(dest,dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False))
            raise
