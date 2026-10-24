"""V193 fixed22/66 opposite-SHAM source-real end-only pair compositor. Root actual entry required.

Preparation reads explicit code/JSON only and invented arrays. Actual mode reads
only original V164-bound TRAIN RGB/crops and the same proved YCB cutouts. No GT,
model, feature, evaluator, coordinate fitting or donor replacement is present.
"""
from pathlib import Path
import argparse, ast, copy, hashlib, html, importlib.util, json, math, os, stat, sys, time, traceback
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import scipy
from scipy.ndimage import distance_transform_edt

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
OLD=P/'query_visibility_native_part_counterfactual_20261024_v164'
SOURCE_PLAN=OLD/'generator_preparation/FIXED22_FRAME_INPUTS_AND_66_ATTEMPTS_BEFORE_PIXELS.json'
DONOR_COMPLETE=P/'query_visibility_public_train_owner_counterfactual_20261024_v161/donor_proof_v1/DONOR_PROOF_COMPLETE.json'
HELPER=P/'query_visibility_public_train_owner_counterfactual_20261024_v161/pixel_builder_preparation/build_coherent_source_pixels.py'
LOCAL=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
TARGET=P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
GEOMETRY=P/'query_visibility_sampled_domain_direct_end_20261024_v189/end_locality_contract_b/finite_end_locality_contract.py'
PROTOCOL=H/'PROTOCOL_BEFORE_REAL_END_PAIR_SOURCE_PIXELS.json'
OUT=V/'real_end_opposite_sham_pairs_v1'
VARIANTS=('BEFORE','COVER','ADJACENT_SHAM')
ACTUAL_ASSET_WHITELIST=None


def require(value,message):
    if not bool(value):raise AssertionError(message)


def safe_path(value):
    p=Path(value)
    require(p.is_absolute() and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Safe explicit path before filesystem access')
    return p


def opened(value):
    p=safe_path(value);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        final=os.open(p.parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    require(stat.S_ISREG(os.fstat(final).st_mode),'Regular file only')
    return os.fdopen(final,'rb')


def sha(path):
    h=hashlib.sha256()
    with opened(path) as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def bind(path):return {'path':str(path),'sha256':sha(path)}
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def checked_bytes(binding):
    with opened(binding['path']) as f:b=f.read()
    require(hashlib.sha256(b).hexdigest()==binding['sha256'],'Bound file drift')
    return b


def json_file(path):
    with opened(path) as f:return json.load(f)


def checked_json(binding):return json.loads(checked_bytes(binding))


def write_new(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def load_bound(binding,keys=None):
    """Same nofollow file descriptor hashed then rewound; actual-entry use only."""
    require(ACTUAL_ASSET_WHITELIST is not None and (binding['path'],binding['sha256']) in ACTUAL_ASSET_WHITELIST,'Root exact source/old5 binary whitelist required before asset access')
    with opened(binding['path']) as f:
        initial=os.fstat(f.fileno());h=hashlib.sha256()
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
        require(h.hexdigest()==binding['sha256'],'Original asset SHA drift')
        f.seek(0);value=np.load(f,allow_pickle=False)
        if isinstance(value,np.lib.npyio.NpzFile):
            with value as z:
                require(keys is not None and set(keys)<=set(z.files),'Exact required array keys')
                result={k:z[k].copy() for k in keys}
        else:
            require(keys is None,'NPY contract');result=value.copy()
        after=os.fstat(f.fileno())
        require((initial.st_size,initial.st_mtime_ns,initial.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Asset drift during read')
    return result


def load_functions(path,names,namespace):
    tree=ast.parse(checked_bytes(bind(path)).decode())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==set(names),'Pinned pure AST functions')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::PURE_ONLY','exec'),namespace)
    return namespace


def operators(protocol):
    for k in ('original_pixel_helper','original_RGB_sampler','original_context_sampler','frozen_geometry'):
        checked_bytes(protocol[k])
    ns=dict(np=np,Image=Image,math=math,distance_transform_edt=distance_transform_edt,exact=exact,ash=ash)
    load_functions(HELPER,('erode1','transformed_cutout','opaque_anchor','paste','materialize','verify_before'),ns)
    load_functions(LOCAL,('crop',),ns)
    load_functions(TARGET,('valid_box','target_context_grid','sample_target_context'),ns)
    spec=importlib.util.spec_from_file_location('frozen_V189_sampled_end_geometry',GEOMETRY)
    geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
    ns['geometry']=geo
    return ns


def materialize(rgb,frame,op):
    a=op['materialize'](rgb,frame,op)
    a['query_available']=np.array(frame['native_continuous_valid'],bool)
    a['owner_target']=np.full(8,-1,np.int8)
    a['original_global_index']=np.array(frame['global_index'],np.int32)
    return a


def original_record(frame,k):
    return dict(original_global_index=frame['global_index'][k],original_row_id=frame['query_id'][k],
        development_role=frame['split'],native_slot=k,query_xy=frame['query_xy'][k],L=frame['L'][k],
        raw_hw=frame['raw_hw'],original_query_available=frame['native_continuous_valid'][k],
        feature_constructed=frame['feature_constructed'][k],original_flags=frame['original_query_input_metadata'][k])


def endpoint_preflight(original,endpoint,geo):
    # Empty/nonlocalized observations remain unknown without opening any asset.
    fields=('selected_instance_supported','intended_part_supported','queried_end_localization_supported',
            'all_plausible_candidates_enclosed','before_end_visible_supported')
    if not original['original_query_available']:return None,'ORIGINAL_INPUT_U'
    if not original['feature_constructed']:return None,'ORIGINAL_CROP_NOT_CONSTRUCTED'
    if endpoint.get('authority')!=geo.REAL_AUTHORITY or endpoint.get('native_slot')!=original['native_slot'] or not all(endpoint.get(k) is True for k in fields) or not endpoint.get('evidence_binding') or not endpoint.get('end_locality_id'):
        return None,'DIRECT_END_WEAK_OBSERVATION_UNKNOWN'
    if not geo.finite_pair(original['query_xy']) or not isinstance(original['L'],(int,float)) or not math.isfinite(original['L']) or original['L']<=0:return None,'ORIGINAL_Q_L_MISSING'
    pixels,outside=geo.candidate_footprint(endpoint['candidate_enclosures_xyxy'],original['raw_hw'])
    support,_=geo.exact96_support(original['query_xy'],original['L'],original['raw_hw'])
    sampled=geo.sampled_candidate_relation(original['query_xy'],original['L'],original['raw_hw'],endpoint['candidate_enclosures_xyxy'])
    if outside or not pixels or not all(sampled[k] for k in ('all_enclosures_inside_original96_sample_extent','every_enclosure_has_positive_valid_original96_sample','all_candidate_raw_footprints_in_frame')):
        return None,'END_OUTSIDE_OR_NO_POSITIVE_ORIGINAL96_SAMPLED_OPPORTUNITY_UNKNOWN_NOT_E'
    return pixels,None


def fixed_transform_plan(end_pixels,native_alpha,hw,op):
    """One V164 EDT scale, then new fixed minimal bbox-disjoint SHAM translation.

    The +4px safety margin and all formulas are frozen before pixel reads; no
    scale/position loop, fallback rectangle, donor replacement or retries.
    """
    require(native_alpha.dtype==bool and native_alpha.ndim==2 and native_alpha.any(),'Original donor alpha nonempty')
    xs,ys=zip(*end_pixels);box=[min(xs),min(ys),max(xs)+1,max(ys)+1]
    rw,rh=box[2]-box[0],box[3]-box[1]
    edt=distance_transform_edt(np.pad(native_alpha,1,constant_values=False))[1:-1,1:-1]
    radius=float(edt.max());require(radius>1,'DONOR_NO_OPAQUE_INTERIOR_NO_FALLBACK')
    required=.5*math.hypot(rw-1,rh-1)+4.
    longside=max(1,math.ceil(max(native_alpha.shape)*required/(radius-1.)))
    return longside,dict(end_support_bbox_xyxy=box,native_EDT_radius=radius,required_radius=required,
                        longside_pixels=longside,margin_pixels=4)


def fixed_placements(end_box,alpha,raw_hw,op):
    anchor=op['opaque_anchor'](alpha);require(anchor is not None,'Transformed alpha empty')
    h,w=raw_hw;ch,cw=alpha.shape;x0,y0,x1,y1=end_box
    center=np.array([(x0+x1-1)/2.,(y0+y1-1)/2.],np.float64)
    cover=np.rint(center-anchor).astype(np.int64)
    direction=1 if center[0]<=(w-1)/2 else -1
    # Sham FULL bbox starts 4 pixels beyond end FULL footprint bbox. No search.
    dx=int(x1+4-cover[0]) if direction==1 else int(x0-4-(cover[0]+cw))
    sham=cover+np.array([dx,0],np.int64)
    bbox=[int(cover[0]),int(cover[1]),int(cover[0]+cw),int(cover[1]+ch)]
    return cover,sham,dict(cover_bbox_xyxy=bbox,sham_translation_xy=[dx,0],
        opaque_anchor_xy=anchor.tolist(),cover_origin_xy=cover.tolist(),sham_origin_xy=sham.tolist(),
        transformed_hw=[ch,cw],horizontal_direction=direction)


def select_original_or_one_opposite_sham(original,endpoint,placement,geo):
    """One prospective geometry-only fallback; no pixel, model or label selection."""
    original_placement=copy.deepcopy(placement)
    first=geo.evaluate_plan(original,endpoint,original_placement)
    policy=dict(original_placement=original_placement,original_geometry=first,
                opposite_attempted=False,opposite_geometry=None,selection='ORIGINAL_RETAINED')
    if first['status']=='GEOMETRY_READY_PIXEL_PROOFS_PENDING':
        return original_placement,first,policy
    if first['status']!='GEOMETRY_FAILURE_RETAINED':
        policy['selection']='SEMANTIC_OR_INPUT_UNKNOWN_RETAINED_NO_OPPOSITE'
        return original_placement,first,policy
    dx,dy=original_placement['sham_translation_xy']
    require(type(dx) is int and dx!=0 and dy==0,'Original integer x-only SHAM required')
    opposite=copy.deepcopy(original_placement)
    opposite['sham_translation_xy']=[-dx,0]
    co=opposite['cover_origin_xy'];opposite['sham_origin_xy']=[co[0]-dx,co[1]]
    opposite['horizontal_direction']=-original_placement['horizontal_direction']
    second=geo.evaluate_plan(original,endpoint,opposite)
    policy.update(opposite_attempted=True,opposite_geometry=second,
                  selection='OPPOSITE_ONE_ATTEMPT_READY' if second['status']=='GEOMETRY_READY_PIXEL_PROOFS_PENDING' else 'OPPOSITE_ONE_ATTEMPT_FAILED_RETAINED')
    require(original_placement['cover_bbox_xyxy']==opposite['cover_bbox_xyxy'] and original_placement['cover_origin_xy']==opposite['cover_origin_xy'] and original_placement['transformed_hw']==opposite['transformed_hw'],'COVER/scale unchanged')
    return opposite,second,policy


def contract14_proof(checks,before_parity,before,cover,donor,evidence):
    """Close the frozen B14 proof using actual source parity/paste/byte checks."""
    q=checks
    proof={
      'original_q_L_native_flags_and_all8_inputs_unchanged':all(q[k] for k in ('all8_input_q_L_flags_validity_and_ID_unchanged','all8_original_initializer_flags_unchanged','all8_new_native_quality_owner_unknown','all8_central32_exact_RGB96_slice')),
      'before_original_RGB96_valid96_target_context_exact':bool(before_parity) and all(before_parity.values()),
      'every_candidate_positive_RGB_tap_cover_alpha_one':q['end_every_candidate_raw_tap_COVER_alpha1'],
      'every_candidate_positive_RGB_tap_sham_alpha_zero':q['end_every_candidate_raw_tap_SHAM_alpha0'],
      'distinct_foreign_material_supported':donor['PASS'] is True and donor['split']=='TRAIN' and donor['donor_id'].startswith('YCB:'),
      'foreign_in_front_paste_composite_or_source_order':q['same_transformed_RGB_alpha_pasted_at_both_locations'] and q['outside_paste_fullRGB_exact'],
      'same_transformed_foreign_RGB_alpha_cover_and_sham':q['same_transformed_RGB_alpha_pasted_at_both_locations'],
      'equal_positive_alpha_pasted_mass':q['equal_positive_pasted_alpha_mass'],
      'background_outside_paste_exact':q['outside_paste_fullRGB_exact'],
      'original_crop_cover_RGB_bytes_changed':bool(np.any(cover['rgb96']!=before['rgb96'])),
      'original_crop_sham_has_donor_RGB_support':q['SHAM_donor_positive_original96_RGB_support'],
      'end_related_original96_COVER_bytes_changed':q['end_support_actual_original96_bytes_changed_COVER'],
      'end_related_original96_SHAM_bytes_unchanged':q['end_support_actual_original96_bytes_unchanged_SHAM'],
      'certified_surrounding_material_RGB_unchanged_both':q['chosen_surrounding_samepart_RGB_unchanged_in_both'],
      'evidence_bindings':evidence}
    return proof


def verify_old_supported_decoded_packets(old,arrays,ca,sa):
    require(old['PASS'] is True,'Old supported pair only')
    masks=(np.zeros(arrays[0]['full_RGB'].shape[:2],bool),ca,sa)
    out={}
    for name,a,mask in zip(VARIANTS,arrays,masks):
        current={**a,'pasted_alpha':mask};saved=load_bound(old['variants'][name],tuple(current))
        out[name]={key:exact(current[key],saved[key]) for key in current}
        require(all(out[name].values()),'Original supported5 decoded packet parity '+name)
    return out


def selected_sample_mask(original,end_pixels,geo):
    return np.array(geo.end_related96_mask(original['query_xy'],original['L'],original['raw_hw'],end_pixels),bool)


def prove(before,cover,sham,ca,sa,end_pixels,endpoint,original,cut,alpha,origins,op):
    geo=op['geometry'];k=original['native_slot'];co,so=origins
    checks={}
    invariant=('query_xy','L','native_slot','query_id','valid96','target_valid','crop_constructed',
               'native_continuous_valid','nearest_valid','central4_available','context_available','fractions','query_available')
    checks['all8_input_q_L_flags_validity_and_ID_unchanged']=all(exact(a[f],before[f]) for a in (cover,sham) for f in invariant)
    checks['all8_original_initializer_flags_unchanged']=all(exact(a[f],before[f]) for a in (cover,sham) for f in before if f.startswith('original_'))
    checks['all8_new_native_quality_owner_unknown']=all((a[f]==-1).all() for a in (before,cover,sham) for f in ('native_visibility_target','point_quality_target','owner_target'))
    checks['all8_central32_exact_RGB96_slice']=all(exact(a['central32'],a['rgb96'][:,32:64,32:64]) for a in (before,cover,sham))
    checks['end_every_candidate_raw_tap_COVER_alpha1']=all(bool(ca[y,x]) for x,y in end_pixels)
    checks['end_every_candidate_raw_tap_SHAM_alpha0']=all(not bool(sa[y,x]) for x,y in end_pixels)
    checks['equal_positive_pasted_alpha_mass']=int(ca.sum())==int(sa.sum())>0
    checks['outside_paste_fullRGB_exact']=all(exact(a['full_RGB'][~mask],before['full_RGB'][~mask]) for a,mask in ((cover,ca),(sham,sa)))
    # Explicit full-bbox/no clipping geometry makes this donor-pixel pairing exact.
    checks['same_transformed_RGB_alpha_pasted_at_both_locations']=True
    for a,origin in ((cover,co),(sham,so)):
        x,y=map(int,origin);ch,cw=alpha.shape
        checks['same_transformed_RGB_alpha_pasted_at_both_locations'] &= exact(a['full_RGB'][y:y+ch,x:x+cw][alpha],cut[alpha])
    samplemask=selected_sample_mask(original,end_pixels,geo)
    checks['end_support_actual_original96_bytes_changed_COVER']=bool(samplemask.any() and np.any(cover['rgb96'][k][samplemask]!=before['rgb96'][k][samplemask]))
    checks['end_support_actual_original96_bytes_unchanged_SHAM']=exact(sham['rgb96'][k][samplemask],before['rgb96'][k][samplemask])
    checks['SHAM_donor_positive_original96_RGB_support']=any(bool(sa[y,x]) for (x,y),weight in geo.exact96_support(original['query_xy'],original['L'],original['raw_hw'])[0].items() if weight>0)
    witnesses=[tuple(w['raw_pixel_xy']) for w in endpoint['surrounding_material_witnesses'] if w.get('material_observed') is True and w.get('same_selected_part_supported') is True and w.get('evidence_binding')]
    checks['chosen_surrounding_samepart_RGB_unchanged_in_both']=bool(witnesses) and all(not ca[y,x] and not sa[y,x] and exact(before['full_RGB'][y,x],cover['full_RGB'][y,x]) and exact(before['full_RGB'][y,x],sham['full_RGB'][y,x]) for x,y in witnesses)
    detail=dict(end_positive_raw_support_xy=sorted(map(list,end_pixels),key=lambda z:(z[1],z[0])),
        end_related_RGB96_sample_count=int(samplemask.sum()),cover_positive_alpha_pixels=int(ca.sum()),sham_positive_alpha_pixels=int(sa.sum()),
        surrounding_raw_witnesses_xy=list(map(list,witnesses)),
        all8_RGB96_array_sha256={name:[ash(a['rgb96'][n]) for n in range(8)] for name,a in zip(VARIANTS,(before,cover,sham))},
        context224_array_sha256={name:ash(a['target224']) for name,a in zip(VARIANTS,(before,cover,sham))})
    return {k:bool(v) for k,v in checks.items()},detail


def save_clean_png(path,array):
    require(array.dtype==np.uint8,'Clean uint8 PNG')
    Image.fromarray(array).save(path)
    with Image.open(path) as im:decoded=np.array(im)
    require(exact(decoded,array),'Saved clean PNG byte parity')
    return {**bind(path),'decoded_array_sha256':ash(decoded)}


def contact(folder,case_id,k,arrays,endpoint,proof):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    canvas=Image.new('RGB',(1140,660),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,8),case_id+' native'+str(k)+' / fixed original q,L / source-real semantic-end weak control',font=font,fill='black')
    draw.text((10,29),'2D opaque foreground composite. No physical-depth/native-gold claim. UNKNOWN/failed proofs retained.',font=font,fill='black')
    for col,(name,a) in enumerate(zip(VARIANTS,arrays)):
        x=10+col*380;draw.text((x,54),name,font=font,fill='black')
        full=Image.fromarray(a['full_RGB']);full.thumbnail((365,250))
        # Mark only the display thumbnail; source/fullRGB/clean96 are untouched.
        marks=ImageDraw.Draw(full);sx=full.width/a['full_RGB'].shape[1];sy=full.height/a['full_RGB'].shape[0]
        for box in endpoint['candidate_enclosures_xyxy']:
            marks.rectangle((box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy),outline='red',width=2)
        for witness in endpoint['surrounding_material_witnesses']:
            px,py=witness['raw_pixel_xy'];px*=sx;py*=sy;marks.ellipse((px-2,py-2,px+2,py+2),fill='yellow')
        q=a['query_xy'][k];qx,qy=float(q[0])*sx,float(q[1])*sy;marks.ellipse((qx-2,qy-2,qx+2,qy+2),outline='cyan',width=1)
        canvas.paste(full,(x,78))
        clean=a['rgb96'][k];canvas.paste(Image.fromarray(clean).resize((192,192),Image.Resampling.NEAREST),(x,340))
        canvas.paste(Image.fromarray(a['central32'][k]).resize((128,128),Image.Resampling.NEAREST),(x+210,370))
        draw.text((x,540),'clean RGB96 (2x) / exact central32 (4x)',font=font,fill='black')
    draw.text((10,572),'Display: RED=end candidates ('+str(len(endpoint['candidate_enclosures_xyxy']))+'); YELLOW=same-part witness; CYAN=original q. Exact coordinates: PROOF.json.',font=font,fill='black')
    draw.text((10,594),'Pixel proof: '+('PASS' if all(proof.values()) else 'FAILED; weak end state remains UNKNOWN'),font=font,fill='black')
    draw.text((10,616),'Other same-part material may stay visible during end COVER. Whole crop presence is not native D.',font=font,fill='black')
    path=folder/'CONTACT.png';canvas.save(path);return bind(path)


def run(entry_path):
    global ACTUAL_ASSET_WHITELIST
    e=json_file(entry_path);require(e['approved_actual_source_real_end_compositing'] is True and e['source_only'] is True,'Root actual source-pixel entry')
    for key in ('training_allowed','model_allowed','encoder_allowed','evaluation_allowed','PnP_allowed','automatic_retry'):require(e[key] is False,'Disallowed operation '+key)
    require(e['code']==bind(Path(__file__).absolute()) and e['protocol']['path']==str(PROTOCOL),'Exact frozen runner')
    p=checked_json(e['protocol']);require(p['code']==e['code'],'Protocol/code closure');checked_json(p['root_scope'])
    checked_json(e['root_scope']);require(e['output_directory']==str(OUT),'Fresh fixed output')
    require(sys.executable==p['runtime']['executable'] and np.__version__==p['runtime']['numpy'] and PIL.__version__==p['runtime']['Pillow'] and scipy.__version__==p['runtime']['scipy'],'Pinned exact CPU runtime')
    require(e['independent_static_review']['path'] and checked_json(e['independent_static_review'])['PASS'] is True,'Independent preflight PASS before source assets')
    inherited=checked_json(p['inherited_asset_bindings']);require(e['exact_binary_asset_whitelist']==inherited['actual_binary_whitelist'],'Exact inherited sources and old5-only parity packet whitelist')
    ACTUAL_ASSET_WHITELIST={(b['path'],b['sha256']) for b in e['exact_binary_asset_whitelist']}
    prior_complete=checked_json(p['predecessor_complete']);require(prior_complete['completed'] and prior_complete['PASS'] and prior_complete['supported_weak_end_pairs']==5 and prior_complete['manifest']==p['predecessor_manifest'],'Immutable predecessor all66/supported5 closure')
    prior_rows=[json.loads(line) for line in checked_bytes(p['predecessor_manifest']).splitlines()];prior={r['pair_id']:r for r in prior_rows};require(len(prior)==66 and sum(r['PASS'] for r in prior_rows)==5,'Original66/5 source metadata')
    inputs=checked_json(p['source_plan']);frames={r['frame_index']:r for r in inputs['frames']};attempts=inputs['attempts']
    require(len(attempts)==66 and len({r['case_id'] for r in attempts})==22 and len(frames)==21,'Original fixed22/66/21 population')
    require(all(r['split']=='SOURCE_DEVELOPMENT' for r in frames.values()),'Every fixed source frame TRAIN role')
    require(len({a['pair_id'] for a in attempts})==66 and all(Path(a['pair_id']).name==a['pair_id'] and a['pair_id'] and '..' not in a['pair_id'] and 'v112' not in a['pair_id'].casefold() for a in attempts),'Safe unique fixed attempt names before destination access')
    donor_complete=checked_json(p['donor_proof_complete']);require(donor_complete['completed'] and donor_complete['PASS'] and donor_complete['attempted']==donor_complete['passed']==59,'Original all59 proved TRAIN donor authority')
    require(donor_complete['records']==p['donor_records'],'Exact original donor records binding')
    donors=[json.loads(s) for s in checked_bytes(p['donor_records']).splitlines()];dm={r['donor_id']:r for r in donors}
    require(len(dm)==59 and all(r['PASS'] and r['split']=='TRAIN' for r in donors),'All59 TRAIN donors')
    for a in attempts:
        require(a['donor']['split']=='TRAIN' and a['donor']['compact_RGBA_cutout']==dm[a['donor_id']]['compact_RGBA_cutout'] and a['donor']['array_proof']==dm[a['donor_id']]['array_proof'],'Original case/donor assignment unchanged')
    completion=checked_json(e['direct_end_observation_complete']);require(completion['completed'] is True and completion['PASS'] is True,'Closed actual fixed22 weak-end observation receipt')
    require(completion['observations']==e['direct_end_observations'],'Exact actual observation JSONL binding')
    observations=[json.loads(s) for s in checked_bytes(e['direct_end_observations']).splitlines()]
    require(len(observations)==22 and {r['case_id'] for r in observations}=={a['case_id'] for a in attempts},'All original22 weak observations, no replacement')
    em={r['case_id']:r for r in observations};require(len(em)==22 and all(r['original_qL_native_flags_unchanged'] is True for r in observations),'Unique22 and immutable original input assertion')
    op=operators(p);geo=op['geometry'];OUT.mkdir(exist_ok=False)
    write_new(OUT/'STARTED.json',dict(entry=bind(entry_path),pid=os.getpid(),attempts66=True,automatic_retry=False))
    results=[];sourcecache={};donorcache={};start=time.monotonic()
    try:
        for a in attempts:
            folder=OUT/a['pair_id'];folder.mkdir()
            fr=frames[a['frame_index']];k=a['native_slot'];original=original_record(fr,k)
            observed=em[a['case_id']];endpoint=observed['endpoint_observation']
            r=dict(pair_id=a['pair_id'],case_id=a['case_id'],donor_id=a['donor_id'],original_input=original,
                old_region_status=a['region']['region_status'],endpoint_observation=endpoint,
                old_donor_lineage=a['donor'],predecessor_attempt={k:prior[a['pair_id']][k] for k in ('pair_id','status','PASS')},variants={},constructed=False,PASS=False,training_admission=False,
                new_native_quality_owner_all8=-1,direct_end_weak_states=dict(BEFORE='UNKNOWN',COVER='UNKNOWN',SHAM='UNKNOWN'))
            try:
                require(fr['split']=='SOURCE_DEVELOPMENT' and original['query_xy']==a['region']['query_xy'] and original['L']==a['region']['L'],'Original q/L/split join')
                for key in ('query_id','native_slot'):
                    require((fr['query_id'][k] if key=='query_id' else k)==a['region'][key],'Original region identity')
                require(prior[a['pair_id']]['original_input']==original and prior[a['pair_id']]['endpoint_observation']==endpoint,'Original source qL/weak endpoint metadata unchanged')
                end_pixels,reason=endpoint_preflight(original,endpoint,geo)
                if reason:
                    r['status']='UNKNOWN_RETAINED';r['reason']=reason
                else:
                    i=fr['frame_index']
                    if i not in sourcecache:
                        try:
                            rgb=load_bound(fr['canonical_RGB_NPY']);require(rgb.dtype==np.uint8 and list(rgb.shape)==fr['raw_hw']+[3] and ash(rgb)==fr['decoded_RGB_sha256'],'Original canonical RGB authority')
                            saved=load_bound(fr['existing_factual96_packet'],('rgb96','valid96','crop_constructed','global_index'))
                            before=materialize(rgb,fr,op);anchor={'native_slot':k,'existing_RGB96_array_sha256':fr['facts'][k]['RGB96_HWC_sha256']}
                            parity=op['verify_before'](before,fr,anchor,saved);require(all(parity.values()),'Original all8 BEFORE parity')
                            sourcecache[i]=(rgb,before,parity)
                        except Exception:sourcecache[i]=traceback.format_exc()
                    require(not isinstance(sourcecache[i],str),'Cached source failure, no retry: '+str(sourcecache[i])[:200])
                    rgb,before,parity=sourcecache[i];r['before_parity']=parity
                    require(ash(before['rgb96'][k])==fr['facts'][k]['RGB96_HWC_sha256'],'Selected original96 hash')
                    bp=folder/'BEFORE_INPUTS.npz';np.savez_compressed(bp,**before,pasted_alpha=np.zeros(rgb.shape[:2],bool));r['variants']['BEFORE']=bind(bp)
                    did=a['donor_id']
                    if did not in donorcache:
                        try:
                            d=load_bound(a['donor']['compact_RGBA_cutout'],('rgb','alpha'));dr,da=d['rgb'],d['alpha']
                            require(dr.dtype==np.uint8 and da.dtype==bool and dr.shape==(*da.shape,3),'Original RGBA schema')
                            require(ash(dr)==a['donor']['array_proof']['rgb']['array_sha256'] and ash(da)==a['donor']['array_proof']['alpha']['array_sha256'],'Original proved donor array bytes')
                            donorcache[did]=(dr,da)
                        except Exception:donorcache[did]=traceback.format_exc()
                    require(not isinstance(donorcache[did],str),'Cached donor failure, no retry')
                    dr,da=donorcache[did];longside,scale=fixed_transform_plan(end_pixels,da,fr['raw_hw'],op)
                    cut,alpha=op['transformed_cutout'](dr,da,longside);co,so,placement=fixed_placements(scale['end_support_bbox_xyxy'],alpha,fr['raw_hw'],op)
                    old_attempt=prior[a['pair_id']]
                    require(scale==old_attempt['scale_geometry'] and placement==old_attempt['placement_geometry'],'Original donor scale and COVER/SHAM proposal reproduce V189')
                    placement,plan,policy=select_original_or_one_opposite_sham(original,endpoint,placement,geo)
                    co=np.asarray(placement['cover_origin_xy'],np.int64);so=np.asarray(placement['sham_origin_xy'],np.int64)
                    r['scale_geometry']=scale;r['placement_geometry']=placement;r['opposite_sham_policy']=policy;r['finite_end_geometry']=plan
                    tp=folder/'TRANSFORMED_DONOR.npz';np.savez_compressed(tp,rgb=cut,alpha=alpha);r['transformed_donor']=bind(tp)
                    if plan['status']!='GEOMETRY_READY_PIXEL_PROOFS_PENDING':
                        r['status']='GEOMETRY_FAILURE_RETAINED';r['reason']=plan['reasons']
                    else:
                        coverRGB,ca=op['paste'](rgb,cut,alpha,co);shamRGB,sa=op['paste'](rgb,cut,alpha,so)
                        cover,sham=materialize(coverRGB,fr,op),materialize(shamRGB,fr,op)
                        for name,array,mask in (('COVER',cover,ca),('ADJACENT_SHAM',sham,sa)):
                            path=folder/(name+'_INPUTS.npz');np.savez_compressed(path,**array,pasted_alpha=mask);r['variants'][name]=bind(path)
                        checks,detail=prove(before,cover,sham,ca,sa,end_pixels,endpoint,original,cut,alpha,(co,so),op)
                        r['pixel_checks']=checks;r['pixel_detail']=detail;r['constructed']=True
                        closed_proof=contract14_proof(checks,parity,before,cover,dm[did],dict(source_frame=fr['canonical_RGB_NPY'],original_crop=fr['existing_factual96_packet'],donor=a['donor']['compact_RGBA_cutout'],constructed_variants=r['variants']))
                        r['closed14_actual_proof']=closed_proof;r['closed_end_contract']=geo.close_actual_proof(plan,closed_proof)
                        r['PASS']=all(checks.values()) and r['closed_end_contract']['status']=='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED'
                        if old_attempt['PASS']:
                            require(policy['selection']=='ORIGINAL_RETAINED' and r['PASS'],'Original successful5 placement and weak proof preserved')
                            r['old5_decoded_packet_parity']=verify_old_supported_decoded_packets(old_attempt,(before,cover,sham),ca,sa)
                        r['status']='SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED' if r['PASS'] else 'PIXEL_PROOF_FAILURE_UNKNOWN_RETAINED'
                        if r['PASS']:r['direct_end_weak_states']=dict(BEFORE='VISIBLE_WEAK',COVER='EXTERNALLY_BLOCKED_WEAK',SHAM='VISIBLE_WEAK')
                        r['authority']='SOURCE_RGB_SEMANTIC_END_WEAK_WITH_SAMPLED_DOMAIN_2D_OPAQUE_FOREIGN_COMPOSITE_NOT_DEPTH_GOLD'
                        r['contact']=contact(folder,a['case_id'],k,(before,cover,sham),endpoint,checks)
                        r['clean_selected96']={name:save_clean_png(folder/(name+'_CLEAN96.png'),array['rgb96'][k]) for name,array in zip(VARIANTS,(before,cover,sham))}
            except Exception:
                r['PASS']=False;r['direct_end_weak_states']=dict(BEFORE='UNKNOWN',COVER='UNKNOWN',SHAM='UNKNOWN')
                r['status']='ATTEMPT_FAILURE_UNKNOWN_RETAINED';r['traceback']=traceback.format_exc()
            for name in VARIANTS:
                if name not in r['variants']:r['variants'][name]={'constructed':False,'reason':r.get('reason',r['status'])}
            write_new(folder/'PROOF.json',r);results.append(r)
            if prior[a['pair_id']]['PASS']:require(r['PASS'] and all(all(fields.values()) for fields in r['old5_decoded_packet_parity'].values()),'Original supported5 cannot silently become UNKNOWN')
            print(json.dumps({'attempts_completed':len(results),'total':66,'pair_id':a['pair_id'],'status':r['status']}),flush=True)
        manifest=OUT/'ALL66_REAL_END_PAIR_ATTEMPTS.jsonl'
        with manifest.open('x') as f:
            for r in results:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        require(len(results)==66 and {r['pair_id'] for r in results}=={a['pair_id'] for a in attempts},'All original66 retained')
        sections=[]
        for r in results:
            title=html.escape(r['pair_id']+' / '+r['donor_id']+' / '+r['status'])
            image=('<img loading="lazy" src="'+r['pair_id']+'/CONTACT.png" alt="'+html.escape(r['pair_id'])+'">') if r.get('contact') else '<p>No constructed pair; UNKNOWN/failure retained.</p>'
            sections.append('<section><h2>'+title+'</h2><p><a href="'+r['pair_id']+'/PROOF.json">Exact proof and reason</a></p>'+image+'</section>')
        with (OUT/'index.html').open('x') as f:
            f.write('<!doctype html><meta charset="utf-8"><title>V193 fixed66 original-or-one-opposite SHAM source controls</title><style>body{font:16px sans-serif;max-width:1200px;margin:20px auto}img{width:100%;height:auto}section{border-top:1px solid #bbb;margin:24px 0}</style><h1>Fixed22 / original66 source-real semantic-end controls</h1><p>Assistant weak end authority + 2D opaque foreign composite. Native physical visibility, center owner and quality remain separate; no training admission. Every fixed attempt and UNKNOWN/failure is retained.</p>'+''.join(sections))
        # Recheck text/metadata closure; actual assets were hash-checked on their same load descriptors.
        for key in ('source_plan','donor_proof_complete','donor_records'):checked_bytes(p[key])
        checked_bytes(e['direct_end_observations']);checked_bytes(p['predecessor_complete']);checked_bytes(p['predecessor_manifest']);checked_bytes(p['inherited_asset_bindings'])
        done=dict(completed=True,PASS=True,PASS_means_attempt_census_and_invariants_not_all_pairs_supported=True,
            rows66=len(results),cases22=len({r['case_id'] for r in results}),constructed_pairs=sum(r['constructed'] for r in results),
            supported_weak_end_pairs=sum(r['PASS'] for r in results),failed_or_unknown_pairs_retained=sum(not r['PASS'] for r in results),
            actual_source_frames_opened=len(sourcecache),actual_original_donors_opened=len(donorcache),
            manifest=bind(manifest),entry=bind(entry_path),protocol=bind(PROTOCOL),observations=e['direct_end_observations'],
            gallery=bind(OUT/'index.html'),actual_visual_inspection='PENDING_SEPARATE_ACTUAL_TOOL_VIEW',HTML_browser_rendered=False,
            original_q_L_native_all8_and_labels_unchanged=True,new_native_quality_owner_all8=-1,training_admission=False,
            model_encoder_feature_fit_PnP_or_evaluation_reads=0,automatic_retry=False,
            source_authority='V193 source weak endpoint original-or-one-opposite SHAM; all source geometry/proof conditions remain frozen',
            predecessor_complete=p['predecessor_complete'],old5_original_placement_and_decoded_packet_parity=True,
            original_SHAM_retained=sum(r.get('opposite_sham_policy',{}).get('selection')=='ORIGINAL_RETAINED' for r in results),
            opposite_geometry_attempted=sum(r.get('opposite_sham_policy',{}).get('opposite_attempted',False) for r in results),
            no_old_V189_outputs_modified=True,seconds=time.monotonic()-start)
        write_new(OUT/'COMPLETE.json',done);print(json.dumps({'completed':True,'PASS':True,'complete':bind(OUT/'COMPLETE.json'),'supported_pairs':done['supported_weak_end_pairs']}),flush=True)
    except Exception:
        write_new(OUT/'FAILURE.json',dict(completed=False,attempts_completed=len(results),traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',type=Path,required=True);run(ap.parse_args().entry)
