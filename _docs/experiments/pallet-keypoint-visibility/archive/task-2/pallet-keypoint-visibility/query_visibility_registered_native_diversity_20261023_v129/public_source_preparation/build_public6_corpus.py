"""Six fixed PUBLIC weak structural cues, exact V122 composites/V125 query grid.

Preparation reads metadata/code only. Real public RGB and bound donor alpha are
read only by --entry after the root entry, binding and output-absence guards.
There are no learned-model imports, calls, coordinate corrections or fits.
"""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, sys, time, traceback
from collections import Counter
import numpy as np

sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
POLICY = V/'ROOT_PUBLIC6_POLICY_BEFORE_COMPOSITES.json'
OLD = P/'query_visibility_flat_pallet_cue_source_20261023_v122'
OLD_CODE = OLD/'composite_preparation/build_flat_cue_corpus.py'
OLD_DONORS = OLD/'actual_corpus/DONOR16_CONDITIONS_BEFORE_PIXELS.jsonl'
OLD_PROOFS = OLD/'actual_corpus/DONOR_OWNERSHIP_AND_ALPHA_FACTS.jsonl'
OLD_SEAL = OLD/'actual_corpus/CORPUS_SEAL.json'
V125_CODE = P/'query_visibility_boundary_query_source_20261023_v125/build_boundary_query_corpus.py'
MATH = dict(local=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py',
            context=P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py',
            side=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py')
OUT = V/'public_corpus_v1'
PROTOCOL = H/'PUBLIC6_PROTOCOL_BEFORE_PIXELS.json'
GENERATED = H/'GENERATED_PUBLIC6_TESTS.json'
CUES = H/'CUES6_INPUT_ONLY.json'
GT_CUES = H/'CUES6_WEAK_EVIDENCE_GT_ONLY.json'
QUERIES = H/'QUERIES54_INPUT_ONLY.json'
RECIPES = H/'DONOR48_RECIPE_PLAN_BEFORE_PIXELS.json'
CASES = [4, 10, 12, 16, 17, 23]
GRID = [(x,y) for y in (-.25,0.,.25) for x in (-.25,0.,.25)]
CHUNK = 576


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def check(b):
    assert sha(b['path']) == b['sha256'], 'Changed binding: '+b['path']
    return Path(b['path'])


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def clean(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    return x


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def pure(path, names):
    body = [n for n in ast.parse(Path(path).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names)
    ns = dict(np=np)
    exec(compile(ast.Module(body=body,type_ignores=[]), str(path)+'::pure_only', 'exec'), ns)
    return ns


def operators():
    op = pure(MATH['local'], ('crop','query_validity'))
    op.update(pure(MATH['context'], ('query_extent','valid_box','target_context_grid','sample_target_context')))
    op.update(pure(MATH['side'], ('native_grid',)))
    return op


def lower():
    spec = importlib.util.spec_from_file_location('v129_pinned_v122', OLD_CODE)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)  # Definitions/imports only; old main is not invoked.
    k, _, _ = h.load_math()
    return h, k


def initial_points(r, box):
    x0,y0,x1,y1 = map(float, box)
    return np.array([r,[x0,y0],[x1,y0],[x1,y1],[x0,y1],
                     [(x0+x1)/2,y0],[(x0+x1)/2,y1],[x0,(y0+y1)/2]],np.float64)


def query_family(cue, op):
    base = np.asarray(cue['base_q8'],np.float64)
    _,_,L0 = op['native_grid'](base,cue['raw_hw'])
    out = []
    for j,xy in enumerate(GRID):
        q8 = base.copy(); delta = np.asarray(xy,np.float64)*L0; q8[0] += delta
        _,roi,L = op['native_grid'](q8,cue['raw_hw'])
        out.append(dict(condition_index=j,jitter_index=j,condition_id=f'Q{j}',
            normalized_offset_xy=list(xy),offset_xy=delta.tolist(),initial_points=q8.tolist(),q8=q8.tolist(),
            query_xy=q8[0].tolist(),L=float(L),L0=float(L0),
            context_box_xyxy=op['query_extent'](q8),ROI_any_valid=bool(roi[0].any())))
    return out


def metadata_plan():
    p = read(POLICY)
    assert p['model_TRAIN_cases']==[4,10,16,17] and p['model_VALIDATION_cases']==[12,23]
    assert p['counts_planned']['rows']==918 and not p['actual_model_forward_fit_current_V112_PnP_allowed']
    for k in ('root_cue_review','candidate_cues','source_group_overlap_audit'): check(p[k])
    root = read(p['root_cue_review']['path'])
    candidates = read(p['candidate_cues']['path'])
    cr = {r['selection_index']:r for r in candidates['rows']}
    rd = {r['selection_index']:r for r in root['root_decision']}
    donors = lines(OLD_DONORS); proofs = lines(OLD_PROOFS)
    assert len(donors)==16 and len(proofs)==10
    assert {(d['fold'],d['rank']) for d in donors}=={(s,r) for s in ('TRAIN','VALIDATION') for r in range(8)}
    cues=[];gt=[];queries=[];recipes=[];op=operators()
    for ci,case in enumerate(CASES):
        c=cr[case];d=rd[case]
        assert d['decision']=='ELIGIBLE_FOR_WEAK_AUX_COMPOSITE_GENERATION'
        assert d['candidate_xy']==c['cue_xy_original_RGB'] and d['uncertainty_bbox']==c['uncertainty_bbox_xyxy_closed']
        assert c['original_fold']=='TRAIN'
        b=c['target_identity']['designated_bbox_xywh'];box=[b[0],b[1],b[0]+b[2],b[1]+b[3]]
        split='TRAIN' if case in p['model_TRAIN_cases'] else 'VALIDATION'
        cue=dict(base_index=ci,cue_index=ci,base_id=f'PUBLIC6_CASE_{case:02d}',selection_index=case,
            source='LOCO_PUBLIC6',style='PUBLIC_LOCO_STRUCTURAL_CUE',split=split,model_split=split,
            raw_split='TRAIN',official_split='TRAIN',original_official_split='TRAIN',
            source_frame_group=c['source_frame_group'],take=c['take'],camera_stream=c['camera_stream'],
            image_id=c['image_id'],original_member=c['original_member'],
            original_full_RGB=c['full_RGB'],original_full_RGB_format='JPEG_RGB',
            original_decoded_RGB_sha256=c['decoded_RGB_sha256'],raw_hw=c['raw_hw'],HW=c['raw_hw'],
            target_bbox_xyxy=box,target_annotation_id=c['target_identity']['official_annotation']['id'],
            base_q8=initial_points(d['candidate_xy'],box).tolist(),
            source_query_emulation=True,actual_initializer_output=False,carrier_slot=0,logical_slot=0,
            native8_mapping='UNKNOWN',human_gold=False,training_admission=False,
            development_exposure='C16_SOURCE_TAKE792_U_ONLY_INPUT_ROWS_NO_KNOWN_LOSS' if case==23 else
              'NO_AUDITED_C16_SOURCE_TAKE_MATCH; same environment/date; source visually reviewed')
        qs=query_family(cue,op);cue['L0']=qs[4]['L0'];cues.append(cue)
        gt.append(dict(cue_index=ci,selection_index=case,reference_xy=d['candidate_xy'],
            original_regions=[dict(kind='CLOSED_NATIVE_UNCERTAINTY',bbox=d['uncertainty_bbox'])],
            before_visible_weak_supported=True,original_candidate=c,root_decision=d,
            root_cue_review=p['root_cue_review'],native_mapping='UNKNOWN',human_gold=False,training_admission=False))
        for q in qs: queries.append(dict(cue,**q,global_query_condition_index=9*ci+q['jitter_index']))
        for rank in range(8):
            donor=next(x for x in donors if x['fold']==split and x['rank']==rank)
            proof=next(x for x in proofs if x['fold']==split and x['donor_id']==donor['donor_id'] and x['kind']==donor['kind'])
            recipes.append(dict(attempt_index=8*ci+rank,cue_index=ci,selection_index=case,split=split,
                donor_rank=rank,assigned_donor=donor,inherited_actual_donor_certificate=proof,
                no_donor_replacement=True,scale_order=[donor['factor']] if donor['kind']=='EXTERNAL_RGBA' else [.6,1.,1.4,2.],
                placement='Exact pinned V122 placement_pair on the complete original closed uncertainty region',
                geometry_result_not_yet_observed=True))
    assert len(cues)==6 and len(queries)==54 and len(recipes)==48
    assert {r['source_frame_group'] for r in cues if r['split']=='TRAIN'}.isdisjoint(
           {r['source_frame_group'] for r in cues if r['split']=='VALIDATION'})
    return p,cues,gt,queries,recipes


def semantic_state(variant, ready, outside, full_cover, full_clear):
    if outside:return 'U'
    if variant=='BEFORE':return 'D'
    if not ready:return 'U'
    if variant=='OPAQUE_COVER':return 'E' if full_cover else 'U'
    if variant=='CUE_PRESERVING_SHAM':return 'D' if full_clear else 'U'
    raise ValueError(variant)


def validate_entry(path):
    e=read(path)
    assert set(e)=={'approved_actual_public_source_build','source_only','model_allowed','fit_allowed',
        'current_allowed','V112_allowed','PnP_allowed','restart_allowed','code','protocol','generated','root_policy','output'}
    assert e['approved_actual_public_source_build'] is True and e['source_only'] is True
    assert all(e[k] is False for k in ('model_allowed','fit_allowed','current_allowed','V112_allowed','PnP_allowed','restart_allowed'))
    for k in ('code','protocol','generated','root_policy'):check(e[k])
    p=read(PROTOCOL)
    assert e['code']==p['code']==bind(__file__) and e['protocol']==bind(PROTOCOL)
    assert e['generated']==p['generated']==bind(GENERATED) and e['root_policy']==p['root_policy']==bind(POLICY)
    assert e['output']==p['output']==str(OUT) and not OUT.exists(), 'Preserve prior complete/failed invocation'
    for b in p['bindings'].values():check(b)  # Code/JSON metadata only, before image/alpha payload reads.
    assert read(GENERATED)['PASS'] is True
    return e,p


def run(entry):
    e,p=validate_entry(entry);op=operators();h,k=lower()
    from PIL import Image, ImageDraw, ImageFont
    cues=read(CUES)['rows'];evidence=read(GT_CUES)['rows'];queries=read(QUERIES)['rows'];recipes=read(RECIPES)['rows']
    OUT.mkdir();(OUT/'native_scenes').mkdir();(OUT/'patches').mkdir();(OUT/'protected').mkdir();(OUT/'figures').mkdir()
    write(OUT/'RUN_STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),source_only=True,restart_allowed=False))
    start=time.monotonic();donor_cache={};donor_records=[];scenes=[];gt_scenes=[];facts=[]
    input_rows=[None]*918;gt_rows=[None]*918
    rgbpack=np.empty((918,96,96,3),np.uint8);validpack=np.empty((918,96,96),bool)
    font=ImageFont.load_default();source_board=Image.new('RGB',(1320,1200),'#f5f7fa')
    try:
        def load_donor(recipe):
            d=recipe['assigned_donor'];cert=recipe['inherited_actual_donor_certificate'];key=(d['fold'],d['donor_id'],d['kind'])
            if key in donor_cache:return donor_cache[key]
            assert cert['admitted'] is True and cert['donor_id']==d['donor_id']
            if d['kind']=='EXTERNAL_RGBA':
                s=d['source'];check(dict(path=s['RGBA_path'],sha256=s['sha256']))
                rgba=np.array(Image.open(s['RGBA_path']).convert('RGBA'),copy=True)
                rgb=rgba[...,:3].copy();alpha=rgba[...,3]==255
                assert int(alpha.sum())==cert['opaque255_pixels']
            else:
                s=d['source'];check(s['image']);check(cert['native_proof'])
                rgb=np.array(Image.open(s['image']['path']).convert('RGB'),copy=True)
                with np.load(cert['native_proof']['path'],allow_pickle=False)as z:
                    selected=z['selected_visible'];other=z['other_visible_union'];exclusive=z['exclusive_alpha'];alpha=z['erode1_exclusive']
                assert cert['complete_owner_instance_set'] and cert['exclusive_alpha']['complete_other_instance_masks']
                assert selected.dtype==other.dtype==exclusive.dtype==alpha.dtype==bool
                assert np.array_equal(exclusive,selected&~other) and np.array_equal(alpha,h.erode1(exclusive))
                assert not(alpha&other).any() and not(alpha&~selected).any()
            assert digest(rgb)==cert['native_decoded_RGB_sha256'] and alpha.any()
            cut=h.native_cutout(rgb,alpha);assert cut is not None
            donor_cache[key]=cut;donor_records.append(dict(key=list(key),source_certificate=cert,
                actual_native_RGB_sha256=digest(rgb),native_cutout_bbox=cut['native_bbox_xyxy'],
                alpha_pixels=int(alpha.sum()),no_new_donor_selection=True))
            return cut

        for ci,(cue,ev)in enumerate(zip(cues,evidence)):
            assert cue['cue_index']==ev['cue_index']==ci
            check(cue['original_full_RGB'])
            native=np.array(Image.open(cue['original_full_RGB']['path']).convert('RGB'),copy=True)
            assert native.shape==tuple(cue['raw_hw'])+(3,) and digest(native)==cue['original_decoded_RGB_sha256']
            qrows=queries[9*ci:9*ci+9];assert [q['jitter_index']for q in qrows]==list(range(9))
            protected,pf=h.protected_mask(cue,ev);pp=OUT/'protected'/f'CUE_{ci:02d}.npz';np.savez_compressed(pp,protected_union=protected)
            ys,xs=np.nonzero(protected);outside=bool(pf['original_uncertainty_has_outside_image_cells'])
            assert len(xs)>0
            basefile=OUT/'native_scenes'/f'CUE_{ci:02d}_BEFORE.npz';np.savez_compressed(basefile,RGB_uint8=native)
            basebinding=bind(basefile);native_sha=digest(native);variants=[dict(variant='BEFORE',donor_rank=None,image=native,
                full_RGB=basebinding,full_RGB_recipe=dict(kind='ORIGINAL',original_full_RGB=cue['original_full_RGB'],original_format='JPEG_RGB',decoded_RGB_sha256=native_sha),
                state=semantic_state('BEFORE',True,outside,False,True),pair_ready=True,protected_all_covered=False,protected_all_clear=True,reason='ROOT_FIXED_VISIBLE_STRUCTURAL_CUE_WEAK_AUX')]
            for rank in range(8):
                recipe=recipes[8*ci+rank];assert recipe['cue_index']==ci and recipe['donor_rank']==rank
                base=load_donor(recipe);selected=None;attempt=dict(attempt_index=8*ci+rank,cue_index=ci,selection_index=cue['selection_index'],
                    donor_rank=rank,split=cue['split'],raw_split='TRAIN',protected_file=bind(pp),protected_facts=pf,
                    donor_recipe=recipe,pair_ready=False,scale_trials=[])
                for factor in recipe['scale_order']:
                    cut=h.resize_cutout(base,factor,cue['L0'],k);pair,geo=h.placement_pair(cut,protected,np.asarray(ev['reference_xy']),cue['L0'])
                    attempt['scale_trials'].append(dict(factor=factor,resized_hw=cut['rounded_resized_hw'],geometry=geo))
                    if pair is not None:selected=(cut,pair,geo);break
                if selected is None:
                    attempt['reason']=attempt['scale_trials'][-1]['geometry']['reason']
                    for variant in ('OPAQUE_COVER','CUE_PRESERVING_SHAM'):
                        variants.append(dict(variant=variant,donor_rank=rank,image=native,full_RGB=basebinding,
                            full_RGB_recipe=dict(kind='ORIGINAL_FAILED_CONSTRUCTION_ALIAS',original_full_RGB=cue['original_full_RGB'],
                                original_format='JPEG_RGB',decoded_RGB_sha256=native_sha),state='U',pair_ready=False,
                            protected_all_covered=False,protected_all_clear=True,reason=attempt['reason']))
                else:
                    cut,(cxy,sxy),geo=selected;patch=OUT/'patches'/f'CUE_{ci:02d}_RANK_{rank:02d}.npz'
                    np.savez_compressed(patch,rgb=cut['rgb'],alpha=cut['alpha']);patchbinding=bind(patch)
                    coverall=bool(h.alpha_at(cut,cxy,xs,ys,protected.shape).all());shamclear=not bool(h.alpha_at(cut,sxy,xs,ys,protected.shape).any())
                    assert coverall and shamclear
                    attempt.update(pair_ready=True,reason=geo['reason'],patch=patchbinding,cover_xy=list(cxy),sham_xy=list(sxy),
                        cover_protected_all_opaque=coverall,sham_protected_all_clear=shamclear,selected_factor=cut['factor_over_L0'],
                        local_alpha_mass96_at_r_L0=geo['local_alpha_mass96_at_r_L0'],mass_equal_claim=False)
                    for variant,xy in [('OPAQUE_COVER',cxy),('CUE_PRESERVING_SHAM',sxy)]:
                        image=h.reconstruct_full_RGB(native,cut['rgb'],cut['alpha'],xy)
                        if variant=='CUE_PRESERVING_SHAM':assert np.array_equal(image[protected],native[protected])
                        f=OUT/'native_scenes'/f'CUE_{ci:02d}_RANK_{rank:02d}_{variant}.npz';np.savez_compressed(f,RGB_uint8=image)
                        variants.append(dict(variant=variant,donor_rank=rank,image=image,full_RGB=bind(f),
                            full_RGB_recipe=dict(kind='OPAQUE_PATCH',original_full_RGB=cue['original_full_RGB'],original_format='JPEG_RGB',
                                patch=patchbinding,patch_RGB_key='rgb',patch_alpha_key='alpha',placement_xy=list(xy),decoded_RGB_sha256=digest(image)),
                            state=semantic_state(variant,True,outside,coverall,shamclear),pair_ready=True,
                            protected_all_covered=variant=='OPAQUE_COVER' and coverall,protected_all_clear=variant=='CUE_PRESERVING_SHAM' and shamclear,
                            reason=geo['reason'],patch_alpha=cut['alpha'],placement_xy=list(xy)))
                facts.append(attempt)
            assert len(variants)==17
            # Source display has a marked copy only; all stored model RGB remains unmarked.
            ci2=ci%2;ri2=ci//2;panel=Image.new('RGB',(640,380),'white');d=ImageDraw.Draw(panel)
            thumb=Image.fromarray(native);td=ImageDraw.Draw(thumb);b=cue['target_bbox_xyxy'];td.rectangle(b,outline='#00ff88',width=4)
            ub=ev['original_regions'][0]['bbox'];td.rectangle(ub,outline='#ff00aa',width=3);thumb.thumbnail((400,320));panel.paste(thumb,(5,35))
            r=np.asarray(ev['reference_xy']);local=Image.fromarray(native).crop([int(r[0])-32,int(r[1])-32,int(r[0])+32,int(r[1])+32]).resize((224,224),Image.Resampling.NEAREST)
            ld=ImageDraw.Draw(local);ld.rectangle([(ub[0]-(r[0]-32))*3.5,(ub[1]-(r[1]-32))*3.5,(ub[2]-(r[0]-32)+1)*3.5,(ub[3]-(r[1]-32)+1)*3.5],outline='#ff00aa',width=2);panel.paste(local,(410,80))
            d.text((8,8),f"case{cue['selection_index']} {cue['split']} | official TRAIN | weak AUX only",fill='black',font=font)
            source_board.paste(panel,(ci2*660+5,ri2*400+5))
            triplets=Image.new('RGB',(660,1900),'#f5f7fa');draw=ImageDraw.Draw(triplets)
            draw.text((10,8),f"CASE {cue['selection_index']} {cue['split']} | centre q, same L | raw 96 shown2x",fill='black',font=font)
            for role,v in enumerate(variants):
                scene_index=17*ci+role;scene_id=f"PUBLIC6_CASE{cue['selection_index']:02d}_{role:02d}"
                sc={key:copy.deepcopy(cue[key])for key in ('base_index','base_id','selection_index','source','style','split','raw_split','official_split','source_frame_group','take','raw_hw')}
                sc.update(scene_index=scene_index,scene_id=scene_id,variant=v['variant'],donor_rank=v['donor_rank'],
                    full_RGB=v['full_RGB'],full_RGB_format='NPZ_RGB_UINT8_HWC',full_RGB_key='RGB_uint8',
                    decoded_RGB_sha256=digest(v['image']),full_RGB_recipe=v['full_RGB_recipe'])
                scenes.append(sc);gt_scenes.append(dict(scene_index=scene_index,scene_id=scene_id,raw_state=v['state'],
                    external_target=0 if v['state']=='D' else 1 if v['state']=='E' else -1,geometry_pair_ready=v['pair_ready'],
                    protected_all_covered=v['protected_all_covered'],protected_all_clear=v['protected_all_clear'],protected_file=bind(pp),
                    reference_xy=ev['reference_xy'],original_regions=ev['original_regions'],reason=v['reason'],weak_AUX_only=True,
                    native_mapping='UNKNOWN',human_gold=False,training_admission=False))
                for j,q in enumerate(qrows):
                    i=153*ci+17*j+role;a,vv=op['crop'](v['image'],np.asarray(q['query_xy']),q['L'],96);rgb=a.transpose(1,2,0).copy();valid=vv[0].copy()
                    rgbpack[i]=rgb;validpack[i]=valid
                    _,cont,near=op['query_validity'](np.asarray([q['query_xy']]),cue['raw_hw']);central=bool(valid[47:49,47:49].all())
                    _,cv=op['sample_target_context'](v['image'],q['context_box_xyxy']);ctx=bool(cv.any());factual=bool(cont[0] and near[0] and central)
                    oid=f"V129_PUBLIC6:C{ci:02d}:J{j}:R{role:02d}";qr=copy.deepcopy(sc)
                    qr.update(occurrence_index=i,row_id=i,id=oid,occurrence_id=oid,query_id=oid,
                        query_group_id=f'V129_PUBLIC6_C{ci:02d}_J{j}',logical_slot=0,carrier_slot=0,query_index=0,
                        condition_index=j,condition_id=f'QUARTER_L0_J{j}',jitter_index=j,offset_xy=q['offset_xy'],
                        normalized_offset_xy=q['normalized_offset_xy'],query_xy=q['query_xy'],initial_points=q['initial_points'],q8=q['initial_points'],
                        L=q['L'],L0=q['L0'],context_box_xyxy=q['context_box_xyxy'],HW=cue['raw_hw'],
                        rgb96_sha256=digest(rgb),valid96_sha256=digest(valid),rgb96_layout='HWC',valid96_layout='HW',
                        q_native_continuous_valid=bool(cont[0]),q_nearest_valid=bool(near[0]),ROI_any_valid=q['ROI_any_valid'],
                        local_central4_valid=central,factual_query_available=factual,factual_available=factual,context_available=ctx,
                        query_available=factual and ctx,original_input_availability=dict(continuous=bool(cont[0]),nearest=bool(near[0]),
                            ROI_any_valid=q['ROI_any_valid'],central4=central,context=ctx),source_query_emulation=True,actual_initializer_output=False)
                    input_rows[i]=qr;qpix=np.floor(np.asarray(q['query_xy'])+.5).astype(int);qalpha=False
                    if 'patch_alpha'in v:
                        qalpha=bool(h.alpha_at(dict(alpha=v['patch_alpha']),v['placement_xy'],np.asarray([qpix[0]]),np.asarray([qpix[1]]),cue['raw_hw'])[0])
                    gt_rows[i]=dict(row_id=i,occurrence_index=i,id=oid,scene_index=scene_index,scene_id=scene_id,cue_index=ci,
                        selection_index=cue['selection_index'],split=cue['split'],raw_split='TRAIN',source_frame_group=cue['source_frame_group'],
                        variant=v['variant'],donor_rank=v['donor_rank'],jitter_index=j,reference_xy=ev['reference_xy'],
                        delta_xy=q['offset_xy'],error_px=float(np.linalg.norm(q['offset_xy'])),raw_state=v['state'],
                        external_target_D0_E1_else_minus1=0 if v['state']=='D' else 1 if v['state']=='E' else -1,
                        candidate_aux_hidden_label_GT_only=0 if v['state']=='D' else 1 if v['state']=='E' else -1,
                        known_geometric_S=False,teacher_reason=v['reason'],geometry_pair_ready=v['pair_ready'],
                        original_uncertainty_regions=ev['original_regions'],weak_cue_provenance=dict(file=bind(GT_CUES),row=ci),
                        nearest_q_foreground_alpha_diagnostic=qalpha,nearest_q_inside_image_diagnostic=bool(near[0]),
                        source_native_registration='A visible 2D component cue only; no public native8 correspondence certified.',
                        human_gold=False,native_mapping='UNKNOWN',training_admission=False)
                    if j==4 and role:
                        rank=(role-1)//2;col=0 if role%2 else 2;yy=55+rank*230
                        # BEFORE/COVER/SHAM: raw, identical query settings; no point/box overlay.
                        if role%2:
                            before=rgbpack[153*ci+17*4];triplets.paste(Image.fromarray(before).resize((192,192),Image.Resampling.NEAREST),(10,yy))
                            triplets.paste(Image.fromarray(rgb).resize((192,192),Image.Resampling.NEAREST),(230,yy))
                        else:triplets.paste(Image.fromarray(rgb).resize((192,192),Image.Resampling.NEAREST),(450,yy))
                        draw.text((10,yy-16),f"rank{rank} BEFORE / COVER / SHAM | {'ready' if v['pair_ready'] else 'FAILED aftersU; base alias'}",fill='black',font=font)
            triplets.save(OUT/'figures'/f'CUE_{ci:02d}_CASE{cue["selection_index"]:02d}_CENTRE_TRIPLETS8.png')
            print(json.dumps(dict(cue_completed=ci+1,total=6,case=cue['selection_index'],accepted=sum(x['pair_ready']for x in facts[-8:]))),flush=True)
        assert all(r is not None for r in input_rows+gt_rows) and len(scenes)==len(gt_scenes)==102
        source_board.save(OUT/'figures/SOURCE_PROTECTED6_CONTACT.png')
        chunks=[]
        for start in range(0,918,CHUNK):
            end=min(918,start+CHUNK);f=OUT/f'RGB96_INPUT_ONLY_{start:04d}_{end:04d}.npz'
            np.savez_compressed(f,rgb96=rgbpack[start:end],valid96=validpack[start:end],occurrence_index=np.arange(start,end,dtype=np.int32));b=bind(f);chunks.append(b)
            for i in range(start,end):input_rows[i].update(rgb96_chunk=b,crop_chunk=b,rgb96_row=i-start,crop_row=i-start)
        for i,r in enumerate(input_rows):assert r['occurrence_index']==gt_rows[i]['occurrence_index']==i
        si=OUT/'SCENES_INPUT_ONLY.json';sg=OUT/'GT_SCENES.json';ip=OUT/'INPUT_ROWS.json';gp=OUT/'GT_ROWS.json'
        write(si,dict(completed=True,rows=scenes,source_only=True,training_admission=False))
        write(sg,dict(completed=True,rows=gt_scenes,weak_AUX_only=True,training_admission=False))
        write(ip,dict(completed=True,rows=input_rows,crop_chunks=chunks,scenes=bind(si),protocol=bind(PROTOCOL),
            source_query_emulation_not_actual_initializer=True,training_admission=False))
        write(gp,dict(completed=True,rows=gt_rows,GT_scenes=bind(sg),input_row_identity=bind(ip),
            teacher_U_separate_from_factual_input_U=True,native_mapping='UNKNOWN',human_gold=False,training_admission=False))
        write(OUT/'FEASIBILITY48_GT_ONLY.json',dict(completed=True,rows=facts))
        write(OUT/'DONOR_REPLAY_CERTIFICATES.json',dict(completed=True,rows=donor_records,original_metadata=bind(OLD_PROOFS)))
        counts=Counter(g['raw_state']for g in gt_rows)
        summary=dict(completed=True,PASS=True,cues=6,scenes=102,query_conditions=54,rows=918,
            internal_split_rows=Counter(r['split']for r in input_rows),all_original_official_TRAIN=True,
            raw_states=counts,query_available=sum(r['query_available']for r in input_rows),
            accepted_pairs=sum(x['pair_ready']for x in facts),attempts=48,
            source_scene_files=len(list((OUT/'native_scenes').glob('*.npz'))),
            source_scene_bytes=sum(x.stat().st_size for x in (OUT/'native_scenes').glob('*.npz')),
            seconds=time.monotonic()-start,model_encoder_fit_current_V112_PnP=0,training_admission=False)
        write(OUT/'SUMMARY.json',summary)
        complete=dict(completed=True,PASS=True,entry=bind(entry),code=bind(__file__),protocol=bind(PROTOCOL),root_policy=bind(POLICY),
            input_manifest=bind(ip),GT_only=bind(gp),scenes=bind(si),GT_scenes=bind(sg),stored_crops=chunks,crop_chunks=chunks,
            summary=bind(OUT/'SUMMARY.json'),feasibility=bind(OUT/'FEASIBILITY48_GT_ONLY.json'),
            donor_replay=bind(OUT/'DONOR_REPLAY_CERTIFICATES.json'),
            prepared_metadata=p['prepared_metadata'],native_files=[bind(x)for x in sorted((OUT/'native_scenes').glob('*.npz'))],
            patches=[bind(x)for x in sorted((OUT/'patches').glob('*.npz'))],protected=[bind(x)for x in sorted((OUT/'protected').glob('*.npz'))],
            figures=[bind(x)for x in sorted((OUT/'figures').glob('*.png'))],rows=918,source_only=True,
            teacher_U_not_input_availability=True,native_mapping='UNKNOWN',human_gold=False,training_admission=False)
        write(OUT/'CORPUS_COMPLETE.json',complete)
        print(json.dumps(dict(completed=True,receipt=bind(OUT/'CORPUS_COMPLETE.json'),summary=summary)),flush=True)
    except BaseException:
        write(OUT/'FAILED_NO_RESTART.json',dict(error=traceback.format_exc(),seconds=time.monotonic()-start,preserve_all_partial_outputs=True,automatic_restart=False))
        raise


def generated():
    h,k=lower();op=operators();tests={}
    c=dict(raw_hw=[100,120],base_q8=initial_points([30,30],[20,20,100,80]).tolist())
    e=dict(before_visible_weak_supported=True,reference_xy=[30,30],original_regions=[dict(kind='CLOSED_NATIVE_UNCERTAINTY',bbox=[28,27,32,33])])
    pm,pf=h.protected_mask(c,e);assert pm.sum()==35 and pm[27:34,28:33].all() and not pf['original_uncertainty_has_outside_image_cells']
    tests['closed_uncertainty_inclusive_without_shrink']=True
    fam=query_family(c,op);assert len(fam)==9 and fam[4]['query_xy']==[30.,30.]and fam[0]['normalized_offset_xy']==[-.25,-.25]
    assert fam[0]['query_xy']==[25.,25.]and fam[0]['L']==20. and fam[0]['context_box_xyxy']==[20.,20.,100.,80.]
    tests['fixed_carrier0_nine_dy_dx_and_recomputed_query_operator']=True
    cut=dict(alpha=np.ones((9,9),bool),rgb=np.full((9,9,3),200,np.uint8),opaque_core_xyxy=[0,0,9,9])
    pair,geo=h.placement_pair(cut,pm,np.asarray([30.,30.]),20.);assert pair is not None
    xs,ys=np.nonzero(pm)[1],np.nonzero(pm)[0];assert h.alpha_at(cut,pair[0],xs,ys,pm.shape).all() and not h.alpha_at(cut,pair[1],xs,ys,pm.shape).any()
    original=np.full((100,120,3),50,np.uint8);cover=h.reconstruct_full_RGB(original,cut['rgb'],cut['alpha'],pair[0]);sham=h.reconstruct_full_RGB(original,cut['rgb'],cut['alpha'],pair[1])
    assert np.array_equal(sham[pm],original[pm]) and (cover[pm]==200).all() and (original==50).all()
    tests['whole_region_cover_clear_sham_opaque_composite_no_mutation']=True
    assert semantic_state('OPAQUE_COVER',True,False,False,False)=='U' and semantic_state('CUE_PRESERVING_SHAM',False,False,True,True)=='U'
    assert semantic_state('OPAQUE_COVER',True,False,True,False)=='E'  # qalpha is deliberately absent from the teacher function.
    tests['partial_or_failed_U_separate_from_qpixel_and_input_availability']=True
    assert [(i//CHUNK,i%CHUNK)for i in (0,575,576,917)]==[(0,0),(0,575),(1,0),(1,341)]
    assert len({153*ci+17*j+role for ci in range(6)for j in range(9)for role in range(17)})==918
    tests['918_order_and_chunk_local_index_global_identity']=True
    assert all(tests.values())
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),tests=tests,source_payload_reads=0,model_calls=0))
    print(json.dumps(dict(generated=bind(GENERATED))),flush=True)


def prepare():
    g=read(GENERATED);assert g['PASS']and g['code']==bind(__file__)
    policy,cues,gt,qs,recipes=metadata_plan()
    write(CUES,dict(completed=True,rows=cues,root_policy=bind(POLICY),source_pixel_reads=0))
    write(GT_CUES,dict(completed=True,rows=gt,root_policy=bind(POLICY),training_admission=False))
    write(QUERIES,dict(completed=True,rows=qs,query_emulation=True,actual_initializer=False))
    write(RECIPES,dict(completed=True,rows=recipes,inherited_donor_conditions=bind(OLD_DONORS),inherited_donor_certificates=bind(OLD_PROOFS),source_pixel_reads=0))
    meta=dict(cues=bind(CUES),GT_cues=bind(GT_CUES),queries=bind(QUERIES),recipes=bind(RECIPES))
    bindings=dict(V122_code=bind(OLD_CODE),V125_code=bind(V125_CODE),V122_donor_conditions=bind(OLD_DONORS),V122_donor_certificates=bind(OLD_PROOFS),V122_seal=bind(OLD_SEAL),**{k:bind(v)for k,v in MATH.items()},**meta)
    protocol=dict(completed=True,code=bind(__file__),root_policy=bind(POLICY),generated=bind(GENERATED),bindings=bindings,prepared_metadata=meta,output=str(OUT),
        counts=dict(cues=6,scenes=102,rows=918,TRAIN=612,VALIDATION=306),
        order='case4,10,12,16,17,23; occurrence=153*cue+17*jitter+role; jitterdyouter/dxinner; role BEFORE then COVER/SHAM ranks0..7',
        crop_layout=dict(rgb96='uint8[N,96,96,3] HWC',valid96='bool[N,96,96] HW',max_chunk_rows=576,rgb96_row='local in chunk',occurrence_index='global'),
        geometry='Exact unchanged V122 protected_mask, native_cutout, resize_cutout, placement_pair and reconstruct_full_RGB; same donor8 by internal fold; deterministic geometric scale enumeration only; no donor replacement.',
        query='Exact V125 normalized quarterL0 Cartesian9; carrier0 only; bbox-derived other7 anchors; V60L and V107unclippedq8context recomputed after each offset; no GT-best/native association.',
        alpha='Reuse complete-owner certified V122 native exclusive/erode1 proof and exact RGBAalpha255; no new masks or donor choice.',
        label_boundary='Complete protected-region E/clearSHAM D is conditional weak cue visibility; partial/failureU. TeacherU never changes factual input availability; nearestqalpha diagnostic GT-only.',
        original_official_TRAIN_preserved=True,internal_VAL_case23_development_exposed=True,no_training_admission=True,
        real_payload_reads_or_model_fit_current_V112_PnP=0)
    write(PROTOCOL,protocol)
    template=dict(approved_actual_public_source_build=False,source_only=True,model_allowed=False,fit_allowed=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,restart_allowed=False,
        code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),root_policy=bind(POLICY),output=str(OUT))
    tp=H/'ROOT_PUBLIC6_BUILD_ENTRY_TEMPLATE.json';write(tp,template)
    print(json.dumps(dict(completed=True,code=bind(__file__),protocol=bind(PROTOCOL),template=bind(tp),
        command=f'{sys.executable} -B {Path(__file__).resolve()} --entry ROOT_ACTUAL_PUBLIC6_BUILD_ENTRY.json',source_payload_reads=0)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--generated',action='store_true');g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path)
    a=ap.parse_args()
    if a.generated:generated()
    elif a.prepare:prepare()
    else:run(a.entry)
