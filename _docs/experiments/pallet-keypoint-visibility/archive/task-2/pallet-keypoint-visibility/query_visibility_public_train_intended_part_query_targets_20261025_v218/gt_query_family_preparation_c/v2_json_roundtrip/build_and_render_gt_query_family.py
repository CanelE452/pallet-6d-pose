"""Separate GT-derived SOURCE query inputs. --prepare is metadata/invented-only."""
import argparse, copy, hashlib, io, json, math, os, re, stat
from collections import Counter
from pathlib import Path
import numpy as np

D = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_train_intended_part_query_targets_20261025_v218')
H = Path(__file__).parent
SCOPE = D / 'ROOT_PROSPECTIVE_SEPARATE_SOURCE_GT_QUERY_FAMILY_CONSTRUCTION.json'
SCOPE_SHA = '15af8d2dacb37c11dda29121dbf01abc9e555287454fe20e6476fbb91273dfa6'
PLAN = H / 'PLAN360_AND_EXACT12_BINDINGS.json'
PROTO = H / 'PROTOCOL_BEFORE_EXACT12_DERIVED_RGB_PNG_BODIES.json'
OUT = D / 'public_source_gt_query_jitter_v1'
FAMILY = 'PUBLIC_SOURCE_GT_QUERY_JITTER_V1'
OFFSETS = [(0., 0.), (.25, 0.), (0., .25)]
AUTHORITY = 'GT-derived training q8; uncertain nominal anatomy, visibility UNKNOWN. No additional GT pose/ref/ROI/mask deployment feature. Original production q unchanged.'

def safe(p):
    p = Path(p)
    assert p.is_absolute() and not re.search(r'v112', str(p), re.I) and '..' not in p.parts
    for a in [*reversed(p.parents), p]:
        assert not a.is_symlink(), str(a)
    return p

def read(p):
    p = safe(p)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        st = os.fstat(f.fileno()); assert stat.S_ISREG(st.st_mode) and st.st_size <= 64 * 1024**2
        return f.read()

def bind(p):
    b = read(p)
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))

def verified(b):
    raw = read(b['path'])
    assert hashlib.sha256(raw).hexdigest() == b['sha256']
    assert 'size_bytes' not in b or len(raw) == b['size_bytes']
    return raw

def write(p, obj):
    p = safe(p)
    raw = obj if isinstance(obj, bytes) else (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    with p.open('xb') as f: f.write(raw)
    return bind(p)

def inside(q, hw):
    return q is not None and len(q) == 2 and all(math.isfinite(float(x)) for x in q) and -.5 <= q[0] < hw[1]-.5 and -.5 <= q[1] < hw[0]-.5

def make_plan(scope, nominal, pixels, complete):
    source12 = scope['future_exact_derived_RGB12']
    assert [x['image'] for x in complete['original_context12']] == source12 and len(source12) == 12
    originals, candidates, rows, groups = {}, {}, [], []
    dims = {(r['frame_position'], r['candidate_gt_list_index']): [r['height'], r['width']] for r in pixels['candidate_rows15']}
    for r in nominal['rows120']:
        o = r['original_input']; k = (o['frame_position'], r['diagnostic_candidate_gt_list_index'])
        assert o['original_role'] == 'TRAIN' and o['internal_source_role'] in ['SOURCE_FIT', 'SOURCE_DEV'] and o['training_admission'] is False
        assert o['row_id'] not in originals or originals[o['row_id']] == o
        originals[o['row_id']] = copy.deepcopy(o); candidates.setdefault(k, []).append(r)
    assert len(originals) == 96 and sum(not x['original_query_available'] for x in originals.values()) == 11 and len(candidates) == 15
    assert Counter(x['internal_source_role'] for x in originals.values()) == {'SOURCE_FIT': 64, 'SOURCE_DEV': 32}
    for (f, c), rr in sorted(candidates.items()):
        rr = sorted(rr, key=lambda r: r['native_slot']); assert [r['native_slot'] for r in rr] == list(range(8))
        assert len({r['original_input']['L'] for r in rr}) == 1
        for s, offset in enumerate(OFFSETS):
            gid = f'F{f:02d}_C{c:06d}_S{s}'; gg = []
            for r in rr:
                o, ref = r['original_input'], r['nominal_reference_xy']; L = o['L']; hw = dims[(f,c)]
                finite = ref is not None and all(math.isfinite(float(x)) for x in ref) and math.isfinite(L) and L > 0
                q = [float(ref[j]) + offset[j]*L for j in range(2)] if finite else None
                row = dict(family=FAMILY, row_id=f'{gid}_N{r["native_slot"]}', group_id=gid, frame_position=f, candidate_gt_list_index=c,
                    selected_GT_instance_index=None, instance_correspondence='UNKNOWN', native_slot=r['native_slot'], canonical_index=r['canonical_index'],
                    anatomical_family='DECK' if r['canonical_index'] in [0,1,4,5] else 'BOTTOM_RUNNER', normalized_offset=list(offset),
                    generated_query_xy=q, L=L, generated_query_available=bool(inside(q,hw)), nominal_reference_xy=ref,
                    nominal_target_in_frame=bool(inside(ref,hw)), source_hw=hw, source_RGB_binding=source12[f],
                    original_input=copy.deepcopy(o), source_role=o['internal_source_role'], component_id=o['component_id'],
                    future_fit_role_eligible=o['internal_source_role']=='SOURCE_FIT', training_admission=False,
                    native_visibility_D_E_SELF_UNKNOWN='UNKNOWN', visible_intended_part='UNKNOWN', q8_origin='GT_DERIVED_TRAINING_INITIALIZATION', authority=AUTHORITY)
                rows.append(row); gg.append(row)
            groups.append(dict(group_id=gid, frame_position=f, candidate_gt_list_index=c, state_index=s,
                normalized_offset=list(offset), generated_q8=[x['generated_query_xy'] for x in gg], L8=[x['L'] for x in gg],
                native8=list(range(8)), availability8=[x['generated_query_available'] for x in gg], row_ids=[x['row_id'] for x in gg]))
    assert len(rows) == 360 and len(groups) == 45 and Counter(r['source_role'] for r in rows) == {'SOURCE_FIT':240,'SOURCE_DEV':120}
    return dict(completed=True, family=FAMILY, authority=AUTHORITY, original_inputs96=list(originals.values()), original_U11_unchanged=True,
        rows360=rows, groups45=groups, exact_source_RGB12=source12, saved_metadata=scope['saved_metadata'], offsets=[list(x) for x in OFFSETS],
        all_labels_UNKNOWN=True, all_DEV_descendants_excluded_from_fit_and_normalizers=True, training_admission=False)

def load_plan_inputs():
    scope = json.loads(verified(dict(path=str(SCOPE), sha256=SCOPE_SHA)))
    meta = [json.loads(verified(b)) for b in scope['saved_metadata']]
    return scope, make_plan(scope, *meta)

def sample96(rgb, q, L):
    """F64 order00,10,01,11, zero outside/no renormalization; uint8 half-even round."""
    acc = np.zeros((96,96,3), np.float64); strict = np.ones((96,96), bool); mass = np.zeros((96,96), np.float64)
    if q is None or not all(math.isfinite(float(v)) for v in [*q,L]) or L <= 0:
        return acc.astype(np.uint8), np.zeros_like(strict), mass
    axis = (np.arange(96, dtype=np.float64)-47.5)*(float(L)/96.)
    x,y = np.meshgrid(float(q[0])+axis, float(q[1])+axis); x0,y0 = np.floor(x).astype(np.int64),np.floor(y).astype(np.int64)
    dx,dy = x-x0,y-y0; h,w = rgb.shape[:2]
    for ox,oy,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        xx,yy = x0+ox,y0+oy; ok = (xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        strict &= (weight <= 0) | ok; mass += weight*ok
        acc += (weight*ok)[...,None]*rgb[np.clip(yy,0,h-1),np.clip(xx,0,w-1)].astype(np.float64)
    return np.rint(np.clip(acc,0,255)).astype(np.uint8), strict, mass

def rle(mask):
    a = np.r_[False,np.asarray(mask,bool).reshape(-1),False].astype(np.int8); starts=np.where(np.diff(a)==1)[0]; ends=np.where(np.diff(a)==-1)[0]
    return [[int(s),int(e-s)] for s,e in zip(starts,ends)]

def fixtures(plan):
    checks=[]; rgb=np.arange(96*96*3,dtype=np.uint32).reshape(96,96,3).astype(np.uint8)
    a,v,m=sample96(rgb,[47.5,47.5],96); assert np.array_equal(a,rgb) and v.all() and (m==1).all(); checks.append('identity96_integer_centers')
    small=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3); q=[.31,5.87]; L=13.2; a,v,m=sample96(small,q,L)
    for iy,ix in [(0,0),(10,23),(47,47),(48,48),(66,60),(95,95)]:
        x=q[0]+(ix-47.5)*L/96; y=q[1]+(iy-47.5)*L/96; xx,yy=math.floor(x),math.floor(y); z=np.zeros(3); mass=0.; valid=True
        for ox,oy in [(0,0),(1,0),(0,1),(1,1)]:
            w=(x-xx if ox else 1-(x-xx))*(y-yy if oy else 1-(y-yy)); ok=0<=xx+ox<9 and 0<=yy+oy<7
            if ok: z+=w*small[yy+oy,xx+ox]; mass+=w
            if w>0 and not ok: valid=False
        assert np.array_equal(a[iy,ix],np.rint(z).astype(np.uint8)) and v[iy,ix]==valid and abs(m[iy,ix]-mass)<1e-14
    checks.append('independent_scalar_four_tap_F64_zero_padding')
    _,v,m=sample96(np.ones((2,2,3),np.uint8),[0,0],96); assert v[47,47]==False and m[47,47]==.25 and v[48,48]
    _,v,m=sample96(rgb,None,96); assert not v.any() and not m.any(); checks.append('positive_tap_strict_partial_and_unavailable_separate')
    a=np.zeros((96,96),bool); a[0,0:3]=True; a[4,2:8]=True; b=np.zeros(9216,bool)
    for start,length in rle(a): b[start:start+length]=True
    assert np.array_equal(a,b.reshape(96,96)); checks.append('row_major_true_run_RLE_roundtrip')
    assert json.loads(json.dumps(plan,allow_nan=False)) == plan
    assert len(plan['rows360'])==360 and len(plan['groups45'])==45 and all(r['training_admission'] is False for r in plan['rows360'])
    assert sum(not r['original_input']['original_query_available'] for r in plan['rows360'])==51
    checks.append('360_all_candidates_states_roles_and_original_U_provenance')
    for off in OFFSETS:
        ref=np.array([200.,100.]); L=80.; q=ref+np.array(off)*L; addr=(ref-q)/L*96+47.5
        assert np.array_equal(addr,47.5-np.array(off)*96)
    checks.append('quarter_L_moves_reference_24_samples_no_q_fit')
    return dict(completed=True,PASS=True,groups6=checks,actual_PNG_reads=0,model_reads=0)

def prepare():
    scope,plan=load_plan_inputs(); plan_b=write(PLAN,plan); code=bind(Path(__file__))
    tests=write(H/'GENERATED_INVENTED_MATH_CHECKS.json',fixtures(plan))
    proto=write(PROTO,dict(completed=True,prepared_only=True,code=code,plan=plan_b,prospective_scope=bind(SCOPE),
        actual_image_read_authorized=False, source12_before_decode='All opaque SHA/size checks first, then exact RGB PNG decode; no other binding followed.',
        sampler='F64 x/y=qgen+(index-47.5)*L/96; taps00,10,01,11; out-of-bounds zero; no renormalization; uint8 np.rint half-even.',
        validity='strict: every positive-weight tap inside source canvas; partial support mass and any-positive support reported separately. No RGB-value/depth/SAM inference.',
        generated_availability='finite q/L and continuous canvas [-.5,W-.5),[-.5,H-.5); independent of original-U and strict full-crop support.',
        RLE='row-major 96x96 true runs [start,length]; no labels inferred from validity.',
        nominal_target_separate=True, original96_immutable=True, all360_preserved=True, nominal_contacts15_only_state0=True,
        training_RGB_unmarked=True, GT_derived_q8_origin_explicit=True, no_additional_GT_features=True,
        all_labels_UNKNOWN=True, DEV_excluded_all_future_fit_and_normalizers=True, no_fit_no_model_no_admission=True,
        root_actual_scope_required_flags=['approved_PUBLIC_SOURCE_GT_QUERY_JITTER_V1','approved_exact12_derived_RGB_PNG']))
    template=write(H/'ROOT_ACTUAL_GT_QUERY_FAMILY_ENTRY_FALSE_TEMPLATE.json',dict(approved_PUBLIC_SOURCE_GT_QUERY_JITTER_V1=False,
        approved_exact12_derived_RGB_PNG=False,root_actual_scope=None,code=code,protocol=proto,plan=plan_b,prospective_scope=bind(SCOPE),
        output_directory=str(OUT),training_admission=False))
    ready=write(H/'FINAL_READY.json',dict(completed=True,PASS=True,prepared_only=True,code=code,protocol=proto,plan=plan_b,checks=tests,
        false_entry=template,rows=360,groups=45,source_PNGs12_inert=True,contacts15=True,source_PNG_stat_hash_decode_reads=0,training_admission=False))
    print(json.dumps(dict(FINAL_READY=ready,rows360=len(plan['rows360']),roles=dict(Counter(r['source_role'] for r in plan['rows360'])))))

def png_bytes(image):
    b=io.BytesIO(); image.save(b,format='PNG'); return b.getvalue()

def render(entry_path):
    from PIL import Image, ImageDraw
    e=json.loads(read(entry_path)); assert e['approved_PUBLIC_SOURCE_GT_QUERY_JITTER_V1'] is True and e['approved_exact12_derived_RGB_PNG'] is True and e['training_admission'] is False
    assert Path(e['root_actual_scope']['path']).parent==D
    actual=json.loads(verified(e['root_actual_scope'])); assert actual['approved_PUBLIC_SOURCE_GT_QUERY_JITTER_V1'] is True and actual['approved_exact12_derived_RGB_PNG'] is True
    for key,p in [('code',Path(__file__)),('protocol',PROTO),('plan',PLAN),('prospective_scope',SCOPE)]: assert e[key]==bind(p)
    assert e['output_directory']==str(OUT); plan=json.loads(verified(e['plan'])); _,expected=load_plan_inputs(); assert plan==expected
    raw12=[verified(b) for b in plan['exact_source_RGB12']]  # All opaque checks precede every image decoder.
    safe(OUT); OUT.mkdir(exist_ok=False); write(OUT/'OPAQUE_ALL12_SHA_BEFORE_DECODE.json',dict(completed=True,PASS=True,bindings=plan['exact_source_RGB12'],decoder_calls_so_far=0))
    images=[]
    for i,raw in enumerate(raw12):
        im=Image.open(io.BytesIO(raw)); assert im.format=='PNG' and im.mode=='RGB'; im.load()
        hw=next(r['source_hw'] for r in plan['rows360'] if r['frame_position']==i); assert im.size==(hw[1],hw[0]); images.append(im)
    ledger=[]; crops={}; manifest=[]
    for row in plan['rows360']:
        a,strict,mass=sample96(np.asarray(images[row['frame_position']]),row['generated_query_xy'],row['L'])
        b=write(OUT/(row['row_id']+'_RGB96.png'),png_bytes(Image.fromarray(a))); crops[row['row_id']]=a
        result=dict(row,RGB96=b,valid96_RLE=rle(strict),any_positive_source_support_RLE=rle(mass>0),
            strict_valid_fraction=float(strict.mean()),any_positive_source_support_fraction=float((mass>0).mean()),mean_source_bilinear_mass=float(mass.mean()),
            sampler='manual_F64_00_10_01_11_zero_pad_np_rint',annotation_markers_in_training_RGB=False)
        ledger.append(result); manifest.append(b)
    contacts=[]
    for group in plan['groups45']:
        if group['state_index']!=0: continue
        rr=[r for r in ledger if r['group_id']==group['group_id']]; f=group['frame_position']; canvas=Image.new('RGB',(1600,1430),'#171b21'); draw=ImageDraw.Draw(canvas)
        draw.text((20,12),f'{FAMILY} | frame {f} candidate {group["candidate_gt_list_index"]} | SOURCE nominal state 0',fill='white')
        draw.text((20,34),'New GT-derived training q8, NOT corrected original query. All visibility UNKNOWN; candidate/part uncertain.',fill='#ffcc88')
        context=images[f].copy(); cd=ImageDraw.Draw(context)
        for r in rr:
            for q,color,prefix in [(r['original_input']['query_xy'],'cyan','q'),(r['nominal_reference_xy'],'magenta','ref')]:
                if inside(q,r['source_hw']):
                    x,y=q; cd.line((x-6,y,x+6,y),fill=color,width=2); cd.line((x,y-6,x,y+6),fill=color,width=2); cd.text((x+7,y+3),prefix+str(r['native_slot']),fill=color)
        context.thumbnail((1500,680),Image.Resampling.LANCZOS); canvas.paste(context,((1600-context.width)//2,68))
        draw.text((25,765),'CONTEXT DISPLAY ONLY: old query cyan / saved nominal reference magenta; outside markers omitted. No pose/query optimization.',fill='white')
        for j,r in enumerate(rr):
            x=25+(j%4)*395; y=800+(j//4)*300; tile=Image.fromarray(crops[r['row_id']]).resize((192,192),Image.Resampling.NEAREST)
            td=ImageDraw.Draw(tile); td.line((87,95.5,104,95.5),fill='cyan',width=1); td.line((95.5,87,95.5,104),fill='cyan',width=1)
            canvas.paste(tile,(x,y+65)); draw.text((x,y),f'native {j} / canonical {r["canonical_index"]} / {r["anatomical_family"]}',fill='white')
            draw.text((x,y+20),f'new q inframe {r["generated_query_available"]}; oldU {not r["original_input"]["original_query_available"]}',fill='white')
            draw.text((x,y+40),f'strict crop-valid {r["strict_valid_fraction"]:.3f}; label UNKNOWN',fill='white')
        b=write(OUT/(group['group_id']+'_CONTACT.png'),png_bytes(canvas)); contacts.append(b); manifest.append(b)
    ledger_b=write(OUT/'ALL360_GENERATED_INPUTS.jsonl',(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in ledger)).encode())
    manifest_b=write(OUT/'IMAGE_MANIFEST.json',dict(completed=True,training_RGB360=[r['RGB96'] for r in ledger],nominal_contacts15=contacts))
    summary=write(OUT/'SUMMARY.json',dict(completed=True,PASS=True,family=FAMILY,rows=360,candidate_states=45,candidates15=True,original_frames12=True,
        original96_unchanged=True,original_U11_unchanged=True,repeated_original_U_provenance_rows=51,roles=dict(Counter(r['source_role'] for r in ledger)),
        generated_query_available=sum(r['generated_query_available'] for r in ledger),nominal_target_in_frame=sum(r['nominal_target_in_frame'] for r in ledger),
        all_strict_crop_pixels_valid_rows=sum(r['strict_valid_fraction']==1 for r in ledger),all_labels_UNKNOWN=True,training_admission=False,
        GT_derived_q8_origin=AUTHORITY,all_DEV_excluded_fit_and_normalizers=True,actual_training_model_or_forward_runs=0,original_source_payload_reads=0,
        derived_context_PNG_decodes=12,training_RGB_PNGs360_unmarked=True,contact_PNGs15_display_only=True))
    md='# 별도 GT-query SOURCE 입력\n\nGT에서 유래한 새 학습용 q8입니다. 원 initializer 96/U11과 별개이며, 원 query 개선이나 실사 성능을 측정하지 않습니다. 원 L·native alias·FIT/DEV를 계승합니다. nominal cuboid와 물리 부위 대응은 불확실하며 모든 D/E/SELF는 UNKNOWN입니다. DEV 120행은 향후 모든 fit/normalizer에서 제외합니다. pose/ref/ROI/mask는 추가 배포 feature로 주지 않습니다.\n\n원본 학습용 RGB96 360개는 marker가 없습니다. 아래 15 contact는 nominal offset(0,0)만 보여 주며 cyan crop 십자는 표시 전용입니다. 나머지 두 jitter 상태 240개도 보존했습니다. strict validity는 양수 bilinear 탭 전체 inbounds이고 partial source support와 별도입니다.\n\n[요약](SUMMARY.json) · [그림 manifest](IMAGE_MANIFEST.json)\n\n전체 360행과 validity RLE는 복원 가능한 `ALL360_GENERATED_INPUTS.jsonl`에 저장되어 있습니다.\n\n'
    md+='\n\n'.join(f'![SOURCE nominal contact {i}]({Path(b["path"]).name})' for i,b in enumerate(contacts))+'\n'
    gallery=write(OUT/'PORTABLE_GALLERY_KO.md',md.encode())
    complete=write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,entry=bind(Path(entry_path)),root_actual_scope=e['root_actual_scope'],code=e['code'],protocol=e['protocol'],plan=e['plan'],
        summary=summary,all360=ledger_b,images=manifest_b,gallery=gallery,rows=360,contacts=15,source_decodes=12,training_admission=False))
    print(json.dumps(dict(COMPLETE=complete,summary=summary,rows360=360,contacts15=15)))

if __name__=='__main__':
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True); g.add_argument('--prepare',action='store_true'); g.add_argument('--entry')
    a=p.parse_args(); prepare() if a.prepare else render(a.entry)
