"""V222 independent saved-byte/pixel QA. Entry-gated; no producer import/model/source GT."""
import argparse, collections, hashlib, io, json, os, stat
from pathlib import Path
import numpy as np
from PIL import Image

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
H = P / 'publication_root_preparation/V222_ACTUAL_PIXEL_QA_A'
SOURCE = P / 'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222/expected_part_counterfactual_pairs_v1'
PLAN = P / 'publication_root_preparation/V222_COUNTERFACTUAL_PRODUCER_C/D48_PLAN_AND_EXACT_PNG_WHITELIST_BEFORE_PIXELS.json'
PLAN_SHA = 'd9e660ef3f19748cfa8370db33297f70b7dcae09730882a8d646583773c775a5'
CODE_SHA = '056161e090c6dc90f996329cffc38f44dc446d8e255b9a56435bd42a32e10a7b'
PROTO_SHA = 'cdef217ce357db15315e651ede50d4dfd0f0f197d219081a9140193fb6cfd702'
OUT = H / 'actual_review_v1'

def need(ok, message):
    if not ok: raise AssertionError(message)

def safe(path):
    s = str(path)
    need('v112' not in s.casefold(), 'V112 lexical reject before FS')
    p = Path(s)
    need(p.is_absolute() and '..' not in p.parts and P in p.parents, 'Exact project path')
    return p

def parentfd(path):
    p = safe(path); fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = n
        return p, fd
    except BaseException:
        os.close(fd); raise

def rb(path):
    p, fd = parentfd(path)
    try: n = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally: os.close(fd)
    try:
        need(stat.S_ISREG(os.fstat(n).st_mode), 'Regular only')
        with os.fdopen(n, 'rb') as f: return f.read()
    except BaseException:
        try: os.close(n)
        except OSError: pass
        raise

def sha(data): return hashlib.sha256(data).hexdigest()
def identity(b): return str(safe(b['path'])), b['sha256']
def file_binding(p):
    b = rb(p); return {'path': str(p), 'sha256': sha(b), 'size_bytes': len(b)}
def bound(b):
    p = safe(b['path']); data = rb(p)
    need(sha(data) == b['sha256'], 'SHA ' + str(p))
    need('size_bytes' not in b or len(data) == b['size_bytes'], 'Size ' + str(p))
    return data
def metadata(b):
    p = safe(b['path'])
    need(p == PLAN or SOURCE in p.parents, 'Only pinned plan/producer-owned metadata')
    need(p.suffix in ('.json', '.jsonl'), 'Metadata suffix')
    raw = bound(b)
    return [json.loads(x) for x in raw.splitlines() if x] if p.suffix == '.jsonl' else json.loads(raw)
def dump(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    p, fd = parentfd(path)
    try: n = os.open(p.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=fd)
    finally: os.close(fd)
    with os.fdopen(n, 'wb') as f: f.write(data)
    return {'path': str(p), 'sha256': sha(data), 'size_bytes': len(data)}
def rle_mask(rle):
    result = np.zeros(9216, bool); end = 0
    for start, length in rle:
        need(type(start) is int and type(length) is int and start >= end and length > 0 and start + length <= 9216, 'Canonical RLE')
        result[start:start+length] = True; end = start + length
    return result.reshape(96, 96)
def rect_mask(box):
    x0, y0, x1, y1 = box
    need(all(type(v) is int and 0 <= v < 96 for v in box) and x0 <= x1 and y0 <= y1, 'Inclusive ROI')
    m = np.zeros((96,96), bool); m[y0:y1+1, x0:x1+1] = True; return m
def bbox(m):
    y,x = np.nonzero(m); need(len(x)>0,'Nonempty target'); return [int(x.min()),int(y.min()),int(x.max()),int(y.max())]
def sham_position(valid, target, box):
    x0,y0,x1,y1 = box; w=x1-x0+1; h=y1-y0+1
    # Independent summed-area rectangle counts, rather than producer's per-window slices.
    def sums(a):
        z=np.pad(a.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
        return z[h:,w:]-z[:-h,w:]-z[h:,:-w]+z[:-h,:-w]
    allowed = (sums(target)==0) & (sums(valid)==w*h)
    yy,xx=np.nonzero(allowed)
    candidates=[(abs(int(x)-x0)+abs(int(y)-y0),int(y),int(x)) for y,x in zip(yy,xx)]
    if not candidates: return None,0
    _,y,x=min(candidates);return [x,y,x+w-1,y+h-1],len(candidates)

def run(entry_path):
    need(Path(entry_path).parent == H, 'Entry in own exact directory')
    e=json.loads(rb(entry_path))
    need(e.get('approved_actual_saved_pixel_QA') is True and e.get('automatic_retry') is False and bool(e.get('root_authorization_message')), 'Explicit root approval')
    need(identity(e['code'])==identity(file_binding(Path(__file__))), 'Frozen QA code')
    need(e['complete']['path']==str(SOURCE/'COMPLETE.json'), 'Exact producer completion')
    complete=metadata(e['complete']);need(complete['completed'] is True and complete['PASS'] is True,'Producer completed PASS')
    need(complete['code']['sha256']==CODE_SHA and complete['protocol']['sha256']==PROTO_SHA and identity(complete['plan'])==(str(PLAN),PLAN_SHA),'Pinned production')
    plan=metadata(complete['plan']);parents=plan['parents48'];lookup={r['row_id']:r for r in parents}
    need(len(parents)==len(lookup)==48 and collections.Counter(r['source_role'] for r in parents)=={'SOURCE_FIT':39,'SOURCE_DEV':9},'Fixed48 roles')
    attempts=metadata(complete['attempts'])['attempts48'];ledger=metadata(complete['variant_rows']);manifest=metadata(complete['image_manifest'])
    need(len(attempts)==48 and len(ledger)==144 and len({v['row_id'] for v in ledger})==144,'All48/all144, unique')
    variants={(v['parent_row_id'],v['variant']):v for v in ledger}
    source_wl={identity(x):x for x in plan['exact_PNG_whitelist48']}
    need(len(source_wl)==48 and set(map(identity,manifest['source_PNG_bindings']))==set(source_wl),'Exact source48 closure')
    own_wl={identity(x):x for x in manifest['owned_PNG_bindings']}
    for p,_ in own_wl: need(SOURCE in safe(p).parents and Path(p).suffix=='.png','Owned PNG prefix')
    allowed={**source_wl,**own_wl};cache={};used={}
    def png(b,mode):
        key=identity(b);need(key in allowed,'Declared image before payload FS')
        if key not in cache: cache[key]=bound(b)
        used[key]=b
        with Image.open(io.BytesIO(cache[key])) as im:
            need(im.mode==mode and im.size==(96,96),'Exact saved96 mode/size')
            a=np.asarray(im).copy()
        need(a.dtype==np.uint8,'Original uint8')
        if mode=='L':need(np.isin(a,[0,255]).all(),'Binary alpha/valid/target');a=a==255
        return a
    pfd,fd=parentfd(OUT)
    try:os.mkdir(pfd.name,dir_fd=fd)
    finally:os.close(fd)
    try:
        checked=[];components={};source_copies=set();owned_RGB=set();alpha_count=valid_count=target_count=0
        for a in attempts:
            r=lookup[a['parent_row_id']];pid=r['row_id'];role=r['source_role']
            need(a['parent_metadata']==r and a['source_role']==role,'Full original parent metadata/role')
            need(components.setdefault(r['component_id'],role)==role,'Component FIT/DEV disjoint')
            before=png(r['RGB96_binding'],'RGB');saved_before=png(a['images']['BEFORE'],'RGB')
            need(cache[identity(r['RGB96_binding'])]==cache[identity(a['images']['BEFORE'])] and np.array_equal(before,saved_before),'Source/BEFORE encoded and pixels exact')
            source_copies.add(identity(a['images']['BEFORE']));owned_RGB.add(identity(a['images']['BEFORE']))
            valid=rle_mask(r['valid96_RLE']);target=rle_mask(r['exact_union_RLE_rowmajor96']);union=np.zeros_like(target)
            for roi in r['ROI96_by_observer'].values():union|=rect_mask(roi)
            need(np.array_equal(target,union) and np.all(valid[target]),'Exact two ROI OR and strict-valid')
            need(np.array_equal(png(a['valid96'],'L'),valid) and np.array_equal(png(a['target_union'],'L'),target),'Stored valid/target RLE exact');valid_count+=1;target_count+=1
            proof=metadata(a['proof']);need(proof['original_input']==r and proof['native_point_DE_target']==-1 and proof['training_admission'] is False,'Proof parent/authority')
            box=bbox(target);ca=rect_mask(box);sb,count=sham_position(valid,target,box)
            geometry=proof['geometry'];need(geometry['COVER_rectangle']==box and geometry['target_union_bbox']==box,'Cover rectangle')
            geometrically_possible=np.all(valid[ca]) and sb is not None
            success=False;actual={}
            if geometrically_possible:
                need(geometry['SHAM_rectangle']==sb,'Independent min-L1/Y/X SHAM geometry')
                cover=png(a['images']['COVER'],'RGB');sham=png(a['images']['SHAM'],'RGB');sa=rect_mask(sb)
                stored_ca=png(a['alpha_images']['COVER'],'L');stored_sa=png(a['alpha_images']['SHAM'],'L');alpha_count+=2
                need(np.array_equal(ca,stored_ca) and np.array_equal(sa,stored_sa),'Stored translated alpha shape')
                x0,y0,x1,y1=box;sx,sy,ex,ey=sb;cd=np.any(cover!=before,axis=2);sd=np.any(sham!=before,axis=2)
                actual={'COVER_covers_full_exact_union':bool(ca[target].all()),'SHAM_disjoint_exact_union':not bool((sa&target).any()),'both_alpha_all_strict_valid':bool(valid[ca].all() and valid[sa].all()),'equal_alpha_mass':int(ca.sum())==int(sa.sum()),'same_rectangle_shape':True,'same_actual_RGB_patch':bool(np.array_equal(cover[y0:y1+1,x0:x1+1],sham[sy:ey+1,sx:ex+1])),'SHAM_target_RGB_byte_unchanged':bool(np.array_equal(sham[target],before[target])),'COVER_outside_alpha_RGB_byte_unchanged':bool(np.array_equal(cover[~ca],before[~ca])),'SHAM_outside_alpha_RGB_byte_unchanged':bool(np.array_equal(sham[~sa],before[~sa])),'all_changes_inside_strict_valid':not bool(((cd|sd)&~valid).any()),'COVER_changed_target_pixels':int(cd[target].sum()),'SHAM_changed_target_pixels':int(sd[target].sum()),'COVER_changed_total':int(cd.sum()),'SHAM_changed_total':int(sd.sum()),'alpha_pixels':int(ca.sum()),'target_union_pixels':int(target.sum()),'SHAM_translation_xy':[sx-x0,sy-y0],'geometry_search_candidates':count,'native_point_gold_certified':False}
                need(actual==proof['proof'],'Every stored proof scalar independently exact')
                success=all(x for k,x in actual.items() if type(x)is bool and k!='native_point_gold_certified') and actual['COVER_changed_target_pixels']>0 and actual['SHAM_changed_total']>0
                owned_RGB.update(identity(a['images'][v]) for v in ('COVER','SHAM'))
            else:
                need(not a['constructed'] and set(a['images'])=={'BEFORE'},'Geometry failure preserves BEFORE only')
            need(a['constructed']==proof['constructed']==bool(success),'Independent actual success/admission')
            for variant in ('BEFORE','COVER','SHAM'):
                v=variants[(pid,variant)];label=0 if variant=='BEFORE' else (int(variant=='COVER') if success else None)
                expected={'native_slot':r['native_slot'],'query_xy':r['generated_query_xy'],'L':r['L'],'query_available':r['generated_query_available'],'q8_xy':r['q8_xy'],'q8_available':r['q8_available'],'q8_L':r['q8_L'],'q8_native_slots':list(range(8)),'condition32':r['condition32'],'valid96_RLE':r['valid96_RLE'],'component_id':r['component_id'],'source_role':role,'exact_target_union_RLE':r['exact_union_RLE_rowmajor96'],'ROI_by_observer':r['ROI96_by_observer']}
                need(all(v[k]==x for k,x in expected.items()),'Every variant same complete conditioning/ROI/role')
                need(v['weak_finite_part_target_D0_E1']==label and v['target_known']==(label is not None) and v['input_available']==(variant=='BEFORE' or bool(success)),'D/E/D or failed C/S unknown')
                need(v['native_point_DE_target']==-1 and v['training_admission']is False and v['proposed_CF_fit_eligible']==bool(success and role=='SOURCE_FIT'),'Weak authority/native-1/FIT only')
                if v['input_available']:need(identity(v['RGB96'])==identity(a['images'][variant]),'Ledger actual image identity')
                else:need(v['RGB96'] is None,'Unconstructed image unavailable')
            checked.append({'parent_row_id':pid,'source_role':role,'constructed':bool(success),**actual})
        for role,b in complete['role_manifests'].items():
            m=metadata(b);rr=[v for v in ledger if v['source_role']==role]
            need(m['source_role']==role and m['CF_variant_rows']==[v for v in rr if v['constructed']] and m['all_parent_BEFORE_rows']==[v for v in rr if v['variant']=='BEFORE'],'Role manifest exact row contents')
            need(set(map(identity,m['exact_RGB96_whitelist']))=={identity(v['RGB96']) for v in rr if v['input_available']},'Role PNG whitelist exact, no sibling leakage')
            need(m['trainer_DEV_payloads_allowed_before_two_new_states_freeze']==(role!='SOURCE_DEV'),'DEV gate metadata')
        success_count=sum(x['constructed'] for x in checked);summary=metadata(complete['summary'])
        need(summary['attempts_constructed']==complete['attempts_constructed']==success_count and summary['failed_retained']==48-success_count and complete['variant_rows144']==144,'Summary denominators')
        for b in used.values():need(sha(rb(b['path']))==b['sha256'],'Final source/owned byte rehash')
        report={'completed':True,'PASS':True,'entry':file_binding(entry_path),'code':file_binding(Path(__file__)),'producer_complete':e['complete'],'plan':complete['plan'],'parents48':48,'variant_rows144':144,'constructed':success_count,'failures_preserved':48-success_count,'roles':{'SOURCE_FIT':39,'SOURCE_DEV':9},'components_by_role':dict(collections.Counter(components.values())),'source_RGB_PNGs_decoded':48,'owned_RGB_PNGs_decoded':len(owned_RGB),'stored_alpha_PNGs_decoded':alpha_count,'valid_PNGs_decoded':valid_count,'target_PNGs_decoded':target_count,'PNG_bindings_read_and_final_rehashed':len(used),'all_same_q_L_native_q8_condition32_valid_exact':True,'all_actual_alpha_RGB_union_proofs_independently_exact':True,'all144_weak_targets_native_minus1_and_role_manifests_exact':True,'point_gold_or_training_admission':False,'original_GT_NPZ_weights_model_private_or_scores_access':0,'producer_imports':0,'results':checked,'limits_KO':['Finite expected-part weak bag를 검산했으며 exact physical native-point visibility·자연 가림·실사 전이를 인증하지 않는다.','Producer의 저장된 단일 procedural patch를 비교한 것이며 새 donor나 sampling을 만들지 않았다.','Trainer의 실제 DEV loss/norm 제외 및 paired 학습은 이 픽셀 검산 범위 밖이다.']}
        result=dump(OUT/'REVIEW.json',report);print(json.dumps({'completed':True,'PASS':True,'review':result,'constructed':success_count,'image_bindings':len(used)},ensure_ascii=False))
    except BaseException as error:
        dump(OUT/'FAILURE_PRESERVED_NO_RETRY.json',{'completed':False,'PASS':False,'error_type':type(error).__name__,'error':str(error),'automatic_retry':False});raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);run(parser.parse_args().entry)
