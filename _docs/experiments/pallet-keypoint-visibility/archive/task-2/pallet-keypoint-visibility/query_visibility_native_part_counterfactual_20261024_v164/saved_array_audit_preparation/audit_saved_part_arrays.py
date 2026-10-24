"""V164 saved-output integrity audit; never reads original RGB/donor assets.

Prepare-only artifact. Actual audit needs a root entry after terminal generation.
CPU frozen crop/paste checks and CUDA C8/C9 preprocessing, no encoder/model.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, runpy, sys, time, traceback
import numpy as np
import torch, PIL, scipy

H = Path(__file__).resolve().parent
V = H.parent
GEN = V/'generator_preparation'
PROTOCOL = H/'PROTOCOL_BEFORE_SAVED_ARRAY_BODIES.json'
OUT = V/'saved_array_audit_v1'
VARIANTS = ('BEFORE', 'COVER', 'ADJACENT_SHAM')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))
def check(b):
    p = Path(b['path']); assert sha(p) == b['sha256'], str(p)
    return p
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p, d):
    with Path(p).open('x') as f:
        json.dump(d, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
def canonical(d): return json.dumps(d, sort_keys=True, ensure_ascii=False, allow_nan=False)

def load_saved(binding, allowed_root, expected_keys=None):
    p = check(binding).resolve()
    assert p.is_relative_to(allowed_root.resolve()), str(p)
    assert p.suffix == '.npz'
    with np.load(p, allow_pickle=False) as z:
        if expected_keys is not None: assert set(z.files) == set(expected_keys), (p, z.files)
        result = {k:z[k].copy() for k in z.files}
    assert all(a.dtype.kind != 'O' for a in result.values())
    return result

def schema_and_metadata(a, fr, keys):
    h,w = fr['raw_hw']
    spec = dict(full_RGB=(np.uint8,(h,w,3)), rgb96=(np.uint8,(8,96,96,3)),
        valid96=(np.bool_,(8,96,96)), central32=(np.uint8,(8,32,32,3)),
        target224=(np.float32,(224,224,3)), target_valid=(np.bool_,(224,224)),
        fractions=(np.float32,(8,3)), query_xy=(np.float64,(8,2)), L=(np.float64,(8,)),
        native_slot=(np.int8,(8,)), crop_constructed=(np.bool_,(8,)),
        native_continuous_valid=(np.bool_,(8,)), nearest_valid=(np.bool_,(8,)),
        central4_available=(np.bool_,(8,)), context_available=(np.bool_,(8,)),
        query_available=(np.bool_,(8,)), native_visibility_target=(np.int8,(8,)),
        point_quality_target=(np.int8,(8,)), owner_target=(np.int8,(8,)),
        pasted_alpha=(np.bool_,(h,w)))
    for name in ('no_detection','finite_non_sentinel','query_RGB96_centre_domain_valid','native_grid_any_valid'):
        spec['original_'+name] = (np.bool_,(8,))
    assert set(a) == set(keys)
    assert set(spec)|{'query_id'} == set(a)
    for k,(dtype,shape) in spec.items():
        assert a[k].dtype == np.dtype(dtype) and a[k].shape == shape, (k,a[k].dtype,a[k].shape)
        if a[k].dtype.kind == 'f': assert np.isfinite(a[k]).all(), k
    assert a['query_id'].dtype.kind == 'U' and a['query_id'].shape == (8,)
    for k,dtype in [('query_xy',np.float64),('L',np.float64),('native_slot',np.int8),
            ('crop_constructed',np.bool_),('native_continuous_valid',np.bool_),('nearest_valid',np.bool_)]:
        source = 'feature_constructed' if k == 'crop_constructed' else k
        assert exact(a[k], np.asarray(fr[source],dtype)), k
    assert a['native_slot'].tolist() == list(range(8))
    assert a['query_id'].tolist() == fr['query_id']
    assert exact(a['query_available'], np.asarray(fr['native_continuous_valid'],bool))
    assert exact(a['central4_available'], np.asarray(fr['local_central4_available'],bool))
    assert exact(a['context_available'], np.asarray(fr['context_available'],bool))
    for k in ('native_visibility_target','point_quality_target','owner_target'): assert (a[k] == -1).all(), k
    for k in ('no_detection','finite_non_sentinel','query_RGB96_centre_domain_valid','native_grid_any_valid'):
        expected = np.array([r[k] for r in fr['original_query_input_metadata']],bool)
        assert exact(a['original_'+k],expected), k
    assert exact(a['central32'], a['rgb96'][:,32:64,32:64])
    assert ((a['target224'] >= 0) & (a['target224'] <= 255)).all()
    return dict(schema=True,metadata=True,all_targets_minus1=True,original_inputU_preserved=True)

def reconstructed_from_saved_RGB(a, fr, op):
    """Independent assembly using only frozen public pure sampling kernels."""
    rgb = a['full_RGB']; local = np.zeros((8,3,96,96),np.uint8)
    valid = np.zeros((8,1,96,96),bool)
    for k in range(8):
        if fr['feature_constructed'][k]:
            local[k],valid[k] = op['crop'](rgb,np.asarray(fr['query_xy'][k],np.float64),float(fr['L'][k]),96)
    box = fr['context_box_xyxy']
    if op['valid_box'](box): target,tv = op['sample_target_context'](rgb,box)
    else: target = np.zeros((224,224,3),np.float32); tv = np.zeros((224,224),bool)
    frac = np.array([[np.float32(tv.mean()),np.float32(valid[k].mean()),
        np.float32(valid[k,:,32:64,32:64].mean())] for k in range(8)],np.float32)
    expected = dict(rgb96=np.ascontiguousarray(local.transpose(0,2,3,1)),valid96=valid[:,0].copy(),
        central32=np.ascontiguousarray(local[:,:,32:64,32:64].transpose(0,2,3,1)),
        target224=target,target_valid=tv,fractions=frac,
        central4_available=valid[:,:,47:49,47:49].all((1,2,3)),context_available=np.full(8,tv.any(),bool))
    for k,value in expected.items(): assert exact(a[k],value), 'full_RGB reconstruction: '+k
    return {k:ash(a[k]) for k in expected}

def before_facts(a, fr):
    assert ash(a['full_RGB']) == fr['decoded_RGB_sha256']
    assert not a['pasted_alpha'].any()
    for k,f in enumerate(fr['facts']):
        assert ash(a['rgb96'][k]) == f['RGB96_HWC_sha256']
        assert ash(a['valid96'][k]) == f['valid96_sha256']
        assert ash(a['target224']) == f['target224_F32_HWC_sha256']
        assert ash(a['target_valid']) == f['target_valid_sha256']
    assert exact(a['fractions'],np.asarray([f['fractions'] for f in fr['facts']],np.float32))
    return dict(decoded_raw_RGB_hash=True,all8_RGB96_valid_fact_hashes=True,target_hashes=True,fractions=True)

def independently_pasted(before, cut, alpha, origin):
    """Reconstruct raw pasted RGB/mask without calling producer paste()."""
    raw = before['full_RGB'];h,w = raw.shape[:2];dh,dw = alpha.shape;x,y = map(int,origin)
    assert 0 <= x and x+dw <= w
    yy,xx = np.nonzero(alpha)
    xx = xx+x;yy = yy+y; keep = (yy>=0)&(yy<h)
    canvas = raw.copy();mask = np.zeros((h,w),bool)
    srcy,srcx = np.nonzero(alpha)
    canvas[yy[keep],xx[keep]] = cut[srcy[keep],srcx[keep]]
    mask[yy[keep],xx[keep]] = True
    return canvas,mask

def actual_preprocess17(a, op):
    local = torch.as_tensor(np.ascontiguousarray(a['rgb96'].transpose(0,3,1,2)),device='cuda')
    valid = torch.as_tensor(a['valid96'][:,None],device='cuda')
    lv,_ = op['preprocess_views'](local,valid)
    tv = op['preprocess_target'](torch.as_tensor(a['target224'].transpose(2,0,1)[None].copy(),device='cuda'),
        torch.as_tensor(a['target_valid'][None,None],device='cuda'))
    out = torch.cat((tv,lv),0).detach().cpu().contiguous().numpy()
    assert out.dtype == np.float32 and out.shape == (17,3,224,224) and np.isfinite(out).all()
    return out

def inspect_pair(r, assignment, fr, root, keys, op, lib):
    assert r['pair_id'] == assignment['pair_id'] and r['case_id'] == assignment['case_id']
    for k in ('frame_index','native_slot','donor_id','region'): assert r[k] == assignment[k], k
    assert r['source_only'] and not r['training_admitted'] and r['all_new_native_quality_owner_targets'] == -1
    folder = root/r['pair_id'];assert read(folder/'PROOF.json') == r
    a = {};details = {};declared_paths = []
    for name in VARIANTS:
        vb = r['variants'][name]
        if 'path' not in vb:
            assert vb['constructed'] is False
            assert not (folder/(name+'_INPUTS.npz')).exists()
            continue
        assert Path(vb['path']).resolve() == (folder/(name+'_INPUTS.npz')).resolve()
        a[name] = load_saved(vb,root,keys); declared_paths.append(str(Path(vb['path']).resolve()))
        details[name] = dict(binding=vb,checks=schema_and_metadata(a[name],fr,keys),
            reconstructed_arrays=reconstructed_from_saved_RGB(a[name],fr,op),
            actual_preprocess17_FP32_sha256=ash(actual_preprocess17(a[name],op)))
    cut = None
    if 'transformed_donor' in r:
        cb = r['transformed_donor'];assert Path(cb['path']).resolve() == (folder/'TRANSFORMED_DONOR.npz').resolve()
        cut = load_saved(cb,root,('rgb','alpha'))
        assert cut['rgb'].dtype == np.uint8 and cut['alpha'].dtype == bool
        assert cut['rgb'].shape == (*cut['alpha'].shape,3)
        declared_paths.append(str(Path(cb['path']).resolve()))
    else: assert not (folder/'TRANSFORMED_DONOR.npz').exists()
    if assignment['region']['region_status'] == 'UNKNOWN':
        assert not a and cut is None and not r['constructed'] and not r['PASS']
        assert r['reason'] == 'REGION_UNKNOWN_NO_GUESSED_ROI_NO_CONSTRUCTED_IMAGES'
        return dict(pair_id=r['pair_id'],integrity_PASS=True,retained_UNKNOWN=True,variants=details,
            generation_proof_PASS=False,payload_paths=declared_paths)
    if 'BEFORE' in a: details['BEFORE']['facts'] = before_facts(a['BEFORE'],fr)
    if 'before_parity' in r: assert all(r['before_parity'].values())
    if any(n in a for n in ('COVER','ADJACENT_SHAM')):
        assert 'BEFORE' in a and cut is not None and 'placement_geometry' in r
        geom = r['placement_geometry'];box = assignment['region']['original_fullRGB_bbox_xyxy']
        anchor = op['opaque_anchor'](cut['alpha']);assert anchor is not None
        x0,y0,x1,y1 = box;h,w = fr['raw_hw'];hh,ww = cut['alpha'].shape
        center = np.array([(x0+x1-1)/2.,(y0+y1-1)/2.])
        cover_origin = np.rint(center-anchor).astype(np.int64)
        direction = 1 if center[0] <= (w-1)/2 else -1
        shift = ww+(x1-x0)+4
        sham_origin = cover_origin+np.array([direction*shift,0],np.int64)
        assert geom == dict(opaque_anchor_xy=anchor.tolist(),cover_origin_xy=cover_origin.tolist(),
            sham_origin_xy=sham_origin.tolist(),horizontal_direction=direction,sham_shift_pixels=shift,transformed_hw=[hh,ww])
        for name,origin in [('COVER',cover_origin),('ADJACENT_SHAM',sham_origin)]:
            if name not in a: continue
            rgb,mask = independently_pasted(a['BEFORE'],cut['rgb'],cut['alpha'],origin)
            assert exact(a[name]['full_RGB'],rgb) and exact(a[name]['pasted_alpha'],mask), name
            details[name]['pasted_RGB_and_alpha_exact_from_saved_donor'] = True
        if all(n in a for n in VARIANTS):
            ca = a['COVER']['pasted_alpha'];sa = a['ADJACENT_SHAM']['pasted_alpha']
            independent_region = dict(cover_alpha1=bool(ca[y0:y1,x0:x1].all()),
                sham_alpha0=bool(not sa[y0:y1,x0:x1].any()),equal_positive_mass=int(ca.sum())==int(sa.sum())>0)
            # A failed criterion is retained as false; agreement with the saved proof is the integrity gate.
            if 'proof' in r:
                assert independent_region['cover_alpha1'] == r['proof']['cover_alpha1_every_supported_box_pixel']
                assert independent_region['sham_alpha0'] == r['proof']['sham_alpha0_every_supported_box_pixel']
                assert independent_region['equal_positive_mass'] == r['proof']['cover_sham_equal_positive_pasted_alpha_mass']
                proof,detail = lib['prove_pair'](a['BEFORE'],a['COVER'],a['ADJACENT_SHAM'],ca,sa,box,r['native_slot'],op,'cuda')
                assert canonical(proof) == canonical(r['proof']), r['pair_id']+' proof'
                assert canonical(detail) == canonical(r['proof_detail']), r['pair_id']+' detail'
                assert r['PASS'] == all(bool(x) for x in proof.values())
                pp = {n:details[n]['actual_preprocess17_FP32_sha256'] for n in VARIANTS}
                assert pp == r['proof_detail']['preprocess17_FP32_sha256']
                details['recomputed_generation_proof'] = dict(criterion_values=independent_region,
                    all_original_proof_fields_equal=True,all_proof_detail_equal=True,actual_preprocess17_hashes=pp)
            else:
                assert not r['PASS'] and not r['constructed'] and 'traceback' in r
                details['recomputed_generation_proof'] = dict(criterion_values=independent_region,
                    generation_proof_absent_due_retained_exception=True,
                    actual_preprocess17_hashes={n:details[n]['actual_preprocess17_FP32_sha256'] for n in VARIANTS})
    if r['PASS']: assert r['constructed'] and all(n in a for n in VARIANTS) and 'proof' in r
    return dict(pair_id=r['pair_id'],integrity_PASS=True,retained_UNKNOWN=False,variants=details,
        generation_proof_PASS=r['PASS'],generation_reason=r['reason'],payload_paths=declared_paths)

def run(entry_path):
    e = read(entry_path)
    assert e['approved_saved_array_audit'] is True and e['source_only'] is True
    assert e['generation_terminal_exit_code'] == 0
    assert e['original_source_or_donor_pixels_allowed'] is False and e['model_encoder_fit_allowed'] is False
    assert e['protocol'] == bind(PROTOCOL)
    p = read(check(e['protocol']));assert p['code'] == bind(__file__) == e['code']
    trigger = read(check(e['root_completion_trigger']));assert trigger['completed'] is True
    gp = read(check(p['generator_protocol']));assert e['generator_protocol'] == p['generator_protocol']
    actual = dict(executable=sys.executable,numpy=np.__version__,torch=torch.__version__,Pillow=PIL.__version__,scipy=scipy.__version__)
    assert actual == gp['runtime'], dict(actual=actual,expected=gp['runtime'])
    assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    complete_path = check(e['source_generation_complete']);complete = read(complete_path)
    root = Path(gp['output']).resolve()
    assert complete_path.resolve() == root/'COMPLETE.json'
    assert complete['completed'] and complete['all66_attempts_retained']
    assert complete['protocol'] == p['generator_protocol'] and complete['inputs'] == gp['inputs']
    assert complete['proofs'] == e['all66_manifest']
    manifest = rows(check(e['all66_manifest']));assert len(manifest) == 66
    assert complete['passed_pixel_input_proofs'] == sum(r['PASS'] for r in manifest)
    assert complete['failed_or_UNKNOWN_attempts_retained'] == sum(not r['PASS'] for r in manifest)
    assert complete['PASS'] == all(r['PASS'] for r in manifest)
    assert complete['fixed_UNKNOWN_region_attempts'] == 15 and complete['supported_region_attempts'] == 51
    inputs = read(check(gp['inputs']));assignments = inputs['attempts'];frames = {f['frame_index']:f for f in inputs['frames']}
    assert [r['pair_id'] for r in manifest] == [a['pair_id'] for a in assignments]
    assert len({r['pair_id'] for r in manifest}) == 66
    assert all(f['split']=='SOURCE_DEVELOPMENT' for f in frames.values())
    assert sum(a['region']['region_status']=='UNKNOWN' for a in assignments) == 15
    for k in ('code','design','generated_checks','safe_helper','local_crop','context_crop','C8_pure_RGB_preprocess','C9_pure_RGB_preprocess'):
        check(gp[k])
    lib = runpy.run_path(str(check(gp['code'])),run_name='V164_SAVED_AUDIT_PURE_LIBRARY_NO_MAIN')
    op = lib['operators']();keys = gp['output_schema']['NPZ_keys']
    OUT.mkdir(exist_ok=False);write(OUT/'STARTED.json',dict(pid=os.getpid(),entry=bind(entry_path),protocol=bind(PROTOCOL)))
    results = [];started = time.monotonic()
    try:
        for r,a in zip(manifest,assignments):
            try: result = inspect_pair(r,a,frames[a['frame_index']],root,keys,op,lib)
            except Exception:
                result = dict(pair_id=r['pair_id'],integrity_PASS=False,traceback=traceback.format_exc(),generation_proof_PASS=r['PASS'])
            results.append(result)
            if len(results)%6 == 0: print(json.dumps(dict(audited=len(results),integrity_passed=sum(x['integrity_PASS'] for x in results))),flush=True)
        observed_paths = {str(x.resolve()) for x in root.glob('*/*.npz')}
        declared_paths = {x for r in results for x in r.get('payload_paths',[])}
        all_payloads_accounted = observed_paths == declared_paths
        passed = all(r['integrity_PASS'] for r in results) and all_payloads_accounted
        record = OUT/'ALL66_SAVED_ARRAY_AUDIT.jsonl'
        with record.open('x') as f:
            for r in results: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        summary = dict(completed=True,PASS=passed,integrity_PASS_not_all_generation_criteria_pass=True,
            attempts=66,integrity_passed=sum(r['integrity_PASS'] for r in results),
            generation_passed=complete['passed_pixel_input_proofs'],generation_failed_or_UNKNOWN=complete['failed_or_UNKNOWN_attempts_retained'],
            fixed_region_UNKNOWN_attempts=15,all_saved_NPZ_paths_accounted=all_payloads_accounted,
            saved_NPZ_files=len(observed_paths),audit_records=bind(record),entry=bind(entry_path),protocol=bind(PROTOCOL),
            source_generation_complete=e['source_generation_complete'],all66_manifest=e['all66_manifest'],
            source_only=True,original_source_RGB_donor_compact_or_masks_read=0,model_encoder_weight_feature_fit_PnP_calls=0,
            new_native_quality_owner_targets=-1,training_admitted=False,original_labels_or_generation_outputs_modified=False,
            actual_runtime=actual,seconds=time.monotonic()-started)
        write(OUT/'COMPLETE.json',summary);print(json.dumps(dict(completed=True,PASS=passed,complete=bind(OUT/'COMPLETE.json'))),flush=True)
        if not passed: raise SystemExit(2)
    except BaseException:
        if not (OUT/'COMPLETE.json').exists():
            write(OUT/'FAILED_EXECUTION.json',dict(completed=False,attempts_audited=len(results),traceback=traceback.format_exc()))
        raise

if __name__ == '__main__':
    ap = argparse.ArgumentParser();ap.add_argument('--entry',required=True,type=Path)
    run(ap.parse_args().entry)
