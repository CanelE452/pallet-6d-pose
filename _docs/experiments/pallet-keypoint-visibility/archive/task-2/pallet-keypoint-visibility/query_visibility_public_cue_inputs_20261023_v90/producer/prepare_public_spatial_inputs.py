"""V90: public RGB-only full P3 + fixed tile input producer.

generated/freeze do not decode source pixels, construct YOLO, or load weights.
run requires a separately root-approved entry binding the reviewed protocol.
Never imports V65 current inference or any semantic/GT/pose/depth/SAM module.
"""
from pathlib import Path
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P = ROOT.parent
V89 = P / 'query_visibility_public_cue_grid_20261023_v89'
FRAMES = V89 / 'inputs_metadata/FRAMES_INPUTS_ONLY.jsonl'
TILES = V89 / 'inputs_metadata/INPUTS_GRID_ONLY.jsonl'
GRID_SEAL = V89 / 'inputs_metadata/INPUT_GRID_SEAL_BEFORE_GT_JOIN.json'
AUTHORITY = P / 'query_visibility_controlled_real_20261022_v66/pallox_features/FEATURE_PROTOCOL.json'
ROI_CODE = P.parent / 'external_ray_proxy_head_20261022_v60/head_and_roi.py'
CROP_CODE = P / 'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
DENY = P / 'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json'
MATH = HERE / 'spatial_input_math.py'
PROTOCOL = ROOT / 'INPUT_PRODUCER_PROTOCOL_BEFORE_PIXELS.json'
GENERATED = HERE / 'GENERATED_SPATIAL_INPUT_TESTS.json'
LEGACY_PYTHON = '/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
CANONICAL_PYTHON = '/home/minjae/anaconda3/bin/python'
FRAME_KEYS = {'frame_input_id','image','image_sha256','raw_hw','pair_id','view','split','source_family','session','seed_V84_reuse'}
TILE_KEYS = FRAME_KEYS | {'tile_id','query_xy','roi_side','window_xyxy_half_open','native_valid_pixels','padding_pixels','native_valid_fraction'}
BACKBONE_CODE_SHA = '296d81b2adde5ce74192f0012e554ae546c95f309a82dd808024ee34105721ab'
WEIGHT_SHA = '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
FRAMES_SHA = 'a7691362fe47824cb9ecd66f41e6e030c94fae9985b8580618b5dc225d185b11'
TILES_SHA = '89b06e032057b620fed8c30e226439082fdaccf116c488f1461d936ae5512960'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def check(b):
    if sha(b['path']) != b['sha256']:
        raise ValueError('Frozen binding changed: ' + b['path'])


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def pure(path, names):
    nodes = [n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    env = dict(np=np, torch=torch, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def assets():
    a = load(AUTHORITY)['assets']
    result = {k: a[k] for k in ('backbone_code','backbone_checkpoint')}
    assert result['backbone_code']['sha256'] == BACKBONE_CODE_SHA
    assert result['backbone_checkpoint']['sha256'] == WEIGHT_SHA
    for b in result.values():
        check(b)
    return result


def input_metadata():
    assert sha(FRAMES) == FRAMES_SHA and sha(TILES) == TILES_SHA
    seal = load(GRID_SEAL)
    assert seal['completed'] and seal['PASS']
    for path in (FRAMES, TILES):
        b = next(b for b in seal['files'] if Path(b['path']) == path)
        check(b)
    frames = [json.loads(s) for s in FRAMES.read_text().splitlines()]
    tiles = [json.loads(s) for s in TILES.read_text().splitlines()]
    assert len(frames) == 100 and len(tiles) == 12900
    assert all(set(r) == FRAME_KEYS for r in frames)
    assert all(set(r) == TILE_KEYS for r in tiles)
    assert len({r['frame_input_id'] for r in frames}) == 100
    assert len({r['tile_id'] for r in tiles}) == 12900
    assert Counter(r['split'] for r in frames) == {'TRAIN_LAB':44, 'VAL_HALL':56}
    assert Counter(r['roi_side'] for r in tiles) == {128:9900, 256:3000}
    denied = set(load(DENY)['image_sha256'])
    frame_index = {r['frame_input_id']: i for i, r in enumerate(frames)}
    base = P / 'query_visibility_controlled_real_20261022_v66/public_pallet/data'
    by_frame = [[] for _ in frames]
    for r in frames:
        assert r['image_sha256'] not in denied and r['raw_hw'] == [480,640]
        assert Path(r['image']).resolve().is_relative_to(base.resolve()) and Path(r['image']).suffix == '.png'
    for i, r in enumerate(tiles):
        fi = frame_index[r['frame_input_id']]
        assert all(r[k] == frames[fi][k] for k in FRAME_KEYS)
        q = np.asarray(r['query_xy'], np.float64)
        assert q.shape == (2,) and np.isfinite(q).all()
        L = float(r['roi_side'])
        assert np.array_equal(np.asarray(r['window_xyxy_half_open']), [q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2])
        by_frame[fi].append((i, r))
    assert all(len(x) == 129 for x in by_frame)
    assert [i for batch in by_frame for i, _ in batch] == list(range(12900))
    return frames, tiles, by_frame


def generated():
    """Analytic/generated tensors only. No source input metadata required."""
    torch.set_num_threads(1)
    math = module(MATH, '_v90_generated_math')
    old = pure(ROI_CODE, ['native_grid','patches','roi_preflight'])
    oldcrop = pure(CROP_CODE, ['crop'])['crop']
    assert ast.dump(next(n for n in ast.parse(MATH.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'crop'), include_attributes=False) == ast.dump(next(n for n in ast.parse(CROP_CODE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'crop'), include_attributes=False)
    rng = np.random.default_rng(20261023)
    rgb_checks = 0
    for hw in [(31,47),(480,640),(540,960)]:
        rgb = rng.integers(0,256,(*hw,3),np.uint8)
        for q in [[0.,0.],[13.3,20.2],[hw[1]-1.,hw[0]-1.],[-.5,-.5],[hw[1]-.5,hw[0]-.5]]:
            for L in (16.,128.,256.):
                a, av = math.crop(rgb,q,L,96)
                b, bv = oldcrop(rgb,q,L,96)
                assert np.array_equal(a,b) and np.array_equal(av,bv)
                assert a.dtype == np.uint8 and av.dtype == bool
                rgb_checks += 96 * 96
    affine = math.expected_affine([480,640])
    assert affine['canvas_shape'] == [680,840] and affine['input_shape'] == [544,640]
    assert affine['p3_actual_hw'] == [68,80] and np.array_equal(affine['offset'],[0.,13.])
    p3 = rng.normal(size=(64,68,80)).astype(np.float16)
    padded = np.pad(p3,((0,0),(0,12),(0,0)))
    q8 = np.array([[100.,100.],[400.,100.],[200.,150.],[250.,250.],[300.,300.],[400.,400.],[150.,200.],[120.,130.]])
    grid, valid, L = old['native_grid'](q8,[480,640])
    own_grid, own_valid = math.native_grid_explicit(q8,L,[480,640])
    assert np.array_equal(grid,own_grid) and np.array_equal(valid,own_valid)
    x, v, _ = math.sample_p3(p3,q8,L,[480,640],affine['gain'],affine['offset'],100)
    expected, ev, el = old['patches'](dict(p3=torch.from_numpy(padded)[None],added_border=100),q8,[480,640],affine['gain'],affine['offset'])
    assert el == L and np.array_equal(x,expected.numpy()) and np.array_equal(v,ev)
    borderq = np.array([[0.,0.],[639.,479.],[320.,240.],[-1.,-1.]])
    masks = []
    for L in (128.,256.):
        x,v,valid = math.sample_p3(p3,borderq,L,[480,640],affine['gain'],affine['offset'],100)
        assert np.array_equal(x[:,64].astype(bool),valid)
        assert not x[:,:64][~np.broadcast_to(valid[:,None],x[:,:64].shape)].any()
        assert not v[-1] and not x[-1].any()
        g,_ = math.native_grid_explicit(borderq,L,[480,640])
        assert np.array_equal(g[:,4,4],borderq)
        assert g[0,0,0,0] == ((.5/9)-.5)*L
        masks.append(int(valid.sum()))
    # Analytic x/y ramp sampled by the same F32 grid, with independently-built indices.
    y,xr = torch.meshgrid(torch.arange(80,dtype=torch.float64),torch.arange(80,dtype=torch.float64),indexing='ij')
    ramp = torch.stack([xr,y])[None]
    query = np.array([[300.,220.]])
    g,_ = math.native_grid_explicit(query,128.,[480,640])
    net = (g+100)*affine['gain']+affine['offset']
    index = net/8.
    sample = F.grid_sample(ramp,torch.from_numpy(2*(index+.5)/80-1),align_corners=False)
    ramp_error = float(np.max(np.abs(sample.numpy().transpose(0,2,3,1)-index)))
    assert ramp_error < 1e-10
    back = (net-affine['offset'])/affine['gain']-100
    assert np.max(np.abs(back-g)) < 1e-10
    valid,continuous,nearest = math.query_validity([[0,0],[639,479],[639.8,479],[-1,-1]],[480,640])
    assert valid.tolist() == [True,True,False,False] and continuous[2] and not nearest[2]
    write(GENERATED, dict(completed=True,PASS=True,code=bind(__file__),math_code=bind(MATH),
          old_ROI_code=bind(ROI_CODE),old_crop_code=bind(CROP_CODE),crop_AST_identical=True,
          generated_crop_pixels=rgb_checks,V60_same_q8_same_L_ROI_bitexact=True,
          explicit_L128_256_edges_and_mask=True,grid9_center_exact_q=True,
          grid96_q_at_index47_5_no_exact_center_sample=True,
          P3_actual_shape=[64,68,80],temporary_sampling_canvas=[64,80,80],
          analytic_coordinate_ramp_max_error=ramp_error,V60_preflight=old['roi_preflight'](),
          source_RGB_metadata_P3_weights_GT_reads=0,model_construct_forward_fit_current_PnP=0,
          runtime=dict(executable=sys.executable,torch=str(torch.__version__),numpy=np.__version__)))
    print(json.dumps(bind(GENERATED)),flush=True)


def freeze():
    frames,tiles,_ = input_metadata()
    proof = load(GENERATED)
    assert proof['PASS'] and proof['code'] == bind(__file__) and proof['math_code'] == bind(MATH)
    a = assets()
    protocol = dict(schema='V90_PUBLIC_FIXED_GRID_SPATIAL_INPUT_PROTOCOL_V1',frozen_before_source_pixels=True,
        code=bind(__file__),math_code=bind(MATH),generated=bind(GENERATED),
        bindings=[bind(p) for p in (GRID_SEAL,FRAMES,TILES,V89/'BEFORE_GRID_PROTOCOL.json',AUTHORITY,ROI_CODE,CROP_CODE,DENY)],
        assets=a,frames=100,tiles=12900,frames_by_split=dict(Counter(r['split'] for r in frames)),
        tile_order_and_q_L_split_exact_V89=True,frame_schema=sorted(FRAME_KEYS),tile_metadata_schema=sorted(TILE_KEYS),
        RGB=dict(native_unpadded=True,read_only_png=True,legacy_PIL_and_cv2_all100_bitexact=True,
                 first_canonical_decoder_python=CANONICAL_PYTHON,first_canonical_Pillow='12.0.0',
                 first_canonical_vs_legacy_pixels_bitexact_required=True,color_to_backbone='nativeRGB -> contiguousBGR',
                 crop96='Exact V74 crop AST: q+((j+.5)/96-.5)*L, nativepixel bilinear/constant0, floor(clip(value)+.5); valid[-.5,W-.5)x[-.5,H-.5). No resize/recenter.',
                 align_corners=False,grid96_q_index=47.5),
        P3=dict(producer='Exact frozen V63 FrozenYoloFeatures.predict(already_padded=False,cpu_features=True)',
                added_border=100,border='OpenCV BORDER_REFLECT_101 exactly once',
                actual_shape='64 x captured.inputH/8 x captured.inputW/8; slice wrapper padded80 back to actual; store unpadded FP16 only',
                sampling_canvas='Temporary RAM right/bottom zero pad to80 solely to reproduce V60 numerical path; never archived as real feature',
                affine='network=(native+100)*gain+integeroffset; gain=min(640/canvasHW); actual dynamic rect input shape/stride8 asserted',
                native9='q+((j+.5)/9-.5)*EXPLICIT_V89_L; no q8-span L recomputation; float64 arithmetic then F32 grid',
                ROI='P3 F16->CPUF32, network/8, norm=2*(index+.5)/80-1; bilinear zeros align_corners=False; native[0,W)x[0,H) validity65th channel; F16save',
                valid_sample_support_inside_actual_P3_required=True,exact_RF_center_claim=False),
        runtime=dict(executable=LEGACY_PYTHON,Python_major_minor=[3,10],torch='2.1.1+cu118',cv2='4.9.0',
                     device='cuda:0',threads=1,FP32_backbone=True,autocast=False,TF32=False,cudnn_benchmark=False),
        execution=dict(unique_source_predicts=100,fixed_first_source_repeat_predicts=1,generated_blank_warmup_predicts=1,
                       internal_lazy_warmup_forwards_logged_separately=True,model_state_SHA_after_blank_warmup_and_after_source_equal=True,
                       first_P3_actual_and_affine_repeat_bitexact=True,checkpoint_file_before_after_SHA_equal=True),
        output=dict(directory=str(ROOT/'input_pack'),full_P3_directory='full_p3',full_P3_count=100,
                    chunks='chunks/tiles_START_END.npz',max_tiles_per_chunk=128,
                    tensor_fields=['p3_roi','rgb96','crop_valid96','query_xy','L','ROI_anyvalid','ROI_center_valid','query_center_valid','query_continuous_valid','query_nearest_valid','image_frame_index','tile_index','tile_id','split'],
                    P3_ROI_shape=[65,9,9],RGB_shape=[3,96,96],valid96_shape=[1,96,96],
                    byte_budget=500000000,GPU_peak_budget=10000000000,
                    frameRGB128_target_context42_SAM_native8_semantic_ID=False),
        no_GT_mask_label_cue_reference_pose_depth_reads=True,no_current_data=True,no_head_forward_or_training=True,no_PnP=True,
        training_admission=False)
    write(PROTOCOL,protocol)
    template=dict(schema='V90_ROOT_APPROVED_SOURCE_INPUT_ENTRY_V1',approved_by_root=False,
        protocol=bind(PROTOCOL),code=bind(__file__),math_code=bind(MATH),generated=bind(GENERATED),
        independent_review={'path':'ROOT_TO_BIND_REVIEW_PATH','sha256':'ROOT_TO_BIND_REVIEW_SHA'},
        source_only=True,current_allowed=False,training_allowed=False,PnP_allowed=False)
    write(ROOT/'SOURCE_INPUT_ENTRYLOCK_TEMPLATE.json',template)
    print(json.dumps(dict(protocol=bind(PROTOCOL),template=bind(ROOT/'SOURCE_INPUT_ENTRYLOCK_TEMPLATE.json')),indent=2),flush=True)


def authorize(entry_path):
    e = load(entry_path)
    assert e['schema'] == 'V90_ROOT_APPROVED_SOURCE_INPUT_ENTRY_V1' and e['approved_by_root'] is True
    assert e['source_only'] is True and e['current_allowed'] is False and e['training_allowed'] is False and e['PnP_allowed'] is False
    for k in ('protocol','code','math_code','generated','independent_review'):
        check(e[k])
    assert Path(e['protocol']['path']) == PROTOCOL and e['code'] == bind(__file__) and e['math_code'] == bind(MATH)
    p = load(PROTOCOL)
    for k in ('code','math_code','generated'):
        check(p[k])
    for b in p['bindings']:
        check(b)
    for b in p['assets'].values():
        check(b)
    assert load(e['independent_review']['path'])['PASS'] is True
    assert load(p['generated']['path'])['PASS'] is True
    return e,p


def state_sha(model):
    h = hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(k.encode('utf8')+b'\0'+str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
    return h.hexdigest()


def canonical_first_RGB_sha(image):
    code = "import sys,json,hashlib,numpy as np,PIL;from PIL import Image;assert PIL.__version__=='12.0.0';a=np.array(Image.open(sys.argv[1]).convert('RGB'));print(json.dumps({'RGB_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'raw_hw':list(a.shape[:2]),'Pillow':PIL.__version__,'executable':sys.executable}))"
    return json.loads(subprocess.check_output([CANONICAL_PYTHON,'-c',code,image],text=True))


def run(entry_path):
    # All immutable review/protocol checks precede source metadata/model imports/pixels.
    entry,prot = authorize(entry_path)
    import cv2
    from PIL import Image
    import PIL
    rt = prot['runtime']
    assert Path(sys.executable).resolve() == Path(rt['executable']).resolve()
    assert list(sys.version_info[:2]) == rt['Python_major_minor'] and str(torch.__version__) == rt['torch'] and cv2.__version__ == rt['cv2']
    frames,tiles,by_frame = input_metadata()
    math = module(MATH,'_v90_run_math')
    out = Path(prot['output']['directory'])
    out.mkdir(exist_ok=False)
    (out/'chunks').mkdir();(out/'full_p3').mkdir()
    write(out/'STARTED.json',dict(entry=bind(entry_path),protocol=bind(PROTOCOL),source_frames=100,tiles=12900,source_pixel_reads_before_entry=0))
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'runtime/ultralytics')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a = prot['assets'];backmod = module(a['backbone_code']['path'],'_v90_bound_frozen_yolo')
    backbone = backmod.FrozenYoloFeatures(a['backbone_checkpoint']['path'],device=rt['device'])
    internal = [0]
    hook = backbone.yolo.model.register_forward_pre_hook(lambda *_:internal.__setitem__(0,internal[0]+1))
    # Generated blank warmup only: settles official fusion/lazy runtime before state freeze.
    backbone.predict(np.zeros((480,640,3),np.uint8),already_padded=False,cpu_features=True)
    warmup_internal = internal[0]
    state_before = state_sha(backbone.yolo.model)
    write(out/'WARMUP_AND_FROZEN_STATE.json',dict(generated_blank_predict_calls=1,internal_forwards=warmup_internal,
          state_after_warmup_SHA256=state_before,model_frozen=not any(p.requires_grad for p in backbone.yolo.model.parameters()),
          source_forward_calls=0,checkpoint=a['backbone_checkpoint']))
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();batch=[];chunks=[];frame_records=[];lineage=[]
    source_predicts=0;repeat_predicts=0;firstproof=None

    def disk_guard():
        n=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if n > prot['output']['byte_budget']:
            raise RuntimeError('Frozen disk budget exceeded; preserve partial outputs, do not change inputs')
        if torch.cuda.max_memory_allocated() > prot['output']['GPU_peak_budget']:
            raise RuntimeError('Frozen GPU budget exceeded')
        return n

    def flush():
        nonlocal batch
        if not batch:return
        begin=int(batch[0]['tile_index']);end=int(batch[-1]['tile_index'])+1
        arrays={k:np.stack([r[k] for r in batch]) for k in batch[0]}
        assert len(batch)<=128 and end-begin==len(batch)
        path=out/'chunks'/f'tiles_{begin:06d}_{end:06d}.npz'
        np.savez_compressed(path,**arrays)
        chunks.append(dict(start=begin,end=end,tiles=len(batch),file=bind(path),
                           schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()}))
        batch=[];disk_guard()

    with torch.inference_mode():
        for fi,fr in enumerate(frames):
            assert sha(fr['image']) == fr['image_sha256']
            rgb=np.array(Image.open(fr['image']).convert('RGB'))
            assert list(rgb.shape[:2]) == fr['raw_hw'] and rgb.dtype == np.uint8
            data=np.fromfile(fr['image'],dtype=np.uint8)
            cv_rgb=cv2.cvtColor(cv2.imdecode(data,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            assert np.array_equal(rgb,cv_rgb), 'Lossless source PNG decoder disagreement'
            rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest()
            if fi == 0:
                firstproof=canonical_first_RGB_sha(fr['image'])
                assert firstproof['RGB_sha256'] == rgbsha and firstproof['raw_hw'] == fr['raw_hw']
            captured=backbone.predict(np.ascontiguousarray(rgb[...,::-1]),already_padded=False,cpu_features=True)
            source_predicts+=1
            affine=math.expected_affine(fr['raw_hw'],100)
            gain,offset=backmod.canvas_affine(captured['canvas_shape'],captured['input_shape'])
            assert captured['added_border']==100 and list(captured['canvas_shape'])==affine['canvas_shape'] and list(captured['input_shape'])==affine['input_shape']
            assert gain==affine['gain'] and np.array_equal(offset,affine['offset'])
            h3,w3=affine['p3_actual_hw'];padded=captured['p3'].numpy()
            assert padded.shape==(64,80,80) and padded.dtype==np.float16
            assert not padded[:,h3:,:].any() and not padded[:,:,w3:].any()
            p3=np.ascontiguousarray(padded[:,:h3,:w3])
            assert p3.shape==(64,captured['input_shape'][0]//8,captured['input_shape'][1]//8) and np.isfinite(p3).all()
            if fi==0:
                repeat=backbone.predict(np.ascontiguousarray(rgb[...,::-1]),already_padded=False,cpu_features=True)
                repeat_predicts+=1
                assert torch.equal(captured['p3'],repeat['p3'])
                for key in ('added_border','canvas_shape','input_shape'):assert captured[key]==repeat[key]
                firstproof.update(first_source_P3_repeat_bitexact=True,first_source_affine_repeat_bitexact=True,
                                  P3_actual_values_SHA256=hashlib.sha256(p3.tobytes()).hexdigest())
            pp=out/'full_p3'/f'frame_{fi:04d}.npz'
            np.savez_compressed(pp,p3=p3,original_hw=np.asarray(fr['raw_hw'],np.int32),
                added_border=np.asarray(100,np.int32),canvas_shape=np.asarray(captured['canvas_shape'],np.int32),
                input_shape=np.asarray(captured['input_shape'],np.int32),gain=np.asarray(gain,np.float64),
                offset=np.asarray(offset,np.float64),p3_actual_hw=np.array([h3,w3],np.int32),
                image_sha256=np.array(fr['image_sha256']),decoded_RGB_sha256=np.array(rgbsha),
                backbone_sha256=np.array(WEIGHT_SHA),frame_input_id=np.array(fr['frame_input_id']))
            frame_records.append(dict(frame_index=fi,**fr,full_P3=bind(pp),P3_shape=list(p3.shape),P3_dtype='float16',
                actual_input_shape=list(captured['input_shape']),canvas_shape=list(captured['canvas_shape']),
                added_border=100,gain=gain,offset=offset.tolist(),decoded_RGB_sha256=rgbsha,
                original_bytes_SHA_and_PIL_cv2_pixels_exact=True,P3_zero_padding_archived=False))
            group=by_frame[fi];q=np.array([r['query_xy'] for _,r in group],np.float64);L=np.array([r['roi_side'] for _,r in group],np.float64)
            x,anyvalid,valid9=math.sample_p3(p3,q,L,fr['raw_hw'],gain,offset,100)
            center,continuous,nearest=math.query_validity(q,fr['raw_hw'])
            assert np.array_equal(x[:,64].astype(bool),valid9)
            for j,(ti,tr) in enumerate(group):
                rgb96,v96=math.crop(rgb,q[j],float(L[j]),96)
                one=dict(p3_roi=x[j],rgb96=rgb96,crop_valid96=v96,query_xy=q[j],L=L[j],
                    ROI_anyvalid=np.asarray(anyvalid[j],bool),ROI_center_valid=np.asarray(valid9[j,4,4],bool),
                    query_center_valid=np.asarray(center[j],bool),query_continuous_valid=np.asarray(continuous[j],bool),
                    query_nearest_valid=np.asarray(nearest[j],bool),image_frame_index=np.asarray(fi,np.int32),
                    tile_index=np.asarray(ti,np.int32),tile_id=np.array(tr['tile_id']),split=np.array(tr['split']))
                batch.append(one)
                lineage.append(dict(tile_index=ti,tile_id=tr['tile_id'],frame_index=fi,frame_input_id=fr['frame_input_id'],
                    image_sha256=fr['image_sha256'],split=tr['split'],query_xy=tr['query_xy'],L=tr['roi_side'],
                    RGB96_values_SHA256=hashlib.sha256(rgb96.tobytes()).hexdigest(),
                    valid96_values_SHA256=hashlib.sha256(v96.tobytes()).hexdigest(),
                    P3_ROI_values_SHA256=hashlib.sha256(x[j].tobytes()).hexdigest()))
                if len(batch)==128:flush()
            disk_guard()
            if (fi+1)%10==0:print(json.dumps(dict(frames=fi+1,total_frames=100,tiles=(fi+1)*129,seconds=time.monotonic()-start,disk_bytes=disk_guard())),flush=True)
    flush()
    state_after=state_sha(backbone.yolo.model)
    assert state_after==state_before and not any(p.requires_grad for p in backbone.yolo.model.parameters())
    assert not any(m.training for m in backbone.yolo.model.modules())
    assert source_predicts==100 and repeat_predicts==1 and len(frame_records)==100 and len(lineage)==12900
    for b in (prot['code'],prot['math_code'],prot['assets']['backbone_checkpoint']):check(b)
    assert sha(FRAMES)==FRAMES_SHA and sha(TILES)==TILES_SHA
    lin=out/'TILE_INPUT_LINEAGE.jsonl';lin.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in lineage))
    frpath=out/'FRAME_FULL_P3_RECORDS.jsonl';frpath.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in frame_records))
    total=disk_guard()
    runtime=dict(**rt,Pillow=PIL.__version__,numpy=np.__version__,actual_python=sys.version)
    manifest=dict(completed=True,frame_count=100,tile_count=12900,protocol=bind(PROTOCOL),entry=bind(entry_path),
        source_frames=bind(FRAMES),source_tiles=bind(TILES),frame_records=bind(frpath),lineage=bind(lin),chunks=chunks,
        full_P3_files=[r['full_P3'] for r in frame_records],runtime=runtime,source_predict_calls=source_predicts,
        first_source_repeat_calls=repeat_predicts,generated_blank_warmup_calls=1,
        internal_forward_calls=internal[0],internal_warmup_forwards=warmup_internal,
        first_replay_and_decoder_proof=firstproof,model_state_before_SHA256=state_before,model_state_after_SHA256=state_after,
        original_grid_order_q_L_split_retained=True,no_old8query_ROIs_reused=True,all_nativevalid_ROI_support_inside_actual_featureplane=True,
        GT_mask_label_cue_reference_pose_depth_SAM_current_reads=0,training_updates=0,head_forward_calls=0,PnP_calls=0,
        disk_bytes_before_manifest=total,GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),seconds=time.monotonic()-start)
    write(out/'PACK_MANIFEST.json',manifest)
    qa=dict(completed=True,PASS=True,manifest=bind(out/'PACK_MANIFEST.json'),first_replay_and_decoder_proof=firstproof,
        all100_PIL_cv2_RGB_exact=True,full_P3_actual_unpadded_shapes_valid=True,model_state_unchanged=True,
        checkpoint_SHA_unchanged=True,all12900_source_q_L_order_preserved=True,all12900_native9_masks_match_channel65=True,
        generated_scalar_analytic_tests=bind(GENERATED),no_native8_context_or_target_created=True,
        no_GT_or_current_read=True,no_training_head_forward_PnP=True)
    write(out/'INPUT_PACK_QA.json',qa)
    disk_guard()
    write(out/'INPUT_PACK_SEAL.json',dict(completed=True,PASS=True,protocol=bind(PROTOCOL),entry=bind(entry_path),
        manifest=bind(out/'PACK_MANIFEST.json'),QA=bind(out/'INPUT_PACK_QA.json'),frame_records=bind(frpath),lineage=bind(lin),
        full_P3_files=manifest['full_P3_files'],chunks=[c['file'] for c in chunks],frame_count=100,tile_count=12900,
        source_only=True,current_allowed=False,training_allowed=False,PnP_allowed=False))
    hook.remove();backbone.close()
    print(json.dumps(dict(completed=True,seal=bind(out/'INPUT_PACK_SEAL.json'),seconds=time.monotonic()-start)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['generated','freeze','run'])
    p.add_argument('--entry')
    a=p.parse_args()
    if a.stage=='generated':generated()
    elif a.stage=='freeze':freeze()
    else:
        if not a.entry:p.error('--entry required for actual source execution')
        run(a.entry)
