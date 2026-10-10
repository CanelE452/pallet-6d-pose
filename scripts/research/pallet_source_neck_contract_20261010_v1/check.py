"""Compare all fixed CAL neck channels with the immutable source FP16 cache.

Freeze once, then run once in an explicitly authorized quiet resource window.
Only the frozen Base detector is executed. The original query/neck expressions
are compiled from AST without importing the model, solver, labels or targets.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import traceback

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
DOC = REPO / '_docs/experiments/pallet_source_neck_contract_20261010_v1'
MECHANISM = REPO / '_docs/experiments/pallet_corner_mechanism_audit_20261010_v1'
OLD_CODE = REPO / 'scripts/research/pallet_observation_refiner_20261009_v1'
INDICES = list(range(768, 896))
NAMES = ('PROTECTION_BEFORE.json', 'PROTOCOL.json', 'STARTED.json',
         'CHECKS.json', 'ROWS.jsonl.gz')
BASE_SHA = '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
BASE_REL = 'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
FEATURE_REL = 'scripts/research/pallet_line_pose_v1/features.py'
SETTINGS = dict(torch_threads=4, opencv_threads=1, matmul_allow_tf32=False,
                cudnn_allow_tf32=True, cudnn_benchmark=False, cudnn_deterministic=False)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = Path(path).resolve()
    public = p.is_relative_to(REPO)
    return dict(path=str(p.relative_to(REPO)) if public else p.name,
                origin='public_repository' if public else 'external_readonly_dependency',
                sha256=sha(p), bytes=p.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def destination(args, name, exists=False):
    out = Path(args.output)
    require(not out.is_symlink(), 'output directory is a symlink')
    out = out.resolve()
    require(not out.is_relative_to(REPO) or out == DOC, 'only the new public audit directory is writable')
    roots = [Path(args.source_root).resolve(), Path(args.baseline_root).resolve(),
             Path(args.features).resolve().parent, Path(args.cache_manifest).resolve().parent,
             Path(args.base_weights).resolve().parent]
    require(all(not out.is_relative_to(p) and not p.is_relative_to(out) for p in roots),
            'output overlaps a source, baseline, cache or checkpoint dependency')
    path = out / name
    require(name in NAMES and not path.is_symlink(), 'invalid output destination')
    require(exists or not path.exists(), 'preserve completed/interrupted output ' + name)
    return path


def paths(args):
    return dict(code=Path(__file__), package_init=CODE/'__init__.py',
        original_model=OLD_CODE/'model.py', original_edge_definition=OLD_CODE/'source_audit.py',
        original_cache_preparation=OLD_CODE/'training.py',
        actual_source_extractor=Path(args.source_root)/FEATURE_REL,
        actual_baseline_extractor=Path(args.baseline_root)/FEATURE_REL,
        current_live_pipeline=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py',
        resource_guard=REPO/'scripts/research/pallet_corner_mechanism_audit_20261010_v1/deployment_smoke.py',
        protection_code=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/protection.py',
        source_family_split=REPO/'_docs/experiments/pallet_observation_refiner_20261009_v1/SOURCE_FAMILY_SPLIT.json',
        preparation_receipt=REPO/'_docs/experiments/pallet_observation_refiner_20261009_v1/SUPERVISION_PREPARATION.json',
        previous_source_contract=MECHANISM/'SOURCE_CONTRACT_CHECKS.json',
        previous_RGB_protocol=MECHANISM/'rgb_resume/RGB_CONTRACT_PROTOCOL.json',
        previous_RGB_checks=MECHANISM/'rgb_resume/RGB_CONTRACT_CHECKS.json',
        cache_manifest=Path(args.cache_manifest), features=Path(args.features),
        base_weights=Path(args.base_weights), protection=destination(args, NAMES[0], exists=True))


def fragment_ast(p):
    """Only query geometry and the original neck stencil; no imported solver."""
    tree = ast.parse(p['original_model'].read_text())
    funcs = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    edges_tree = ast.parse(p['original_edge_definition'].read_text())
    edges_nodes = [n for n in edges_tree.body if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id == 'EDGES' for t in n.targets)]
    require(len(edges_nodes) == 1, 'one fixed edge definition required')
    edges = ast.literal_eval(edges_nodes[0].value)
    require(edges == [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)],
            'original physical edge IDs changed')
    query = funcs['query_geometry']
    body = funcs['inputs'].body
    def assignment(n, key):
        return isinstance(n,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id == key for t in n.targets)
    first = next(i for i,n in enumerate(body) if assignment(n,'gain') or
                 (isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) and
                  any(isinstance(e,ast.Name) and e.id == 'gain' for e in t.elts) for t in n.targets)))
    last = next(i for i,n in enumerate(body) if isinstance(n,ast.For) and
                isinstance(n.target,ast.Name) and n.target.id == 'name')
    selected = body[first:last+1]
    require(len(selected) == 6 and isinstance(selected[-1],ast.For), 'neck stencil statement layout changed')
    require(not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in
                    ast.walk(ast.Module(body=selected,type_ignores=[]))), 'neck fragment must not import')
    # The original concatenation promotes float32 neck samples to float64 through
    # the RGB array, then rounds the entire input to FP16 before the float32 head.
    prefix = ast.parse("cc=q['candidate'];device=captured['p3'].device").body
    suffix = ast.parse('return torch.cat(neck,1).to(torch.float64).to(torch.float16)').body
    stencil = ast.FunctionDef(name='neck_stencil', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg='captured'),ast.arg(arg='q')],vararg=None,kwonlyargs=[],
        kw_defaults=[],kwarg=None,defaults=[]),body=prefix+selected+suffix,decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[query,stencil],type_ignores=[]))
    f_tree = ast.parse(p['actual_source_extractor'].read_text())
    affine = next(n for n in f_tree.body if isinstance(n,ast.FunctionDef) and n.name=='canvas_affine')
    spec = dict(query_geometry_ast_sha256=hashlib.sha256(ast.dump(query,include_attributes=False).encode()).hexdigest(),
        neck_stencil_ast_sha256=hashlib.sha256(ast.dump(stencil,include_attributes=False).encode()).hexdigest(),
        canvas_affine_ast_sha256=hashlib.sha256(ast.dump(affine,include_attributes=False).encode()).hexdigest(),
        query_geometry_lines=[query.lineno,query.end_lineno],
        original_neck_stencil_lines=[body[first].lineno,body[last].end_lineno],
        edges=[list(e) for e in edges], rounding='original float32 neck -> float64 concatenation -> FP16',
        excluded=['RGB stencil','initial_geometry','CorrespondenceHead','loss','model module imports'])
    return module, edges, spec


def metadata(p):
    manifest = read(p['cache_manifest'])
    split = read(p['source_family_split'])
    prior = read(p['previous_RGB_protocol'])
    checks = read(p['previous_RGB_checks'])
    require(checks['passed'] and prior['population']['indices'] == INDICES,
            'previous successful fixed CAL RGB identity evidence required')
    require(manifest['complete'] and len(manifest['records']) == 1024 and
        manifest['specs']['features'] == dict(shape=[1024,84,28,65],dtype='float16'), 'cache shape contract')
    require(split['counts'] == dict(train=768,calibration=128,source_test=128), 'unchanged family split')
    selected = prior['selected_CAL']
    require([r['index'] for r in selected] == INDICES and len({r['id'] for r in selected}) == 128 and
            len({r['family'] for r in selected}) == 128, 'all 128 fixed CAL identities required')
    for r in selected:
        m,s = manifest['records'][r['index']],split['records'][r['index']]
        require(m['index']==r['index'] and m['id']==r['id']==s['id'] and
            m['family']==r['family']==s['family'] and m['partition']==r['partition']==s['partition']=='calibration',
            'cache/family/CAL identity mismatch')
        require(m['variant']==0 and m['composition']==dict(kind='existing_P0_original',generated=False),
                'only existing original P0 images')
        require(m['original_rgb']==r['original_rgb'] and m['rgb_sha256']==r['decoded_BGR_sha256'] and
                m['original_rgb']['path']==s['raw']['rgb'], 'recorded original PNG identity')
        require(m['selected_detector_candidate'] is not None and len(m['selected_points'])==9 and
                all(len(q)==2 for q in m['selected_points']), 'recorded selected Base coordinates required')
    return manifest, selected


def RGB_bindings(args, selected):
    root=Path(args.source_root).resolve(); result=[]
    for row in selected:
        declared=row['original_rgb']; relative=Path(declared['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe source PNG path')
        path=(root/relative).resolve()
        require(path.is_relative_to(root), 'source PNG escapes source-root')
        actual=dict(path=declared['path'],origin='source',sha256=sha(path),bytes=path.stat().st_size)
        require(actual==declared, 'original PNG bytes differ from source cache manifest')
        result.append(dict(index=row['index'],id=row['id'],original_rgb=actual))
    return result


def prerequisites(args, frozen=True):
    require(args.source_root and args.baseline_root and args.features and args.cache_manifest and args.base_weights,
            'explicit dependency paths required')
    for name in NAMES[2:]:
        destination(args,name)
    p=paths(args); identity={k:binding(v) for k,v in p.items()}
    require(identity['base_weights']['sha256']==BASE_SHA, 'only original frozen Base weights supported')
    require(identity['actual_source_extractor']['sha256']==identity['actual_baseline_extractor']['sha256'],
            'source and actual baseline extractors differ')
    previous=read(p['previous_RGB_protocol'])['authoritative_inputs']
    for name in ('features','cache_manifest','original_model','original_cache_preparation','original_edge_definition'):
        require(identity[name]['sha256']==previous[name]['sha256'] and identity[name]['bytes']==previous[name]['bytes'],
                'immutable original input changed '+name)
    manifest,selected=metadata(p)
    originals=RGB_bindings(args,selected)
    _,_,spec=fragment_ast(p)
    from scripts.research.pallet_boundary_corner_refiner_20261010_v2 import protection
    preserved=protection.verify(read(p['protection']),args.source_root)
    require(preserved['passed'],'protected publication or original user checkout changed')
    if frozen:
        protocol=read(destination(args,'PROTOCOL.json',exists=True))
        require(protocol['authoritative_inputs']==identity and protocol['selected_CAL']==selected and
                protocol['original_RGB']==originals and protocol['compiled_scope']==spec and
                protocol['current_numeric_settings']==SETTINGS,'frozen neck contract drift')
    return p,identity,manifest,selected,originals,spec,preserved


def freeze(args):
    for name in NAMES:
        destination(args,name)
    # Preserve all existing tracked/sparse publication bytes and user changes.
    from scripts.research.pallet_boundary_corner_refiner_20261010_v2 import protection
    write_new(destination(args,'PROTECTION_BEFORE.json'),protection.snapshot(args.source_root))
    _,identity,_,selected,originals,spec,preserved=prerequisites(args,frozen=False)
    write_new(destination(args,'PROTOCOL.json'),dict(schema='fixed_source_CAL_16_neck_contract_protocol_v1',
        frozen_before_comparison=True,authoritative_inputs=identity,selected_CAL=selected,original_RGB=originals,
        compiled_scope=spec,current_numeric_settings=SETTINGS,
        historical_numeric_settings='Original preparation recorded torch_threads=1/OpenCV_threads=1; TF32/cuDNN/version settings were not recorded.',
        population=dict(split='calibration',indices=INDICES,original_RGB_frames=128,
            selected_Base_points_shape=[9,2],feature_shape=[1024,84,28,65],
            compared_slice='features[768:896,:,3:19,:]',queries_per_frame=84,bins_per_query=65,
            channels_compared=16,compared_FP16_values=128*84*16*65),
        fixed_policy=dict(detector='original FrozenYoloFeatures.predict(BGR);already_padded=False;cpu_features=False',
            query='unchanged recorded source Base coordinates; exact original query_geometry AST',
            stencil='exact original neck AST;8 channel-mean groups each P3/P4; bilinear;zero padding;align_corners=False',
            acceptance='strict zero FP16 bit mismatches AND zero numerical mismatches; PNG/query/source preservation and fixed detector contracts also required',
            fresh_Base_coordinate_parity_atol_px=1e-7,coordinate_parity_rtol=0,
            no_selected_after_results_tolerance=True),
        expected_image_decode_calls=128,expected_detector_predict_calls=128,
        detector_internal_initialization_forwards='count actual model-root/first-layer forwards separately, including lazy warmup',
        prohibited_counts=dict(N3=0,ROLE_head=0,initial_pose=0,final_pose=0,GT_scoring=0,rays=0,training_updates=0,new_RGB=0),
        no_automatic_retry=True,protection_summary={k:v for k,v in preserved.items() if k!='current_source_checkout'},
        limits=['This checks current extractor against recorded source neck channels, not accuracy or feature usefulness.',
                'Historical backend flags are unknown; any measured mismatch is reported without a tuned tolerance.',
                'Public digests do not authenticate private cache acquisition; no physical GT or first-surface claim.',
                'No latency benchmark or image/head/pose performance experiment is performed.']))
    print('SOURCE_NECK_CONTRACT_FROZEN',flush=True)


@contextmanager
def no_truth_reads():
    originals=(builtins.open,io.open,Path.open)
    tokens=('TARGETS','READY_SOURCE','STATIC_VISIBILITY','GT_CACHE','GROUND_TRUTH','POSTHOC',
            'PREDICTIONS','OBSERVATIONS','GEOMETRY_SEALED','_label.json','mask_amodal','mask_visible','depth.npy')
    def guard(fn):
        def wrapped(path,*a,**kw):
            if isinstance(path,(str,os.PathLike)):
                name=str(path).upper()
                require(not any(token.upper() in name for token in tokens),'SOURCE_NECK_TRUTH_CANARY:'+Path(path).name)
            return fn(path,*a,**kw)
        return wrapped
    builtins.open,io.open,Path.open=map(guard,originals)
    try:
        yield
    finally:
        builtins.open,io.open,Path.open=originals


def distribution(values,np):
    a=np.asarray(values,dtype=np.float64).reshape(-1)
    require(np.isfinite(a).all(),'non-finite absolute difference')
    if not np.count_nonzero(a):
        return dict(n=int(a.size),mean=0.,median=0.,P90=0.,P99=0.,max=0.)
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.quantile(a,.5)),
                P90=float(np.quantile(a,.9)),P99=float(np.quantile(a,.99)),max=float(a.max()))


def run(args):
    p,identity,manifest,selected,originals,spec,preserved=prerequisites(args)
    from scripts.research.pallet_corner_mechanism_audit_20261010_v1.deployment_smoke import quiet
    probe=quiet()
    require(probe['quiet'] and probe['temperature_under_80'],'SOURCE_NECK_PENDING: competing research/GPU or thermal guard')
    write_new(destination(args,'STARTED.json'),dict(schema='fixed_CAL_neck_execution_claim_v1',status='STARTED',
        started_at_UTC=utc(),protocol=binding(destination(args,'PROTOCOL.json',exists=True)),
        detector_predicts=128,no_automatic_retry=True,resource_snapshot=probe))
    result=dict(schema='actual_fixed_CAL_16_neck_contract_checks_v1',passed=False,complete=False,status='FAILED',
        protocol=binding(destination(args,'PROTOCOL.json',exists=True)),authoritative_inputs=identity,
        actual_counts=dict(image_decode_calls=0,detector_predict_calls=0,detector_predicts_completed=0,
                          compiled_neck_calls=0,frames_compared=0),
        prohibited_counts=dict(N3=0,ROLE_head=0,initial_pose=0,final_pose=0,GT_scoring=0,rays=0,training_updates=0,new_RGB=0),
        resource_snapshots=[dict(phase='before_models',**probe)],truth_canary_active=True,
        no_automatic_retry=True,timing_intervals=0,accuracy_evaluation=False)
    rows=[]; model_counts=Counter(); extractor=None; global_handle=None; first_handle=None; diffs=[]
    try:
        import numpy as np
        import cv2
        import torch
        from torch.nn import functional as F
        import ultralytics
        torch.set_num_threads(SETTINGS['torch_threads']);cv2.setNumThreads(SETTINGS['opencv_threads'])
        torch.backends.cuda.matmul.allow_tf32=SETTINGS['matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32=SETTINGS['cudnn_allow_tf32']
        torch.backends.cudnn.benchmark=SETTINGS['cudnn_benchmark']
        torch.backends.cudnn.deterministic=SETTINGS['cudnn_deterministic']
        result['runtime_versions']=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,
            torch=torch.__version__,ultralytics=ultralytics.__version__,cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(0),
            historical_preparation_versions='not recorded',historical_TF32_settings='not recorded')
        result['actual_numeric_settings']=dict(torch_threads=torch.get_num_threads(),opencv_threads=cv2.getNumThreads(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            opencv_optimized=cv2.useOptimized())
        cache=np.load(p['features'],mmap_mode='r')
        require(cache.shape==(1024,84,28,65) and cache.dtype==np.float16 and not cache.flags.writeable,
                'read-only immutable FP16 cache required')
        module,edges,_=fragment_ast(p)
        loader=importlib.util.spec_from_file_location('_fixed_source_neck_features',p['actual_source_extractor'])
        source=importlib.util.module_from_spec(loader)
        namespace=dict(np=np,torch=torch,F=F,EDGES=edges)
        phase='constructor'
        def root_hook(m,args):
            if m.__class__.__name__=='PoseModel':
                model_counts['detector_model_root_forwards']+=1
                model_counts['detector_model_root_forwards_'+phase]+=1
        global_handle=torch.nn.modules.module.register_module_forward_pre_hook(root_hook)
        with no_truth_reads():
            loader.loader.exec_module(source)
            namespace['canvas_affine']=source.canvas_affine
            exec(compile(module,'<bound_original_query_and_neck_AST>','exec'),namespace)
            extractor=source.FrozenYoloFeatures(weights=p['base_weights'],device='cuda')
            def count_first(m,args):
                model_counts['detector_first_layer_forwards']+=1
            first_handle=extractor.yolo.model.model[0].register_forward_pre_hook(count_first)
            require(extractor.yolo.model.__class__.__name__=='PoseModel','fixed detector root class changed')
            result['detector_root_class']=extractor.yolo.model.__class__.__name__
            phase='predict'
            for selected_row in selected:
                i=selected_row['index'];record=manifest['records'][i]
                path=Path(args.source_root)/record['original_rgb']['path']
                image=cv2.imread(str(path),cv2.IMREAD_COLOR);result['actual_counts']['image_decode_calls']+=1
                require(image is not None and image.dtype==np.uint8 and list(image.shape[:2])==selected_row['raw_hw'],
                        'original BGR decode contract')
                decoded_sha=hashlib.sha256(image.tobytes()).hexdigest()
                require(decoded_sha==selected_row['decoded_BGR_sha256'],'decoded source BGR differs')
                points=np.asarray(record['selected_points'],dtype=np.float64)
                require(points.shape==(9,2) and np.isfinite(points).all() and
                    hashlib.sha256(points.tobytes()).hexdigest()==selected_row['selected_points_float64_sha256'],
                    'recorded immutable Base query coordinates differ')
                result['actual_counts']['detector_predict_calls']+=1
                captured=extractor.predict(image)
                result['actual_counts']['detector_predicts_completed']+=1
                fresh_index=captured['selected_index'];cached_index=record['selected_detector_candidate']
                fresh_points=None if fresh_index is None else captured['candidates'][fresh_index]['keypoints_xy']
                candidate_equal=fresh_index==cached_index
                point_delta=None if fresh_points is None or not np.isfinite(fresh_points).all() else float(np.max(np.abs(fresh_points-points)))
                coordinate_equal=point_delta is not None and point_delta<=1e-7
                q=namespace['query_geometry'](points)
                with torch.no_grad():
                    live=namespace['neck_stencil'](captured,q).cpu().numpy()
                result['actual_counts']['compiled_neck_calls']+=1
                expected=np.asarray(cache[i,:,3:19,:])
                require(live.shape==expected.shape==(84,16,65) and live.dtype==expected.dtype==np.float16 and
                        np.isfinite(live).all() and np.isfinite(expected).all(),'finite FP16 neck slice contract')
                bit_difference=live.view(np.uint16)!=expected.view(np.uint16)
                numeric_difference=live!=expected
                delta=np.abs(live.astype(np.float64)-expected.astype(np.float64))
                diffs.append(delta)
                channels=[]
                for ch in range(16):
                    channels.append(dict(channel=ch+3,scale='P3' if ch<8 else 'P4',group=ch%8,
                        FP16_bit_mismatches=int(np.count_nonzero(bit_difference[:,ch,:])),
                        numeric_mismatches=int(np.count_nonzero(numeric_difference[:,ch,:])),
                        absolute_difference=distribution(delta[:,ch,:],np)))
                gain,offset=source.canvas_affine(captured['canvas_shape'],captured['input_shape'])
                rows.append(dict(index=i,id=selected_row['id'],family=selected_row['family'],partition='calibration',
                    original_RGB=record['original_rgb'],decoded_BGR_sha256=decoded_sha,
                    recorded_Base_points_sha256=selected_row['selected_points_float64_sha256'],
                    query_candidate_float64_sha256=hashlib.sha256(q['candidate'].tobytes()).hexdigest(),
                    fresh_selected_index=fresh_index,recorded_selected_index=cached_index,
                    selected_candidate_equal=candidate_equal,fresh_Base_points_parity=coordinate_equal,
                    fresh_Base_points_max_abs_delta_px=point_delta,query_points_source='immutable original cache coordinates',
                    input_shape=list(captured['input_shape']),canvas_shape=list(captured['canvas_shape']),
                    reflected_border=captured['added_border'],canvas_gain=gain,canvas_offset=offset.tolist(),
                    captured_necks={name:dict(shape=list(captured[name].shape),dtype=str(captured[name].dtype),
                        device=str(captured[name].device)) for name in ('p3','p4')},
                    detector_eval=True,detector_parameters_frozen=True,
                    live_FP16_neck_sha256=hashlib.sha256(live.tobytes()).hexdigest(),
                    cached_FP16_neck_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                    FP16_bit_mismatches=int(np.count_nonzero(bit_difference)),
                    numeric_mismatches=int(np.count_nonzero(numeric_difference)),channels=channels))
                result['actual_counts']['frames_compared']+=1
        a=np.stack(diffs); channels=[]
        for ch in range(16):
            channels.append(dict(channel=ch+3,scale='P3' if ch<8 else 'P4',group=ch%8,
                FP16_bit_mismatches=sum(r['channels'][ch]['FP16_bit_mismatches'] for r in rows),
                numeric_mismatches=sum(r['channels'][ch]['numeric_mismatches'] for r in rows),
                absolute_difference=distribution(a[:,:,ch,:],np)))
        result.update(complete=len(rows)==128,channels=channels,compared_FP16_values=int(a.size),
            FP16_bit_mismatches=sum(r['FP16_bit_mismatches'] for r in rows),
            numeric_mismatches=sum(r['numeric_mismatches'] for r in rows),
            absolute_difference=distribution(a,np),
            fresh_Base_candidate_mismatch_frames=sum(not r['selected_candidate_equal'] for r in rows),
            fresh_Base_coordinate_mismatch_frames=sum(not r['fresh_Base_points_parity'] for r in rows))
        require(result['complete'] and result['compared_FP16_values']==128*84*16*65,'full fixed CAL comparison required')
        require(result['FP16_bit_mismatches']==result['numeric_mismatches']==0,
                'NECK_CACHE_MISMATCH: strict bit/numerical equality failed; no tolerance tuning or retry')
        require(result['fresh_Base_candidate_mismatch_frames']==result['fresh_Base_coordinate_mismatch_frames']==0,
                'FRESH_BASE_QUERY_PARITY_MISMATCH: cached coordinates remain unchanged')
        result.update(passed=True,status='PASS')
    except Exception as exc:
        result['exception_type']=type(exc).__name__;result['exception']=str(exc)
        result['traceback']=traceback.format_exc().replace(str(REPO),'<public_repository>').replace(
            str(Path(args.source_root).resolve()),'<source_root>').replace(
            str(Path(args.baseline_root).resolve()),'<baseline_root>')
    finally:
        cleanup_errors=[]
        for label,callback in (
                ('first_layer_hook',None if first_handle is None else first_handle.remove),
                ('extractor',None if extractor is None else extractor.close),
                ('global_model_hook',None if global_handle is None else global_handle.remove)):
            if callback is not None:
                try:callback()
                except Exception as exc:cleanup_errors.append(dict(phase=label,error=str(exc)))
        result['cleanup_errors']=cleanup_errors
        result['actual_model_forwards']=dict(model_counts)
        result['detector_internal_initialization_forwards']=(
            model_counts['detector_model_root_forwards']-result['actual_counts']['detector_predict_calls']
            if result['actual_counts']['detector_predicts_completed']==result['actual_counts']['detector_predict_calls'] else None)
        result['internal_initialization_count_scope']='All PoseModel root forwards including constructor and lazy first-predict warmup; first-layer crosscheck attached after constructor.'
        from scripts.research.pallet_boundary_corner_refiner_20261010_v2 import protection
        try:
            result['protection_after']=protection.verify(read(p['protection']),args.source_root)
            after={k:binding(v) for k,v in p.items()};result['input_bindings_unchanged']=identity==after
            result['original_PNG_bindings_unchanged']=originals==RGB_bindings(args,selected)
            result['resource_snapshots'].append(dict(phase='after_models',**quiet()))
        except Exception as exc:
            result['final_preservation_exception']=str(exc)
            result.update(passed=False,status='FAILED_FINAL_PRESERVATION_CHECK')
        if cleanup_errors or not result.get('protection_after',{}).get('passed') or not result.get('input_bindings_unchanged') or not result.get('original_PNG_bindings_unchanged'):
            result.update(passed=False,status='FAILED_PRESERVATION')
        out=destination(args,'ROWS.jsonl.gz')
        with out.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:
                for row in rows:zipped.write((json.dumps(row,allow_nan=False)+'\n').encode())
        result['rows']=binding(out);result['rows_written']=len(rows)
        result['finished_at_UTC']=utc()
        write_new(destination(args,'CHECKS.json'),result)
        print(json.dumps({k:result.get(k) for k in ('passed','status','complete','actual_counts','actual_model_forwards',
            'FP16_bit_mismatches','numeric_mismatches','exception')},ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','preflight','run'])
    parser.add_argument('--source-root',required=True);parser.add_argument('--baseline-root',required=True)
    parser.add_argument('--features',required=True);parser.add_argument('--cache-manifest',required=True)
    parser.add_argument('--base-weights',required=True)
    parser.add_argument('--output',default=str(DOC))
    args=parser.parse_args()
    if args.stage=='freeze':freeze(args)
    elif args.stage=='preflight':
        *_,preserved=prerequisites(args)
        print(json.dumps(dict(passed=True,model_imports=0,CUDA_calls=0,frames=128,channels=16,
                             protection_passed=preserved['passed'])))
    else:run(args)


if __name__=='__main__':main()
