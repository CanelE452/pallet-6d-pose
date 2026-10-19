"""Single-worker resumable S3/S4 producer with sealed row checkpoints.

The same callable executes real238 and original synthetic1003.  Production
imports no supervision/scorer and reads sanitized model-identity metadata only.
The 24 timing rows are selected by sorted ID before any new accuracy scoring.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import contextlib
import gzip
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
import unittest

import cv2
import numpy as np

from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import PredictionFirewall
from . import material_assets as A,material_support as S,render_refine as R


PLAN_SHA='e5b965e312fc96e6ef245d868285be2bf48e7ba4ee5a7dc5a612368d9485c023'
ARMS=('S3_COMMON','S3_BI','S4_PL','S4_CONTOUR','S4_IMAGE')


def finite(value):
    if isinstance(value,dict):return {str(k):finite(v)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [finite(v)for v in value]
    if isinstance(value,np.ndarray):return finite(value.tolist())
    if isinstance(value,np.generic):return finite(value.item())
    if isinstance(value,Path):return str(value)
    if isinstance(value,float)and not math.isfinite(value):return None
    return value


def canonical(value):return json.dumps(finite(value),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with (gzip.open(path,'rt')if str(path).endswith('.gz')else open(path))as f:
        for line in f:
            if line.strip():yield json.loads(line)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=canonical(value)+'\n'
    if path.exists():
        if path.read_text()!=text:raise ValueError('Immutable artifact differs: '+str(path))
        return
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}');temp.write_text(text);temp.replace(path)


def save(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with open(temp,'wb')as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0)as z:
            for value in values:z.write((canonical(value)+'\n').encode())
    if path.exists():
        if A.sha(path)!=A.sha(temp):raise ValueError('Immutable rows differ: '+str(path))
        temp.unlink()
    else:temp.replace(path)


def append(path,value):
    with open(path,'a')as f:f.write(canonical(value)+'\n')


def source_bindings():
    paths=[Path(__file__),Path(A.__file__),Path(S.__file__),Path(R.__file__),
        Path(__file__).with_name('image_tests.py'),Path(__file__).with_name('material_README.md'),
        Path(__file__).with_name('render_thread_parity.py'),
        Path(inspect.getfile(PredictionFirewall))]
    import pyrender.camera,pyrender.renderer,pyrender.offscreen
    from scipy.optimize import least_squares
    paths += [Path(m.__file__)for m in (pyrender.camera,pyrender.renderer,pyrender.offscreen)]
    paths += [Path(inspect.getfile(least_squares))]
    return {str(p):A.sha(p)for p in paths}


def environment():
    if os.environ.get('LP_NUM_THREADS')not in ('1','4'):raise ValueError('Explicit frozen software raster worker setting required')
    versions={}
    for name in ('numpy','scipy','opencv-python','trimesh','pyrender','PyOpenGL','Pillow','usd-core'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='MODULE_AVAILABLE_VERSION_NOT_DISTRIBUTION_NAMED'
    return dict(packages=versions,cv2_version=cv2.__version__,EGL_device=1,
        BLAS_threads={k:os.environ.get(k)for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        CPU_workers=1,LP_NUM_THREADS=int(os.environ['LP_NUM_THREADS']),GL_implementation=A.hardware_info(),source_new_packages_installed=False)


def paths(d1,domain):
    name='REAL238'if domain=='real'else'SYNTHETIC1003'
    return Path(d1)/'d1_upstream_v1'/f'{name}_UPSTREAM.jsonl.gz',Path(d1)/f'd1_{domain}_candidates_v1',Path(d1)/f'd1_{domain}_observations_v1'


def guard(old_run):
    old=Path(old_run)
    forbidden=[old/'synthetic_v1/SYNTH_GT_LABELS.jsonl.gz',old/'synthetic_v1/GEOMETRY_SIDETABLE.jsonl.gz',
        old.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json',
        old.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1/oracle_v1/ORACLE_ROWS.jsonl.gz']
    return PredictionFirewall(forbidden).install()


def identities(old_run,d1):
    # This pre-existing split/identity lock contains asset names and scheduling
    # metadata, not poses, cuboid labels, GT boxes or GT masks. Sanitize it here.
    lock_path=Path(old_run)/'synthetic_v1/DATA_SPLIT_LOCK.json'
    lock=read(lock_path)
    source={r['id']:r['source_asset']for r in lock['all_rows']}
    result=[]
    for domain in ('real','synthetic'):
        inputs=list(rows(paths(d1,domain)[0]))
        for f in inputs:
            named=source[f['id']]if domain=='synthetic'else None
            key,reason=A.choose_asset(f['known_physical_dimensions_xyz_m'],named)
            result.append(dict(id=f['id'],domain=domain,named_source_asset=named,
                proxy_key=key,selection_reason=reason,
                known_physical_dimensions_xyz_m=f['known_physical_dimensions_xyz_m'],
                source='PREEXISTING_ASSET_IDENTITY_ONLY_SPLIT_LOCK'if named else'UPSTREAM_KNOWN_PHYSICAL_DIMENSIONS',
                ID_or_asset_or_material_is_H1_feature=False,GT_pose_reads=0))
    return result,dict(source_split_identity_lock=str(lock_path),sha256=A.sha(lock_path),
        consumed_fields=['all_rows.id','all_rows.source_asset'],GT_pose_or_GEOMETRY_SIDETABLE_read=False)


def prepare(run,d1,old_run,asset_root,output):
    started=time.monotonic();run,out=Path(run),Path(output)
    if A.sha(run/'CAMPAIGN_PLAN.json')!=PLAN_SHA:raise ValueError('Frozen parent campaign plan differs')
    out.mkdir(parents=True,exist_ok=True)
    firewall=guard(old_run);metadata,provenance=identities(old_run,d1)
    save(out/'MODEL_IDENTITY_INPUTS1241.jsonl.gz',metadata)
    from . import image_tests
    suite=unittest.defaultTestLoader.loadTestsFromModule(image_tests)
    stream=io.StringIO();tick=time.monotonic();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt=dict(PASS=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        output=stream.getvalue(),wall_seconds=time.monotonic()-tick,authored_constructed_poses_only=True,
        current_real_GT_reads=0,scientific_training_fits=0,scientific_real_pose_updates=0,
        pixel_contract='OpenGL index+.5 raster centres -> stored integer-centre K via renderer principal-point+.5; input K unchanged',
        depth_contract='textured RGB pass plus nonMSAA visible depth pass; near/far from current known proxy bounds',
        analytic_jacobian=True,optimizer_residual_calls_max_per_update=10)
    write(out/'CONSTRUCTED_TESTS.json',receipt)
    if not result.wasSuccessful():raise AssertionError(stream.getvalue())
    assets=A.AssetRepository(asset_root)
    registry=[]
    # Geometry-only checks of all selected assets and both target ranks.
    for source,dims in [(None,[1.1,.11,1.3]),(None,[.8,.14,.59]),('scene.usd',[1.,.15,1.2]),('scene_1.usd',[1.,.14,1.2])]:
        model,registration=assets.get(dims,source)
        np.testing.assert_allclose(np.ptp(model.body_vertices(dims),axis=0),dims,atol=1e-8)
        registry.append(registration)
    write(out/'MATERIAL_ASSET_AUDIT.json',dict(assets=registry,all_asset_bindings=assets.bindings,
        missing_named_GLBs=['eur_pallet_bk_cc0.glb','woodpallet_block_jtoastie_ccby.glb'],
        search_scope=['repo tracked+ignored data','research directories','Downloads extracted files and ZIP member inventories','.cache'],
        physical_CAD_identity_verified=False,physical_owner_oracle='NOT_APPLICABLE_WITH_REASON_NO_INDEPENDENT_LABELS',
        earlier_unverified_registration_retained=True,exact_named_asset_does_not_imply_verified_signed_axes=True))
    write(out/'INFORMATION_TABLE.json',dict(
        shared_known_inputs=['raw RGB','K','known XYZ dimensions','proxy mesh registration','N0','original D1 automatic target masks','same <=2 physical starts'],
        S3_COMMON=dict(actual_image_cues=['RGB Canny normal support','tight mask boundary','full mask boundary'],
            policy='minimum normalized cue cost, RGB tie first',dimension='two supported actual mesh directions2D; one1D; none0D',
            covariance='shared conditional4+RMS²; junction propagation; no fitted role floor',old_downstream_role_features='unchanged'),
        S3_BI=dict(actual_image_cues=['RGB Canny normal support','tight mask boundary','full mask boundary'],
            policy='B tight/full mask thenRGB; I RGB then tight/full mask; same support gates',dimension='same evidence rule as COMMON',
            covariance='same conditional rule; no extra CAL fit',old_downstream_role_features='same unchanged features'),
        S4_PL=dict(actual_consumption=['original native points','typed native normal observations','current native cuboid outline to automatic mask'],
            mesh_access='same registered proxy available; structural mesh not its residual'),
        S4_CONTOUR=dict(actual_consumption=['rendered visible structured proxy silhouette/depth','automatic target mask boundary']),
        S4_IMAGE=dict(actual_consumption=['whole native grayscale','textured RGB converted to grayscale','nonMSAA mesh depth','automatic target mask'],
            matcher='OpenCV4.9 DIS FAST bidirectional; depth->object3D; RefineLM',all_RGB_channels_consumed=False),
        all_new_candidates='producer validity/properR/positive original-native depth; same original62scalar+valid/188D downstream feature path',
        physical_oracle='NA only; automatic S3/S4 runs remain executable',GT_inputs=False))
    lock=dict(schema='S3_S4_FIXED_PRODUCTION_METHOD_V1',parent_plan_sha256=PLAN_SHA,
        sources=source_bindings(),environment=environment(),material_support_configuration=S.CONFIG,
        refinement_configuration=R.CONFIG,model_identity_inputs_sha256=A.sha(out/'MODEL_IDENTITY_INPUTS1241.jsonl.gz'),
        model_identity_source=provenance,asset_bindings=assets.bindings,
        input_files={domain:dict(path=str(paths(d1,domain)[0]),sha256=A.sha(paths(d1,domain)[0]))for domain in ('real','synthetic')},
        physical_registration='fixed proper footprint-rank axis map, bbox centring, known-axis metric scaling; registered proxy only',
        registration_and_pixel_convention_tests_passed=True,current_real_reference_decode=False,
        normalized_none_prior_source='existing old CONFIG_FROZEN none_prior=.01; no new owner permutation',
        renderer_numerical_details=receipt['pixel_contract']+'; '+receipt['depth_contract'])
    write(out/'METHOD_LOCK.json',lock)
    write(out/'PREPARATION_RECEIPT.json',dict(complete=True,wall_seconds=time.monotonic()-started,
        source_sha256=digest(lock['sources']),method_lock_sha256=A.sha(out/'METHOD_LOCK.json'),
        reference_firewall=firewall.receipt(),real_pose_updates=0,H1_fits=0))
    return dict(complete=True,constructed_tests=receipt['tests_run'],method_lock_sha256=A.sha(out/'METHOD_LOCK.json'))


def validate_lock(run,output):
    active=Path(output)/'METHOD_LOCK_RESOURCE_LP4.json'
    lock=read(active if active.exists()else Path(output)/'METHOD_LOCK.json')
    if A.sha(Path(run)/'CAMPAIGN_PLAN.json')!=PLAN_SHA or lock['parent_plan_sha256']!=PLAN_SHA:
        raise ValueError('Campaign plan changed')
    if lock['sources']!=source_bindings()or lock['environment']!=environment():
        raise ValueError('Frozen S3/S4 source/environment changed')
    for path,value in lock['asset_bindings'].items():
        if A.sha(path)!=value:raise ValueError('Registered proxy bytes changed')
    return lock


def active_lock_path(output):
    amended=Path(output)/'METHOD_LOCK_RESOURCE_LP4.json'
    return amended if amended.exists()else Path(output)/'METHOD_LOCK.json'


def resource_amend(run,output):
    """Explicit root-authorized worker change; scientific modules stay bytesame."""
    out=Path(output);old=read(out/'METHOD_LOCK.json')
    if os.environ.get('LP_NUM_THREADS')!='4':raise ValueError('This amendment freezes LP_NUM_THREADS4 only')
    a=read(out.parent/'S3_S4_RENDER_THREADS1.json');b=read(out.parent/'S3_S4_RENDER_THREADS4.json')
    parity=dict(PASS=a['records']==b['records'],strict_bitwise_RGB_depth_mask=True,cases=len(a['records']),
        input_sha256={n:A.sha(out.parent/n)for n in ('S3_S4_RENDER_THREADS1.json','S3_S4_RENDER_THREADS4.json')})
    if not parity['PASS']or parity['cases']!=12:raise ValueError('LP4 numerical renderer parity failed; retain LP1')
    write(out.parent/'S3_S4_RENDER_THREAD_PARITY_LP4.json',parity)
    current_sources=source_bindings()
    numeric_names=('material_assets.py','material_support.py','render_refine.py','image_tests.py','render_thread_parity.py')
    equality={name:old['sources'][str(Path(__file__).with_name(name))]==current_sources[str(Path(__file__).with_name(name))]for name in numeric_names}
    if not all(equality.values()):raise ValueError('Resource amendment must not change a scientific module')
    from . import image_tests
    suite=unittest.defaultTestLoader.loadTestsFromModule(image_tests);stream=io.StringIO()
    tests=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    if not tests.wasSuccessful():raise ValueError(stream.getvalue())
    adopted={str(p.relative_to(out)):dict(seal_sha256=A.sha(p),frame_sha256=read(p)['frame_sha256'],
        dependencies=read(p)['dependencies'])for p in (out/'frames').glob('*/*/SEAL.json')}
    amendment=dict(schema='S3_S4_RESOURCE_ONLY_AMENDMENT_LP4_V1',
        root_authorization='2026-10-19 root measured16logical/8physical,18.91GiB available; permit2 solver workers +1 renderer process with4 internal graphics workers',
        parent_plan_sha256=PLAN_SHA,predecessor_method_lock_sha256=A.sha(out/'METHOD_LOCK.json'),
        predecessor_sources_snapshot='LP1_SOURCE_SNAPSHOT',scientific_module_byte_equality=equality,
        old_environment=old['environment'],new_environment=environment(),
        current_source_bindings=current_sources,constructed_tests_PASS=True,constructed_tests_run=tests.testsRun,
        renderer_worker_parity_sha256=A.sha(out.parent/'S3_S4_RENDER_THREAD_PARITY_LP4.json'),
        adopted_numerically_identical_predecessor_rows=adopted,
        model_preset_or_objective_or_budget_changed=False,quality_or_reference_used_to_choose=False,
        prior_actual_costs_preserved=True,current_real_reference_reads=0)
    write(out/'RESOURCE_AMENDMENT_LP4.json',amendment)
    new=dict(old,sources=current_sources,environment=amendment['new_environment'],
        predecessor_method_lock_sha256=amendment['predecessor_method_lock_sha256'],
        resource_amendment_sha256=A.sha(out/'RESOURCE_AMENDMENT_LP4.json'))
    write(out/'METHOD_LOCK_RESOURCE_LP4.json',new)
    return dict(PASS=True,scientific_modules_identical=True,adopted_rows=len(adopted),
        active_method_lock_sha256=A.sha(out/'METHOD_LOCK_RESOURCE_LP4.json'),LP_NUM_THREADS=4)


def _render_binding(rgb,depth,mask):
    return dict(rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),
        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        native_hw=list(depth.shape),positive_depth_pixels=int(mask.sum()),
        depth_units='metres',pixel_centres='stored integer-index original pixels')


def _unchanged_function_closure(folder):
    names={
        'material_assets.py':{'axis_registration','_usd_meshes','project','unproject'},
        'material_support.py':{'boundary','compile_factors','generic_residuals'},
        'render_refine.py':{'_image_update','_nearest_contour_factors','optimize_factors','body_geometry',
            'pose','pose_valid','_step_valid','candidate'}}
    result={}
    for name,wanted in names.items():
        tree=ast.parse((Path(folder)/name).read_text())
        for node in tree.body:
            if isinstance(node,ast.FunctionDef)and node.name in wanted:
                result[name+'/'+node.name]=digest(ast.dump(node,include_attributes=False))
            if name=='material_assets.py'and isinstance(node,ast.ClassDef)and node.name in ('VisibleRenderer','AssetRepository'):
                for method in node.body:
                    if isinstance(method,ast.FunctionDef)and method.name in ('__init__','get','render','close'):
                        result[name+'/'+node.name+'.'+method.name]=digest(ast.dump(method,include_attributes=False))
            if name=='render_refine.py'and isinstance(node,ast.FunctionDef)and node.name=='refine_s4':
                selected=[n for n in ast.walk(node)if isinstance(n,ast.If)and ast.unparse(n.test)=="arm == 'PL'"]
                if len(selected)!=1:raise ValueError('Unique PL versus contour/image branch expected')
                result[name+'/refine_s4.CONTOUR_IMAGE_BODY']=digest([ast.dump(x,include_attributes=False)for x in selected[0].orelse])
    return result


def _without_timing(value):
    if isinstance(value,dict):return {k:_without_timing(v)for k,v in value.items()if k not in ('wall_seconds',)}
    if isinstance(value,list):return [_without_timing(v)for v in value]
    return value


def substage_reuse_proof(output,prior):
    """Function closure plus strict numeric replay; no tolerance-based reuse."""
    out,prior=Path(output),Path(prior)
    if not (prior/'SOURCE_SNAPSHOT').exists():return None
    old=_unchanged_function_closure(prior/'SOURCE_SNAPSHOT')
    new=_unchanged_function_closure(Path(__file__).parent)
    thread=read(out.parent/'S3_S4_RENDER_THREAD_PARITY.json')
    oldlock,newlock=read(prior/'METHOD_LOCK.json'),read(out/'METHOD_LOCK.json')
    parity=[]
    pilot=read(out/'PILOT_RECEIPT.json')
    wanted=set(pilot['selected_ids'])
    for path in sorted((out/'frames').glob('*/*/FRAME.jsonl.gz')):
        row=next(rows(path))
        if row['id']not in wanted:continue
        oldpath=prior/path.relative_to(out)
        previous=next(rows(oldpath))
        equality={arm:canonical(_without_timing(row['s4'][arm]))==canonical(_without_timing(previous['s4'][arm]))for arm in ('CONTOUR','IMAGE')}
        parity.append(dict(id=row['id'],strict_numeric_and_correspondence_equality=equality,
            old_frame_sha256=A.sha(oldpath),new_frame_sha256=A.sha(path)))
    passed=(old==new and thread['PASS'] and len(parity)==24 and all(all(r['strict_numeric_and_correspondence_equality'].values())for r in parity)
        and oldlock['asset_bindings']==newlock['asset_bindings']
        and oldlock['refinement_configuration']==newlock['refinement_configuration'])
    proof=dict(PASS=passed,prior_root=str(prior),function_closure_old=old,function_closure_new=new,
        function_AST_exact=old==new,raster_thread_parity_sha256=A.sha(out.parent/'S3_S4_RENDER_THREAD_PARITY.json'),
        actual24row_replays=parity,original_and_new_asset_bindings_equal=oldlock['asset_bindings']==newlock['asset_bindings'],
        exact_no_tolerance_reuse=True,
        independence='CONTOUR/IMAGE branch reads neither original normal weights nor original/generic overlap packet; their shared body XYZ output is unchanged',
        excluded_timing_fields=['wall_seconds'],all_numeric_endpoints_correspondences_render_hashes_compared=True)
    write(out/'UNCHANGED_CONTOUR_IMAGE_REUSE_PROOF.json',proof)
    return proof if passed else None


def produce_frame(frame,d1_result,mask_arrays,pf_candidates,assets,renderer,asset_identity,reuse_branches=None):
    """Actual generic support and five pose-producing arms for either domain."""
    started,cpu=time.monotonic(),time.process_time()
    calls,passes,seconds=renderer.calls,renderer.raster_passes,renderer.seconds
    image=cv2.imread(frame['image'])
    if image is None or list(image.shape[:2])!=frame['raw_hw']or A.sha(frame['image'])!=frame['image_sha256']:
        raise ValueError('Exact original RGB binding differs')
    if d1_result['id']!=frame['id']or asset_identity['id']!=frame['id']:
        raise ValueError('Per-row identities differ')
    dims=frame['known_physical_dimensions_xyz_m']
    model,registration=assets.get(dims,asset_identity['named_source_asset'])
    starts,start_audit=R.choose_starts(frame,d1_result,pf_candidates)
    full=np.asarray(mask_arrays['pallet_mask_clean'],np.float32)
    tight=np.asarray(mask_arrays['pallet_mask_tight'],np.float32)
    if full.shape!=tuple(frame['raw_hw'])or tight.shape!=full.shape:raise ValueError('Automatic masks must use exact original canvas')
    if not starts:
        # Preserve a genuinely unavailable/no-pose producer without inventing
        # successful output or treating an executed fallback as an interruption.
        return dict(id=frame['id'],complete=True,status='EXECUTED_NO_VALID_INITIAL_POSE',
            initializations=[],initialization_audit=start_audit,asset_registration=registration,
            s3={arm:dict(candidates=[],iterations=[])for arm in ('COMMON','BI')},
            s4={arm:dict(candidates=[],iterations=[])for arm in ('PL','CONTOUR','IMAGE')},
            material_observations=None,resource=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu))
    initial=frame['initial_pose'];K=np.asarray(frame['K'])
    rgb,depth,visible=renderer.render(model,dims,np.asarray(initial['R_physical']),np.asarray(initial['centroid']),K,frame['raw_hw'])
    shared=S.extract_supports(frame,model,dims,np.asarray(initial['R_physical']),np.asarray(initial['centroid']),
        K,depth,full,tight,image)
    s3={arm:R.refine_material(frame,starts,shared['policy_packets'][arm],d1_result)for arm in ('COMMON','BI')}
    s4=R.refine_s4(frame,starts,d1_result,model,dims,renderer,image,full,reuse_branches=reuse_branches)
    for arm in ('PL','CONTOUR','IMAGE'):
        if canonical(s4[arm]['exact_common_initializations'])!=canonical(starts):raise AssertionError('S4 starts differ')
    new_s4=[value for arm,value in s4.items()if not reuse_branches or arm not in reuse_branches]
    alliterations=[r for branch in list(s3.values())+new_s4 for r in branch['iterations']]
    counts=Counter()
    for record in alliterations:
        for name in ('objective_calls','jacobian_calls','DIS_calls','PnP_calls'):
            counts[name]+=int(record.get(name,0))
    return dict(id=frame['id'],complete=True,status='COMPLETE_NEW',image=frame['image'],image_sha256=frame['image_sha256'],
        initializations=starts,initialization_audit=start_audit,asset_registration=registration,
        material_observations=shared,initial_visible_render_binding=_render_binding(rgb,depth,visible),
        s3=s3,s4=s4,GT_inputs=False,current_real_reference_reads=0,
        exact_reused_substages=sorted(reuse_branches or {}),
        resource=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
            logical_render_calls=renderer.calls-calls,actual_raster_passes=renderer.raster_passes-passes,
            renderer_wall_seconds=renderer.seconds-seconds,actual_calls=dict(counts),
            process_peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            hardware_GPU_render_jobs=0,software_EGL_contexts=1,CPU_workers=1,BLAS_threads=1,LP_NUM_THREADS=1))


def load_pf(run,old_run):
    paths_=[Path(run)/'s1_synthetic_pf/PREDICTIONS.jsonl.gz',
        Path(old_run).parent/'pallet-accuracy-closeout-20261017-v1/posefix192_v1/PREDICTIONS.jsonl.gz']
    result={}
    for path in paths_:
        for row in rows(path):
            if row.get('method')in ('POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR'):
                result.setdefault(row['id'],[]).append(row)
    return result,{str(p):A.sha(p)for p in paths_}


def batch(run,d1,old_run,asset_root,output,mode):
    started,cpu=time.monotonic(),time.process_time();cv2.setNumThreads(1)
    out=Path(output);lock=validate_lock(run,out);firewall=guard(old_run)
    identity={r['id']:r for r in rows(out/'MODEL_IDENTITY_INPUTS1241.jsonl.gz')}
    pf,pfbindings=load_pf(run,old_run)
    assets=A.AssetRepository(asset_root);renderer=A.VisibleRenderer()
    selected=[];all_frames=[]
    for domain in ('real','synthetic'):
        input_path,bank,observations=paths(d1,domain)
        data=list(rows(input_path))
        if A.sha(input_path)!=lock['input_files'][domain]['sha256']:raise ValueError('Input bytes differ')
        wanted={f['id']for f in sorted(data,key=lambda f:f['id'])[:12]}if mode=='pilot'else{f['id']for f in data}
        for index,frame in enumerate(data):
            entry=(domain,index,frame,bank,observations)
            all_frames.append(entry)
            if frame['id']in wanted:selected.append(entry)
    if mode=='pilot'and len(selected)!=24:raise ValueError('Frozen24 timing rows required')
    write(out/(mode.upper()+'_EXECUTION_LOCK.json'),dict(method_lock_sha256=A.sha(out/'METHOD_LOCK.json'),
        mode=mode,ordered_ids=[f[2]['id']for f in selected],selection='first12 lexicographically sorted IDs/domain; before accuracy scoring',
        PF_inputs=pfbindings,CPU_workers=1,software_EGL_contexts=1,hardware_GPU_render_jobs=0,
        LP_NUM_THREADS=int(os.environ['LP_NUM_THREADS']),BLAS_threads=1))
    counts=Counter();timings=[]
    prior=out.parent/'s3_s4_technical_attempt2'
    reuse_proof=substage_reuse_proof(out,prior)if mode=='full'else None
    try:
        for domain,index,frame,bank,observation_dir in selected:
            row_started=time.monotonic()
            source=bank/'frames'/f'{index:04d}';mask_path=observation_dir/'frames'/f'{index:04d}'/'MASK.npz'
            erun_path=source/'E_RUN.jsonl.gz'
            folder=out/'frames'/domain/f'{index:04d}';folder.mkdir(parents=True,exist_ok=True)
            dest=folder/'FRAME.jsonl.gz';sealpath=folder/'SEAL.json'
            key=dict(method_lock_sha256=A.sha(active_lock_path(out)),input_payload_sha256=digest(frame),
                model_identity_payload_sha256=digest(identity[frame['id']]),E_RUN_sha256=A.sha(erun_path),
                automatic_mask_sha256=A.sha(mask_path),PF_rows_payload_sha256=digest(pf.get(frame['id'],[])))
            reuse_branches=None
            oldfolder=prior/'frames'/domain/f'{index:04d}'
            previous_current_seal=read(sealpath)if sealpath.exists()else None
            if previous_current_seal is not None and (out/'RESOURCE_AMENDMENT_LP4.json').exists():
                amendment=read(out/'RESOURCE_AMENDMENT_LP4.json')
                entry=amendment['adopted_numerically_identical_predecessor_rows'].get(str(sealpath.relative_to(out)))
                if entry is not None:
                    if A.sha(sealpath)!=entry['seal_sha256']or A.sha(dest)!=entry['frame_sha256']:
                        raise ValueError('Adopted resource-predecessor checkpoint changed')
                    key['method_lock_sha256']=amendment['predecessor_method_lock_sha256']
            want_reuse=previous_current_seal is None or'unchanged_substage_reuse'in previous_current_seal['dependencies']
            if reuse_proof and want_reuse and(oldfolder/'SEAL.json').exists():
                oldseal=read(oldfolder/'SEAL.json');oldrow=oldfolder/'FRAME.jsonl.gz'
                fields=('input_payload_sha256','model_identity_payload_sha256','E_RUN_sha256','automatic_mask_sha256','PF_rows_payload_sha256')
                if all(oldseal['dependencies'][k]==key[k]for k in fields)and oldseal['frame_sha256']==A.sha(oldrow):
                    previous=next(rows(oldrow));reuse_branches={arm:previous['s4'][arm]for arm in ('CONTOUR','IMAGE')}
                    key['unchanged_substage_reuse']=dict(prior_frame_sha256=A.sha(oldrow),proof_sha256=A.sha(out/'UNCHANGED_CONTOUR_IMAGE_REUSE_PROOF.json'))
            if sealpath.exists():
                seal=read(sealpath)
                if seal['dependencies']!=key or seal['frame_sha256']!=A.sha(dest):raise ValueError('Completed S3/S4 row cache changed')
                counts['exact_completed_row_cache_hits']+=1
                append(out/'EXECUTION_LEDGER.jsonl',dict(event='EXACT_ROW_REUSE',id=frame['id'],domain=domain,
                    frame_sha256=seal['frame_sha256'],wall_seconds=time.monotonic()-row_started,old_cost_not_charged=True))
                continue
            try:
                erun=next(rows(erun_path))
                with np.load(mask_path,allow_pickle=False)as z:mask={k:z[k]for k in z.files}
                result=produce_frame(frame,erun,mask,pf.get(frame['id'],[]),assets,renderer,identity[frame['id']],reuse_branches=reuse_branches)
                result['resource']['LP_NUM_THREADS']=int(os.environ['LP_NUM_THREADS'])
                save(dest,[result])
                seal=dict(id=frame['id'],complete=True,dependencies=key,frame_sha256=A.sha(dest),
                    resource=result['resource'],status=result['status'],current_real_reference_reads=0)
                write(sealpath,seal);counts['new_completed_rows']+=1
                if reuse_branches:counts['exact_unaffected_substage_reuses']+=len(reuse_branches)
                timings.append(dict(id=frame['id'],domain=domain,**result['resource']))
                append(out/'EXECUTION_LEDGER.jsonl',dict(event='NEW_ROW_COMPLETE',id=frame['id'],domain=domain,
                    frame_sha256=seal['frame_sha256'],resource=result['resource']))
                print(canonical(dict(stage='S3_S4_'+mode.upper(),done=counts['new_completed_rows']+counts['exact_completed_row_cache_hits'],total=len(selected),
                    id=frame['id'],row_seconds=result['resource']['wall_seconds'],elapsed_seconds=time.monotonic()-started)),flush=True)
            except Exception as error:
                failure=dict(id=frame['id'],status='BLOCKED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),
                    scientific_pose_failure=False,completed_previous_rows_preserved=True)
                write(folder/f'FAILURE_{time.time_ns()}.json',failure)
                append(out/'EXECUTION_LEDGER.jsonl',dict(event='TECHNICAL_FAILURE',**failure))
                raise
    finally:renderer.close()
    resource_totals=Counter();call_totals=Counter()
    for domain,index,frame,bank,observations in selected:
        seal=read(out/'frames'/domain/f'{index:04d}'/'SEAL.json')
        resource_=seal['resource']
        for key in ('wall_seconds','CPU_seconds','logical_render_calls','actual_raster_passes','renderer_wall_seconds'):
            resource_totals[key]+=resource_.get(key,0)
        call_totals.update(resource_.get('actual_calls',{}))
    receipt=dict(schema='S3_S4_'+mode.upper()+'_RECEIPT_V1',complete=True,rows=len(selected),
        selected_ids=[f[2]['id']for f in selected],actual_calls=dict(call_totals),
        completed_row_resource_totals=dict(resource_totals),current_invocation_counts=dict(counts),
        invocation_wall_seconds=time.monotonic()-started,invocation_CPU_seconds=time.process_time()-cpu,
        method_lock_sha256=A.sha(active_lock_path(out)),reference_firewall=firewall.receipt(),
        fit_count=0,accuracy_evaluation_executed=False)
    if mode=='pilot':
        times=[read(out/'frames'/d/f'{i:04d}'/'SEAL.json')['resource']['wall_seconds']for d,i,*_ in selected]
        receipt.update(row_seconds=dict(mean=float(np.mean(times)),median=float(np.median(times)),P95=float(np.quantile(times,.95)),maximum=float(max(times))),
            full1241_extrapolated_row_wall_seconds=float(np.mean(times)*1241),
            extrapolation_is_runtime_estimate_not_performance=True,
            full_execution_follows_without_reselecting_rows_or_thresholds=True)
    else:
        extras={arm:[]for arm in ARMS};summary=[]
        for domain,index,frame,bank,observations in all_frames:
            result=next(rows(out/'frames'/domain/f'{index:04d}'/'FRAME.jsonl.gz'))
            for arm in ARMS:
                stage,branch=arm.split('_',1)
                branch_result=result[stage.lower()][branch]
                candidates=branch_result['candidates']
                if not candidates:
                    # Explicit unavailable action preserves full-ID union
                    # denominator while carrying no fabricated numeric pose.
                    candidates=[dict(id=frame['id'],method=arm,candidate_id=arm+'/NO_VALID_INITIAL_POSE',
                        producer_available=False,actual_pose=dict(available=False),producer_status=result['status'])]
                extras[arm].extend(candidates)
                summary.append(dict(id=frame['id'],domain=domain,arm=arm,visited_endpoints=len(candidates),
                    producer_available_endpoints=sum(bool(c.get('producer_available',c['actual_pose'].get('available',False)))for c in candidates),
                    initialization_count=len(result['initializations']),asset_proxy_key=result['asset_registration']['proxy_key']))
        for arm,records in extras.items():
            for domain in ('real','synthetic'):
                ids={f[2]['id']for f in all_frames if f[0]==domain}
                save(out/f'EXTRAS_{arm}_{domain}.jsonl.gz',[r for r in records if r['id']in ids])
        save(out/'ENDPOINT_SUMMARY.jsonl.gz',summary)
        receipt.update(all1241_completed=True,arms=list(ARMS),
            output_artifacts={p.name:A.sha(p)for p in out.glob('EXTRAS_*.jsonl.gz')},
            visited_pose_endpoints={arm:len(value)for arm,value in extras.items()},
            source_lock_validated_after_production=validate_lock(run,out)==lock)
    write(out/(mode.upper()+'_RECEIPT.json'),receipt)
    return {k:receipt[k]for k in ('complete','rows','actual_calls','current_invocation_counts','invocation_wall_seconds')} | (
        {'full1241_extrapolated_row_wall_seconds':receipt['full1241_extrapolated_row_wall_seconds']}if mode=='pilot'else{})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','pilot','full','resource_amend'])
    for name in ('run','d1','old-run','asset-root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.mode=='prepare':result=prepare(a.run,a.d1,a.old_run,a.asset_root,a.output)
    elif a.mode=='resource_amend':result=resource_amend(a.run,a.output)
    else:result=batch(a.run,a.d1,a.old_run,a.asset_root,a.output,a.mode)
    print(canonical(result),flush=True)


if __name__=='__main__':main()
