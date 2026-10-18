"""Independent raw-coordinate/query/bilinear audit; no GT or neural forward."""
from __future__ import annotations
import argparse,ast,gzip,hashlib,json,math,resource,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from .independent_checks import independent_actions


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line)
def write(p,value):
    with Path(p).open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def queries(xyz,edges):
    a=np.asarray(xyz,float);result=list(a)
    if a.shape!=(8,3)or len(edges)!=12 or any(sum(i in edge for edge in edges)!=3 for i in range(8)):raise ValueError('Original degree-three cuboid adjacency required')
    for first,second in edges:
        if sum(x!=y for x,y in zip(a[first],a[second]))!=1:raise ValueError('Not an actual native cuboid edge')
        result.append([(float(x)+float(y))*.5 for x,y in zip(a[first],a[second])])
    return np.asarray(result,float)


def project(pose,geometry,edges):
    xyz=np.asarray(geometry['native_xyz'],float);q=queries(xyz,edges);Q=np.asarray(geometry['physical_registration_Q'],float)
    rotation=np.asarray(pose['R_physical'],float)@Q.T;center=np.asarray(pose['centroid'],float);K=np.asarray(geometry['K'],float)
    origin=[math.fsum(float(x[j])for x in xyz)/8 for j in range(3)];pixel=[];depth=[]
    for point in q:
        camera=[math.fsum(float(rotation[i,j])*(float(point[j])-origin[j])for j in range(3))+float(center[i])for i in range(3)]
        homogeneous=[math.fsum(float(K[i,j])*camera[j]for j in range(3))for i in range(3)]
        pixel.append([homogeneous[j]/homogeneous[2]if homogeneous[2]!=0 else math.nan for j in(0,1)]);depth.append(camera[2])
    return np.asarray(pixel),np.asarray(depth)


def bilinear(image,pixels,depths,side):
    image=np.asarray(image);h,w=image.shape[:2];result=np.zeros((20,4,16,16),np.float32);query_valid=np.zeros(20,bool)
    if side is None or not math.isfinite(side)or side<=0:return result,np.zeros(20,np.float32),query_valid
    for region,(point,z)in enumerate(zip(pixels,depths)):
        if not(math.isfinite(float(z))and z>0 and all(math.isfinite(float(v))for v in point)):continue
        query_valid[region]=True
        for iy in range(16):
            for ix in range(16):
                x=float(point[0])+((ix+.5)/16.-.5)*side;y=float(point[1])+((iy+.5)/16.-.5)*side
                if not(0.<=x<=w-1 and 0.<=y<=h-1):continue
                x0,y0=math.floor(x),math.floor(y);x1,y1=min(x0+1,w-1),min(y0+1,h-1);dx,dy=x-x0,y-y0
                for channel in range(3):
                    value=((1.-dx)*(1.-dy)*float(image[y0,x0,channel])+dx*(1.-dy)*float(image[y0,x1,channel])+(1.-dx)*dy*float(image[y1,x0,channel])+dx*dy*float(image[y1,x1,channel]))/255.
                    result[region,channel,iy,ix]=value
                result[region,3,iy,ix]=1.
    validity=np.asarray([sum(float(v)for v in row.flat)/256. for row in result[:,3]],np.float32)
    return result,validity,query_valid


def source_contracts(repo):
    namespace=Path(repo)/'scripts/research/pallet_risk_information_v1';trees={n:ast.parse((namespace/n).read_text())for n in('predict.py','visual_patches.py','risk_models.py','train.py')};checks={}
    # Compare explicit API argument whitelists, rather than filename rejection.
    functions=[n for n in ast.walk(trees['visual_patches.py'])if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    call=next(n for n in functions if n.name=='__call__');checks['provider_no_supervised_argument']=[a.arg for a in call.args.args]==['self','reference_free_frame','actual_entries','indices']
    proj=next(n for n in functions if n.name=='project_queries');checks['projection_no_reference_argument']=[a.arg for a in proj.args.args]==['pose','geometry']
    forward=next(n for n in ast.walk(trees['risk_models.py'])if isinstance(n,ast.FunctionDef)and n.name=='forward')
    checks['model_forward_whitelist']=[a.arg for a in forward.args.args]==['self','f0','extra','patches','region_validity']
    imports=[n for n in trees['risk_models.py'].body if isinstance(n,(ast.Import,ast.ImportFrom))]
    checks['model_has_no_supervision_import']=all('supervision'not in ast.unparse(n)and'evaluate'not in ast.unparse(n)for n in imports)
    null=[n for n in ast.walk(forward)if isinstance(n,ast.If)and'V_ORDER_NULL'in ast.unparse(n.test)]
    checks['NULL_only_RGB_zero_mask_preserved']=len(null)==1 and ast.unparse(null[0].body[0])=="patches = torch.cat((torch.zeros_like(patches[:, :, :3]), patches[:, :, 3:]), dim=2)"
    runner=next(n for n in trees['predict.py'].body if isinstance(n,ast.FunctionDef)and n.name=='run')
    with_nodes=[n for n in runner.body if isinstance(n,ast.With)and any(ast.unparse(i.context_expr)=='guard'for i in n.items)]
    checks['prediction_guard_present']=len(with_nodes)==1
    guarded=set(id(n)for n in ast.walk(with_nodes[0]))if with_nodes else set()
    input_calls=[n for n in ast.walk(runner)if isinstance(n,ast.Call)and ast.unparse(n.func)in('read','rows','torch.load','OLD.bound')]
    checks['all_prediction_input_loads_inside_guard']=all(id(n)in guarded for n in input_calls)
    checks['prediction_deny_resolved_original_references']='denied_files' in ast.unparse(runner)and 'FIXED_REFERENCES.json'in ast.unparse(runner)
    checks['source_read_root_scripts_only']='Path(__file__).resolve().parents[2]'in ast.unparse(runner)
    provider_calls=[n for n in ast.walk(trees['train.py'])if isinstance(n,ast.Call)and ast.unparse(n.func)=='visual_provider']
    checks['training_provider_receives_reference_free_fields']=len(provider_calls)==1 and[ast.unparse(a)for a in provider_calls[0].args[:2]]==["item['reference_free_frame']","item['entries']"]
    return checks,{n:sha(namespace/n)for n in trees}


def run(repo,archive,run_dir,output):
    from . import visual_patches as production
    from scripts.research.pallet_boundary_action_repair_v1.gallery import EDGES
    repo=Path(repo).resolve();archive=Path(archive).resolve();run_dir=Path(run_dir).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();cpu=time.process_time();checks=Counter();failures=[];stats=Counter();sourcechecks,sourcehash=source_contracts(repo)
    def check(name,condition,detail=None):
        checks[name]+=1
        if not bool(condition):failures.append(dict(check=name,detail=detail))
    try:
        for name,value in sourcechecks.items():check('source/'+name,value)
        path=run_dir/'visual_inputs_v1/VISUAL_INPUTS.jsonl.gz';sealpath=path.with_name('INPUT_SEAL.json');seal=read(sealpath)
        boxpath=run_dir/'v_box_inputs_v1/V_BOX_INPUTS.jsonl.gz';boxseal=read(boxpath.with_name('V_BOX_SEAL.json'))
        check('visual_input_seal',sha(path)==seal['artifacts'][path.name]and seal['reference_inputs']is False)
        check('box_input_seal',sha(boxpath)==boxseal['output_sha256']==seal['box_inputs_sha256'])
        check('actual_original_adjacency',seal['adjacency']==[list(v)for v in EDGES])
        values=list(rows(path));boxes={r['id']:r for r in rows(boxpath)};up={};frame_paths={}
        for domain,filename in(('real','REAL238_UPSTREAM.jsonl.gz'),('synthetic','SYNTHETIC1003_UPSTREAM.jsonl.gz')):
            for index,r in enumerate(rows(archive/'d1_upstream_v1'/filename)):
                up[r['id']]=r;frame_paths[r['id']]=archive/f'd1_{domain}_candidates_v1/frames'/f'{index:04d}'/'E_RUN.jsonl.gz'
        check('exact1241_ID_coverage',len(values)==1241 and len({r['id']for r in values})==1241 and{r['id']for r in values}==set(up)==set(boxes))
        selected=[r for r in values if r['domain']=='synthetic'and r['split']=='train'][:2];selected_by={r['id']:r for r in selected};selected_frames={}
        for f in rows(archive/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz'):
            if f['id']in selected_by:selected_frames[f['id']]=f
        for row in values:
            fid=row['id'];original=up[fid];box=boxes[fid];e_path=frame_paths[fid]
            check('exact_raw_geometry_ID_join',all(row[k]==original[k]for k in('partition','split','image','image_sha256','raw_hw','K','native_xyz')),fid)
            check('original_detector_box_join',row['detector_box_xyxy_raw']==box['detector_box_xyxy_raw']and row['box_binding']==box,fid)
            check('original_E_RUN_hash',sha(e_path)==row['e_run_sha256'],fid)
            e=next(rows(e_path));family=e['families']['ORIGINAL_WD']
            check('actual_original_native_Q',e['id']==fid and row['native_xyz']==family['native_xyz']and row['physical_registration_Q']==family['physical_registration_Q'],fid)
            q=np.asarray(row['physical_registration_Q']);check('proper_registered_Q',q.shape==(3,3)and np.max(np.abs(q.T@q-np.eye(3)))<=1e-7 and abs(np.linalg.det(q)-1.)<=1e-7,fid)
            check('adjacency_matches_native',np.array_equal(queries(row['native_xyz'],EDGES),production.original_model_queries(row['native_xyz'])),fid)
            coordinates=row['detector_box_xyxy_raw'];expected_side=None if coordinates is None else .05*math.hypot(coordinates[2]-coordinates[0],coordinates[3]-coordinates[1])
            check('raw_original_box_size',row['patch_side_raw_px']is None if expected_side is None else abs(row['patch_side_raw_px']-expected_side)<=8*np.finfo(float).eps*max(1.,expected_side),fid)
            stats[row['domain']+'/frames']+=1
            if fid not in selected_by:continue
            frame=selected_frames[fid];actions=independent_actions(frame);action=next(a for a in actions if a['is_N0'])
            with Image.open(row['image'])as image:rgb=np.asarray(image.convert('RGB'),np.uint8)
            check('actual_train_RGB_hash_canvas',sha(row['image'])==row['image_sha256']and list(rgb.shape[:2])==row['raw_hw'],fid)
            xy,z=project(action['actual_pose'],row,EDGES);q2,z2=production.project_queries(action['actual_pose'],row)
            pd=float(np.max(np.abs(xy-q2)));zd=float(np.max(np.abs(z-z2)));check('independent_actual_projection',pd<=128*np.finfo(float).eps*max(1.,float(np.max(np.abs(q2))))and zd<=128*np.finfo(float).eps*max(1.,float(np.max(np.abs(z2)))),fid)
            expected,valid,query_valid=bilinear(rgb,q2,z2,row['patch_side_raw_px']);actual,av,aq=production.sample(rgb,q2,z2,row['patch_side_raw_px'])
            check('independent_actual_bilinear_FP32_bitwise',np.array_equal(expected,actual),fid)
            check('independent_actual_masks_bitwise',np.array_equal(valid,av)and np.array_equal(query_valid,aq),fid)
            check('technical_FP16_FP32_contract',np.array_equal(expected.astype(np.float16).astype(np.float32),actual.astype(np.float16).astype(np.float32))and np.array_equal(actual[:,3],actual[:,3].astype(np.float16).astype(np.float32)),fid)
            stats['technical_train_N0_actions']+=1;stats['technical_regions']+=20;stats['technical_projection_max_abs_px']=max(stats['technical_projection_max_abs_px'],pd)
        check('fixed_first2train_only_actual_RGB',stats['technical_train_N0_actions']==2)
        result=dict(schema='RISK_INDEPENDENT_VISUAL_INPUT_CHECK_V1',status='PASS'if not failures else'FAIL',checks=sum(checks.values()),checks_by_kind=dict(checks),failure_count=len(failures),failures=failures,counts=dict(stats),source_contracts=sourcechecks,source_sha256=sha(__file__),production_source_sha256=sourcehash,
            visual_inputs_sha256=sha(path),visual_input_seal_sha256=sha(sealpath),box_inputs_sha256=sha(boxpath),
            independent_scope='Independent scalar projection/bilinear on fixed first2train N0;1241 raw/K/native/Q/box/E_RUN joins; AST input-whitelist/guard/NULL contract. No neural model instantiated or forwarded and no optimizer created.',
            real_reference_reads=0,neural_forward_calls=0,optimizer_created=0,new_PnP_solves=0,bulk_V_cache_generated=False,
            cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1))
        write(output/'VERIFICATION.json',result);return result
    except BaseException as exc:
        write(output/'FAILED_ATTEMPT.json',dict(error=repr(exc),traceback=traceback.format_exc(),checks=dict(checks),failures=failures,wall_seconds=time.monotonic()-started));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.repo,a.archive,a.run_dir,a.output);print(json.dumps({k:r[k]for k in('status','checks','failure_count','counts','cost')}));raise SystemExit(0 if r['status']=='PASS'else 1)
