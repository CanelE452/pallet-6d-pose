"""Portable CPU replay of six saved V270 SYN slot-prior diagnosis tables.

--prepare snapshots only declared SYN metadata and numeric saved predictions.
Replay reads four compact NPZs and source code; no RGB, original annotation
path, real data, model, or optimizer is needed. Prior labels stay unchanged.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import shutil
import types

import numpy as np

HERE=Path(__file__).absolute().parent
GENERATOR_SHA='64aa2fcaf35d7d5c1d6b971fdf2afbb4d8f9e5fafc25f8fc1e6142230f2f9baf'
TABLES=('SYN_SLOT_LABEL_CENSUS.csv','SYNDEV_SLOT_PREDICTIONS.csv','SOURCE_FRACTION_COUNTS.csv',
        'WORLD_Z_SLOT_GEOMETRY.csv','POSTHOC_SLOT_PRIOR_BASELINES.csv','NO_GLOBAL_M0_M4_DIFFERENCE_FRACTIONS.csv')


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112','cache','.pyc','..')):raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def json_read(path):return json.loads(safe(path).read_text())


def checked(bound):
    assert binding(bound['path'])==bound,bound
    return safe(bound['path'])


def savej(path,x):
    p=safe(path);assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def npz(path):
    with np.load(safe(path),allow_pickle=False)as z:return {k:z[k] for k in z.files}


def prepare(directory):
    directory=safe(directory);assert not directory.exists(),directory;directory.mkdir()
    generator=safe(HERE/'diagnose_slot_priors_v270.py');assert binding(generator)['sha256']==GENERATOR_SHA
    nc=json_read(HERE/'syn_joint_targets_v1/COMPLETE.json')
    rows=[json.loads(x) for x in checked(nc['outputs']['FRAME_TARGETS1003.jsonl']).read_text().splitlines()]
    manifest=[json.loads(x) for x in safe(HERE.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK/data_audit/FRAME_MANIFEST.jsonl').read_text().splitlines()]
    world=[];ground=[];owner=[];m0=[]
    for r,m in zip(rows,manifest):
        assert r['id']==m['input']['id']
        obj=json.loads(checked(r['annotation']).read_text())['objects'][0]
        world.append(obj['cuboid']);sp=obj['scene_placement_v2'];ground.append(sp['support_ground_z'])
        owner.append(str(sp['support_surface_name']));m0.append(m['target']['M0_only_area'])
    np.savez_compressed(safe(directory/'GEOMETRY_INPUTS.npz'),world_cuboid=np.asarray(world,np.float64),
        support_ground_z=np.asarray(ground,np.float64),support_owner=np.asarray(owner),M0_only_area=np.asarray(m0,np.int64))
    sources={'NATIVE_PROXY_TARGETS8.npz':checked(nc['outputs']['TARGETS.npz']),
        'VIS_ONLY_SYNDEV235.npz':safe(HERE/'syn_training_v1/VIS_ONLY_SYNDEV235.npz'),
        'JOINT_SYNDEV235.npz':safe(HERE/'syn_training_v1/JOINT_SYNDEV235.npz')}
    for name,p in sources.items():shutil.copyfile(p,safe(directory/name))
    files={}
    for name in ('GEOMETRY_INPUTS.npz',*sources):
        b=binding(directory/name);files[name]={'file':name,'sha256':b['sha256'],'size_bytes':b['size_bytes']}
    expected={}
    for n in TABLES:
        b=binding(HERE/'slot_prior_diagnosis_v1'/n);expected[n]={'sha256':b['sha256'],'size_bytes':b['size_bytes']}
    savej(directory/'COMPACT_INPUTS.json',{'completed':True,'PASS':True,'code':binding(Path(__file__).absolute()),
        'generator_code':binding(generator),'source_diagnosis_complete':binding(HERE/'slot_prior_diagnosis_v1/COMPLETE.json'),
        'source_annotation_bindings':binding(HERE/'slot_prior_diagnosis_v1/ANNOTATION_BINDINGS.json'),
        'files':files,'expected_tables':expected,'geometry_frame_order':'Same original1003 frame_index order as native targets.',
        'RGB_or_mask_or_current_reads':0,'model_forwards':0,'optimizer_steps':0,
        'prepare_only_reads_original_SYN_metadata':True})


def replay(directory,out):
    directory=safe(directory);manifest=json_read(directory/'COMPACT_INPUTS.json');assert manifest['PASS']
    data={}
    for n,b in manifest['files'].items():
        assert Path(b['file']).name==b['file'];p=safe(directory/b['file']);a=binding(p)
        assert a['sha256']==b['sha256'] and a['size_bytes']==b['size_bytes'];data[n]=npz(p)
    generator=safe(HERE/'diagnose_slot_priors_v270.py');assert binding(generator)['sha256']==GENERATOR_SHA
    mod=types.ModuleType('v270_slot_generator_for_replay');mod.__file__=str(generator)
    exec(compile(generator.read_bytes(),str(generator),'exec'),mod.__dict__)
    a=data['NATIVE_PROXY_TARGETS8.npz'];g=data['GEOMETRY_INPUTS.npz']
    train=np.flatnonzero(a['fit']);dev=np.flatnonzero(~a['fit']);assert len(train)==768 and len(dev)==235
    census=[];fractions=[]
    for fold,ix in [('train',train),('validation',dev)]:
        for k in range(8):
            loc=a['loc_valid'][ix,k];sv=a['source_convex_self_visible'][ix,k];q=a['external_ray_fraction'][ix,k];y=a['visibility_y'][ix,k]
            D=int((y==0).sum());E=int((y==1).sum());valid=loc&sv
            census.append({'split':fold,'raw_model_slot':k,'N':len(ix),'D_strict_fraction0':D,'E_fraction_GE_half':E,
                'partial_0_LT_fraction_LT_half_excluded':int((valid&(q>0)&(q<.5)).sum()),
                'self_inside_excluded':int((loc&~sv).sum()),'outside_excluded':int((~loc).sum()),
                'strict_E_prevalence':E/(D+E) if D+E else None,
                'source_LT_half_negative_count_diagnostic_only':int((valid&(q<.5)).sum()),
                'source_LT_half_count_is_not_relabeling':True})
            for scope,mask in [('ALL',np.ones(len(ix),bool)),('SOURCE_SELF_VISIBLE_INFRAME',valid)]:
                values=collections.Counter(f'{x:.4f}' for x in q[mask])
                for fraction,n in sorted(values.items()):fractions.append({'split':fold,'raw_model_slot':k,'scope':scope,'fraction':fraction,'N':n})
    geoby=collections.defaultdict(list);noocc=collections.Counter()
    for i,pts in enumerate(g['world_cuboid']):
        z=pts[:,2];zmin=float(z.min());zmax=float(z.max());span=zmax-zmin;assert span>0
        ground=float(g['support_ground_z'][i])
        for k in range(8):
            relative=(float(z[k])-zmin)/span
            role='LOWER_Z' if abs(relative)<1e-4 else 'UPPER_Z' if abs(relative-1)<1e-4 else 'OTHER_Z'
            geoby[k].append({'relative_height':relative,'ground_offset':float(z[k]-ground),'role':role})
            if g['M0_only_area'][i]==0 and a['loc_valid'][i,k] and a['source_convex_self_visible'][i,k]:
                # Original JSON fractions are 4 decimal values; float32 formats back identically.
                noocc[('train' if a['fit'][i] else 'validation',k,f"{a['external_ray_fraction'][i,k]:.4f}")]+=1
    geometry=[]
    for k,values in sorted(geoby.items()):
        offset=[v['ground_offset'] for v in values]
        geometry.append({'raw_model_slot':k,'N':len(values),'LOWER_Z':sum(v['role']=='LOWER_Z' for v in values),
            'UPPER_Z':sum(v['role']=='UPPER_Z' for v in values),'OTHER_Z':sum(v['role']=='OTHER_Z' for v in values),
            'relative_height_min':min(v['relative_height'] for v in values),'relative_height_max':max(v['relative_height'] for v in values),
            'support_ground_z_available':len(offset),'support_ground_offset_min_m':float(min(offset)),
            'support_ground_offset_median_m':float(np.median(offset)),'support_ground_offset_max_m':float(max(offset))})
    y=a['visibility_y'][dev];known=y>=0
    priors=[float((a['visibility_y'][train,k]==1).sum()/max(1,(a['visibility_y'][train,k]>=0).sum())) for k in range(8)]
    prior=np.tile(priors,(235,1));baselines=[{'method':'POSTHOC_STATIC_SLOT_FIT_PREVALENCE',**mod.summarize(y[known],prior[known])}]
    slot_rows=[]
    for arm in ('VIS_ONLY','JOINT'):
        pred=data[f'{arm}_SYNDEV235.npz'];assert np.array_equal(pred['frame_index'],dev)
        prob=1/(1+np.exp(-np.clip(pred['E_logit'][:,:8].astype(np.float64),-60,60)));binary=prob>=.5
        baselines.append({'method':arm,**mod.summarize(y[known],prob[known])})
        for k in range(8):slot_rows.append({'method':arm,'raw_model_slot':k,'N_frames':235,
            'pred_D_all_frames':int((~binary[:,k]).sum()),'pred_E_all_frames':int(binary[:,k].sum()),
            'strict_D':int((y[:,k]==0).sum()),'strict_E':int((y[:,k]==1).sum()),
            'D_to_E':int(((y[:,k]==0)&binary[:,k]).sum()),'E_to_D':int(((y[:,k]==1)&~binary[:,k]).sum()),
            'E_logit_min':float(pred['E_logit'][:,k].min()),'E_logit_median':float(np.median(pred['E_logit'][:,k])),
            'E_logit_max':float(pred['E_logit'][:,k].max()),'slot_FIT_E_prevalence':priors[k]})
    noocc_rows=[{'split':k[0],'raw_model_slot':k[1],'fraction':k[2],'N':n} for k,n in sorted(noocc.items())]
    tables=dict(zip(TABLES,[census,slot_rows,fractions,geometry,baselines,noocc_rows]))
    out=safe(out);assert not out.exists(),out;out.mkdir();outputs={}
    for name,rows in tables.items():
        mod.csvwrite(out/name,rows);b=binding(out/name);expected=manifest['expected_tables'][name]
        assert b['sha256']==expected['sha256'] and b['size_bytes']==expected['size_bytes'],name
        outputs[name]=b
    savej(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':binding(Path(__file__).absolute()),
        'generator_code':binding(generator),'compact_inputs':binding(directory/'COMPACT_INPUTS.json'),
        'outputs':outputs,'six_tables_byte_identical':True,'bottom_slots_LOWER_Z_all1003':True,
        'raw_annotation_or_RGB_or_mask_or_current_reads_in_replay':0,'model_forwards':0,'optimizer_steps':0,
        'prior_baseline':baselines[0]})
    print(json.dumps({'COMPLETE':binding(out/'COMPLETE.json'),'six_tables_byte_identical':True,'prior_baseline':baselines[0]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    p.add_argument('--inputs',default='slot_prior_compact_inputs_v1');p.add_argument('--out',default='slot_prior_metrics_replay_v1')
    args=p.parse_args();directory=safe(HERE/args.inputs)
    if args.prepare:prepare(directory)
    replay(directory,safe(HERE/args.out))
