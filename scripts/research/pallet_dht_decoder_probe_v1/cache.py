"""Frozen eval-mode image/line cache for the small decoder pilot.

The CNN receives raw BGR only. Synthetic targets are read after top-confidence
instance selection and are used solely for loss matching. Real GT is not read.
FP32 features, logits and coordinate transforms are retained for replay.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODES = ('point_only', 'line_fusion', 'wrong_image_line')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value: raise ValueError(f'Bound artifact changed: {path}')
    else: write(path, value)


def save_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.npz')
    np.savez(temporary, **values)
    temporary.replace(path)


def source_hashes():
    names = ['scripts/research/pallet_dht_decoder_probe_v1/'+n for n in ['cache.py','model.py','geometry.py']]
    names += ['scripts/research/pallet_dht_joint_v1/'+n for n in ['evaluate.py','integration.py','hough_block.py']]
    names += ['scripts/research/deep_hough_side_v1/dht.py','scripts/research/pallet_line_pose_v1/source_data.py']
    return {str(ROOT / n): sha(ROOT / n) for n in names}


def verify_hashes(values):
    for path, expected in values.items():
        if sha(path) != expected: raise ValueError(f'Input/source SHA mismatch: {path}')


def prepare_smoke(parent):
    parent = Path(parent).resolve(); run = parent / 'smoke'
    config = copy.deepcopy(read(parent / 'TRAIN_PROTOCOL.json'))
    config['stage'] = 'smoke'
    config['data'].update(train_count=8, val_count=4)
    config['training'].update(steps=4, batch=4, val_every=2)
    config['evaluation'].update(real_count=0)
    config['parent_protocol_sha256'] = sha(parent / 'TRAIN_PROTOCOL.json')
    freeze(run / 'TRAIN_PROTOCOL.json', config)
    purpose = run / 'PURPOSE.md'
    if not purpose.exists(): purpose.write_text('[소비처] 부모 decoder pilot의 구현 스모크 검증.\n[문장] 합성12장·4업데이트에서 좌표/초기동일성/학습 연결을 검증하며 본 실험 결과로 취급하지 않는다.\n')
    return run


def metadata(run):
    run = Path(run).resolve(); config = read(run / 'TRAIN_PROTOCOL.json')
    data = config['data']
    verify_hashes({data['source_manifest']: data['source_manifest_sha256'],
        data['real_manifest']: data['real_manifest_sha256'],
        config['backbone']['checkpoint']: config['backbone']['sha256']})
    source = read(data['source_manifest'])
    records = []
    for split, count in [('train', data['train_count']), ('val', data['val_count'])]:
        candidates = [r for r in source['records'] if r['source_split'] == split]
        candidates.sort(key=lambda r: hashlib.sha256(f"{data['subset_salt']}:{split}:{r['id']}".encode()).hexdigest())
        for item in candidates[:count]:
            records.append(dict(index=len(records), id=item['id'], population='synth_'+split,
                image=item['image'], image_key=item['image'], image_sha256=item['image_sha256'],
                width=item['raw_shape_hw'][1], height=item['raw_shape_hw'][0], session_id=item['source'],
                source_index=item['index'], source=item['source'], source_record=item,
                prepared_image=True))
    if config['stage'] == 'main':
        from scripts.research.pallet_dht_joint_v1.evaluate import population
        pair = population()
        for item in pair.positive.items:
            path = (ROOT / item.image).resolve()
            # Metadata/image bytes are permitted; no real GT annotation is read.
            import cv2
            image = cv2.imread(str(path))
            if image is None: raise ValueError(f'Cannot read image: {path}')
            records.append(dict(index=len(records), id=item.frame_id, population='real_dev',
                image=str(path), image_key=str(item.image), image_sha256=sha(path),
                width=image.shape[1], height=image.shape[0], session_id=getattr(item,'session_id',None) or item.frame_id.split(':')[0],
                source_index=None, source='real_DEV', prepared_image=False))
    if len({r['id'] for r in records}) != len(records): raise ValueError('Duplicate selected ID')
    train_ids = {r['id'] for r in records if r['population']=='synth_train'}
    val_ids = {r['id'] for r in records if r['population']=='synth_val'}
    if train_ids & val_ids: raise ValueError('Synthetic train/validation overlap')
    train_hashes={r['image_sha256'] for r in records if r['population']=='synth_train'}
    val_hashes={r['image_sha256'] for r in records if r['population']=='synth_val'}
    if train_hashes & val_hashes: raise ValueError('Synthetic image-byte train/validation overlap')
    result = dict(schema='decoder_probe_manifest_v1', complete=True, stage=config['stage'],
        protocol_sha256=sha(run/'TRAIN_PROTOCOL.json'), records=records,
        populations={p:[r['index'] for r in records if r['population']==p] for p in ['synth_train','synth_val','real_dev']},
        decoder_train_val_disjoint=True, decoder_train_val_image_sha_disjoint=True,
        backbone_saw_original_train_and_validation=True,
        held_out_final=False, real_GT_read=False)
    freeze(run/'MANIFEST.json', result)
    return result


class FrozenCapture:
    def __init__(self, checkpoint, device='0'):
        from scripts.research.pallet_dht_joint_v1.evaluate import CanonicalPredictor
        self.predictor = CanonicalPredictor(checkpoint, device)
        self.evidence = None
        modules = [m for m in self.predictor.model.model.modules() if m.__class__.__name__ == 'HoughFeatureFusion']
        if len(modules) != 1: raise ValueError('Expected one HoughFeatureFusion')
        self.handle = modules[0].register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        lattice = module.lattice
        p4 = inputs[0][1]
        self.evidence = dict(p4=p4[0].detach().float().cpu().numpy().copy(),
            logits=module.line_logits[0].detach().float().cpu().numpy().copy(),
            theta=lattice['theta_values'].detach().float().cpu().numpy().copy(),
            rho=lattice['rho_values'].detach().float().cpu().numpy().copy(),
            lattice_valid=lattice['valid'].detach().cpu().numpy().copy(),
            feature_shape_hw=np.array(p4.shape[-2:], np.int64), input_shape_hw=np.array(p4.shape[-2:], np.int64)*16)

    def predict(self, raw):
        self.evidence = None
        candidates, ms = self.predictor.predict(raw)
        if self.evidence is None: raise ValueError('Missing Hough/P4 hook')
        return candidates, self.evidence, ms

    def state_sha(self):
        h = hashlib.sha256()
        for name, value in self.predictor.model.model.state_dict().items():
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


def load_raw(record):
    import cv2
    if sha(record['image']) != record['image_sha256']: raise ValueError('Source image changed')
    image = cv2.imread(record['image'])
    if image is None: raise ValueError('Image decode failed')
    raw = image[100:-100, 100:-100].copy() if record['prepared_image'] else image
    if raw.shape[:2] != (record['height'], record['width']): raise ValueError('Raw image shape differs')
    if record['prepared_image']:
        reconstructed = cv2.copyMakeBorder(raw,100,100,100,100,cv2.BORDER_REFLECT_101)
        if not np.array_equal(reconstructed,image): raise ValueError('Source border is not the declared reflect101')
    return raw


def extract(run, manifest, *, device='0'):
    import torch
    import cv2
    from scripts.research.pallet_line_pose_v1 import source_data as SD
    run = Path(run); config = read(run/'TRAIN_PROTOCOL.json')
    bindings = dict(protocol_sha256=sha(run/'TRAIN_PROTOCOL.json'),manifest_sha256=sha(run/'MANIFEST.json'),
        checkpoint_sha256=config['backbone']['sha256'],source_sha256=source_hashes())
    freeze(run/'EXTRACTION_PROTOCOL.json',bindings)
    target = run/'EXTRACTION_COMPLETE.json'
    if target.exists():
        done = read(target)
        if not done['complete'] or done['bindings']!=bindings: raise ValueError('Existing extraction binding mismatch')
        verify_hashes(done['frame_sha256']);return done
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    model = FrozenCapture(config['backbone']['checkpoint'],device)
    first = load_raw(manifest['records'][0])
    for _ in range(5): model.predict(first)
    before = model.state_sha(); started=time.perf_counter();frames={};durations=[]
    for record in manifest['records']:
        path = run/'frames'/f"{record['index']:06d}.npz"
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                cached= json.loads(str(data['record_json']))
                if cached['image_sha256']!=record['image_sha256'] or str(data['bindings_json'])!=json.dumps(bindings,sort_keys=True):
                    raise ValueError('Partial frame binding differs')
            frames[str(path.resolve())]=sha(path);continue
        raw=load_raw(record)
        candidates,evidence,ms=model.predict(raw);durations.append(ms)
        selected = int(np.argmax([c['score'] for c in candidates])) if candidates else None
        item = candidates[selected] if selected is not None else None
        points = np.array(item['keypoints_xy'],np.float32) if item and item['keypoints_xy'] is not None else np.zeros((9,2),np.float32)
        valid = np.isfinite(points).all(-1) if item and item['keypoints_xy'] is not None else np.zeros(9,bool)
        confidence = np.array(item['keypoints_conf'],np.float32) if item and item['keypoints_conf'] is not None else np.zeros(9,np.float32)
        points[~valid]=0
        baseline=dict(points=points.tolist(),point_valid=valid.tolist(),point_conf=confidence.tolist(),
            box_xyxy=item['box_xyxy'] if item else [0.,0.,0.,0.],score=item['score'] if item else 0.,
            detected=item is not None,selected_instance=selected,all_candidates=candidates)
        tf=SD.transform((record['height']+200,record['width']+200),evidence['input_shape_hw'])
        affine=np.array([[tf.gain,0,100*tf.gain+tf.pad_ltrb[0]],[0,tf.gain,100*tf.gain+tf.pad_ltrb[1]]],np.float64)
        info={k:v for k,v in record.items() if k!='source_record'}
        info.update(baseline=baseline,raw_to_input_affine=affine.tolist(),prediction_inverse_pad_xy=list(tf.scale_coords_pad_xy),
            input_shape_hw=evidence['input_shape_hw'].tolist(),feature_shape_hw=evidence['feature_shape_hw'].tolist(),
            inference_ms=ms,transform=tf.to_dict())
        gt=np.zeros((9,2),np.float32);gt_valid=np.zeros(9,bool)
        match={'target_index':None,'iou':0.,'matched':False}
        if record['population'].startswith('synth_'):
            source_record=record['source_record']
            if sha(source_record['label'])!=source_record['label_sha256']: raise ValueError('Synthetic label changed')
            # This is downstream of the already-frozen score-only selection.
            targets=SD.load_loss_targets(source_record,tf)
            raw_boxes=(targets['boxes_xyxy'].reshape(-1,2,2)-np.array(tf.pad_ltrb[:2]))/tf.gain-100
            if item is not None: match=SD.match_loss_target(baseline['box_xyxy'],raw_boxes.reshape(-1,4),minimum_iou=.5)
            if match['matched']:
                index=match['target_index'];gt_valid=targets['kp_valid'][index]
                gt=tf.input_to_original(targets['kps'][index]).astype(np.float32);gt[~gt_valid]=0
            info.update(synth_gt=gt.tolist(),synth_gt_valid=gt_valid.tolist(),loss_matched=match['matched'],
                matched_gt_index=match['target_index'],matching_iou=match['iou'],synthetic_targets_only_for_loss=True)
        evidence.update(raw_to_input_affine=affine,gt_points=gt,gt_valid=gt_valid,
            record_json=np.array(json.dumps(info,sort_keys=True)),bindings_json=np.array(json.dumps(bindings,sort_keys=True)))
        save_npz(path,**evidence);frames[str(path.resolve())]=sha(path)
        if (record['index']+1)%100==0: print(f"Frozen extraction {record['index']+1}/{len(manifest['records'])}",flush=True)
    after=model.state_sha()
    if before!=after: raise ValueError('Frozen model/BN state changed')
    verify_hashes(bindings['source_sha256'])
    result=dict(schema='decoder_probe_frozen_extraction_v1',complete=True,PASS=True,bindings=bindings,
        frames=len(frames),frame_sha256=frames,warmup_forwards=5,actual_new_forwards=len(durations),
        resumed_frames=len(frames)-len(durations),frozen_state_sha_before=before,frozen_state_sha_after=after,
        frozen_eval_state_unchanged=True,real_GT_read=False,elapsed_seconds=time.perf_counter()-started,
        median_inference_ms=float(np.median(durations)) if durations else None,
        precision='FP32',P4='Pre-Hough-fusion neck P4,128channels; includes the previously trained backbone but no extra explicit Hough feedback in this tensor.',
        torch_version=torch.__version__,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    write(target,result);model.handle.remove();del model
    if torch.cuda.is_initialized():torch.cuda.empty_cache()
    return result


def sample_visual(p4, raw_xy, affine, input_shape_hw):
    """Bilinear actual-content mapping, feature centres at(input/16)-0.5.

    Out-of-input candidates receive zero; no GT crop or point correction.
    """
    import torch
    import torch.nn.functional as F
    xy=np.asarray(raw_xy,float); flat=xy.reshape(-1,2)@affine[:,:2].T+affine[:,2]
    ih,iw=map(int,input_shape_hw)
    grid=2*flat/np.array([iw,ih])-1
    with torch.no_grad():
        sampled=F.grid_sample(torch.from_numpy(p4)[None],torch.tensor(grid,dtype=torch.float32)[None,None],
            mode='bilinear',padding_mode='zeros',align_corners=False)[0,:,0].t().numpy()
    return sampled.reshape(*xy.shape[:-1],128).copy()


def point_bank(points,conf,valid,diagonal,affine,input_shape):
    offsets=[(x,y) for y in range(-3,4) for x in range(-3,4)
        if (x,y)!=(0,0) and not (abs(x)==3 and abs(y)==3) and not (x==0 and abs(y)==3) and not (y==0 and abs(x)==3)]
    assert len(offsets)==40
    xy=np.zeros((8,49,2),np.float64);mask=np.zeros((8,49),bool);features=np.zeros((8,49,18),np.float64)
    ih,iw=input_shape;inputdiag=math.hypot(ih,iw)
    for i in range(8):
        order=[i]+[j for j in range(9) if j!=i]
        xy[i,:9]=points[order];mask[i,:9]=valid[order];mask[i,0]=True
        xy[i,9:]=points[i]+np.array(offsets)*(.05*diagonal/3)
        mask[i,9:]=valid[i]
        q=xy[i]@affine[:,:2].T+affine[:,2];p=points[i]@affine[:,:2].T+affine[:,2]
        features[i,0,0]=1;features[i,:,1:3]=(q-p)/inputdiag
        features[i,:,3:5]=(q-np.array([iw/2,ih/2]))/inputdiag;features[i,:,5]=conf[i]
        mask[i,9:] &= ((q[9:]>=0)&(q[9:]<=np.array([iw,ih]))).all(-1)
    return dict(candidates_xy=xy,candidate_valid=mask,candidate_features=features)


COMMON_SPECS={'points':((9,2),'float32'),'point_conf':((9,),'float32'),'point_valid':((9,),'bool'),
    'boxes':((4,),'float32'),'diagonal':((),'float32'),'detected':((),'bool'),
    'point_visual':((8,128),'float32'),'global_visual':((128,),'float32'),
    'gt_points':((9,2),'float32'),'gt_valid':((9,),'bool'),'loss_valid':((8,),'bool')}
CANDIDATE_SPECS={'candidate_xy':((8,49,2),'float32'),'candidate_valid':((8,49),'bool'),
    'candidate_geometry':((8,49,18),'float32'),'candidate_visual':((8,49,128),'float32')}


def features(run,manifest):
    from .geometry import build_candidates,FEATURE_NAMES
    run=Path(run);config=read(run/'TRAIN_PROTOCOL.json');n=len(manifest['records'])
    binding=dict(protocol_sha256=sha(run/'TRAIN_PROTOCOL.json'),manifest_sha256=sha(run/'MANIFEST.json'),
        extraction_sha256=sha(run/'EXTRACTION_COMPLETE.json'),source_sha256=source_hashes())
    if (run/'CACHE_COMPLETION.json').exists():
        done=read(run/'CACHE_COMPLETION.json')
        if not done['complete'] or done['bindings']!=binding:raise ValueError('Completed cache binding differs')
        verify_hashes(done['array_sha256']);return done
    arrays={};descriptions={}
    for mode,specs in [('common',COMMON_SPECS)]+[(m,CANDIDATE_SPECS) for m in MODES]:
        arrays[mode]={};descriptions[mode]={}
        for key,(tail,dtype) in specs.items():
            path=run/'arrays'/mode/(key+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
            arrays[mode][key]=np.lib.format.open_memmap(path,mode='w+',shape=(n,*tail),dtype=dtype)
            descriptions[mode][key]=dict(path=str(path.resolve()),shape=[n,*tail],dtype=dtype)
    records=[];donors={};evidence_hashes={}
    for pop in ['synth_val','real_dev']:
        ids=sorted(manifest['populations'][pop],key=lambda i:hashlib.sha256(manifest['records'][i]['id'].encode()).hexdigest())
        donors.update({i:ids[(j+1)%len(ids)] for j,i in enumerate(ids)})
    started=time.perf_counter()
    for record in manifest['records']:
        index=record['index'];path=run/'frames'/f'{index:06d}.npz'
        with np.load(path,allow_pickle=False) as source:
            info=json.loads(str(source['record_json']));b=info['baseline'];points=np.array(b['points'],np.float32)
            conf=np.array(b['point_conf'],np.float32);valid=np.array(b['point_valid'],bool)
            affine=source['raw_to_input_affine'];p4=source['p4'];shape=source['input_shape_hw'];diagonal=math.hypot(record['width'],record['height'])
            common=dict(points=points,point_conf=conf,point_valid=valid,boxes=b['box_xyxy'],diagonal=diagonal,
                detected=b['detected'],point_visual=sample_visual(p4,points[:8],affine,shape),global_visual=p4.mean((1,2)),
                gt_points=source['gt_points'],gt_valid=source['gt_valid'],
                loss_valid=source['gt_valid'][:8]&valid[:8]&bool(info.get('loss_matched',False)))
            for key,value in common.items():arrays['common'][key][index]=value
            def line_bank(data,used_affine):
                return build_candidates(data['logits'],data['theta'],data['rho'],data['lattice_valid'],
                    data['feature_shape_hw'],data['input_shape_hw'],used_affine,points,conf,config=config['geometry'])
            line=line_bank(source,affine)
            point=point_bank(points,conf,valid,diagonal,affine,shape)
            banks={'line_fusion':line,'point_only':point}
            donor_index=donors.get(index)
            if donor_index is not None:
                with np.load(run/'frames'/f'{donor_index:06d}.npz',allow_pickle=False) as donor:
                    # Recipientraw -> recipientinput normalized -> donorinput.
                    scale=np.array(donor['input_shape_hw'][::-1])/np.array(shape[::-1])
                    wrong_affine=affine*scale[:,None]
                    banks['wrong_image_line']=line_bank(donor,wrong_affine)
            else:
                # No wrong-line training arm. These rows are deliberately marked unavailable.
                banks['wrong_image_line']=point
            evidence={}
            for mode,bank in banks.items():
                mapping=dict(candidate_xy=bank['candidates_xy'],candidate_valid=bank['candidate_valid'],
                    candidate_geometry=bank['candidate_features'],
                    candidate_visual=sample_visual(p4,bank['candidates_xy'],affine,shape))
                for key,value in mapping.items():arrays[mode][key][index]=value
                for key in ['candidates_xy','candidate_valid','candidate_sources','line_peaks_h_raw','line_peak_valid','line_peak_theta_rho','incident_roles']:
                    if key in bank:evidence[mode+'__'+key]=np.asarray(bank[key])
            evidence_path=run/'candidate_evidence'/f'{index:06d}.npz';save_npz(evidence_path,**evidence)
            evidence_hashes[str(evidence_path.resolve())]=sha(evidence_path)
            info.update(candidate_evidence_npz=str(evidence_path.resolve()),donor_id=manifest['records'][donor_index]['id'] if donor_index is not None else None,
                wrong_line_available=donor_index is not None)
            records.append(info)
        if (index+1)%100==0:print(f'Candidate/P4 sampling {index+1}/{n}',flush=True)
    hashes={}
    for mode,items in arrays.items():
        for key,array in items.items():
            array.flush();path=descriptions[mode][key]['path'];hashes[path]=sha(path)
    cache=dict(schema='decoder_probe_model_cache_v1',complete=True,bindings=binding,frames=n,
        arrays=descriptions,feature_names=list(FEATURE_NAMES),candidate_count=49,
        wrong_line_available_populations=['synth_val','real_dev'],precision='FP32')
    write(run/'CACHE_MANIFEST.json',cache)
    write(run/'CACHE_RECORDS.json',dict(schema='decoder_probe_cache_records_v1',complete=True,records=records,
        manifest_sha256=sha(run/'MANIFEST.json'),cache_manifest_sha256=sha(run/'CACHE_MANIFEST.json')))
    result=dict(schema='decoder_probe_cache_completion_v1',complete=True,PASS=True,stage=config['stage'],bindings=binding,
        frames=n,array_sha256=hashes,candidate_evidence_sha256=evidence_hashes,cache_manifest_sha256=sha(run/'CACHE_MANIFEST.json'),
        cache_records_sha256=sha(run/'CACHE_RECORDS.json'),elapsed_feature_seconds=time.perf_counter()-started,
        counts={pop:dict(frames=len(indices),detected=int(np.asarray(arrays['common']['detected'])[indices].sum()),
            loss_supervised_frames=int(np.asarray(arrays['common']['loss_valid'])[indices].any(1).sum()),
            supervised_corner_count=int(np.asarray(arrays['common']['loss_valid'])[indices].sum()))
            for pop,indices in manifest['populations'].items()},real_GT_read=False)
    verify_hashes(binding['source_sha256']);write(run/'CACHE_COMPLETION.json',result)
    return result


def load_model_arrays(run_dir, mode='line_fusion'):
    run=Path(run_dir);cache=read(run/'CACHE_MANIFEST.json');done=read(run/'CACHE_COMPLETION.json')
    if not done['complete'] or sha(run/'CACHE_MANIFEST.json')!=done['cache_manifest_sha256']:
        raise ValueError('Incomplete/changed model cache')
    if mode not in MODES:raise ValueError('Unknown candidate mode')
    return {key:np.load(spec['path'],mmap_mode='r') for group in ['common',mode] for key,spec in cache['arrays'][group].items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--phase',choices=['metadata','extract','features','all'],default='all')
    parser.add_argument('--device',default='0');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run=prepare_smoke(args.run_dir) if args.smoke else args.run_dir.resolve()
    if not (run/'PURPOSE.md').is_file():raise ValueError('PURPOSE.md required')
    manifest=metadata(run)
    if args.phase in ['extract','all']:extract(run,manifest,device=args.device)
    if args.phase in ['features','all']:features(run,manifest)


if __name__=='__main__':main()
