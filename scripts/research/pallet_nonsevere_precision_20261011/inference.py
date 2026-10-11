"""One frozen detector pass, native correction, common final production F."""
import argparse,json,math,sys,time
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,torch
import precision_model as M
from train import read,write,sha

def configure():
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    torch.set_num_threads(2);cv2.setNumThreads(1)

def canonical(method):
    return {'N3_DIM_SYM':'N3','N3_THEN_SUBPIX':'N3_SUBPIX'}.get(method,method)

class InferenceEngine:
    def __init__(self,args,seed=1):
        configure();self.args=args;self.seed=seed
        env,refiner,_,Base,_,decode,_=M.source_modules(args.source_root)
        self.env,self.refiner,self.decode=env,refiner,decode
        self.features=env.old('features')
        self.extractor=self.features.FrozenYoloFeatures(env.R0,device='cuda')
        self.pose=__import__('pose')
        self.norm=read(env.DOC/'DIM_NORMALIZATION_LOCK.json')
        self.old_selection=read(env.C.B/'P_SELECTION.json')
        self.n3_selection=read(env.DOC/'CALIBRATION_AND_SELECTION.json')
        self.new_calibration=read(Path(args.public_dir)/'CALIBRATION.json') if (Path(args.public_dir)/'CALIBRATION.json').exists() else None
        self.heads={};self.parameters={};self.detector_forwards=0
        self.hook=self.extractor.yolo.model.model[0].register_forward_pre_hook(lambda *x:setattr(self,'detector_forwards',self.detector_forwards+1))
        methods=['OLD_P','N3','POSEFIX',*M.ARMS] if getattr(args,'method','ALL')=='ALL' else [canonical(args.method).removesuffix('_SUBPIX')]
        methods=[m for m in methods if m not in ('BASE','SUBPIX')]
        config=read(env.DOC/'TRAIN_PROTOCOL_LOCK.json')['config']
        for method in methods:
            if method=='POSEFIX':
                sys.path.insert(0,str(env.ROOT/'scripts/research/pallet_sensors_submission_v1'))
                import prior_model,posefix_contract_math
                self.prior_model,self.crop_math=prior_model,posefix_contract_math
                path=env.ROOT/f'data/pallet/results/pallet_sensors_submission_v1/runs/PRIOR{seed}/last.pt'
                ck=torch.load(path,map_location='cpu');assert ck['complete'] and ck['step']==6000
                head=prior_model.PoseFixPallet9();head.load_state_dict(ck['model_state_dict'])
            elif method=='OLD_P':
                path=env.C.BRAW/f'runs/seed{seed}/last.pt';ck=torch.load(path,map_location='cpu')
                head=Base(**ck['config']);head.load_state_dict(ck['model_state_dict'])
            elif method=='N3':
                path=env.RAW/f'runs/N3_DIM_SYM_seed{seed}/last.pt';ck=torch.load(path,map_location='cpu')
                head=refiner.model('N3_DIM_SYM',ck['config']);head.load_state_dict(ck['model_state_dict'])
            else:
                path=Path(args.private_dir)/f'checkpoints/{method}_seed{seed}.pt'
                if not path.exists():continue
                ck=torch.load(path,map_location='cpu');head=M.NativeEvidence(Base(**config));head.load_state_dict(ck['state_dict'])
            self.parameters[method]=sum(p.numel() for p in head.parameters())
            self.heads[method]=head.cuda().eval().requires_grad_(False)
            del ck
        self.methods=__import__('importlib.util').util.spec_from_file_location('nonsevere_sp',env.ROOT/'scripts/research/pallet_training_free_compare_20261007_v1/methods.py')
        module=__import__('importlib.util').util.module_from_spec(self.methods);self.methods.loader.exec_module(module);self.methods=module

    @torch.no_grad()
    def correct(self,image,method,captured,force_uncapped=False):
        method=canonical(method);subpix=method=='SUBPIX' or method.endswith('_SUBPIX')
        owner='BASE' if method=='SUBPIX' else method.removesuffix('_SUBPIX')
        selected=captured['selected_index']
        if selected is None:
            self.last_subpix_input=None
            return None,dict(detector_calls=1,corrector_forwards=0,subpix_corner_calls=0)
        candidate=captured['candidates'][selected];q0=np.asarray(candidate['keypoints_xy'],float).copy()
        support=np.isfinite(q0).all(-1)&~(q0==-1).all(-1);q=q0.copy();head_calls=0
        if owner=='POSEFIX':
            box=np.asarray(candidate['box_xyxy'],float)
            if np.isfinite(box).all() and (box[2:]>box[:2]).all():
                matrix=self.crop_math.axis_aligned_crop_matrix(box)
                crop=cv2.warpAffine(image,matrix[:2],(288,384),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                rgb=(crop[:,:,::-1].astype(np.float32)-np.array([123.68,116.78,103.94],np.float32)).transpose(2,0,1)
                points=self.crop_math.transform_points(np.where(support[:,None],q0,0),matrix).astype(np.float32)
                logits=self.heads[owner](torch.as_tensor(rgb,device='cuda')[None],torch.as_tensor(points,device='cuda')[None],torch.as_tensor(support,device='cuda')[None])
                qp=self.prior_model.expectation(logits)[0].cpu().numpy();q=self.crop_math.transform_points(qp,np.linalg.inv(matrix))
                q[~support]=q0[~support];q[8]=q0[8];head_calls=1
        elif owner!='BASE':
            inputs=self.features.branch_inputs(captured)
            batch={k:torch.as_tensor(inputs[k],device='cuda')[None] for k in ('points','boxes','point_valid','input_shape')}
            batch.update(p3=captured['p3'],p4=captured['p4'])
            head=self.heads[owner];tiles=None
            if owner=='N3':
                # The five-component original N3 context does not encode group order.
                dims=self.current_xyz[[0,2,1]]
                batch['context']=torch.as_tensor(self.refiner.context(dims[None],[2],self.norm,False),device='cuda')
                output=self.refiner.forward(head,batch)
                temperature=self.n3_selection['temperatures'][f'N3_DIM_SYM_seed{self.seed}']['temperature']
            elif owner=='OLD_P':
                output=head(*(batch[k] for k in ('p3','p4','points','boxes','point_valid','input_shape')),lam=0)
                temperature=self.old_selection['temperatures'][str(self.seed)]['temperature']
            else:
                if owner in ('M3','M4'):
                    h,w=image.shape[:2]
                    fake=SimpleNamespace(indices=np.array([0]),source={'records':[dict(raw_shape_hw=[h,w],prepared_shape_hw=list(captured['canvas_shape']),reflect_pad_px=captured['added_border'])]},arrays={'gain':np.array([inputs['gain']]),'input_shape':np.array([inputs['input_shape']])},_rgb_reader=lambda _:image[:,:,::-1])
                    tiles=M.native_tiles(fake,np.array([0]),batch,head.base.displacements)
                output=head(batch,owner in ('M3','M4'),tiles)
                temperature=self.new_calibration['temperatures'][f'{owner}_seed{self.seed}']['temperature']
            cap=.01*math.hypot(*image.shape[:2])*inputs['gain']
            uncapped_decoded=self.decode(output,temperature,1.,None)[0].cpu().numpy()
            self.last_subpix_input=q0+(uncapped_decoded.astype(float)-inputs['points'].astype(float))/inputs['gain']
            self.last_subpix_input[8]=q0[8];self.last_subpix_input[~support]=q0[~support]
            decoded=uncapped_decoded if subpix or force_uncapped else self.decode(output,temperature,1.,cap)[0].cpu().numpy()
            q=q0+(decoded.astype(float)-inputs['points'].astype(float))/inputs['gain'];q[8]=q0[8];q[~support]=q0[~support];head_calls=1
        corners=0
        if owner in ('BASE','POSEFIX'):self.last_subpix_input=q.copy()
        if subpix:
            refined,diag=self.methods.correct(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),q,support,'SUBPIX')
            q=self.methods.cap_points(q0,refined,image.shape[1],image.shape[0],support)
            corners=diag['algorithm_corner_calls']
        return q,dict(detector_calls=1,corrector_forwards=head_calls,subpix_corner_calls=corners,native_rgb_tiles=8 if owner in ('M3','M4') else 0)

    @torch.no_grad()
    def run(self,image,method,K,xyz):
        self.current_xyz=np.asarray(xyz,float)
        start=self.detector_forwards;captured=self.extractor.predict(image)
        q,counters=self.correct(image,method,captured)
        actual=self.pose.infer(q,np.asarray(K,float),self.current_xyz,False)
        counters.update(detector_forwards=self.detector_forwards-start,F_calls=1)
        return dict(points=q,actual_pose=actual,counters=counters,q0=None if captured['selected_index'] is None else np.asarray(captured['candidates'][captured['selected_index']]['keypoints_xy'],float))

    def close(self):self.hook.remove();self.extractor.close()

    def metadata(self,method):
        owner=canonical(method).removesuffix('_SUBPIX')
        if owner=='SUBPIX':owner='BASE'
        backbone=sum(p.numel() for p in self.extractor.yolo.model.parameters())
        additional=self.parameters.get(owner,0)
        return dict(total_params=backbone+additional,additional_params=additional,
            inactive_RGB_branch_params=2768 if owner in ('M0','M1','M2') else 0,
            inactive_RGB_scorer_column_params=512 if owner in ('M0','M1','M2') else 0,
            inherited_neck_storage='FP16 rounded, sampled as FP32; identical source contract',
            FP32=True,matmul_TF32=False,cudnn_TF32=False,backbone_params=backbone)

def accuracy(args):
    doc=Path(args.public_dir);private=Path(args.private_dir)
    assert (doc/'SYNTHETIC_SELECTION.json').exists(), 'Synthetic selection must precede REAL predictions'
    inputs=np.load(private/'BASELINE_INPUTS.npz');ids=inputs['ids'];predictions=[];audit=[];captures=[];execution=[];begin=time.monotonic()
    # Decode once; existing raw RGB never exported.
    for path,expected in zip(inputs['image_paths'],inputs['image_sha256']):
        assert sha(Path(args.source_root)/str(path))==str(expected), 'Allowed REAL image hash changed'
    images=[cv2.imread(str(Path(args.source_root)/str(p)),cv2.IMREAD_COLOR) for p in inputs['image_paths']]
    for image,p,h in zip(images,inputs['image_paths'],inputs['raw_hw']):assert image is not None and list(image.shape[:2])==list(h)
    existing={}
    import gzip
    with gzip.open(private/'BASELINE_PREDICTIONS_SEALED.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);existing[r['id'],int(r['seed']),r['method']]=np.array(r['qFinal'],float)
    for seed in (1,2,3):
        args.method='ALL';engine=InferenceEngine(args,seed)
        arms=['BASE','OLD_P','N3','POSEFIX']+([a for a in M.ARMS if a in engine.heads] if seed in (1,2) else [])
        corrector_counts={method:0 for method in arms}
        tile_counts={method:0 for method in arms}
        for i,fid in enumerate(ids):
            engine.current_xyz=np.asarray(inputs['xyz'][i],float)
            captured=engine.extractor.predict(images[i])
            selected=captured['selected_index']
            captures.append(dict(id=str(fid),seed=seed,selected_index=selected,candidates=[
                dict(score=float(c['score']),box_xyxy=np.asarray(c['box_xyxy'],float).tolist())
                for c in captured['candidates']],selection_rule='argmax detector confidence; no reference inputs'))
            for method in arms:
                q,counters=engine.correct(images[i],method,captured)
                corrector_counts[method]+=counters['corrector_forwards']
                tile_counts[method]+=counters.get('native_rgb_tiles',0)
                uncapped=engine.last_subpix_input
                name='N3_DIM_SYM' if method=='N3' else method
                predictions.append(dict(id=str(fid),seed=seed,method=name,qFinal=np.full((9,2),np.nan) if q is None else q,subpix_input=np.full((9,2),np.nan) if uncapped is None else uncapped,detected=q is not None,reference_inputs=False))
                if method not in M.ARMS:
                    old=existing[str(fid),seed,name];difference=np.abs(np.full((9,2),np.nan) if q is None else q-old)
                    audit.append(dict(id=str(fid),seed=seed,method=name,max_abs_px=float(np.nanmax(difference)),changed_components=int(np.count_nonzero(difference>1e-7))))
        execution.append(dict(seed=seed,detector_prediction_calls=len(ids),
            actual_detector_network_forwards=engine.detector_forwards,
            detector_warmup_forwards_included=True,corrector_network_forwards=corrector_counts,
            native_RGB_tiles=tile_counts,RGB_CNN_forward_definition='One batched tile CNN per RGB arm/image; eight native corner tiles',
            new_training_updates=0,pose_F_calls=0))
        engine.close();del engine;torch.cuda.empty_cache();print('COMMON_FP32_REAL_PREDICTIONS',seed,len(predictions),flush=True)
    torch.cuda.synchronize()
    p=private/'RAW_NEW_PREDICTIONS.npz';assert not p.exists()
    np.savez_compressed(p,ids=np.array([r['id'] for r in predictions]),seeds=np.array([r['seed'] for r in predictions]),methods=np.array([r['method'] for r in predictions]),points=np.array([r['qFinal'] for r in predictions]),subpix_input=np.array([r['subpix_input'] for r in predictions]),detected=np.array([r['detected'] for r in predictions]))
    capture_path=private/'CURRENT_DETECTOR_CAPTURES.json';write(capture_path,captures)
    write(doc/'PRECISION_PREDICTION_SEAL.json',dict(status='SEALED_COMMON_FP32_BEFORE_NEW_REAL_REFERENCE_SCORING',frames=110,rows=len(predictions),seeds=[1,2,3],new_method_seeds=[1,2],FP32=True,AMP=False,matmul_TF32=False,cudnn_TF32=False,inherited_neck_storage='FP16 rounded, FP32 sampled; common source contract',first_RAW_PoseFix_pass=True,posefix_crop_hw=[384,288],all_source_backbones_frozen=True,protected_or_Severe_reads=0,source_cache_replaced_for_all_accuracy_routes=True,old_cached_prediction_delta=audit,old_cached_rows_preserved=True,predictions_sha256=sha(p),detector_captures_sha256=sha(capture_path),allowed_RGB_hashes_verified=110,execution_counts=execution,inference_source_sha256=sha(Path(__file__)),actual_device_seconds=time.monotonic()-begin))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--private-dir',required=True);p.add_argument('--public-dir',required=True);a=p.parse_args();accuracy(a)
