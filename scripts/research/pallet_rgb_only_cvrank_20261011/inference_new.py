"""New accuracy routes reuse the sealed FP32 detector output, never old logits."""
import argparse, gzip, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace
import cv2, numpy as np, torch
import rgb_model as R
import precision_model as M
import train_rgb as T
import inference as prior_inference

def load_head(args,seed):
    env,refiner,Data,Base,targets,decode,loss=M.source_modules(args.source_root)
    path=Path(args.private_dir)/f'checkpoints/RGB_ONLY_seed{seed}.pt'
    receipt=T.read(Path(args.public_dir)/f'RGB_ONLY_TRAIN_RECEIPT_seed{seed}.json')
    assert receipt['status']=='DONE' and T.sha(path)==receipt['private_checkpoint']['sha256']
    ck=torch.load(path,map_location='cpu');assert ck['step']==3000 and ck['seed']==seed
    original=torch.load(env.C.BRAW/f'runs/seed{seed}/last.pt',map_location='cpu')
    head=R.RGBOnlyEvidence(Base(**original['config']))
    head.load_state_dict(ck['state_dict']);counts=head.parameter_counts()
    return env,head.cuda().eval().requires_grad_(False),decode,counts

def correct_rgb(image,captured,head,decode,temperature,features,methods,subpix=False):
    if captured['selected_index'] is None:return None,None,dict(corrector_forwards=0,subpix_corner_calls=0)
    candidate=captured['candidates'][captured['selected_index']]
    q0=np.asarray(candidate['keypoints_xy'],float);support=np.isfinite(q0).all(-1)&~(q0==-1).all(-1)
    inputs=features.branch_inputs(captured)
    batch={k:torch.as_tensor(inputs[k],device='cuda')[None] for k in ('points','boxes','point_valid','input_shape')}
    h,w=image.shape[:2]
    fake=SimpleNamespace(indices=np.array([0]),source={'records':[dict(raw_shape_hw=[h,w],prepared_shape_hw=list(captured['canvas_shape']),reflect_pad_px=captured['added_border'])]},
        arrays={'gain':np.array([inputs['gain']]),'input_shape':np.array([inputs['input_shape']])},_rgb_reader=lambda _:image[:,:,::-1])
    tiles=M.native_tiles(fake,np.array([0]),batch,head.base.displacements)
    output=head(batch,True,tiles)
    uncapped=decode(output,temperature,1.,None)[0].cpu().numpy()
    native=q0+(uncapped.astype(float)-inputs['points'].astype(float))/inputs['gain'];native[8]=q0[8];native[~support]=q0[~support]
    if subpix:
        refined,diag=methods.correct(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),native,support,'SUBPIX')
        final=methods.cap_points(q0,refined,w,h,support);calls=diag['algorithm_corner_calls']
    else:
        cap=.01*math.hypot(h,w)*inputs['gain']
        capped=decode(output,temperature,1.,cap)[0].cpu().numpy()
        final=q0+(capped.astype(float)-inputs['points'].astype(float))/inputs['gain'];final[8]=q0[8];final[~support]=q0[~support];calls=0
    return final,native,dict(corrector_forwards=1,subpix_corner_calls=calls,native_rgb_tiles=8,p3_p4_reads=0)

class InferenceEngine(prior_inference.InferenceEngine):
    def __init__(self,args,seed=1):
        route=args.method
        simple=SimpleNamespace(**vars(args));simple.method='BASE'
        super().__init__(simple,seed)
        self.route=route
        if route.startswith('RGB_ONLY'):
            _,head,decode,counts=load_head(args,seed)
            self.head=head;self.decode=decode;self.rgb_counts=counts
            self.temperature=T.read(Path(args.public_dir)/'CALIBRATION.json')['temperatures'][f'RGB_ONLY_seed{seed}']['temperature']

    @torch.no_grad()
    def correct(self,image,method,captured,force_uncapped=False):
        if captured['selected_index'] is None:return None,dict(detector_calls=1,corrector_forwards=0,subpix_corner_calls=0)
        if method.startswith('RGB_ONLY'):
            q,native,counters=correct_rgb(image,captured,self.head,self.decode,self.temperature,self.features,self.methods,method.endswith('_SUBPIX'))
            self.last_subpix_input=native
        else:
            q0=np.asarray(captured['candidates'][captured['selected_index']]['keypoints_xy'],float)
            support=np.isfinite(q0).all(-1)&~(q0==-1).all(-1)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            q,diag=self.methods.correct(gray,q0,support,'CVRANK');self.last_subpix_input=q.copy();corners=0
            if method=='CVRANK_THEN_SUBPIX':q,d=self.methods.correct(gray,q,support,'SUBPIX');corners=d['algorithm_corner_calls']
            if method!='CVRANK_NATIVE':q=self.methods.cap_points(q0,q,image.shape[1],image.shape[0],support)
            counters=dict(corrector_forwards=0,CVRANK_calls=1,subpix_corner_calls=corners)
        counters['detector_calls']=1
        return q,counters

    def metadata(self,method):
        backbone=sum(p.numel() for p in self.extractor.yolo.model.parameters())
        c=self.rgb_counts if method.startswith('RGB_ONLY') else dict(serialized_parameters=0,executed_parameter_tensors_elements=0,input_active_parameter_elements=0)
        return dict(total_params=backbone+c['serialized_parameters'],additional_params=c['serialized_parameters'],
            backbone_params=backbone,active_params=backbone+c['input_active_parameter_elements'],
            executed_parameter_elements=backbone+c['executed_parameter_tensors_elements'],corrector_parameter_counts=c,
            FP32=True,AMP=False,matmul_TF32=False,cudnn_TF32=False,
            inherited_neck_storage='Common frozen detector extractor still captures FP16-rounded p3/p4; new RGB-only head never reads them. Whole detector remains unchanged.')

@torch.no_grad()
def accuracy(args):
    doc,private=Path(args.public_dir),Path(args.private_dir)
    assert T.read(doc/'SYNTHETIC_SELECTION.json')['status']=='SELECTED_BEFORE_REAL_ON_FIXED_C2_RENDERER_VAL'
    x=np.load(private/'COMMON_INPUTS.npz');start=time.monotonic();T.configure()
    env,refiner,Data,Base,targets,decode,loss=M.source_modules(args.source_root)
    features=env.old('features')
    import importlib.util
    sp=importlib.util.spec_from_file_location('rgb_real_sp',env.ROOT/'scripts/research/pallet_training_free_compare_20261007_v1/methods.py')
    methods=importlib.util.module_from_spec(sp);sp.loader.exec_module(methods)
    captures=T.read(Path(args.prior_private_dir)/'CURRENT_DETECTOR_CAPTURES.json')
    capture_lookup={(r['id'],int(r['seed'])):r for r in captures}
    from ultralytics.data.augment import LetterBox
    records=[];panel_shapes=set()
    for seed in (1,2):
        _,head,decode,counts=load_head(args,seed)
        temperature=T.read(doc/'CALIBRATION.json')['temperatures'][f'RGB_ONLY_seed{seed}']['temperature']
        for k,fid in enumerate(x['ids'].tolist()):
            image_path=Path(str(x['image_paths'][k]));image_path=image_path if image_path.is_absolute() else Path(args.source_root)/image_path
            assert T.sha(image_path)==str(x['image_sha256'][k])
            image=cv2.imread(str(image_path));h,w=image.shape[:2];q0=np.array(x['q0'][k],float)
            saved=capture_lookup[fid,seed];selected=saved['selected_index']
            canvas_shape=(h+200,w+200)
            # Actual installed official preprocessing, not a guessed affine.
            transformed=LetterBox((640,640),auto=True,stride=32)(image=np.zeros((*canvas_shape,3),np.uint8))
            input_shape=transformed.shape[:2];panel_shapes.add(tuple(input_shape))
            assert tuple(x['canvas_shape'][k])==canvas_shape and tuple(x['input_shape'][k])==input_shape
            assert features.canvas_affine(canvas_shape,input_shape)[0]==float(x['gain'][k])
            candidates=[] if selected is None else [dict(c, keypoints_xy=q0.copy(),
                box_xyxy=np.asarray(c['box_xyxy'],dtype=np.float64)) for c in saved['candidates']]
            captured=dict(selected_index=selected,candidates=candidates,canvas_shape=canvas_shape,input_shape=input_shape,added_border=100)
            q,native,counters=correct_rgb(image,captured,head,decode,temperature,features,methods,False)
            records.append(dict(id=fid,seed=seed,points=q if q is not None else q0.copy(),subpix_input=native if native is not None else q0.copy(),detected=selected is not None,
                q0=q0.copy(),support=np.array(x['support'][k]),boxes=np.array(x['boxes'][k])))
        del head;torch.cuda.empty_cache();print('RGB_ONLY_REAL_COORDINATES_SEALED',seed,110,flush=True)
    torch.cuda.synchronize();path=private/'RAW_RGB_ONLY_PREDICTIONS.npz';assert not path.exists()
    np.savez_compressed(path,ids=np.array([r['id'] for r in records]),seeds=np.array([r['seed'] for r in records]),points=np.array([r['points'] for r in records]),
        subpix_input=np.array([r['subpix_input'] for r in records]),detected=np.array([r['detected'] for r in records]),
        q0=np.array([r['q0'] for r in records]),support=np.array([r['support'] for r in records]),boxes=np.array([r['boxes'] for r in records]))
    T.write(doc/'RGB_ONLY_PREDICTION_SEAL.json',dict(status='SEALED_BEFORE_NEW_REFERENCE_SCORING',frames=110,rows=220,seeds=[1,2],
        detector_forwards=0,existing_current_FP32_base_reused=True,GT_inputs=False,p3_p4_reads=0,
        input_shapes=sorted(map(list,panel_shapes)),letterbox_source='Installed official Ultralytics LetterBox auto=True stride=32; source branch_inputs and canvas_affine unchanged',
        common_input_sha256=T.sha(private/'COMMON_INPUTS.npz'),predictions_sha256=T.sha(path),source_sha256=T.sha(__file__),
        actual_device_seconds=time.monotonic()-start,protected_or_Severe_reads=0,actual_PnP_calls=0))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source-root','private-dir','public-dir','prior-private-dir','prior-public-dir'):p.add_argument('--'+name,required=True)
    accuracy(p.parse_args())
