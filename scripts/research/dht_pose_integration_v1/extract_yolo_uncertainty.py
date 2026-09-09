"""Read preserved YOLO26 RLE sigma heads from the unchanged eval prediction.

Sigma heads are copied before standard Ultralytics inference fusion removes them.
Read-only hooks capture the actual one2one pose features and final head output.
Training mode, model parameters, library code and the frozen baseline are never
changed. Sigma is an uncalibrated RLE residual scale, not a Gaussian error bound.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics.nn.modules.head import Pose26
from ultralytics.utils import ops, nms
from ultralytics.utils.loss import PoseLoss26, RLELoss

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yolo_baseline as Y


def source_hashes():
    result = Y.source_hashes()
    for obj in (Pose26, PoseLoss26, RLELoss, nms.non_max_suppression):
        path = Path(inspect.getfile(obj)).resolve()
        result[str(path)] = Y.file_sha(path)
    result[str(Path(__file__).resolve())] = Y.file_sha(Path(__file__))
    return result


def state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()+b'\0'+str(value.dtype).encode()+b'\0')
        digest.update(np.asarray(value.shape,dtype='<i8').tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def compare(actual, expected):
    a,b = np.asarray(actual,float),np.asarray(expected,float)
    if a.shape!=b.shape or not np.array_equal(np.isfinite(a),np.isfinite(b)):
        return dict(exact=False,max_abs_delta=None)
    mask=np.isfinite(a)
    delta=float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0.
    exact=bool(np.array_equal(a[mask].view(np.uint64),b[mask].view(np.uint64)))
    return dict(exact=exact,max_abs_delta=delta)


class UncertaintyExtractor:
    def __init__(self, config, device='cuda'):
        self.baseline=Y.Baseline(config,device)
        self.head=self.baseline.model.model.model[-1]
        if not isinstance(self.head,Pose26) or list(self.head.kpt_shape)!=[9,3] or not self.head.end2end or self.head.nc!=1:
            raise ValueError('Expected the exact single-class end2end YOLO26 nine-keypoint head')
        if self.head.one2one_cv4_sigma is None or self.head.flow_model is None:
            raise ValueError('Checkpoint is already fused or lacks trained sigma/flow modules')
        self.sigma_heads=copy.deepcopy(self.head.one2one_cv4_sigma).float().eval()
        self.sigma_source_sha256=state_hash(self.head.one2one_cv4_sigma)
        if state_hash(self.sigma_heads)!=self.sigma_source_sha256:
            raise ValueError('Preserved sigma heads differ from original checkpoint')
        self.sigma_heads=self.sigma_heads.requires_grad_(False).to(device)
        self.capture={}
        self.handles=[]
        self.handles.append(self.baseline.model.model.model[0].register_forward_pre_hook(self._input_hook))
        for level,block in enumerate(self.head.one2one_cv4):
            self.handles.append(block.register_forward_hook(self._feature_hook(level)))
        self.handles.append(self.head.register_forward_hook(self._head_hook))
        self.maximum_selected_sigma_manual_delta=0.

    def _input_hook(self,module,inputs):
        self.capture['input_shape_chw']=list(inputs[0].shape[1:])

    def _feature_hook(self,level):
        def hook(module,inputs,output):
            self.capture.setdefault('features',{})[level]=output.detach()
        return hook

    def _head_hook(self,module,inputs,output):
        if not isinstance(output,tuple) or len(output)!=2:
            raise ValueError('Expected standard eval (decoded,raw) head output')
        self.capture['decoded'],self.capture['raw']=output

    @torch.no_grad()
    def predict(self,image):
        self.capture={}
        Y.synchronize();started=time.perf_counter()
        baseline=self.baseline.predict(image)
        if any(m.training for m in self.baseline.model.model.modules()) or any(m.training for m in self.sigma_heads.modules()):
            raise ValueError('A module entered training mode during extraction')
        if self.head.one2one_cv4_sigma is not None:
            raise ValueError('Expected standard inference fusion to remove model sigma heads; preserved copy is separate')
        raw=self.capture['raw']['one2one']
        if 'kpts_sigma' in raw:
            raise ValueError('Unexpected training-mode sigma output in standard eval path')
        features=self.capture['features']
        if set(features)!={0,1,2}:
            raise ValueError('Missing actual one2one pose feature hooks')
        decoded=self.capture['decoded']
        scores,labels,anchor_indices=self.head.get_topk_index(raw['scores'].sigmoid().permute(0,2,1),self.head.max_det)
        if not torch.equal(scores,decoded[...,4:5]) or not torch.equal(labels,decoded[...,5:6]):
            raise ValueError('Reconstructed head top-k mapping differs from actual prediction')
        eligible=torch.nonzero(decoded[0,:,4]>self.baseline.recipe['box_confidence'],as_tuple=False).flatten()[:self.baseline.recipe['max_det']]
        if len(eligible)!=baseline['n_instances']:
            raise ValueError('Instance filtering does not match unchanged baseline')
        feature_shapes=[list(features[level].shape[2:]) for level in range(3)]
        counts=[int(np.prod(shape)) for shape in feature_shapes]
        boundaries=np.cumsum([0]+counts)
        if sum(counts)!=raw['kpts'].shape[-1]:
            raise ValueError('Sigma/keypoint anchor flattening lengths differ')
        # Validate level boundaries even when the chosen object uses only one level.
        for level,(h,w) in enumerate(feature_shapes):
            for local in (0,h*w-1):
                anchor=int(boundaries[level]+local)
                expected=torch.tensor([local%w+.5,local//w+.5],device=self.head.anchors.device)
                if not torch.equal(self.head.anchors[:,anchor],expected) or float(self.head.strides[0,anchor])!=float(self.head.stride[level]):
                    raise ValueError('Anchor level/row/column/stride mapping failed')
        input_shape=self.capture['input_shape_chw']
        height,width=image.shape[:2];pad=self.baseline.recipe['pad_px']
        canvas_shape=(height+2*pad,width+2*pad)
        gain=min(input_shape[1]/canvas_shape[0],input_shape[2]/canvas_shape[1])
        extra=dict(sigma_logits=None,sigma_stride_units=None,sigma_original_px=None,
                   selected_anchor_index=None,feature_level=None,stride=None,feature_row_col=None,
                   input_shape_chw=input_shape,feature_shapes=feature_shapes,letterbox_gain=gain,
                   anchor_mapping_exact=True,keypoint_mapping_max_abs_delta_px=0.,box_mapping_max_abs_delta_px=0.)
        if baseline['detected']:
            position=int(eligible[baseline['selected_instance']])
            anchor=int(anchor_indices[0,position,0])
            level=int(np.searchsorted(boundaries[1:],anchor,side='right'))
            local=anchor-int(boundaries[level]);h,w=feature_shapes[level];row,col=divmod(local,w)
            stride=float(self.head.strides[0,anchor])
            # Verify decoded selected corner/box/confidence identity in original pixels.
            keypoints=self.head.kpts_decode(raw['kpts'])[0,:,anchor].reshape(9,3).clone()
            keypoints=ops.scale_coords(input_shape[1:],keypoints[None],canvas_shape)[0].cpu().numpy()
            keypoints[:,:2]-=pad
            box=ops.scale_boxes(input_shape[1:],decoded[0,position,:4].clone()[None],canvas_shape)[0].cpu().numpy()-pad
            kp_parity=compare(keypoints[:,:2],baseline['kps']);box_parity=compare(box,baseline['box_xyxy'])
            if not kp_parity['exact'] or not box_parity['exact'] or not compare(keypoints[:,2],baseline['kp_conf'])['exact']:
                raise ValueError(f'Selected anchor does not reproduce baseline: keypoints{kp_parity},box{box_parity}')
            if float(scores[0,position,0])!=baseline['box_conf']:
                raise ValueError('Selected anchor box confidence differs')
            # The unchanged baseline retains its original cuDNN settings. Only
            # the separately preserved sigma heads use full FP32 arithmetic;
            # scoped flags restore the preceding backend settings afterward.
            baseline_cudnn_tf32=torch.backends.cudnn.allow_tf32
            with torch.backends.cudnn.flags(allow_tf32=False):
                all_logits=[self.sigma_heads[i](features[i]) for i in range(3)]
            if torch.backends.cudnn.allow_tf32!=baseline_cudnn_tf32:
                raise ValueError('Sigma arithmetic context changed baseline cuDNN settings')
            flat=torch.cat([x.view(1,18,-1) for x in all_logits],2)
            logits=flat[0,:,anchor].reshape(9,2)
            if not torch.equal(logits,all_logits[level][0,:,row,col].reshape(9,2)):
                raise ValueError('Sigma anchor flattening mismatch')
            module=self.sigma_heads[level]
            manual=torch.nn.functional.linear(features[level][0,:,row,col],module.weight[:,:,0,0],module.bias).reshape(9,2)
            manual_delta=float((manual-logits).abs().max())
            if not torch.allclose(manual,logits,atol=2e-5,rtol=2e-5):
                raise ValueError('Selected sigma convolution does not match direct feature/channel calculation')
            self.maximum_selected_sigma_manual_delta=max(self.maximum_selected_sigma_manual_delta,manual_delta)
            sigma=logits.sigmoid()
            sigma_px=sigma*stride/gain
            if not torch.isfinite(logits).all() or not torch.isfinite(sigma_px).all() or not (sigma_px>0).all():
                raise ValueError('Nonfinite/nonpositive extracted sigma')
            extra.update(sigma_logits=logits.cpu().tolist(),sigma_stride_units=sigma.cpu().tolist(),
                         sigma_original_px=sigma_px.cpu().tolist(),selected_anchor_index=anchor,
                         feature_level=level,stride=stride,feature_row_col=[row,col],
                         keypoint_mapping_max_abs_delta_px=kp_parity['max_abs_delta'],
                         box_mapping_max_abs_delta_px=box_parity['max_abs_delta'])
        Y.synchronize()
        return baseline|extra|dict(extraction_wall_ms=(time.perf_counter()-started)*1000)


def baseline_parity(prediction,reference):
    comparisons={key:compare(prediction[key],reference[key]) for key in ('kps','kp_conf','box_xyxy','box_conf')}
    flags={key:prediction[key]==reference[key] for key in ('kp_valid','detected','n_instances','selected_instance')}
    return dict(PASS=all(v['exact'] for v in comparisons.values()) and all(flags.values()),
                comparisons=comparisons,flags=flags)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True,help='Existing integration live directory')
    parser.add_argument('--phase',choices=('smoke','all'),default='all')
    args=parser.parse_args();root=args.run_dir.resolve();out=root/'uncertainty_fusion_v1';out.mkdir(exist_ok=True)
    if not (out/'PURPOSE.md').is_file():
        raise ValueError('New uncertainty experiment PURPOSE.md must exist before extraction')
    config,manifest,baseline=[Y.read_json(root/name) for name in ('CONFIG.json','manifest.json','BASELINE_YOLO.json')]
    records=manifest['records'];references={r['id']:r for r in baseline['records']}
    if len(records)!=692 or [r['id'] for r in records]!=[r['id'] for r in baseline['records']]:
        raise ValueError('Frozen692-frame baseline/manifest identity mismatch')
    identity=dict(config_sha256=Y.file_sha(root/'CONFIG.json'),manifest_sha256=Y.file_sha(root/'manifest.json'),
                  baseline_sha256=Y.file_sha(root/'BASELINE_YOLO.json'),weights_sha256=config['models']['yolo']['weights_sha256'],
                  source_sha256=source_hashes())
    if baseline['config_sha256']!=identity['config_sha256'] or baseline['manifest_sha256']!=identity['manifest_sha256']:
        raise ValueError('Baseline belongs to different experiment inputs')
    for path,digest in baseline['source_sha256'].items():
        if Y.file_sha(path)!=digest:raise ValueError(f'Baseline source changed:{path}')
    protocol=dict(schema='yolo_sigma_extraction_protocol_v2',**identity,
                  inference='Original Baseline.predict unchanged; separate deepcopy of one2one_cv4_sigma before standard fusion; eval-only hooks on actual one2one_cv4 features and Pose26 output.',
                  selection='Actual head.get_topk_index plus originalconfidencefilter, then highestboxconfidence instance. Verify complete point/box/confidence reconstruction; noGTinput.',
                  sigma_units='sigmoid(sigma_logits) normalizes coordinate residuals in anchor-stride units during RLE loss. sigma_original_px=sigma_stride_units*selected_stride/letterbox_gain.',
                  sigma_arithmetic='Preserved sigma heads only: FP32 convolution with scoped cuDNN allow_tf32=False, restoring original backend flags before the next unchanged baseline inference. Independent selected-anchor FP32 linear calculation verifies mapping at atol=rtol=2e-5. V1 TF32 control recorded separately; baseline outputs unchanged.',
                  limits='RLE residual scale with a learned flow is not calibrated Gaussian standard deviation. Original-coordinate scale is the affine inverse before any display/predictor clipping; translations includingreflectpadding do not alter scale.',
                  mutation_policy='No training=True, optimizer, library patch, checkpoint modification, or change to frozen baseline source.')
    protocol_path=out/'EXTRACTION_PROTOCOL_YOLO.json'
    if protocol_path.exists() and Y.read_json(protocol_path)!=protocol:
        raise ValueError('Existing extraction protocol differs; investigate before changing it')
    Y.write_json(protocol_path,protocol)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    extractor=UncertaintyExtractor(config,device='cuda')
    for _ in range(5):extractor.predict(Y.load_image(records[0]))
    smoke=[]
    for record in records[:3]:
        if record['population']!='synth_val':raise ValueError('Smoke must use syntheticvalidation only')
        result=extractor.predict(Y.load_image(record));agreement=baseline_parity(result,references[record['id']])
        smoke.append(dict(id=record['id'],parity=agreement,selected_anchor_index=result['selected_anchor_index'],
                          feature_level=result['feature_level'],stride=result['stride'],letterbox_gain=result['letterbox_gain']))
        if not agreement['PASS']:
            Y.write_json(out/'EXTRACTION_FAILURE_YOLO.json',dict(stage='smoke',records=smoke))
            raise ValueError('Sigma extraction baseline parity failed; report discrepancy before evaluating uncertainty')
    Y.write_json(out/'EXTRACTION_SMOKE_YOLO.json',dict(PASS=True,records=smoke,**identity))
    print('YOLO sigma smoke PASS3/3 exactbaseline plusselectedanchor reconstruction',flush=True)
    if args.phase=='smoke':return
    output,agreements=[],[]
    for index,record in enumerate(records):
        result=extractor.predict(Y.load_image(record));agreement=baseline_parity(result,references[record['id']])
        agreements.append(dict(id=record['id'],**agreement))
        if not agreement['PASS']:
            Y.write_json(out/'EXTRACTION_FAILURE_YOLO.json',dict(stage='full',records=agreements))
            raise ValueError(f"Exact frozen baseline parity failed:{record['id']}; stopbeforeuncertainty evaluation")
        output.append({k:record[k] for k in ('id','population','group','width','height')}|result)
        if (index+1)%100==0 or index+1==len(records):print(f'YOLO sigma {index+1}/692',flush=True)
    if state_hash(extractor.sigma_heads)!=extractor.sigma_source_sha256:
        raise ValueError('Sigma weights changed during extraction')
    if source_hashes()!=identity['source_sha256'] or Y.file_sha(root/'CONFIG.json')!=identity['config_sha256'] or Y.file_sha(root/'manifest.json')!=identity['manifest_sha256'] or Y.file_sha(root/'BASELINE_YOLO.json')!=identity['baseline_sha256']:
        raise ValueError('Frozen sources/inputs changed during extraction')
    payload=dict(schema='yolo_uncertainty_predictions_v1',complete=True,**identity,records=output,n_frames=692,
                 n_detected=sum(r['detected'] for r in output),sigma_head_state_sha256=extractor.sigma_source_sha256,
                 sigma_interpretation=protocol['sigma_units'],limitations=protocol['limits'],
                 runtime=dict(python=sys.executable,torch=torch.__version__,ultralytics=ultralytics.__version__,
                              cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),batch=1,eval_only=True))
    Y.write_json(out/'YOLO_UNCERTAINTY.json',payload)
    history_path=Path(config['models']['yolo']['weights']).parent.parent/'results.csv'
    with history_path.open(newline='') as handle:
        history=[{k.strip():v for k,v in row.items()} for row in csv.DictReader(handle)]
    history_evidence=dict(path=str(history_path),sha256=Y.file_sha(history_path),epochs=len(history),
                          first_three=[{key:float(row[key]) for key in ('epoch','train/rle_loss')} for row in history[:3]],
                          last_three=[{key:float(row[key]) for key in ('epoch','train/rle_loss')} for row in history[-3:]],
                          interpretation='Nonzero early training RLE loss confirms this term was active. Later zero loss is not evidence of an untrained sigma: PoseLoss26 clamps RLE loss at zero. Validation eval does not expose sigma heads.')
    audit=dict(schema='yolo_uncertainty_extraction_audit_v1',complete=True,PASS=True,**identity,
               prediction_sha256=Y.file_sha(out/'YOLO_UNCERTAINTY.json'),n_frames=692,
               exact_baseline_frames=sum(r['PASS'] for r in agreements),baseline_parity=agreements,
               anchor_keypoint_box_confidence_mapping_exact=True,level_anchor_boundary_checks=True,
               max_selected_sigma_manual_convolution_delta=extractor.maximum_selected_sigma_manual_delta,
               manual_sigma_tolerance=dict(atol=2e-5,rtol=2e-5),sigma_heads_unchanged=True,eval_only=True,
               original_baseline_code_unchanged=True,images_sha256_verified=True,
               sigma_arithmetic=protocol['sigma_arithmetic'],training_history=history_evidence,
               scope='Extraction/coordinate/parity PASS only. Sigma calibration and usefulness remain separate downstream evaluations.')
    Y.write_json(out/'EXTRACTION_AUDIT_YOLO.json',audit)
    print(f'Complete YOLO sigma extraction692/692 exactbaseline:{out}',flush=True)


if __name__=='__main__':main()
