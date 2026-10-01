"""Native DINO input operators and guarded post-certification runtime extraction.

Pure import: no files, hooks, CUDA, model loads, downloads or label reads.
All runtime descriptors use fixed TRAIN normalization and preserve pose validity.
"""
from pathlib import Path
import numpy as np
from . import common as C
from . import direction_features as D
from scripts.research.pallet_pose_dino_native_inputs_20261001_v1 import visual_features as V

MODELS = D.MODELS
HYP = D.HYP
DIM = 385
RULE = V.RULE
SUPPORT_RULE = V.SUPPORT_RULE
NORMALIZATION = 'R0_TRAIN_valid_float32_mean_std_floor_1e-6_then_float64_candidate_minus_R0_anchor'
array_sha = D.array_sha


def validate_normalization(mean, std, expected_sha=None):
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    assert mean.shape == std.shape == (DIM,)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert (std >= np.float32(1e-6)).all()
    digest = array_sha(np.stack([mean, std]))
    if expected_sha is not None:
        assert digest == expected_sha, 'VISUAL_NORMALIZATION_DRIFT'
    return mean, std


def appearance_difference(raw385, valid, anchor_index, mean385, std385):
    raw, valid, anchor = map(np.asarray, (raw385, valid, anchor_index))
    assert raw.dtype == np.float32 and valid.dtype == bool and valid.ndim == 2
    assert raw.shape == (*valid.shape, DIM) and valid.shape[1] in (2, 4)
    assert anchor.shape == (len(valid),) and np.issubdtype(anchor.dtype, np.integer)
    assert np.isin(anchor, [-1,0,1]).all()
    present = valid.any(1)
    assert np.array_equal(anchor >= 0, present), 'VISUAL_ANCHOR_SUPPORT_CONTRACT'
    rows = np.flatnonzero(present)
    assert valid[rows, anchor[rows]].all()
    assert np.isfinite(raw[valid]).all()
    mean, std = validate_normalization(mean385, std385)
    normalized = np.zeros(raw.shape, np.float64)
    normalized[valid] = ((raw[valid]-mean)/std).astype(np.float64)
    result = np.zeros_like(normalized)
    i,j = np.nonzero(valid)
    result[i,j] = normalized[i,j]-normalized[i,anchor[i]]
    assert np.isfinite(result).all() and not result[~valid].any()
    assert not result[rows,anchor[rows]].any()
    return result


def normalization_receipt(protocol):
    bindings = {k:protocol['inputs'][k] for k in ('visual_receipt','visual_protocol','visual_verification')}
    expected = dict(visual_receipt=C.NATIVE_DOC/'TRAIN_APPEARANCE_INPUTS.json',
                    visual_protocol=C.NATIVE_DOC/'INPUT_PROTOCOL.json',
                    visual_verification=C.NATIVE_DOC/'INPUT_VERIFICATION.json')
    for key,path in expected.items():
        assert bindings[key] == C.bind(path), ('NATIVE_TRAIN_BINDING',key)
    receipt = C.read(expected['visual_receipt']); design = C.read(expected['visual_protocol'])
    checked = C.read(expected['visual_verification'])
    assert receipt['complete'] and receipt['input_construction_pass'] and receipt['source_TRAIN_only']
    assert checked['complete'] and checked['PASS'] and checked['source_TRAIN_only']
    assert checked['verification_PASS'] and checked['normalization_exact']
    assert checked['public_input_receipt'] == bindings['visual_receipt']
    assert receipt['protocol'] == checked['protocol'] == bindings['visual_protocol']
    assert receipt['descriptors'] == checked['descriptors'] == protocol['inputs']['visual_features']
    # Verification may bind the byte-identical RAW producer receipt.
    assert checked['input_receipt']['sha256'] == bindings['visual_receipt']['sha256']
    assert receipt['frames'] == design['frames'] == 2598 and design['output_dim'] == DIM
    assert design['descriptor_rule'] == receipt['descriptor_rule'] == RULE
    assert design['support_rule'] == receipt['support_rule'] == SUPPORT_RULE
    assert design['train_only'] and design['no_fit'] and design['no_label_values']
    assert receipt['original_candidate_validity_preserved'] and not receipt['source_label_values_read']
    norm = receipt['normalization']
    assert norm['source'] == 'R0_original_valid_TRAIN_candidates' and norm['count'] == 5194
    assert norm['dtype'] == 'float32' and norm['std_floor'] == 1e-6
    assert norm['valid_zero_support_candidates_included']
    assert checked['normalization']['array_sha'] == norm['array_sha']
    mean,std = validate_normalization(norm['mean385'],norm['std385'],norm['array_sha'])
    fields = dict(visual_receipt_binding=bindings['visual_receipt'],
        visual_protocol_binding=bindings['visual_protocol'], visual_verification_binding=bindings['visual_verification'],
        visual_normalization_sha=norm['array_sha'])
    return fields,mean,std


def verify_checkpoint(checkpoint, protocol):
    fields,mean,std = normalization_receipt(protocol)
    for key,value in fields.items(): assert checkpoint[key] == value
    a,b = validate_normalization(checkpoint['visual_mean'],checkpoint['visual_std'],fields['visual_normalization_sha'])
    np.testing.assert_array_equal(a,mean);np.testing.assert_array_equal(b,std)
    return fields


def operator_paths():
    # The completed producer's image preparation and frozen-backbone loader are
    # reused verbatim. Import does not install its optional input-only hook.
    from scripts.research.pallet_pose_dino_input_audit_20261001_v1 import freeze_train as H
    from scripts.research.pallet_sensors_submission_v1 import posefix_contract_math as M
    return [Path(__file__),Path(V.__file__),Path(V.PREVIOUS.__file__),Path(H.__file__),Path(M.__file__)]


def paths(scope):
    assert scope in ('SOURCE_VAL','REAL')
    return C.RAW/f'{scope}_APPEARANCE.npz',C.DOC/f'{scope}_APPEARANCE_INPUTS.json',C.RAW/f'{scope}_TOKENS.npy'


def image_padding(scope, row):
    if scope == 'SOURCE_VAL':
        assert row['split'] == 'VAL' and row['pad'] == 100
        return 100
    assert scope == 'REAL' and 'pad' not in row
    assert all(key in row for key in ('id','image','hw','K','xyz','recording','severity'))
    # Native EVAL_METADATA binds exactly the original RGB and its K/hw. No
    # prepare-image padding or K/q translation is applied to real images.
    return 0


def verify_independent_inputs(scope, routing_lock):
    """Require separate input-only verification before opening either GT gate."""
    assert scope in ('SOURCE_VAL','REAL')
    lock=C.read(routing_lock)
    receipt_path=C.DOC/f'{scope}_APPEARANCE_VERIFICATION.json'
    checked=C.read(receipt_path)
    assert checked['complete'] and checked['PASS'] and checked['scope']==scope
    assert checked['frames']==lock['frames'] and checked['protocol']==lock['protocol']
    assert checked['routing_lock']==C.bind(routing_lock)
    assert checked['input_receipt']==lock['visual_inputs']
    assert checked['descriptors']==lock['visual_features']
    producer=C.read(C.ROOT/lock['visual_inputs']['path'])
    assert checked['tokens']==producer['tokens']
    assert checked['source_label_values_read'] is False and checked['real_reference_values_read'] is False
    assert checked['normalization_recomputed'] is False
    C.verify(checked['code'])
    return C.bind(receipt_path)


def load_frozen(scope, rows, indices, anchor, masks, protocol_binding, training_protocol, inputs):
    from scripts.research.pallet_pose_dino_input_audit_20261001_v1 import freeze_train as H
    path,receipt_path,token_path = paths(scope)
    receipt=C.read(receipt_path)
    assert receipt['complete'] and receipt['PASS'] and receipt['scope']==scope
    assert receipt['frames']==len(rows) and receipt['protocol']==protocol_binding and receipt['inputs']==inputs
    assert receipt['descriptors']==C.bind(path) and receipt['tokens']==H.bind(token_path)
    assert receipt['training_verification']==inputs['training_verification']==C.bind(C.DOC/'TRAIN_CONVERGENCE.json')
    fixed,_,_=normalization_receipt(training_protocol)
    for key,value in fixed.items(): assert receipt[key]==value
    assert receipt['operators']==[C.bind(p) for p in operator_paths()]
    assert receipt['rule']==RULE and receipt['support_rule']==SUPPORT_RULE
    assert receipt['source_label_values_read'] is False and receipt['real_reference_values_read'] is False
    assert receipt['normalization_recomputed'] is False and receipt['additional_padding']==0
    assert receipt['original_candidate_validity_preserved'] and receipt['new_PnP_solves']==receipt['fits_executed']==0
    with np.load(path,allow_pickle=False) as z:
        expected={'ids','source_index','anchor_index','crop_matrices','padding_px','native_rectangles_xyxy'}|{
            m+s for m in MODELS for s in ('_appearance385','_valid','_support8')}
        assert set(z.files)==expected
        np.testing.assert_array_equal(z['ids'],[r['id'] for r in rows])
        np.testing.assert_array_equal(z['source_index'],np.asarray(indices,np.int64))
        np.testing.assert_array_equal(z['anchor_index'],anchor)
        np.testing.assert_array_equal(z['padding_px'],[image_padding(scope,r) for r in rows])
        np.testing.assert_array_equal(z['native_rectangles_xyxy'],np.stack([
            V.native_rectangle(r['hw'],image_padding(scope,r)) for r in rows]))
        assert z['crop_matrices'].shape==(len(rows),3,3)
        values={}
        for m in MODELS:
            x=z[m+'_appearance385'];mask=z[m+'_valid'];support=z[m+'_support8']
            assert x.dtype==np.float32 and x.shape==(len(rows),2,DIM)
            assert mask.dtype==support.dtype==bool and support.shape==(len(rows),2,8)
            np.testing.assert_array_equal(mask,masks[m])
            assert np.isfinite(x).all() and not x[~mask].any() and not support[~mask].any()
            np.testing.assert_array_equal(x[:,:,-1],support.sum(2).astype(np.float32)/8)
            summary=receipt['models'][m]
            assert summary['descriptor_sha']==array_sha(x) and summary['valid_sha']==array_sha(mask)
            assert summary['support_sha']==array_sha(support)
            values[m]=x.copy()
    assert receipt['image_forwards']==int(np.logical_or.reduce([v.any(1) for v in masks.values()]).sum())
    assert len(receipt['frame_receipts'])==len(rows)
    for i,(row,frame) in enumerate(zip(rows,receipt['frame_receipts'])):
        assert frame['id']==row['id'] and frame['image']==row['image'] and frame['source_index']==int(indices[i])
        assert frame['padding_px']==image_padding(scope,row)
        assert frame['image_forward']==bool(anchor[i]>=0)
    return values,dict(visual_inputs=C.bind(receipt_path),visual_features=C.bind(path),**fixed)


def freeze_inputs(scope, rows, indices, anchor, masks, poses, prediction_for,
                  protocol_binding, training_protocol, inputs):
    """One shared frozen DINO forward per available RGB, before any GT scoring."""
    from scripts.research.pallet_pose_dino_input_audit_20261001_v1 import freeze_train as H
    import cv2
    import torch
    path,receipt_path,token_path=paths(scope)
    failed=C.DOC/f'{scope}_APPEARANCE_INPUTS_FAILED.json'
    assert not failed.exists(), 'Failed input extraction requires explicit incident review; no automatic retry.'
    fixed,_,_=normalization_receipt(training_protocol)
    train_audit=D.verify_training_audit(C.bind(C.DOC/'TRAIN_PROTOCOL.json'))
    assert inputs['training_verification']==train_audit
    complete=C.read(C.DOC/'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['all_certified'] and complete['fit_count']==4
    assert complete['models']==list(C.MODEL_NAMES) and complete['protocol']==C.bind(C.DOC/'TRAIN_PROTOCOL.json')
    assert inputs['training_complete']==C.bind(C.DOC/'TRAINING_COMPLETE.json')
    if receipt_path.exists():
        return load_frozen(scope,rows,indices,anchor,masks,protocol_binding,training_protocol,inputs)
    assert not path.exists() and not token_path.exists(), 'Partial extraction must not be overwritten.'
    native=C.read(C.ROOT/fixed['visual_protocol_binding']['path'])
    original=C.read(C.ROOT/native['inputs']['previous_protocol']['path'])
    H.verify(native['inputs']['previous_protocol'])
    assert original['backbone']==native['frozen_parent']['backbone']
    assert original['crop']==native['frozen_parent']['crop'] and original['tokens']==native['frozen_parent']['tokens']
    assert original['tokens']['shape']==[384,56,42] and original['tokens']['inference_dtype']=='float32'
    assert original['tokens']['matmul_allow_tf32'] is False and original['tokens']['cudnn_allow_tf32'] is False
    backbone_binding=original['backbone'];H.verify(backbone_binding)
    assert scope=='SOURCE_VAL' and len(rows)==1024 or scope=='REAL' and len(rows)==173
    present=np.logical_or.reduce([v.any(1) for v in masks.values()])
    assert np.array_equal(present,anchor>=0)
    assert all(mask.dtype==bool and mask.shape==(len(rows),2) for mask in masks.values())
    pads=np.array([image_padding(scope,r) for r in rows],np.int64)
    rectangles=np.stack([V.native_rectangle(r['hw'],int(pad)) for r,pad in zip(rows,pads)])
    # All allowed image bytes come from already bound metadata, never reference
    # annotation paths. No system setting, clock or download is modified.
    try:
        C.RAW.mkdir(parents=True,exist_ok=True)
        tokens=np.lib.format.open_memmap(token_path,mode='w+',dtype=np.float16,shape=(len(rows),384,56,42))
        arrays=dict(ids=np.asarray([r['id'] for r in rows]),source_index=np.asarray(indices,np.int64),
            anchor_index=np.asarray(anchor,np.int64),crop_matrices=np.zeros((len(rows),3,3),np.float64),
            padding_px=pads,native_rectangles_xyxy=rectangles)
        values={m:np.zeros((len(rows),2,DIM),np.float32) for m in MODELS}
        support={m:np.zeros((len(rows),2,8),bool) for m in MODELS}
        torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=H.load_backbone(backbone_binding)
        frames=[];forwards=0
        for j,row in enumerate(rows):
            prediction,pb=prediction_for('R0',j,row)
            frame=dict(id=row['id'],source_index=int(indices[j]),image=row['image'],prediction=pb,
                image_forward=False,padding_px=int(pads[j]),native_rectangle_xyxy=rectangles[j].tolist())
            if not present[j]:
                tokens[j]=0
            else:
                H.verify(row['image'])
                image=cv2.imdecode(np.frombuffer((C.ROOT/row['image']['path']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
                assert image is not None and list(image.shape[:2])==row['hw']
                selected=prediction['selected_index'];assert selected is not None
                candidate=prediction['candidates'][selected]
                matrix=H.wide_matrix(candidate['box_xyxy']);arrays['crop_matrices'][j]=matrix
                rgb=H.crop_rgb(image,matrix);feature=H.extract(model,rgb)
                tokens[j]=feature;forwards+=1
                frame.update(image_forward=True,matrix=matrix.tolist(),crop_rgb_sha=array_sha(rgb),token_sha=array_sha(feature))
                for m in MODELS:
                    hs={h['name']:h['pose'] for h in poses[m][row['id']]['hypotheses']}
                    for k,name in enumerate(HYP):
                        if not masks[m][j,k]: continue
                        pose=hs[name];assert pose['available']
                        result=V.describe(pose,row['K'],matrix,row['hw'],feature,int(pads[j]))
                        values[m][j,k]=result['descriptor'];support[m][j,k]=result['support8']
            frames.append(frame)
            if (j+1)%100==0 or j==len(rows)-1:
                tokens.flush();print('FROZEN_NATIVE_DINO_INPUT',scope,j+1,'/',len(rows),'forwards',forwards,flush=True)
        tokens.flush();del tokens;del model
        summaries={}
        for m in MODELS:
            x=values[m];mask=masks[m];s=support[m]
            assert np.isfinite(x).all() and not x[~mask].any() and not s[~mask].any()
            np.testing.assert_array_equal(x[:,:,-1],s.sum(2).astype(np.float32)/8)
            arrays.update({m+'_appearance385':x,m+'_valid':mask.copy(),m+'_support8':s})
            count=s.sum(2)[mask]
            summaries[m]=dict(valid_candidates=int(mask.sum()),descriptor_sha=array_sha(x),valid_sha=array_sha(mask),
                support_sha=array_sha(s),supported_corner_histogram={str(k):int((count==k).sum()) for k in range(9)})
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        C.save(receipt_path,dict(complete=True,PASS=True,scope=scope,frames=len(rows),protocol=protocol_binding,
            inputs=inputs,training_verification=train_audit,descriptors=C.bind(path),tokens=H.bind(token_path),
            **fixed,operators=[C.bind(p) for p in operator_paths()],rule=RULE,support_rule=SUPPORT_RULE,
            frame_receipts=frames,models=summaries,backbone=backbone_binding,
            precision=dict(matmul_allow_tf32=False,cudnn_allow_tf32=False,cudnn_benchmark=False,inference_dtype='float32',token_dtype='float16'),
            original_candidate_validity_preserved=True,allinvalid_rows=int((~present).sum()),
            normalization_recomputed=False,additional_padding=0,image_forwards=forwards,new_PnP_solves=0,fits_executed=0,
            source_label_values_read=False,real_reference_values_read=False,
            native_contract='Source uses bound pad100; real uses native image hw/K and explicit pad0, no coordinate shifts.',
            context_padding_caveat=native['caveat']))
    except Exception as error:
        if not failed.exists():
            C.save(failed,dict(complete=False,PASS=False,scope=scope,error_type=type(error).__name__,error=str(error),
                no_automatic_retry=True,source_label_values_read=False,real_reference_values_read=False))
        raise
    return load_frozen(scope,rows,indices,anchor,masks,protocol_binding,training_protocol,inputs)


def selfcheck():
    rng=np.random.default_rng(20261001656)
    x=rng.normal(size=(3,4,DIM)).astype(np.float32);x[2]=np.nan
    valid=np.array([[1,1,1,1],[0,1,1,0],[0,0,0,0]],bool);index=np.array([0,1,-1])
    mean=rng.normal(size=DIM).astype(np.float32);std=np.abs(rng.normal(size=DIM)).astype(np.float32)+np.float32(.1)
    actual=appearance_difference(x,valid,index,mean,std);expected=np.zeros_like(actual)
    for i,j in zip(*np.nonzero(valid)):
        for k in range(DIM):
            expected[i,j,k]=float(np.float32((x[i,j,k]-mean[k])/std[k]))-float(np.float32((x[i,index[i],k]-mean[k])/std[k]))
    np.testing.assert_array_equal(actual,expected)
    assert not actual[~valid].any()
    native=V.selfcheck();assert native['PASS']
    assert image_padding('SOURCE_VAL',dict(split='VAL',pad=100))==100
    assert image_padding('REAL',dict.fromkeys(('id','image','hw','K','xyz','recording','severity')))==0
    return dict(PASS=True,invented_only=True,normalization_and_anchor_difference_exact=True,native_support=native,
                input_files_read=0,new_forwards=0,new_fits=0)


if __name__=='__main__':
    import json
    print(json.dumps(selfcheck(),ensure_ascii=False,indent=2))
