"""Read-only TRAIN loss/assignment probes; no optimizer is constructed.

Hooks call the installed true-ignore criterion unchanged. Arrays and examples
are private; public output contains aggregates, bindings and limitations only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
from pathlib import Path
import random
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import common as C
from scripts.self_training_yolo.v3.true_ignore_pose_loss import TrueIgnorePoseLoss26, make_criterion
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import pose_parameter

RAW = C.RAW / 'loss_signal'
C3 = C.OLD.RAW / 'cycles/C3_MANUAL38_CAPABILITY'
C3DOC = C.OLD.DOC / 'cycles/C3_MANUAL38_CAPABILITY'


def normalized_smooth_l1(pred, target, supervise, area, sigmas, beta=1.):
    """Same support/denominator as location; transition sqrt(e)=1, not pixels.

    Existing OKS exponent is ||delta / normalizer||**2. All normalization is
    detached. Returning a graph-connected zero makes fully ignored rows safe.
    """
    scale = ((2 * sigmas.detach()).square() * (area.detach() + 1e-9) * 2).sqrt()
    z = (pred[..., :2] - target[..., :2]) / scale[..., None]
    value = F.smooth_l1_loss(z, torch.zeros_like(z), reduction='none', beta=beta).sum(-1)
    factor = supervise.shape[1] / (supervise.sum(1) + 1e-9)
    return (factor[:, None] * value * supervise).mean()


class SupplementLoss(TrueIgnorePoseLoss26):
    coefficient = 0.
    beta = 1.

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        result = super().calculate_keypoints_loss(masks, target_gt_idx, keypoints,
            batch_idx, stride_tensor, target_bboxes, pred_kpts)
        if not masks.any() or self.coefficient == 0:
            return result
        from ultralytics.utils.ops import xyxy2xywh
        gt = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        gt[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        area = xyxy2xywh((target_bboxes / stride_tensor)[masks])[:, 2:].prod(1, keepdim=True)
        extra = normalized_smooth_l1(pred_kpts[masks], gt[masks], gt[masks][..., 2] == 2,
                                     area, self.keypoint_loss.sigmas, self.beta)
        return result[0] + self.coefficient * extra, result[1], result[2]


def make_supplement_criterion(model, coefficient, beta=1.):
    """Opt-in factory for a separately locked paired fit; not used by probes."""
    from ultralytics.utils.loss import E2ELoss
    assert model.end2end and coefficient >= 0 and beta > 0
    criterion = E2ELoss(model, SupplementLoss)
    for branch in (criterion.one2many, criterion.one2one):
        branch.coefficient, branch.beta = coefficient, beta
    return criterion


class SignalHooks:
    def __init__(self, criterion):
        self.criterion, self.rows, self.restore, self.handles = criterion, {}, [], []

    def __enter__(self):
        for name in ('one2many', 'one2one'):
            branch = getattr(self.criterion, name)
            row = self.rows[name] = dict(branch=branch)
            original = branch.calculate_keypoints_loss
            self.restore.append((branch, original))
            def wrapped(*args, _row=row, _original=original):
                _row['args'] = args
                result = _original(*args)
                _row['terms'] = result
                return result
            branch.calculate_keypoints_loss = wrapped
            def location(module, inputs, output, _row=row):
                _row['location_inputs'], _row['location'] = inputs, output
            def rle(module, inputs, output, _row=row):
                _row['rle_inputs'], _row['rle_before_clamp'] = inputs, output
            self.handles.append(branch.keypoint_loss.register_forward_hook(location))
            self.handles.append(branch.rle_loss.register_forward_hook(rle))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        for branch, original in self.restore:
            branch.calculate_keypoints_loss = original


def attach_signal_hooks(criterion):
    return SignalHooks(criterion)


def distribution(values):
    a = np.asarray(values, float).reshape(-1)
    return dict(n=len(a), min=float(a.min()), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), max=float(a.max())) if len(a) else dict(n=0)


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_model(binding, args, device):
    from ultralytics.cfg import get_cfg
    C.verify(binding)
    model = torch.load(C.ROOT / binding['path'], map_location='cpu', weights_only=False)['model'].float()
    model.args = get_cfg(overrides=args)
    for name, p in model.named_parameters():
        p.requires_grad_(pose_parameter(name))
    model.to(device).train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    return model


def curve():
    """CPU single-point input curve through actual installed modules."""
    from ultralytics.utils.loss import KeypointLoss, PoseLoss26, RLELoss
    from scripts.research.pallet_oracle_mechanism_followup_v1.cycle_manual import read_protocol
    torch.set_num_threads(4)
    p = read_protocol(); binding = C.read(C3 / 'FIT_MANUAL9.json')['checkpoint']
    model = load_model(binding, p['args'], 'cpu')
    flow = copy.deepcopy(model.model[-1].flow_model).double().eval()
    loss = SimpleNamespace(flow_model=flow, rle_loss=RLELoss(use_target_weight=True), target_weights=torch.ones(9, dtype=torch.float64))
    loc = KeypointLoss(torch.full((9,), 1 / 9, dtype=torch.float64))
    targets = C.read(C3 / 'TRAIN_TARGETS_PRIVATE.json')
    if isinstance(targets, dict): targets = targets['records']
    target = next(r for r in targets if r['id'] == 'wood_day_01:002141')
    box = np.asarray(target['box']); native_area = float(np.prod(box[2:] - box[:2]))
    rows = []
    # Native->padded training640 geometry ratio; no random affine in historical C3 real data.
    gain = 640 / 840
    for stride in (8, 16, 32):
        area = torch.tensor([[native_area * gain**2 / stride**2]], dtype=torch.float64)
        for logit in (-4., 0., 4.):
            for error_px in (0., 1., 2., 4., 8., 16., 32., 64., 128., 256., 512.):
                pred = torch.zeros((1, 9, 5), dtype=torch.float64)
                pred[0, 0, 0] = error_px * gain / stride
                pred[..., -2:] = logit; pred.requires_grad_()
                gt = torch.zeros((1, 9, 3), dtype=torch.float64)
                mask = torch.zeros((1, 9), dtype=torch.bool); mask[0, 0] = True
                location = loc(pred, gt, mask, area)
                raw = PoseLoss26.calculate_rle_loss(loss, pred, gt, mask)
                rle = raw.clamp(min=0)
                extra = normalized_smooth_l1(pred, gt, mask, area, loc.sigmas)
                terms = {}
                for name, value in [('location', location), ('rle_before_clamp', raw), ('rle', rle), ('smooth_l1', extra)]:
                    grad = torch.autograd.grad(value, pred, retain_graph=True)[0]
                    terms[name] = dict(value=float(value), xy_gradient=float(grad[0, 0, :2].norm()),
                                       xy_gradient_axes=grad[0, 0, :2].tolist(),
                                       sigma_logit_gradient=float(grad[0, 0, -2:].norm()))
                e = (pred[0, 0, :2].square().sum() / ((2 / 9)**2 * area[0, 0] * 2)).item()
                standardized = error_px * gain / stride / (float(torch.sigmoid(torch.tensor(logit))) + 1e-9)
                rows.append(dict(native_error_px=error_px, stride=stride, sigma_logit=logit,
                    sigma=float(torch.sigmoid(torch.tensor(logit))), e=e, exp_minus_e=float(np.exp(-e)),
                    standardized_error=standardized, standardized_clipped=standardized > 100,
                    aggregate_clamped=float(raw) < 0, terms=terms))
    result = dict(kind='ARTIFICIAL_SINGLE_POINT_CURVE', actual_installed_calls=True, checkpoint=binding,
        training_area_source='C3 failed TRAIN frame R0 predicted box; no eval data', native_box_area=native_area,
        native_to_network_gain=gain, rows=rows, optimizer_updates=0, fits=0,
        caveat='Artificial errors and sigma logits isolate mechanisms; not frequency or causal failure evidence.')
    C.save(RAW / 'SINGLE_POINT_CURVE.json', result)
    print('CURVE_COMPLETE', len(rows), flush=True)


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def dataset_fixture(kind):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from scripts.research.pallet_oracle_mechanism_followup_v1.cycle_manual import read_protocol
    from scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine import wrap_affine
    p = read_protocol() if kind == 'C3' else C.read(C.P.REC / 'pose_only/PROTOCOL.json')
    arm = 'MANUAL9' if kind == 'C3' else 'REF'
    paths = [Path(x) for x in (C.ROOT / p['datasets'][arm]['train_list']['path']).read_text().splitlines()]
    source = sorted({x for x in paths if x.name.startswith('syn__')})
    real = sorted({x for x in paths if not x.name.startswith('syn__')})
    if kind == 'C3':
        bad = next(x for x in real if 'wood_day_01__002141' in x.name)
        real = [bad, next(x for x in real if 'plastic' in x.name)]
    else:
        # Use first two mask-feasible samples from the already fixed A preflight,
        # never select by prediction, target error or any evaluation reference.
        preflight=C.read(C.RAW/'OCCLUSION_PREFLIGHT_PLASTIC_PRIVATE.json')
        chosen=[r['index'] for r in preflight if r['role']=='REAL' and r['arms']['REF']['plan']['applied']][:2]
        assert len(chosen)==2
        ordered=sorted(paths)
        real=[ordered[i] for i in chosen]
    selected = real + [source[len(source) // 4], source[3 * len(source) // 4]]
    folder = RAW / 'fixtures' / kind
    for path in selected:
        for src, dst in [(path, folder / 'images' / path.name),
                         (path.parent.parent / 'labels' / path.with_suffix('.txt').name,
                          folder / 'labels' / path.with_suffix('.txt').name)]:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists(): dst.symlink_to(src.resolve())
            assert dst.resolve() == src.resolve()
    C.save(folder / 'train.txt', '\n'.join(str(folder / 'images' / p.name) for p in selected) + '\n', True)
    dataset = YOLODataset(img_path=str(folder / 'train.txt'), imgsz=640, batch_size=4,
        augment=True, hyp=get_cfg(overrides=p['args']), rect=False, cache=False, stride=32, pad=0.,
        task='pose', data=dict(names={0: 'pallet'}, nc=1, kpt_shape=[9, 3], flip_idx=[1,0,3,2,5,4,7,6,8]), prefix='SIGNAL ')
    if kind == 'C3': dataset.transforms, n = wrap_affine(dataset.transforms); assert n >= 1
    from .occlusion import SharedOcclusion
    from scripts.research.pallet_clean19_structured_easyhard_v1.augmentation import cover
    canonical={}
    for path in selected:
        if not path.name.startswith('syn__'):
            text=(path.parent.parent/'labels'/path.with_suffix('.txt').name).read_text()
            canonical[path.name]=np.asarray([line.split() for line in text.splitlines()],np.float32)[:,5:].reshape(-1,9,3)
    wrapped=SharedOcclusion(copy.deepcopy(dataset.transforms),canonical)
    entries, masked, plans = [], [], []
    for i in range(len(dataset)):
        original_index=next(j for j,p0 in enumerate(sorted(paths)) if p0.name==Path(dataset.im_files[i]).name)
        sample_seed=280901+original_index if kind=='MAIN' else 290901+i
        seed(sample_seed); entry=dataset[i]; entries.append(entry)
        seed(sample_seed); other=wrapped(copy.deepcopy(dataset.get_image_and_label(i)))
        plans.append(dict(image=Path(dataset.im_files[i]).name,**other.pop('occlusion_info')))
        assert torch.equal(entry['keypoints'],other['keypoints']) and torch.equal(entry['bboxes'],other['bboxes'])
        masked.append(other)
    batch = dataset.collate_fn(entries)
    masked_batch=dataset.collate_fn(masked)
    covered=torch.zeros_like(batch['keypoints'][...,2],dtype=torch.bool)
    for row,bi in enumerate(batch['batch_idx'].long().tolist()):
        plan=plans[bi]
        if plan['applied']:
            covered[row]=torch.from_numpy(cover(batch['keypoints'][row,:,:2].numpy()*640,plan['rectangle'])) & (batch['keypoints'][row,:,2]==2)
    assert len(batch['img']) == 4 and len(batch['keypoints']) == 4
    if kind=='MAIN':assert covered.any(),'Actual A-wrapper masked examples required'
    return p, batch, masked_batch, covered, plans


def grad_norm(grads):
    return float(sum((g.detach().double().square().sum() for g in grads if g is not None), torch.tensor(0., device=next(g.device for g in grads if g is not None))).sqrt())


def finite_difference_actual(row):
    """CPU float64 coordinate perturbations, actual loss method, fixed TAL assignment.

    Only selected anchors are reindexed as separate one-anchor images. This
    preserves the criterion's location/RLE/visibility aggregate denominators.
    No head weights, RGB or assignment are optimized/recomputed here.
    """
    from ultralytics.utils.loss import KeypointLoss, RLELoss
    pred,gt,mask,area=row['location_inputs']; branch=row['branch']
    fixed=object.__new__(TrueIgnorePoseLoss26)
    fixed.flow_model=copy.deepcopy(branch.flow_model).cpu().double().eval()
    fixed.rle_loss=RLELoss(use_target_weight=True)
    fixed.target_weights=branch.target_weights.detach().cpu().double()
    fixed.keypoint_loss=KeypointLoss(branch.keypoint_loss.sigmas.detach().cpu().double())
    fg,ti,kp,bi,st,box,pr=row['args']; n=len(pred)
    boxes=(box/st)[fg].detach().cpu().double().unsqueeze(1)
    # The installed target selector intentionally allocates float32, retained here.
    targets=gt.detach().cpu().float()
    p=pred.detach().cpu().double().requires_grad_()
    args=(torch.ones(n,1,dtype=torch.bool),torch.zeros(n,1,dtype=torch.long),targets,
          torch.arange(n)[:,None],torch.ones(1,1,dtype=torch.float64),boxes)
    def values(x):
        a,b,c=fixed.calculate_keypoints_loss(*args,x.unsqueeze(1))
        d=normalized_smooth_l1(x,targets,mask.detach().cpu(),area.detach().cpu().double(),fixed.keypoint_loss.sigmas)
        return dict(location=a,visibility=b,rle=c,smooth_l1_unit=d)
    baseline=values(p); candidates=mask.detach().cpu().nonzero()
    error=(p[...,:2]-targets[...,:2]).norm(dim=-1)
    ranks=error[mask.detach().cpu()].argsort()
    picks=[candidates[ranks[0]].tolist(),candidates[ranks[-1]].tolist()]
    out=[]
    for term,value in baseline.items():
        grad=torch.autograd.grad(value,p,retain_graph=True)[0]
        checks=[]
        for i,j in picks:
            axis=int((p[i,j,:2]-targets[i,j,:2]).abs().argmax())
            for channel in (axis,3):
                eps=1e-5;delta=torch.zeros_like(p);delta[i,j,channel]=eps
                fd=float((values(p+delta)[term]-values(p-delta)[term])/(2*eps))
                ag=float(grad[i,j,channel]);error_abs=abs(fd-ag)
                checks.append(dict(channel='xy' if channel<2 else 'sigma_logit',automatic=ag,
                    finite_difference=fd,absolute_error=error_abs,
                    passed=error_abs<=1e-6+1e-4*max(abs(fd),abs(ag))))
        direction=grad.detach()/max(1.,float(grad.detach().norm()))
        after=float(values(p-1e-3*direction)[term]); before=float(value)
        out.append(dict(term=term,checks=checks,coordinate_leaf_descent_before=before,
            coordinate_leaf_descent_after=after,descent_nonincreasing=after<=before+1e-9))
    assert all(r['descent_nonincreasing'] and all(c['passed'] for c in r['checks']) for r in out)
    return out


def summarize_hook(name, row, weight, batch, covered, params):
    pred, gt, mask, area = row['location_inputs']
    branch = row['branch']; fg, target_idx, _, batch_idx, strides, boxes, all_pred = row['args']
    positions = fg.nonzero(); stride = strides[positions[:,1], 0]
    sigma = pred[..., -2:].sigmoid(); standardized = (pred[..., :2] - gt[..., :2]) / (sigma + 1e-9)
    e = (pred[..., :2]-gt[..., :2]).square().sum(-1) / ((2*branch.keypoint_loss.sigmas).square() * (area + 1e-9) * 2)
    extra = normalized_smooth_l1(pred, gt, mask, area, branch.keypoint_loss.sigmas)
    terms = dict(location=row['terms'][0]*branch.hyp.pose*weight,
                 visibility=row['terms'][1]*branch.hyp.kobj*weight,
                 rle=row['terms'][2]*branch.hyp.rle*weight,
                 smooth_l1_unit=extra*branch.hyp.pose*weight)
    terms['combined'] = terms['location'] + terms['visibility'] + terms['rle']
    gradients, info = {}, {}
    ignored = gt[...,2] == 1
    for term, value in terms.items():
        g = torch.autograd.grad(value, [pred]+params, retain_graph=True, allow_unused=True)
        coord = g[0]; gradients[term] = coord.detach()
        ignored_max = float(coord[ignored].abs().max()) if ignored.any() else 0.
        assert ignored_max == 0., (name, term, ignored_max)
        radial = (coord[...,:2]*(pred[...,:2]-gt[...,:2])).sum(-1)
        info[term] = dict(value=float(value.detach()), head_gradient_norm=grad_norm(g[1:]),
            supervised_xy_gradient=distribution(coord[...,:2].norm(dim=-1)[mask].detach().cpu()),
            sigma_logit_gradient=distribution(coord[...,-2:].norm(dim=-1)[mask].detach().cpu()),
            target_descent_positive=int((radial[mask]>0).sum()), target_descent_negative=int((radial[mask]<0).sum()),
            target_descent_zero=int((radial[mask]==0).sum()), ignored_gradient_max=ignored_max)
    assignments = []
    for j, (bi, anchor) in enumerate(positions.tolist()):
        candidates = (batch['batch_idx'].long()==bi).nonzero().flatten()
        instance = int(candidates[int(target_idx[bi,anchor])]); filename=Path(batch['im_file'][bi]).name
        for corner in range(9):
            assignments.append(dict(image=filename, role='SOURCE' if filename.startswith('syn__') else 'REAL',
                anchor=anchor, corner=corner, stride=float(stride[j]), support=int(gt[j,corner,2]),
                covered=bool(covered[instance,corner]), error_stride=float((pred[j,corner,:2]-gt[j,corner,:2]).norm()),
                error_network_px=float((pred[j,corner,:2]-gt[j,corner,:2]).norm()*stride[j]),
                e=float(e[j,corner]), exp_minus_e=float(torch.exp(-e[j,corner])), sigma=sigma[j,corner].detach().tolist(),
                standardized=standardized[j,corner].detach().tolist(), clipped_axes=int((standardized[j,corner].abs()>100).sum()),
                gradients={k:dict(xy=float(v[j,corner,:2].norm()), sigma=float(v[j,corner,-2:].norm()),
                    confidence=float(v[j,corner,2].abs())) for k,v in gradients.items()}))
    supervised = [r for r in assignments if r['support']==2]
    result = dict(branch=name, foreground_anchors=len(positions), supervised_assigned_points=len(supervised),
        ignored_assigned_points=int(ignored.sum()), e=distribution(e[mask].detach().cpu()),
        exp_minus_e=distribution(torch.exp(-e[mask]).detach().cpu()),
        sigma=distribution(sigma[mask].detach().cpu()), standardized_abs=distribution(standardized[mask].abs().detach().cpu()),
        clipped_points=sum(r['clipped_axes']>0 for r in supervised), e_above10=sum(r['e']>10 for r in supervised),
        rle_before_clamp=float(row['rle_before_clamp'].detach()), rle_after_clamp=float(row['terms'][2].detach()),
        rle_aggregate_clamped=float(row['rle_before_clamp'].detach())<0, terms=info,
        groups={})
    for group, test in [('source',lambda r:r['role']=='SOURCE'), ('real',lambda r:r['role']=='REAL'),
                        ('covered',lambda r:r['covered']), ('remaining_real',lambda r:r['role']=='REAL' and not r['covered']),
                        ('e_gt10',lambda r:r['e']>10), ('e_le1',lambda r:r['e']<=1)]:
        sub=[r for r in supervised if test(r)]
        result['groups'][group]=dict(n=len(sub), e=distribution([r['e'] for r in sub]),
            gradients={t:distribution([r['gradients'][t]['xy'] for r in sub]) for t in terms})
    return result, assignments


def probe():
    start=time.monotonic(); torch.set_num_threads(4); cv2.setNumThreads(1)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.benchmark=False
    prepared={kind:dataset_fixture(kind) for kind in ('C3','MAIN')}
    results, private, inputs=[] , [], []
    jobs=[('C3_MANUAL9','C3',C.read(C3/'FIT_MANUAL9.json')['checkpoint'],5),
          ('MAIN_R0','MAIN',C.checkpoint('PLASTIC','R0'),0),
          ('MAIN_REF','MAIN',C.checkpoint('PLASTIC','REF_LR5'),5)]
    for label,kind,binding,updates in jobs:
        p,b0,bocc,occ_covered,occ_plans=prepared[kind]; model=load_model(binding,p['args'],'cuda:0')
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.criterion=make_criterion(model)
        for _ in range(updates):model.criterion.update()
        params=[v for v in model.parameters() if v.requires_grad]
        for mode in ('CLEAR','OCCLUDED'):
            if mode=='OCCLUDED':batch,covered,plans=copy.deepcopy(bocc),occ_covered,occ_plans
            else:batch=copy.deepcopy(b0);covered=torch.zeros_like(batch['keypoints'][...,2],dtype=torch.bool);plans=[]
            inputs.append(dict(job=label,mode=mode,images=tensor_digest(batch['img']),
                keypoints=tensor_digest(batch['keypoints']),boxes=tensor_digest(batch['bboxes']),plans=plans))
            batch={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in batch.items()}
            batch['img']=batch['img'].float()/255
            with attach_signal_hooks(model.criterion) as hook:
                predictions=model(batch['img'])
                loss,items=model.criterion(predictions,batch)
                captured=[]
                for name,row in hook.rows.items():
                    weight=model.criterion.o2m if name=='one2many' else model.criterion.o2o
                    summary,assignments=summarize_hook(name,row,weight,batch,covered,params)
                    if mode=='CLEAR' and label in ('C3_MANUAL9','MAIN_R0'):
                        summary['finite_difference_actual_fixed_assignment']=finite_difference_actual(row)
                    captured.append(summary);private.append(dict(job=label,mode=mode,branch=name,assignments=assignments))
                # Same tensor highest-score one2one candidate: independent of TRAIN target choice.
                branch=hook.rows['one2one'];fg,idx,kp,bi,st,box,pr=branch['args']
                score=predictions['one2one']['scores'].sigmoid().max(1).values
                deployment=[]
                for i in range(len(batch['img'])):
                    anchor=int(score[i].argmax()); instance=(batch['batch_idx'].long()==i).nonzero().flatten()
                    assert len(instance)==1
                    gt=batch['keypoints'][instance[0]]; pred=pr[i,anchor,:,:2]*st[anchor]
                    err=(pred-gt[:,:2]*torch.tensor([640,640],device=pred.device)).norm(dim=-1)
                    deployment.append(dict(image=Path(batch['im_file'][i]).name,anchor=anchor,score=float(score[i,anchor]),
                        is_assigned=bool(fg[i,anchor]),errors_network_px=err.detach().tolist(),support=gt[:,2].detach().tolist()))
                # Hook transparency: same deterministic raw predictions through unwrapped original loss.
            check,_=model.criterion(predictions,batch)
            assert torch.equal(loss.detach(),check.detach()),'Hooks changed criterion values'
            results.append(dict(job=label,mode=mode,checkpoint=binding,branches=captured,
                hook_value_exact=True,branch_weights=dict(o2m=model.criterion.o2m,o2o=model.criterion.o2o),
                gains=dict(pose=model.args.pose,kobj=model.args.kobj,rle=model.args.rle),
                deployment=deployment,AMP=False,batch_size=4,
                gradient_units='weighted objective per image: original gain x branch coefficient; actual trainer loss multiplies batch4'))
            print('PROBE',label,mode,[(r['branch'],r['e_above10'],r['clipped_points'],r['rle_aggregate_clamped']) for r in captured],flush=True)
            del predictions,loss,check,hook
        assert set(state)==set(model.state_dict())
        assert all(torch.equal(v,state[k]) for k,v in ((k,v.detach().cpu()) for k,v in model.state_dict().items()))
        for r in results:
            if r['job']==label:r['all_state_tensors_exact']=len(state)
        del model;torch.cuda.empty_cache()
    C.save(RAW/'ASSIGNMENTS_PRIVATE.json',private,True)
    C.save(RAW/'INPUTS_PRIVATE.json',inputs,True)
    C.save(RAW/'PROBE_RESULTS.json',dict(results=results,fits=0,optimizer_updates=0,
        optimizer_constructed=False,evaluation_reference_read=False,seconds=time.monotonic()-start,
        limitations=['Few fixed TRAIN examples, not population prevalence or causal proof.',
        'C3 historical real-affineOFF and main vanilla are separate contexts, not a paired intervention.',
        'Frozen EMA checkpoint, FP32 diagnostic, not recovered online optimizer/AMP state.',
        'Direct-click provenance does not establish signed physical corner identity.']),True)
    print('PROBE_COMPLETE_SECONDS',time.monotonic()-start,flush=True)


def report():
    import importlib.metadata
    import ultralytics.utils.loss as installed_loss
    import ultralytics.nn.modules.head as installed_head
    import ultralytics.utils.tal as installed_tal
    raw=C.read(RAW/'PROBE_RESULTS.json'); results=raw['results']
    assignments=C.read(RAW/'ASSIGNMENTS_PRIVATE.json')
    c3_native=[r for r in C.read(C3/'TRAIN_POINT_METRICS.json')
        if r['arm']=='MANUAL9' and r['errors']['MANUAL9']>20]
    assert {(r['id'],r['corner']) for r in c3_native}=={('wood_day_01:002141',j) for j in (0,4,5)}
    bad=[]
    for corner in (0,4,5):
        private=next(r for r in assignments if r['job']=='C3_MANUAL9' and r['mode']=='CLEAR' and r['branch']=='one2one')
        row=next(r for r in private['assignments'] if 'wood_day_01__002141' in r['image'] and r['corner']==corner)
        native=next(r for r in c3_native if r['corner']==corner)
        dep=next(r for r in next(r for r in results if r['job']=='C3_MANUAL9' and r['mode']=='CLEAR')['deployment']
                 if 'wood_day_01__002141' in r['image'])
        assert dep['is_assigned'] and dep['anchor']==row['anchor']
        assert abs(dep['errors_network_px'][corner]-row['error_network_px'])<1e-5
        bad.append(dict(id=native['id'],corner=corner,native_cached_error_px=native['errors']['MANUAL9'],
            current_probe_error_network_px=row['error_network_px'],stride=row['stride'],e=row['e'],
            exp_minus_e=row['exp_minus_e'],clipped_axes=row['clipped_axes'],
            highest_score_is_same_assigned_anchor=True,gradients=row['gradients']))
    base=next(r for r in results if r['job']=='MAIN_R0' and r['mode']=='CLEAR')
    loc_norm=float(np.sqrt(sum(b['terms']['location']['head_gradient_norm']**2 for b in base['branches'])))
    extra_norm=float(np.sqrt(sum(b['terms']['smooth_l1_unit']['head_gradient_norm']**2 for b in base['branches'])))
    coefficient=.1*loc_norm/extra_norm
    finite=[t for r in results for b in r['branches'] for t in b.get('finite_difference_actual_fixed_assignment',[])]
    assert len(finite)==16 and sum(len(t['checks']) for t in finite)==64
    branches=[b for r in results for b in r['branches']]
    assert all(r['hook_value_exact'] and r['all_state_tensors_exact']==879 for r in results)
    assert all(t['ignored_gradient_max']==0 for b in branches for t in b['terms'].values())
    installed=[]
    for module in (installed_loss,installed_head,installed_tal):
        path=Path(inspect.getfile(module))
        installed.append(dict(path=str(path),sha256=C.sha(path),bytes=path.stat().st_size))
    # Public aggregate excludes native/target arrays and individual private filenames.
    public_rows=[]
    for r in results:
        public_rows.append({k:r[k] for k in ('job','mode','checkpoint','branches','hook_value_exact',
            'branch_weights','gains','AMP','batch_size','gradient_units','all_state_tensors_exact')})
    fixture_bindings=[]
    for folder in (RAW/'fixtures/C3',RAW/'fixtures/MAIN'):
        for path in sorted((folder/'images').iterdir()):
            fixture_bindings.append(C.bind(path))
            fixture_bindings.append(C.bind(folder/'labels'/path.with_suffix('.txt').name))
    source=Path(__file__)
    payload=dict(status='PASS_BOUNDED_TRAIN_DIAGNOSTIC',utc=C.now(),fits=0,optimizer_updates=0,
        optimizer_constructed=False,gpu_elapsed_seconds=raw['seconds'],evaluation_reference_read=False,
        installed_ultralytics=importlib.metadata.version('ultralytics'),installed_files=installed,
        implementation=[C.bind(source),C.bind(source.with_name('test_loss_signal.py')),
            C.bind(C.ROOT/'scripts/self_training_yolo/v3/true_ignore_pose_loss.py')],
        artifacts=[C.bind(RAW/f) for f in ('SINGLE_POINT_CURVE.json','PROBE_RESULTS.json','ASSIGNMENTS_PRIVATE.json','INPUTS_PRIVATE.json')],
        input_bindings=fixture_bindings+[C.bind(C3/'TRAIN_POINT_METRICS.json'),C.bind(C3/'TRAIN_TARGETS_PRIVATE.json'),
            C.bind(C3DOC/'PROTOCOL.json'),C.bind(C3DOC/'PROTOCOL_V2.json'),C.bind(C.P.REC/'pose_only/PROTOCOL.json')],
        execution_note='Core probe and objective logic unchanged after GPU run; report aggregation and unused helper cleanup followed. Full current source is bound for subsequent B lock.',
        sample='C3 failed TRAIN frame + one Plastic manual + two source; MAIN first two mask-feasible fixed A preflight REAL plus two source. No model/error ranking. Each clear/masked batch4 has 1:1 real:source.',
        actual_occlusion='SharedOcclusion with same original main index seeds280901+index; same masks in RAW/REF A policy. Labels unchanged. Not independent augmentation implementation.',
        units=dict(location='stride-normalized xy and GT box area; e=||delta||²/((2sigma_OKS)²·area·2)',
            oks_sigma='all9 keypoints 1/9; fixed, not RLE predicted uncertainty',
            rle='(pred_xy-gt_xy)/(sigmoid(pred_sigma_logits)+1e-9), stride-coordinate residual, then per-axis [-100,100]',
            rle_aggregate='Learned flow + residual density; averaged by visible point count, coordinate sum; final whole branch clamp(min=0)',
            visibility='BCE sum over visibility!=1 / count(visibility!=1), target(visibility==2)',
            gradient='hyp gain x E2E branch weight, per-image objective; actual batch-summed loss multiplies batch4',
            native='Cached historical native errors kept separate from current padded640 HSV probe errors; no native==network assumption'),
        checks=dict(hook_value_exact=6,all_state_tensors_exact_per_checkpoint=879,checkpoints=3,
            ignored_gradient_max=0.,finite_difference_checks=64,finite_difference_passed=True,
            fixed_assignment_leaf_descent_checks=16,leaf_descent_passed=True,unit_tests=5),
        C3_failed_original_ID_points=bad,results=public_rows,
        calibration=dict(candidate='additive coordinate-wise SmoothL1 on detached existing OKS normalization',
            function='make_supplement_criterion(model, coefficient, beta=1.)',coefficient=coefficient,beta=1.,
            desired_initial_head_gradient_ratio=.1,measured_existing_location_head_norm=loc_norm,
            measured_unit_supplement_head_norm=extra_norm,batch='MAIN_R0 CLEAR same actual TRAIN batch4',
            branch_weights=base['branch_weights'],gains=base['gains'],
            branch_joint_norm='sqrt(sum(branch norms squared)): location and supplement one2many/one2one pose parameters disjoint; flow unused by both',
            reduction='K/n_supervised per anchor, mean over foreground anchors x K; identical location support and normalization',
            transition='Each abs(z_axis)=1; ||z||²=e. Equals sqrt(e)=1 only for one-axis residual, not a radial transition.',
            ignored='visibility1 exactly zero xy/sigma/confidence gradients; visibility0 retains original negative BCE',
            unchanged='original detector, assignments, RLE/clamps/gain, BCE, LR, branch weights, EMA, source ratio, augmentation; B recipe chooses vanilla no input mask',
            rationale='Small 10% auxiliary signal capability test selected by TRAIN norms. Not tuned against PCK/DEV, not a proven causal repair.'),
        supported=['Actual MAIN sample 8/8 branch-context RLE aggregate clamps to zero; its coordinate, sigma and flow head gradients are zero in those contexts.',
            'C3 original corner4 has exp(-e)=0.00222 and attenuated location signal, but RLE and combined gradients remain target-directed.',
            'C3 failed3 use the same highest-score/one2one assigned anchor on the probe input; a candidate/assignment mismatch does not explain these samples.',
            'No supervised point reaches standardized RLE +-100 clipping in these bounded actual batches.'],
        unsupported=['Not all C3 failure is caused by saturation, and no total gradient starvation established.',
            'Not population frequencies; chosen few TRAIN frames/source and EMA states do not reconstruct complete training dynamics.',
            'No claim of physical signed-axis correctness, true corner identity, unseen generalization, or lower pose error from a proxy loss change.',
            'Positive coordinate-leaf target descent is not a guarantee that a shared-parameter optimizer step improves every keypoint.'])
    C.save(C.DOC/'LOSS_SIGNAL_AUDIT.json',payload)
    table=C.table(['context','branch','supervised assigned','e max','RLE before→after','location head norm','RLE head norm'],[
        [r['job']+' '+r['mode'],b['branch'],b['supervised_assigned_points'],f"{b['e']['max']:.4f}",
         f"{b['rle_before_clamp']:.5f}→{b['rle_after_clamp']:.5f}",f"{b['terms']['location']['head_gradient_norm']:.5f}",
         f"{b['terms']['rle']['head_gradient_norm']:.5f}"] for r in results for b in r['branches']])
    badtable=C.table(['original corner','cached native error px','probe input640 error px','e','exp(-e)','location xy grad','RLE xy grad','combined xy grad'],[
        [r['corner'],f"{r['native_cached_error_px']:.3f}",f"{r['current_probe_error_network_px']:.3f}",f"{r['e']:.5f}",
         f"{r['exp_minus_e']:.6f}",*[f"{r['gradients'][t]['xy']:.6f}" for t in ('location','rle','combined')]] for r in bad])
    text=f'''# LOSS SIGNAL AUDIT — 실제 TRAIN graph, optimizer 0

결론: MAIN 소표본에서는 RLE의 branch aggregate clamp가 실제로 켜졌지만 위치 신호는 남았다. C3 실패점 중 corner4의 위치 신호는 강하게 감쇠했으나 RLE/combined 신호가 남았다. 따라서 “C3가 전체 gradient 소실 때문에 실패했다”는 결론은 지지하지 않는다. 작은 보조 회귀항 B는 **능력 시험**이지 확정 원인 수정이 아니다.

## 범위와 검증

실제 설치 Ultralytics {payload['installed_ultralytics']} + 기존 TrueIgnorePoseLoss26/E2ELoss를 hook했다. 복제한 유사 criterion으로 원래 결과를 대체하지 않았다. 3 checkpoint, clear/occluded 각 batch4(REAL2+SOURCE2), optimizer 생성/step/fit 0, GPU 구간 {raw['seconds']:.3f}s. 879개 state tensor/checkpoint 모두 bit-exact, 원 loss와 hook loss 6회 bit-exact, ignored 전 channel gradient 최대0, actual fixed-assignment finite difference64/64 및 leaf descent16/16 통과. CPU 단위시험5개 통과.

C3는 원래 real-affineOFF의 과거 조건이다. MAIN은 기존 vanilla affine/HSV를 유지하며 실제 A의 SharedOcclusion과 원 index seed를 재사용했다. MAIN REAL2는 이미 고정한 A preflight에서 처음 mask가 가능했던 사례이며 prediction/error/DEV로 고르지 않았다. batch4는 전체 batch16 학습 역학을 대체하지 않는다. 원 recipe AMP=False와 같은 FP32이며 BN/EMA checkpoint는 고정했다. raw arrays, RGB 및 실제 좌표는 private data namespace에만 있다.

## 실제 branch 결과

{table}

MAIN 8/8 branch-context에서 pre-clamp RLE가 음수여서 coordinate뿐 아니라 sigma/flow gradient도0이었다. 이 8/8은 선택한 배치의 결과이지 전체 TRAIN 빈도가 아니다. C3는 RLE clamp가 꺼져 있고 location/RLE 둘 다 양의 target-radial gradient를 보였다. 실제 측정한 supervised point의 ±100 표준화 clamp는0건이다.

## C3 실패3의 identity/assignment 분리

원래 ID wood_day_01:002141의 corner0/4/5를 그대로 사용했다. native cached deployment error와 현재 padded640+HSV probe 오차는 입력·단위가 다르므로 동일 수치로 취급하지 않는다. 현재 입력 최고 confidence one2one anchor는 실제 TAL assigned anchor와 같다. corner의 nearest matching이나 GT 기반 candidate 교체를 하지 않았다.

{badtable}

corner4 exp(-e)=0.00222는 위치항 감쇠의 실제 증거지만 RLE xy gradient0.12524와 combined0.12970이 남는다. 세 점 모두 RLE standardized clip이 없고 coordinate-leaf combined 신호가 target 방향이다. manual_click provenance는 물리 signed-axis/corner 정답을 확정하지 않는다.

## 단위·감쇠·clamp

GT는 normalized→입력640px→각 anchor stride 좌표로 변환되고 GT box area도 stride²로 나뉜다. 기존 location은 `e=||delta||²/((2σ_OKS)²·area·2)`, `1-exp(-e)`이며 9점 σ_OKS=1/9다. 이는 RLE의 예측 sigma와 다르다. RLE는 stride residual을 sigmoid(scale logit)으로 나눈 뒤 각 축 ±100 clamp, learned flow+residual density, supervised point 평균을 거친 뒤 **branch 전체 scalar**를 min0 clamp한다. visibility는 ignore(v1)를 분자·분모에서 빼고 v0 negative를 유지한다.

99개 artificial single-point curve는 실제 installed KeypointLoss/RLE/learned C3 flow를 호출했다. native256px 단일축 예시에서 e8.4406, exp(-e)0.0002159, location gradient0.0001495 대 normalizedSmoothL1 0.11916이었다. 이는 mechanism isolate이며 관측 빈도/실패 인과가 아니다. 한 축 ±100 clamp 시 그 축 residual 경로는0이어도 flow coupling으로 다른 축/scale gradient는 남을 수 있다. 실제 curve에는 축별 gradient도 저장했다.

## B 최소 후보와 TRAIN calibration

`z=(pred-target)/sqrt((2σ_OKS)²·detached(area)·2)`에 coordinate-wise SmoothL1을 더한다. β=1은 각 |z_axis|=1 전환이며 ||z||²=e; 일반2D에서 radial sqrt(e)=1 전환이라고 하지 않는다. mask와 K/n_supervised × mean(anchor,K) denominator는 기존 location과 같다. area/σ detach, v1은 xy/confidence/scale 모두0, v0 BCE 기존 동작 보존. coefficient0에서 원 location/visibility/RLE tuple bit-exact를 확인했다.

MAIN R0 clear TRAIN batch의 weighted head norm: 원 location {loc_norm:.12f}, unit supplement {extra_norm:.12f}. `λ=.1*original/unit={coefficient:.17g}`로 초기 보조 신호를 원 위치항의10%로 잠근다. branch .8/.2, pose gain12, kobj1, rle{base['gains']['rle']}를 적용한 per-image objective 기준이며 실제 batch loss에는 batch4가 곱해진다. 양 branch location parameter support가 분리되어 joint norm은 branch norm 제곱합의 제곱근이다. PCK10/DEV나 후속 B 결과로 β/λ를 선택하지 않았다. B fit은 부모의 별도 protocol 승인 대상이다.

API: `make_supplement_criterion(model, {coefficient:.17g}, beta=1.)`. top-level SupplementLoss가 기존 criterion을 먼저 호출하고 location 보조항만 추가한다. `_setup_train` 뒤 model.criterion을 직접 설치하면 이미 존재한 criterion 누락을 피한다. EMA validation에 설치할 때는 EMA 자신의 flow를 참조하는 criterion을 별도 생성한다. E2E update schedule, RLE clamp/gain, LR, mask/source ratio, assignment 및 detector는 유지한다.

## 해석 제한

가장 가까운 원인 후보를 분리한 작은 TRAIN 진단이며 보조항이 자연 Moderate+Severe의 T/R을 개선한다는 보장은 없다. aggregate clamp 활성만으로 제거가 옳다는 결론도 없다. frozen EMA/소배치 graph는 online optimizer 전 궤적이 아니다. coordinate-leaf target descent는 공유 head weight 업데이트 후 모든 점 개선을 보장하지 않는다. 실제 물리 정답은 여전히 미확정이다.

정확한 구현/설치본/입력/checkpoint/hash와 세부 값: [LOSS_SIGNAL_AUDIT.json](LOSS_SIGNAL_AUDIT.json). 실행 core는 GPU 진단 이후 변경하지 않았고 report/미사용 helper 정리 후 source 전체를 후속 B lock용으로 바인딩했다.
'''
    C.save(C.DOC/'LOSS_SIGNAL_AUDIT.md',text)
    print('REPORT_COMPLETE_COEFFICIENT',coefficient,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['curve','probe','report'])
    parser.add_argument('--allow-gpu',action='store_true');args=parser.parse_args()
    if args.phase=='probe':assert args.allow_gpu,'Parent GPU authorization required';probe()
    elif args.phase=='curve':curve()
    else:report()
