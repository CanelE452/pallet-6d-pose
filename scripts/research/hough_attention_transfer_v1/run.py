"""Run synthetic-only A/B training through real DEV evaluation and visualization.

Usage: python run.py --run-dir ... [--phase prepare|check|sanity|train|evaluate|all]
All arm settings are fixed before real-image inference; evaluation never selects
a checkpoint. Existing runs are resumed only when configuration matches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import core as C


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def settings(args):
    return dict(schema='hough_attention_transfer_v1', steps=args.steps,
                seeds=args.seeds, batch=args.batch, lr=.001, weight_decay=.0001,
                attention_lambda=args.attention_lambda, pad_px=100, input_size=400,
                train_n=2048, val_n=256, test_n=256, cross_n=128, split_seed=17,
                backbone=str(args.backbone.resolve()), backbone_sha256=C.sha(args.backbone),
                arms={'A': 'line CE only', 'B': 'line CE + lambda * -log foreground attention mass'},
                checkpoint_selection='fixed final step, identical for both arms; no real selection',
                target_semantics='12 camera-dynamic cuboid supporting lines, NOT visible material edges',
                attention_target='visible pallet mask, shared across supported roles; padded border zero',
                primary_metric='original-image mean GT-endpoint to predicted infinite-line distance / image diagonal',
                inference='full padded image; no GT crop, GT masks or GT points fed to model',
                limitations=['Real population is development, not a new final test.',
                             'Frozen backbone inherited synthetic source manifest was deleted; exact pretraining image overlap unverified.',
                             'Foreground attention is a coarse localization prior, not role-specific visible-edge supervision.',
                             'Attention weights measure token mixing, not proof of causal pixel importance.'])


def load_records(out):
    doc = json.loads((out/'manifest.json').read_text())
    pops = doc.get('populations', doc)
    records = []
    for name, rows in pops.items():
        for row in rows:
            row = dict(row, population=name, index=len(records))
            records.append(row)
    return records


def cache_data(out, cfg, dh):
    from dataset import build_manifest
    if not (out/'manifest.json').exists():
        build_manifest(C.ROOT, out, seed=cfg['split_seed'], train_n=cfg['train_n'],
                       val_n=cfg['val_n'], test_n=cfg['test_n'], cross_n=cfg['cross_n'])
    records = load_records(out)
    signature = {'config_sha256': C.sha(out/'CONFIG.json'), 'manifest_sha256': C.sha(out/'manifest.json')}
    if (out/'CACHE.json').exists():
        old = json.loads((out/'CACHE.json').read_text())
        if old['signature'] != signature:
            raise RuntimeError('Cache provenance does not match config/manifest')
        return records
    backbone = C.load_backbone(cfg['backbone'], dh.DEV)
    feats = np.lib.format.open_memmap(out/'features.npy', mode='w+', dtype=np.float16,
                                     shape=(len(records),128,50,50))
    masks = np.zeros((len(records),50,50), np.float32)
    grids = np.zeros((len(records),8,2), np.float64)
    for start in range(0, len(records), 12):
        batch = records[start:start+12]
        prepared = [C.prepare(r, cfg['pad_px']) for r in batch]
        ims, gr, ma = map(np.stack, zip(*prepared))
        with torch.inference_mode():
            f = backbone(torch.from_numpy(ims).to(dh.DEV)).float()
        if f.shape[1:] != (128,50,50) or not torch.isfinite(f).all():
            raise RuntimeError('Invalid frozen features')
        feats[start:start+len(batch)] = f.cpu().numpy().astype(np.float16)
        grids[start:start+len(batch)] = gr
        masks[start:start+len(batch)] = ma
        if start % 240 == 0:
            log(f"cache {start}/{len(records)} {batch[0]['population']}")
    feats.flush()
    tc, rc, sup = C.geometry(dh, records, grids)
    np.savez_compressed(out/'targets.npz', theta=tc.cpu().numpy(), rho=rc.cpu().numpy(),
                        supported=sup.cpu().numpy(), masks=masks, grids=grids)
    C.write_json(out/'CACHE.json', {'signature':signature, 'frames':len(records),
                                  'features_dtype':'float16 stored, float32 training',
                                  'geometry':'canonical Hough target; original-image segment support',
                                  'backbone_frozen':True})
    del backbone, feats
    torch.cuda.empty_cache()
    return records


class Data:
    def __init__(self, out, dh):
        self.out, self.dh = out, dh
        self.records = load_records(out)
        self.features = np.load(out/'features.npy', mmap_mode='r')
        a = np.load(out/'targets.npz')
        self.theta = torch.from_numpy(a['theta']).to(dh.DEV)
        self.rho = torch.from_numpy(a['rho']).to(dh.DEV)
        self.support = torch.from_numpy(a['supported']).to(dh.DEV)
        self.masks = torch.from_numpy(a['masks']).to(dh.DEV)
        gt, gr, valid = dh.lattice()
        self.gt, self.gr = gt[valid], gr[valid]
        self.valid = torch.ones_like(self.gt, dtype=torch.bool)
        self.hypothesis = dh.hypothesis_features(self.gt, self.gr)
        self.populations = {}
        for r in self.records:
            self.populations.setdefault(r['population'], []).append(r['index'])

    def batch(self, idx):
        f = torch.from_numpy(np.asarray(self.features[idx])).to(self.dh.DEV, dtype=torch.float32)
        return f, self.theta[idx], self.rho[idx], self.support[idx], self.masks[idx]

    def target(self, t, r):
        return self.dh.target_distribution(t.reshape(-1), r.reshape(-1), self.gt,
                                           self.gr, self.valid).reshape(*t.shape, -1)


def model_for(dh, seed):
    dh.CAP.SEED = seed  # Original constructor seeds from this constant.
    return dh.DirectHoughModel().to(dh.DEV)


def model_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def checks(data, cfg):
    dh = data.dh
    m = model_for(dh, 123)
    f,t,r,s,mask = data.batch(data.populations['synth_train'][:2])
    m.eval()
    original = m(f, data.hypothesis)
    got, weights = C.forward_attention(m, f, data.hypothesis)
    error = float((original-got).abs().max())
    if not torch.allclose(original, got, atol=2e-5, rtol=2e-4):
        raise RuntimeError(f'Attention extraction changes forward: {error}')
    if not torch.allclose(weights.sum((-1,-2)), torch.ones_like(s, dtype=torch.float32), atol=1e-5):
        raise RuntimeError('Attention weights not normalized')
    loss, mass = C.foreground_loss(weights, mask, s)
    m.zero_grad(); loss.backward()
    norm = float(m.encoder.attention.in_proj_weight.grad.norm())
    if not norm > 0 or not torch.isfinite(loss):
        raise RuntimeError('Foreground loss does not train attention')
    # Analytical behavior: moving mass into foreground must lower supervision loss.
    toy = torch.zeros((1,12,2,2)); toy[...,0,0]=.1; toy[...,1,1]=.9
    tm = torch.tensor([[[1.,0.],[0.,0.]]]); ts = torch.ones((1,12), dtype=torch.bool)
    low,_ = C.foreground_loss(toy.flip((-1,-2)),tm,ts)
    high,_ = C.foreground_loss(toy,tm,ts)
    assert low < high
    # Exact known line survives non-square original/padded coordinate transforms.
    theta = torch.tensor([0.,90.], device=dh.DEV)
    rho = torch.tensor([-20.,10.], device=dh.DEV)
    lines = C.line_pixels(dh,theta,rho,640,480,cfg['pad_px'])
    rad, cr = dh.canonical_from_centred(theta,rho)
    back = (lines+cfg['pad_px'])*[50/840,50/680]
    residual = np.abs((back*np.stack([rad.cos().cpu(),rad.sin().cpu()],-1)[:,None]).sum(-1)-cr.cpu().numpy()[:,None]).max()
    assert residual < 1e-4
    report = {'forward_max_abs_difference':error,'attention_gradient_norm':norm,
              'coordinate_roundtrip_residual_cell':float(residual),'PASS':True}
    C.write_json(data.out/'CHECKS.json', report)
    log(f'checks PASS {report}')


@torch.no_grad()
def evaluate_model(data, model, indices, cfg, arm, seed, collect=False):
    model.eval(); rows=[]; artifacts={}; dh=data.dh
    for start in range(0,len(indices),cfg['batch']):
        idx=indices[start:start+cfg['batch']]
        f,t,r,s,mask=data.batch(idx)
        scores,att=C.forward_attention(model,f,data.hypothesis)
        th,rh=dh.decode(scores.reshape(-1,len(data.gt)),data.gt,data.gr,data.valid)
        th=th.reshape(-1,12); rh=rh.reshape(-1,12)
        for b,i in enumerate(idx):
            rec=data.records[i]
            lines=C.line_pixels(dh,th[b],rh[b],rec['width'],rec['height'],cfg['pad_px'])
            angle,dist=C.pixel_errors(lines,rec['gt_points'])
            supported=s[b].cpu().numpy()
            ca,co=dh.measure(th[b],rh[b],t[b],r[b])
            a=att[b].cpu().numpy()
            ent=-(a*np.log(a.clip(1e-12))).sum((1,2))/np.log(2500)
            has_mask=bool(rec.get('mask'))
            area=float(mask[b].mean())
            mass=(a*mask[b].cpu().numpy()[None]).sum((1,2))
            diag=np.hypot(rec['width'],rec['height'])
            for role in np.flatnonzero(supported):
                rows.append(dict(id=rec['id'],population=rec['population'],group=rec.get('group',''),
                                 arm=arm,seed=seed,role=int(role),angle_deg=float(angle[role]),
                                 distance_px=float(dist[role]),distance_diagonal=float(dist[role]/diag),
                                 canonical_angle_deg=float(ca[role]),canonical_offset_cell=float(co[role]),
                                 attention_entropy=float(ent[role]),
                                 foreground_mass=float(mass[role]) if has_mask else None,
                                 foreground_lift=float(mass[role]/max(area,1e-9)) if has_mask else None))
            if collect:
                artifacts[rec['id']]={'attention':a,'lines':lines,'angle':angle,'distance':dist,
                                      'supported':supported,'foreground_mask':mask[b].cpu().numpy()}
    return rows, artifacts


def summarize(rows):
    result={}
    for key in ('angle_deg','distance_px','distance_diagonal','canonical_angle_deg',
                'canonical_offset_cell','attention_entropy','foreground_mass','foreground_lift'):
        v=[r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        if v:
            result[key]={'median':float(np.median(v)),'p90':float(np.percentile(v,90)),
                         'mean':float(np.mean(v))}
    result.update(n_roles=len(rows),n_frames=len(set(r['id'] for r in rows)))
    if rows:
        result['fraction_angle_le5_and_distance_le1pct']=float(np.mean([
            r['angle_deg']<=5 and r['distance_diagonal']<=.01 for r in rows]))
    return result


def train_one(data,cfg,arm,seed,steps=None,indices=None,tag=None):
    steps=steps or cfg['steps']; tag=tag or f'{arm}_seed{seed}'
    target=data.out/f'{tag}_final.pth'
    if target.exists() and (data.out/f'{tag}_HISTORY.json').exists():
        log(f'{tag} complete; verified resume')
        return
    indices=indices or data.populations['synth_train']
    assert all(data.records[i]['population']=='synth_train' for i in indices)
    model=model_for(data.dh,seed)
    initial_sha=model_sha(model)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed)
    schedule=np.asarray(indices).copy()
    pointer=len(schedule); history=[]; losses=[]; at_losses=[]; t0=time.monotonic()
    for step in range(1,steps+1):
        if pointer+cfg['batch']>len(schedule):
            rng.shuffle(schedule); pointer=0
        idx=schedule[pointer:pointer+cfg['batch']].tolist(); pointer+=cfg['batch']
        model.train()
        f,t,r,s,mask=data.batch(idx)
        targets=data.target(t,r)
        scores,weights=C.forward_attention(model,f,data.hypothesis)
        line=data.dh.cross_entropy(scores,targets,s,data.valid)
        attention,mass=C.foreground_loss(weights,mask,s)
        loss=line+(cfg['attention_lambda'] if arm=='B' else 0.)*attention
        if not torch.isfinite(loss):
            raise RuntimeError(f'{tag}: nonfinite loss at {step}')
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(line.detach())); at_losses.append(float(attention.detach()))
        if step%250==0 or step==steps:
            log(f'{tag} {step}/{steps} line={np.mean(losses[-250:]):.4f} '
                f'att={np.mean(at_losses[-250:]):.4f} elapsed={(time.monotonic()-t0)/60:.1f}m')
        if step in (500,1500,3000,6000) or step==steps:
            train_rows,_=evaluate_model(data,model,indices[:min(64,len(indices))],cfg,arm,seed)
            entry={'step':step,'line_loss':float(np.mean(losses[-250:])),
                   'attention_loss':float(np.mean(at_losses[-250:])),
                   'train_diagnostic':summarize(train_rows)}
            if tag.startswith('SANITY'):
                entry['synthetic_val']=None
            else:
                val_rows,_=evaluate_model(data,model,data.populations['synth_val'],cfg,arm,seed)
                entry['synthetic_val']=summarize(val_rows)
            history.append(entry)
            C.write_json(data.out/f'{tag}_progress.json',history)
    torch.save({'model':model.state_dict(),'config':cfg,'arm':arm,'seed':seed,'steps':steps,
                'initial_state_sha256':initial_sha,'manifest_sha256':C.sha(data.out/'manifest.json')},target)
    report={'arm':arm,'seed':seed,'steps':steps,'train_count':len(indices),
            'initial_state_sha256':initial_sha,'checkpoint_sha256':C.sha(target),
            'elapsed_seconds':time.monotonic()-t0,'history':history}
    C.write_json(data.out/f'{tag}_HISTORY.json',report)
    log(f'{tag} DONE')
    del model,opt; torch.cuda.empty_cache()


def final_evaluation(data,cfg):
    """Real inference happens only here, after all six fixed-step runs finish."""
    rows=[]; art={}; reports={}
    populations={k:v for k,v in data.populations.items() if k!='synth_train'}
    # Predetermined visualization samples: hashed IDs, independent of prediction.
    viz_ids=[]
    for name,indices in populations.items():
        if name=='synth_val': continue
        ranked=sorted(indices,key=lambda i:hashlib.sha256(data.records[i]['id'].encode()).hexdigest())
        if name=='real_dev':
            groups=sorted(set(data.records[i].get('group','') for i in ranked))
            selected=[]
            for group in groups:
                selected.extend([i for i in ranked if data.records[i].get('group','')==group][:2])
            viz_ids.extend(selected[:12])
        else:
            viz_ids.extend(ranked[:3])
    all_eval=[i for inds in populations.values() for i in inds]
    for seed in cfg['seeds']:
        initial=[]
        for arm in ('A','B'):
            p=data.out/f'{arm}_seed{seed}_final.pth'
            checkpoint=torch.load(p,map_location='cpu',weights_only=False)
            if checkpoint['config']!=cfg or checkpoint['manifest_sha256']!=C.sha(data.out/'manifest.json'):
                raise RuntimeError('Evaluation checkpoint provenance mismatch')
            initial.append(checkpoint['initial_state_sha256'])
            model=model_for(data.dh,seed)
            model.load_state_dict(checkpoint['model'],strict=True)
            rr,_=evaluate_model(data,model,all_eval,cfg,arm,seed)
            rows.extend(rr)
            reports[f'{arm}_seed{seed}']={name:summarize([r for r in rr if r['population']==name])
                                         for name in populations}
            if seed==cfg['seeds'][0]:
                _,art[arm]=evaluate_model(data,model,viz_ids,cfg,arm,seed,collect=True)
            log(f'evaluated {arm} seed{seed} real DEV '+json.dumps(reports[f'{arm}_seed{seed}'].get('real_dev',{})))
        if initial[0]!=initial[1]:
            raise RuntimeError('A/B initial weights differ')
    with open(data.out/'PER_ROLE.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    comparison={}
    for name in populations:
        perseed=[]
        for seed in cfg['seeds']:
            a=reports[f'A_seed{seed}'][name];b=reports[f'B_seed{seed}'][name]
            perseed.append({'seed':seed,
                            'A_distance_median_px':a['distance_px']['median'],
                            'B_distance_median_px':b['distance_px']['median'],
                            'A_angle_median_deg':a['angle_deg']['median'],
                            'B_angle_median_deg':b['angle_deg']['median'],
                            'distance_relative_reduction':1-b['distance_diagonal']['median']/max(a['distance_diagonal']['median'],1e-12)})
        comparison[name]={'per_seed':perseed,
                          'mean_distance_relative_reduction':float(np.mean([s['distance_relative_reduction'] for s in perseed])),
                          'B_distance_better_seed_count':sum(s['distance_relative_reduction']>0 for s in perseed)}
    results={'status':'COMPLETE','config':cfg,'populations':{k:len(v) for k,v in populations.items()},
             'by_run':reports,'comparison':comparison,
             'interpretation':'Exploratory fixed-budget A/B evidence; real DEV is not an independent final test.',
             'visualization_seed':cfg['seeds'][0],
             'visualization_selection':'sha256 ID order, 2 per real session; no cherry-picking by model output'}
    C.write_json(data.out/'RESULTS.json',results)
    vis=data.out/'visualization_data';vis.mkdir(exist_ok=True)
    samples=[]
    for i in viz_ids:
        rec=data.records[i]; key=hashlib.sha256(rec['id'].encode()).hexdigest()[:12]
        bgr=cv2.imread(rec['image']);pad=cfg['pad_px']
        canvas=cv2.copyMakeBorder(bgr,pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
        image=vis/f'{key}_input.png';cv2.imwrite(str(image),canvas)
        arrays={'gt_points':np.asarray(rec['gt_points'])+pad,
                'gt_valid':np.asarray(rec.get('gt_valid',[True]*8)),
                'supported':art['A'][rec['id']]['supported']}
        for arm in ('A','B'):
            a=art[arm][rec['id']]
            for field in ('attention','angle','distance'):
                arrays[f'{arm}_{field}']=a[field]
            arrays[f'{arm}_lines']=a['lines']+pad
        if rec.get('mask'):
            arrays['foreground_mask']=art['A'][rec['id']]['foreground_mask']
        artifact=vis/f'{key}.npz';np.savez_compressed(artifact,**arrays)
        samples.append({'id':rec['id'],'population':rec['population'],'image':str(image.resolve()),
                        'original_image':rec['image'],'pad_px':pad,
                        'artifact':str(artifact.relative_to(data.out))})
    C.write_json(data.out/'visualization_manifest.json',{'edges':C.EDGES,'samples':samples,'summary':'RESULTS.json'})
    report=['# 합성 전용 Direct Hough attention 지도 A/B 실험','',
            'A: 선 손실만. B: 동일 조건 + 팔레트 foreground attention 손실.',
            '백본 고정, 동일 초기값·배치 순서·학습량. 실제 이미지는 학습·체크포인트 선택에 사용하지 않음.',
            '실사 평가는 기존 DEV이며, 새로운 최종 테스트가 아님. 시각화 seed는 사전 고정한 첫 seed.', '',
            '| 모집단 | seed | A 각도° | B 각도° | A 거리px | B 거리px |',
            '|---|---:|---:|---:|---:|---:|']
    for name,comp in comparison.items():
        for s in comp['per_seed']:
            report.append(f"| {name} | {s['seed']} | {s['A_angle_median_deg']:.2f} | {s['B_angle_median_deg']:.2f} | {s['A_distance_median_px']:.2f} | {s['B_distance_median_px']:.2f} |")
    report+=['','거리 = GT 선분 양 끝점에서 예측 무한직선까지의 평균 수직거리; 표는 role 전체 중앙값.',
             '두 모델의 잘못된 위치·각도를 모두 볼 수 있도록 각도와 거리를 함께 보고한다.',
             'attention은 12개 role별 cross-attention의 head 평균이며, 인과적 중요도를 뜻하지 않는다.',
             '이미지는 입력과 동일하게 반사 패딩 100px을 포함한다. 점선 사각형 안이 원본 영상이다.',
             '마스크 지도는 팔레트 전체를 향하게 하는 약한 지도이며, 각 선에 집중하도록 직접 지도한 실험은 아니다.',
             '합성 사전학습 원본이 삭제되어 백본 사전학습과 평가 합성의 전체 중복 여부는 검증하지 못했다.',
             '', '[attention gallery](attention_gallery.html)']
    (data.out/'REPORT.md').write_text('\n'.join(report)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,default=C.ROOT/'data/pallet/results/hough_attention_transfer_v1')
    ap.add_argument('--phase',choices=['prepare','check','sanity','train','evaluate','all'],default='all')
    ap.add_argument('--steps',type=int,default=3000)
    ap.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
    ap.add_argument('--batch',type=int,default=12)
    ap.add_argument('--attention-lambda',type=float,default=1.)
    ap.add_argument('--backbone',type=Path,default=C.ROOT/'weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth')
    args=ap.parse_args();out=args.run_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=settings(args)
    if (out/'CONFIG.json').exists() and json.loads((out/'CONFIG.json').read_text())!=cfg:
        raise RuntimeError('Config changed: use a separate run directory')
    C.write_json(out/'CONFIG.json',cfg)
    (out/'PURPOSE.md').write_text('# PURPOSE\n\n합성만으로 학습한 Direct Hough에서 팔레트 영역 attention 지도를 추가하면 실사 선 정확도가 개선되는지 확인하고 실제 가중치를 시각화한다.\n\n설정은 CONFIG.json, 데이터 역할은 manifest.json. 실사 결과로 설정이나 체크포인트를 고르지 않는다.\n')
    dh=C.load_dh()
    if args.phase in ('prepare','all'):
        cache_data(out,cfg,dh)
    if args.phase=='prepare': return
    data=Data(out,dh)
    if args.phase in ('check','all'):
        checks(data,cfg)
    if args.phase in ('sanity','all'):
        train_one(data,cfg,'A',101,steps=1500,indices=data.populations['synth_train'][:32],tag='SANITY32')
    if args.phase in ('train','all'):
        for seed in cfg['seeds']:
            for arm in ('A','B'):
                train_one(data,cfg,arm,seed)
    if args.phase in ('evaluate','all'):
        final_evaluation(data,cfg)
        subprocess.run([sys.executable,str(Path(__file__).with_name('visualize.py')),'--run-dir',str(out)],check=True)
        log(f'COMPLETE {out}/attention_gallery.html')


if __name__=='__main__':
    main()
