"""Frozen synthetic calibration/selection followed by heldout reporting."""
import math
import numpy as np
import torch
from common import *
from generic_point_refiner import GenericPointRefiner,decode,targets,finite
from preflight import dataset
from train_point import forward

@torch.no_grad()
def cache_logits():
    assert read(B/'P_TRAINING_COMPLETE.json')['complete'];data=dataset()
    write(B/'GPU_VALIDATION_START.json',gpu())
    bindings={}
    for seed in (1,2,3):
        path=BRAW/f'runs/seed{seed}/last.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        logits_path=BRAW/f'validation_P{seed}.npz'
        if not logits_path.exists():
            head=GenericPointRefiner(**ck['config']).cuda().eval();head.load_state_dict(ck['model_state_dict'])
            logs=[];supports=[]
            for start in range(0,len(data.validation_rows),16):
                rows=data.validation_rows[start:start+16];out=forward(head,data.batch(rows))
                assert torch.isfinite(out['logits']).all()
                logs.append(out['logits'].cpu().numpy());supports.append(out['point_support'].cpu().numpy())
                if start%512==0:print('P_VALIDATION_LOGITS',seed,start,len(data.validation_rows),flush=True)
            with logits_path.open('xb') as f:np.savez(f,rows=data.validation_rows,logits=np.concatenate(logs),support=np.concatenate(supports))
            del head,out;torch.cuda.empty_cache()
        bindings[str(seed)]=dict(checkpoint_sha256=sha(path),logits_sha256=sha(logits_path))
    write(B/'P_VALIDATION_LOGITS.json',dict(complete=True,seeds=bindings,rows=len(data.validation_rows),accuracy_not_read=True))


class Inputs:
    def __init__(self):
        self.data=dataset();self.arrays=self.data.arrays;self.records=self.data.source['records']
        self.logs={s:np.load(BRAW/f'validation_P{s}.npz') for s in (1,2,3)}
        self.rowmap={int(r):i for i,r in enumerate(self.data.validation_rows)}
        self.head=GenericPointRefiner(**read(B/'PARAMETER_BUDGET_LOCK.json')['config'])
        for s,v in self.logs.items():
            assert np.array_equal(v['rows'],self.data.validation_rows)
            assert sha(BRAW/f'validation_P{s}.npz')==read(B/'P_VALIDATION_LOGITS.json')['seeds'][str(s)]['logits_sha256']

    def batch(self,rows,seed):
        a={k:np.array(v[rows]) for k,v in self.arrays.items() if k not in ('p3','p4')}
        t={k:torch.from_numpy(a[k].copy()) for k in ('points','boxes','point_valid','input_shape','gt_points','gt_valid')}
        ix=[self.rowmap[int(r)] for r in rows]
        box=t['boxes'];bv=torch.isfinite(box).all(-1)&(box[:,2:]>box[:,:2]).all(-1)
        safe=torch.where(bv[:,None],box,box.new_tensor([0,0,1,1]));diag=(safe[:,2:]-safe[:,:2]).norm(dim=-1).clamp_min(1)
        out=dict(logits=torch.from_numpy(self.logs[seed]['logits'][ix].copy()),points_raw=t['points'],
            point_valid=finite(t['points'],t['point_valid']),point_support=torch.from_numpy(self.logs[seed]['support'][ix].copy()),
            candidate_displacements=diag[:,None,None]*self.head.displacements[None],box_diagonal=diag)
        records=[self.records[int(self.data.indices[r])] for r in rows]
        return a,t,out,records

    def rows(self,partition):return np.flatnonzero(self.data.partitions==partition)


@torch.no_grad()
def calibration(inputs,seed):
    grid=read(LINE/'TRAIN_PROTOCOL.json')['calibration']['temperature_grid'];totals=np.zeros(len(grid));n=0;corners=0
    rows=inputs.rows('calibration')
    for start in range(0,len(rows),64):
        a,t,o,records=inputs.batch(rows[start:start+64],seed)
        target=targets(o,t['gt_points'],t['gt_valid']);mask=target['support'];count=mask.sum(-1);used=count>0
        n+=int(used.sum());corners+=int(count.sum())
        for i,T in enumerate(grid):
            ce=-(target['distribution'].double()*(o['logits'].double()/T).log_softmax(-1)).sum(-1)
            per=(ce*mask).sum(-1)/count.clamp_min(1);totals[i]+=float(per[used].sum())
    assert n>0
    candidates=[dict(temperature=T,score=float(v/n)) for T,v in zip(grid,totals)]
    return dict(**old('select_synthetic').choose_temperature(candidates),candidates=candidates,supported_frames=n,supported_corners=corners,
                score_definition='mean supported-frame point-candidate CE; same reduction/grid, different geometric target semantics from L')


@torch.no_grad()
def evaluate(inputs,seed,partition,T,rules,method='P'):
    output=[[] for _ in rules];rows=inputs.rows(partition);extras=[[] for _ in rules]
    if method=='L':
        manifest=read(LINE/'LOGITS_MANIFEST.json');entry=next(r for r in manifest['runs'] if r['arm']=='image_line_only' and r['seed']==seed)
        lp=Path(entry['logits']);lp=lp if lp.is_absolute() else LINE/lp
        line_logits=np.load(lp,mmap_mode='r')
    for start in range(0,len(rows),64):
        selected=rows[start:start+64];a,t,o,records=inputs.batch(selected,seed)
        if method=='L':
            ix=[inputs.rowmap[int(r)] for r in selected]
            logs=torch.from_numpy(np.array(line_logits[ix],copy=True))
            state=old('readout').prepare_readout(logs,t['points'],t['boxes'],t['point_valid'],t['input_shape'],T)
        for i,rule in enumerate(rules):
            fraction=rule['max_move_image_diagonal_fraction'];gain=a['gain'].reshape(-1)
            cap=None if fraction is None else fraction*np.array([math.hypot(*r['raw_shape_hw']) for r in records])*gain
            q=(old('readout').apply_readout(state,rule['lam'],cap)['points'] if method=='L' else decode(o,T,rule['lam'],cap)).numpy()
            output[i].extend(old('select_synthetic').frame_metrics(q,a['point_valid'],a['gt_points'],a['gt_valid'],gain,
                records,a['matched'],a['detected'],a['matched_gt_index']))
            for j,r in enumerate(records):
                valid=(a['gt_valid'][j]&a['point_valid'][j]&np.isfinite(q[j]).all(-1)&a['matched'][j])
                errors=np.linalg.norm(q[j]-a['gt_points'][j],axis=-1)/gain[j]
                residual=np.linalg.norm(a['gt_points'][j,:8]-a['points'][j,:8],axis=-1)
                radius=float(o['box_diagonal'][j])*.08
                denominator=sum(sum(p[2]>0 and np.isfinite(p[:2]).all() and p[:2]!=[-1,-1] for p in tar['keypoints_normalized']) for tar in r['targets'])
                extras[i].append(dict(record_index=int(r['index']),errors9_px=errors[valid].tolist(),gt9=denominator,
                    available9=int(valid.sum()),move8_px=(np.linalg.norm(q[j,:8]-a['points'][j,:8],axis=-1)/gain[j]).tolist(),
                    support8=o['point_support'][j].tolist(),residual_in_bank=(residual[valid[:8]]<=radius).tolist(),
                    mean_null_probability=float((o['logits'][j]/T).softmax(-1)[:,-1].mean()) if method=='P' else None))
    return output,extras


def summarize9(rows):
    e=np.array([v for r in rows for v in r['errors9_px']]);moves=np.array([v for r in rows for v in r['move8_px']])
    denominator=sum(r['gt9'] for r in rows);covered=sum(r['available9'] for r in rows)
    bank=[v for r in rows for v in r['residual_in_bank']]
    null=[r['mean_null_probability'] for r in rows if r['mean_null_probability'] is not None]
    return dict(frames=len(rows),pooled9_median_px=float(np.median(e)),pooled9_p90_px=float(np.quantile(e,.9)),
        frame_mean_px=float(np.mean([np.mean(r['errors9_px']) for r in rows if r['errors9_px']])),
        pck10_all_gt=float((e<=10).sum()/denominator),observed9=covered,GT9=denominator,coverage=covered/denominator,
        candidate_target_radius_coverage=float(np.mean(bank)),movement_mean_px=float(moves.mean()),movement_p90_px=float(np.quantile(moves,.9)),
        mean_null_probability=float(np.mean(null)) if null else None)


def select_and_report():
    inputs=Inputs();legacy=old('select_synthetic');protocol=read(LINE/'TRAIN_PROTOCOL.json')
    destination=B/'P_SELECTION.json'
    if not destination.exists():
        temperatures={str(s):calibration(inputs,s) for s in (1,2,3)}
        rules=[dict(lam=l,max_move_image_diagonal_fraction=c) for l in protocol['selection']['lambda_grid'] for c in protocol['selection']['max_move_image_diagonal_fractions']]
        byseed={s:evaluate(inputs,s,'selection',temperatures[str(s)]['temperature'],rules)[0] for s in (1,2,3)}
        candidates=[]
        for i,r in enumerate(rules):
            values=np.array([[row['score'] for row in byseed[s][i]] for s in (1,2,3)])
            candidates.append(dict(**r,score=float(values.mean(0).mean()),per_seed={str(s):legacy.summarize(byseed[s][i]) for s in (1,2,3)}))
        selection=dict(complete=True,no_real_selection=True,selection_population='synth_val',
            timestamp=datetime.now(timezone.utc).isoformat(),temperatures=temperatures,selected_rule=dict(legacy.choose_rule(candidates)),
            candidates=candidates,checkpoints={str(s):sha(BRAW/f'runs/seed{s}/last.pt') for s in (1,2,3)},
            logits_manifest_sha256=sha(B/'P_VALIDATION_LOGITS.json'),source_manifest_sha256=sha(LINE/'SOURCE_MANIFEST.json'),
            selection_code_sha256=sha(Path(__file__)),heldout_accuracy_not_used=True,real_accuracy_not_used=True,
            selection_objective='Exact old all-source-GT-denominator capped diagonal-normalized 8-corner frame mean, then 3-seed mean; exact old tie rules')
        write(destination,selection)
        print('P_SELECTION_FROZEN',selection['selected_rule'],'T',[temperatures[str(s)]['temperature'] for s in (1,2,3)],flush=True)
    selection=read(destination);summaries={};allrows={}
    # No heldout accuracy is read above the persisted selection boundary.
    for seed in (1,2,3):
        rule=selection['selected_rule'];T=selection['temperatures'][str(seed)]['temperature']
        rows,extra=evaluate(inputs,seed,'heldout',T,[dict(lam=0,max_move_image_diagonal_fraction=None),rule])
        for i,name in enumerate(['R0',f'P{seed}']):
            if name=='R0' and seed!=1:continue
            summaries[name]=dict(old8=legacy.summarize(rows[i]),nine=summarize9(extra[i]));allrows[name]=extra[i]
        lr=read(B/'LINE_SOURCE_BINDING.json')['seeds'][str(seed)]
        rows,extra=evaluate(inputs,seed,'heldout',lr['temperature'],[dict(lam=lr['lam'],max_move_image_diagonal_fraction=lr['cap'])],method='L')
        summaries[f'L{seed}']=dict(old8=legacy.summarize(rows[0]),nine=summarize9(extra[0]));allrows[f'L{seed}']=extra[0]
    write(BRAW/'SYNTH_HELDOUT_FRAMES.json',allrows)
    write(B/'SYNTH_HELDOUT_RESULTS.json',dict(complete=True,selection_sha256=sha(destination),summaries=summaries,
        rows_sha256=sha(BRAW/'SYNTH_HELDOUT_FRAMES.json'),architecture_unchanged=True,scope='Branch heldout synthetic; not independent of historical R0/probes'))
    print('SYNTH_HELDOUT_COMPLETE',summaries,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cache','select']);args=p.parse_args()
    cache_logits() if args.phase=='cache' else select_and_report()
