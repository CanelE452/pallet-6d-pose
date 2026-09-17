"""Phase 2: fixed checkpoints -> GT-free logits -> calibration -> heldout."""
import argparse,math
import numpy as np
import torch,cv2
import dcp_env as E
from data import PaperData
from refiner import forward,targets,GenericPointRefiner,decode
from inference import load_head,preservation,serial
from eval_math import measure,summary

@torch.no_grad()
def cache():
    assert E.read(E.DOC/'PAPER_TRAINING_COMPLETE.json')['complete'];E.gpu();data=PaperData()
    records={}
    for arm in E.ARMS:
        for seed in [1,2,3]:
            dst=E.RAW/f'logits/{arm}_seed{seed}.npz';dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():
                head,path=load_head(arm,seed);logs=[];supports=[]
                for start in range(0,len(data.validation_rows),16):
                    rows=data.validation_rows[start:start+16];o=forward(head,data.batch(rows,arm,supervision=False))
                    assert torch.isfinite(o['logits']).all();logs.append(o['logits'].cpu().numpy());supports.append(o['point_support'].cpu().numpy())
                with dst.open('xb') as f:np.savez(f,rows=data.validation_rows,logits=np.concatenate(logs),support=np.concatenate(supports))
                del head,o;torch.cuda.empty_cache();print('VALIDATION_LOGITS',arm,seed,flush=True)
            records[f'{arm}_seed{seed}']=E.bound(dst)
    E.write(E.DOC/'PAPER_LOGITS_COMPLETE.json',dict(complete=True,files=records,GT_for_forward=False,images=15*len(data.validation_rows)))

def light_batch(data,logs,ix,supervision=False):
    keys=['points','boxes','point_valid']+(['gt_points','gt_valid'] if supervision else [])
    rows=logs['rows'][ix];a=data.arrays;t={k:torch.from_numpy(np.array(a[k][rows])) for k in keys}
    box=t['boxes'];valid=torch.isfinite(box).all(-1)&(box[:,2:]>box[:,:2]).all(-1)
    box=torch.where(valid[:,None],box,box.new_tensor([0,0,1,1]));diag=(box[:,2:]-box[:,:2]).norm(dim=-1).clamp_min(1)
    displacements=GenericPointRefiner(**E.read(E.C.B/'PARAMETER_BUDGET_LOCK.json')['config']).displacements
    o=dict(points_raw=t['points'],logits=torch.from_numpy(logs['logits'][ix].copy()),point_support=torch.from_numpy(logs['support'][ix].copy()),
      candidate_displacements=diag[:,None,None]*displacements[None],box_diagonal=diag)
    return rows,t,o

@torch.no_grad()
def calibrate():
    assert E.read(E.DOC/'PAPER_LOGITS_COMPLETE.json')['complete'];data=PaperData();lock=E.read(E.DOC/'TRAIN_PROTOCOL_LOCK.json');out={}
    E.verify(list(E.read(E.DOC/'PAPER_LOGITS_COMPLETE.json')['files'].values()))
    for arm in E.ARMS:
        for seed in [1,2,3]:
            logs=np.load(E.RAW/f'logits/{arm}_seed{seed}.npz');indices=np.flatnonzero(data.partitions[logs['rows']]=='calibration');scores=np.zeros(len(lock['temperature_grid']));n=0
            for start in range(0,len(indices),64):
                rows,t,o=light_batch(data,logs,indices[start:start+64],supervision=True);target=targets(o,t['gt_points'],t['gt_valid']);mask=target['support'];count=mask.sum(-1);used=count>0;n+=int(used.sum())
                for j,T in enumerate(lock['temperature_grid']):
                    ce=-(target['distribution'].double()*(o['logits'].double()/T).log_softmax(-1)).sum(-1)
                    scores[j]+=float(((ce*mask).sum(-1)/count.clamp_min(1))[used].sum())
            candidates=[dict(temperature=T,score=float(s/n)) for T,s in zip(lock['temperature_grid'],scores)]
            chosen=E.old('select_synthetic').choose_temperature(candidates)
            out[f'{arm}_seed{seed}']=dict(**chosen,candidates=candidates,supported_frames=n,checkpoint=E.bound(E.RAW/f'runs/{arm}_seed{seed}/last.pt'))
    E.freeze(E.DOC/'CALIBRATION_AND_SELECTION.json',dict(complete=True,temperatures=out,rule=lock['selected_rule'],new_lambda_cap_sweep=False,
      objective=lock['calibration_target'],real_access=False,heldout_performance_access_before_lock=False))
    print('CALIBRATION_LOCKED',flush=True)

@torch.no_grad()
def baseline():
    """Serialize full R0 candidates, verify source cache adapter on ALL heldout."""
    assert E.read(E.DOC/'CALIBRATION_AND_SELECTION.json')['complete'];E.gpu();data=PaperData();fx=E.old('features');extractor=fx.FrozenYoloFeatures(E.R0)
    files=[];count=0
    for row in np.flatnonzero(data.partitions=='heldout'):
        r=data.source['records'][data.indices[row]];dst=E.RAW/f'source_baseline/{row:05d}.json'
        if not dst.exists():
            im=cv2.imread(r['image']);assert im is not None and E.sha(r['image'])==r['image_sha256']
            cap=extractor.predict(im,already_padded=True);inp=fx.branch_inputs(cap);selected=cap['selected_index'];a=data.arrays
            assert (-1 if selected is None else selected)==a['selected_candidate_index'][row]
            if inp is not None:
                for k in ['points','boxes','point_valid','input_shape']:assert np.array_equal(np.array(inp[k]),a[k][row]),(r['id'],k)
                for k in ['p3','p4']:assert np.array_equal(cap[k][0].cpu().numpy(),a[k][row]),(r['id'],k)
            candidates=serial(cap['candidates'])
            for c in candidates:
                c['keypoints_xy']=(np.array(c['keypoints_xy'])-100).tolist();c['box_xyxy']=(np.array(c['box_xyxy'])-100).tolist()
            E.write(dst,dict(id=r['id'],candidates=candidates,selected_index=selected,cache_adapter_exact=True,GT_input=False))
            count+=1
        files.append(E.bound(dst))
        if len(files)%250==0:print('SOURCE_R0_CONTRACT',len(files),flush=True)
    extractor.close();E.write(E.DOC/'SYNTH_DETECTION_AUDIT.json',dict(complete=True,frames=len(files),new_R0_forwards_this_execution=count,files=files,cache_feature_and_geometry_exact=True))

@torch.no_grad()
def evaluate():
    from point_inference import replace_selected
    data=PaperData();sel=E.read(E.DOC/'CALIBRATION_AND_SELECTION.json');assert E.read(E.DOC/'SYNTH_DETECTION_AUDIT.json')['complete']
    oldsel=E.read(E.C.B/'P_SELECTION.json');allrows={};summaries={};predfiles={};detection=0
    assets={r['frame_id']:r['source_asset'] for r in E.read(E.RAW/'DIMENSION_SIDECAR.json')['records']}
    oldlogs=E.read(E.C.B/'P_VALIDATION_LOGITS.json')
    for seed in [1,2,3]:assert E.sha(E.C.BRAW/f'validation_P{seed}.npz')==oldlogs['seeds'][str(seed)]['logits_sha256']
    for arm in ['OLD_P',*E.ARMS]:
        for seed in [1,2,3]:
            name=f'{arm}_seed{seed}';logs=np.load(E.C.BRAW/f'validation_P{seed}.npz' if arm=='OLD_P' else E.RAW/f'logits/{name}.npz')
            assert np.array_equal(logs['rows'],data.validation_rows)
            T=oldsel['temperatures'][str(seed)]['temperature'] if arm=='OLD_P' else sel['temperatures'][name]['temperature'];rule=sel['rule'];indices=np.flatnonzero(data.partitions[logs['rows']]=='heldout')
            preds=[];scored=[]
            for start in range(0,len(indices),64):
                rows,t,o=light_batch(data,logs,indices[start:start+64]);records=[data.source['records'][data.indices[row]] for row in rows];gain=data.arrays['gain'][rows]
                cap=None if rule['max_move_image_diagonal_fraction'] is None else rule['max_move_image_diagonal_fraction']*np.array([math.hypot(*r['raw_shape_hw']) for r in records])*gain
                refined=decode(o,T,rule['lam'],cap).numpy()
                for j,(row,r) in enumerate(zip(rows,records)):
                    before=E.read(E.RAW/f'source_baseline/{row:05d}.json');idx=before['selected_index']
                    after=replace_selected(before['candidates'],idx,data.arrays['points'][row],refined[j],gain[j],rule['lam']);preservation(before['candidates'],after,idx);detection+=1
                    preds.append(dict(id=r['id'],candidates=serial(after),selected_index=idx))
            # Persist all predictions before opening evaluation targets below.
            predpath=E.RAW/f'predictions/SYNTH_HELDOUT/{name}.json';E.write(predpath,dict(records=preds,GT_input=False,complete=True));predfiles[name]=E.bound(predpath)
            for row,p in zip(logs['rows'][indices],preds):
                r=data.source['records'][data.indices[row]];assert len(r['targets'])==1
                kp=np.array(r['targets'][0]['keypoints_normalized']);gt=kp[:,:2]*np.array(r['prepared_shape_hw'])[::-1]-100
                valid=kp[:,2]>0;idx=p['selected_index'];matched=bool(data.arrays['matched'][row]);point=np.full((9,2),np.nan) if idx is None else p['candidates'][idx]['keypoints_xy']
                order=int(data.side['order'][row]);perms=data.side['permutations'][row,:order]
                m=measure(point,gt,valid,perms,r['raw_shape_hw'],matched,idx is not None)
                d=data.side['dimensions'][row];ratio=max(d[0]/d[1],d[1]/d[0])
                m.update(id=r['id'],session=r['scenario_id'],group=f'C{order}',object=assets[r['id']],source=r['source'],ratio_bin='<=1.05' if ratio<=1.05 else '<=1.2' if ratio<=1.2 else '>1.2');scored.append(m)
            allrows[name]=scored;summaries[name]=summary(scored);print('SYNTH_RESULT',name,summaries[name],flush=True)
    E.write(E.RAW/'SYNTH_HELDOUT_METRICS.json',allrows)
    E.write(E.DOC/'SYNTH_HELDOUT_RESULTS.json',dict(complete=True,summary=summaries,predictions=predfiles,detection_contract_checked=detection,GT_single_branch=True,matched_metrics_separate_from_full_denominator=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cache','calibrate','baseline','evaluate']);args=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    globals()[args.phase]()
