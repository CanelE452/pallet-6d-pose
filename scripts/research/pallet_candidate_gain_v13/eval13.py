"""Evaluate actual candidate selection, regret and candidate-oracle ceiling."""
import argparse
from context13 import *
from scorer13 import GainScorer
E=import_file('v13_score_compat',KIT.parent/'pallet_optional_lines_v11'/'evaluate.py')
COMPARE=import_file('v13_compare_compat',KIT.parent/'pallet_observability_v12'/'eval12.py').compare

def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['complete'];lock=RUN/'EVAL_PROTOCOL.json'
    if lock.exists():raise SystemExit('Refusing to overwrite inference')
    write(lock,dict(source_sha256=sha(Path(__file__)),train_protocol_sha256=sha(RUN/'TRAIN_PROTOCOL.json'),
        policy='Exact same current input observations and detection mask as V12; no GT, labels, image corruption masks or condition identifier supplied to scoring model.',
        oracle='GT candidate winner is computed only after all actual inference files are saved; diagnostic only, excluded from acceptance.',
        readout='argmax learned pixel improvement. Candidate0 exact zero, ties keep. No posthoc threshold selection.'))
    receipts=[]
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            m,O=P.C.load_observations(split,condition,'cuda:0');save_predictions(split,condition,'stock',m,O,O['points'])
            for seed in SEEDS:
                rep=parent_cache(split,condition,seed,'cuda:0');cand=candidates(rep)
                for k,arm in enumerate(('base','half','full')):save_predictions(split,condition,f'{arm}_seed{seed}',m,O,cand[:,:,k])
                for arm in ARMS:
                    ck=RUN/'models'/f'{arm}_seed{seed}'/'final.pt';model=GainScorer(seed).cuda().eval()
                    model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state_dict']);parts=[]
                    with torch.inference_mode():
                        for first in range(0,len(cand),32):
                            sl=slice(first,first+32);out=model(subset(O,sl),subset(rep,sl),cand[sl],arm)
                            parts.append({k:v.cpu() for k,v in out.items()})
                    pred={k:torch.cat([p[k] for p in parts]) for k in parts[0]};points=pred.pop('points');assert torch.isfinite(points).all()
                    pred.update(candidates=cand.cpu(),base=rep['base'].cpu(),full=rep['full'].cpu())
                    save_predictions(split,condition,f'{arm}_seed{seed}',m,O,points,pred)
                    receipts.append(dict(split=split,condition=condition,seed=seed,arm=arm,checkpoint_sha256=sha(ck),frames=len(cand)))
                    del model
            print('INFERRED',split,condition,len(O['points']),flush=True)
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,rows=receipts,targets_used=False))

def scoring():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};comparisons={};diagnostics={}
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            key=f'{split}/{condition}';m=read(V11/'cache'/split/condition/'manifest.json');targets=read(V11/'targets'/f'{split}_{condition}.json')['rows']
            gt=np.array([r['points'] for r in targets]);valid=np.array([r['valid'] for r in targets]);kind=P.target_type(targets).numpy()
            rows={};arrays={};diag={}
            for path in (RUN/'predictions'/split/condition).glob('*.npz'):
                rows[path.stem],arrays[path.stem]=E.score_file(path,m,targets)
                p=np.load(path)
                if 'gain' not in p:continue
                det=p['detected'];errors=np.linalg.norm(p['candidates']-gt[:,:,None],axis=-1);truth=errors[...,:1]-errors
                choice=p['choice'][det].astype(int);chosen=np.take_along_axis(errors[det],choice[...,None],axis=-1)[...,0]
                best=errors[det].min(-1);selectedgain=np.take_along_axis(p['gain'][det],choice[...,None],axis=-1)[...,0]
                change=chosen-errors[det,:,0];d={}
                for name,k in [('all',None),('intact',0),('hidden',1),('outside',2)]:
                    mask=valid[det] if k is None else valid[det]&(kind[det]==k)
                    if not mask.any():continue
                    apply=(choice>0)&mask;false=apply&(change>0)
                    d[name]=dict(points=int(mask.sum()),choice_counts=[int(((choice==j)&mask).sum()) for j in range(3)],
                        candidate_oracle_mean_px=float(best[mask].mean()),selected_mean_px=float(chosen[mask].mean()),
                        base_mean_px=float(errors[det,:,0][mask].mean()),mean_regret_px=float((chosen-best)[mask].mean()),
                        candidate_gain_mae_px=float(np.abs(p['gain'][det,:,1:]-truth[det,:,1:])[mask].mean()),
                        applied_points=int(apply.sum()),harmful_applied_points=int(false.sum()),
                        predicted_positive_but_harmful=int((false&(selectedgain>0)).sum()))
                diag[path.stem]=d
            sessions=[r['session'] for r in targets];comp={}
            for a,b in [('image','base'),('image','full'),('image','geometry'),('image','constant'),('geometry','base'),('image','stock')]:
                c=COMPARE(arrays,rows,a,b,sessions,split)
                c['subgroups']={name:COMPARE(arrays,rows,a,b,sessions,split,mask=valid&(kind==k)) for k,name in enumerate(('intact','hidden','outside'))}
                comp[f'{a}_vs_{b}']=c
            results[key]=rows;comparisons[key]=comp;diagnostics[key]=diag
            print('SCORED',key,{a:[round(rows[f'{a}_seed{s}']['frame_mean_px'],3) for s in SEEDS] for a in ('base','full')+ARMS},flush=True)
    gates={}
    for seed in SEEDS:
        clean=results['real/clean'];mild=results['real/crop20'];g=dict(
            clean_within1=clean[f'image_seed{seed}']['frame_mean_px']<=1.01*clean[f'base_seed{seed}']['frame_mean_px'],
            mild_outside_within1=mild[f'image_seed{seed}']['strata']['outside']['mean_px']<=1.01*mild[f'base_seed{seed}']['strata']['outside']['mean_px'],
            mild_outside_better_full=mild[f'image_seed{seed}']['strata']['outside']['mean_px']<mild[f'full_seed{seed}']['strata']['outside']['mean_px'],
            mild_outside_better_geometry=mild[f'image_seed{seed}']['strata']['outside']['mean_px']<mild[f'geometry_seed{seed}']['strata']['outside']['mean_px'])
        for condition,typ in [('corner20','hidden'),('crop40','outside')]:
            r=results['real/'+condition];b=r[f'base_seed{seed}']['strata'][typ]['mean_px'];full=r[f'full_seed{seed}']['strata'][typ]['mean_px'];actual=r[f'image_seed{seed}']['strata'][typ]['mean_px']
            g[condition+'_retain80']=b-actual>=.8*(b-full)
            g[condition+'_geometry_nonworse']=actual<=r[f'geometry_seed{seed}']['strata'][typ]['mean_px']
        gates[str(seed)]=g
    ci=comparisons['real/crop20']['image_vs_full']['subgroups']['outside']['ci95'][1]<0
    passed=ci and all(all(g.values()) for g in gates.values())
    write(RUN/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,
        mild_vs_full_session_ci_below0=ci,stage1_pass=passed,joint_followup_required=passed,
        scope='Reused synthetic and DEV real artificial stress. Real manual-click GT only. Candidate oracle is privileged diagnostic. No real tuning.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['inference','score'],default='inference');args=p.parse_args()
    inference() if args.phase=='inference' else scoring()
