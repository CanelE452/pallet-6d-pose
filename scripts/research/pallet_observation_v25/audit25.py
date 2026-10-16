"""Independent coordinate, metric, exposure, and integrated-inference checks."""
from context25 import *
from prepare25 import ARMS,new_load25
from eval25 import SPLITS,conditions,population,inputs
from model25 import Correction25
from network18 import VectorPointLineNetwork
from network25 import ObservationCorrectionNetwork

def main():
    setup();s=read(RUN/'SUMMARY.json');protocol=read(RUN/'TRAIN_PROTOCOL.json');assert sha(RUN/'PROTOCOL.json')==protocol['protocol_sha256']
    for mapping in (protocol['sources'],protocol['frozen_sources'],protocol['new_parents'],read(RUN/'EVALUATION_BINDINGS.json')['files']):
        for p,h in mapping.items():assert sha(ROOT/p)==h,p
    assert sha(KIT/'eval25.py')==read(RUN/'EVAL_PROTOCOL.json')['source_sha256']
    updates=0;exposures=[]
    for seed in SEEDS:
        g=read(RUN/f'GROUPS_seed{seed}.json');n=g['original_count'];seen={k:{i for row in plan for i in row} for k,plan in g['plans'].items()};assert set(range(n)).issubset(seen['original']) and set(range(len(g['ids']))).issubset(seen['paired'])
        init=[]
        for arm in ARMS:
            d=RUN/'models'/f'{arm}_seed{seed}';start=read(d/'START.json');init.append(start['initial_sha256']);assert start['plan_sha256']==g['plan_sha256']['original' if arm=='gate_original' else 'paired'];rows=[json.loads(l) for l in (d/'TRACE.jsonl').read_text().splitlines()];assert len(rows)==2000
            assert all(r['step']==i+1 and all(np.isfinite(r[k]) for k in ('loss','point_error','gradient_norm')) for i,r in enumerate(rows));assert sha(d/'final.pt')==read(d/'COMPLETE.json')['sha256'];updates+=len(rows)
        assert len(set(init))==1;exposures.append(dict(seed=seed,**g['exposures']))
    assert updates==42000
    pops=read(PREV.RUN/'POPULATIONS.json');groups=[{r['pair_group_id'] for r in pops[k]} for k in ('new_train','new_check')];assert not groups[0]&groups[1]
    raw=read(PREV.RUN/'RAW_TARGETS.json')['rows'];gtmax=0.;rowschecked=0
    for split in ('new_train','new_check'):
        m,O=new_load25(split);ts=read(RUN/'targets'/f'{split}_opposite20.json')['rows'];assert [r['id'] for r in m['rows']]==[t['id'] for t in ts]
        for name,h in m['chunk_sha256'].items():assert sha(RUN/'cache'/split/'opposite20'/name)==h
        old={r['id']:r for r in read(PREV.RUN/'cache'/split/'crop20/manifest.json')['rows']}
        for r,t in zip(m['rows'],ts):
            source=raw[r['id']];assert sha(ROOT/source['label'])==source['label_sha256'];values=np.fromstring((ROOT/source['label']).read_text(),sep=' ')[5:].reshape(9,3)[:8];h,w=r['original_hw'];gt=values[:,:2]*[w+200,h+200]-100-r['origin'];err=float(np.abs(gt-t['points']).max());gtmax=max(gtmax,err);assert err<1e-9;assert (values[:,2]>0).tolist()==t['valid'];assert old[r['id']]['crop_side']!=r['crop_side'];rowschecked+=1
            if r['detected']:
                ii=r['observation_index'];K=np.asarray(r['K']).copy();K[:2,2]-=r['origin'];assert np.allclose(O['K'][ii],K,atol=1e-4);assert O['image_hw'][ii].tolist()==r['image_hw']
    metricmax=0.;predictionrows=0
    for split in SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);det=np.array([r['detected'] for r in m['rows']])
            if split=='real':
                for r,t in zip(m['rows'],ts):
                    assert sha(ROOT/r['label'])==r['label_sha256'];ann=read(ROOT/r['label'])['objects'][0]['keypoint_annotations'][:8];g=np.array([a['xy'] if a.get('xy') is not None else [0,0] for a in ann]);assert np.allclose(g-r['origin'],t['points'],atol=1e-9);assert [a.get('source')=='manual_click' and a.get('xy') is not None for a in ann]==t['valid']
            for name,row in s['results'][split+'/'+c].items():
                p=np.load(RUN/'predictions'/split/c/f'{name}.npz');assert p['ids'].tolist()==[t['id'] for t in ts];assert np.array_equal(p['detected'],det);points=p['points'].astype(float);assert np.isfinite(points[det]).all();error=np.linalg.norm(points-gt,axis=-1);per=np.where(valid[det],error[det],0).sum(-1)/valid[det].sum(-1);diff=abs(float(per.mean())-row['frame_mean_px']);metricmax=max(metricmax,diff);assert diff<1e-9
                diag=np.linalg.norm(np.array([r['original_hw'] for r in m['rows']]),axis=-1);norm=np.ones(len(det));norm[det]=np.where(valid[det],np.minimum(error[det]/diag[det,None],1),0).sum(-1)/valid[det].sum(-1);assert abs(norm.mean()-row['full_normalized_mean'])<1e-9
                for typ in ('hidden','outside'):
                    mask=valid&det[:,None]&np.array([t[typ] for t in ts])
                    if mask.any():assert abs(error[mask].mean()-row['strata'][typ]['mean_px'])<1e-9
                assert np.allclose(p['base'][det]+p['applied_delta'][det],p['points'][det],atol=.001,rtol=0);predictionrows+=len(det)
    parity=[];maximum=0.;extra_effect=[]
    for seed in SEEDS:
        parent=VectorPointLineNetwork(seed,'line_vector').eval().requires_grad_(False)
        for c in ('clean','crop20','crop40'):
            m,O,cached=inputs('real',c,seed);O={k:v[:4].cuda() for k,v in O.items()};cached={k:v[:4].cuda() for k,v in cached.items()}
            with torch.inference_mode():
                p=parent.parent({k:v for k,v in O.items() if k!='edge'});r=dict(p['rep'],lines=p['lines'],line_weights=p['line_weights']);proposal=parent.evaluator(O,r,'line_vector')['points']-r['base'];r['proposal_delta']=proposal
                diff=max(float((r[k]-cached[k]).abs().max()) for k in cached);assert diff<.001
                for arm in ARMS:
                    model=Correction25(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);out=model(O,r,arm);saved=np.load(RUN/'predictions/real'/c/f'{arm}_seed{seed}.npz');want=saved['points'][saved['detected']][:4];err=float(np.abs(out['points'].cpu().numpy()-want).max());maximum=max(maximum,err);assert err<.001
                    if arm in ('vector_line','vector_line_edge'):
                        local=model(O,r,arm.replace('line','local'));off=model(O,r,arm,evidence_off=True);assert torch.equal(local['points'],off['points']);flip=model(O,dict(r,lines=-r['lines']),arm);assert torch.allclose(flip['points'],out['points'],atol=.001,rtol=0);extra_effect.append(dict(seed=seed,condition=c,arm=arm,max_new_line_effect_px=float((local['points']-out['points']).abs().max())))
                    if c=='crop20':
                        integrated=ObservationCorrectionNetwork(seed,arm);whole=integrated(O);joined=float((whole['points']-out['points']).abs().max());assert joined<.001;del integrated
                    parity.append(dict(seed=seed,condition=c,arm=arm,max_error_px=err));del model
        del parent
    write(RUN/'INTEGRATION_CHECK.json',dict(status='PASS',rows=parity,max_error_px=maximum,extra_line_effect=extra_effect,no_GT_forward=True,scope='Full frozen representation and final correction replay from cached YOLO features. YOLO image extraction separately performed for new paired observations.'))
    write(RUN/'AUDIT.json',dict(status='PASS',finite_updates=updates,models=21,original_group_split_disjoint=True,all_original_detected_training_observations_retained=True,all_paired_detected_training_observations_exposed=True,batch_exposures=exposures,paired_GT_rows=rowschecked,GT_coordinate_max_px=gtmax,manual_GT_unchanged=True,metric_max_error=metricmax,prediction_frame_rows=predictionrows,integration_max_px=maximum,frozen_sources_unchanged=True))
    record('독립 감사 통과:42,000개 유한 업데이트/21모델, 원문 수동GT·좌표 이동·분모 포함 지표 재계산·세 시드 통합 경로 재생 및 선 off 대조. AUDIT.json/INTEGRATION_CHECK.json 기록.');print('AUDIT25 PASS',updates,maximum,flush=True)

if __name__=='__main__':main()
