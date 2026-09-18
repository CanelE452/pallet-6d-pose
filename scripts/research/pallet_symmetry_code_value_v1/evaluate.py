"""Fixed-T inference serialized before target access; same evaluator for every code."""
import argparse,math
import numpy as np
import torch
import cv_env as E
from code_adapter import model,alter,mode_for,PaperData,SquareData,forward,decode,context
from inference import predict_captured,preservation,serial
from point_inference import replace_selected
from eval_math import measure,summary
from dev_evaluate import population_metadata,iou

def name(arm,seed,mode='PRIMARY'):return f'{arm}_seed{seed}_{mode}'
def path(pop,arm,seed,mode='PRIMARY',kind='predictions'):
    return E.RAW/kind/pop/(name(arm,seed,mode)+('.npz' if kind=='logits' else '.json'))
def evaluation_bindings():
    return [E.bound(E.HERE/f) for f in ['evaluate.py','code_statistics.py','inference_manifest.py']]+[E.bound(E.DOC/f) for f in ['PROTOCOL_LOCK.json','CODE_DONOR_LOCK.json','INFERENCE_METADATA_ONLY.json']]
def head_for(arm,seed):
    record=E.read(E.DOC/f'fits/{arm}_seed{seed}.json');E.verify([record['checkpoint']])
    ck=torch.load(E.ROOT/record['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert ck['complete'] and ck['step']==6000
    head=model().cuda().eval();head.load_state_dict(ck['state']);head.requires_grad_(False);return head,record['checkpoint']

class Population:
    def __init__(self,pop):
        self.pop=pop;self.norm=E.read(E.D.DOC/'DIM_NORMALIZATION_LOCK.json')
        if pop=='DEV':
            self.records=[torch.load(E.ROOT/r['path'],map_location='cpu',weights_only=False) for r in E.read(E.D.DOC/'DEV_CACHE_COMPLETE.json')['records']]
            self.ids=[r['id'] for r in self.records];self.groups=[r['order'] for r in self.records]
        else:
            self.data=InferenceData(pop)
            self.rows=self.data.rows
            self.records=self.data.records
            self.ids=[r['id'] for r in self.records]
            self.groups=self.data.orders[self.rows].tolist() if pop=='SYNTH' else [4]*len(self.rows)
            self.baselines=[E.read(E.D.RAW/(f'source_baseline/{i:05d}.json' if pop=='SYNTH' else f'square_cache/{i:04d}.json')) for i in self.rows]
        self.donors=E.read(E.DOC/'CODE_DONOR_LOCK.json')['populations'][pop]
    def codes(self,z,mode,start,end):
        donor=None
        if mode=='SHUFFLED':
            records=self.donors['records'][start:end];assert [r['recipient'] for r in records]==self.ids[start:end]
            donor=[r['donor_group'] for r in records]
        return alter(z,mode,donor)
    @torch.no_grad()
    def infer(self,arm,seed,mode='PRIMARY'):
        dst=path(self.pop,arm,seed,mode);lp=path(self.pop,arm,seed,mode,'logits')
        if dst.exists() and lp.exists():
            saved=E.read(dst);assert saved['evaluation_bindings']==evaluation_bindings();E.verify([saved['checkpoint'],saved['logits']]);return
        actual=mode_for(arm) if mode=='PRIMARY' else mode;head,ck=head_for(arm,seed);preds=[];logs=[];supports=[];activations=[]
        rule=E.protocol()['rule'];E.gpu()
        if self.pop=='DEV':
            for j,r in enumerate(self.records):
                cap=dict(r['captured']);cap.update({k:cap[k].cuda() for k in ['p3','p4']})
                z=torch.from_numpy(context(np.array(r['dimensions'])[None],[r['order']],self.norm,True)).cuda();z=self.codes(z,actual,j,j+1)
                p,d=predict_captured(head,'N4_META_SYM',cap,r['dimensions'],r['order'],1.,rule,r['raw_hw'],self.norm,z.cpu().numpy())
                p.update(id=r['id'],key=r['key']);preds.append(p)
                logs.append(d['logits'] if d else np.zeros((8,222),np.float32));supports.append(d['support'] if d else np.zeros(8,bool))
                activations.append(head.metadata_encoder[0](z).cpu().numpy()[0])
        else:
            for start in range(0,len(self.rows),16):
                rows=self.rows[start:start+16];b=self.data.batch(rows,'N4_META_SYM',supervision=False)
                b['context']=self.codes(b['context'],actual,start,start+len(rows));o=forward(head,b)
                hw=[r['raw_shape_hw'] if self.pop=='SYNTH' else r['raw_hw'] for r in self.records[start:start+len(rows)]]
                gain=self.data.arrays['gain'][rows];cap=rule['max_move_image_diagonal_fraction']*np.array([math.hypot(*x) for x in hw])*gain
                q=decode(o,1.,rule['lam'],cap).cpu().numpy();logs.extend(o['logits'].cpu().numpy());supports.extend(o['point_support'].cpu().numpy())
                activations.extend(head.metadata_encoder[0](b['context']).cpu().numpy())
                for j,i in enumerate(rows):
                    base=self.baselines[start+j];idx=base['selected_index']
                    after=replace_selected(base['candidates'],idx,self.data.arrays['points'][i],q[j],gain[j],rule['lam']);preservation(base['candidates'],after,idx)
                    preds.append(dict(id=base['id'],selected_index=idx,candidates=serial(after)))
        assert [p['id'] for p in preds]==self.ids
        lp.parent.mkdir(parents=True,exist_ok=True)
        np.savez(lp,logits=np.array(logs),support=np.array(supports),activation=np.array(activations),ids=np.array(self.ids))
        assert np.isfinite(np.array(logs)).all()
        E.write(dst,dict(complete=True,records=preds,GT_input=False,GT_arrays_read=False,detector_contract_exact=True,T=1.,rule=rule,mode=actual,checkpoint=ck,logits=E.bound(lp),evaluation_bindings=evaluation_bindings()))
        del head;torch.cuda.empty_cache();print('PREDICTIONS_SERIALIZED',self.pop,name(arm,seed,mode),flush=True)
    def score(self,arm,seed,mode='PRIMARY'):
        dst=path(self.pop,arm,seed,mode,kind='metrics')
        if dst.exists():return E.read(dst)
        pred=E.read(path(self.pop,arm,seed,mode));assert pred['complete'] and not pred['GT_input'];rows=[]
        if self.pop=='DEV':
            pe,pop=population_metadata();targets={}
            for item,m in pop:
                t=pe.E._legacy_forbidden_target(item);targets[item.frame_id]=(t,m)
            groups={r['object_type']:r for r in E.read(E.D.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
        elif self.pop=='SYNTH':
            perms=np.load(E.D.RAW/'DIMENSION_SIDECAR.npz')['permutations'];assets={r['frame_id']:r['source_asset'] for r in E.read(E.D.RAW/'DIMENSION_SIDECAR.json')['records']}
            target_records={r['id']:r for r in E.read(E.D.LINE/'SOURCE_MANIFEST.json')['records'] if r['partition']=='heldout'}
        else:
            from square_data import membership
            target_records={r['id']:r for r in membership() if r['split']=='val'}
            perms=np.array(E.read(E.D.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][-1]['permutations'])
        matched_array=None if self.pop=='DEV' else np.load(self.data.cache/'matched.npy',mmap_mode='r')
        for j,(p,r) in enumerate(zip(pred['records'],self.records)):
            assert p['id']==r['id'];idx=p['selected_index'];c=None if idx is None else p['candidates'][idx];point=np.full((9,2),np.nan) if c is None else c['keypoints_xy'];g=self.groups[j]
            if self.pop=='DEV':
                t,m=targets[p['id']];gt=t.keypoints_xy;valid=t.keypoint_supervision_mask;ps=groups[r['object_type']]['permutations'];hw=r['raw_hw'];matched=c is not None and iou(c['box_xyxy'],t.box_xyxy)>=.5
                session=m['session_id'];obj=r['object_type']
            elif self.pop=='SYNTH':
                a=np.array(target_records[r['id']]['targets'][0]['keypoints_normalized']);gt=a[:,:2]*np.array(r['prepared_shape_hw'])[::-1]-100;valid=a[:,2]>0
                ps=perms[self.rows[j],:g];hw=r['raw_shape_hw'];matched=bool(matched_array[self.rows[j]]);session=r['scenario_id'];obj=assets[r['id']]
            else:
                a=np.array(target_records[r['id']]['target'])[0,5:].reshape(9,3);gt=a[:,:2]*(np.array(r['raw_hw'])[::-1]+200)-100;valid=a[:,2]>0;ps=perms;hw=r['raw_hw'];matched=bool(matched_array[self.rows[j]])
                session=r['id'].split('__')[0];obj='plastic_standard_110x110x15'
            m=measure(point,gt,valid,ps,hw,matched,idx is not None);m.update(id=p['id'],session=session,group=f'C{g}',object=obj);rows.append(m)
        E.write(dst,rows);return rows

class InferenceData:
    """Never opens GT arrays or computes supervised train-row masks."""
    def __init__(self,pop):
        self.pop=pop;self.norm=E.read(E.D.DOC/'DIM_NORMALIZATION_LOCK.json')
        self.cache=E.D.LINE/'cache' if pop=='SYNTH' else E.D.RAW/'square_cache'
        self.arrays={k:np.load(self.cache/(k+'.npy'),mmap_mode='r') for k in ['p3','p4','points','boxes','point_valid','input_shape','gain']}
        manifest=E.read(E.DOC/'INFERENCE_METADATA_ONLY.json')['populations'][pop]
        self.rows=np.array(manifest['rows']);self.records=manifest['records']
        if pop=='SYNTH':
            cache_manifest=E.read(self.cache/'CACHE_MANIFEST.json');indices=np.asarray(cache_manifest['record_indices'])
            with np.load(E.D.RAW/'DIMENSION_SIDECAR.npz') as side:
                assert np.array_equal(side['record_index'],indices);self.dimensions=side['dimensions'];self.orders=side['order']
        else:
            self.dimensions=np.tile([1.1,1.1,.15],(851,1));self.orders=np.full(851,4)
    def batch(self,rows,arm='N4_META_SYM',device='cuda',supervision=False):
        assert not supervision and arm=='N4_META_SYM'
        b={k:torch.from_numpy(np.array(self.arrays[k][rows])).to(device) for k in ['p3','p4','points','boxes','point_valid','input_shape']}
        b['context']=torch.from_numpy(context(self.dimensions[rows],self.orders[rows],self.norm,True)).to(device);return b

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['A','B','C']);args=p.parse_args();torch.set_num_threads(4)
    E.freeze(E.DOC/'EVALUATION_CODE_LOCK.json',dict(files=evaluation_bindings(),T=1.,GT_before_prediction=False))
    tracks=['A','B'] if args.stage=='C' else [args.stage]
    for track in tracks:
        assert E.read(E.DOC/f'{track}_TRAINING_COMPLETE.json')['complete']
        for pop in E.POPS[track]:
            data=Population(pop);arms=[E.ARMS[track][1]] if args.stage=='C' else E.ARMS[track]
            modes=['NEUTRAL','ZERO','WRONG']+(['SHUFFLED'] if data.donors['applicable'] else []) if args.stage=='C' else ['PRIMARY']
            # All predictions in this population are serialized before opening evaluation targets.
            for arm in arms:
                for seed in [1,2,3]:
                    for mode in modes:data.infer(arm,seed,mode)
            for arm in arms:
                for seed in [1,2,3]:
                    for mode in modes:data.score(arm,seed,mode)
            del data
        if args.stage!='C':
            from code_statistics import track_report
            track_report(track)
    if args.stage=='C':
        from code_statistics import perturbations
        perturbations()
if __name__=='__main__':main()
