"""One GT-free shared-backbone pass for ALL 319 positive + 2689 negative images."""
import math,time,copy
import numpy as np
import torch,cv2
import env as E
from b_model import FrozenHeads,decode_raw
from b_cache import cpu
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.export import sample_roi

def main():
    E.gpu();torch.set_num_threads(4);cv2.setNumThreads(1);assert torch.cuda.is_available()
    selection=E.read(E.DOC/'B/selection_lock.json');assert selection['selected_eta']['B3_THREE_AMBIG']>0
    heads=FrozenHeads();features=E.C.old('features');extractor=features.FrozenYoloFeatures(E.R0,device='cuda')
    baseline=E.read(E.C.LINE/'baseline/FULL_CANDIDATES.json');keys=sorted(baseline['frames'])
    perms=E.read(E.DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][0]['permutations']
    start=time.monotonic();records=[];r0max=0.;pmax=0.;counts=dict(R0=0,P=0,DHT=0)
    prior={s:E.read(E.C.BRAW/f'evaluation/P{s}/PREDICTIONS.json')['frames'] for s in [1,2,3]}
    for i,key in enumerate(keys):
        path=E.RAW/f'B/cache/DEV/{i:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():records.append(E.bound(path));continue
        im=cv2.imread(str(E.ROOT/key));assert im is not None
        assert E.sha(E.ROOT/key)==baseline['frame_metadata'][key]['image_sha256']
        captured=extractor.predict(im);counts['R0']+=1
        candidates=captured['candidates'];ref=baseline['frames'][key]
        assert len(candidates)==len(ref),(key,len(candidates),len(ref))
        for c,r in zip(candidates,ref):
            for field in ['score','box_xyxy','keypoints_xy']:
                delta=float(np.max(np.abs(np.array(c[field])-np.array(r[field]))));r0max=max(r0max,delta)
                assert delta==0,(key,field,delta)
        c=dict(frame_id=baseline['frame_metadata'][key]['frame_id'],image_key=key,
            source_image_sha256=baseline['frame_metadata'][key]['image_sha256'],
            candidates=copy.deepcopy(candidates),selected_index=captured['selected_index'],raw_hw=im.shape[:2],
            perms=perms,source='paper_known_C2_task',asset='paper_task_C2',P={})
        if captured['selected_index'] is not None:
            inputs=features.branch_inputs(captured);gain,offset=features.canvas_affine(captured['canvas_shape'],captured['input_shape'])
            chosen=candidates[captured['selected_index']];box_raw=np.array(chosen['box_xyxy'])
            p3,content3=sample_roi(captured['p3'][0].cpu().numpy(),box_raw,gain,offset,8,im.shape[:2])
            p4,content4=sample_roi(captured['p4'][0].cpu().numpy(),box_raw,gain,offset,16,im.shape[:2]);content=content3&content4
            obs=dict(features=(torch.cat([p3,p4],1)*content).cuda(),content=content.cuda())
            logits,valid=heads.observe(obs);counts['DHT']+=1
            def tensor(x,dtype=torch.float32):return torch.as_tensor(x,device='cuda',dtype=dtype)[None]
            c.update(line_logits=logits[0].cpu(),line_valid=valid[0].cpu(),box_raw=torch.tensor(box_raw,dtype=torch.float32),
              base_raw=torch.tensor(chosen['keypoints_xy']),base_points=torch.tensor(chosen['keypoints_xy'],dtype=torch.float32),
              point_valid=torch.tensor(inputs['point_valid']),gain=gain,offset=offset)
            with torch.no_grad():
                for seed,head in heads.points.items():
                    out=head(captured['p3'],captured['p4'],tensor(inputs['points']),tensor(inputs['boxes']),
                      tensor(inputs['point_valid'],torch.bool),tensor(inputs['input_shape']),lam=0);counts['P']+=1
                    scores=heads.score(out,logits[0],valid[0],c['box_raw'].cuda(),c['base_points'].cuda(),
                      c['point_valid'].cuda(),perms,gain,torch.tensor(offset,device='cuda',dtype=torch.float32))
                    T=heads.selection['temperatures'][str(seed)]['temperature']
                    b0,moved=decode_raw(out,scores['biases']['B3_THREE_AMBIG'],T,0,heads.selection['selected_rule'],gain,im.shape[:2],c['base_raw'])
                    # Historical P serialization refined positives only; negatives remain valid new inference.
                    if baseline['frame_metadata'][key]['kind']=='positive':
                        previous=prior[seed][key][captured['selected_index']]['keypoints_xy']
                        difference=float(np.max(abs(b0.numpy()-np.array(previous))));pmax=max(pmax,difference)
                        assert difference*gain<.0003,(key,seed,difference)
                    c['P'][seed]=dict(output=cpu({k:v for k,v in out.items() if k!='coverage'}),B0_raw=b0,**scores)
        torch.save(c,path);records.append(E.bound(path))
        if i%100==0:print('DEV cache',i,len(keys),'seconds',round(time.monotonic()-start,1),flush=True)
    extractor.close()
    E.write(E.DOC/'B/DEV_CACHE_COMPLETE.json',dict(complete=True,frames=len(keys),new_forwards_this_execution=counts,
      max_R0_delta=r0max,max_P_raw_pixel_delta=pmax,detection_candidate_identity=True,
      GT_for_inference=False,metadata_for_history_parity_only=True,files=records,
      elapsed_seconds=time.monotonic()-start,physical_visibility_not_inferred=True,main_training_updates=0))
if __name__=='__main__':main()
