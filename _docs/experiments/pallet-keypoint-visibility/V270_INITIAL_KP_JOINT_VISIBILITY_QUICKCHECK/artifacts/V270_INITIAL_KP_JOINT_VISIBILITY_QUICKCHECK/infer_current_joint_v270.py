"""Whole-frame223 initial-model inference. Never reads human GT or query sidecar."""
import os,sys,json,types,time,hashlib,shutil
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from PIL import Image
W=Path(__file__).parent
p=W/'model_initial_joint_v270.py'
for a in [p,*p.parents]:assert not a.is_symlink()
b=p.read_bytes();m=types.ModuleType('v270_model');m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(b,str(p),'exec'),m.__dict__)
INPUT=W.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK/current_fullframe_inputs_v2/FRAMES223_INPUT_ONLY.jsonl'
def main():
    assert not (W/'CURRENT_PREDICTIONS_FROZEN.json').exists()
    configpath=W/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json';config=json.loads(m.safe(configpath).read_text());assert config['PASS'];m.verify(config['model_code'])
    for value in config['checkpoints'].values():m.verify(value)
    identity=W/'INPUT_CHANNEL_NATIVE_IDENTITY_BEFORE_GT.json';assert json.loads(m.safe(identity).read_text())['PASS']
    weights=m.safe(W/'weights');weights.mkdir(exist_ok=True);alias=weights/'BASE_R0.pt'
    if not alias.exists():shutil.copyfile(m.safe(m.R0),alias)
    assert m.binding(alias)['sha256']==m.R0_SHA
    # Relocation changes only the path: exact original checkpoint bytes remain mandatory.
    m.R0=alias
    rows=[json.loads(x) for x in m.safe(INPUT).read_text().splitlines()];assert len(rows)==223
    base=m.InitialKPJoint(device='cuda');base.eval();models={}
    for arm in m.ARMS:
        model=m.InitialKPJoint(arm,device='cuda');cp=torch.load(m.verify(config['checkpoints'][arm]),map_location='cpu',weights_only=False);assert cp['steps']==300;model.load_head_state(cp['head_state']);model.eval();models[arm]=model
    result=dict(frame_index=np.arange(223,dtype=np.int32),frame_available=np.zeros(223,bool),anchor_index=np.zeros(223,np.int64),candidate_score=np.zeros(223,np.float32),raw_to_input_affine=np.zeros((223,2,3),np.float32),R0_kobj_probability=np.zeros((223,9),np.float32))
    result.update(R0_xy_original=np.zeros((223,9,2),np.float32),R0_xy_input=np.zeros((223,9,2),np.float32),R0_canvas_logit=np.zeros((223,9),np.float32),R0_raw27=np.zeros((223,27),np.float32))
    for arm in m.ARMS:
        result[arm+'_xy_original']=np.zeros((223,9,2),np.float32);result[arm+'_xy_input']=np.zeros((223,9,2),np.float32);result[arm+'_canvas_logit']=np.zeros((223,9),np.float32);result[arm+'_raw27']=np.zeros((223,27),np.float32)
        if arm!='XY_ONLY':result[arm+'_E_probability']=np.zeros((223,9),np.float32);result[arm+'_E_logit']=np.zeros((223,9),np.float32)
    fixtureids=[0,56,112,168,222];fixture=dict(frame_index=np.array(fixtureids,np.int32));fixturelists={};start=time.time();before=base.frozen_sha()
    for i,row in enumerate(rows):
        assert row['frame_index']==i;m.verify(row['image_binding'])
        rgb=np.asarray(Image.open(m.safe(row['image_binding']['path'])).convert('RGB'));assert list(rgb.shape[:2])==row['raw_hw'];assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['decoded_RGB_sha256']
        image,A=m.prepare_rgb(rgb);result['raw_to_input_affine'][i]=A
        with torch.no_grad():
            features=base.extract_pyramid(image[None].cuda());original=base.forward_pyramid(features);score=original['det']['scores'].sigmoid().amax(1);anchor=score.argmax(1);conf=float(score[0,anchor[0]])
            result['anchor_index'][i]=int(anchor[0]);result['candidate_score'][i]=conf;result['frame_available'][i]=conf>=.001
            old=m.selected(original,anchor);result['R0_kobj_probability'][i]=old['canvas_logit'].sigmoid().cpu().numpy()[0]
            oldxy=old['xy_px'].cpu().numpy()[0];result['R0_xy_input'][i]=oldxy;result['R0_xy_original'][i]=(oldxy-A[:,2])/np.diag(A[:,:2]);result['R0_canvas_logit'][i]=old['canvas_logit'].cpu().numpy()[0];result['R0_raw27'][i]=original['raw9'][0,anchor[0]].cpu().numpy().reshape(27)
            if i in fixtureids:
                for k,f in enumerate(features):fixturelists.setdefault(f'pyramid_P{k+3}',[]).append(f.cpu().numpy()[0])
            for arm,model in models.items():
                packet=model.forward_pyramid(features,with_det=False);selected=m.selected(packet,anchor);xy=selected['xy_px'].cpu().numpy()[0]
                result[arm+'_xy_input'][i]=xy;result[arm+'_xy_original'][i]=(xy-A[:,2])/np.diag(A[:,:2]);result[arm+'_canvas_logit'][i]=selected['canvas_logit'].cpu().numpy()[0]
                raw=packet['raw9'][0,anchor[0]].cpu().numpy().reshape(27);result[arm+'_raw27'][i]=raw
                if arm!='XY_ONLY':result[arm+'_E_logit'][i]=selected['E_logit'].cpu().numpy()[0];result[arm+'_E_probability'][i]=selected['E_logit'].sigmoid().cpu().numpy()[0]
                if i in fixtureids:
                    fixturelists.setdefault(arm+'_raw27',[]).append(raw)
                    fixturelists.setdefault(arm+'_E_logit',[]).append(selected['E_logit'].cpu().numpy()[0])
        if i%40==0:print(f'CURRENT {i+1}/223 elapsed {time.time()-start:.1f}s',flush=True)
    assert base.frozen_sha()==before
    for key,values in fixturelists.items():fixture[key]=np.stack(values)
    for key in ('anchor_index','candidate_score','raw_to_input_affine'):fixture[key]=result[key][fixtureids]
    out=W/'CURRENT_RAW223_PREDICTIONS.npz';np.savez_compressed(m.safe(out),**result)
    replay=m.safe(W/'fullframe_replay_v1');replay.mkdir();np.savez_compressed(replay/'FORWARD_INPUTS5.npz',**fixture)
    # Every full-frame and raw9 prediction is sealed before any query sidecar or GT join.
    m.dump(W/'CURRENT_PREDICTIONS_FROZEN.json',dict(PASS=True,completed=True,predictions_npz=m.binding(out),configuration=m.binding(configpath),input_channel_native_identity=m.binding(identity),frame_inputs=m.binding(INPUT),forward_replay_inputs=m.binding(replay/'FORWARD_INPUTS5.npz'),R0=m.binding(alias),code=m.binding(Path(__file__)),model_code=m.binding(p),frame_count=223,raw_slot_count=9,semantic_slot_count=8,available_frames=int(result['frame_available'].sum()),semantic_threshold=.5,candidate_threshold=.001,frame_or_candidate_only_abstention=True,old45queryU_forcing=False,query_sidecar_reads=0,human_GT_reads=0,query_neural_inputs=0,frame_metadata_note='Frame manifest is read for RGB bindings/order/rawHW only; its old query fields remain unused. Query sidecar is never opened.',elapsed_seconds=time.time()-start))
    print('FROZEN223',int(result['frame_available'].sum()),flush=True)
if __name__=='__main__':main()
