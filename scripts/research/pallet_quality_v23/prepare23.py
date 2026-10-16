"""Select source-group-disjoint YOLO-val images, rerun all frozen parents, then build GT."""
import argparse,cv2
from context23 import *
from cli_export import load_probe,load_population,read_label,project,corners
from network18 import VectorPointLineNetwork
from scipy.optimize import linear_sum_assignment

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    meta=load_probe();old={x for rows in load_population().values() for x in rows}
    previous=read(ROOT/'data/pallet/results/pallet_residual_v21/SOURCES.json')['rows']
    excluded={r['pair_group_id'] for r in meta.values() if r['dataset_split']=='train'}|{meta[x]['pair_group_id'] for x in old}|{r['pair_group_id'] for r in previous}
    candidates=sorted([r for r in meta.values() if r['dataset_split']=='val' and r['merged_stem'] not in old and r['pair_group_id'] not in excluded],key=lambda r:digest('v23-intact-quality:'+r['pair_group_id']))
    unique={}
    for r in candidates:unique.setdefault(r['pair_group_id'],r)
    selected=list(unique.values())[:640];assert len(selected)==640
    pops={};sources=[]
    for split,recs in [('new_train',selected[:512]),('new_check',selected[512:])]:
        rows=[]
        for r in recs:
            lab=read_label(r['renderer_label_locator']);ci=lab['camera_data']['intrinsics'];fixed=r['fixed_renderer_dimensions_m_xyz_model_input']
            rows.append(dict(id=r['merged_stem'],image=r['image_path'],padded=True,K=[[ci['fx'],0,ci['cx']],[0,ci['fy'],ci['cy']],[0,0,1]],dims=[fixed[0],fixed[2],fixed[1]],session=r['source'],pair_group_id=r['pair_group_id']))
            sources.append(r)
        pops[split]=rows
    write(RUN/'POPULATIONS.json',pops);write(RUN/'SOURCES.json',dict(rows=sources,probe_sha256=sha(ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl')))
    write(RUN/'PROTOCOL.json',dict(arms=ARMS,seeds=SEEDS,steps=2000,lr=3e-4,weight_decay=1e-4,clip=5,float32=True,
      hypothesis='Intact image observations with genuinely accurate frozen parent points under mild cropping are underrepresented. Expanding independent synthetic source groups and balancing training-only quality bins may preserve accurate points without sacrificing hard errors.',
      design='All arms use original253 plus new512 image groups. 2x2 original movement-weighted BCE versus training-only crop20-outside quality weights x extra-line evidence versus local control. Same batches, initialization, 2000 final updates. Compare existing V22 unit_bce on old253 as historical data-expansion reference; that comparison changes data exposure.',
      weights='Only valid outside points in crop20: frozen-base error bins <=5px, (5,15]px, >15px. For each seed compute bin counts from TRAIN GT only. Inverse-frequency weights N/(3*n), clipped to [0.125,8], renormalized to mean1 over the targeted points. Other points weight1. All originals, hard points, GT and features retained. No GT edit, no inference filter or pseudo-label.',
      input='Image-level six conditions, rerun YOLO and all frozen parents. Parent inference does not receive GT. Fixed raw coordinates and IDs.',
      population='New512 train and128 prospective check groups disjoint from original YOLO train, original requested refinement all splits, and V21 both pools. Historical YOLO validation, not untouched whole-pipeline holdout or certified asset/scene independence.',
      evaluation='Original real297/synth505 and V21new_check253 reused DEV plus new128 synthetic check. Real GT evaluation only; no selection by evaluation results.',
      acceptance='Every real seed clean/crop10/20/30outside <=1.01base; cornerhidden/crop40outside retain>=90% V18 vector gain. Pairwise everyseed>=1% and real9session bootstrap95%CI upper<0. Fixed last checkpoint; report all arms.',
      budget='4arms x3seeds x2000=24000updates. No further loss sweep or PCGrad. Fixed frozen V18 direction; gate only adjusts its magnitude.'))
    oldhash=read(ROOT/'data/pallet/results/pallet_residual_v21/IMAGE_BINDINGS.json')
    forbidden=set(oldhash['forbidden'].values())|set(oldhash['selected'].values());own={r['merged_stem']:sha(ROOT/r['image_path']) for r in selected}
    assert len(set(own.values()))==640 and not (set(own.values())&forbidden)
    write(RUN/'IMAGE_BINDINGS.json',dict(selected=own,prior_binding_sha256=sha(ROOT/'data/pallet/results/pallet_residual_v21/IMAGE_BINDINGS.json')))
    write(RUN/'SPLIT_AUDIT.json',dict(status='PASS',available_groups=len(unique),new_train=512,new_check=128,excluded_old_requested_ids=len(old),excluded_pair_groups=len(excluded),pair_group_overlap=0,exact_image_hash_overlap=0,unique_selected_image_hashes=640,YOLO_historical_val=True))

def extract():
    setup();cv2.setNumThreads(1);register();pop=read(RUN/'POPULATIONS.json');ex=PREP.Extractor()
    for split,recs in pop.items():
        for c in TRAIN_CONDITIONS:
            d=RUN/'cache'/split/c
            if (d/'manifest.json').exists():continue
            d.mkdir(parents=True,exist_ok=True);rows=[];parts=[];chunks=[];n=0
            if c!='clean':cm,co=new_load(split,'clean');lookup={r['id']:r for r in cm['rows']}
            for i,r in enumerate(recs):
                im=PREP.raw_image(r);h,w=im.shape[:2];clean=lookup[r['id']] if c!='clean' else {};box=clean.get('box') or [.2*w,.2*h,.8*w,.8*h]
                if c!='corner20':canvas,info=PREP.stress(im,box,c,r['id'])
                else:
                    box=np.array(box);corner=[0,1,4,5][int(digest('v11-corner:'+r['id'])[:8],16)%4];center=co['points'][clean['observation_index'],corner].numpy() if clean['detected'] else (box[:2]+box[2:])/2;side=.2*np.linalg.norm(box[2:]-box[:2])
                    a=max(0,min(w-1,int(round(center[0]-side/2))));b=max(a+1,min(w,int(round(center[0]+side/2))));u=max(0,min(h-1,int(round(center[1]-side/2))));v=max(u+1,min(h,int(round(center[1]+side/2))))
                    color=np.random.default_rng(int(digest('v11-corner-color:'+r['id'])[:8],16)).integers(40,216,3).astype(np.uint8);canvas=im.copy();canvas[u:v,a:b]=color;info=dict(origin=[0,0],mask=[a,u,b,v],original_hw=[h,w],fill_bgr=color.tolist(),targeted_predicted_corner=corner)
                K=np.array(r['K']);K[:2,2]-=info['origin'];o,det=ex(canvas,K.tolist(),r['dims']);rows.append(dict(**r,condition=c,**info,**det,image_hw=list(canvas.shape[:2]),image_sha256=sha(ROOT/r['image']),observation_index=n if o is not None else None))
                if o is not None:parts.append(o);n+=1
                if len(parts)>=64 or (i+1==len(recs) and parts):
                    p=f'obs_{len(chunks):03d}.pt';torch.save({k:torch.cat([x[k] for x in parts]) for k in KEYS},d/p);chunks.append(p);parts=[]
            write(d/'manifest.json',dict(rows=rows,chunks=chunks,chunk_sha256={p:sha(d/p) for p in chunks},no_GT_detection=True,checkpoint_sha256=PREP.R0_SHA));print('EXTRACT',split,c,n,len(rows),flush=True)
    ex.cap.handle.remove();write(RUN/'EXTRACTION_COMPLETE.json',dict(complete=True))

def parents():
    setup();assert read(RUN/'EXTRACTION_COMPLETE.json')['complete'];rows=[]
    for seed in SEEDS:
        model=VectorPointLineNetwork(seed,'line_vector').eval().requires_grad_(False)
        for split in ('new_train','new_check'):
            for c in TRAIN_CONDITIONS:
                d=RUN/'parents'/split/c;d.mkdir(parents=True,exist_ok=True);p=d/f'seed{seed}.pt'
                if p.exists():continue
                m,O=new_load(split,c,'cuda:0');parts=[]
                with torch.inference_mode():
                    for first in range(0,len(O['points']),32):
                        o=subset(O,slice(first,first+32));out=model.parent(o);rep=dict(out['rep'],lines=out['lines'],line_weights=out['line_weights']);pred=model.evaluator(o,rep,'line_vector');rep['proposal_delta']=pred['points']-rep['base'];parts.append({k:rep[k].cpu() for k in V.REP_KEYS+('proposal_delta',)})
                torch.save({k:torch.cat([x[k] for x in parts]) for k in parts[0]},p);rows.append(dict(seed=seed,split=split,condition=c,frames=len(O['points']),sha256=sha(p)));print('PARENTS',seed,split,c,flush=True)
        del model
    write(RUN/'PARENTS_COMPLETE.json',dict(complete=True,no_GT_forward=True,rows=rows,files={str(p.relative_to(ROOT)):sha(p) for p in (RUN/'parents').rglob('*.pt')}))

def targets23():
    assert read(RUN/'PARENTS_COMPLETE.json')['complete'];sources={r['merged_stem']:r for r in read(RUN/'SOURCES.json')['rows']};raw={};audit=[]
    for fid,r in sources.items():
        label=ROOT/r['yolo_label_path'];txt=[np.fromstring(s,sep=' ') for s in label.read_text().splitlines() if s.strip()];assert len(txt)==1 and txt[0].shape==(32,);kp=txt[0][5:].reshape(9,3)[:8];im=cv2.imread(str(ROOT/r['image_path']));h,w=im.shape[:2];g=kp[:,:2]*[w,h]-100.;valid=kp[:,2]>0;assert valid.any()
        lab=read_label(r['renderer_label_locator']);ci=lab['camera_data']['intrinsics'];K=torch.tensor([[[ci['fx'],0,ci['cx']],[0,ci['fy'],ci['cy']],[0,0,1.]]]);T0=torch.tensor(lab['objects'][0]['pose_transform'],dtype=torch.float32)[None];f=r['fixed_renderer_dimensions_m_xyz_model_input'];uv,_=project(K,T0[:,:3,:3],T0[:,:3,3],corners(torch.tensor([[f[0],f[2],f[1]]])))
        cost=np.linalg.norm(g[valid,None]-uv[0,None,:8].numpy(),axis=-1);a,b=linear_sum_assignment(cost);err=float(cost[a,b].max());assert err<.01,(fid,err)
        raw[fid]=dict(points=g.tolist(),valid=valid.tolist(),label=str(label.relative_to(ROOT)),label_sha256=sha(label));audit.append(dict(id=fid,renderer_set_max_error_px=err))
    write(RUN/'RAW_TARGETS.json',dict(rows=raw))
    for split in ('new_train','new_check'):
        for c in TRAIN_CONDITIONS:
            m,_=new_load(split,c);rows=[]
            for r in m['rows']:
                t=raw[r['id']];g=np.array(t['points']);p=g-np.array(r['origin']);h,w=r['image_hw'];outside=(p[:,0]<0)|(p[:,0]>=w)|(p[:,1]<0)|(p[:,1]>=h);hidden=np.zeros(8,bool)
                if r['mask'] is not None:a,u,b,v=r['mask'];hidden=(g[:,0]>=a)&(g[:,0]<b)&(g[:,1]>=u)&(g[:,1]<v)
                rows.append(dict(id=r['id'],points=p.tolist(),valid=t['valid'],hidden=hidden.tolist(),outside=outside.tolist(),session=r['session'],original_points=g.tolist()))
            write(RUN/'targets'/f'{split}_{c}.json',dict(rows=rows))
    write(RUN/'TARGET_AUDIT.json',dict(status='PASS',rows=audit,renderer_max_px=max(x['renderer_set_max_error_px'] for x in audit),no_inference_identity_reassignment=True));print('TARGETS PASS',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['register','extract','parents','targets'],required=True);a=ap.parse_args();{'register':register,'extract':extract,'parents':parents,'targets':targets23}[a.phase]()
