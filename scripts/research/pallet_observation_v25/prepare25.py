"""Paired opposite-side crops from original pixels, with complete parent reruns."""
from context25 import *
import cv2
from network18 import VectorPointLineNetwork

ARMS=('gate_original','gate_paired','gate_paired_local','vector_local','vector_line','vector_local_edge','vector_line_edge')

def register():
    if (RUN/'PROTOCOL.json').exists():return
    assert read(RUN/'DIAGNOSIS.json')['complete']
    write(RUN/'PROTOCOL.json',dict(arms=ARMS,seeds=SEEDS,steps=2000,learning_rate=3e-4,clip=5,float32=True,
        diagnosis_sha256=sha(RUN/'DIAGNOSIS.json'),
        intervention='Original V23 train253+512 retained. For new512 only, rerun opposite-side20% image crop. At each grouped batch draw, add one crop slot per original: original crop20 in gate_original; original/opposite alternation in paired arms. The original six slots always remain. Thus original observations and hard examples never removed; exposure slot budget matched where detector availability permits. New128 paired images remain check only.',
        rationale='Real right-side mild crop is especially harmful; train/real direction mismatch survives side and error matching. Opposite-side paired imagery tests within-original observation coverage, not a proven sole cause or camera-pose intervention.',
        stages='Stage2 original versus paired gate and paired local gate. Stage3 paired local/line vector, crossed with raw signed image-gradient evidence. All seven are fixed in advance and will be reported even if stage2 fails; stage3 is then a bounded diagnostic, not evidence that data intervention passed.',
        models='Frozen YOLO and V18 parents. Gates use original CauseGate with continuous oracle-alpha movement-weighted BCE. Vector models reuse point and observed-line encoding, predict zero-initialized2D residual within15% box diagonal, mean Euclidean point loss. Stage3 changes output and objective together versus historical gate; does not isolate direction freedom alone. Point/line and edge/noedge within vector family are matched controls.',
        evidence='Image-gradient branch uses grayscale and signed Sobel derivatives from actual pixels. Point and line edge arms both get identical local9x3 evidence. Only line-edge arm additionally reads gradient normal/tangent and along-line differences at36 observed DHT sample positions. These are image cues, NOT annotated visible edges/endpoints or proven attention explanations. No renderer-visible boundary labels are claimed.',
        acceptance='Each seed clean frame mean and crop10/20/30 outside <=1.01base; corner20hidden/crop40outside retain>=90% V18 gain. Extra line improvement eachseed>=1% AND paired real-session95%CI upper<0, all conditions reported. No best-checkpoint selection.',
        population='Existing reused real297/9sessions and synth505/8conditions, V23new128/6conditions plus new opposite crops128. Same original-ID GT. No new independent real holdout available. No training from real GT or per-condition inference filter.',
        budget='7arms x3seeds x2000updates=42000. Final checkpoint only. No grid search.',
        optimizer='AdamW3e-4 weight_decay1e-4 clip5 float32. Group3originals, six original condition slots plus1mild slot; batch plans matched.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'prepare25.py',KIT/'context25.py',KIT/'diagnose25.py')}))
    with NOTE.open('a') as f:f.write('\n### 1단계 완료·후속 설계 등록\n\n실사 약한 잘림에서 안쪽 이동이 필요한 점은 약79%, 신규 합성512에서는 약40%. 실사 오른쪽 잘림에서 방향 오류가 더 크며, 점ID·오차구간·잘리는 쪽을 통제해도 차이가 유지됐다. 이는 단독 원인 증명이 아니다. 같은 원본의 반대쪽20% 잘림을 추가 관측으로 검증한다.\n\n학습 전에7대조×3시드×2000업데이트(42,000)를 고정했다. 기존/짝지은 입력의 적용률3대조, 점/선2D보정×실제 영상 미분 특징 유무4대조. 2단계가 실패하더라도 사용자가 요청한3단계는 한정된 진단으로 끝까지 실행하며 실패 대조도 공개한다. 가시 경계·끝점 GT는 확보됐다고 가정하지 않는다.\n')
    record('1단계에 근거해 반대쪽 잘림 대조 등록. PROTOCOL.json에7대조/3시드/총42,000업데이트 및 재사용DEV 판정 기준 고정. 실제 signed gradient 특징은 관측 단서일 뿐 가시 경계 정답 아님.')

def crop_opposite(im,box,oldside):
    h,w=im.shape[:2];x1,y1,x2,y2=box;bw=max(x2-x1,8.);left=oldside=='right'
    cut=max(1,min(w-32,int(round(x1+.2*bw)))) if left else max(32,min(w-1,int(round(x2-.2*bw))))
    canvas=im[:,cut:].copy() if left else im[:,:cut].copy()
    return canvas,dict(origin=[cut,0] if left else [0,0],mask=None,original_hw=[h,w],crop_side='left' if left else 'right',cut=cut)

def extract():
    setup();cv2.setNumThreads(1);register();ex=PREP.Extractor();pops=read(PREV.RUN/'POPULATIONS.json');raw=read(PREV.RUN/'RAW_TARGETS.json')['rows'];all_audit=[]
    for split,recs in pops.items():
        folder=RUN/'cache'/split/'opposite20'
        if (folder/'manifest.json').exists():continue
        folder.mkdir(parents=True,exist_ok=True);clean={r['id']:r for r in read(PREV.RUN/'cache'/split/'clean/manifest.json')['rows']};old={r['id']:r for r in read(PREV.RUN/'cache'/split/'crop20/manifest.json')['rows']};rows=[];ts=[];parts=[];chunks=[];n=0
        for i,r in enumerate(recs):
            im=PREP.raw_image(r);h,w=im.shape[:2];box=clean[r['id']]['box'] or [.2*w,.2*h,.8*w,.8*h];canvas,info=crop_opposite(im,box,old[r['id']]['crop_side']);assert info['crop_side']!=old[r['id']]['crop_side']
            K=np.array(r['K']);K[:2,2]-=info['origin'];o,det=ex(canvas,K.tolist(),r['dims']);row=dict(**r,condition='opposite20',**info,**det,image_hw=list(canvas.shape[:2]),image_sha256=sha(ROOT/r['image']),transformed_sha256=__import__('hashlib').sha256(canvas.tobytes()).hexdigest(),observation_index=n if o is not None else None);rows.append(row)
            source=raw[r['id']];assert sha(ROOT/source['label'])==source['label_sha256'];g=np.asarray(source['points']);points=g-info['origin'];hh,ww=canvas.shape[:2];outside=(points[:,0]<0)|(points[:,0]>=ww)|(points[:,1]<0)|(points[:,1]>=hh)
            ts.append(dict(id=r['id'],points=points.tolist(),valid=source['valid'],outside=outside.tolist(),hidden=[False]*8,session=r['session'],original_points=g.tolist()))
            # Audit directly from label file rather than cached GT.
            v=np.fromstring((ROOT/source['label']).read_text(),sep=' ')[5:].reshape(9,3)[:8];independent=v[:,:2]*[w+200,h+200]-100-info['origin'];assert np.allclose(independent,points,atol=1e-9,rtol=0)
            if o is not None:parts.append(o);n+=1
            if len(parts)>=64 or (i+1==len(recs) and parts):
                name=f'obs_{len(chunks):03d}.pt';torch.save({k:torch.cat([a[k] for a in parts]) for k in KEYS},folder/name);chunks.append(name);parts=[]
            if (i+1)%128==0:print('PAIRED EXTRACT',split,i+1,n,flush=True)
        write(folder/'manifest.json',dict(rows=rows,chunks=chunks,chunk_sha256={p:sha(folder/p) for p in chunks},no_GT_detection=True,checkpoint_sha256=PREP.R0_SHA));write(RUN/'targets'/f'{split}_opposite20.json',dict(rows=ts));all_audit.append(dict(split=split,requested=len(rows),detected=n,GT_label_replay=True))
    ex.cap.handle.remove();write(RUN/'EXTRACTION_COMPLETE.json',dict(complete=True,rows=all_audit))

def new_load25(split,device='cpu'):
    folder=RUN/'cache'/split/'opposite20';m=read(folder/'manifest.json');parts=[torch.load(folder/p,weights_only=True,map_location='cpu') for p in m['chunks']]
    return m,{k:torch.cat([a[k] for a in parts]).to(device) for k in KEYS}

def parents():
    setup();assert read(RUN/'EXTRACTION_COMPLETE.json')['complete']
    for seed in SEEDS:
        model=VectorPointLineNetwork(seed,'line_vector').eval().requires_grad_(False)
        for split in ('new_train','new_check'):
            path=RUN/'parents'/split/f'seed{seed}.pt'
            if path.exists():continue
            m,O=new_load25(split,'cuda:0');parts=[]
            with torch.inference_mode():
                for start in range(0,len(O['points']),32):
                    o=subset(O,slice(start,start+32));p=model.parent(o);rep=dict(p['rep'],lines=p['lines'],line_weights=p['line_weights']);out=model.evaluator(o,rep,'line_vector');rep['proposal_delta']=out['points']-rep['base'];parts.append({k:rep[k].cpu() for k in V.REP_KEYS+('proposal_delta',)})
            path.parent.mkdir(parents=True,exist_ok=True);torch.save({k:torch.cat([r[k] for r in parts]) for k in parts[0]},path);print('PAIRED PARENTS',seed,split,len(O['points']),flush=True)
        del model
    write(RUN/'PARENTS_COMPLETE.json',dict(complete=True,no_GT_forward=True,files={str(p.relative_to(ROOT)):sha(p) for p in (RUN/'parents').rglob('*.pt')}));record('反대쪽20% 잘림640개 실제 입력과 세 시드 전체 부모 재추론 완료. 새 GT는 기존 라벨에서 crop origin만 이동하고 독립 파일 재계산으로 검증.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['extract','parents'],required=True);a=p.parse_args();extract() if a.phase=='extract' else parents()
