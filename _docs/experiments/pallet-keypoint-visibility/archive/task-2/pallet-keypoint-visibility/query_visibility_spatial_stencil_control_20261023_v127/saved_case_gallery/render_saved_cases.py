"""Fixed-final PUBLIC VAL diagnostic gallery. Saved scores/crops only."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,html,json
import numpy as np
V=Path(__file__).resolve().parents[1];H=Path(__file__).resolve().parent;P=V.parent
RUN=V/'runs/C19_MATCHED_STENCIL_FIXED1000'
POLICY=V/'ROOT_C19_SOURCE_CONTROL_POLICY_BEFORE_FEATURES.json'
CUE_GT=P/'query_visibility_flat_pallet_cue_source_20261023_v122/input_preparation/CUES32_WEAK_EVIDENCE_GT_ONLY.jsonl'
CORPUS=P/'query_visibility_boundary_query_source_20261023_v125/actual_corpus/CORPUS_COMPLETE.json'
ARMS=('POOLED9_REPEAT8','ORDERED_OUTER8')
GROUPS=('improvement','regression','both_wrong')
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,v):
    with Path(p).open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def select():
    assert not(H/'SELECTION_MANIFEST_VAL_BEFORE_CROPS.json').exists()
    control=read(RUN/'CONTROL_COMPLETE.json');assert control['completed']and control['PASS']and control['fixed_updates_each']==1000
    policy=read(POLICY);rs=rows(policy['bindings']['input_occurrences']['path']);gs=rows(policy['bindings']['GT_occurrences']['path']);cue=rows(CUE_GT);cp=read(CORPUS)
    assert len(rs)==len(gs)==4896 and len(cue)==32
    scores={};fitbindings={}
    for arm in ARMS:
        fp=RUN/arm/'FIT_COMPLETE.json';fit=read(fp);assert fit['completed']and fit['PASS']and fit['optimizer_updates']==1000 and fit['fixed_final_not_selector']and not fit['checkpoint_selection']
        assert fit['scores']==bind(Path(fit['scores']['path']))
        with np.load(fit['scores']['path'],allow_pickle=False)as f:scores[arm]={k:f[k]for k in('query_logit','prediction','query_available','global_occurrence_index','occurrence_id','external_target')}
        fitbindings[arm]=dict(fit=bind(fp),scores=fit['scores'])
    s0,s1=(scores[a]for a in ARMS);y=np.array([g['candidate_aux_hidden_label_GT_only']for g in gs]);a=s0['query_available']
    assert np.array_equal(a,s1['query_available'])and int((~a).sum())==51
    for arm,s in scores.items():
        assert np.array_equal(s['global_occurrence_index'],np.arange(4896))and np.array_equal(s['occurrence_id'],np.array([r['occurrence_id']for r in rs]))and np.array_equal(s['external_target'],y)
        assert np.array_equal(s['prediction'],np.where(a,(s['query_logit']>=0).astype(np.int8),-1))
    alpha=np.array([-1 if g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only']is None else int(g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only'])for g in gs])
    variant=np.array([r['variant']for r in rs]);fold=np.array([r['raw_split']for r in rs]);split=np.array([r['split']for r in rs])
    pc=s0['prediction']==y;oc=s1['prediction']==y
    cond={'E_qOUT':(y==1)&(variant=='OPAQUE_COVER')&(alpha==0),'SHAM_D_qIN':(y==0)&(variant=='CUE_PRESERVING_SHAM')&(alpha==1)}
    outcome=dict(improvement=~pc&oc,regression=pc&~oc,both_wrong=~pc&~oc,both_correct=pc&oc)
    counts={};cases=[]
    for name,cm in cond.items():
        counts[name]={}
        for ff in ['ALL']+sorted(set(fold)):
            mask=cm&a&(np.ones(4896,bool)if ff=='ALL'else fold==ff)
            counts[name][ff]=dict(known_available=int(mask.sum()),POOLED_correct=int((mask&pc).sum()),ORDERED_correct=int((mask&oc).sum()),outcomes={k:int((mask&om).sum())for k,om in outcome.items()},unique_cues=len({rs[i]['cue_index']for i in np.flatnonzero(mask)}),unique_source_frame_groups=len({rs[i]['source_frame_group']for i in np.flatnonzero(mask)}),unique_original_RGB=len({rs[i]['full_RGB']['sha256']for i in np.flatnonzero(mask)}))
        for group in GROUPS:
            candidates=np.flatnonzero(cm&a&(split=='VALIDATION')&outcome[group]).tolist();chosen=[];seen=set()
            for i in candidates:
                ci=rs[i]['cue_index']
                if ci not in seen:chosen.append(i);seen.add(ci)
                if len(chosen)==4:break
            if len(chosen)<4:
                chosen.extend(i for i in candidates if i not in chosen and len(chosen)<4)
            for i in chosen:
                r=rs[i];g=gs[i];ci=r['cue_index'];cg=cue[ci];assert r['cue_id']==cg['cue_id']and cg['split']=='VALIDATION'
                ref=np.array(r['query_xy'])-np.array(r['offset_xy']);assert np.array_equal(ref,np.array(cg['reference_xy']))
                # Exact inverse of frozen V90 crop: native=q+((j+.5)/96-.5)*L.
                tolocal=lambda p:((np.asarray(p,np.float64)-r['query_xy'])*96/r['L']+47.5).tolist()
                ql=tolocal(r['query_xy']);rl=tolocal(ref);assert ql==[47.5,47.5]
                assert np.allclose(np.asarray(r['query_xy'])+((np.asarray(rl)+.5)/96-.5)*r['L'],ref,rtol=0,atol=1e-12)
                # Triplets are the same query/donor rank; no nearest/best query selection.
                base=17*r['query_index'];rank=r['donor_rank'];ix=[base,base+1+2*rank,base+2+2*rank]
                assert [rs[j]['variant']for j in ix]==['BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM']and[y[j]for j in ix]==[0,1,0]and a[ix].all()
                for j in ix:
                    assert all(rs[j][k]==r[k]for k in('query_index','cue_index','query_xy','L','split','raw_split'))
                regions=[]
                for region in cg['original_regions']:
                    assert region['kind']in('HALF_OPEN_PIXEL_EDGE','CLOSED_NATIVE_UNCERTAINTY')
                    bb=region['bbox'];mapped=tolocal([bb[:2],bb[2:]])
                    regions.append(dict(original_region=region,display_xyxy96=mapped[0]+mapped[1],display_only_no_raster_redefinition=True))
                trip=[]
                for j in ix:
                    trip.append(dict(occurrence_index=j,occurrence_id=rs[j]['occurrence_id'],variant=rs[j]['variant'],weak_target=int(y[j]),rgb96_sha256=rs[j]['rgb96_sha256'],valid96_sha256=rs[j]['valid96_sha256'],POOLED_logit=float(s0['query_logit'][j]),POOLED_prediction=int(s0['prediction'][j]),ORDERED_logit=float(s1['query_logit'][j]),ORDERED_prediction=int(s1['prediction'][j])))
                cases.append(dict(case_index=len(cases),condition=name,outcome=group,selected_occurrence_index=i,selected_occurrence_id=r['occurrence_id'],available_population=len(candidates),selection='VAL-only, lowest occurrence per distinct cue first, then earliest repeats if<4 unique cues',source=r['source'],split=r['split'],raw_split=r['raw_split'],source_frame_group=r['source_frame_group'],cue_index=ci,cue_id=r['cue_id'],query_xy=r['query_xy'],L=r['L'],donor_rank=rank,reference_GT_display_only=ref.tolist(),q_in_crop96=ql,reference_in_crop96_display_only=rl,original_uncertainty_regions_display_only=regions,triplet=trip,native_mapping='UNKNOWN',human_gold=False))
    lock=dict(completed_before_RGB96_payload_read=True,selection_scope='VALIDATION only LOCO_VAL+TOMIE_DEVELOPMENT; TRAIN counted but excluded from demonstrations by root steering before any lock/crop read.',rule='Per E_qOUT/SHAM_D_qIN and improvement/regression/bothwrong, up to4 distinct cues before repeated variants, lowest original occurrence. Missing groups remain empty. No image-based replacement.',counts_all_original_folds=counts,cases=cases,missing_groups=[n+'::'+g for n in cond for g in GROUPS if not any(c['condition']==n and c['outcome']==g for c in cases)],bindings=dict(control=bind(RUN/'CONTROL_COMPLETE.json'),policy=bind(POLICY),arms=fitbindings,input=policy['bindings']['input_occurrences'],GT=policy['bindings']['GT_occurrences'],source_cue_GT=bind(CUE_GT),stored_crops=cp['stored_crops'],code=bind(Path(__file__))),coordinate_formula='j=(native_coordinate-q)*96/L+47.5; q at47.5. Exact inverse checked for unchanged reference. Region endpoint convention retained; boxes are visual display only.',labels='Conditional AI-weak component/cue visibility; alpha outside does not certify target pixel. No native/human gold.',model_current_V112_fit=0)
    write(H/'SELECTION_MANIFEST_VAL_BEFORE_CROPS.json',lock);print(json.dumps(dict(selection=bind(H/'SELECTION_MANIFEST_VAL_BEFORE_CROPS.json'),cases=len(cases),counts=counts)),flush=True)

def render():
    from PIL import Image,ImageDraw,ImageFont
    lockpath=H/'SELECTION_MANIFEST_VAL_BEFORE_CROPS.json';lock=read(lockpath);assert lock['bindings']['code']==bind(Path(__file__))
    binding=lock['bindings']['stored_crops'];assert binding==bind(Path(binding['path']))
    with np.load(binding['path'],allow_pickle=False)as f:rgb=f['rgb96'];valid=f['valid96'];order=f['occurrence_index']
    assert rgb.shape==(4896,3,96,96)and rgb.dtype==np.uint8 and valid.shape==(4896,1,96,96)and valid.dtype==bool and np.array_equal(order,np.arange(4896))
    for d in('raw_crops','figures'):(H/d).mkdir(exist_ok=False)
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';title=ImageFont.truetype(font,26);normal=ImageFont.truetype(font,20);small=ImageFont.truetype(font,17)
    cards=[];sheets=[];groups={};rawcache={}
    def panel(case,marked):
        canvas=Image.new('RGB',(1280,610),'#f8fafc');dr=ImageDraw.Draw(canvas)
        dr.text((16,10),f'{case["case_index"]:02d} | {case["condition"]} | {case["outcome"]} | {case["source"]} {case["raw_split"]}',font=title,fill='#193e55')
        dr.text((16,48),f'cue{case["cue_index"]} query={case["query_xy"]} L={case["L"]:.3f} rank={case["donor_rank"]} selected={case["selected_occurrence_index"]}',font=small,fill='#426274')
        dr.text((16,77),'DISPLAY MARKS ONLY: yellow q; cyan weak r + original uncertainty'if marked else'PRIMARY: byte-exact stored RGB96; no display markers',font=small,fill='#845723')
        for k,t in enumerate(case['triplet']):
            i=t['occurrence_index'];assert ash(rgb[i])==t['rgb96_sha256']and ash(valid[i])==t['valid96_sha256']
            im=Image.fromarray(rgb[i].transpose(1,2,0));raw=H/'raw_crops'/f'OCC_{i:04d}.png'
            if i not in rawcache:im.save(raw);rawcache[i]=bind(raw)
            if marked:
                im=im.resize((384,384),Image.Resampling.NEAREST);md=ImageDraw.Draw(im)
                # Pixel-centre display phase is (j+.5)*4-.5.
                loc=lambda xy:tuple((np.asarray(xy)+.5)*4-.5)
                qx,qy=loc(case['q_in_crop96']);rx,ry=loc(case['reference_in_crop96_display_only'])
                md.line((qx-13,qy,qx+13,qy),fill='#ffe02b',width=3);md.line((qx,qy-13,qx,qy+13),fill='#ffe02b',width=3)
                md.line((rx-11,ry,rx+11,ry),fill='#23e9ff',width=3);md.line((rx,ry-11,rx,ry+11),fill='#23e9ff',width=3)
                for reg in case['original_uncertainty_regions_display_only']:
                    bb=np.asarray(reg['display_xyxy96']).reshape(2,2);ll=(bb+.5)*4-.5;md.rectangle(tuple(ll.reshape(-1)),outline='#23e9ff',width=2)
            else:im=im.resize((384,384),Image.Resampling.NEAREST)
            x=16+424*k;dr.text((x,110),f'{t["variant"]} GTweak {"D"if t["weak_target"]==0 else"E"}',font=normal,fill='#193e55');canvas.paste(im,(x,145))
            dr.text((x,540),f'pool z={t["POOLED_logit"]:+.3f} -> {"D"if t["POOLED_prediction"]==0 else"E"}',font=small,fill='#345b70')
            dr.text((x,566),f'order z={t["ORDERED_logit"]:+.3f} -> {"D"if t["ORDERED_prediction"]==0 else"E"}',font=small,fill='#345b70')
        return canvas
    for case in lock['cases']:
        paths={}
        for marked in(False,True):
            im=panel(case,marked);name=f'CASE_{case["case_index"]:02d}_{"MARKED"if marked else"RAW"}.png';path=H/'figures'/name;im.save(path);paths['marked'if marked else'raw']=bind(path)
        cards.append(dict(case_index=case['case_index'],selected_occurrence_index=case['selected_occurrence_index'],**paths))
        groups.setdefault((case['condition'],case['outcome']),[]).append(paths)
    for condition in('E_qOUT','SHAM_D_qIN'):
        for outcome in GROUPS:
            cc=groups.get((condition,outcome),[])
            for marked in(False,True):
                sheet=Image.new('RGB',(2560,1220),'#e3ebf1')
                if not cc:ImageDraw.Draw(sheet).text((40,40),f'{condition}/{outcome}: no eligible VAL case',font=title,fill='#193e55')
                for j,c in enumerate(cc):
                    im=Image.open(c['marked'if marked else'raw']['path']);sheet.paste(im,((j%2)*1280,(j//2)*610))
                f=H/'figures'/f'CONTACT_{condition}_{outcome}_{"MARKED"if marked else"RAW"}.png';sheet.save(f);sheets.append(dict(condition=condition,outcome=outcome,marked=marked,count=len(cc),file=bind(f)))
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>C19 PUBLIC VAL saved cases</title><style>body{font-family:system-ui;max-width:1500px;margin:22px auto;color:#163b52}img{max-width:100%}section{background:#f1f6fa;padding:12px;margin:18px 0}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #bdccd6}</style><h1>C19 공개 VAL 저장 사례: pooled ↔ ordered</h1><p>고정1000 update 두 arm. 원 VAL만 사례 선택, TRAIN은 별도 전체 분모에만 포함. 서로 다른 cue를 우선한 occurrence 순서 선택이며 독립 장면 표본이나 accuracy 추정용 gallery가 아닙니다. AI-weak cue 가시성·native UNKNOWN·human gold false. current/V112/새 model call 없음.</p><p><a href="SELECTION_MANIFEST_VAL_BEFORE_CROPS.json">선택·전체 fold 분모·logit manifest</a></p>']
    for s in sheets:
        if s['marked']:continue
        page.append(f'<section><h2>{s["condition"]}: {s["outcome"]} ({s["count"]})</h2><img src="figures/{Path(s["file"]["path"]).name}">')
        marked=next(x for x in sheets if x['condition']==s['condition']and x['outcome']==s['outcome']and x['marked'])
        page.append(f'<details><summary>q / weak r / original uncertainty display</summary><img src="figures/{Path(marked["file"]["path"]).name}"></details></section>')
    page.append('</html>');(H/'GALLERY_KO.html').write_text('\n'.join(page))
    write(H/'RENDER_COMPLETE.json',dict(completed=True,PASS=True,selection=bind(lockpath),code=bind(Path(__file__)),stored_crops=binding,cases=len(cards),cards=cards,contacts=sheets,raw_crops=list(rawcache.values()),raw_stored_pixel_hashes_all_exact=True,gallery=bind(H/'GALLERY_KO.html'),originals_labels_q_L_folds_or_models_changed=False,new_model_fit_current_V112_PnP=0))
    print(json.dumps(dict(completed=True,cases=len(cards),receipt=bind(H/'RENDER_COMPLETE.json'),gallery=bind(H/'GALLERY_KO.html'))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['select','render'],required=True);args=ap.parse_args();(select if args.stage=='select'else render)()
