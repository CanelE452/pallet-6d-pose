"""Actual public-real opaque reference-cover source data; no model/pose code."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,importlib.util,json,math,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1];PARENT=ROOT.parent
V71=PARENT/'query_visibility_reference_jitter_20261023_v71'
GRID=V71/'data/ycb_proposal_grid';CUE=V71/'data/ycb_cue_review'
POOL=PARENT/'query_visibility_controlled_real_20261022_v66/provenance/OCCLUDER_POOL.jsonl'
DENY=PARENT/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json'
SYNTH=V71/'data/synth_reference_jitter/build_actual_pilot_v2.py'
SAMPLER=GRID/'build_proposal_grid.py'
REVIEW=V71/'source_audit/ROOT_YCB_CUE_CONDITIONAL_TEACHER_REVIEW.json'
KEYS={'rgb96','crop_valid96','query_valid','query_xy','box_xyxy','L'}
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def need(v,m):
    if not v:raise ValueError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):need(sha(b['path'])==b['sha256'],'Binding changed '+b['path']);return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def write(p,d):need(not p.exists(),'Preserve '+str(p));p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def lines(p,d):need(not p.exists(),'Preserve '+str(p));p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'for r in d))
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def rgb(b):return np.asarray(Image.open(checked(b)).convert('RGB')).copy()
def rank(r):return hashlib.sha256((r['image']['sha256']+':'+r['id']).encode()).hexdigest()
def bbox_cells(bb):return np.arange(math.ceil(bb[1]),math.ceil(bb[3])),np.arange(math.ceil(bb[0]),math.ceil(bb[2]))
def mask_region(mask,left,top,ys,xs):
    yy=ys[:,None]-top;xx=xs[None,:]-left;valid=(yy>=0)&(yy<mask.shape[0])&(xx>=0)&(xx<mask.shape[1]);a=np.zeros((len(ys),len(xs)),bool)
    Y=np.broadcast_to(yy,a.shape);X=np.broadcast_to(xx,a.shape);a[valid]=mask[Y[valid],X[valid]];return a


def freeze():
    need(not(HERE/'COMPOSITION_PROTOCOL.json').exists(),'Protocol already frozen')
    p=dict(schema='V72_YCB_FIXED_REFERENCE_REAL_CUTOUT_COMPOSITION_V1',source_only=True,
      parent='V71 fixed12 cue refs/root-approved conditionalD7/U5 plus all1944 existing grid inputs. Native8semanticID UNKNOWN/nonhuman conditional source evidence.',
      variants=['FULL_COVER','PARTIAL_COVER','NEUTRAL_D'],expected=dict(base_rows=1944,new_refs=7,new_variants=21,queries_per_variant=81,new_rows=1701,total_rows=3645),
      donor_rule='Targets remain frozen proposal order, separately per source split. SHA256(imageSHA+":"+donorID) sorted; same split/different sourceRGB SHA/different objectID. Prefer unused donor objectclass in split; otherwise same SHA order. First geometrically feasible row, at most128 attempted per target; reject reason recorded. No score-based choice.',
      geometry_feasibility='Native donor visible-mask tight bbox, no resize/AA/color transforms. Largest all-opaque native square must fit uncertainty bbox native cell width/height plus2. Entire visible-object cutout retained, not a square texture replacement.',
      full_placement='Integer translate native cutout so its largest opaque square centers over fixed reference uncertainty native cell rectangle; assert every actual cell alpha1.',
      partial_placement='Same donor cutout/y translation, scan integer dx in[-(donorW+uncW),+(donorW+uncW)] from FULL. Choose strict partial cell count closest0.5 coverage; ties minabs(dx), then dx. No alpha truncation, no cue relocation.',
      neutral_placement='Same native cutout, first row-major position of3x3 canvas top/center/bottom and left/center/right alignments with uncertainty allalpha0 and some visible paste. No-op forbidden.',
      compositing='Binary source visible mask opaque front layer after full target RGB. Outsidealpha exactoriginal; insidealpha exactdonorRGB; no blending. This is a controlled intervention certificate, not natural scene depth/semantic gold.',
      query_preservation='All81 source9x9 queries and original oldL/roughbox/refs/uncertainty unchanged. Prefix1944 model arrays byte-copied. All567 rows per new variant retained including MISS/PARTIAL/U.',
      supervision='Fixed reference state FULL_COVER E, PARTIAL_COVER U, NEUTRAL_D D for the7 conditionalD refs. Training target D/E onlywhen observed crop FULL; otherwiseU. Existing5sourceU and every old label unchanged.',
      model_input_NPZ_keys=sorted(KEYS),GT_only='reference/state/uncertainty/alpha/donor/source/semanticID/offset are separate lineage/GT files, never model tensors.',
      sample_recipe='Reuse frozen V71 YCB crop96 exactly, including oldL native span and query/crop validity; do not reuse a different border sampler.',
      conflict_rule='Audit samefullRGB/qLbox and effectivecrop/valid/qLbox. Preserve referenceD/E collisions hidden by crop as diagnostics, but any identical-input conflicting supervisedD/E is a hard data admission failure.',
      code=ref(__file__),bindings=[ref(x)for x in [GRID/'FINAL_SEAL.json',GRID/'INPUT_ONLY.npz',GRID/'INPUT_INDEX_LINEAGE_ONLY.jsonl',GRID/'GT_ONLY_GRID_TEACHERS.jsonl',CUE/'FROZEN_AI_CUE_PROPOSALS_GT_ONLY.json',CUE/'SOURCE_LINEAGE_GT_ONLY.jsonl',REVIEW,POOL,DENY,SYNTH,SAMPLER]],
      GT_recenter_or_L_expansion=False,new_model_forward_training_current_PnP=0)
    write(HERE/'COMPOSITION_PROTOCOL.json',p);print(json.dumps(ref(HERE/'COMPOSITION_PROTOCOL.json')))


def compose_variants(rgb0,donor_rgb,donor_mask,bb,helpers):
    h,w=rgb0.shape[:2];ys,xs=bbox_cells(bb);uw,uh=len(xs),len(ys);sx,sy,K=helpers.largest_square(donor_mask)
    need(K>=max(uw,uh)+2,'Donor opaque square too small')
    fullleft=int(xs[0]-sx-(K-uw)//2);fulltop=int(ys[0]-sy-(K-uh)//2);dh,dw=donor_mask.shape
    def make(left,top):
        center=[left+(dw-1)/2,top+(dh-1)/2];alpha,color,offset=helpers.paste_layer((h,w),donor_rgb,donor_mask.astype(np.uint8),center)
        need(offset==[left,top],'Exact integer paste translation')
        image=np.where(alpha[...,None]>0,color,rgb0).astype(np.uint8)
        need(np.array_equal(image[alpha==0],rgb0[alpha==0]),'Outsidepaste RGB unchanged')
        need(np.array_equal(image[alpha!=0],color[alpha!=0]),'Insidepaste donorRGB exact')
        return image,alpha,dict(integer_translation_xy=offset,uncertainty_opaque_cells=int(alpha[np.ix_(ys,xs)].sum()),uncertainty_total_cells=uw*uh,changed_RGB_pixels=int(np.any(image!=rgb0,axis=2).sum()))
    full=make(fullleft,fulltop);need(full[2]['uncertainty_opaque_cells']==uw*uh,'Full uncertainty opaque')
    choices=[]
    for dx in range(-(dw+uw),dw+uw+1):
        count=int(mask_region(donor_mask,fullleft+dx,fulltop,ys,xs).sum())
        if 0<count<uw*uh:choices.append((abs(2*count-uw*uh),abs(dx),dx,count))
    need(choices,'No actual partial silhouette placement');_,_,dx,_=min(choices);partial=make(fullleft+dx,fulltop)
    neutral=None
    for top in [0,(h-dh)//2,h-dh]:
        for left in [0,(w-dw)//2,w-dw]:
            if not mask_region(donor_mask,left,top,ys,xs).any():
                candidate=make(left,top)
                if candidate[1].any():neutral=candidate;break
        if neutral is not None:break
    need(neutral is not None,'No unchanged-cue neutral placement')
    need(0<partial[2]['uncertainty_opaque_cells']<uw*uh and neutral[2]['uncertainty_opaque_cells']==0,'Actual native certificates')
    return [('FULL_COVER',*full),('PARTIAL_COVER',*partial),('NEUTRAL_D',*neutral)]


def build():
    protocol=read(HERE/'COMPOSITION_PROTOCOL.json');need(checked(protocol['code'])==Path(__file__).resolve(),'Frozen code')
    for b in protocol['bindings']:checked(b)
    need(not(HERE/'DATASET_RECEIPT.json').exists(),'Existing dataset')
    for name in ['RGB_INPUT','ALPHA_GT_ONLY','GALLERIES']:(HERE/name).mkdir(exist_ok=False)
    helpers=load('v71_frozen_paste',SYNTH);sampler=load('v71_exact_ycb_sampler',SAMPLER)
    rootreview=read(REVIEW);accepted=set(rootreview['accepted_for_source_raster_teacher_only_targetIDs']);need(len(accepted)==7,'RootD7')
    proposals={r['target_id']:r for r in read(CUE/'FROZEN_AI_CUE_PROPOSALS_GT_ONLY.json')['proposals']}
    lineage=rows(CUE/'SOURCE_LINEAGE_GT_ONLY.jsonl');targets=[r for r in lineage if r['id']in accepted]
    donors=sorted(rows(POOL),key=rank);deny=set(read(DENY)['image_sha256']);used=defaultdict(set);donorcache={};rejections=[]
    with np.load(GRID/'INPUT_ONLY.npz',allow_pickle=False)as z:need(set(z.files)==KEYS,'Base model input whitelist');base={k:z[k]for k in z.files}
    oldindex=rows(GRID/'INPUT_INDEX_LINEAGE_ONLY.jsonl');oldgt=rows(GRID/'GT_ONLY_GRID_TEACHERS.jsonl');need(len(oldindex)==len(oldgt)==1944,'Base1944')
    arrays={k:[v]for k,v in base.items()};index=[];gt=[];variants=[];gallerydata=[]
    for i,(r,g)in enumerate(zip(oldindex,oldgt)):
        group=r['original_target_id'];scene=int(group.split(':')[1]);f=proposals[group]
        index.append(dict(r,row_id=f'V72:BASE:{i:06d}',source_row_index=i,variant='BASE_ORIGINAL'if r['source_lineage_only']=='YCB_ORIGINAL'else'BASE_EXISTING_NEUTRAL',cue_group=f['cue_refID'],scene_family_group=f'YCB_SCENE:{scene:06d}'))
        gt.append(dict(g,row_id=index[-1]['row_id'],reference_state=g['source_conditional_state'],availability=g['crop_availability'],ureason=g['why'],reference_or_GT_is_not_model_input=True))
    original_lookup={(r['original_target_id'],r['grid_row'],r['grid_col']):r['input_index']for r in oldindex if r['source_lineage_only']=='YCB_ORIGINAL'}
    newdata={k:[]for k in KEYS}
    for t in targets:
        original=t['original'];target_id=t['id'];split=t['split'];cue=proposals[target_id];bb=cue['uncertainty_bbox_xyxy'];ys,xs=bbox_cells(bb);minimum=max(len(ys),len(xs))+2
        need(original['image']['sha256']not in deny,'Target excluded');a=rgb(original['image']);eligible=[d for d in donors if d['split']==split and d['object_id']!=original['object_id']and d['image']['sha256']!=original['image']['sha256']and d['image']['sha256']not in deny]
        fresh=[d for d in eligible if d['object_id']not in used[split]];ordered=fresh+[d for d in eligible if d['object_id']in used[split]]
        chosen=None
        for attempt,d in enumerate(ordered[:128]):
            if d['id']not in donorcache:
                image=rgb(d['image']);m=np.asarray(Image.open(checked(d['visible'])))>0;yy,xx=np.nonzero(m)
                need(len(xx)>0,'Visible donor');db=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
                donorcache[d['id']]=(image[db[1]:db[3],db[0]:db[2]],m[db[1]:db[3],db[0]:db[2]],db)
            drgb,dm,db=donorcache[d['id']];sq=helpers.largest_square(dm)
            if sq[2]<minimum:rejections.append(dict(target_id=target_id,donor_id=d['id'],reason='Native opaque square smaller than fixed uncertainty+2',attempt=attempt));continue
            chosen=(d,drgb,dm,db,sq);break
        need(chosen is not None,'No feasible donor within128');d,drgb,dm,db,sq=chosen;used[split].add(d['object_id'])
        need(d['scene_id']in(set(range(8))if split=='TRAIN'else{8,9}),'Same split donor scene')
        local=compose_variants(a,drgb,dm,bb,helpers);localrows=[]
        original_sites=sorted([r for r in oldindex if r['original_target_id']==target_id and r['source_lineage_only']=='YCB_ORIGINAL'],key=lambda r:(r['grid_row'],r['grid_col']))
        need(len(original_sites)==81,'All81sourcegrid')
        for variant,image,alpha,certificate in local:
            stem=target_id.replace(':','_')+'_'+variant;f=HERE/'RGB_INPUT'/(stem+'.png');ap=HERE/'ALPHA_GT_ONLY'/(stem+'.png');Image.fromarray(image).save(f);Image.fromarray(alpha*255).save(ap)
            state={'FULL_COVER':'E','PARTIAL_COVER':'U','NEUTRAL_D':'D'}[variant];fullsha=hashlib.sha256(image.tobytes()).hexdigest();variantindex=[]
            for old in original_sites:
                q=old['query_xy'];L=old['L'];box=old['box_xyxy'];patch,valid=sampler.crop96(image,q,L);qvalid=sampler.center_valid(q,image.shape[:2]);av=sampler.availability(q,L,image.shape[:2],bb);training=state if state in['D','E']and av=='FULL'else'U';i=len(index)
                for key,value in dict(rgb96=patch,crop_valid96=valid,query_valid=qvalid,query_xy=q,box_xyxy=box,L=L).items():newdata[key].append(value)
                row=dict(input_index=i,row_id=f'V72:{stem}:GRID:{old["grid_row"]}:{old["grid_col"]}',view_index=24+len(variants),view_id=stem,original_target_id=target_id,source_lineage_only='YCB_REFERENCE_COVER',split=split,image=str(f),image_sha256=sha(f),raw_hw=old['raw_hw'],query_xy=q,grid_row=old['grid_row'],grid_col=old['grid_col'],box_xyxy=box,L=L,native_query_valid_from_metadata=qvalid,decoded_fullRGB_sha256=fullsha,variant=variant,cue_group=cue['cue_refID'],scene_family_group=f'YCB_SCENE:{original["scene_id"]:06d}',paired_original_input_index=original_lookup[(target_id,old['grid_row'],old['grid_col'])])
                index.append(row);variantindex.append(i)
                gt.append(dict(input_index=i,row_id=row['row_id'],view_id=stem,original_target_id=target_id,split=split,source_lineage_only=row['source_lineage_only'],grid_row=old['grid_row'],grid_col=old['grid_col'],cue_refID=cue['cue_refID'],reference_xy=cue['reference_xy'],uncertainty_bbox_xyxy=bb,source_conditional_state=state,reference_state=state,crop_availability=av,availability=av,training_state=training,why='conditional intervention/reference and FULL'if training!='U'else'partial-reference-cover or nonFULL/INVALID',ureason=None if training!='U'else('PARTIAL_REFERENCE_COVER'if state=='U'else'CROP_'+av),native8_semantic_ID='UNKNOWN',model_input_contains_this_GT=False,reference_or_GT_is_not_model_input=True))
            v=dict(target_id=target_id,cue_refID=cue['cue_refID'],split=split,variant=variant,reference_state=state,image=ref(f),GT_alpha=ref(ap),reference_xy=cue['reference_xy'],uncertainty_bbox_xyxy=bb,certificate=certificate,donor=d,donor_crop_native_bbox=db,donor_largest_opaque_square_xyK=list(sq),native_donor_resize=False,source_queries_L_box_unchanged=True,input_indices=variantindex)
            variants.append(v);localrows.append((variant,image,alpha,certificate))
        gallerydata.append((original,cue,d,a,localrows))
    for k in KEYS:
        new=np.asarray(newdata[k],dtype=base[k].dtype);arrays[k].append(new);arrays[k]=np.concatenate(arrays[k],axis=0)
        need(np.array_equal(arrays[k][:1944],base[k]),'Base1944prefix exact '+k)
    need(all(len(a)==3645 for a in arrays.values())and len(index)==len(gt)==3645,'Complete3645')
    dest=HERE/'MODEL_INPUT_ONLY.npz';need(not dest.exists(),'No cache overwrite');np.savez_compressed(dest,**arrays)
    lines(HERE/'INPUT_INDEX_LINEAGE_ONLY.jsonl',index);lines(HERE/'GT_ONLY_TEACHERS.jsonl',gt)
    write(HERE/'INTERVENTION_CERTIFICATES_GT_ONLY.json',dict(completed=True,variants=variants,donor_rejections=rejections,conditional_refs=7,variants_per_ref=3,prefix_old1944_unchanged=True))
    audit={}
    for kind in ['full_RGB_query_L_box','effective_crop_valid_query_L_box']:
        groups=defaultdict(list)
        for i,r in enumerate(index):
            params=arrays['query_xy'][i].tobytes()+arrays['box_xyxy'][i].tobytes()+np.asarray(arrays['L'][i],np.float64).tobytes()
            observed=bytes.fromhex(r['decoded_fullRGB_sha256'])if kind.startswith('full_')else arrays['rgb96'][i].tobytes()+arrays['crop_valid96'][i].tobytes()+np.asarray(arrays['query_valid'][i],bool).tobytes()
            groups[hashlib.sha256(observed+params).hexdigest()].append(i)
        semantic=[];supervised=[]
        for key,ix in groups.items():
            if {'D','E'}<={gt[i]['reference_state']for i in ix}:semantic.append(dict(hash=key,indices=ix))
            if {'D','E'}<={gt[i]['training_state']for i in ix}:supervised.append(dict(hash=key,indices=ix))
        audit[kind]=dict(groups=len(groups),duplicate_groups=sum(len(ix)>1 for ix in groups.values()),reference_D_E_collision_groups=semantic,supervised_D_E_conflicts=supervised)
        need(not supervised,'Identical-input supervisedD/E conflict')
    write(HERE/'INPUT_CONFLICT_AUDIT.json',dict(completed=True,N=3645,audits=audit,strict_supervised_D_E_conflicts=0,global_identifiability_proved=False))
    galleries=render_gallery(gallerydata,gt)
    def counts(rr):return dict(N=len(rr),reference_states={k:sum(r['reference_state']==k for r in rr)for k in['D','E','U']},availability={k:sum(r['availability']==k for r in rr)for k in['FULL','PARTIAL','MISS','INVALID']},training={k:sum(r['training_state']==k for r in rr)for k in['D','E','U']})
    receipt=dict(schema='V72_PUBLIC_YCB_REFERENCE_COVER_DATASET_RECEIPT_V1',completed=True,status='ACTUAL_DATA_BUILT_NO_MODEL_FIT',protocol=ref(HERE/'COMPOSITION_PROTOCOL.json'),code=ref(__file__),counts=counts(gt),by_split={s:counts([r for r in gt if r['split']==s])for s in['TRAIN','VALIDATION']},by_variant={s:counts([r for r in gt if index[r['input_index']]['variant']==s])for s in['BASE_ORIGINAL','BASE_EXISTING_NEUTRAL','FULL_COVER','PARTIAL_COVER','NEUTRAL_D']},old1944_prefix_byte_identical=True,all81sites_per_variant_preserved=True,fixed_refs_queries_boxes_L_unchanged=True,model_input_keys=sorted(arrays),GT_reference_alpha_source_semantic_ID_offset_in_model_input=0,conditional_AI_source_teacher_only=True,human_or_native8_gold=False,source_family_and_donor_split_preserved=True,strict_supervised_input_conflicts=0,outputs=[ref(HERE/x)for x in['MODEL_INPUT_ONLY.npz','INPUT_INDEX_LINEAGE_ONLY.jsonl','GT_ONLY_TEACHERS.jsonl','INTERVENTION_CERTIFICATES_GT_ONLY.json','INPUT_CONFLICT_AUDIT.json']],galleries=galleries,direct_visual_QA='Separate additive QA after actual views; receipt is immutable.',new_model_forward_training_current_PnP=0,model_fit_or_current_gate_authorized=False)
    write(HERE/'DATASET_RECEIPT.json',receipt);print(json.dumps(dict(receipt=ref(HERE/'DATASET_RECEIPT.json'),counts=receipt['counts'],by_split=receipt['by_split']),ensure_ascii=False))


def render_gallery(data,gt):
    files=[]
    for k,(original,cue,donor,before,local)in enumerate(data):
        im=Image.new('RGB',(1720,1040),'#f5f7fa');d=ImageDraw.Draw(im)
        def text(xy,s,size=21):d.text(xy,s,font=ImageFont.truetype(FONT,size),fill='#172536')
        text((20,12),original['split']+' '+original['id']+' | donor object '+str(donor['object_id'])+' '+donor['id'],26)
        text((20,57),'Fixed RGB cue/uncertainty; same81 querygrid +oldL/box. Real native cutout, no resize/AA. Conditional source teacher, not human/native8 gold.',19)
        xs=[('ORIGINAL',before,None,None)]+local
        x,y=cue['reference_xy'];bb=cue['uncertainty_bbox_xyxy']
        for j,(name,rgbv,alpha,cert)in enumerate(xs):
            ox=20+j*425;text((ox,115),name,23)
            full=Image.fromarray(rgbv).resize((384,288),Image.Resampling.NEAREST);fd=ImageDraw.Draw(full);fd.rectangle(tuple(v*.6 for v in bb),outline='#ff00bb',width=2);im.paste(full,(ox,155))
            raw=Image.fromarray(rgbv).crop((x-32,y-32,x+32,y+32)).resize((384,384),Image.Resampling.NEAREST);rd=ImageDraw.Draw(raw);rd.rectangle(tuple((v-(x-32 if n%2==0 else y-32))*6 for n,v in enumerate(bb)),outline='#ffdd00',width=2);rd.line((180,192,204,192),fill='#eeeeee',width=2);rd.line((192,180,192,204),fill='#eeeeee',width=2);im.paste(raw,(ox,485))
            n=0 if cert is None else cert['uncertainty_opaque_cells'];N=(bb[2]-bb[0])*(bb[3]-bb[1]);text((ox,890),f'opaque uncertainty cells {n}/{N}',20)
            text((ox,930),'D'if name in['ORIGINAL','NEUTRAL_D']else'E'if name=='FULL_COVER'else'U',26)
        text((20,990),'Top: entire nativeRGB. Bottom: diagnostic native64 around fixed reference, NOT the model crop. QueryL and reference geometry were never moved.',19)
        f=HERE/'GALLERIES'/f'{k+1:02d}_{original["id"].replace(":","_")}.png';im.save(f);files.append(ref(f))
    return files


def mock(path):
    helpers=load('mock_native_paste',SYNTH);rgb0=np.zeros((40,50,3),np.uint8);drgb=np.full((15,17,3),127,np.uint8);mask=np.ones((15,17),bool);bb=[20,15,27,22]
    result=compose_variants(rgb0,drgb,mask,bb,helpers);need([r[0]for r in result]==['FULL_COVER','PARTIAL_COVER','NEUTRAL_D'],'All variants')
    need(result[0][3]['uncertainty_opaque_cells']==49 and 0<result[1][3]['uncertainty_opaque_cells']<49 and result[2][3]['uncertainty_opaque_cells']==0,'Native coverage certificates')
    write(Path(path),dict(completed=True,status='PASS',code=ref(__file__),paste_helpers=ref(SYNTH),generated_only=True,coverage_D_E_U_and_outside_RGB_checks=True,actual_source_RGB_GT_or_scores_read=0,new_model_training_current_PnP=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['mock','freeze','build']);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.mode=='mock':mock(a.out)
    elif a.mode=='freeze':freeze()
    else:build()
