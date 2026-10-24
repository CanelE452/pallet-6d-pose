"""Saved PUBLIC metadata/alpha audit, fixed-first selection, existing crop display only."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');V=P/'query_visibility_boundary_query_source_20261023_v125';C=V/'actual_corpus';OLD=P/'query_visibility_flat_pallet_cue_source_20261023_v122/actual_corpus';O=Path(__file__).resolve().parent
FONT='/usr/share/fonts/truetype/nanum/NanumGothic.ttf';BOLD='/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def check(b):assert bind(Path(b['path']))==b;return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def text(d,xy,s,n=20,bold=False,color='#182d3b'):d.text(xy,s,font=ImageFont.truetype(BOLD if bold else FONT,n),fill=color)
def main():
 assert not(O/'REVIEW.json').exists();receipt=read(C/'CORPUS_COMPLETE.json');assert receipt['completed']and receipt['PASS']and receipt['occurrences']==4896
 for b in receipt['outputs'].values():check(b)
 check(receipt['policy']);check(receipt['code']);oldr=rows(OLD/'OCCURRENCES4896_INPUT_ONLY.jsonl');oldg=rows(OLD/'ORIGINAL_WEAK_AND_COMPOSITE_LABELS4896_GT_ONLY.jsonl');rs=rows(check(receipt['input_manifest']));gs=rows(check(receipt['weak_evidence']));assert len(rs)==len(gs)==len(oldr)==len(oldg)==4896
 ev=rows(OLD.parent/'input_preparation/CUES32_WEAK_EVIDENCE_GT_ONLY.jsonl');alpha={};relations={};counters=defaultdict(Counter);states=Counter();centres=[];Lchanges=0;contextchanges=0
 unchanged=('source','split','raw_split','source_frame_group','target_instance_id','target_bbox_xyxy','raw_hw','full_RGB','full_RGB_format','decoded_RGB_sha256','L0','full_RGB_recipe','variant','donor_rank','cue_id')
 for i,(r,g)in enumerate(zip(rs,gs)):
  cue,j,role=i//153,(i%153)//17,i%17;pi=153*cue+68+role;p=oldr[pi];og=oldg[pi];b=oldr[153*cue+68]
  assert r['occurrence_index']==g['occurrence_index']==r['rgb96_row']==i and r['query_index']==9*cue+j and r['cue_index']==cue and r['jitter_index']==g['jitter_index']==j
  assert r['parent_recipe_occurrence_index']==g['inherited_label_from_parent_occurrence']==pi
  assert all(r[k]==p[k]for k in unchanged)
  for k,v in og.items():
   if k not in ('occurrence_index','occurrence_id','jitter_index'):assert g[k]==v,(i,k)
  off=np.array([(-.25,0,.25)[j%3],(-.25,0,.25)[j//3]])*b['L0'];q=np.array(b['query_xy'])+off;points=np.array(b['initial_points'],float);points[0]=q;good=np.isfinite(points).all(1)&~(points==[-1,-1]).all(1);ctx=np.r_[points[good].min(0),points[good].max(0)];L=float(np.clip(.25*np.ptp(points[good],axis=0).max(),16,max(r['raw_hw'])))
  assert np.array_equal(q,r['query_xy'])and np.array_equal(points,r['initial_points'])and np.array_equal(ctx,r['context_box_xyxy'])and L==r['L']and np.array_equal(off,r['offset_xy'])
  H,W=r['raw_hw'];x,y=np.floor(q+.5).astype(int);cont=bool(0<=q[0]<W and 0<=q[1]<H);nearest=bool(0<=x<W and 0<=y<H)
  assert r['q_native_continuous_valid']==cont and r['q_nearest_valid']==nearest and r['q_native_valid']==(cont and nearest)
  if j==4:
   assert r['query_xy']==p['query_xy']and r['L']==p['L']and r['context_box_xyxy']==p['context_box_xyxy'];centres.append((i,pi))
  if L!=p['L']:Lchanges+=1
  if r['context_box_xyxy']!=p['context_box_xyxy']:contextchanges+=1
  states[str(g['candidate_aux_hidden_label_GT_only'])]+=1
  rec=r['full_RGB_recipe'];ins=None
  if rec['kind']=='OPAQUE_PATCH':
   path=rec['patch']['path']
   if path not in alpha:
    check(rec['patch'])
    with np.load(path,allow_pickle=False)as f:alpha[path]=np.array(f['alpha'],copy=True)
    assert alpha[path].dtype==bool
   pa=alpha[path];px,py=rec['placement_xy'];lx,ly=x-px,y-py;ins=bool(nearest and 0<=lx<pa.shape[1]and 0<=ly<pa.shape[0]and pa[ly,lx])
  assert g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only']==ins and g['nearest_q_in_image_GT_diagnostic_only']==nearest
  relations[i]=ins
  if g['candidate_aux_hidden_label_GT_only']in(0,1)and ins is not None:
   key=r['source']+'::'+r['raw_split'];counters[key][r['variant']+'::'+('inside'if ins else'outside')+'::'+('valid'if(cont and nearest)else'OOB')]+=1
 assert len(centres)==544 and states==Counter({'0':2358,'1':2070,'-1':468})
 validEout=sum(1 for i in range(4896)if gs[i]['candidate_aux_hidden_label_GT_only']==1 and relations[i]is False and rs[i]['q_native_valid']);validDin=sum(1 for i in range(4896)if rs[i]['variant']=='CUE_PRESERVING_SHAM'and gs[i]['candidate_aux_hidden_label_GT_only']==0 and relations[i]is True and rs[i]['q_native_valid']);assert(validEout,validDin)==(639,471)
 # Selection is metadata/alpha-only and fixed before opening new or old RGB96 arrays.
 selected=[]
 for source,fold in [('LOCO','TRAIN'),('LOCO','VALIDATION'),('TOMIE','TOMIE_DEVELOPMENT_STRESS')]:
  for typ in ('E_q_outside','D_q_inside'):
   eligible=[i for i,r in enumerate(rs)if r['source']==source and r['raw_split']==fold and r['query_available']and r['q_native_valid']and ((typ=='E_q_outside'and r['variant']=='OPAQUE_COVER'and gs[i]['candidate_aux_hidden_label_GT_only']==1 and relations[i]is False)or(typ=='D_q_inside'and r['variant']=='CUE_PRESERVING_SHAM'and gs[i]['candidate_aux_hidden_label_GT_only']==0 and relations[i]is True))]
   if not eligible:selected.append(dict(source=source,raw_split=fold,type=typ,missing=True));continue
   i=eligible[0];r=rs[i];qi=r['query_index'];rank=r['donor_rank'];triplet=[17*qi,17*qi+1+2*rank,17*qi+2+2*rank]
   selected.append(dict(source=source,raw_split=fold,type=typ,missing=False,selected_occurrence_index=i,query_index=qi,cue_index=r['cue_index'],jitter_index=r['jitter_index'],donor_rank=rank,triplet_occurrence_indices=triplet,q=r['query_xy'],L=r['L'],reference_xy=ev[r['cue_index']]['reference_xy'],cover_q_inside=relations[triplet[1]],sham_q_inside=relations[triplet[2]]))
 lock=dict(completed=True,selection_rule='Original occurrence order first native+factual available known E/COVER qalpha outside and known D/SHAM qalpha inside, per LOCO TRAIN/LOCO VALIDATION/TOMIE DEVELOPMENT. No replacement.',corpus_complete=bind(C/'CORPUS_COMPLETE.json'),input_manifest=receipt['input_manifest'],GT=receipt['weak_evidence'],selected=selected,all_new_stored_RGB96_arrays_unopened_before_this_lock=True,model_scores_read=False)
 (O/'SELECTION_LOCK_BEFORE_CROP_ARRAYS.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
 with np.load(check(receipt['stored_crops']),allow_pickle=False)as f:rgb=np.array(f['rgb96'],copy=True);valid=np.array(f['valid96'],copy=True);order=np.array(f['occurrence_index'],copy=True)
 assert rgb.shape==(4896,3,96,96)and rgb.dtype==np.uint8 and valid.shape==(4896,1,96,96)and valid.dtype==bool and np.array_equal(order,np.arange(4896))
 availability=Counter()
 for i,r in enumerate(rs):
  q=np.asarray(r['query_xy']);L=r['L'];H,W=r['raw_hw'];u=(np.arange(96)+.5)/96-.5;xs=q[0]+u*L;ys=q[1]+u*L;expect=(ys[:,None]>=-.5)&(ys[:,None]<H-.5)&(xs[None,:]>=-.5)&(xs[None,:]<W-.5)
  assert np.array_equal(valid[i,0],expect)and not rgb[i,:,~expect].any();assert ash(rgb[i])==r['rgb96_sha256']and ash(valid[i])==r['valid96_sha256']
  b=r['context_box_xyxy'];tx=b[0]+(np.arange(224)+.5)*(b[2]-b[0])/224;ty=b[1]+(np.arange(224)+.5)*(b[3]-b[1])/224
  ctx=bool(((tx>=0)&(tx<=W-1)).any()and((ty>=0)&(ty<=H-1)).any());factual=bool(r['q_native_valid']and expect[47:49,47:49].all());a=factual and ctx
  assert r['context_available']==ctx and r['factual_query_available']==factual and r['query_available']==a;availability['available'if a else'U']+=1
 with np.load(OLD/'INPUT_CROPS4896_INPUT_ONLY.npz',allow_pickle=False)as f:
  oldrgb=f['rgb96'];oldvalid=f['valid96']
  for i,pi in centres:assert np.array_equal(rgb[i],oldrgb[pi])and np.array_equal(valid[i],oldvalid[pi])
 assert availability==Counter({'available':4845,'U':51})
 cards=[];markedcards=[];figures=[]
 for n,s in enumerate(selected):
  card=Image.new('RGB',(1600,390),'white');markedcard=card.copy()
  for dst,mark in [(card,False),(markedcard,True)]:
   d=ImageDraw.Draw(dst);text(d,(20,20),f"{n+1}. {s['source']} · {s['raw_split']}",24,True)
   if s['missing']:text(d,(20,70),'MISSING · 대체 사례 없음',23);continue
   text(d,(20,68),'고정 첫 '+('E · q alpha 밖'if s['type']=='E_q_outside'else'D · q alpha 안'),22,True)
   text(d,(20,112),f"cue{s['cue_index']} · jitter{s['jitter_index']} · rank{s['donor_rank']}",19)
   text(d,(20,148),f"선택 occurrence {s['selected_occurrence_index']}",18)
   text(d,(20,184),f"q=({s['q'][0]:.2f}, {s['q'][1]:.2f})",18)
   text(d,(20,220),f"L={s['L']:.3f} native px",18)
   text(d,(20,256),f"q alpha: COVER {s['cover_q_inside']} / SHAM {s['sham_q_inside']}",18)
   text(d,(20,296),'동일 cue 의미 D/E · q pixel gold 아님',18)
   text(d,(20,332),'모델/점수 읽기·새 crop 추출 없음',18)
   for j,(ix,title)in enumerate(zip(s['triplet_occurrence_indices'],['BEFORE · weak D','COVER · semantic E','SHAM · semantic D'])):
    raw=Image.fromarray(rgb[ix].transpose(1,2,0));text(d,(510+350*j,22),title,21,True)
    if mark:
     dd=ImageDraw.Draw(raw);qx=qy=47.5;ref=np.array(s['reference_xy']);rr=(ref-np.array(s['q']))/s['L']*96+47.5
     dd.line((qx-4,qy,qx+4,qy),fill='#ffe133',width=1);dd.line((qx,qy-4,qx,qy+4),fill='#ffe133',width=1);dd.ellipse((rr[0]-3,rr[1]-3,rr[0]+3,rr[1]+3),outline='#00eaff',width=1)
    dst.paste(raw.resize((288,288),Image.Resampling.NEAREST),(510+350*j,65))
  cp=O/f'CASE_{n:02d}_RAW_TRIPLET.png';mp=O/f'CASE_{n:02d}_MARKED_TRIPLET.png';card.save(cp);markedcard.save(mp);cards.append(card);markedcards.append(markedcard);figures.append(dict(raw=bind(cp),marked=bind(mp)))
 for mark,allcards,name in [(False,cards,'FIXED6_RAW_6x3.png'),(True,markedcards,'FIXED6_MARKED_6x3.png')]:
  board=Image.new('RGB',(1600,2460),'#edf2f5');d=ImageDraw.Draw(board);text(d,(20,15),'공개 V125 · semantic cue와 q pixel 경계의 고정 6조합',30,True);text(d,(20,60),'같은 원 합성물·q/L의 BEFORE/COVER/SHAM · weak AUX · native UNKNOWN · 예측/정확도 아님',21)
  text(d,(20,98),'표시 별도: 노랑+ q / 하늘원 원 weak reference r'if mark else'원 저장 RGB96 무표시 · 선택은 source/fold별 metadata/alpha 첫 순서로 고정',19)
  for j,im in enumerate(allcards):board.paste(im,(0,120+390*j))
  board.save(O/name)
 contacts=[]
 for k in range(3):
  c=Image.new('RGB',(1600,780),'white');c.paste(cards[2*k],(0,0));c.paste(cards[2*k+1],(0,390));p=O/f'CONTACT_{k+1:02d}_RAW.png';c.save(p);contacts.append(bind(p))
 review=dict(completed=True,PASS=True,blockers=[],corpus_complete=bind(C/'CORPUS_COMPLETE.json'),selection_lock=bind(O/'SELECTION_LOCK_BEFORE_CROP_ARRAYS.json'),script=bind(Path(__file__)),all4896_order_and_parent_identity_fold_recipe_semantic_labels_unchanged=True,all_q8_context_and_L_recomputed_exact=True,centres544_RGB_valid_qL_context_exact=True,stored_all_RGB_and_valid_hashes_exact=True,all_valid96_geometry_and_invalid_zero_exact=True,availability=dict(availability),teacher_states=dict(states),valid_E_qalpha_out=validEout,valid_SHAM_D_qalpha_in=validDin,alpha_counts_native_valid_vs_OOB={k:dict(v)for k,v in counters.items()},L_changed_occurrences_vs_parent=Lchanges,context_changed_occurrences_vs_parent=contextchanges,selected=selected,figures=figures,raw_board=bind(O/'FIXED6_RAW_6x3.png'),marked_board=bind(O/'FIXED6_MARKED_6x3.png'),raw_contacts=contacts,new_RGB96_noncentre_values_reconstructed_from_native_RGB=False,limitation='Noncentre RGB checked via bound rowhash, exact factual validity and preserved native recipes; no raw RGB re-extraction. Centre544 independently exact original crops. Semantic labels remain weak samecue visibility, not q-pixel/native gold.',actual_model_scores_current_V112_PnP_reads_or_calls=0,actual_visual_view_pending=True)
 (O/'REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(completed=True,PASS=True,review=bind(O/'REVIEW.json'),selection=selected,raw_board=review['raw_board'],contacts=contacts),ensure_ascii=False))
if __name__=='__main__':main()
