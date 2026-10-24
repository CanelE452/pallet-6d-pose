"""PUBLIC alpha/query arithmetic only: no RGB decoding/model/current/protected data."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,math
import numpy as np
from scipy.ndimage import distance_transform_edt
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');V=P/'query_visibility_training_duration_control_20261023_v124';C=P/'query_visibility_flat_pallet_cue_source_20261023_v122/actual_corpus'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def stats(v):
 a=np.array(v,np.float64)
 return dict(N=len(a),min=float(a.min())if len(a)else None,median=float(np.median(a))if len(a)else None,max=float(a.max())if len(a)else None)
def main():
 out=V/'ROOT_PUBLIC_QUERY_BOUNDARY_COVERAGE_AUDIT.json';assert not out.exists(),'Preserve any existing audit'
 ip=C/'OCCURRENCES4896_INPUT_ONLY.jsonl';gp=C/'ORIGINAL_WEAK_AND_COMPOSITE_LABELS4896_GT_ONLY.jsonl';seal=json.loads((C/'CORPUS_SEAL.json').read_text())
 assert sha(C/'CORPUS_SEAL.json')=='a023bee62b75596eb79fb0d9919453db92d9c1ba7e79a4c012114e83062e3668'
 inputs=rows(ip);gt=rows(gp);assert len(inputs)==len(gt)==4896
 groups=defaultdict(list);teacher=Counter();missing=Counter()
 for i,(r,g)in enumerate(zip(inputs,gt)):
  assert r['occurrence_index']==g['occurrence_index']==i and r['occurrence_id']==g['occurrence_id']and r['variant']==g['variant']
  for k in('cue_index','jitter_index','donor_rank'):assert r[k]==g[k]
  y=g['candidate_aux_hidden_label_GT_only'];teacher['D'if y==0 else'E'if y==1 else'U']+=1
  if r['variant']=='BEFORE':assert y==0 and r['full_RGB_recipe']['kind']=='ORIGINAL';missing['BEFORE_no_foreground_expected']+=1;continue
  if y<0:
   assert not g['geometry_pair_ready']and r['full_RGB_recipe']['kind']=='ORIGINAL';missing['failed_after_teacherU_original_alias']+=1;continue
  assert g['geometry_pair_ready']and r['full_RGB_recipe']['kind']=='OPAQUE_PATCH'
  assert y==(1 if r['variant']=='OPAQUE_COVER'else 0)
  groups[r['composition_key']].append((r,g))
 alpha_cache={};seen_bindings={};measurements=[]
 for key,records in groups.items():
  assert len(records)==9 and {r['jitter_index']for r,g in records}==set(range(9))
  r0=records[0][0];rec=r0['full_RGB_recipe'];patch=rec['patch'];path=Path(patch['path'])
  if str(path)not in alpha_cache:
   assert sha(path)==patch['sha256']
   with np.load(path,allow_pickle=False)as a:alpha=np.array(a['alpha'],copy=True)
   assert alpha.dtype==bool and alpha.ndim==2 and alpha.any();alpha_cache[str(path)]=alpha;seen_bindings[str(path)]=patch
  alpha=alpha_cache[str(path)];H,W=r0['raw_hw'];px,py=map(int,rec['placement_xy']);ah,aw=alpha.shape
  for r,g in records:assert r['full_RGB_recipe']==rec and r['raw_hw']==[H,W]
  nearest=[(int(math.floor(r['query_xy'][0]+.5)),int(math.floor(r['query_xy'][1]+.5)))for r,g in records]
  assert all(0<=x<W and 0<=y<H for x,y in nearest),'Retain/report OOB rather than clamp'
  # Native clip alpha; EDT raster bounds also include every query. One zero border is explicit.
  x0=max(px,0);y0=max(py,0);x1=min(px+aw,W);y1=min(py+ah,H);assert x1>x0 and y1>y0
  lx=min(x0,min(x for x,y in nearest))-1;ly=min(y0,min(y for x,y in nearest))-1
  rx=max(x1,max(x for x,y in nearest)+1)+1;by=max(y1,max(y for x,y in nearest)+1)+1
  mask=np.zeros((by-ly,rx-lx),bool);mask[y0-ly:y1-ly,x0-lx:x1-lx]=alpha[y0-py:y1-py,x0-px:x1-px]
  assert mask.any();insideEDT=distance_transform_edt(mask);outsideEDT=distance_transform_edt(~mask)
  for (r,g),(qx,qy)in zip(records,nearest):
   mx,my=qx-lx,qy-ly;inside=bool(mask[my,mx]);localx,localy=qx-px,qy-py
   direct=bool(alpha[localy,localx])if 0<=localx<aw and 0<=localy<ah else False
   assert inside==direct
   measurements.append(dict(occurrence_index=r['occurrence_index'],cue_index=r['cue_index'],query_index=r['query_index'],jitter_index=r['jitter_index'],donor_rank=r['donor_rank'],composition_key=key,source=r['source'],raw_split=r['raw_split'],variant=r['variant'],label='E'if g['candidate_aux_hidden_label_GT_only']==1 else'D',q_inside_foreground_alpha=inside,distance_to_opposite_cell_center_px=float(insideEDT[my,mx]if inside else outsideEDT[my,mx])))
 assert len(groups)==460 and len(alpha_cache)==230 and len(measurements)==4140
 def summarize(rs):
  covers=[r for r in rs if r['variant']=='OPAQUE_COVER'];shams=[r for r in rs if r['variant']=='CUE_PRESERVING_SHAM']
  ret={}
  for name,vs in [('known_COVER_E',covers),('known_SHAM_D',shams)]:
   inside=sum(r['q_inside_foreground_alpha']for r in vs);ds=[r['distance_to_opposite_cell_center_px']for r in vs]
   ret[name]=dict(N=len(vs),q_inside_foreground_alpha=inside,q_outside_foreground_alpha=len(vs)-inside,nearest_q_OOB=0,distance_to_opposite_cell_center_px=stats(ds),opposite_center_distance_le1=sum(d<=1 for d in ds),opposite_center_distance_le2=sum(d<=2 for d in ds),opposite_center_distance_le3=sum(d<=3 for d in ds))
  return ret
 allsum=summarize(measurements);breakdown={}
 for source,fold in sorted({(r['source'],r['raw_split'])for r in measurements}):breakdown[source+'::'+fold]=summarize([r for r in measurements if(r['source'],r['raw_split'])==(source,fold)])
 centre=summarize([r for r in measurements if r['jitter_index']==4]);byrank={str(k):summarize([r for r in measurements if r['donor_rank']==k])for k in range(8)}
 exceptions=[r for r in measurements if(r['label']=='E'and not r['q_inside_foreground_alpha'])or(r['variant']=='CUE_PRESERVING_SHAM'and r['q_inside_foreground_alpha'])]
 text='V122의 known COVER E 2070행은 모두 nearest q pixel이 foreground alpha 내부였고, known SHAM D 2070행은 모두 외부였다. 따라서 이 고정 source에는 의미 cue는 숨김(E)이지만 q pixel만 alpha 경계를 넘어 원본 쪽에 남는 사례가 확인되지 않았다. 다만 alpha 외부라는 사실만으로 그 pixel이 target임을 인증할 수는 없다. 이 결과는 합성물의 q/alpha 관계에 대한 source 관찰이며 실제 crop 실패의 전체 원인이나 자연 가림/목표 단서 association을 증명하지 않는다. 9개 jitter는 같은 230 cue-rank 합성물을 공유하므로 2070개의 독립 장면으로 해석하지 않는다.'
 result=dict(completed=True,PASS=True,scope='PUBLIC V122 source alpha/query arithmetic only; no prediction or current evidence.',inputs=bind(ip),GT_sidecar=bind(gp),corpus_seal=bind(C/'CORPUS_SEAL.json'),script=bind(Path(__file__)),original_all4896_retained=True,teacher_states=dict(teacher),before_and_failed_separate=dict(missing),known_after_rows=4140,unique_successful_cue_rank_pairs=230,unique_compositions=460,unique_alpha_files_read=230,alpha_bindings=list(seen_bindings.values()),query_sampling='nearest native cell floor(q+0.5) each axis; exact integer placement_xy; no q movement/clamp',EDT_convention='Cached per native-clipped composition; Euclidean distance between pixel centers to nearest opposite alpha class. This is not continuous-boundary signed distance; one-cell-inside edge gives1, not0.',EDT_composition_cache_entries=460,all=allsum,centre_offset0=centre,by_source_rawsplit=breakdown,by_donor_rank=byrank,exceptions=exceptions,Korean_interpretation=text,semantic_E_with_q_outside_foreground_alpha=len([r for r in measurements if r['label']=='E'and not r['q_inside_foreground_alpha']]),SHAM_D_with_q_inside_foreground_alpha=len([r for r in measurements if r['variant']=='CUE_PRESERVING_SHAM'and r['q_inside_foreground_alpha']]),actual_RGB_decodes_model_fit_current_V112_PnP=0,labels_queries_or_original_files_modified=False,fully_solved=False)
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');note=V/'ROOT_PUBLIC_QUERY_BOUNDARY_COVERAGE_INTERPRETATION_KO.txt';assert not note.exists();note.write_text(text+'\n');print(json.dumps(dict(report=bind(out),interpretation=bind(note),summary=allsum,exceptions=len(exceptions)),ensure_ascii=False))
if __name__=='__main__':main()
