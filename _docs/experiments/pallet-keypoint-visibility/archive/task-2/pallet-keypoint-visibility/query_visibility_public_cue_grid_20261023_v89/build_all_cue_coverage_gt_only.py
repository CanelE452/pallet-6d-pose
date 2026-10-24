"""All cue x fixed grid geometry diagnostic. Metadata only, no input selection."""
from pathlib import Path
from collections import Counter,defaultdict
import json,csv,math,hashlib,datetime,argparse
ROOT=Path(__file__).resolve().parent
P=ROOT.parent
V85=P/'query_visibility_source_cue_corpus_20261023_v85'
V84=P/'query_visibility_paired_cue_20261023_v84'

def b(path):
 p=Path(path);z=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(z).hexdigest(),'bytes':len(z)}
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]
def save(path,obj):
 with Path(path).open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def dump(path,rows):
 with Path(path).open('x')as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def continuous(box,window):
 a,c=box,window
 if c[0]<=a[0] and c[1]<=a[1] and a[2]<=c[2] and a[3]<=c[3]:return 'FULL'
 return 'PARTIAL' if max(a[0],c[0])<min(a[2],c[2]) and max(a[1],c[1])<min(a[3],c[3]) else 'MISS'
def native_raster(box,window):
 x0,y0,x1,y1=math.ceil(box[0]),math.ceil(box[1]),math.ceil(box[2]),math.ceil(box[3])
 if x1<=x0 or y1<=y0:return 'EMPTY_UNCERTAINTY_RASTER_U'
 if x0>=window[0]and y0>=window[1]and x1-1<window[2]and y1-1<window[3]:return 'FULL'
 ax=max(x0,math.ceil(window[0]));bx=min(x1,math.ceil(window[2]));ay=max(y0,math.ceil(window[1]));by=min(y1,math.ceil(window[3]))
 return 'PARTIAL'if ax<bx and ay<by else 'MISS'
def fixtures():
 tests=[([0.,0.,2.,2.],[0.,0.,2.,2.],'FULL'),([0.,0.,2.,2.],[2.,0.,4.,2.],'MISS'),([0.,0.,2.,2.],[1.,0.,3.,2.],'PARTIAL'),([-.5,-.5,.5,.5],[-64.,-64.,64.,64.],'FULL'),([.1,.1,.9,.9],[0.,0.,1.,1.],'FULL')]
 for box,win,want in tests:
  assert continuous(box,win)==want
  points=[(x,y)for x in range(math.floor(box[0])-1,math.ceil(box[2])+1)for y in range(math.floor(box[1])-1,math.ceil(box[3])+1)if box[0]<=x<box[2]and box[1]<=y<box[3]]
  included=[win[0]<=x<win[2]and win[1]<=y<win[3]for x,y in points]
  expected='EMPTY_UNCERTAINTY_RASTER_U' if not points else 'FULL'if all(included)else 'PARTIAL'if any(included)else 'MISS'
  assert native_raster(box,win)==expected
 # Continuous extent across a tile edge can differ from integer-pixel support.
 assert continuous([-.1,0.,1.,1.],[0.,0.,2.,2.])=='PARTIAL'
 assert native_raster([-.1,0.,1.,1.],[0.,0.,2.,2.])=='FULL'
 return {'PASS':True,'fixtures':len(tests)+1,'edge_semantics':'Half-open equality FULL; touching-only MISS; empty native support U; continuous/raster phase mismatch preserved; noGT/RGBread'}
def main():
 lock=read(ROOT/'COVERAGE_GT_ENTRYLOCK_BEFORE_EXECUTION.json');assert lock['builder']==b(__file__)
 for z in lock['GT_metadata_bindings'].values():assert b(z['path'])==z
 inputs=ROOT/'inputs_metadata';seal=read(inputs/'INPUT_GRID_SEAL_BEFORE_GT_JOIN.json')
 for z in seal['files']:assert b(z['path'])==z
 gridpath=inputs/'INPUTS_GRID_ONLY.jsonl';input_hash_before=b(gridpath)
 frames=lines(inputs/'FRAMES_INPUTS_ONLY.jsonl');grid=lines(gridpath);byframe=defaultdict(list)
 for t in grid:byframe[t['frame_input_id']].append(t)
 assert len(grid)==12900 and len(frames)==100 and len(byframe)==100 and all(len(ts)==129 for ts in byframe.values())
 fmap={(f['pair_id'],f['view']):f for f in frames};assert len(fmap)==100
 primary=lines(V85/'all49_completed_corpus/PROPOSED_294_CUE_AUXILIARY_GT_ONLY.jsonl');assert len(primary)==294
 cues=[]
 for q in primary:
  ref=q['reference_xy'];box=q['uncertainty_bbox_xyxy_half_open'];original=q['primary_record_unchanged']
  assert original['reference_xy']==ref and original['uncertainty_bbox_xyxy_half_open']==box
  cues.append({'cue_id':'V85:'+q['pair_id']+':'+q['part_slot'],'corpus':'V85_CANONICAL294','pair_id':q['pair_id'],'slot_or_proposal_id':q['part_slot'],'reference_xy':ref,'uncertainty_bbox_xyxy_half_open':box,'before_state':original['before_state'],'after_state':q['proposed_aux_after_state'],'source_record_unchanged':q,'native8_semantic_id':None,'training_admitted':False})
 union=read(V84/'ROOT_BEFORE_REFERENCE_UNION_LOCK.json');assert union['proposals_N']==8 and union['qualitative_distinct_part_cue_regions_N']==6
 aa=read(V84/'reviewer_a/after_review/AFTER_CUE_EVIDENCE_REVIEW.json');bb=read(V84/'reviewer_b/after_review/AFTER_CUE_EVIDENCE_REVIEW.json')
 amap={x['union_id']:x for x in aa['records']};bmap={x['union_id']:x for x in bb['observations']};assert len(amap)==len(bmap)==8
 for q in union['proposals']:
  key=q['union_id'];ref=q['reference_xy_unchanged'];box=q['uncertainty_bbox_xyxy_half_open_unchanged']
  assert amap[key]['reference_xy_unchanged']==ref and amap[key]['uncertainty_bbox_unchanged']==box
  assert bmap[key]['reference_xy_unchanged']==ref and bmap[key]['uncertainty_bbox_xyxy_half_open_unchanged']==box
  cues.append({'cue_id':'V84_SEPARATE_SEED:'+key,'corpus':'V84_SEED8_SEPARATE_6LINKED_REGIONS','pair_id':union['pair_id'],'slot_or_proposal_id':key,'reference_xy':ref,'uncertainty_bbox_xyxy_half_open':box,'before_state':'APPARENT_VISIBLE_STRUCTURAL_CUE_CONDITIONAL_AI','after_state':{'reviewer_A_unchanged':amap[key]['after_state'],'reviewer_B_unchanged':bmap[key]['after_state']},'source_record_unchanged':q,'after_observations_unchanged':{'A':amap[key],'B':bmap[key]},'duplicate_groups_link_only_unchanged':union['possible_duplicate_groups_before_only'],'native8_semantic_id':None,'training_admitted':False})
 assert len(cues)==302 and len({q['cue_id']for q in cues})==302
 out=ROOT/'cue_coverage_gt_only';out.mkdir(exist_ok=False)
 dump(out/'CUES302_GT_ONLY_UNCHANGED.jsonl',cues)
 reports=[];counts=Counter();strata=defaultdict(Counter);differ=0;combinations=0
 with(out/'ALL77916_CUE_VIEW_TILE_COVERAGE_GT_ONLY.csv').open('x',newline='')as f:
  w=csv.writer(f);w.writerow(['cue_id','corpus','pair_id','view','split','tile_id','L','continuous_coverage','native_raster_coverage'])
  for q in cues:
   box=q['uncertainty_bbox_xyxy_half_open'];ref=q['reference_xy'];missing=box is None or ref is None
   assert (box is None)==(ref is None)
   if not missing:
    assert len(box)==4 and all(math.isfinite(v)for v in box)and box[0]<box[2]and box[1]<box[3]
    assert box[0]<=ref[0]<box[2]and box[1]<=ref[1]<box[3]
   for view in ('before','after'):
    frame=fmap[(q['pair_id'],view)];H,W=frame['raw_hw'];tiles=byframe[frame['frame_input_id']]
    groups=defaultdict(list);raster_groups=defaultdict(list);perL=defaultdict(Counter)
    for tile in tiles:
     c=n='MISSING_REFERENCE_U'
     if not missing:c=continuous(box,tile['window_xyxy_half_open']);n=native_raster(box,tile['window_xyxy_half_open'])
     groups[c].append(tile['tile_id']);raster_groups[n].append(tile['tile_id']);perL[str(tile['roi_side'])][c]+=1
     counts[c]+=1;strata[(q['corpus'],frame['split'],view)][c]+=1;differ+=c!=n;combinations+=1
     w.writerow([q['cue_id'],q['corpus'],q['pair_id'],view,frame['split'],tile['tile_id'],tile['roi_side'],c,n])
    outside=None;native_outside=None;outside_details=None
    if not missing:
     outside=box[0]<0 or box[1]<0 or box[2]>W or box[3]>H
     x0,y0,x1,y1=math.ceil(box[0]),math.ceil(box[1]),math.ceil(box[2]),math.ceil(box[3]);allpix=max(0,x1-x0)*max(0,y1-y0)
     validpix=max(0,min(W,x1)-max(0,x0))*max(0,min(H,y1)-max(0,y0));native_outside=allpix>validpix
     outside_details={'uncertainty_raster_pixels':allpix,'inside_native_image_pixels':validpix,'outside_native_image_pixels':allpix-validpix,'full_box_not_clipped_for_coverage':True}
    reports.append({'cue_id':q['cue_id'],'corpus':q['corpus'],'pair_id':q['pair_id'],'view':view,'split':frame['split'],'frame_input_id':frame['frame_input_id'],'image':frame['image'],'image_sha256':frame['image_sha256'],'raw_hw':frame['raw_hw'],'reference_xy_unchanged':ref,'uncertainty_bbox_unchanged':box,'observation_state_unchanged':q[view+'_state'],'null_reference_preserved_U':missing,'uncertainty_continuous_outside_image':outside,'uncertainty_native_raster_outside_image':native_outside,'uncertainty_extent_details':outside_details,'all_grid_tiles':len(tiles),'full_tile_ids':groups['FULL'],'partial_tile_ids':groups['PARTIAL'],'miss_tile_ids':groups['MISS'],'missing_reference_tile_ids':groups['MISSING_REFERENCE_U'],'counts_by_L':dict(perL),'native_raster_coverage_tile_ids':dict(raster_groups),'diagnostic_only_not_best_tile_selection':True,'training_admission':False})
 assert combinations==77916 and len(reports)==604
 assert all(sum(len(r[k])for k in ('full_tile_ids','partial_tile_ids','miss_tile_ids','missing_reference_tile_ids'))==129 for r in reports)
 dump(out/'ALL604_CUE_VIEW_COVERAGE_GT_ONLY.jsonl',reports)
 known=[r for r in reports if not r['null_reference_preserved_U']]
 summary={'completed':True,'pairs':50,'frames':100,'input_tiles':12900,'canonical_nonseed_cues':294,'separate_seed_proposals':8,'separate_seed_linked_regions':6,'cue_views':604,'all_cue_view_tile_combinations':combinations,'continuous_coverage_counts':dict(counts),'continuous_vs_native_raster_state_mismatches':differ,'known_reference_cue_views':len(known),'null_reference_cue_views':604-len(known),'known_reference_cue_views_with_any_FULL':sum(bool(r['full_tile_ids'])for r in known),'known_reference_cue_views_with_no_FULL':sum(not r['full_tile_ids']for r in known),'uncertainty_outside_native_image_cue_views':sum(r['uncertainty_continuous_outside_image']is True for r in reports),'native_uncertainty_outside_image_cue_views':sum(r['uncertainty_native_raster_outside_image']is True for r in reports),'strata_counts':[{'corpus':k[0],'split':k[1],'view':k[2],'counts':dict(v)}for k,v in strata.items()],'primary_proposed_states':dict(Counter(q['after_state']for q in cues if q['corpus']=='V85_CANONICAL294')),'seed_after_states':'Both original reviewer states preserved separately; no new consensus. Duplicate8 proposals remain linked6regions, not8independenttruths.','all_null_U_retained':True,'input_manifest_before':input_hash_before,'input_manifest_after':b(gridpath),'input_metadata_unchanged_after_GT_join':input_hash_before==b(gridpath),'GTbesttile_or_recenter_or_input_filter':False,'actual_RGB_P3_tensors_or_crops_prepared':False,'training_admission':False,'interpretation_KO':'모든 grid와 모든 cue/view 조합의 영역 포함 관계만 계산했다. FULL이어도 원본이 실제로 관측 가능한지, 경계가 선명한지, padding이 cue를 포함하는지 또는 모델이 cue를 읽는지 보증하지 않는다. 원래 U/null과 native8UNKNOWN을 보존했고 source별 최적 tile을 입력으로 고르지 않았다. 원래 물리 대상은1개이며 native8/context42를 만들지 않았다.','actual_RGB_mask_depth_feature_model_current_reads':0,'new_forward_fit_PnP':0,'protocol':b(ROOT/'BEFORE_GRID_PROTOCOL.json'),'input_seal_before_GT_join':b(inputs/'INPUT_GRID_SEAL_BEFORE_GT_JOIN.json'),'entrylock':b(ROOT/'COVERAGE_GT_ENTRYLOCK_BEFORE_EXECUTION.json'),'builder':b(__file__)}
 assert summary['input_metadata_unchanged_after_GT_join']
 save(out/'ALL_GRID_CUE_COVERAGE_SUMMARY_KO.json',summary)
 files=sorted(p for p in out.iterdir()if p.is_file());save(out/'GT_ONLY_COVERAGE_SEAL.json',{'completed':True,'PASS':True,'files':[b(p)for p in files],'input_seal_unchanged':b(inputs/'INPUT_GRID_SEAL_BEFORE_GT_JOIN.json'),'no_input_selection_or_semantic_gold_or_training_admission':True,'current_forward_fit_PnP':0})
 print(json.dumps({'summary':summary,'seal':b(out/'GT_ONLY_COVERAGE_SEAL.json')},ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fixtures-only',action='store_true');a=ap.parse_args()
 if a.fixtures_only:
  f=fixtures();save(ROOT/'GENERATED_CUE_COVERAGE_EDGE_FIXTURES.json',f);print(json.dumps(f))
 else:main()
