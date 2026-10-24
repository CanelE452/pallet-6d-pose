"""Separate arithmetic replay of V89 JSON metadata, no image/source tensor reads."""
from pathlib import Path
from collections import Counter,defaultdict
import json,csv,hashlib,math,datetime
R=Path(__file__).resolve().parent

def b(p):
 p=Path(p);z=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(z).hexdigest(),'bytes':len(z)}
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def save(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def area_class(a,t):
 area=(a[2]-a[0])*(a[3]-a[1]);intersection=max(0.,min(a[2],t[2])-max(a[0],t[0]))*max(0.,min(a[3],t[3])-max(a[1],t[1]))
 return 'MISS'if intersection==0 else 'FULL'if intersection==area else 'PARTIAL'
def enum_class(a,t):
 xs=[x for x in range(math.floor(a[0])-1,math.ceil(a[2])+1)if a[0]<=x<a[2]];ys=[y for y in range(math.floor(a[1])-1,math.ceil(a[3])+1)if a[1]<=y<a[3]]
 if not xs or not ys:return 'EMPTY_UNCERTAINTY_RASTER_U'
 support=[t[0]<=x<t[2]and t[1]<=y<t[3]for x in xs for y in ys]
 return 'FULL'if all(support)else 'PARTIAL'if any(support)else 'MISS'
def main():
 pr=read(R/'BEFORE_GRID_PROTOCOL.json');source=read(pr['source_cohort']['path'])['rows'];assert b(pr['source_cohort']['path'])==pr['source_cohort']
 frames=lines(R/'inputs_metadata/FRAMES_INPUTS_ONLY.jsonl');tiles=lines(R/'inputs_metadata/INPUTS_GRID_ONLY.jsonl');td={x['tile_id']:x for x in tiles};assert len(td)==len(tiles)==12900
 assert len(source)==50 and len(frames)==100 and len({r['pair_id']for r in source})==50
 byframe=defaultdict(list)
 allowed={'tile_id','frame_input_id','image','image_sha256','raw_hw','pair_id','view','split','source_family','session','seed_V84_reuse','query_xy','roi_side','window_xyxy_half_open','native_valid_pixels','padding_pixels','native_valid_fraction'}
 for t in tiles:assert set(t)==allowed;byframe[t['frame_input_id']].append(t)
 fdict={(r['pair_id'],r['view']):r for r in frames};geometry_pairs=[]
 for s in source:
  geoms=[]
  for view in ('before','after'):
   f=fdict[(s['pair_id'],view)];assert f['image']==s[view+'_RGB']['path']and f['image_sha256']==s[view+'_RGB']['sha256']and f['split']==s['split']and f['seed_V84_reuse']==s['seed_V84_reuse']
   H,W=f['raw_hw'];expected=[]
   for L in (128,256):
    xx=[];x=0
    while x<W:xx.append(x);x+=L//2
    if xx[-1]!=W-1:xx.append(W-1)
    yy=[];y=0
    while y<H:yy.append(y);y+=L//2
    if yy[-1]!=H-1:yy.append(H-1)
    for y in yy:
     for x in xx:expected.append((L,float(x),float(y)))
   local=byframe[f['frame_input_id']]
   assert [(t['roi_side'],*t['query_xy'])for t in local]==expected
   for t in local:
    L,x,y=t['roi_side'],*t['query_xy'];assert x.is_integer()and y.is_integer()
    win=[x-L/2,y-L/2,x+L/2,y+L/2];assert t['window_xyxy_half_open']==win
    nx=sum(0<=int(x)+dx<W for dx in range(-L//2,L//2));ny=sum(0<=int(y)+dy<H for dy in range(-L//2,L//2))
    assert t['native_valid_pixels']==nx*ny and t['padding_pixels']==L*L-nx*ny and t['native_valid_fraction']==nx*ny/L**2
   geoms.append(expected)
  assert geoms[0]==geoms[1];geometry_pairs.append(s['pair_id'])
 croot=R/'cue_coverage_gt_only';cues=lines(croot/'CUES302_GT_ONLY_UNCHANGED.jsonl');reports=lines(croot/'ALL604_CUE_VIEW_COVERAGE_GT_ONLY.jsonl');cmap={r['cue_id']:r for r in cues};rd={(r['cue_id'],r['view']):r for r in reports}
 assert len(cmap)==302 and len(rd)==604
 canonical=lines(R.parent/'query_visibility_source_cue_corpus_20261023_v85/all49_completed_corpus/PROPOSED_294_CUE_AUXILIARY_GT_ONLY.jsonl')
 canonical_map={(r['pair_id'],r['part_slot']):r for r in canonical};count_canonical=count_seed=0
 for c in cues:
  if c['corpus']=='V85_CANONICAL294':assert canon(c['source_record_unchanged'])==canon(canonical_map[(c['pair_id'],c['slot_or_proposal_id'])]);count_canonical+=1
  else:count_seed+=1
 assert count_canonical==294 and count_seed==8
 counts=Counter();replayed=0;diffs=0;seen=set();coverage_by_report=defaultdict(lambda:defaultdict(list));raster_by_report=defaultdict(lambda:defaultdict(list))
 with(croot/'ALL77916_CUE_VIEW_TILE_COVERAGE_GT_ONLY.csv').open()as f:
  for row in csv.DictReader(f):
   key=row['cue_id'],row['view'];q=cmap[row['cue_id']];t=td[row['tile_id']];assert t['pair_id']==q['pair_id']==row['pair_id']and t['view']==row['view']and t['split']==row['split']and row['corpus']==q['corpus']and int(row['L'])==t['roi_side']
   k=(*key,row['tile_id']);assert k not in seen;seen.add(k)
   box=q['uncertainty_bbox_xyxy_half_open'];win=t['window_xyxy_half_open'];c=n='MISSING_REFERENCE_U'
   if box is not None:c=area_class(box,win);n=enum_class(box,win)
   assert row['continuous_coverage']==c and row['native_raster_coverage']==n
   counts[c]+=1;replayed+=1;diffs+=c!=n;coverage_by_report[key][c].append(row['tile_id']);raster_by_report[key][n].append(row['tile_id'])
 assert replayed==77916
 for key,rep in rd.items():
  q=cmap[key[0]];assert rep['reference_xy_unchanged']==q['reference_xy']and rep['uncertainty_bbox_unchanged']==q['uncertainty_bbox_xyxy_half_open']
  assert rep['observation_state_unchanged']==q[key[1]+'_state']
  for c,f in [('FULL','full_tile_ids'),('PARTIAL','partial_tile_ids'),('MISS','miss_tile_ids'),('MISSING_REFERENCE_U','missing_reference_tile_ids')]:assert rep[f]==coverage_by_report[key][c]
  for c,ids in raster_by_report[key].items():assert rep['native_raster_coverage_tile_ids'][c]==ids
  assert sum(len(v)for v in coverage_by_report[key].values())==129
 summary=read(croot/'ALL_GRID_CUE_COVERAGE_SUMMARY_KO.json');assert summary['continuous_coverage_counts']==dict(counts)and summary['continuous_vs_native_raster_state_mismatches']==diffs
 for folder,name in [(R/'inputs_metadata','INPUT_GRID_SEAL_BEFORE_GT_JOIN.json'),(croot,'GT_ONLY_COVERAGE_SEAL.json')]:
  for z in read(folder/name)['files']:assert b(z['path'])==z
 audit={'completed':True,'PASS':True,'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code':b(__file__),'scope':'Separate JSON arithmetic replay, no producer imports/imagepixels/GTarrays/model; area-basedcontinuous and explicitintegerpointenumraster versus producerboundslogic.','all50pair100frame_original_lineage_exact':True,'all12900_fixed_centres_windows_and_padding_counts_exact':True,'all50_before_after_grids_identical':True,'all302_original_cue_records_unchanged':True,'all77916_unique_combinations_replayed':replayed,'all604_tileIDlists_exact':True,'continuous_counts':dict(counts),'continuous_raster_mismatches':diffs,'all_input_and_GT_output_file_SHA_bindings_exact':True,'input_manifest':b(R/'inputs_metadata/INPUTS_GRID_ONLY.jsonl'),'input_seal':b(R/'inputs_metadata/INPUT_GRID_SEAL_BEFORE_GT_JOIN.json'),'GT_coverage_seal':b(croot/'GT_ONLY_COVERAGE_SEAL.json'),'no_GTbased_inputfilter_or_besttile_selection':True,'native8_or_human_gold_or_training_admission':False,'actual_RGB_feature_model_forward_fit_current_PnP':0}
 save(R/'METADATA_LINEAGE_AND_ALL_COVERAGE_QA.json',audit)
 final={'completed':True,'PASS_metadata_geometry_only':True,'protocol':b(R/'BEFORE_GRID_PROTOCOL.json'),'input_manifest':b(R/'inputs_metadata/INPUTS_GRID_ONLY.jsonl'),'input_seal_before_GT_join':b(R/'inputs_metadata/INPUT_GRID_SEAL_BEFORE_GT_JOIN.json'),'GT_coverage_seal':b(croot/'GT_ONLY_COVERAGE_SEAL.json'),'QA':b(R/'METADATA_LINEAGE_AND_ALL_COVERAGE_QA.json'),'input_generated_edgefixtures':b(R/'GENERATED_GRID_EDGE_FIXTURES.json'),'coverage_generated_edgefixtures':b(R/'GENERATED_CUE_COVERAGE_EDGE_FIXTURES.json'),'metadata_not_RGB_P3_tensors':True,'training_admission':False,'current_forward_fit_PnP':0}
 save(R/'FINAL_METADATA_GEOMETRY_SEAL.json',final)
 print(json.dumps({'QA':b(R/'METADATA_LINEAGE_AND_ALL_COVERAGE_QA.json'),'final_seal':b(R/'FINAL_METADATA_GEOMETRY_SEAL.json'),'all_combinations':replayed,'counts':dict(counts)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
