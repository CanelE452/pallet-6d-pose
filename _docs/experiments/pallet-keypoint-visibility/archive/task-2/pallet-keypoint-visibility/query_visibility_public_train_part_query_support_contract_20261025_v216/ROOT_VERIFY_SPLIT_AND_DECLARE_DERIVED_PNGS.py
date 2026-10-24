"""Independent SOURCE metadata checks; declare PNG bindings without opening images."""
from pathlib import Path
import collections,datetime,hashlib,json
V=Path(__file__).parent;P=V.parent
PREV=P/'query_visibility_public_train_visible_part_coverage_expansion_20261025_v215'
def read(p,sha):
 assert p.is_absolute()and'v112'not in str(p).casefold()
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha
 return json.loads(b)
def new(name,z):
 p=V/name
 with p.open('x')as f:json.dump(z,f,ensure_ascii=False,indent=2);f.write('\n')
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 s=read(V/'component_split_preparation_b/INTERNAL_SOURCE_FIT_DEV_COMPONENT_SPLIT.json','1ee8271189e82c72a89461f880a5c4464ce474035ba12189ddaf175fce407ce9')
 m=read(PREV/'fixed12_public_TRAIN_input_previews_v1/ANNOTATOR_INPUT_MANIFEST.json','d7982e3c61c85c82c56d630a8316b459504fbb99ebeb957f155f29cc43b13d18')
 by=collections.defaultdict(list);a={f['position']:f for f in s['frames']}
 for f in m['frames']:by[f['prefix']].append(f);assert f['role']=='TRAIN'
 for prefix,frames in by.items():
  ranked=sorted(frames,key=lambda f:(hashlib.sha256((s['seed']+f['component_id']).encode()).hexdigest(),f['component_id'],f['frame_key']))
  assert len(ranked)==3
  for rank,f in enumerate(ranked):
   z=a[f['position']]
   assert z['internal_source_role']==('SOURCE_DEV'if rank==0 else'SOURCE_FIT')and z['rank_within_prefix']==rank and z['component_id']==f['component_id']and z['original_role']=='TRAIN'
 original={r['row_id']:r for r in m['slots96']};counts=collections.Counter();u=collections.Counter()
 assert len(s['rows96'])==len(original)==96
 for r in s['rows96']:
  o=original[r['row_id']]
  for k in ['position','frame_key','native_slot','query_xy','L','original_query_available','original_flags','original_global_index','component_id','clean96','valid96']:assert r[k]==o[k],k
  assert r['role']==o['role']=='TRAIN'and r['internal_source_role']==a[o['position']]['internal_source_role']
  counts[r['internal_source_role']]+=1;u[r['internal_source_role']]+=not r['original_query_available']
 assert counts=={'SOURCE_FIT':64,'SOURCE_DEV':32}and u=={'SOURCE_FIT':9,'SOURCE_DEV':2}
 wh=[]
 for f in m['frames']:
  for k in ['original_fullRGB','display_original_q8','all8contact']:wh.append(dict(kind=k,position=f['position'],binding=f[k]))
 for r in m['slots96']:
  for k in ['clean96','valid96']:wh.append(dict(kind=k,position=r['position'],native_slot=r['native_slot'],binding=r[k]))
 assert len(wh)==228 and len({x['binding']['path']for x in wh})==228
 for rec in wh:
  path=Path(rec['binding']['path']);assert PREV in path.parents and path.suffix=='.png'and'v112'not in str(path).casefold()
 failure=new('ROOT_METADATA_QA_FIELD_ASSUMPTION_FAILURE_AND_FIX.json',dict(completed=True,first_inline_attempt_terminal_chunk='f14a51',exit_code=1,error="KeyError: original_role",cause='Frame grouping uses original_role; original per-slot input rows preserve role. The root checker incorrectly assumed the frame key at row level.',repair='Use r.role==original.role==TRAIN at row level. No producer artifacts changed.',before_failure_source_image_or_model_opens=0,before_failure_output_files_created=0,bounded_metadata_rerun_only=True))
 q=dict(completed=True,PASS=True,code=dict(path=str(Path(__file__).absolute()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),split_ranking_and_all96_original_ID_q_L_native_flags_exact=True,counts=dict(counts),U=dict(u),no_source_image_array_GT_model_score_opens=True,images_declared_not_opened=len(wh),split_is_not_freshblind=True,first_checker_field_error_preserved=failure,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
 qa=new('ROOT_INDEPENDENT_COMPONENT_SPLIT_METADATA_QA.json',q)
 w=new('ROOT_DECLARED228_DERIVED_PUBLIC_PNG_BINDINGS_BEFORE_NEW_ANNOTATION.json',dict(completed=True,metadata_declaration_only=True,actual_image_views_approved=False,images=wh,array_or_model_binding_followed=0,source_image_body_reads=0))
 print(json.dumps(dict(PASS=True,metadata_QA=qa,declared_image_whitelist=w)))
if __name__=='__main__':main()
