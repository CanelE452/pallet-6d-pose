"""V204 closed source report numeric/provenance/link check; no arrays/images decode."""
from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
import json,hashlib,re,os,stat,csv,io
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent;n=0

def need(v,m):
 global n
 if not v:raise AssertionError(m)
 n+=1

def raw(p):
 p=Path(p).absolute();need(P in p.parents and '..'not in p.parts and all('v112'not in s.casefold()for s in p.parts),'Finite path before access');fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for s in p.parts[1:-1]:z=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=z
  z=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(z,'rb')as f:need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'Regular no symlink');return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(raw(p)).hexdigest())
def js(p):return json.loads(raw(p))
def bound(b):z=raw(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Frozen artifact hash');return z
class Refs(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,t,a):self.links +=[v for k,v in a if k in('href','src')and v]
build=js(V/'ROOT_REPORT_BUILD_COMPLETE.json');need(build['completed']and build['PASS'],'Root report build complete');bound(build['code'])
for b in build['outputs'].values():bound(b)
r=js(V/'ROOT_RESULTS_KO.json');need(r['completed']and r['PASS']and r['source_TRAIN_only']and not r['goal_solved']and not r['new_model_forward_or_fit']and not r['newlabels']and not r['label_error_or_transfer_cause_proved'],'Source-only factual claims')
for b in r['proofs'].values():need(json.loads(bound(b))['PASS'],'A/B/C closed proofs')
a=js(V/'paired_pixel_alignment_a/actual_train_pixel_alignment_v1/SUMMARY.json');ind=js(V/'independent_pixel_diagnostic_review_c/NUMERICAL_QA.json');c=js(V/'label_loss_contract_c/TRAIN_METADATA_LABEL_LOSS_CENSUS.json');b=js(V/'public_source_coverage_b/SUMMARY.json')
need(ind['PASS']and r['window_threshold1']==[x for x in a['window_summary']if x['threshold']==1],'Root all30 threshold1 rows equal independently audited savedsummary')
need((r['original_TRAIN_rows'],r['known_TRAIN_triplets'],r['known_D'],r['known_E'],r['UNKNOWN'],r['input_U'])==(2784,109,218,109,2457,112),'TRAIN denominators exact')
need(r['known_D']==c['TRAIN_D218']and r['known_E']==c['TRAIN_E109'],'Label contract census exact')
need((r['public_original_TRAIN_frames'],r['public_direct_TRAIN_frames'],r['public_direct_TRAIN_triplets'],r['public_direct_TRAIN_components'])==(b['public_real']['original_frames'],b['direct_weak_TRAIN']['real_source_frames'],b['direct_weak_TRAIN']['real_triplets'],b['direct_weak_TRAIN']['real_component_count']),'Public TRAIN coverage exact')
md=raw(V/'PORTABLE_REPORT_KO.md').decode();need(hashlib.sha256(md.encode()).hexdigest()=='a47d80479bd0e26d1c660ffe72df84eada5c327fceac8fc996444e0399e0753c'and raw(V/'ROOT_REPORT_KO.md')==md.encode(),'Root/portable fixed bytes')
blocks=re.findall(r'(?m)(^\|.*(?:\n\|.*)+)',md);rows=[x.split('|')[1:-1]for x in blocks[0].splitlines()[2:]]
expected=[]
for d in ('SYNTH_FINITE_END','REAL_SEMANTIC_END_WEAK'):
 for v in('COVER','SHAM'):
  ss=[x for x in a['window_summary']if x['domain']==d and x['variant']==v and x['threshold']==1];expected.append([d,v,str(ss[0]['analyzed_TRAIN_triplets'])]+[str(x['triplets_with_changed_query_window'])for x in ss])
need(rows==expected,'Rendered four central109 table rows exact')
rr=[list(csv.DictReader(io.StringIO(raw(V/'paired_pixel_alignment_a/actual_train_pixel_alignment_v1/PAIR_WHOLE96_THRESHOLDS.csv').decode())))][0]
for d,lo,hi in [('SYNTH_FINITE_END',36,100),('REAL_SEMANTIC_END_WEAK',283,2633)]:
 z=[int(x['changed_pixels'])for x in rr if x['domain']==d and x['variant']=='COVER'and x['absRGB_max_channel_threshold']=='1'];need((min(z),max(z))==(lo,hi),'Rendered whole changed pixel ranges')
need(all(int(x['changed_pixels'])>0for x in rr if x['absRGB_max_channel_threshold']=='1'),'Whole96 bothvariants109 all changed')
root=js(V/'ROOT_ACTUAL_FIXED2_SOURCE_VISUAL_QA.json');av=js(V/'paired_pixel_alignment_a/ACTUAL_FIXED2_GENERATED_CARD_VISUAL_QA.json');cv=js(V/'independent_pixel_diagnostic_review_c/ACTUAL_FIXED3_CARD_VISUAL_QA.json')
rootset={x['path']for x in root['cards']};aset={x['path']for x in av['actual_viewed_cards']};cset={x['card']['path']for x in cv['viewed_cards']};need(rootset==aset and len(aset)==2 and len(cset)==3 and not aset&cset and len(aset|cset)==5,'Actual view A2=root2 plus C3 means five unique only')
# Publisher metadata counts only: do not follow any dataset file path.
ext=V/'external_primary_sources_root';meta=js(ext/'MR6D_REPOSITORY_API_METADATA.json');need(meta['sha']=='3c4ed8453312655499c3d3a3cf38b4929fae77ee'and meta['id']=='anas-gouda/mr6d','Existing MR6D same repo revision');need(Counter(x['rfilename'].split('/')[0]for x in meta['siblings']if x['rfilename'].endswith('/scene_camera.json'))==dict(dynamic=40,mr=14,odyn=19,val=16),'Official index directory counts distinct from paper92')
ku=js(ext/'DJLDFS_DATASET_API_METADATA.json')['data']['latestVersion'];counts=Counter(x.get('directoryLabel','').split('/')[-1]for x in ku['files']);need((counts['rgb'],counts['depth'],counts['masks'],counts['annotated_poses'])==(322,322,290,284),'API modality directory counts 322/322/290/284')
old=raw(ext/'ATTRIBUTION_AND_READOUT_KO_V1_COUNT_CORRECTION_SOURCE.md');note=raw(ext/'ATTRIBUTION_AND_READOUT_KO.md');need(hashlib.sha256(old).hexdigest()=='20d65791103c4339b7f6ca5c3b2d84226adf017ff50490bfb57bdcd7fabcb53b'and note==old.replace(b'RGB335',b'RGB322'),'Preserved first count error + one-token correction only');need(hashlib.sha256(note).hexdigest()=='b6e4ade396a4b8def3e42011e3be35d39d61fed427141828778d36a35226fbfe','Latest corrected attribution exact')
links={}
for rel in ['ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','paired_pixel_alignment_a/KO.md','paired_pixel_alignment_a/actual_train_pixel_alignment_v1/PORTABLE_GALLERY_KO.md','paired_pixel_alignment_a/actual_train_pixel_alignment_v1/index.html','public_source_coverage_b/KO.md','label_loss_contract_c/KO.md','independent_pixel_diagnostic_review_c/KO.md','external_primary_sources_root/ATTRIBUTION_AND_READOUT_KO.md']:
 p=V/rel;t=raw(p).decode();parser=Refs();parser.feed(t)if rel.endswith('.html')else None;refs=parser.links if rel.endswith('.html')else re.findall(r'\]\(([^)]+)\)',t);local=0
 for ref in refs:
  if ref.startswith(('https://','http://','#')):continue
  need(not ref.startswith(('/','//'))and 'v112'not in ref.casefold(),'Relative safe local link');q=Path(os.path.normpath(p.parent/ref));need(q.suffix.lower()not in('.npz','.npy','.pt','.pth'),'No array/model follow');raw(q);local+=1
 links[rel]=local
review=dict(completed=True,PASS=True,code=bind(__file__),root_report_build=bind(V/'ROOT_REPORT_BUILD_COMPLETE.json'),root_results=bind(V/'ROOT_RESULTS_KO.json'),root_portable=bind(V/'PORTABLE_REPORT_KO.md'),root_HTML=bind(V/'ROOT_READOUT_KO.html'),latest_attribution=bind(ext/'ATTRIBUTION_AND_READOUT_KO.md'),correction_record=bind(ext/'ROOT_PALLOX_METADATA_COUNT_CORRECTION.json'),closed_C_numeric=bind(V/'independent_pixel_diagnostic_review_c/REVIEW.json'),closed_C_contract=bind(V/'label_loss_contract_c/REVIEW.json'),actual_views_A2_root_same2_C3_five_unique_only=True,new_image_views_during_final_QA=0,finite_checks=n,local_link_counts=links,first_count_finding_preserved=dict(original_note_sha='20d65791103c4339b7f6ca5c3b2d84226adf017ff50490bfb57bdcd7fabcb53b',incorrect_RGB=335,metadata_directory_RGB=322,source_API_unchanged=True),source_facts_semantics='Querycenter unchanged does not prove weak finite-end label wrong. Separate broad BOTH/finite E is not a forced sameoutput contradiction. Gradient conflict/render shortcut/real cause unproven. New MR6D dataset claim corrected; public natural weak observation is future pilot, no admission/newlabel/fit here.',original_arrays_targets_pixels_weights_model_protected_eval_or_new_download_reads=0,scientific_existing_files_modified=0,selected_model=None,adopted=False,goal_solved=False,limits=['Numeric verification reuses closed independent TRAIN scalar QA and B source metadata audit; no original RGB/target computations repeated.','External sources checked only saved API metadata, model-name/bounds metadata and prose; no image/GT/model payload follows.','No browser rendering or new image view. Human D/E/S authority remains valid; no mesh certification blocker.'])
with(H/'REVIEW.json').open('x')as f:json.dump(review,f,ensure_ascii=False,indent=2);f.write('\n')
with(H/'KO.md').open('x')as f:f.write('V204 최종 사실·수치·링크 QA PASS. TRAIN109 창표와 2784/known327/UNKNOWN2457/U112, B의 공개TRAIN155 및 감독단위, oldloss 의미차이를 닫힌 근거와 대조했다. 실제뷰는 A2=root동일2+C다른3으로 고유5개다. 최초출처노트의 Pallox RGB335는 API의 rgb디렉터리322와 달랐고 root가 원문을 보존한 후322로 한곳만 정정했다. q중심불변을 라벨오류로 해석하지 않으며 새학습/새영상도입/실사개선/원인확정 주장은 없다.\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),checks=n,local_links=links),ensure_ascii=False))
