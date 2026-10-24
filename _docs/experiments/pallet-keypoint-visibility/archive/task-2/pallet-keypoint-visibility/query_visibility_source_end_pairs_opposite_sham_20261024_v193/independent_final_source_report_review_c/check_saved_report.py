"""Authorized V193 saved JSON/CSV/report/PNG-byte QA only; no array/image decode."""
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import csv, hashlib, json, os, re, stat, struct
V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_source_end_pairs_opposite_sham_20261024_v193')
G=V/'source_pair_visual_readout_v2_caption_fit';O=V/'independent_final_source_report_review_c'
def need(v,m):
 if not v:raise AssertionError(m)
def raw(path):
 p=Path(path);need(p.is_absolute() and '..'not in p.parts and all('v112'not in x.casefold()for x in p.parts) and V in p.parents,'exact authorized phase path')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for x in p.parts[1:-1]:
   n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'regular file')
 with os.fdopen(n,'rb')as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(raw(p)).hexdigest())
def boundbytes(b):
 r=raw(b['path']);need(hashlib.sha256(r).hexdigest()==b['sha256'],'SHA drift');return r
def load(p):return json.loads(raw(p))
def checked(b):return json.loads(boundbytes(b))
def rows(p):return list(csv.DictReader(raw(p).decode().splitlines()))
def alltrue(d):return all(alltrue(v)if isinstance(v,dict)else v is True for v in d.values())
scopepath=V/'ROOT_INDEPENDENT_FINAL_SOURCE_REPORT_QA_SCOPE.json';s=load(scopepath);need(s['approved_independent_saved_source_report_QA']is True,'root scope')
r=checked(s['root_results']);md=boundbytes(s['report']);c=checked(s['source_complete']);a=checked(s['independent_actual_source_audit']);g=checked(s['current_gallery']);repair=checked(s['caption_display_repair_QA'])
boundbytes(g['code']);checked(g['protocol']);cover=checked(g['coverage']);im=checked(g['images']);m=[json.loads(x)for x in boundbytes(c['manifest']).splitlines()]
S=rows(G/'ALL66_SOURCE_STATUS.csv');C=rows(G/'ALL22_SOURCE_CASES.csv');N=rows(G/'ALL1584_NATIVE_VARIANT_CENSUS.csv');checks={}
def ck(k,v):need(v,k);checks[k]=True
ck('all_scope_and_report_evidence_SHA_bindings',r['source_generation']==s['source_complete']and r['saved_visual_gallery']==s['current_gallery']and r['independent_all66_source_audit']==s['independent_actual_source_audit'])
ck('66_unique_attempts22_cases',len(m)==len(S)==66 and len({x['pair_id']for x in m})==66 and len({x['case_id']for x in m})==22)
ck('22_original_cases21_frames_SOURCE_DEVELOPMENT',len(C)==22 and len({x['case_id']for x in C})==22 and len({x['frame_index']for x in C})==21 and all(x['source_role']=='SOURCE_DEVELOPMENT'for x in C))
known=[x for x in m if x['PASS']];old=[x for x in m if x['predecessor_attempt']['PASS']];status=Counter(x['status']for x in m)
ck('27supported_21geometry_18UNKNOWN',status==Counter(SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED=27,GEOMETRY_FAILURE_RETAINED=21,UNKNOWN_RETAINED=18))
ck('old5_plusnew22_all27',len(old)==5 and all(x['PASS']for x in old)and len(known)==27 and sum(not x['predecessor_attempt']['PASS']for x in known)==22)
ck('known12_frames12_cases_old2frames',len({x['original_input']['original_flags']['frame_index']for x in known})==12 and len({x['case_id']for x in known})==12 and len({x['original_input']['original_flags']['frame_index']for x in old})==2)
ck('closed14_proof_on27_supported',all(len(x['closed14_actual_proof'])==15 and all(v is True for k,v in x['closed14_actual_proof'].items()if k!='evidence_bindings')for x in known))
ck('old5_all3_allarray_storedparity',all(alltrue(x['old5_decoded_packet_parity'])for x in old))
ck('all66_nativequalityownerUNKNOWN_admissionfalse',all(x['new_native_quality_owner_all8']==-1 and x['training_admission']is False for x in m))
ck('UNKNOWN18_no_packets_or_fabricated_RGB',all(not any(b.get('path')for b in x['variants'].values())for x in m if x['status']=='UNKNOWN_RETAINED'))
ck('geometry21_BEFOREonly_unknownweak',all([v for v,b in x['variants'].items()if b.get('path')]==['BEFORE']and set(x['direct_end_weak_states'].values())=={'UNKNOWN'}for x in m if x['status']=='GEOMETRY_FAILURE_RETAINED'))
ck('102_packets_by_manifest_no_body',sum(bool(b.get('path'))for x in m for b in x['variants'].values())==102)
byid={x['pair_id']:x for x in m};group={x:[]for x in byid}
for x in N:group[x['pair_id']].append(x)
ck('1584_census_each66_x3_x8',len(N)==1584 and all(len(rr)==24 and {(x['variant'],int(x['native_slot']))for x in rr}=={(v,k)for v in ('BEFORE','COVER','ADJACENT_SHAM')for k in range(8)}for rr in group.values()))
ck('saved816_inputU112_and_allstatusU207_distinct',sum(x['pixel_packet_present']=='True'for x in N)==816 and sum(x['pixel_packet_present']=='True'and x['original_input_available']=='False'for x in N)==112 and sum(x['original_input_available']=='False'for x in N)==207==cover['original_inputU_native_variant_rows'])
ck('96_missing_packets_768_missing_native_statuses',sum(x['pixel_packet_present']=='False'for x in N)==768)
ck('selected81_weak_D54_E27_other1503UNKNOWN',Counter(x['declared_direct_weak_state']for x in N)==Counter(UNKNOWN=1503,VISIBLE_WEAK=54,EXTERNALLY_BLOCKED_WEAK=27)and all(x['declared_direct_weak_state']=='UNKNOWN'or x['selected_weak_end_slot']=='True'for x in N))
ck('all_census_old_native_quality_owner_unknown',all(x['native_quality_owner_targets']=='ALL_MINUS1'and x['training_admission']=='False'for x in N))
ck('all66_status_csv_exact_join',all(x['case_id']==byid[x['pair_id']]['case_id']and int(x['selected_native'])==byid[x['pair_id']]['original_input']['native_slot']and(x['supported_weak_pair']=='True')==byid[x['pair_id']]['PASS']and(x['old_supported_pair']=='True')==byid[x['pair_id']]['predecessor_attempt']['PASS']and x['status']==byid[x['pair_id']]['status']for x in S))
native=[]
for k in range(8):
 rr=[x for x in m if x['original_input']['native_slot']==k]
 native.append(dict(native=k,original_attempts=len(rr),old_supported=sum(x['predecessor_attempt']['PASS']for x in rr),supported=sum(x['PASS']for x in rr),new_supported=sum(x['PASS']and not x['predecessor_attempt']['PASS']for x in rr),geometry_failure=sum(x['status']=='GEOMETRY_FAILURE_RETAINED'for x in rr),UNKNOWN=sum(x['status']=='UNKNOWN_RETAINED'for x in rr)))
ck('native8_table_exact_from66_manifest',native==r['by_selected_native'])
expected=dict(original_attempts=66,original_cases=22,original_unique_frames=21,old_supported_triplets=5,new_supported_triplets=22,all_supported_triplets=27,old_supported_unique_source_frames=2,supported_unique_source_frames=12,supported_unique_cases=12,selected_direct_weak_D_rows=54,selected_direct_weak_E_rows=27,geometry_failure_retained=21,semantic_or_sample_UNKNOWN_retained=18,actual_saved_packets=102,all8_saved_input_rows=816,saved_inputU_rows=112,attempted_variant_native_status_rows=1584,missing_variant_packets=96)
ck('all18_report_countfields_exact',r['counts']==expected)
ck('A_independent_all66_source_checks_PASS',a['completed']and a['PASS']and a['counts']['attempts66']==66 and a['counts']['actual_allnative_input_rows']==816 and a['counts']['actual_original_inputU_rows']==112 and a['stored_status_counts']==dict(status)and all(all(x['checks'].values())for x in a['independent_all66_checks']))
ck('source_only_nofit_noadoption_active_unsolved',r['goal_status']=='active_unsolved'and r['training_admission']is False and r['new_training_or_model_inference_or_real_evaluation']is False and r['adopted']is False and r['fully_solved']is False and r['selected_model']is None)
block=[];tables=[]
for l in md.decode().splitlines()+['']:
 if l.startswith('|'):block.append(l)
 elif block:tables.append(block);block=[]
nt=[[int(v.strip())for v in l.strip('|').split('|')]for l in tables[1][2:]]
ck('Markdown_native_table_all8_exact',nt==[[x[k]for k in ['native','original_attempts','old_supported','supported','new_supported','geometry_failure','UNKNOWN']]for x in native])
class Href(HTMLParser):
 def __init__(self):super().__init__();self.targets=[]
 def handle_starttag(self,t,attrs):self.targets.extend(v for k,v in attrs if k in {'href','src'}and v)
def ordinary(p):
 need(V==p or V in p.parents,'link inside phase');need(all('v112'not in x.casefold()for x in p.parts),'safe link');q=Path(p.anchor)
 for x in p.parts[1:]:q/=x;mode=q.lstat().st_mode;need(not stat.S_ISLNK(mode),'no symlink')
 return stat.S_ISREG(mode)or stat.S_ISDIR(mode)
def links(p,html=False):
 text=raw(p).decode();h=Href();h.feed(text);ts=h.targets if html else re.findall(r'!?\[[^\]\n]*\]\(([^\s)]+)\)',text);out=[]
 for t in ts:
  u=urlsplit(t)
  if u.scheme:need(u.scheme in {'https','http','data','mailto'},'URI scheme');continue
  value=unquote(u.path)
  if not value:continue
  need(not value.startswith('/'),'relative link');target=Path(os.path.normpath(str(p.parent/value)));need(ordinary(target),'existing target');out.append(t)
 return dict(document=str(p),count=len(out),relative_references=out)
linkresults=[links(p,h)for p,h in [(V/'PORTABLE_REPORT_KO.md',False),(V/'ROOT_REPORT_KO.md',False),(V/'ROOT_READOUT_KO.html',True),(G/'PORTABLE_GALLERY_KO.md',False),(G/'index.html',True)]]
ck('root_MD_and_PORTABLE_byteidentical',raw(V/'ROOT_REPORT_KO.md')==md);checks['all5_report_gallery_pages_relative_links_exist']=True
allimages=im['cards']+im['clean_valid_and_context_PNG'];need(len({x['path']for x in allimages})==len(allimages),'unique generated images')
for b in allimages:need(G in Path(b['path']).parents and Path(b['path']).suffix=='.png','generatedPNG only');boundbytes(b)
ck('1791_generated_PNG_hashes_all_bound',len(im['cards'])==132 and len(im['clean_valid_and_context_PNG'])==1659)
G1=V/'source_pair_visual_readout_v1';oldim=load(G1/'IMAGE_MANIFEST.json');oldmap={str(Path(x['path']).relative_to(G1)):x['sha256']for x in oldim['clean_valid_and_context_PNG']};newmap={str(Path(x['path']).relative_to(G)):x['sha256']for x in im['clean_valid_and_context_PNG']}
ck('captionrepair1659_clean_valid_context_hashes_unchanged',oldmap==newmap and len(newmap)==1659)
ck('captionrepair3CSV_coverage_MD_HTML_byteidentical',all(raw(G1/n)==raw(G/n)for n in ['ALL66_SOURCE_STATUS.csv','ALL22_SOURCE_CASES.csv','ALL1584_NATIVE_VARIANT_CENSUS.csv','SOURCE_COVERAGE_KO.json','PORTABLE_GALLERY_KO.md','index.html']))
ck('root_captionrepair_receipt_preserves_v1',repair['CSV3_coverage_MD_HTML_byteidentical']is True and repair['all_clean_valid_context_png_hashes_identical']==1659 and repair['all_v1_outputs_preserved']is True and repair['source_weak_labels_or_model_or_fit_changes']==0)
views=[('NEW_native1','Q0033_D2/SELECTED_END_TRIPLET.png','Red donor visible in both COVER/SHAM central32; captions fit; local endpoint semantic identity is not newly adjudicated.'),('NEW_native4','Q0052_D1/SELECTED_END_TRIPLET.png','Donor shifted to opposite positions and present in both central32; q marker is not endpoint truth.'),('OLD_native0','Q1088_D2/SELECTED_END_TRIPLET.png','Original red-can contrast readable; retained weak captions distinguish COVER/SHAM, not physical/native gold.'),('semanticUNKNOWN','Q0057_D1/SELECTED_END_TRIPLET.png','All3 variants show no saved RGB/not constructed with explicit UNKNOWN reason; no fabricated image.'),('geometryfailure','Q0024_D1/CONTACT_ALL8.png','All8 BEFORE rows and native3 inputU padding visible; missing COVER/SHAM have reason-only text/UNKNOWN. Readable at documented resize.'),('inline_native5','Q0277_D2/SELECTED_END_TRIPLET.png','Blue-can COVER changes central32 while shifted SHAM central32 resembles BEFORE; image contrast alone does not certify endpoint/native accuracy or transfer.')]
vr=[]
for cat,rel,obs in views:
 p=G/rel;dims=list(struct.unpack('>II',raw(p)[16:24]));vr.append(dict(category=cat,image=bind(p),tool='tools.view_image via functions.exec',actual_calls=1,detail='high',original_dimensions=dims,display_dimensions=[1131,2048]if rel.endswith('CONTACT_ALL8.png')else dims,observation=obs))
review=dict(completed=True,PASS=True,review_type='INDEPENDENT_FINAL_SOURCE_REPORT_NUMERIC_LINK_AND_GENERATED_VISUAL_QA',scope=bind(scopepath),code=bind(__file__),root_results=s['root_results'],report=s['report'],source_complete=s['source_complete'],source_manifest=c['manifest'],independent_source_audit=s['independent_actual_source_audit'],gallery=s['current_gallery'],caption_repair=s['caption_display_repair_QA'],checks=checks,counts=expected,native8=native,relative_links=linkresults,generated_PNG_byte_bindings_checked=len(allimages),caption_clean_valid_context_parity_count=1659,actual_visual_tool_calls=6,actual_views=vr,manual_interpretation=['27triplets/81selected weak labels use12sourceframes; all22cases21frames preserved.','Originalq, finiteend and central32 presence differ. BothCOVER/SHAM may contain donor centrally; proof does not guarantee readable semantic identity or transfer.','Source assistantweak endpoint+2Dcomposite controls are not natural occlusion or physical/native gold; other7 native/owner/quality remain UNKNOWN.','No new classifierfit/inference/real eval/threshold or model choice; trainingadmissionfalse and active_unsolved.'],preserved_own_helper_failure=bind(O/'FIRST_QA_HELPER_CODE_AS_JSON_FAILURE_PRESERVED.json'),failure_resolution='SHA-only validation for Python code; JSON decode only for JSON metadata. Scientific files/pixels unchanged. Initial absent ROOT_PHASE read18254d was normal parent-confirmed ordering: root creates phase after this QA.',limits=['C actually viewed6 generated sheets, not all66 contacts or all1584 rows visually.','All1791 generated PNG bytes hashed; no additional image/sourcearray/alpha decoding.','A bound numeric PASS supplies independent sourcepixel proof; C did not rerun generator/model/sampler.','HTML links checked as source; no browser rendering claim.'],actual_model_sourcepacket_alpha_or_eval_reads=0,scientific_outputs_modified=False,HTML_browser_rendered=False,selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved',phase_write_closure='After this QA no further scientific phase writes by C.')
with(O/'REVIEW.json').open('x')as f:json.dump(review,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(O/'REVIEW.json'),checks=len(checks),relative_link_counts=[dict(document=x['document'],count=x['count'])for x in linkresults],generated_PNGs=len(allimages),actual_tool_views=6),ensure_ascii=False))
