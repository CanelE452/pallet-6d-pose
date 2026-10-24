"""공개 SOURCE metadata만 재조합. PNG/NPZ/GT/model/평가/배포파일 접근 없음."""
import ast,csv,hashlib,io,json,math,os,stat
from collections import Counter
from pathlib import Path
H=Path(__file__).absolute().parent;R=H.parent;P=R.parent
V=P/'query_visibility_public_train_intended_part_query_targets_20261025_v218'
W=P/'query_visibility_expected_part_weak_paired_source_training_20261025_v219'
INPUTS={
 'canonical360':(V/'joined_weak_intended_part_targets_v1/ALL360_CHECKED_WEAK_TARGET_CANDIDATES.jsonl','52ac5bb8d0fb19a1329b9bdb7f7b56964fed954a2a4d50489e3dcb7d598d139d'),
 'nominal120':(V/'joined_weak_intended_part_targets_v1/ALL120_JOINED_NOMINAL_OBSERVATIONS.jsonl','7c7695fba4a65bbe0362a9276da1a2f7e40247205544bae583d0d0eb4059e197'),
 'pair_complete':(V/'weak_part_pair_controls_v1/COMPLETE.json','36153d428ef246428c839923ac8f7ba74fa31c23bf3dc2def5f67f6f7a7cfda5'),
 'attempts9':(V/'weak_part_pair_controls_v1/ALL9_ATTEMPTS.json','b126e8eedd1fc17b7f82bbf08eb70cd7f8bfaa7405300b71666a1d1d509e9d78'),
 'variants27':(V/'weak_part_pair_controls_v1/ALL27_ATTEMPTED_VARIANT_METADATA.jsonl','933d02e4e7168244a3ac876df0c858f965ab9f69e0919501f0867c8b5b217d3a'),
 'proofs9':(V/'weak_part_pair_controls_v1/ALL9_ACTUAL_PIXEL_ALPHA_PROOFS.json','d9696152cf2f2db1481cf00d2e8364950a74b01fdf782a9bf2aa0a717c7f588c'),
 'targets9':(V/'intended_part_targets_a/ALL9_EXPLICIT_FINITE_LOCAL_PART_TARGETS.json','51844ddd5c142206a93a0c68c2880186a8a7d45507c29ec838ccaeb7fc5aa9bc'),
 'pair_plan':(V/'weak_part_pair_preparation_a/FIXED9_PAIR_INPUTS_AND_EXACT18_PNG_BINDINGS_BEFORE_PIXELS.json','96eee6f533cac1833d0684a93ef6569a6f5398c383f9861805b4347f8900731a'),
 'pair_code':(V/'weak_part_pair_preparation_a/build_weak_part_pairs.py','9f8df828320f01f1e8edc95a5af5c6d1e4c3e9d75902cc77da1314391ecbacb9'),
 'V219_fit_complete':(W/'paired_source_fit_v1/COMPLETE.json','e6c820488e111049056384bce482a2d1bfb9c9b964ea285da5093d8d0d941f8f'),
 'V219_actual_entry':(W/'ROOT_ACTUAL_FIXED_SOURCE_PAIR_FIT_ENTRY.json','91035379fa50fed323d96219b6b98ed2a9ad830a9668b811686bcb40e6067028'),
 'V219_weak_contract':(R/'V219_TRAINING_PREPARATION_C/source_training_contract.py','fc0a161e8a4c1a413d528ce1e66eed1b2addf2d4593c632506280f10732451d7')}

def rb(p):
 s=str(p);assert'v112'not in s.casefold()and p.is_absolute()and'..'not in p.parts and p.suffix in{'.json','.jsonl','.py'}
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 assert stat.S_ISREG(os.fstat(n).st_mode)
 with os.fdopen(n,'rb')as f:return f.read()

def enc(x):return(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def save(name,x):
 b=x.encode()if isinstance(x,str)else enc(x);p=H/name
 with p.open('xb')as f:f.write(b)
 return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))
def csvsave(name,rows):
 s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader()
 for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':'))if isinstance(v,(dict,list))else v for k,v in r.items()})
 return save(name,s.getvalue())
def unpack_rle(rle):
 seen=set();last=0
 for start,length in rle:
  assert type(start)is int and type(length)is int and start>=last and length>0 and start+length<=9216
  seen.update(range(start,start+length));last=start+length
 return seen
def rle(ids):
 out=[]
 for i in sorted(ids):
  if out and i==out[-1][0]+out[-1][1]:out[-1][1]+=1
  else:out.append([i,1])
 return out
def boxmask(box):
 assert len(box)==4 and all(math.isfinite(x)and x==int(x)for x in box)
 x0,y0,x1,y1=map(int,box);assert 0<=x0<=x1<=95 and 0<=y0<=y1<=95
 return {y*96+x for y in range(y0,y1+1)for x in range(x0,x1+1)}
def bbox(ids):
 return [min(i%96 for i in ids),min(i//96 for i in ids),max(i%96 for i in ids),max(i//96 for i in ids)]if ids else None
def role_summary(rows,rolekey,framekey,componentkey):
 return {role:dict(rows=len(rr),frames=len({r[framekey]for r in rr}),components=len({r[componentkey]for r in rr}))for role in sorted({r[rolekey]for r in rows})if(rr:=[r for r in rows if r[rolekey]==role])}

def main():
 save('FINAL_COMPUTATION_INPUT_WHITELIST_AND_SCOPE.json',dict(completed=True,source_only=True,authority='Root V221 metadata readiness messages including D48 exactOR and originalquery9 request',inputs={k:dict(path=str(p),sha256=sha)for k,(p,sha)in INPUTS.items()},binary_paths_in_records_NEVER_followed=True,new_labels_controls_training_or_admission=0,pixels_arrays_GT_model_evaluation_Git=0))
 docs={};bindings={}
 for k,(p,sha)in INPUTS.items():
  b=rb(p);assert hashlib.sha256(b).hexdigest()==sha;bindings[k]=dict(path=str(p),sha256=sha,size_bytes=len(b))
  docs[k]=b.decode()if p.suffix=='.py'else[json.loads(z)for z in b.splitlines()]if p.suffix=='.jsonl'else json.loads(b)
 allrows=docs['canonical360'];nom={r['nominal_row_id']:r for r in docs['nominal120']};assert len(allrows)==360 and len(nom)==120
 assert docs['V219_actual_entry']['weak_rows360']=={k:bindings['canonical360'][k]for k in('path','sha256')}
 assert docs['V219_fit_complete']['entry']['sha256']==bindings['V219_actual_entry']['sha256']
 assert "fit=known and r['source_role']=='SOURCE_FIT'"in docs['V219_weak_contract']
 assert "known=r['weak_loss_candidate'] is True"in docs['V219_weak_contract']
 used=[r for r in allrows if r['weak_loss_candidate']and r['source_role']=='SOURCE_FIT'];assert len(used)==45 and Counter(r['weak_target_E_binary']for r in used)=={0:39,1:6}
 assert all(r['weak_loss_fit_candidate']for r in used)
 d48=[r for r in allrows if r['target_state']=='D_VISIBLE_EXPECTED_PART_WEAK'and r['weak_loss_candidate']];assert len(d48)==48
 d19=[r for r in d48 if r['generated_input']['normalized_offset']==[0,0]];assert len(d19)==19
 dtable=[]
 for r in d48:
  g=r['generated_input'];o=g['original_input'];nr=nom[r['nominal_row_id']];boxes={k:v['transformed_roi96']for k,v in r['support_checks']['evidence_boxes'].items()};assert set(boxes)=={'A','C'}
  union=boxmask(boxes['A'])|boxmask(boxes['C']);valid=unpack_rle(g['valid96_RLE']);assert union<=valid
  delta=[g['generated_query_xy'][i]-o['query_xy'][i]for i in range(2)];L=g['L'];assert L==o['L']and o['original_role']=='TRAIN'and g['q8_origin']=='GT_DERIVED_TRAINING_INITIALIZATION'
  dtable.append(dict(row_id=r['row_id'],nominal_row_id=r['nominal_row_id'],source_role=r['source_role'],frame_position=g['frame_position'],candidate_gt_list_index=g['candidate_gt_list_index'],component_id=g['component_id'],native_slot=g['native_slot'],canonical_index=g['canonical_index'],anatomical_family=g['anatomical_family'],original_row_id=o['row_id'],original_role=o['original_role'],original_query_xy=o['query_xy'],generated_query_xy=g['generated_query_xy'],L=L,query_delta_raw96=[x*96/L for x in delta],normalized_offset=g['normalized_offset'],original_query_available=o['original_query_available'],generated_query_available=g['generated_query_available'],query_family=g['family'],q8_origin=g['q8_origin'],ROI96_by_observer=boxes,exact_union_RLE_rowmajor96=rle(union),exact_union_pixels=len(union),union_bbox_DISPLAY_ONLY=bbox(union),valid96_RLE=g['valid96_RLE'],valid96_pixels=len(valid),all_union_pixels_strict_valid=True,all_support_checks=r['support_checks'],part_evidence_A=nr['A']['part_identity_evidence'],part_evidence_C=nr['C']['part_identity_evidence'],RGB96_binding=g['RGB96'],source_context_binding=g['source_RGB_binding'],condition32=r['deployable_condition32'],original_detector_to_GT_instance='UNKNOWN',V219_FIT_pool_member=r['weak_loss_fit_candidate'],native_point_gold=False,new_counterfactual_constructed=False))
 targets={t['target_id']:t for t in docs['targets9']['targets9']};proofs={tuple(t['case_key']):t for t in docs['proofs9']['proofs9']};plans={tuple(t['case_key']):t for t in docs['pair_plan']['cases9']};pairs=[]
 palette=next(n for n in ast.parse(docs['pair_code']).body if isinstance(n,ast.Assign)and any(isinstance(z,ast.Name)and z.id=='PALETTE'for z in n.targets));palette_code=ast.unparse(palette)
 for a in docs['attempts9']['attempts9']:
  t=targets[a['target_id']];o=a['input'];pr=proofs[tuple(a['case_key'])]['proof'];plan=plans[tuple(a['case_key'])];support=a['original_target_support'];ids={y*96+x for y,x0,x1 in support['positive_runs_y_x0_x1']for x in range(x0,x1+1)}
  assert len(ids)==support['positive_pixels']and ids==(boxmask(support['original_ROIs_preserved'][0])|boxmask(support['original_ROIs_preserved'][1]))
  assert a['images']['BEFORE']['sha256']==t['clean96_binding']['sha256']==plan['clean96']['sha256']
  assert o['role']=='TRAIN'and a['training_admission']is False and t['native_DE_target']==-1
  assert pr['COVER_target_whole_OR_opaque']and pr['SHAM_alpha_disjoint_target_OR']and pr['SHAM_target_RGB_byte_unchanged']and pr['RGB_pattern_exact_same']and pr['alpha_same_mass_pattern_translation']
  pairs.append(dict(target_id=a['target_id'],case_key=a['case_key'],original_row_id=o['row_id'],source_role=o['internal_source_role'],frame_position=o['position'],component_id=o['component_id'],native_slot=o['native_slot'],query_xy=o['query_xy'],L=o['L'],query_available=o['original_query_available'],query_family='ORIGINAL_INITIALIZER_QUERY_UNCHANGED',ROI96_by_observer=dict(A=support['original_ROIs_preserved'][0],C=support['original_ROIs_preserved'][1]),exact_union_RLE_rowmajor96=rle(ids),exact_union_pixels=len(ids),union_bbox_DISPLAY_ONLY=support['union_bbox'],valid96_binding=plan['valid96'],valid96_decoded_bool_SHA256_declared=plan['valid96_decoded_bool_sha256'],validity_proof_from_saved_metadata=pr['both_alpha_all_original_valid'],semantic_alignment=t['expected_native_part_correspondence'],feature_A=t['observed_witness_A']['description'],feature_C=t['observed_witness_C']['description'],query_support_A=t['query_support_A'],query_support_C=t['query_support_C'],image_bindings=a['images'],source_context_binding=t['original_context_binding'],occluder_provenance=dict(type='SYNTHETIC_POST_RAW96_OPAQUE_CHECKER_PATTERN_NOT_SCANNED_REAL_DONOR',palette_code=palette_code,checker_cell=4,pattern_anchor='target union bbox minx/miny; same RGB pattern translated to SHAM',SHAM_translation_xy=pr['SHAM_translation_xy'],placement='minimum(L1_translation,shifted_min_y,shifted_min_x,dx,dy), all original-valid and target-disjoint',depth_or_physical_owner='NOT_INFERRED'),saved_pixel_proof=pr,finite_part_targets=dict(BEFORE=0,COVER=1,SHAM=0),native_DE_target=-1,query_support_from_variant='NOT_INFERRED',training_admission=False))
 assert len(pairs)==9
 usedrows=[]
 for r in used:
  g=r['generated_input'];nr=nom[r['nominal_row_id']];usedrows.append(dict(row_id=r['row_id'],nominal_row_id=r['nominal_row_id'],target=r['weak_target_E_binary'],source_role=r['source_role'],frame_position=g['frame_position'],component_id=g['component_id'],native_slot=g['native_slot'],original_row_id=g['original_input']['row_id'],normalized_offset=g['normalized_offset'],RGB96_binding=g['RGB96'],natural_foreign_evidence_A=nr['A'].get('foreign_evidence'),natural_foreign_evidence_C=nr['C'].get('foreign_evidence'),occluder_A=nr['A'].get('occluder'),occluder_C=nr['C'].get('occluder')))
 pair_hashes={b['sha256']for r in pairs for k,b in r['image_bindings'].items()if k in('BEFORE','COVER','SHAM')};used_hashes={r['RGB96_binding']['sha256']for r in usedrows}
 underlying_overlap=sorted({r['original_row_id']for r in pairs if r['source_role']=='SOURCE_FIT'}&{r['original_row_id']for r in usedrows})
 summary=dict(nominal_D19=role_summary([r for r in dtable if r['normalized_offset']==[0,0]],'source_role','frame_position','component_id'),D48_variants=role_summary(dtable,'source_role','frame_position','component_id'),old_COVER9=role_summary(pairs,'source_role','frame_position','component_id'),old_COVER27_roles=dict(Counter(v['input']['internal_source_role']for v in docs['variants27'])),D19_anatomy=dict(Counter(r['anatomical_family']for r in dtable if r['normalized_offset']==[0,0])),old9_semantic=dict(Counter(r['semantic_alignment']for r in pairs)),V219_FIT45=dict(known=45,D=39,E=6,canonical_binding_exact=True,selection='weak_loss_candidate & SOURCE_FIT',training_pool_not_independent_new_samples=True),old27_vs_V219_FIT45_encoded_PNG_SHA_intersection=sorted(pair_hashes&used_hashes),same_original_row_FIT_intersection=underlying_overlap,nominal_D19_in_V219_FIT=sum(r['V219_FIT_pool_member']for r in dtable if r['normalized_offset']==[0,0]),D48_in_V219_FIT=sum(r['V219_FIT_pool_member']for r in dtable),D48_all_union_in_strict_valid=True,PNG_pixels_rehashed_or_decoded=0)
 outputs={}
 outputs['all_rows']=save('SOURCE_ROI_IMAGE_OCCLUDER_PROVENANCE.json',dict(completed=True,SOURCE_metadata_only=True,D48=dtable,nominal_D19_row_ids=[r['row_id']for r in dtable if r['normalized_offset']==[0,0]],original_query9=pairs,V219_FIT45=usedrows,summary=summary))
 outputs['D48_CSV']=csvsave('D48_EXACT_ROI_UNION_VALIDITY_AND_QUERY_PROVENANCE.csv',dtable)
 outputs['D19_CSV']=csvsave('NOMINAL_D19_SOURCE_ROLE_AND_QUERY_PROVENANCE.csv',[r for r in dtable if r['normalized_offset']==[0,0]])
 outputs['pair9_CSV']=csvsave('ORIGINAL_QUERY_COVER9_FINITE_PART_AND_OCCLUDER_PROVENANCE.csv',pairs)
 outputs['used45_CSV']=csvsave('V219_ACTUAL_DEFINED_FIT45_AND_NATURAL_E6.csv',usedrows)
 save('READINESS.json',dict(completed=True,PASS=True,source_only=True,scope='Existing frozen public SOURCE metadata/code only; no new observations, targets, images, training or admission',inputs=bindings,outputs=outputs,summary=summary,available_now=['19 nominal expected-part weak D (FIT15/DEV4), expanded valid known D48 (FIT39/DEV9), all evidence ROI exactOR RLE/valid96 metadata/RGB96 provenance and existing condition32','Original-query finite witness controls9 (FIT7/DEV2), existing27PNG bindings and frozen opaque-raster proofs; native labels remain -1'],new_declared_families_needed=[dict(name='PROPOSED_PUBLIC_GT_QUERY_EXPECTED_PART_MATCHED_INTERVENTION_V1',not_yet_constructed=True,parents='Canonical D48 with existing GT-derived q8/native32/originalL and inherited SOURCE roles; nominal19 are the source semantic observations',authority='Weak visibility of same declared candidate/anatomical finite expected-part bag; not physical point0 gold',BEFORE='existing D weak input',COVER='opaque intervention covers exact declared evidenceOR only after future actual proof',SHAM='same RGB+alpha mass/pattern translated, targetOR unchanged, same q8/native/L as BEFORE',future_training_limit='FIT parents only; every DEV descendant excluded all loss/normalization; inherited D39 FIT already used in V219',ROI='supervision only, no additional pose/ref/ROI/mask deployment channel',UNKNOWN='retain if target address/support or matched construction unresolved; never force E from arbitrary texture'),dict(name='PROPOSED_ORIGINAL_QUERY_FINITE_WITNESS_AUXILIARY_V1',existing_controls_not_new=True,parents='Existing finite9 original detector-query bags',authority='Local finite visible feature weak only; no automatic native or query-support D/E from D/E/D variant',limitation='4 top-role review +2 deck/runner unresolved +3 f6 nonnative-addressable. Conditioning/target name must identify witness before labels can supervise an unconditioned native head.')],limits=['Encoded PNG binding comparisons are metadata identity, not a new pixel rehash/decode or decoded-pixel dedup test.','Frozen reported RGB/alpha proofs were not re-executed. ExactOR/validity checks here use saved ROI and RLE metadata only.','GT-derived training q8 is itself GT-origin information; original detector query/availability remains separate and unmodified. Detector-to-GT-instance association remains UNKNOWN.','No blanket physical-gold requirement: intended finite-part weak supervision is useful under its declared authority, while native point0 remains a distinct target.','No new current/publicheld/VAL/HOLD/private scores or images used; no performance/cause/adoption claim. Public real+GT is currently allowed; historical synthetic-only advice is not a current restriction.'],new_control_construction_or_training_admission=0,model_pixel_array_GT_or_protected_read=0))
 print(json.dumps(dict(completed=True,PASS=True,summary=summary,outputs=outputs),ensure_ascii=False))

if __name__=='__main__':main()
