from pathlib import Path
import collections,csv,hashlib,io,json,math,os
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation');H=Path(__file__).parent;pkg=R/'V236_PUBLICATION_PACKAGE';scope=json.loads((H/'SCOPE_BEFORE_SOURCE_ERROR_JOIN.json').read_text());inputs=scope['inputs']
# Root exact-path clarification replaces no body and adds no linked target traversal.
inputs['prior_coverage']['path']=str(pkg/'artifacts/V234_YCB_SHIFT_AUDIT_B/OBJECT_AUX_LABEL_COVERAGE.csv')
inputs['prior_report']['path']=str(pkg/'artifacts/V234_YCB_SHIFT_AUDIT_B/PORTABLE_REPORT_KO.md')
def read(p):
 p=Path(p);s=str(p).casefold();assert p.is_absolute() and '..' not in p.parts and not any(x in s for x in ('v112','__pycache__','.pyc','current223','publicheld','oldval','source_holdout','v177','v178','v182','private_evaluation'))
 for q in reversed((p,*p.parents)):assert not q.is_symlink()
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()
def bind(p):
 z=read(p);return {'path':str(p),'sha256':hashlib.sha256(z).hexdigest(),'size_bytes':len(z)}
manifest_raw=read(inputs['prior_manifest']['path']);manifest=json.loads(manifest_raw);lookup={x['path']:x for x in manifest['files']};raws={}
for name,b in inputs.items():
 if name=='prior_manifest':continue
 if name.startswith('prior'):
  item=lookup[Path(b['path']).relative_to(pkg).as_posix()];b.update(sha256=item['sha256'],size_bytes=item.get('bytes',item.get('size_bytes')))
 z=read(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256']
 if 'size_bytes' in b:assert len(z)==b['size_bytes']
 b['size_bytes']=len(z);raws[name]=z
inputs['prior_manifest'].update(sha256=hashlib.sha256(manifest_raw).hexdigest(),size_bytes=len(manifest_raw))
rr={k:list(csv.DictReader(io.StringIO(v.decode()))) for k,v in raws.items() if k!='prior_report'}
control,treat=rr['CONTROL'],rr['TREAT'];assert len(control)==len(treat)==7617
for a,b in zip(control,treat):assert all(a[k]==b[k] for k in ('sample_index','row_id','family','role','available','target'))
new=[(a,b) for a,b in zip(control,treat) if a['family']=='1'];old=rr['prior_diagnostic'];assert len(new)==len(old)==1521
prior={r['row_id']:r for r in old};assert len(prior)==1521 and len({r['sample_index'] for r in old})==1521
joined=[]
for a,b in new:
 p=prior.get(a['row_id']);assert p is not None and a['sample_index']==p['sample_index'] and a['role']==p['role'] and 'DE'[int(a['target'])]==p['associated_r_state']
 assert a['available']==b['available']=='True'
 joined.append((a,b,p))
coverage={};byobjE=collections.Counter();byauxE=collections.Counter()
for a,b,p in joined:
 key=(p.get('object_id','UNKNOWN'),p.get('aux_slot','UNKNOWN'));item=coverage.setdefault(key,dict.fromkeys(('FIT_D','FIT_E','DEV_D','DEV_E'),0));item[a['role']+'_'+'DE'[int(a['target'])]]+=1
 if a['role']=='FIT' and a['target']=='1':byobjE[p['object_id']]+=1;byauxE[p['aux_slot']]+=1
oldcov={(r['object_id'],r['logical_aux_slot']):{k:int(r[k]) for k in ('FIT_D','FIT_E','DEV_D','DEV_E')} for r in rr['prior_coverage']};assert coverage==oldcov
rows=[]
for a,b,p in joined:
 if a['role']!='DEV' or a['target']!='1':continue
 dist=p.get('q_reference_distance_px','');distance=float(dist) if dist else None
 if distance is not None and not math.isfinite(distance):distance=None
 category='UNKNOWN' if distance is None else ('LE3' if distance<=3 else 'GT3')
 ca=a['prediction']=='E';cb=b['prediction']=='E';status='both_correct' if ca and cb else 'recovered' if cb else 'damaged' if ca else 'still_missed'
 cov=coverage.get((p.get('object_id','UNKNOWN'),p.get('aux_slot','UNKNOWN')),dict.fromkeys(('FIT_D','FIT_E'),None))
 rows.append({'row_id':a['row_id'],'sample_index':int(a['sample_index']),'scene':p.get('scene','UNKNOWN'),'frame_id':p.get('frame_id','UNKNOWN'),'object_id':p.get('object_id','UNKNOWN'),'logical_aux_slot':p.get('aux_slot','UNKNOWN'),'original_role':a['role'],'associated_r_state':'E','q_reference_distance_px':distance,'q_r_category':category,'CONTROL_prediction':a['prediction'],'TREAT_prediction':b['prediction'],'change_TREAT_vs_CONTROL':status,'FIT_same_object_E':byobjE[p['object_id']],'FIT_same_object_aux_D':cov['FIT_D'],'FIT_same_object_aux_E':cov['FIT_E'],'FIT_same_aux_E':byauxE[p['aux_slot']]})
assert len(rows)==57
counts=collections.Counter(r['change_TREAT_vs_CONTROL'] for r in rows);groups={}
for r in rows:
 key=(r['scene'],r['object_id'],r['logical_aux_slot'],r['q_r_category']);g=groups.setdefault(key,{'scene':key[0],'object_id':key[1],'logical_aux_slot':key[2],'q_r_category':key[3],'DEV_E':0,'CONTROL_correct':0,'TREAT_correct':0,**dict.fromkeys(('recovered','damaged','still_missed','both_correct'),0),'FIT_same_object_aux_D':r['FIT_same_object_aux_D'],'FIT_same_object_aux_E':r['FIT_same_object_aux_E']})
 g['DEV_E']+=1;g['CONTROL_correct']+=r['CONTROL_prediction']=='E';g['TREAT_correct']+=r['TREAT_prediction']=='E';g[r['change_TREAT_vs_CONTROL']]+=1
for filename,table in [('YCB_DEV_E57_ERROR_JOIN.csv',rows),('YCB_DEV_E_GROUP_COUNTS.csv',[groups[k] for k in sorted(groups)])]:
 with (H/filename).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
distgroups={}
for r in rows:
 g=distgroups.setdefault(r['q_r_category'],{'DEV_E':0,'CONTROL_correct':0,'TREAT_correct':0,**dict.fromkeys(('recovered','damaged','still_missed','both_correct'),0)})
 g['DEV_E']+=1;g['CONTROL_correct']+=r['CONTROL_prediction']=='E';g['TREAT_correct']+=r['TREAT_prediction']=='E';g[r['change_TREAT_vs_CONTROL']]+=1
summary={'completed':True,'PASS':True,'input_bindings':inputs,'same_saved_SOURCE_composition':{'new_YCB_rows':1521,'all_row_id_and_sample_index_both_exact':True,'original_role_and_associated_r_label_exact':True,'original_object_aux_coverage31cells_exact':True,'FIT_known':sum(a['role']=='FIT' for a,b,p in joined),'DEV_known':sum(a['role']=='DEV' for a,b,p in joined)},'YCB_DEV_E':{'rows':57,'unique_saved_scene_frame_pairs':len({(r['scene'],r['frame_id']) for r in rows}),'scene6_object7':sum(r['scene']=='6' and r['object_id']=='7' for r in rows),'FIT_matching_object_aux_E_supported_rows':sum(r['FIT_same_object_aux_E']>0 for r in rows),'CONTROL_correct':sum(r['CONTROL_prediction']=='E' for r in rows),'TREAT_correct':sum(r['TREAT_prediction']=='E' for r in rows),'change_counts':dict(counts),'by_q_r':distgroups},'scene_object_aux_q_r_groups':[groups[k] for k in sorted(groups)],'missing_join_values':{'unmatched_original_row_or_sample_id':0,'unknown_scene_object_aux':sum(any(r[k]=='UNKNOWN' for k in ('scene','object_id','logical_aux_slot')) for r in rows),'unknown_q_r_category':sum(r['q_r_category']=='UNKNOWN' for r in rows)},'limits':['Descriptive saved SOURCE join only; associated-r teacher E is not literal q-owner/native physical pallet gold. q is prior GT-jitter emulation, not a measured deployment initializer distribution.','scene/frame/object/logicalaux metadata are grouping proxies; not independent physical occlusion events. Camera/lighting/occluder causes and model mechanism remain UNKNOWN.','q-r<=3 here uses existing saved diagnostic reference distance, not new rawGT access or a 3px correctness certificate.','Same object×aux lacks FIT E support descriptively; this does not prove the cause of the new errors. No row/target/split/model/data/threshold/architecture choice or training.','Prior report links and old original audit directories were not followed. No rawGT/images/weights/features/NPZ/current/held bodies.']}
p=H/'SUMMARY.json';p.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['# V240 저장 SOURCE 오류 조인','',f"기존 YCB known 1,521행을 새 두 CSV와 row_id **및** sample_index로 전수 연결했습니다. 역할·associated-r D/E와 기존 object×aux 31개 구성표가 모두 동일합니다. FIT/DEV known 분모는 1,124/397이며 DEV E는 57행입니다.",'',f"CONTROL E 정답 {summary['YCB_DEV_E']['CONTROL_correct']}/57 → TREAT {summary['YCB_DEV_E']['TREAT_correct']}/57입니다. 회복 {counts['recovered']}, 손상 {counts['damaged']}, 계속 미검출 {counts['still_missed']}, 둘 다 정답 {counts['both_correct']}입니다.",'','|scene|object|logical aux|q–r 범주|DEV E|회복|손상|계속 미검출|FIT 같은 object×aux D/E|','|---|---|---|---|---:|---:|---:|---:|---|']
for g in [groups[k] for k in sorted(groups)]:lines.append(f"|{g['scene']}|{g['object_id']}|{g['logical_aux_slot']}|{g['q_r_category']}|{g['DEV_E']}|{g['recovered']}|{g['damaged']}|{g['still_missed']}|{g['FIT_same_object_aux_D']}/{g['FIT_same_object_aux_E']}|")
lines+=['',f"기존과 같이 53/57은 scene6/object7이며, 같은 object×aux의 FIT E 지원은 57행 모두 0입니다. saved frame 구분은 {summary['YCB_DEV_E']['unique_saved_scene_frame_pairs']}개이나 독립된 자연가림 사건 수로 인증하지 않습니다. 이 구성이 오류의 원인이라는 결론도 내리지 않습니다.",'','q–r 분류는 기존 진단표의 저장 거리입니다. nominal/associated reference와 query의 거리이지 새 물리점 gold나 현재 실사 원인 분석이 아닙니다. YCB native는 UNKNOWN·aux는 논리 슬롯이고, SOURCE 내부 DEV는 이미 노출·재사용됐습니다. 모델/data/architecture 선택, 추가 학습, label 수정은 하지 않았습니다.','', '[57행 조인표](YCB_DEV_E57_ERROR_JOIN.csv) · [그룹표](YCB_DEV_E_GROUP_COUNTS.csv) · [요약·한계](SUMMARY.json)']
(H/'PORTABLE_REPORT_KO.md').write_text('\n'.join(lines)+'\n')
receipt={'completed':True,'PASS':True,'scope':bind(H/'SCOPE_BEFORE_SOURCE_ERROR_JOIN.json'),'code':bind(Path(__file__)),'summary':bind(H/'SUMMARY.json'),'tables':[bind(H/'YCB_DEV_E57_ERROR_JOIN.csv'),bind(H/'YCB_DEV_E_GROUP_COUNTS.csv')],'report':bind(H/'PORTABLE_REPORT_KO.md'),'inputs':inputs,'actual_model_or_raw_GT_image_feature_weight_NPZ_current_held_reads':0,'new_training_or_selection':False,'source_gate_success_claimed':False}
(H/'REVIEW.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'REVIEW':bind(H/'REVIEW.json'),'E_summary':summary['YCB_DEV_E'],'groups':summary['scene_object_aux_q_r_groups']},ensure_ascii=False))
