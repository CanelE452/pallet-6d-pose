from pathlib import Path
import json,hashlib,datetime,sys
V=Path(__file__).resolve().parents[2]; R=Path(__file__).resolve().parent; E=V/'evidence_recovered_v3'
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
manual={}
for line in (R/'manual_observations.tsv').read_text().splitlines():
 k,i,v,q,reason=line.split('\t'); assert k not in manual;manual[k]=(i,v,q,reason)
I={'C':'CLEAR','A':'AMBIGUOUS','U':'NOT_VISUALLY_RESOLVABLE'}
VI={'V':'VISIBLE_SUPPORTED','E':'EXTERNAL_BLOCK_SUPPORTED','S':'SELF_HIDE_SUPPORTED','G':'BOUNDARY_OR_GAP_AMBIGUOUS','U':'UNKNOWN'}
Q={'O':'ON_INTENDED_PART_SUPPORTED','N':'NEAR_INTENDED_PART_SAME_SELECTED_PALLET','P':'OTHER_SELECTED_PALLET_PART','F':'FOREIGN_OBJECT','B':'BACKGROUND_OR_GAP','U':'UNKNOWN'}
mode=sys.argv[1]; name='BLIND_CROSS40_ALL_REVIEWERS.json' if mode=='cross' else 'REVIEW_INPUT_GROUP_A.json'; manifest=E/name; x=json.loads(manifest.read_text())
cross=json.loads((E/'BLIND_CROSS40_ALL_REVIEWERS.json').read_text());crossids={c['case_id'] for c in cross['cases']}
viewed=x['contacts'] if mode=='cross' else [c for c in x['contacts'] if not set(c['case_ids'])<=crossids]
# Primary overlap uses already actually viewed cross contacts; no extra vote.
if mode!='cross':viewed += [c for c in cross['contacts'] if any(z in {r['case_id'] for r in x['cases']} for z in c['case_ids'])]
image_for={}
for ct in viewed:
 f=Path(ct['image']['path']);assert bind(f)==ct['image']
 for k in ct['case_ids']:image_for[k]=ct['image']
rows=[]
for c in x['cases']:
 k=c['case_id'];i,v,q,reason=manual[k]; assert k in image_for
 if q in ['O','N','P']:assert i=='C'
 row={k:c[k] for k in ['case_id','global_index','query_id','native_slot','native_name','frame_index','source_id','split','query_xy','L','selected_gt_instance_index']}
 row.update(reviewer='A',review_set='BLIND_CROSS40' if mode=='cross' else 'PRIMARY130',selected_instance_correspondence=I[i],native_part_visibility=VI[v],query_relation=Q[q],reason_KO=reason,ambiguity_KO='명목 CAD/pose와 실제 끝의 픽셀 일치는 보증하지 않는다. 구조적 약한 관찰이며 정확한 코너 좌표나 native 가시성 gold가 아니다.' if v!='U' and q!='U' else '부분 위치 또는 q 경계의 미해결 불확실성을 유지한다. 명목 투영이나 이전 라벨로 보완하지 않았다.',part_support_extent=None,part_cover_readiness='WEAK_CANDIDATE_PART_VISIBLE_AND_Q_RELATED' if i=='C' and v=='V' and q in ['O','N'] else 'UNKNOWN_NOT_ADMITTED',actually_viewed={'manifest':bind(manifest),'images':[image_for[k]],'tool':'tools.view_image','viewed':True},assistant_weak_observation_only=True,native_coordinate_gold=False,training_admitted=False,original_labels_modified=False)
 if mode!='cross' and k in crossids:row['same_reviewer_cross_observation_reused']=True
 rows.append(row)
f=R/('BLIND_CROSS40_NATIVE_PART_WITNESS.jsonl' if mode=='cross' else 'PRIMARY130_NATIVE_PART_WITNESS.jsonl');f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
receipt={'schema':'V163_REVIEWER_A_'+('CROSS40_SEALED' if mode=='cross' else 'PRIMARY130_SEALED'),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'PASS':True,'manifest':bind(manifest),'guidance':bind(V/'reviewer_a/preparation/SHARED_OBSERVATION_GUIDANCE.json'),'schema_binding':bind(V/'reviewer_a/preparation/SHARED_OBSERVATION_ROW_SCHEMA.json'),'trigger':bind(V/'ROOT_ACTUAL_BLINDED_RGB_REVIEW_TRIGGER.json'),'output':bind(f),'rows':len(rows),'all_expected_cases_exact':{r['case_id'] for r in rows}=={c['case_id'] for c in x['cases']},'actually_viewed_images':[z['image'] for z in viewed],'other_reviewer_votes_read':False,'root_votes_read':False,'old_labels_or_model_outputs_read':False,'source_pixels_changed':False,'new_labels_admitted':False}
r=R/('CROSS40_SEALED.json' if mode=='cross' else 'PRIMARY130_SEALED.json');dump(r,receipt);print(json.dumps(bind(r)))
