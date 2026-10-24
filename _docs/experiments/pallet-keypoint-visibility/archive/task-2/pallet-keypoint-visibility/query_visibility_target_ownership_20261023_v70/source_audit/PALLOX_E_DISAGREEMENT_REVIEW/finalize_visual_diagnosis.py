"""Save observations after actual fixed-pair RGB review. No new inference."""
from pathlib import Path
import csv,hashlib,json
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    target=OUT/'PALLOX_E_DISAGREEMENT_REVIEW.json';assert not target.exists()
    lock=json.loads((OUT/'SELECTION_LOCK.json').read_text());data=json.loads((OUT/'SAVED_SOURCE_CERTIFICATE_AND_CROPS.json').read_text())
    notes=[{
        'observed_material_at_selected_q':'FOREIGN_CARDBOARD_FLAP',
        'observed_material_at_paired_q':'TARGET_WOODEN_BIN_BOARD',
        'qualitative_boundary_assessment':'The selected q lies on the light cardboard flap in front of the wooden bin. The flap/wood edge is visible elsewhere in the96 crop, but q is not on directly exposed wood. It is closer to a foreground edge than case2; RGB/depth registration uncertainty remains conditional.',
        'visual_evidence_ko':'A의 동일 q는 별도 cardboard 상자의 펼쳐진 flap 위에 있고, B에서는 그 상자가 왼쪽으로 옮겨져 동일 q의 목재 bin 판재가 드러난다. 주변 crop에 목재가 함께 있어도 q 소유자는 cardboard이다.',
        'diagnosis':'CONSISTENT_WITH_REAL_FOREGROUND_QUERY_OWNERSHIP_FALSE_VISIBLE; conditional teacher not human-gold',
        'confidence':'AI visual material distinction high; exact RGB-depth pixel registration unverified',
    },{
        'observed_material_at_selected_q':'FOREIGN_CARDBOARD_BOX_FRONT_INTERIOR',
        'observed_material_at_paired_q':'TARGET_WOODEN_BIN_BOARD',
        'qualitative_boundary_assessment':'The selected q lies well inside the broad cardboard front face. In the paired view it lies on a wooden bin board, above an opening. This case is qualitatively difficult to explain by a1–3pixel border displacement alone.',
        'visual_evidence_ko':'A의 q 주변은 큰 cardboard 상자 앞면 내부이며 실제 목재가 보이는 점이 아니다. B의 동일 좌표에는 목재 bin 하부 판재가 나타난다. 현재 q를 다른 목재나 모서리로 옮기지 않았다.',
        'diagnosis':'CONSISTENT_WITH_REAL_FOREGROUND_QUERY_OWNERSHIP_FALSE_VISIBLE; strong local paired-depth margin despite conditional global alignment',
        'confidence':'AI visual material distinction high; exact RGB-depth pixel registration unverified',
    }]
    cases=[];score_rows=[]
    for row,obs in zip(data['cases'],notes):
        a=row['saved_scores_selected_A'];b=row['saved_scores_paired_B'];diff={m:float(b[m]['p_visible'])-float(a[m]['p_visible']) for m in a}
        cases.append({**row,'actual_RGB_visual_observation':obs,
            'saved_score_direction_pV_B_minus_A':diff,'V70_pair_direction_correct_B_gt_A':diff['V70_OWNERSHIP_LOO_PAIR_V1']>0,
            'fixed_binary_threshold':.5,'V70_selected_A_false_visible':float(a['V70_OWNERSHIP_LOO_PAIR_V1']['p_visible'])>=.5,
            'other_requested_baselines_selected_A_nonvisible':all(float(a[m]['p_visible'])<.5 for m in ['MOBILE_SAM_RAW','V65_BASELINE','V66_CONTROLLED_PLUS_PALLOX']),
            'teacher_or_label_changed':False,'query_or_crop_changed':False})
        for view,scores in [('A',a),('B',b)]:
            for model,s in scores.items():
                score_rows.append({'selected_case_sample_id':row['sample_id'],'source_pair_id':row['source_pair_id'],'view':view,'model':model,'sample_id':s['sample_id'],'conditional_exact_GT':s['exact_GT'],'p_visible':float(s['p_visible']),'fixed_pV_ge_0_5_visible':float(s['p_visible'])>=.5,'source_label_tier':'CONDITIONAL_TEACHER_NOT_GOLD'})
    result={'schema':'FIXED_TWO_PUBLIC_PALLOX_E_DISAGREEMENT_REVIEW_V1','status':'COMPLETED_DIAGNOSIS_ONLY_FROZEN_SOURCE_MODEL_LABELS_UNCHANGED',
        'selection':'SHA256(sample_id) first2 distinct PALLOX target IDs among native-valid knownE with V70pV>=.5 and rawMobilepV<=.5;11 eligible. Frozen before this task RGB reads. No confidence ranking.',
        'selection_lock':{'path':str(OUT/'SELECTION_LOCK.json'),'sha256':sha(OUT/'SELECTION_LOCK.json')},
        'population_limit':'These2 deliberately selected disagreement cases are not an unbiased estimate of all31 knownE or independent recording families. Both are1.23m_hall from the same public session, different pairs/target rows.',
        'main_finding_ko':'두 고정 사례 모두 q는 실제 외부 cardboard에 있고 paired reference view에서는 같은 q의 대상 목재가 보인다. 기존5x5 조건부 E 증거도25/25이고 깊이 분리는 충분히 커서, 이 사례들은 단순1–3px 경계/약한 teacher 문제보다는 V70의 실제 foreground 소유권 false-visible과 더 일치한다. 이것은 source component 진단이며 full semantic corner 목표나 현재평가 결과의 성공 주장이 아니다.',
        'certificate_support_ko':'첫 사례25점 B-A 깊이차 min149/median161mm, 두번째 min812/median843mm. 두 사례 모두 pairedB teacher core/visible/mesh-agree25/25. supplied T와K는 쌍마다 동일하고 기존 reprojection 수치는 수치상0에 가깝지만, 이는 실제 nativeRGB-depth pixel registration 인증이 아니다.',
        'remaining_teacher_limits':['The original visible support uses supplied SAM teacher plus depth/mesh agreement; mask agreement with rawMobileSAM is not independent human validation.','The protocol assumes depth in the RGB native pixel grid; official explicit calibration/alignment certification is absent.','Pairwise unchanged-area depth residual p95 is20mm for0004/5 and48mm for0020/21; query-specific saved certificate margins are separately reported, not used to change thresholds or exclude pairs.','The public object is a tall wooden pallet-bin, not a flat pallet; these query slots have no canonical pallet-corner semantic truth.'],
        'mechanism_interpretation':'V70 has pV_B>pV_A in both pairs, but both covered-viewA values remain>=.5. Stored scores demonstrate insufficient binary separation, not total invariance to the occluder. Whole-target/context evidence overwhelming local ownership is a plausible hypothesis only; no branch causal/gradient/model-weight analysis was performed.',
        'baseline_scope':'Reported p_visible is saved exact-query visible marginal for each model. Binary nonvisible is not causal E classification. No pE is fabricated from1-pV.',
        'preservation':{'all_source_scores_weights_labels_queries_boxes_L_and_previous_protocols_unchanged':True,'new_forward_fit_threshold_radius_or_checkpoint_selection':False,'current_data_access':False,'new_PnP_or_alignment_optimization':False,'semantic_or_human_GT_admitted':0},
        'read_scope':{'public_source_RGB_images':4,'public_depth_images_for_unchanged_certificate_replay':4,'saved_evidence_source_NPZ_only':2,'saved_public_score_CSV':True,'source_feature_weight_arrays':0,'current_RGB_GT_scores_features':0},
        'contact_sheet':data['contact_sheet'],'evidence_manifest':{'path':str(OUT/'SAVED_SOURCE_CERTIFICATE_AND_CROPS.json'),'sha256':sha(OUT/'SAVED_SOURCE_CERTIFICATE_AND_CROPS.json')},
        'cases':cases,'code':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))}}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'FIXED_CASE_SAVED_SCORES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(score_rows[0]));w.writeheader();w.writerows(score_rows)
    files=[p for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='REVIEW_SEAL.json']
    seal={'schema':'PALLOX_FIXED_DISAGREEMENT_REVIEW_SEAL_V1','status':result['status'],'report_sha256':sha(target),'selection_lock_sha256':sha(OUT/'SELECTION_LOCK.json'),'source_only':True,'new_fit_forward_PnP_current':False,'bindings':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]}
    sp=OUT/'REVIEW_SEAL.json';assert not sp.exists();sp.write_text(json.dumps(seal,indent=2)+'\n')
    print(json.dumps({'report_sha256':sha(target),'seal_sha256':sha(sp),'contact_sheet_sha256':data['contact_sheet']['sha256']}))
if __name__=='__main__':main()
