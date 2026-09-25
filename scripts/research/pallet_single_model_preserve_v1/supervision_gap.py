"""Conditional fixed-rule hard support audit, never a training target generator."""
from collections import Counter
from . import common as C

def main():
    decision=C.read(C.DOC/'DECISION.json');assert decision['primary']!='PRESERVATION_SUPPORTED','Annotation phase must be skipped on preservation success'
    # Fixed reference remains frozen S1, NOT the newly trained adapter.
    anchors=C.read(C.PREV_RAW/'ANCHOR_POINT_METRICS.json');hard=[r for r in anchors if r['severity']!='CLEAN'];assert len(hard)==36
    role=C.read(C.PREV_DOC/'H10_ROLE_PREVALENCE.json');strong=set(role['strong_frames']);targets=[r for r in C.read(C.ROOT/'data/pallet/results/pallet_existing_data_transfer_v1/TARGETS.json') if r['role']=='H'];assert len(targets)==10
    trusted=[];excluded=[];support_total=0
    for r in targets:
        obj=C.read(C.ROOT/r['annotation']['path'])['objects'][0];entries=obj['keypoint_annotations'];meta=obj.get('camera_facing_pnp',{});axis=meta.get('axis_assignment_confirmed')
        # Stored camera_dynamic name alone is not a human-confirmed front role.
        explicit_role=obj.get('role_confident') is True or obj.get('role_status')=='ROLE_CONFIDENT' or obj.get('manual_role_confirmed') is True
        unresolved=not (axis is True or explicit_role) or obj.get('migration_status')=='MANUAL_REVIEW_REQUIRED'
        for ci,mask in enumerate(r['mask'][:8]):
            if not mask:continue
            support_total+=1;e=entries[ci];reason=[]
            if r['id'] in strong:reason.append('STRONG_ROLE_MISMATCH_FRAME')
            if axis is not True and unresolved:reason.append('AXIS_NOT_CONFIRMED_AND_ROLE_UNRESOLVED')
            if e.get('source') not in ('manual','manual_click'):reason.append('NOT_DIRECT_MANUAL_OR_UNKNOWN_PROVENANCE')
            if any('identity' in k.lower() and str(v).lower() in ('unknown','unverified','unresolved') for k,v in e.items()):reason.append('POINT_IDENTITY_UNRESOLVED')
            row=dict(frame=r['id'],corner_id=ci,severity=r['severity'],source=e.get('source'),axis_assignment_confirmed=axis,role_explicit=explicit_role)
            if reason:excluded.append(dict(row,reasons=reason))
            else:trusted.append(row)
    assert support_total==36;cells=Counter((r['severity'],r['corner_id']) for r in trusted);rows=[]
    for r in hard:
        s,t=r['errors']['S1'],r['errors']['TEACHER'];category='STUDENT_OK' if s<=10 else 'TEACHER_CAN_TEACH' if t<=10 else 'BOTH_FAIL_VISIBLE';count=cells[(r['severity'],r['corner_id'])]
        rows.append(dict(frame=r['frame_id'],recording=r['recording'],severity=r['severity'],corner_id=r['corner_id'],student_error=s,teacher_error=t,category=category,
            gross_both_fail=s>20 and t>20,trusted_support_count=count,missing_trusted_support=count<2))
    both=[r for r in rows if r['category']=='BOTH_FAIL_VISIBLE'];missing=[r for r in both if r['missing_trusted_support']];frames={r['frame'] for r in both};corners={r['corner_id'] for r in both};strata={r['severity'] for r in both}
    gates=dict(BOTH_FAIL_VISIBLE_ge6=len(both)>=6,distinct_frames_ge3=len(frames)>=3,distinct_corners_ge3=len(corners)>=3,missing_support_points_ge6=len(missing)>=6,
        distinct_severity_view_strata_ge2=len(strata)>=2,T2_not_claimed_to_solve_deployment=True)
    justified=all(gates.values());status='MIN_HARD_LABELING_JUSTIFIED' if justified else 'MIN_HARD_LABELING_NOT_JUSTIFIED'
    out=dict(created_at=C.now(),run=True,reference_pipeline='S1+frozen GEO_LINEAR (2D unchanged)',hard_visible_points=36,student_wrong10=sum(r['student_error']>10 for r in rows),
        teacher_can_teach=sum(r['category']=='TEACHER_CAN_TEACH' for r in rows),both_fail_visible=len(both),gross_both_fail=sum(r['gross_both_fail'] for r in rows),missing_trusted_support=len(missing),
        distinct_frames=len(frames),distinct_corners=len(corners),strata=sorted(strata),trusted_existing_hard_support=len(trusted),original_H10_T2_support=36,
        trusted_cells=[dict(severity=k[0],corner_id=k[1],count=v) for k,v in sorted(cells.items())],trusted=trusted,excluded=excluded,points=rows,gates=gates,decision=status,
        rule=C.read(C.DOC/'PROTOCOL_LOCK.json')['gap_rule'],teacher_binding=C.bind(C.PREV_DOC/'VERIFIED_ANCHOR_TRANSFER.json'),
        caveat='Low trusted support can mean unresolved metadata rather than absence of historical clicks. Both-fail counts only support a minimal labeling pilot, not proof labeling will solve deployment. T2 is not used as a deployment success argument.')
    C.freeze(C.DOC/'SUPERVISION_GAP_DIAGNOSTIC.json',out)
    lines=['# Conditional hard-supervision gap audit','',f'**{status}**','',
        'PRES1 실패 자체는 어노테이션 필요 근거가 아니다. 아래는 기존 frozen S1+GEO_LINEAR 기준의 verified HARD36과 기존 teacher를 비교한 별도 판정이다.','',
        f"student >10px {out['student_wrong10']}/36, teacher가 가르칠 수 있는 점 {out['teacher_can_teach']}, 둘 다 >10px {len(both)}, 둘 다 >20px {out['gross_both_fail']}. Both-fail은 {len(frames)}프레임/{len(corners)}corner IDs/{len(strata)}난도 strata에 걸친다.",'',
        f"H10/T2 common-manual36 중 역할·직접 클릭 출처가 확인되는 trusted point {len(trusted)}, 제외 {len(excluded)}. Both-fail 중 같은 (severity,corner_id) trusted support<2인 점 {len(missing)}.",'',
        'axis 미확인 + 역할 확신 기록 부재는 보수적으로 unresolved로 두었다. 과거 좌표가 틀렸다고 단정하거나 자동 재라벨하지 않았다. stored camera_dynamic 이름만으로 사람이 front role을 확인했다고 간주하지 않았다.','',
        '|fixed gate|pass|','|---|---|',*[f'|{k}|{v}|' for k,v in gates.items()],'',
        '|frame|corner|severity|S1 px|teacher px|category|trusted cell count|','|---|---:|---|---:|---:|---|---:|']
    for r in rows:lines.append(f"|{r['frame']}|{r['corner_id']}|{r['severity']}|{r['student_error']:.2f}|{r['teacher_error']:.2f}|{r['category']}|{r['trusted_support_count']}|")
    lines+=['','TEACHER_CAN_TEACH는 수동 라벨 부족보다 증류/표현 병목의 근거다. T2의 학습점 적합을 배포 성공 증거로 사용하지 않았다. 이 보고서에는 exact annotation xy를 공개하지 않는다.']
    C.save(C.DOC/'SUPERVISION_GAP_REPORT_KO.md','\n'.join(lines)+'\n');print('GAP_DECISION',status,{k:out[k] for k in ('teacher_can_teach','both_fail_visible','missing_trusted_support','distinct_frames','distinct_corners')},flush=True)
if __name__=='__main__':main()
