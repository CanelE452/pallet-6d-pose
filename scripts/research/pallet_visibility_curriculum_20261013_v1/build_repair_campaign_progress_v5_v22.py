"""Append-only uniform O/S-prior comparison from sealed score archives."""
from datetime import datetime,timezone
from pathlib import Path
import html

from . import build_repair_campaign_progress_v3_v18 as P
from . import build_repair_campaign_progress_v4_v19 as PAIR
from . import evaluate_bounded_geometric_priors_v4_22 as V

OUTPUT=P.OUTPUT
LABELS={P.BASELINE:'기존 e',
    'O_LOCALIZATION_ABSTAINED_NATIVE':'v19 범위 제한 보류',
    'O_LOCALIZATION_ABSTAINED_PLUS_MASK':'v19 범위 제한 보류 + 마스크',
    'O_SOFT_NATIVE':'가림 위치 범위의 제한된 벌점',
    'O_SOFT_S_NATIVE':'가림 범위 + S 기하학의 제한된 벌점',
    'O_SOFT_PLUS_MASK':'가림 범위 벌점 + 마스크',
    'O_SOFT_S_PLUS_MASK':'가림 범위·S 벌점 + 마스크'}


def build():
    C=P.C;out=OUTPUT
    paths=[out/name for name in ('PROGRESS_V5_V22.json','RESULTS_V5_V22.html','PROGRESS_V5_V22_SEAL.json')]
    if any(p.exists()for p in paths):raise FileExistsError('Preserve previous campaign documents')
    oldprotocol,old=P._load_run(V.V.OUTPUT);protocol,new=P._load_run(V.OUTPUT)
    auditpath=V.OUTPUT/'independent_root_audit_v1/AUDIT.json';audit=C.read(auditpath)
    auditseal=C.read(auditpath.with_name('AUDIT_SEAL.json'))
    if (audit['rows']!=238 or audit['failure_count']or audit['failures']
            or C.sha(auditpath)!=auditseal['receipt_sha256']
            or audit['prediction_seal']['sha256']!=C.sha(V.OUTPUT/'PREDICTION_SEAL.json')):
        raise ValueError('Independent current all238 audit required')
    if oldprotocol['ids']!=protocol['ids']:raise ValueError('Keep original ordered cohort')
    for part in P.COUNTS:
        if [r['id']for r in old[part]['rows']]!=[r['id']for r in new[part]['rows']]:raise ValueError('Paired rows changed')
        if [(r['metrics'][P.BASELINE],r['accepted'][P.BASELINE])for r in old[part]['rows']]!=[
                (r['metrics'][P.BASELINE],r['accepted'][P.BASELINE])for r in new[part]['rows']]:
            raise ValueError('Frozen e score changed')
    methods=[]
    for reports,folder,arms in ((old,V.V.OUTPUT,(P.BASELINE,)+V.V.ARMS[1:]),(new,V.OUTPUT,V.ARMS[1:])):
        for arm in arms:
            populations={}
            for part in P.COUNTS:
                populations[part]=P._summary(reports[part]['rows'],arm)
                P._same_summary(populations[part],reports[part]['summary'][arm])
            populations['PRIMARY192']=P._summary(reports['TEST']['rows']+reports['REMAINDER']['rows'],arm)
            methods.append(dict(method=arm,label_KO=LABELS[arm],populations=populations,
                source_output=str(folder.relative_to(C.ROOT)),current_default=arm==P.BASELINE,
                original_TEST49_all_translation_le5=populations['TEST']['translation_le5']==49,
                numerical_primary192_target_met=populations['PRIMARY192']['translation_le5']==192))
    paired={}
    for profile,arms in V.PROFILE_ARMS.items():
        for index,arm in enumerate(arms):
            previous=V.V.ARMS[index+1]
            paired[arm]={part:PAIR._paired(old[part]['rows'],new[part]['rows'],previous,arm)for part in P.COUNTS}
            paired[arm]['PRIMARY192']=PAIR._paired(old['TEST']['rows']+old['REMAINDER']['rows'],
                new['TEST']['rows']+new['REMAINDER']['rows'],previous,arm)
    bindings=[Path(__file__),Path(P.__file__),Path(PAIR.__file__),auditpath,auditpath.with_name('AUDIT_SEAL.json')]
    for folder in (V.V.OUTPUT,V.OUTPUT):
        bindings.extend(folder/name for name in ('PROTOCOL.json','INPUT_SEAL.json','PREDICTION_SEAL.json',
            'SCORE_SEAL.json','CAL_SCORE.json','TEST_SCORE.json','REMAINDER_SCORE.json'))
    for folder in (V.OCAL.OUTPUT,V.SCAL.OUTPUT):
        bindings.extend(folder/name for name in ('MODEL.json','MODEL_SEAL.json','CAL_LOGO_REPORT.json'))
    active=C.read(out/'ACTIVE_GOAL_PROTOCOL_V1.json');P._verify(active['protected_user_sha256'])
    data=dict(schema='ACTIVE_REPAIR_CAMPAIGN_PROGRESS_V5_BOUNDED_O_S',
        generated_UTC=datetime.now(timezone.utc).isoformat(),goal_status='active',default_changed=False,
        default_method=P.BASELINE,methods=methods,paired_v19=paired,primary=dict(TEST49=49,REMAINDER143=143,total=192,severe=0),
        separate_CAL=46,no_per_frame_best_or_GT_ID_partition_routing=True,
        numerical_target_met_any_new_method=any(m['numerical_primary192_target_met']for m in methods[3:]),
        change='same native image observations/guards/starts; bounded coarse O frame prior and optional model-only S frame prior',
        S_image_2D_or_regions=0,U_regions_only=True,independent_new_image_information_claimed=False,
        original_O_radius_model_unchanged=True,all_methods_uniform_full_population=True,
        additional_reference_is_annotation_PnP_derived_not_independent_metrology=True,
        reused_exploratory_TEST_not_new_holdout=True,evidence=[P._binding(p)for p in bindings],
        independent_root_audit=P._binding(auditpath),protected_user_sha256=active['protected_user_sha256'])
    C.write(paths[0],data)
    content=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>전체 평가: 제한된 가림 범위·S 벌점</title>',
        '<style>body{font:16px/1.6 sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#192536}table{border-collapse:collapse;width:100%;margin:20px 0 40px}th,td{border-bottom:1px solid #ccd4df;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}tr:first-child{background:#eef4fa}p{max-width:1000px}</style>',
        '<p>기존 e를 유지합니다. severe를 제외한 기존 49장과 추가 143장 전체를 같은 방법으로 평가했습니다. CAL46은 별도입니다. 표의 각 행은 사진별 최선 결과를 섞지 않은 단일 방법입니다.</p>',
        '<p>O의 기존 타원 범위를 강제 조건에서 제한된 벌점으로 바꾸고, S에는 선택적으로 초기 모델 기하학의 제한된 벌점을 추가했습니다. S는 계속 자세 투영으로만 출력하며, U는 영역으로만 사용합니다. 기존 영상 관측·hard guard·시작점·파렛트 크기는 같습니다.</p>']
    for part,title in (('PRIMARY192','전체 192장'),('TEST','기존 49장'),('REMAINDER','추가 143장'),('CAL','별도 CAL46')):
        content.append(f'<p><b>{title}</b> — T≤5cm: 위치 오차 5cm 이하, 공동: 5cm·5° 모두 만족. 오차 단위는 cm입니다.</p>')
        content.append('<table><tr><th>방법</th><th>T≤5cm</th><th>공동</th><th>평균 위치 오차</th><th>최대 위치 오차</th><th>fallback</th></tr>')
        for method in methods:
            q=method['populations'][part]
            content.append(f'<tr><td>{html.escape(method["label_KO"])}</td><td>{q["translation_le5"]}</td><td>{q["successes_5cm_5deg"]}</td><td>{q["translation_cm"]["mean"]:.3f}</td><td>{q["translation_cm"]["max"]:.3f}</td><td>{q["fallback_count"]}</td></tr>')
        content.append('</table>')
    content.extend(['<p>추가 143장의 참조는 저장된 annotation/PnP에서 유도한 값입니다. 독립 물리 계측이나 새 holdout의 일반화 결과로 해석하지 않습니다. 회전 오차·개별 사례·대응 비교·봉인은 아래 JSON에 보존했습니다.</p>',
        '<p><a href="PROGRESS_V5_V22.json">전체 수치와 검산 근거</a></p></html>'])
    paths[1].write_text('\n'.join(content),encoding='utf-8')
    P._verify(active['protected_user_sha256'])
    C.write(paths[2],dict(source=P._binding(Path(__file__)),json=P._binding(paths[0]),html=P._binding(paths[1]),
        evidence=data['evidence'],no_new_inference_pose_fits_images_or_reference_decode=True))
    print(dict(methods=len(methods),json_sha256=C.sha(paths[0]),html_sha256=C.sha(paths[1])),flush=True)


if __name__=='__main__':build()
