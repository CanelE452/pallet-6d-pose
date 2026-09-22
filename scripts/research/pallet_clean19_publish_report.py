"""Publish existing results and representative images; no training or selection."""
from pathlib import Path
import cv2
import numpy as np
from scripts.research import pallet_clean19_student_v1 as S


def main():
    doc=S.DOC
    two=S.P.read(doc/'RESULTS.json');six=S.P.read(doc/'pose/RESULTS.json')
    rr=S.P.read(S.P.DOC/'SPLIT.json')['evaluation']
    old=S.P.read(S.V.RAW/'PREDICTIONS.json')['predictions']
    preds={'R0':old['R0_DIRECT'],'CORRECTED_TEACHER':old['TYPE_REPLAY_PIPELINE'],'STUDENT':{}}
    for arm in ('PLASTIC','WOOD'):
        preds['STUDENT'].update({r['id']:r['prediction'] for r in S.P.read(S.RAW/f'EVAL_PREDICTIONS_{arm}.json')['records']})
    metrics={'R0':S.P.read(S.P.RAW/'FRAME_METRICS.json')['R0'],
        'CORRECTED_TEACHER':S.P.read(S.V.RAW/'FRAME_METRICS.json')['TYPE_REPLAY_PIPELINE'],
        'STUDENT':S.P.read(S.RAW/'FRAME_METRICS.json')}
    names={'R0':'R0','CORRECTED_TEACHER':'보정 적용','STUDENT':'Self-training 학생'}
    groups={'ALL300':'전체300장','plastic':'일반 플라스틱184장','wood':'목재116장','CLEAN':'Clean132장','MODERATE_OCCLUSION':'중간 가림87장','SEVERE_OCCLUSION':'심한 가림81장'}
    lines=['# Clean19 보정·자기 가림 PnP·종류별 Self-training 종합 보고서','',
        '작성: 2026-09-22. 추가 학습 없이 완료된 결과를 모아 작성했다. 이번 사용자 요청에 따라 코드·보고서·표·비교 이미지를 main에 공개한다. 이전 문서의 “commit/push 없음”은 당시 상태이며 이번 게시 요청으로 변경되었다.', '',
        '## 결론','',
        '**평가300장에서는 보정기를 직접 적용하는 방법이 학생 단독보다 코너 정밀도와 주요6D 지표가 높았다.** 학생은 Clean의 일부6D 지표는 개선했지만 심한 가림에서는6D가 악화했다. 이번320step 실험에서 보정 지식을 학생에게 충분히 전달하지 못했으며, 최종모델 교체나 후속 튜닝은 하지 않았다.', '',
        '## 데이터와 실험 순서','',
        '- 기존 어노테이션319장 중 Clean19장(플라스틱10·목재9) 학습 / 나머지300장 평가. 원본 이미지 SHA 교차0. 같은 촬영 세션 사용은 사용자 승인 사항이며 독립환경 검증은 아니다.',
        '- 1단계: 합성전용 PRIOR1에서 혼합19장 보정기, 이어 종류별10/9장 보정기를 각각300step 학습. 직접클릭 코너 감독+합성 replay.',
        '- 2단계: R0 PnP로 자기 가림을 판정하고 비가림점에만 종류별 보정 적용 → 비가림점만으로 PnP 재추정 → 자기 가림점만 대체. 직육면체 근사이며 외부물체 가림 검출이 아니다.',
        '- 3단계: 19장의 위 보정 결과를 고정 수도레이블로 사용. 17장 숨은점 대체,2장 판정변화 fallback. 두 학생 모두 원래 R0에서 시작; 종류별 실사512복원추출 슬롯+동일 혼합합성512슬롯,5epoch/320update,seed42,최종last 고정.',
        '- 학생 평가는 보정기/필터 없는 단독추론. 재학습한 것은 YOLO 학생이며 보정기를 또 학습한 것이 아니다. 학생 실사 target에 수동GT 좌표를 직접 넣지 않았지만 보정기는 이19장의 수동GT로 학습되었으므로 완전 무라벨 학습이 아닌 보정 지식 증류다.',
        '- 2D 평가는2347코너; 이미지 제외 없음. PCK는 매칭실패 penalty 포함, median/P90은 매칭 성공 코너만. 중심점은 코너 평가에서 제외.',
        '- 6D는 동일 prediction-only W/D selector+SQPnP/LM, canonical 물리축/C2대칭. GT로 축 선택 없음. ADD AUC는 객체별 대각선 정규화 후0~0.1적분, 성공률 아님. PoseCov=계산가능 비율이지 정답률 아님.',
        '- 6D참조는 어노테이션에서 기하학적으로 복원한 값으로 독립 실측GT가 아니다. 이전 CF-only 표와 수치를 혼합하지 않는다. AP/negative오검출은 이번 학생에 대해 새로 측정하지 않았다.', '',
        '## 검증','',
        '두 학생 각각320 optimizer update·R0 초기화일치 확인. 수도레이블171좌표 정규화 왕복검사 통과. R0 300장 재추론 좌표차이0px 및 점수 재현 확인. 평가예측은 채점 전 고정. 모델선택·GT변경 없음. 보정 없는 수도레이블 학생 대조군은 없으므로 개선 원인을 보정 단독효과로 주장하지 않는다.', '']
    for key,title in groups.items():
        lines += ['## '+title,'','### 2D','', '| 방법 | PCK5↑ | PCK10↑ | PCK20↑ | median px↓ | P90 px↓ | >20px 코너↓ | 매칭↑ |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm,label in names.items():
            s=two[key][arm]
            lines.append(f'| {label} | {100*s["PCK"]["5"]:.2f}% | {100*s["PCK"]["10"]:.2f}% | {100*s["PCK"]["20"]:.2f}% | {s["matched_pooled_corner8_median_px"]:.2f} | {s["matched_pooled_corner8_P90_px"]:.2f} | {round(s["gross20"]*s["corners"])} | {s["matched"]}/{s["total_frames"]} |')
        lines += ['', '### 6D','', '| 방법 | PoseCov↑ | AxisAcc↑ | R med°↓ | Yaw med°↓ | t med cm↓ | IoU3D med↑ | ADDsym AUC↑ |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm,label in names.items():
            s=six[key][arm]
            lines.append(f'| {label} | {100*s["coverage"]:.2f}% | {100*s["axis_accuracy"]:.2f}% | {s["rotation_deg"]["median"]:.3f} | {s["yaw_deg"]["median"]:.3f} | {s["translation_cm"]["median"]:.3f} | {s["IoU3D"]["median"]:.4f} | {s["ADDsym_AUC_full"]:.4f} |')
        lines.append('')
    lines += ['## 난도별 실제 예측 비교','',
        '왼쪽 R0 / 가운데 보정 적용 / 오른쪽 학생. 노랑은 예측 모서리, 초록 점은 평가 reference 위치(번호 대응은 표시하지 않음). GT는 그림/채점에만 사용했다. 각 난도에서 세 모델 모두 검출매칭된 이미지 중 학생-R0 평균코너오차 차이가 가장 작은/큰 사례를 각각 표시했다. 평가 후 설명용 선택이며 대표 평균이나 학습 선택 기준이 아니다.']
    truth=S.P.read(S.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json')
    edges=[(0,1),(1,5),(5,4),(4,0),(3,2),(2,6),(6,7),(7,3),(0,3),(1,2),(5,6),(4,7)]
    selections=[]
    for sev in S.P.SEVERITIES:
        candidates=[r for r in rr if r['severity']==sev and all(metrics[a][r['id']]['matched'] for a in names)]
        candidates.sort(key=lambda r:metrics['STUDENT'][r['id']]['frame_mean_px']-metrics['R0'][r['id']]['frame_mean_px'])
        for tag,r in [('best',candidates[0]),('worst',candidates[-1])]:
            fid=r['id'];im=cv2.imread(str(S.ROOT/r['image']['path']));scale=480/im.shape[1];panels=[]
            for arm in names:
                view=cv2.resize(im,(480,round(im.shape[0]*scale)));p=S.P.C.selected(preds[arm][fid]);q=np.array(p['keypoints_xy'])*scale
                for a,b in edges:
                    if np.isfinite(q[[a,b]]).all():cv2.line(view,tuple(q[a].astype(int)),tuple(q[b].astype(int)),(0,220,255),1,cv2.LINE_AA)
                g=np.array(truth[fid]['gt'])*scale
                for i in range(8):
                    if truth[fid]['valid'][i] and np.isfinite(g[i]).all():cv2.circle(view,tuple(g[i].astype(int)),3,(40,230,40),-1)
                view=cv2.copyMakeBorder(view,44,0,0,0,cv2.BORDER_CONSTANT)
                label=f'{arm} mean {metrics[arm][fid]["frame_mean_px"]:.2f}px'
                cv2.putText(view,label,(6,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.putText(view,fid,(6,35),cv2.FONT_HERSHEY_SIMPLEX,.33,(255,255,255),1);panels.append(view)
            name=f'{sev.lower()}_{tag}.jpg';dest=doc/'images'/name
            assert not dest.exists();assert cv2.imwrite(str(dest),np.hstack(panels))
            selections.append(dict(severity=sev,selection=tag,id=fid,image=S.C.bound(dest),reference_display_only=True))
            lines += ['',f'### {groups[sev]} — {tag} 사례','',f'![{sev} {tag}](images/{name})','']
    lines += ['## 검출매칭 변화까지 포함한 개선·악화 사례','',
        '아래는 종류별 개선 최대2장·악화 최대2장. GT점 없이 예측만 표시. 800px 표시는 실제 이동량이 아니라 box IoU<0.5일 때 이미지 대각선으로 부여한 penalty다.', '',
        '![일반 플라스틱](images/plastic.jpg)','', '![목재](images/wood.jpg)','',
        '## 상세 기록과 재현 코드','',
        '- [혼합19장 보정기 결과](../pallet_replay_clean19_v1/SUMMARY_KO.md)',
        '- [종류별 보정기 결과](../pallet_replay_by_type_v1/RESULTS_KO.md)',
        '- [비가림 보정→숨은점PnP 결과 및 적용수](../pallet_visible_refine_hidden_pnp_v1/RESULTS_KO.md)',
        '- [학생2D 상세](RESULTS_KO.md) / [학생6D 상세](pose/RESULTS_KO.md)',
        '- [학생 실행 코드](../../../scripts/research/pallet_clean19_student_v1.py) / [6D평가](../../../scripts/research/pallet_clean19_student_pose_v1.py) / [이 보고서 생성](../../../scripts/research/pallet_clean19_publish_report.py)', '',
        '원본 데이터·체크포인트·대용량300장 로컬갤러리는 Git에 포함하지 않는다. 재현 시 기존 로컬 데이터/모델이 필요하며 프로토콜JSON에 경로·해시가 기록되어 있다. 이 게시물만으로 전체 학습을 재현할 수 있는 독립 데이터 배포는 아니다.']
    S.save(doc/'REPORT_KO.md','\n'.join(lines)+'\n')
    S.save(doc/'REPORT_IMAGE_SELECTION.json',selections)
    print(doc/'REPORT_KO.md',flush=True)


if __name__=='__main__':main()
