"""Generate the bounded-stop report; do not fabricate downstream student results."""
from .audit_mechanism import DOC,POINT,DHT,ROOT,read_json


def report():
    a=read_json(DOC/'LINE_MECHANISM_AUDIT.json');q=read_json(DOC/'PSEUDO_SUPERVISION_QUALITY.json')
    final=read_json(DOC/'FINAL_AUDIT.json');independent=read_json(DOC/'INDEPENDENT_GRADIENT_AUDIT.json')
    detection=read_json(DOC/'BASELINE_DETECTION_AUDIT.json')['synthetic_R0']
    assert final['PASS'] and independent['PASS']
    lines=['# DHT pseudo-line self-training v1 — Stage A에서 종료','',
        '## [관찰]','',f'최종 판정: `{a["status"]}`. 사용자가 사전 지정한 teacher-source mechanism gate를 통과하지 못해 학생 C0/C1/C2 × seed1/2/3의9-fit을 실행하지 않았다. optimizer updates는 전부0이다.', '',
        '| 모집단 | frame N | common edge N | P-line 가중 평균 px | DHT-line 가중 평균 px | DHT better % | frame-bootstrap95% CI | 평균 gradient cosine | positive alignment % |',
        '|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for split,s in a['splits'].items():
        lo,hi=s['better_edge_fraction_frame_bootstrap_CI95']
        lines.append(f'| {split} | {s["frame_N"]} | {s["N"]} | {s["weighted_P_mean_px"]:.6f} | {s["weighted_DHT_mean_px"]:.6f} | {100*s["better_edge_fraction"]:.3f} | {100*lo:.3f}–{100*hi:.3f}% | {s["mean_gradient_cosine"]:.6f} | {100*s["positive_alignment_fraction"]:.3f} |')
    lines += ['', '선 오차는 동일GT endpoint와 동일DHT soft weight를 사용한 normal incidence distance다. P-line/DHT-line 비교에 GT로 teacher edge를 선택하지 않았다. 양쪽에 공통으로 평가 가능한edge만 line quality 통계에 사용했다. 실제line-loss weight에는GT가 들어가지 않는다.', '',
        'Stage A gate는 synth_val 512장에서 cosine>0 AND DHT 가중 선 오차<P AND better-edge CI 하한>.5다. cosine 조건만 통과했다. 선 각도/ambiguity/Point-error/source/asset/C1-C2/edge별 통계는 SUBGROUP_RESULTS.json에 N과 함께 기록했다.', '',
        '| 합성 Point pseudo-label 진단 | frame N | pooled entry N | median px | P90 px | gross20 % | frame mean/diag |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split,quality in q.items():
        p=quality['Point'];lines.append(f'| {split} | {p["N_frames"]} | {p["N_keypoints"]} | {p["median_px"]:.6f} | {p["p90_px"]:.6f} | {100*p["gross20_rate"]:.4f} | {p["frame_mean_normalized"]:.9f} |')
    s=q['synth_val'];d=s['DHT_line']['DHT_line']
    lines += ['', 'pooled entry N은 no-evaluable-target frame당 diagonal penalty 8개를 포함하며, 관측 keypoint 수와 다르다.']
    for split,det in detection.items():
        lines.append(f'R0 {split}: detection {det["stock_detected_frames"]}/{det["N"]}, matched {det["stock_matched_frames"]}/{det["N"]}, covered target corners {det["covered_target_corners"]}/{det["target_corner_count"]}, no-target penalty frame {det["no_evaluable_target_frames"]}.')
    lines += ['',f'Test DHT 선 오차(unweighted)는 mean/median/P90={d["mean"]:.6f}/{d["median"]:.6f}/{d["p90"]:.6f}px. available edges={s["available_edges"]}/{s["total_edges"]}, weight mean/median/P90={s["line_weight"]["mean"]:.6f}/{s["line_weight"]["median"]:.6f}/{s["line_weight"]["p90"]:.6f}.', '',
        f'선의 normal 방향 성분에서 teacher residual이 GT 오차를 줄이는 방향인 비율은 {100*s["normal_help_fraction"]:.3f}%, 반대 방향 {100*s["normal_harm_fraction"]:.3f}% (nonzero evaluated endpoint N={s["normal_endpoint_N"]}). 학생 parameter-space의 효과나 실제 정확도 개선과 같지 않다.', '',
        '### Teacher 및 데이터','',
        f'- Point: `{POINT.relative_to(ROOT)}`',f'  SHA `{final["Point_before_sha256"]}`',
        f'- DHT: `{DHT.relative_to(ROOT)}`',f'  SHA `{final["DHT_before_sha256"]}`',
        '- 두 teacher의 before/after SHA 동일. synthetic cached P/line의 기준 checkpoint도 확인했다.',
        '- unlabeled real pool 1000장(주간500/야간500), 실제 파일 해시 검증. 기존 V3A pseudo-point 273장, C1/C2가 동일 label 파일을 사용하도록 고정. 이 실험의 실제 real training exposure는0.',
        '- PAPER_EVAL319+negative2689의 membership image hash/filename과 pool 교집합0. 실사 GT annotation을 열지 않았고 real GT access count0.',
        '- synthetic pseudo-line cache 256+512장에는 GT 필드 없음. real DHT line cache는 Stage A 실패로 생성하지 않았다.', '',
        '### 실행·검증','',
        '- C0/C1/C2 각 seed1/2/3: actual updates0. 학습/평가 미실시이며 R0와의 개선율, C1-C0, C2-C1, C2-R0 모두 N/A.',
        '- λ_line / g_point / g_line / achieved ratio: N/A. gate 통과 후 실행하는 parameter-gradient calibration이므로 실행하지 않았다. output-gradient norm으로 대체하지 않았다.',
        '- 20개 회귀검사 통과. exposure/order와 student-only optimizer/stock inference graph 검사는 static contract이며 full trainer 실행 증거가 아니다.',
        '- 초기 진단에서 ignored GT NaN의 autograd 오염을 발견했고 엄격 JSON 출력이 실패했다. invalid GT를 미분 전에 sanitization하는 수정과 회귀검사만 추가했다. gate/weight/teacher/cache는 바꾸지 않았다. 수정 전후 선 cache는 byte-identical이다.',
        '- 최종 768frame gradient/cosine/virtual step을 독립 NumPy 해석 gradient로 검증했다. finite output 및 허용 오차 통과.',
        f'- 원본 산출물 {final["source_artifacts_preserved"]}개 보존 해시 통과. paper stop lock/최종 paper 문서/V1–V5 불변.',
        '- GPU 상태는 실행 중 확인하고 GPU_*.json에도 기록했다. 다른 프로젝트의 GPU 작업을 종료하거나 변경하지 않았다. 재부팅/시스템 driver 변경 없음.', '',
        '## [해석]','',
        'DHT line의 normal 방향 gradient는 일부 GT point 오차를 줄이는 방향이었지만, 고정된 soft weighting으로 전체 available semantic line을 보조 교사로 쓸 때 Point에서 직접 유도한 선보다 좋은 source라는 사전 조건을 만족하지 못했다. 따라서 이번 teacher/weight/population의 Stage A를 종료한다. 새 weight/threshold/teacher/epoch 탐색으로 전환하지 않는다.', '',
        '## [미확정]','',
        '학생을 학습하지 않았으므로 「DHT를 더한 학생이 Point-ST보다 나쁘다」 또는 「DHT privileged teacher가 불가능하다」고 결론낼 수 없다. output-space alignment와 학생 weight-space gradient/후속 정확도는 별개다. pseudo-supervision 품질과 학생 성능도 동일하지 않다.', '',
        'real DEV/PAPER_EVAL/FINAL 평가 미실행, canonical 6D 미평가. untouched target evaluation을 사용하지 않았고 증거 등급은 POSTHOC_DEVELOPMENT_ONLY다. confirmatory/held-out/unseen-pallet/6D 개선 주장을 할 수 없다. paper-facing claim/table/abstract는 자동 수정하지 않았다.', '',
        '학생 primary를 측정하지 않았으므로 이전 P/Q oracle2.18%와의 비교나 「2% 초과」 판정은 N/A다. 이번 기록은 선 teacher mechanism의 negative development evidence로만 활용할 수 있다.']
    (DOC/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':report()
