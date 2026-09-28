"""Render traceable aggregate tables and CLI handoff; never fit or alter main."""
from collections import Counter
import subprocess
from . import common as C


def main():
    c2=C.read(C.DOC/'cycles/C2_REAL_AFFINE_OFF/RESULTS.json')
    c3=C.read(C.DOC/'cycles/C3_MANUAL38_CAPABILITY/RESULTS.json')
    ledger=C.read(C.DOC/'RESOURCE_LEDGER.json')
    assert ledger['totals']['fits']==6 and ledger['totals']['optimizer_updates']==1920
    rows=[];deltas=[]
    for mat in ('PLASTIC','WOOD'):
        base=c2['materials'][mat]['groups']['ALL']
        extra=c3['materials'][mat]['groups']['ALL']
        for arm in ('R0','OLD_RAW','OLD_REF','SYN','NEW_RAW','NEW_REF','RAW9','MANUAL9'):
            v=(extra if arm in ('RAW9','MANUAL9') else base)[arm];q=v['twoD'];p=v['sixD']
            rows.append([mat,arm,f"{q['correct']['10']}/{q['corners']}",f"{100*q['PCK']['10']:.4f}",
                         f"{100*q['PCK']['5']:.3f}",f"{100*q['PCK']['20']:.3f}",
                         f"{q['full_penalty_median_px']:.3f}",f"{q['full_penalty_P90_px']:.3f}",
                         q['tail_gt20_count'],f"{p['ADDsym_AUC']:.9f}",f"{p['pose_coverage']:.3f}"])
        for cycle,l,r,data in [('C2','NEW_RAW','NEW_REF',base),('C2','OLD_REF','NEW_REF',base),
                              ('C2','R0','NEW_REF',base),('C3','RAW9','MANUAL9',extra),
                              ('C3 descriptive','OLD_REF','MANUAL9',extra),('C3 descriptive','R0','MANUAL9',extra)]:
            a,b=data[l],data[r]
            deltas.append([mat,cycle,f'{r}−{l}',b['twoD']['correct']['10']-a['twoD']['correct']['10'],
                f"{100*(b['twoD']['PCK']['10']-a['twoD']['PCK']['10']):+.4f}",
                f"{b['sixD']['ADDsym_AUC']-a['sixD']['ADDsym_AUC']:+.9f}"])
    train=c3['TRAIN_capability']['groups']
    train_rows=[[group,arm,v['MANUAL9']['corners'],f"{v['RAW9']['mean_px']:.4f}",
                 f"{v['MANUAL9']['mean_px']:.4f}",f"{100*v['MANUAL9']['PCK']['10']:.3f}"]
                for group,models in train.items() for arm,v in models.items()]
    text=['# 최종 수치 연결표','',
          'SOURCE: C2/C3 RESULTS.json. NEW_RAW/REF=C2 실사 affine OFF. RAW9/MANUAL9=C3의 별도 혼합재료9장·38점 support 대조다. 서로 다른 학습 모집단의 절대값은 참고용이며 같은 인과 비교로 합치지 않는다. median/P90은 full-denominator mismatch penalty 포함.', '',
          C.table(['Material','arm','correct10/N','PCK10 %','PCK5 %','PCK20 %','median px','P90 px','>20px','D9 AUC','pose coverage'],rows),
          '', '## 세 가지 기준과 감독 대안의 차이','',
          C.table(['Material','cycle','contrast','correct10 delta','PCK10 pp','AUC delta'],deltas),
          '', '## C3 finite TRAIN 적합성','',
          C.table(['group','arm','manual points','raw-target mean px','manual-target mean px','manual-target PCK10 %'],train_rows),
          '', 'TRAIN38은 이미 교사가 사용한 직접 클릭 정보다. RAW9도 동일 manual support를 쓰며 center와 PnP 보완점은 ignore한다. 물리 signed-axis나 독립 reference 검증으로 격상하지 않는다.',
          '', '전체 severity/recording, PCK5/20, paired 진입·이탈, rotation/yaw/translation/axis/IoU, LORO는 각 cycle RESULTS.json에 보존한다. Wood severe는 NA, 모든 결과는 반복 DEV다.']
    C.save(C.DOC/'FINAL_TABLES.md','\n'.join(text)+'\n')
    sources=C.read(C.DOC/'SOURCE_REGISTRY.json')['sources']
    levels=Counter(s['read_status'] for s in sources)
    audit=C.read(C.DOC/'AUDIT.json') if (C.DOC/'AUDIT.json').exists() else {}
    c3rows=c3['materials'];summary=[]
    for mat in ('PLASTIC','WOOD'):
        g=c3rows[mat]['groups']['ALL']
        summary.append(f"{mat}: "+' / '.join(f"{a} {g[a]['twoD']['correct']['10']}/{g[a]['twoD']['corners']}, AUC {g[a]['sixD']['ADDsym_AUC']:.9f}" for a in ('R0','OLD_REF','RAW9','MANUAL9')))
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()
    cli=[
        '# 최종 CLI 보고 — 실행 수치 및 공개 범위','',
        'STATUS: THREE_CYCLES_COMPLETED_NO_METHOD_PROMOTION',
        'HEAD_START: '+C.read(C.DOC/'INPUT_BINDINGS.json')['head_start'],
        'HEAD_AT_REPORT_RENDER: '+head,
        'REMOTE_TRACKING_AT_RENDER: '+remote,
        'HEAD_END / REMOTE_HEAD: 최종 push 후 실제 조회 결과는 사용자 최종 응답 및 private RELEASE_VERIFICATION.json에 기록한다. 이 파일을 포함하는 commit은 git log -1 -- 이파일경로로 확인한다.',
        '', 'RESEARCH_QUESTION: 보정 자기학습의 제한된 이득을 타깃/학습/선택/참조로 구분하고 한정된 실제 개입으로 회수할 수 있는가?',
        'WHAT_PRIOR_RESULTS_ACTUALLY_RULED_OUT: 같은 낮은LR 연장·기존 local/융합/필터 설정의 제한된 효과. 모든 최적화/표현/selector의 불가능성은 아님; PRIOR_ATTEMPTS 30항목 참조.',
        '', '## CURRENT_BASELINES', '',
        'PLASTIC_R0_RAW_REF: 484/468/507 of985; AUC .337964844/.334722656/.359015625.',
        'WOOD_R0_RAW_REF: 167/163/165 of346; AUC .670500/.656433333/.665033333.',
        '', '## ORACLE_RESULTS', '',
        'FIXED_CANDIDATE_POSE_BY_MATERIAL: REF Plastic .359015625→.450093750; Wood .665033333→.667400000.',
        'WHOLE_OUTPUT_TEACHER_STUDENT_ORACLE: whole pose Plastic .441824219 / Wood .710688889; whole2D589/985,210/346.',
        'PER_POINT_ORACLE: 613/985,228/346; verified66 whole/point53/66; coordinate mixtures not reported as rigid6D.',
        'LOCAL_MOVE_CEILING_IF_RELEVANT: GT-direction8px REF700/985,252/346; not image-based attainability.',
        'REFERENCE_INPUT_GEOMETRY_CHECK: legacy xy→pose AUC1 circular; exactsynthetic64 production .905797; privileged renderer-correspondence solvermaxreprojection8.13e-6px.',
        'PRIVILEGED_TRAIN_CAPABILITY_OR_NA: C3 existing manual9/38; see FINAL_TABLES and C3 REPORT.',
        'ORACLE_NOT_INTERPRETABLE_ITEMS: synthetic physical-axis/camera-facing mismatch NA; Wood verified-visible NA; no GTcrop or unsupported pointcompletion.',
        '', '## DIAGNOSIS', '',
        'TARGET_QUALITY: verified66 teacher50 vsREF43, teacher-only10/student-only3 at10px; Wood trustedvisible unresolved.',
        'TRAINING_SIGNAL_TRANSFER: partial; correctedTRAIN residual Plastic RAW4.078→REF3.070px, Wood3.689→3.325px; not physicalGT accuracy.',
        'AUGMENTATION_OR_SOURCE_CONFLICT: source/real gradient signs mixed; huge affineprobeoutlier is instance switch. C2 primary improvement absent.',
        'CANDIDATE_GENERATION: existing set only; no claim about all possible representations.',
        'CANDIDATE_SELECTION: Plastic W/D gap meaningful, Wood REF gap too small to exceedR0; C1 recovered0%.',
        'REFERENCE_OR_IDENTIFIABILITY: repeatedDEV/legacygeometry, missingWooddirectclickprovenance; C3 unconfirmedphysicalsignedaxes.',
        '', '## LITERATURE', '',
        'PRIMARY_SOURCES_READ: '+str(dict(levels)),
        'ABSTRACT_ONLY_OR_BLOCKED: '+', '.join(s['id'] for s in sources if 'ABSTRACT' in s['read_status']),
        'SELECTED_PRINCIPLES: fixed robust residual; controlled augmentation; direct same-budget TRAIN capability.',
        'REJECTED_AND_WHY: no calibrated localization-reliability evidence, mixed gradient evidence, CAD/render/depth/extra framework prerequisites; not blanket paper failure.',
        '', '## EXECUTED_CYCLES', '',
        'C1: 0fits; same candidates, scoreonly, deltaAUC0.',
        'C2: 4fits×320; paired Plastic/Wood real-affine-off.',
        'C3: 2fits×320; RAW9/MANUAL9 existing38point support.',
        f"PAIRED_NEW_FITS: {ledger['totals']['fits']}; OPTIMIZER_UPDATES: {ledger['totals']['optimizer_updates']}",
        f"GPU_HOURS: {ledger['totals']['gpu_seconds']/3600:.6f}; WALL_HOURS_AT_LEDGER: {ledger['totals']['elapsed_wall_seconds']/3600:.6f}",
        'TECHNICAL_FAILURES_AND_RETRIES: EXPERIMENT_LOG and C3 technical records; no result-driven refit. Sandbox failure before model/optimizer0step distinguished from training.',
        '', '## RESULT', '',
        'GAIN_VS_RAW: C2 Plastic+34/985,+.026992188AUC; Wood−3/346,+.007511111AUC.',
        'GAIN_VS_OLD_REF: C2 Plastic−4/985,−.000281250; Wood−1/346,−.002466667.',
        'GAIN_VS_R0: C2 Plastic+19/985,+.020769531; Wood−3/346,−.007933333.',
        *summary,
        'COST_AND_HARMS: sixfits; sameexistingmanualbudget butC3 directstudentpath; allseverity/tail/recordingretained. No hiddenbestcheckpoint/threshold/seed selection.',
        'ATTAINABLE_RECOVERY_DEMONSTRATED: C1 fixedset0%; C2 no gain overoldREF primary. C3 is a distinct finite supervised control; interpretation in REPORT.',
        'REMAINING_HEADROOM_AND_MISSING_INFORMATION: GT-free discriminating cue and out-of-sample targetquality/independentreference remain unresolved.',
        'GENERALIZATION_STATUS: REUSED_DEV_ONLY.',
        '', '## FINAL_FOUR_AXES', '',
        'EVIDENCE_VALIDITY: LIMITED — paired computations valid, repeatedDEV and reference scope limited.',
        'HEADROOM: MEASURED — within stated sets/metrics only.',
        'RECOVERY: NOT_DEMONSTRATED against existingREF; C3 TRAIN fitting partial, DEV effects material-mixed. No automatic best-run promotion.',
        'CAUSE: MULTIPLE_EXPLANATIONS.',
        '', 'WHAT_WAS_RESOLVED: currentoracleheadroom, partialTRAINimitation, testedHuber/affineeffects, finite9/38capability.',
        'WHAT_WAS_NOT_RESOLVED: physicaltargettruth, robustGT-freeselection, independentgeneralization, separatecapacity/optimization causality.',
        'WHAT_MUST_NOT_BE_CLAIMED: oracleattainability, allmaterialgeneralization, allaugmentationfailure, physical6Dclosure.',
        'NEXT_ONE_DECISION: STOP this3-cyclebatch; anynextstudy asks whether availableRGBcan discriminate complementaryfrozenoutputs on eligibleTRAINwithoutDEVlabels.',
        'HUMAN_ACTION_REQUIRED: NO.',
        'REPORT: REPORT_KO.md; FINAL_TABLES.md; AUDIT.json; REPRODUCE.md.',
        'COMMITS_AND_PUSH: diagnosis4bc8e41a and C2 6fb81b1e already verifiedremote; finalrelease verifiedafter commit.',
        'GIT_STATUS: original untracked files preserved; only followupnamespace tracked/staged changes intended.',
        '', '현재 REF의 W/D 선택 단계에 Plastic AUC .091078 / Wood .002367의 여지가 관측됐고, Huber12 방법은 그중0%를 회수했다. 정답을 모르는 입력만으로 이 후보를 고르는 방법과 독립 실사 일반화는 아직 확인하지 못했다.',
    ]
    C.save(C.DOC/'CLI_REPORT_KO.md','\n\n'.join(cli)+'\n')
    print('FINAL_TABLES_AND_CLI_RENDERED')


if __name__=='__main__':main()
