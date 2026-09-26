"""Freeze reserve provenance, manuscript numerical bindings and build locally."""
import argparse
import csv
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from . import common as C

def prepare():
    assert (C.DOC/'TABLE_FIGURE_FREEZE.json').exists()
    for b in C.read(C.DOC/'TABLE_FIGURE_FREEZE.json')['tables']:C.verify(b)
    hist=C.ROOT/'_docs/experiments/pallet_sensors_submission_v1'
    old=C.read(hist/'CONFIRMATION_COMPLETE.json');incoming=Path(old['incoming_path'])
    manifest=C.ROOT/'data/evaluation/pallet_eval_v1/manifests/FINAL_POSITIVE.csv'
    with manifest.open() as f:final=list(csv.DictReader(f))
    seen={r['session'] for r in C.read(C.OLD/'EVAL_PROTOCOL.json')['records']}
    sessions=sorted({r['session_id'] for r in final});unknown=set(sessions)-seen
    assert not incoming.exists(), 'New confirmation panel exists: inspect provenance before deciding no independent data'
    assert not unknown, ('Physical FINAL contains unaudited sessions',unknown)
    prior=C.read(hist/'CONFIRMATION_READINESS.json')
    sources=[manifest,hist/'CONFIRMATION_COMPLETE.json',hist/'CONFIRMATION_READINESS.json',C.OLD/'EVAL_PROTOCOL.json',C.ROOT/'data/evaluation/pallet_eval_v1/README.md']
    audit=dict(utc=C.now(),after_method_and_claim_freeze=C.bind(C.DOC/'TABLE_FIGURE_FREEZE.json'),
        status='NO_CERTIFIED_INDEPENDENT_PANEL_AVAILABLE',confirmation_executed=False,
        current_incoming_exists=False,physical_FINAL_rows=len(final),physical_FINAL_sessions=sessions,
        physical_FINAL_sessions_already_in_historical_evaluation=sorted(set(sessions)&seen),
        manifest_FINAL_name_does_not_imply_independence=True,
        previous_registered_confirmation_status=prior['status'],sources=[C.bind(p) for p in sources],
        scope='Audit of registered labeled reserves and current confirmation input, not proof that no uncatalogued RGB exists.',
        action='Finish DEV-only draft. Independent capture is optional for a future stronger claim; not a blocker for this scoped manuscript.',
        human_action_for_current_research=None)
    if (C.DOC/'INDEPENDENT_CONFIRMATION_AUDIT.json').exists():
        audit['utc']=C.read(C.DOC/'INDEPENDENT_CONFIRMATION_AUDIT.json')['utc']
    C.save(C.DOC/'INDEPENDENT_CONFIRMATION_AUDIT.json',audit,True)
    C.save(C.DOC/'INDEPENDENT_CONFIRMATION_REPORT_KO.md','# 독립 확인 데이터 감사\n\n현재 등록된 independent incoming panel은 없다. physical FINAL manifest의 '+str(len(final))+'행은 '+', '.join(sessions)+' 세션으로, 모두 과거 평가에 노출된 세션이다. FINAL_EVAL은 원래DEV의 실행 alias라는 저장소 계약도 유지한다. 파일명이 FINAL이라는 이유로 독립 test로 바꾸지 않았다.\n\n따라서 확인되지 않은 독립 성능을 만들지 않고 DEV-only 원고를 완성한다. 이 판단은 등록된 labeled reserve 감사이며 미등록 RGB가 전혀 없다는 뜻은 아니다. 새로운 촬영·어노테이션은 현재 원고를 완성하기 위한 필수 요청이 아니다.\n',True)
    res=C.read(C.DOC/'CORE_RESULTS.json')['groups']['ALL'];q=C.read(C.DOC/'PSEUDO_LABEL_QUALITY.json')['groups']['ALL'];d=C.read(C.DOC/'PAIRED_ANALYSIS.json')['REF_LR5-minus-RAW_LR5']
    values={}
    for prefix,arm in [('Base','R0'),('Raw','RAW_LR5'),('Corr','REF_LR5')]:
        values[prefix+'PCK']=(f'{100*res[arm]["twoD"]["PCK"]["10"]:.2f}','CORE_RESULTS.json',f'groups.ALL.{arm}.twoD.PCK.10 *100')
        values[prefix+'AUC']=(f'{res[arm]["sixD"]["ADDsym_AUC"]:.5f}','CORE_RESULTS.json',f'groups.ALL.{arm}.sixD.ADDsym_AUC')
        values[prefix+'Pninety']=(f'{res[arm]["twoD"]["matched_pooled_corner8_P90_px"]:.3f}','CORE_RESULTS.json',f'groups.ALL.{arm}.twoD.matched_pooled_corner8_P90_px')
    for prefix,arm in [('AnchorRaw','R0'),('AnchorCorr','TEACHER')]:
        values[prefix]=(str(q[arm]['PCK']['10']['correct']),'PSEUDO_LABEL_QUALITY.json',f'groups.ALL.{arm}.PCK.10.correct')
        values[prefix+'Med']=(f'{q[arm]["median_px"]:.3f}','PSEUDO_LABEL_QUALITY.json',f'groups.ALL.{arm}.median_px')
    values.update(StudentDelta=(f'{d["PCK10_delta_pp"]:.2f}','PAIRED_ANALYSIS.json','REF_LR5-minus-RAW_LR5.PCK10_delta_pp'),
        CILow=(f'{d["recording_bootstrap_CI95_pp"][0]:.2f}','PAIRED_ANALYSIS.json','REF_LR5-minus-RAW_LR5.recording_bootstrap_CI95_pp[0]'),
        CIHigh=(f'{d["recording_bootstrap_CI95_pp"][1]:.2f}','PAIRED_ANALYSIS.json','REF_LR5-minus-RAW_LR5.recording_bootstrap_CI95_pp[1]'))
    C.save(C.PAPER/'generated_tables/numbers.tex',''.join('\\newcommand{\\'+k+'}{'+v[0]+'}\n' for k,v in values.items()))
    C.save(C.PAPER/'generated_tables/MANUSCRIPT_NUMBER_PROVENANCE.json',{k:dict(display=v[0],source=C.bind(C.DOC/v[1]),json_path=v[2]) for k,v in values.items()})
    C.save(C.PAPER/'CLAIMS_KO.md','# 문장별 주장–근거\n\n'+C.table(['원고 문장/주장','근거/범위'],[
        ['보정 pseudo 가시점 PCK10 개선','PSEUDO_LABEL_QUALITY.json: FINAL_V2 fixed-ID44/66→50/66; 217 unlabeled 학습영상 자체의 GT 품질 측정은 아님'],
        ['동일 조건 corrected 학생 > raw 학생','CORE_COMPARABILITY_AUDIT.json + CORE_RESULTS.json: LR5 PCK10/공통D9 AUC; 모든LR/order도 보존'],
        ['R0보다 추가 가치','CORE_RESULTS.groups.ALL: PCK10/AUC 개선; 전체 지표 우월성 아님'],
        ['분산·실패·악화','PAIRED_ANALYSIS:92/28/8,51/12; TABLE3 severe/tail, TABLE6 recording'],
        ['실사 감독 공개','기존 teacher INPUT_LOCK/TRAIN_SUPPORT:9이미지38점; 66평가점과 분리'],
        ['순수 teacher-free raw 정책 비교 아님','RAW/REF 동일 보정후 승인집합. 좌표 intervention만 격리; 선택정책효과는 측정하지 않음'],
        ['독립 일반화/물리6D 검증 아님','INDEPENDENT_CONFIRMATION_AUDIT + geometry reference 계약'],
        ['신규기법 주장 아님','PoseFix/STAC/Self6D/BOP 원문 확인; application-specific controlled evidence'],
        ['Hard8/selector/occlusion main으로 섞지 않음','EXPERIMENT_ROLE_MAP.md']])+'\n각 수치의 source path/hash는 generated_tables/NUMBER_PROVENANCE.json 및 MANUSCRIPT_NUMBER_PROVENANCE.json에 있다.\n')
    C.save(C.PAPER/'EXPERIMENT_MAPPING.md',(C.DOC/'EXPERIMENT_ROLE_MAP.md').read_text()+'\n## Core mapping\n\nR0 / RAW_LR5 / REF_LR5 main. LR4 and ORDER43/44 complete sensitivity, SYN source-only controls. Same frozen Replay9/38 for Q1 and pseudo generation. Clean19 teacher never substituted. Newfits=0. Existing12 student fits reused.\n')
    C.save(C.PAPER/'LIMITATIONS.md','# Limits that remain in the paper\n\n- Reused DEV, seven recording groups; historical LR5 selection; no independent confirmation.\n- Ordinary plastic only, no wood/green transfer guarantee.\n-9-image/38-corner teacher manual adaptation;66 evaluation anchors extra; upstream generic pretraining and development history disclosed.\n- Raw control shares teacher-based selection and support; only the coordinate intervention is isolated.\n- No annotation-time efficiency or direct-manual-supervision superiority claim.\n- P90 and several pose/statified measures worsen; main student20→10px gross recovery count0.\n-16 selected images/66 visible points do not validate hidden points or unlabeled217 exact quality.\n- Legacy full-panel points have mixed provenance;6D is geometry-derived, not independent measured ground truth.\n- Pose-head-only adaptation; order repeats are not independent init-seed repeats.\n- Private data/checkpoints are hash-bound but not automatically redistributed as a full public dataset.\n')
    repro='''# Reproducibility

All commands run from repository root. Python: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`.

```bash
python -m scripts.research.pallet_selftraining_paper_closure_v1.prepare
python -m scripts.research.pallet_selftraining_paper_closure_v1.freeze_predictions
python -m scripts.research.pallet_selftraining_paper_closure_v1.evaluate
python -m scripts.research.pallet_selftraining_paper_closure_v1.report
python -m scripts.research.pallet_selftraining_paper_closure_v1.package prepare
python -m scripts.research.pallet_selftraining_paper_closure_v1.package build
python -m unittest scripts.research.pallet_selftraining_paper_closure_v1.test_closure
python -m scripts.research.pallet_selftraining_paper_closure_v1.audit
```

Completed expensive stages verify existing locks and do not rerun fitting. The closure includes NO training entry point. A fresh clone requires the private hash-bound datasets/checkpoints and frozen result caches, or regeneration from the historical training scripts. It is not an independently downloadable public benchmark. Do not overwrite historical artifacts to satisfy a changed hash.

Historical fit implementation: `scripts/research/pallet_type_selftrain_v1/recovery_pose.py` and `recovery_repeat.py`. Arguments and final checkpoint SHA256 for every arm: `CORE_COMPARABILITY_AUDIT.json`. Exact input paths, masks, source data, images, teacher, split, and evaluator dependencies: `INPUT_BINDINGS.json`, `PREDICTIONS_LOCK.json`, `PAPER_AUDIT.json`. The paper's 2D and6D values are generated, not copied from incompatible194/300-image historical reports.

CPU: existing environment, four evaluation workers. GPU is needed only for128 frozen Replay inference when no saved output exists; RTX3080 checked, thermal stop80C, observed48–54C. No reboot/driver changes or unrelated process termination.

Tectonic is reused from the existing local compiler. Its cache is COPIED to the new namespace; old paper and compiler cache are not modified. PDF build logs and rendered-page checks are retained in the new result namespace.

Numbers: generated_tables/NUMBER_PROVENANCE.json binds each table to source path/hash; MANUSCRIPT_NUMBER_PROVENANCE.json binds result macros to JSON paths. Counts that describe the protocol are in the comparability audit; transitions and per-frame categories are in PAIRED_ANALYSIS.json. MAIN128 corner denominator985, matched120frames/931corners, anchor16frames/66points remain distinct.
'''
    C.save(C.PAPER/'REPRODUCIBILITY.md',repro)
    C.save(C.PAPER/'SUBMISSION_READINESS.md','# Submission readiness\n\nEvidence closure: COMPLETE for a scoped DEV-only manuscript; method development STOP. Independent confirmation is NOT AVAILABLE and no number is invented. No new annotation is requested to complete this draft.\n\nThis is an author-review draft, not a submitted paper or acceptance guarantee. Authors must approve the actual author list/affiliations, funding and data-distribution declarations, journal fit, and the DEV-only claim scope before submission. Author identity is not fabricated. Strong unseen-session or physical-accuracy claims require genuinely independent evidence. The existing local-refiner paper was preserved, not overwritten.\n\nBuild and audit result: see PAPER_AUDIT.json and BUILD_RESULT.json in the experiment namespace.\n')
    refs=[('PoseFix','https://openaccess.thecvf.com/content_CVPR_2019/html/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.html'),('STAC','https://arxiv.org/abs/2005.04757'),('Self6D','https://arxiv.org/abs/2004.06468'),('BOP','https://arxiv.org/abs/1808.08319'),('BOP Challenge2023','https://openaccess.thecvf.com/content/CVPR2024W/CV4MR/html/Hodan_BOP_Challenge_2023_on_Detection_Segmentation_and_Pose_Estimation_of_CVPRW_2024_paper.html')]
    C.save(C.PAPER/'RELATED_WORK_SOURCES.md','# Verified primary sources\n\n'+''.join(f'- [{name}]({url})\n' for name,url in refs)+'\nChecked during this closure via primary publication/arXiv records. BOP2024 HTML open returned403, but the official CVF indexed bibliographic record provided author/title/pages. No benchmark numbers borrowed as matched evidence.\n')
    from .report import tab
    tab('TABLE7_verified_students',['Arm','PCK10','PCK20','Median px','P90 px'],[[a,f"{q[a]['PCK']['10']['correct']}/66",f"{q[a]['PCK']['20']['correct']}/66",f"{q[a]['median_px']:.3f}",f"{q[a]['p90_px']:.3f}"] for a in ('R0','RAW_LR5','REF_LR5')],
        'Verified-visible student sensitivity: both students tie at PCK10; the legacy-panel gain is not reproduced at this threshold on the small manually reverified subset. Same66 points, fixed identity.','verified')
    note='\n## 추가 근거 감사: 수동 재확인66점의 학생 결과\n\n'+(C.DOC/'TABLE7_verified_students.md').read_text().split('\n\n',1)[1]+'\n이 작은 subset에서는 학생 PCK10이 raw43/66, corrected43/66으로 동일하고 R0는44/66이다. 따라서 학생 개선 주장은 전체128장 기존 reference의 pooled결과에 한정한다. 보정기44→50/66과 학생43→43/66은 서로 다른 질문이다. 보정 학생의 PCK20과 tail은 이66점에서는 개선되지만 median은 소폭 악화된다. 이 민감도도 원고와 보고서에 명시하고 유리한 subset으로 평가를 대체하지 않는다.\n'
    current=(C.DOC/'REPORT_KO.md').read_text()
    if '## 추가 근거 감사:' not in current:C.save(C.DOC/'REPORT_KO.md',current+note)
    limits=(C.PAPER/'LIMITATIONS.md').read_text()
    C.save(C.PAPER/'LIMITATIONS.md',limits+'\n- Verified66 student PCK10 ties43/66 in both arms (R0=44/66); full128 legacy-label gain is not independently confirmed on that threshold.\n')
    matrix=C.read(C.DOC/'CLAIM_MATRIX.json');matrix['C2']['verified_visible_sensitivity']='PCK10 raw43/66=corrected43/66; no claim of improvement on every annotation reference'
    C.save(C.DOC/'CLAIM_MATRIX.json',matrix)
    numbers=C.read(C.PAPER/'generated_tables/NUMBER_PROVENANCE.json');numbers['table_mapping']['TABLE7']='PSEUDO_LABEL_QUALITY.groups.ALL student arms'
    numbers['tables']=[C.bind(p) for p in sorted((C.PAPER/'generated_tables').glob('TABLE*.tex'))]
    C.save(C.PAPER/'generated_tables/NUMBER_PROVENANCE.json',numbers)
    print('MANUSCRIPT_BINDINGS_AND_RESERVE_AUDIT_COMPLETE',len(final),sessions,flush=True)

def build():
    compiler=C.ROOT/'data/pallet/results/pallet_sensors_submission_v1/tectonic'
    cache=C.RAW/'tex_cache';old=C.ROOT/'data/pallet/results/pallet_sensors_submission_v1/tex_cache'
    if not cache.exists():shutil.copytree(old,cache)
    if not compiler.exists():
        C.save(C.PAPER/'BUILD_BLOCKED.md','Missing local Tectonic executable: '+str(compiler)+'\n');return
    env={**os.environ,'XDG_CACHE_HOME':str(cache)}
    run=subprocess.run([str(compiler),'--keep-logs','--keep-intermediates','manuscript.tex'],cwd=C.PAPER,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    C.save(C.RAW/'BUILD_LOG.txt',run.stdout);print(run.stdout[-6500:],flush=True)
    assert run.returncode==0, 'See BUILD_LOG.txt'
    pdf=C.PAPER/'manuscript.pdf';assert pdf.exists()
    subprocess.run(['pdftotext','-layout',str(pdf),str(C.RAW/'manuscript.txt')],check=True)
    text=(C.RAW/'manuscript.txt').read_text()
    log=(C.PAPER/'manuscript.log').read_text()
    errors=[pat for pat in ('undefined references','undefined citations','Citation .* undefined','Reference .* undefined','Missing character','Overfull') if re.search(pat,log)]
    assert '??' not in text and not re.search(r'TODO|PLACEHOLDER|\[xx\]',text)
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    C.save(C.DOC/'BUILD_RESULT.json',dict(complete=True,compiler=C.bind(compiler),pdf=C.bind(pdf),info=info,log=C.bind(C.RAW/'BUILD_LOG.txt'),warnings_to_inspect=errors))
    C.save(C.RAW/'PDF_INFO.txt',info)
    preview=C.RAW/'pdf_preview';preview.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm','-r','90','-png',str(pdf),str(preview/'page')],check=True)
    print('PDF_BUILT',info,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);a=p.parse_args()
    prepare() if a.action=='prepare' else build()
