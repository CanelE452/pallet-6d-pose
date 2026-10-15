"""Postseal scoring of the two prespecified fixed-observation XY controls."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import numpy as np
from . import scoring as S
from .batch import DIAGNOSTIC_NAMES, rows, write
from .statistics import distribution


def run(*,folder,input_dir,score_dir,source_root,reference_manifest,output,
        expected_prediction_seal_sha256,expected_score_seal_sha256):
    folder,input_dir,score_dir,source_root,output=map(Path,(folder,input_dir,score_dir,source_root,output))
    if output.exists():raise FileExistsError('Preserve every coordinate analysis attempt')
    compact,main_seal,protocol=S.prediction_gate(folder=folder,input_dir=input_dir,
        expected_prediction_seal_sha256=expected_prediction_seal_sha256)
    if S.sha(score_dir/'SCORE_SEAL.json')!=expected_score_seal_sha256:
        raise ValueError('Same complete main score seal required')
    score_seal=S.read(score_dir/'SCORE_SEAL.json')
    if score_seal['prediction_seal_sha256']!=expected_prediction_seal_sha256:
        raise ValueError('Main prediction/score ancestry changed')
    for name,h in score_seal['assets'].items():
        if S.sha(score_dir/name)!=h:raise ValueError('Main score changed')
    path=folder/'COORDINATE_DIAGNOSTIC_PREDICTIONS.jsonl.gz'
    diagnostic=list(rows(path));wanted=[r['id']for r in compact if r['partition']!='CAL']
    if len(diagnostic)!=192 or [r['id']for r in diagnostic]!=wanted:
        raise ValueError('All192 paired coordinate controls, no selection permitted')
    if protocol['coordinate_diagnostics']!=list(DIAGNOSTIC_NAMES):
        raise ValueError('Prespecified diagnostic names changed')
    output.mkdir(parents=True)
    write(output/'DIAGNOSTIC_PREDICTION_SEAL.json',dict(names=list(DIAGNOSTIC_NAMES),
        frame_rows=192,branch_rows=384,ordered_ids=wanted,prediction_sha256=S.sha(path),
        main_prediction_seal_sha256=expected_prediction_seal_sha256,
        protocol_sha256=main_seal['protocol_sha256'],
        fixed_post_observation_states_covariance_support_and_solver=True,
        replaced_only_xy_with_own_precorrection_prediction=True,before_diagnostic_scoring=True,
        natural_independent_no_correction_baseline=False,new_RGB_or_state_mask_forward_calls=0))
    manifest=S.read(reference_manifest)
    if manifest['input_seal_sha256']!=main_seal['input_seal_sha256']:
        raise ValueError('Same fixed reference ancestry required')
    references,_=S.read_fixed_references(source_root=source_root,manifest=manifest,compact=compact)
    values=S.score_compact(diagnostic,references)
    main={r['id']:r for r in S.read(score_dir/'SCORED_ROWS.json')['rows']}
    summary={}
    for arm,name in zip(('R1','N1'),DIAGNOSTIC_NAMES):
        controls=[r['metrics'][arm]for r in values]
        current=[main[r['id']]['metrics'][arm]for r in values]
        transitions=Counter()
        for a,b in zip(current,controls):
            transitions[f"main_{a['success_le5']}_replaced_{b['success_le5']}"]+=1
        summary[name]=dict(N=192,main_successes=sum(x['success_le5']is True for x in current),
            replaced_successes=sum(x['success_le5']is True for x in controls),
            transitions=dict(transitions),
            main_errors={k:distribution([r[k]for r in current])for k in('translation_cm','rotation_deg')},
            replaced_errors={k:distribution([r[k]for r in controls])for k in('translation_cm','rotation_deg')},
            main_fallback_n=sum(main[r['id']]['prediction_status'][arm]['fallback']is True for r in values),
            replaced_fallback_n=sum(r['prediction_status'][arm]['fallback']is True for r in values))
    write(output/'COORDINATE_DIAGNOSTICS.json',dict(names=list(DIAGNOSTIC_NAMES),summary=summary,rows=values,
        caution='Conditional on the same postcorrection observation IDs/states/support/covariance; not a natural baseline or causal proof',
        not_a_main_comparison_or_selection_candidate=True,primary_Holm_family_member=False))
    lines=['# 두 좌표 교체 진단','',
        '자기 보정 후 상태·B/I ID·지지·공분산·solver를 유지하고, xy만 자기 보정 전 좌표로 바꿨다. '
        '192장×2 초기화=384 pose-only 계산이며 새 영상·상태·마스크 계산은0이다. '
        '보정 후 관측에 조건화된 진단으로 자연스러운 독립 무보정 기준선이나 전체 메커니즘의 인과 증명은 아니다.','',
        '| 초기화 | 주경로 성공/192 | xy 복귀 성공/192 | 주경로만 성공 | 복귀만 성공 | 주경로 위치평균 cm | 복귀 위치평균 cm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in summary.items():
        t=v['transitions'];lines.append(f"| {name.split('_')[0]} | {v['main_successes']} | {v['replaced_successes']} | "
            f"{t.get('main_True_replaced_False',0)} | {t.get('main_False_replaced_True',0)} | "
            f"{v['main_errors']['translation_cm']['mean']:.4f} | {v['replaced_errors']['translation_cm']['mean']:.4f} |")
    (output/'COORDINATE_DIAGNOSTICS_KO.md').write_text('\n'.join(lines)+'\n')
    write(output/'DIAGNOSTIC_SCORE_SEAL.json',dict(assets={p.name:S.sha(p)for p in output.iterdir()if p.is_file()},
        source_sha256={Path(__file__).name:S.sha(__file__)},main_score_seal_sha256=expected_score_seal_sha256))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in('folder','input-dir','score-dir','source-root','reference-manifest','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prediction-seal-sha256',required=True);p.add_argument('--score-seal-sha256',required=True)
    a=p.parse_args();print(run(folder=a.folder,input_dir=a.input_dir,score_dir=a.score_dir,source_root=a.source_root,
        reference_manifest=a.reference_manifest,output=a.output,expected_prediction_seal_sha256=a.prediction_seal_sha256,
        expected_score_seal_sha256=a.score_seal_sha256))
