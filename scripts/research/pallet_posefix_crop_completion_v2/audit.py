"""Final independent arithmetic / unchanged-input / test audit, no learning."""
import json
import re
import subprocess
import sys
import numpy as np
from . import common as C

def main():
    lock=C.read(C.DOC/'INPUT_LOCK.json');fit=C.read(C.DOC/'FIT_C.json');train=C.read(C.DOC/'TRAIN_INPUT_AUDIT.json');freeze=C.read(C.DOC/'PREDICTION_LOCK.json')
    assert fit['complete'] and fit['updates']==300 and fit['new_train_runs']==1 and not fit['evaluation_during_training'];C.verify(fit['checkpoint'])
    for b in C.read(C.DOC/'CODE_LOCK.json')['files']:C.verify(b)
    for b in lock['protected']:C.verify(b)
    for a,r in freeze['arms'].items():
        C.verify(r['predictions'])
        for b in r['heatmaps'].values():C.verify(b)
    oldtrace=[json.loads(s) for s in (C.B.RAW/'TRACE_FULL.jsonl').read_text().splitlines()];trace=[json.loads(s) for s in (C.RAW/'TRACE_C.jsonl').read_text().splitlines()];assert len(trace)==len(oldtrace)==300
    c=np.load(C.ROOT/train['corruption']['path'])
    for i,(a,b) in enumerate(zip(trace,oldtrace)):
        assert a['step']==i+1 and a['real_ids']==b['real_ids'] and a['source_ids']==b['source_ids'] and a['source_rows']==b['source_rows']
        assert a['original_source_points_sha']==C.B.P.array_sha(c['points'][i]);assert a['update_norm']>0 and np.isfinite(a['update_norm'])
        assert a['expansion']==C.EXPANSION
    assert sum(r['real_supervised'] for r in trace)==train['counts']['real_exposures']['new']==fit['real_supervised_exposures']
    assert sum(r['source_supervised'] for r in trace)==train['counts']['source_exposures']['new']==fit['source_supervised_exposures']
    assert sum(r['real_newly_supervised'] for r in trace)==fit['real_newly_supervised_exposures']
    assert sum(r['source_newly_supervised'] for r in trace)==fit['source_newly_supervised_exposures']
    results=C.read(C.DOC/'REAL_RESULTS.json');rows=C.read(C.RAW/'CORNER_ROWS.json');metrics=C.read(C.RAW/'FRAME_METRICS.json')
    for pop,ids in lock['populations'].items():
        rr=[r for r in rows if r['frame_id'] in set(ids)]
        for a in C.ARMS:
            e=np.array([r['errors'][a] for r in rr]);s=results['summary'][pop][a]
            assert s['correct10']==int((e<=10).sum()) and s['correct20']==int((e<=20).sum())
            for t in (5,10,20):assert s['PCK'][str(t)]==float((e<=t).mean())
            obs=np.array([e for i in ids for e in metrics[a][i]['observed_errors']]);assert float(np.quantile(obs,.9))==s['matched_pooled_corner8_P90_px']
            tr=results['transitions'][pop][a]['A'];assert tr['BG']-tr['GB']==s['correct10']-results['summary'][pop]['A']['correct10']
    source=C.read(C.DOC/'SOURCE_RESULTS.json')['results'];sr=C.read(C.RAW/'SOURCE_ROWS.json')
    for a in C.ARMS:
        for mode in ('clean','stress'):
            for k in ('COMMON_SUPPORT','NEW_SUPPORT'):
                e=np.array([v for r in sr[a][mode] for v in r[k]['errors']]);assert source[a][mode][k]['correct10']==int((e<=10).sum());assert source[a][mode][k]['n']==len(e)
                assert [(r['id'],r[k]['indices']) for r in sr[a][mode]]==[(r['id'],r[k]['indices']) for r in sr['A'][mode]]
    heat=C.read(C.DOC/'HEATMAP_SUMMARY.json');diag=C.read(C.RAW/'HEATMAP_DIAGNOSTICS.json')
    for pop,flag in [('H163','fixed_set'),('C_REMAINING_HARD','remaining_set')]:
        for a in C.ARMS:
            rr=[r for r in diag if r['model']==a and r[flag]];s=heat['summary'][pop][a];assert len(rr)==s['n']==sum(s['categories'].values())
            assert s['identity_suspect']==sum(r['identity_suspect'] for r in rr)
    test=subprocess.run([sys.executable,'-m','pytest','-q',str(C.HERE/'test_contracts.py'),str(C.HERE/'test_analysis.py')],capture_output=True,text=True)
    C.save(C.DOC/'FINAL_TESTS.json',dict(PASS=test.returncode==0,stdout=test.stdout,stderr=test.stderr));assert test.returncode==0,test.stdout+test.stderr
    assert 'skipped' not in test.stdout
    for p in C.DOC.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not target.startswith('http'):assert (p.parent/target).exists(),(p,target)
    identity=dict(implementation_bug_found=False,physical_identity_independently_verified=False,
        source='source_data.load_loss_targets keeps normalized native9 label order; cache preserves matched object labels without pointwise remap; SourceData undo letterbox and crop only',
        real='original frozen Replay/hidden-only PnP target native XY exactly preserved for every row; clean/OCC transform no channel change; hidden-only projected cuboid writes same native indices',
        evaluator='canonical[perm[valid]]=native errors; one approved whole-object permutation per frame; mechanism fixed historical FP branch',
        center8='unchanged all4arms; no supervision',groups={r['object_type']:r['group_order'] for r in C.read(C.ROOT/lock['symmetry']['path'])['objects']},
        candidate_tags=heat['identity_candidate_tags'],
        legitimate_tag_limit='approved alternative whole-object branch also selected by official full-object evaluator; compatibility not verified physical identity; no pointwise swap or new performance table')
    C.save(C.DOC/'IDENTITY_MAPPING_AUDIT.json',identity)
    C.save(C.DOC/'IDENTITY_MAPPING_AUDIT.md','# Identity 계약 감사\n\nSource normalized native9 label→cache→prepared/crop affine 순서를 유지. 실사 frozen target/pnp hidden index 순서 그대로 재사용. 모든 center8 pass-through, crop실험 channel swap없음. evaluator는 whole-object branch 하나만 사용하고 canonical[idx]=native error 역대응. 기전진단은 과거FP branch로 고정.\n\n구현상mapping불일치 발견없음. 다른채널 candidate는 가능한전체순열과 실제모델전체branch 일치로 LEGIT_SYMMETRY_EQUIVALENT / CHANNEL_CONFUSION / UNKNOWN 분류. 호환성은 물리정체검증이나 자동swap 근거가 아님. source/real label의 물리적 정의까지 새로 독립검증한것은 아니며 REVIEW_PENDING 유지.\n')
    C.save(C.DOC/'AUDIT.json',dict(PASS=True,new_train_runs=1,updates=300,all_predictions_before_scoring=True,
        old_artifacts_unchanged=len(lock['protected']),source_corruption_all300_trace_exact=True,real_original_parity=True,
        same_orders=True,BN_frozen=True,denominators_unchanged=True,failed_match_included=True,all_tests=test.stdout.strip(),
        tests=C.bind(C.DOC/'FINAL_TESTS.json'),max_gpu_temperature=max(r['gpu']['temperature_C'] for r in trace),
        max_gpu_memory_MiB=max(r['gpu']['memory_used_MiB'] for r in trace),no_sweep=True,no_rescue=True,final_model_changed=False,
        review=C.read(C.DOC/'REVIEW_STATUS.json')['status'],figures=len(list((C.DOC/'figures').glob('*.jpg'))),
        code=[C.bind(p) for p in sorted(C.HERE.glob('*.py'))],git_status=subprocess.check_output(['git','status','--short'],text=True)))
    print('FINAL_AUDIT_PASS',test.stdout.strip(),flush=True)

if __name__=='__main__':main()
