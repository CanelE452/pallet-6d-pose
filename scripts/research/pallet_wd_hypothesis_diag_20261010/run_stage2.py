"""Run each preregistered Stage2 rule once, after Stage1 publication.

This wrapper records rule-specific verdicts. It does not invent a combined
scientific verdict across independent selectors or choose a rule from results.
"""
import time
from . import common as C
from . import inputs as I
from . import stage2
from . import s3
from . import verdict as V


def skipped_real(rule, synthetic):
    ids=sorted({r['id'] for r in I.real_inputs()})
    assert len(ids)==319
    return dict(status='SKIPPED_SYNTH_WORSENED',population='REAL_DEV',rule=rule,
        reason='Frozen synthetic primary gate forbids this rule on REAL',
        expected_frames=319,expected_rows=3828,skipped_frame_ids=ids,
        methods=list(C.METHODS),seeds=list(C.SEEDS),primary=synthetic['primary'],
        actual_rule_selector_calls=0,source_lock_sha256=C.sha(C.DOC/'SOURCE_LOCK_STAGE2.json'))


def run():
    stage2.gate('S1','SYNTH')
    names=('EXECUTION_STAGE2_START.json','EXECUTION_STAGE2.json','VERDICT_STAGE2.json',
           'VERDICT.json','METRICS.json','PAIRED.json','FAILURES.json')
    assert not any((C.DOC/name).exists() for name in names), 'Preserve previous Stage2 orchestration'
    assert not any(C.DOC.glob('RESULTS_S[123]_*.json')), 'Preserve prior rule execution; no implicit resume'
    started=time.monotonic()
    C.write(C.DOC/'EXECUTION_STAGE2_START.json',dict(status='STARTED_AFTER_STAGE1_PUBLICATION',
        stage1_publication_sha256=C.sha(C.DOC/'STAGE1_PUBLICATION.json'),
        method_lock_sha256=C.sha(C.DOC/'METHOD_LOCK.json'),
        source_lock_sha256=C.sha(C.DOC/'SOURCE_LOCK_STAGE2.json'),
        order=['S1_SYNTH','S1_REAL_if_gate_pass','S2_SYNTH','S2_REAL_if_gate_pass','S3_SYNTH','S3_REAL'],
        training_updates=0,network_forwards=0,additional_SubPix_calls=0))
    results={};verdicts={};skips={}
    for rule in ('S1','S2'):
        print('STAGE2_ORCHESTRATION',rule,'SYNTH',flush=True)
        synthetic=stage2.run(rule,'SYNTH')
        results[rule]={'SYNTH':synthetic}
        if V.synth_gate(synthetic['primary'])['run_real']:
            print('STAGE2_ORCHESTRATION',rule,'REAL',flush=True)
            real=stage2.run(rule,'REAL');results[rule]['REAL']=real
        else:
            real=None;skip=skipped_real(rule,synthetic);skips[rule]=skip
            results[rule]['REAL']=skip
            C.write(C.DOC/f'SKIPPED_{rule}_REAL.json',skip)
            print('STAGE2_ORCHESTRATION',rule,'REAL SKIPPED_SYNTH_WORSENED',flush=True)
        verdicts[rule]=V.evaluate(synthetic['primary'],None if real is None else real['primary'],rule)
    results['S3']={}
    for population in ('SYNTH','REAL'):
        print('STAGE2_ORCHESTRATION','S3',population,flush=True)
        results['S3'][population]=s3.run(population)
    verdicts['S3']=dict(V.evaluate(None,rule='S3'),
        real=results['S3']['REAL'].get('primary'),
        eligible_sessions=results['S3']['REAL'].get('eligible_sessions',[]),
        eligible_frames=results['S3']['REAL'].get('eligible_frames',0),
        not_estimable_metadata=results['S3']['REAL'].get('not_estimable_metadata',False),
        synthetic_oracle_diagnostic_only=True,
        synthetic_oracle_has_method_verdict=False)
    packet=dict(phase='STAGE2',status='COMPLETE',definitions=V.definitions(),rules=verdicts,
        verdict_summary={rule:value['verdict'] for rule,value in verdicts.items()},
        no_combined_method_verdict=True,
        selection='No result-dependent selector, coefficients, bins or thresholds were chosen')
    for name in ('VERDICT_STAGE2.json','VERDICT.json'):C.write(C.DOC/name,packet)
    for field in ('metrics','paired','failures'):
        wrapper={}
        for rule,populations in results.items():
            wrapper[rule]={population:result[field] if field in result else result
                           for population,result in populations.items()}
        C.write(C.DOC/f'{field.upper()}.json',dict(phase='STAGE2',rules=wrapper,
            stage1_metrics_path='STAGE1_METRICS.json',
            definitions_path='METHOD_LOCK.json',combined_method_verdict=False))
    executions={}
    for rule,populations in results.items():
        executions[rule]={}
        for population,result in populations.items():
            name=f'EXECUTION_{rule}_{population}.json'
            executions[rule][population]=result if result.get('status')=='SKIPPED_SYNTH_WORSENED' else dict(
                path=name,sha256=C.sha(C.DOC/name),**C.read(C.DOC/name))
    C.write(C.DOC/'EXECUTION_STAGE2.json',dict(status='COMPLETE',phase='STAGE2',
        elapsed_seconds=time.monotonic()-started,executions=executions,skipped_real=skips,
        verdict_summary=packet['verdict_summary'],
        source_lock_sha256=C.sha(C.DOC/'SOURCE_LOCK_STAGE2.json'),
        method_lock_sha256=C.sha(C.DOC/'METHOD_LOCK.json'),
        stage1_publication_sha256=C.sha(C.DOC/'STAGE1_PUBLICATION.json'),
        network_forwards=0,training_updates=0,additional_SubPix_calls=0,
        coordinates_changed=0,S4_executed=False,depth_model_inference=0))
    print('STAGE2 COMPLETE',packet['verdict_summary'],flush=True)
    return packet


if __name__=='__main__':run()
