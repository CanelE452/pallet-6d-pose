"""Audit completed CPU accuracy independently of the deferred GPU runtime panel."""
from env import *
def run():
    if complete('DEV_COMPLETE'): return
    start=now(); verify(); inputs=[DOC/'PRIOR_DEV_LOCK.json',HERE/'prior_analysis.py',HERE/'dev_completion.py']
    for s in (1,2,3):
        for name in (f'PRIOR{s}_INFERENCE',f'PRIOR{s}_SCORED',f'PRIOR{s}_raw_SCORED'):
            assert complete(name),name;inputs.append(DOC/(name+'.json'))
    result=read(DOC/'UNIFIED_DEV_RESULTS.json');paired=read(DOC/'P_VS_PRIOR_PAIRED.json')
    assert result['complete'] and result['historical_result_recomputed_exact']
    for records in result['sources'].values():
        for b in records:assert sha(ROOT/b['path'])==b['sha256']
    assert paired['session']['resamples']==paired['frame']['resamples']==10000
    assert paired['session']['frames']==319 and paired['session']['units']==13
    receipt('DEV_COMPLETE',inputs,[DOC/'UNIFIED_DEV_RESULTS.json',DOC/'P_VS_PRIOR_PAIRED.json',RAW/'DEV_FULL_PRECISION_ERRORS.json'],start,
        seeds=3,actual_image_forwards=3*3008,wrapped_and_raw_arms=6,runtime_is_separate=True)
    print('DEV_ACCURACY_COMPLETE_RUNTIME_SEPARATE',flush=True)
if __name__=='__main__':run()
