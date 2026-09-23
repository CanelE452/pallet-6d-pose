"""Read-only assertions plus an exclusive final public audit record."""
import re
import subprocess
from . import common as E

def main():
    E.immutable();checks={}
    g=E.read(E.DOC/'GPU_GRADIENT_PARITY.json');checks['old_old_float32_close']=g['old_old']['passed'];checks['old_m0_float32_close']=g['old_new']['passed']
    checks['same_initial_and_batch']=all(r['initial']==g['runs'][0]['initial'] and r['batch']==g['runs'][0]['batch'] for r in g['runs'])
    for m in E.MATS:
        for a in E.ARMS:
            f=E.read(E.DOC/f'FIT_{m}_{a}.json');checks[m+'_'+a+'_320_steps']=f['steps']==320
            checks[m+'_'+a+'_actual_trainable_rule']=f['inventory']==g['runs'][0]['inventory']
            checks[m+'_'+a+'_actual_initial_state']=f['initial_state']==g['runs'][0]['initial']
            checks[m+'_'+a+'_fixed_exposures']=f['real_exposures']==f['synthetic_exposures']==2560
            checks[m+'_'+a+'_no_eval_no_rescue']=f['no_eval_during_train'] and f['no_rescue']
            E.verify(f['checkpoint']);E.verify(f['trace'])
    for b in E.read(E.DOC/'EVAL_PREDICTIONS_LOCK.json')['files']:E.verify(b)
    E.verify(E.read(E.DOC/'POSE_PREDICTIONS_LOCK.json')['file'])
    checks['math']=E.read(E.DOC/'LOSS_MATH_TEST.json')['passed']
    checks['source_exact256']=E.read(E.DOC/'SOURCE_GEOMETRY_BINDING.json')['bound']==256
    checks['occurrence_parity']=all(v['passed'] for v in E.read(E.DOC/'TRAINING_PARITY.json').values())
    old=E.read(E.C.RAW/'FRAME_METRICS.json')['S1'];new=E.read(E.RAW/'FRAME_METRICS.json')['M0']
    checks['M0_old_S1_2D_exact']=old==new
    report=(E.DOC/'REPORT_KO.md').read_text();links=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report)
    checks['all40_images_linked']=len(links)==40 and all((E.DOC/p).exists() for p in links)
    status=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines()
    prefixes=('scripts/research/'+E.NAME+'/', '_docs/experiments/'+E.NAME+'/')
    checks['only_requested_paths_staged']=all(p.startswith(prefixes) for p in status)
    checks['no_large_checkpoint_cache_staged']=not any(p.endswith(('.pt','.npz','.npy','.pkl')) for p in status)
    assert all(checks.values()),checks
    E.save(E.DOC/'FINAL_TESTS.json',dict(passed=True,checks=checks))
    print('FINAL_TESTS_PASS',len(checks))

if __name__=='__main__':main()
