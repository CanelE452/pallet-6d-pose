from context34 import *
import train34,subprocess
from sampler34 import install,precheck

def worker(arm,seed,repeat):
    install();original=train34.deterministic
    def flags():return dict(original(),sampler='fixed-grid dense bilinear bmm')
    train34.deterministic=flags;train34.RUN=RUN/'dense';train34.worker(arm,seed,repeat)

def main():
    precheck();source={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'sampler34.py',KIT/'train_dense34.py',KIT/'train34.py',KIT/'context34.py')}
    write(RUN/'DENSE_PROTOCOL.json',dict(reason='Requested deterministic settings and equivalent pooling still produced grid_sample input-gradient variation with identical input/state. Replace fixed-grid bilinear sampling with an equivalent dense linear operator; keep old12runs intact. Same initialization/data/loss/budget, an additional12models24000updates. Actual total24models48000updates.',source_hashes=source,primary_gates_unchanged=True,sampler_preflight_sha256=sha(RUN/'SAMPLER_PREFLIGHT.json')))
    for seed in SEEDS:
        for repeat in REPEATS:
            for arm in ARMS:subprocess.run([sys.executable,str(KIT/'train_dense34.py'),'--arm',arm,'--seed',str(seed),'--repeat',str(repeat)],cwd=ROOT,check=True,env=dict(os.environ,PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            a,b=[read(RUN/'dense/models'/f'{arm}_r{r}_seed{seed}'/'COMPLETE.json') for r in REPEATS];rows.append(dict(arm=arm,seed=seed,state_equal=a['state_sha256']==b['state_sha256'],trace_equal=a['trace_sha256']==b['trace_sha256']))
    write(RUN/'DENSE_REPLICATION.json',dict(rows=rows,all_equal=all(r['state_equal'] and r['trace_equal'] for r in rows)));write(RUN/'DENSE_COMPLETE.json',dict(complete=True,models=12,updates=24000,total_models=24,total_updates=48000));print('DENSE34 COMPLETE',rows,flush=True)

def load_model(arm,seed,repeat=0,phase='dense'):
    from model33 import Correspondence
    install();folder=RUN/'dense' if phase=='dense' else RUN;model=Correspondence(seed).cuda().eval();model.load_state_dict(torch.load(folder/'models'/f'{arm}_r{repeat}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);return model

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int);p.add_argument('--repeat',type=int);a=p.parse_args()
    if a.arm:worker(a.arm,a.seed,a.repeat)
    else:main()
