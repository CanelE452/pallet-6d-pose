"""Audit all nine fits, then reuse canonical 2D/6D evaluation on the fixed145."""
import numpy as np
import torch
import reweight as W
from experiment import E,S,ROOT,RAW,DOC,read,write,sha,METHODS,SEEDS

def audit():
    torch.set_num_threads(4);baseline=S.load_model();initial=S.tensor_sha(baseline.state_dict())
    buffers=S.tensor_sha(dict(baseline.named_buffers()));expected_params=sum(p.numel() for p in baseline.parameters())
    audits={};traces={}
    for seed in SEEDS:
        for method in METHODS:
            name=f'{method}_seed{seed}';d=W.BASE/'runs'/name
            start=read(d/'START.json')
            assert start['source_code_sha256']==sha(S.Path(W.__file__))
            assert start['split_sha256']==sha(DOC/'SPLIT.json')
            a=read(d/'TRAINING_AUDIT.json');assert a['optimizer_updates']==300 and a['init_state_sha256']==initial
            assert a['checkpoint_sha256']==sha(d/'last.pt')
            ck=torch.load(d/'last.pt',map_location='cpu');m=ck['model']
            assert ck['train_args']['task']=='pose' and m.args['task']=='pose'
            assert sum(p.numel() for p in m.parameters())==expected_params
            assert all(torch.isfinite(v).all() for v in m.state_dict().values())
            assert S.tensor_sha(dict(m.named_buffers()))==buffers
            trace=read(d/'EXPOSURE.json');assert len(trace)==300
            for row in trace:
                expected=W.weights(method,torch.tensor(row['losses']),torch.tensor(row['alignment'])).numpy()
                np.testing.assert_allclose(expected,row['weights'],atol=1e-6,rtol=1e-6)
            audits[name]=a;traces[name]=trace
    for seed in SEEDS:
        for step in range(300):
            for field in ('synthetic','real','meta'):
                assert len({traces[f'{m}_seed{seed}'][step][field] for m in METHODS})==1
    for rows in read(DOC/'reweight/LABEL_BINDINGS.json').values():
        for r in rows:
            assert sha(ROOT/r['source_label'])==r['source_label_sha256']
            assert sha(ROOT/r['exported_label'])==r['exported_label_sha256']
    write(DOC/'reweight/TRAINING_AUDIT.json',dict(status='PASS',fits=9,optimizer_updates=2700,
        initial_state_parity=True,BN_buffers_frozen=True,actual_synthetic_real_meta_input_parity=True,
        saved_weight_formula_recomputation=True,source_exported_labels_unchanged=True,audits=audits))

def evaluate():
    E.training_audit=audit;E.RAW=W.BASE;E.DOC=DOC;E.S.METHODS=METHODS
    E.evaluate()

if __name__=='__main__':evaluate()
