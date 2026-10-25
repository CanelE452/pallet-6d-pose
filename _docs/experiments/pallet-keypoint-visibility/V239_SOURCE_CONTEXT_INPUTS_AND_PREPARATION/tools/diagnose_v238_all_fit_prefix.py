"""Locate all SOURCE FIT prefix/direct descriptor discrepancies, without fit."""
from pathlib import Path
import csv
import hashlib
import json
import types
import numpy as np
import torch

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R/'V238_ALL_FIT_DESCRIPTOR_DIAG'


def safe(p):
    p = Path(p)
    assert not any(x in str(p).lower() for x in ('v112', '__pycache__', '.pyc', 'current223', 'publicheld', 'oldval', 'v177', 'v178', 'source_holdout', 'v182', 'private_evaluation'))
    assert not p.is_symlink()
    return p


def main():
    assert not H.exists(); H.mkdir()
    epath = R/'V238_ROOT_CONTEXT_RESIDUAL/ACTUAL_SOURCE_CONTEXT_RESIDUAL_FIT_ENTRY.json'
    e = json.loads(safe(epath).read_text())
    rows = [json.loads(x) for x in safe(e['sample_map']['path']).read_text().splitlines()]
    chosen = sorted([r for r in rows if r['sample_index'] is not None and r['fit'] and r['crop_available'] and r['target'] >= 0], key=lambda r: r['sample_index'])
    assert len(chosen) == 5876
    indices = np.asarray([r['sample_index'] for r in chosen], np.int64)
    gi = np.asarray([r['group_index'] for r in chosen], np.int64); si = np.asarray([r['active_slot'] for r in chosen], np.int64)
    scope = {'approved_SOURCE_FIT_descriptor_diagnosis': True, 'source_FIT_samples': indices.tolist(),
             'allowed': 'Exact previous V238 official model/code, original samples metadata, only these5876 FIT full/centre prefix views, stored new NODE_CLS on these active nodes. No node/context RGB numeric read needed. Whole SOURCE metadata may be parsed; DEV pixel/prefix/feature numeric rows and head outputs zero.',
             'fixed_before_actual': 'Compare original actual same64 FIT batch schedule, locate failed batch/rows; do not assume first batch failed.',
             'threshold_or_tolerance_change': False, 'optimizer_or_newhead_outputs': 0}
    (H/'SCOPE_BEFORE_MODEL_AND_FEATURE_READ.json').write_text(json.dumps(scope,indent=2)+'\n')
    raw = safe(e['code']['path']).read_bytes(); assert hashlib.sha256(raw).hexdigest() == e['code']['sha256']
    a = types.ModuleType('v238_fit_diag'); a.__file__ = e['code']['path']; exec(compile(raw, a.__file__, 'exec'), a.__dict__)
    utility = a.load_code(e['utility'], e['code_whitelist'], 'v238_fit_diag_utility', a.PINS['utility'])
    source, crop, runtime = utility.load_sources(e); source.configure_runtime()
    model = source.load_backbone(e['model_loader_config'], 'cuda', treatment=False)
    with np.load(safe(e['samples_npz']['path']),allow_pickle=False) as z: data = {k:z[k] for k in z.files}
    prefix = np.load(safe(e['prefix_npy']['path']),mmap_mode='r',allow_pickle=False)
    nodes = np.load(safe(Path(e['output_directory'])/'NODE_CLS.npy'),mmap_mode='r',allow_pickle=False)
    store = source.PrefixViews(prefix,data); records=[]
    with torch.no_grad():
        for first in range(0,len(indices),64):
            ids = indices[first:first+64]
            old = source.features_for(ids,store,model,runtime,None,'cuda',False)[:,:384].cpu().numpy()
            new = np.array(nodes[gi[first:first+64],si[first:first+64]],copy=True)
            diff=np.abs(old-new); cosine=np.sum(old*new,axis=1)
            for j,i in enumerate(ids):
                records.append({'sample_index':int(i),'FIT_batch':first//64,'family':int(data['family'][i]),'group':int(gi[first+j]),'node':int(si[first+j]),'max_abs':float(diff[j].max()),'mean_abs':float(diff[j].mean()),'cosine':float(cosine[j]),'exceeds_original_1e_6':bool(diff[j].max()>1e-6)})
    with (H/'ALL5876_DESCRIPTOR_DIFFERENCES.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    bad=[r for r in records if r['exceeds_original_1e_6']]
    result={'completed':True,'PASS':True,'diagnostic_not_equivalence_PASS':True,'FIT_rows':5876,
            'first_failing_batch':None if not bad else min(r['FIT_batch'] for r in bad),
            'exceed_1e_6_rows':len(bad),'max_abs':max(r['max_abs'] for r in records),
            'min_cosine':min(r['cosine'] for r in records),'bad_rows':bad,
            'new_head_or_optimizer_updates':0,'DEV_or_current_numeric_feature_or_logits':0,
            'tolerance_or_threshold_changed':False}
    (H/'DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
