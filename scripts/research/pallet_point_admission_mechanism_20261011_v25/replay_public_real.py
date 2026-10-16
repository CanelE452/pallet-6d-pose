"""Replay the unchanged saved real join with explicitly relocated public inputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

DOC='_docs/experiments/pallet_point_admission_mechanism_20261011_v25'
V21='_docs/experiments/pallet_whole_u_closure_20261011_v21'
V23='_docs/experiments/pallet_consensus_cause_audit_20261011_v23'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();checkout=a.checkout.resolve();out=a.output.resolve()
    assert not out.exists(),'Choose a new output directory; preserve previous results.'
    code=Path(__file__).with_name('audit_saved_real_loo.py')
    protocol=json.loads((checkout/DOC/'real_loo/PROTOCOL.json').read_text())
    expected={Path(k).name:v for k,v in protocol['inputs'].items()};assert len(expected)==4
    paths={'G':checkout/V21/'geometry/GEOMETRY.jsonl.gz',
        'P':checkout/V21/'evaluation/POSTHOC_ROWS.jsonl.gz',
        'SELECTED':checkout/V23/'SELECTED_BOUNDARY_ROWS.jsonl','code':code}
    before={k:dict(path=str(v),bytes=v.stat().st_size,sha256=sha(v)) for k,v in paths.items()}
    for k,v in paths.items():
        target=expected[v.name];assert before[k]['sha256']==target['sha256'] and before[k]['bytes']==target['bytes'],k
    out.mkdir(parents=True)
    save(out/'PUBLIC_REAL_REPLAY_PROTOCOL.json',dict(inputs=before,original_code_unchanged=True,
        relocation_only=['ROOT','G','P','SELECTED'],new_LOO=0,new_PnP=0,new_projection=0,new_pose_accuracy=0,
        new_training=0,physical_truth_certified=False))
    spec=importlib.util.spec_from_file_location('pallet_v25_saved_real_original',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=out
    for key in ('G','P','SELECTED'):setattr(module,key,paths[key])
    try:
        module.main()
        for name in ('ROWS.jsonl.gz','SUMMARY.json','CHECKS.json'):
            assert (out/'real_loo'/name).read_bytes()==(checkout/DOC/'real_loo'/name).read_bytes(),name
        assert before=={k:dict(path=str(v),bytes=v.stat().st_size,sha256=sha(v)) for k,v in paths.items()}
    except BaseException as e:
        save(out/'PUBLIC_REAL_REPLAY_CHECKS.json',dict(passed=False,error=repr(e)))
        raise
    save(out/'PUBLIC_REAL_REPLAY_CHECKS.json',dict(passed=True,original_code_unchanged=True,
        relocation_only=True,actual_all245_join=True,actual_selected_points=133,
        derived_rows_summary_checks_byte_identical=True,inputs_unchanged=True,
        new_LOO=0,new_PnP=0,new_pose_accuracy=0,new_training=0,
        physical_truth_certified=False))


if __name__=='__main__':main()
