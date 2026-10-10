"""Run the same fresh timing after a separately declared metadata repair.

Preserve the original failed first call. Only the metadata packet adapter and
exclusive output location change; unchanged frozen runtime does the timings.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
from pathlib import Path
from types import SimpleNamespace

from . import runtime as Original
from . import runner as R
from . import deployment as D
from .deployment_review import Pipeline
from ..pallet_sparse_local_line_20261010_v9 import common as C

DOC=R.DOC/'runtime_review'


def history():
    p=R.DOC;receipt=C.read(p/'RUNTIME_RECEIPT.json')
    C.require(receipt['complete'] is False and receipt['status']=='FAILED_RUNTIME'
        and receipt['serialized_rows']==0 and receipt['summaries'] is None
        and receipt['error']['type']=='TypeError'
        and receipt['error']['message']=="dict() got multiple values for keyword argument 'id'"
        and receipt['execution']['calls_started']==1 and receipt['cleanup_errors']==[],
        'specific preserved first runtime packaging failure required')
    C.bound(p/'RUNTIME_PROTOCOL.json',receipt['protocol'],'original failed runtime protocol')
    C.require(all(receipt['cleanup'][k] for k in ('instruments_restored','opencv_threads_restored',
        'head_hooks_removed','pipeline_closed','environment_restored')),'original cleanup required')
    return receipt


def output(args,name):
    C.require(Path(args.output).resolve()==DOC.resolve() and name==Path(name).name,
              'exclusive metadata-repair runtime output required')
    p=DOC/name;C.require(not p.exists() and not any(x.is_symlink() for x in (p,*p.parents)),
                          'preserve runtime review '+name)
    return p


@contextmanager
def adapted():
    original_R=Original.R;original_inputs=Original.inputs;original_pre=Original.prerequisites
    original_class=D.Pipeline
    def parent_call(fn,*a):
        active=Original.R;Original.R=original_R
        try:return fn(*a)
        finally:Original.R=active
    def inputs(args):
        data=parent_call(original_inputs,args);history()
        paths=dict(review_code=Path(__file__),packet_adapter=Path(__file__).with_name('deployment_review.py'),
            review_contract=R.DOC/'RUNTIME_REVIEW_CONTRACT_KO.md')
        for n in ('RUNTIME_PROTOCOL.json','RUNTIME_PREFLIGHT.json','RUNTIME_STARTED.json','RUNTIME_RECEIPT.json',
            'RUNTIME_RESOURCES.json','INTERRUPTED_RUNTIME_ROWS.jsonl.gz','INTERRUPTED_RUNTIME_PACKET.json'):
            paths['preserved:'+n]=R.DOC/n
        data.update({'metadata_review:'+k:C.binding(p) for k,p in paths.items()})
        return data
    proxy=SimpleNamespace(**vars(original_R));proxy.DOC=DOC;proxy.output=output
    Original.R=proxy;Original.inputs=inputs
    Original.prerequisites=lambda args:parent_call(original_pre,args)
    D.Pipeline=Pipeline
    try:yield
    finally:
        D.Pipeline=original_class;Original.R=original_R
        Original.inputs=original_inputs;Original.prerequisites=original_pre


def main():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('freeze','preflight','measure')
    existing={a.dest for a in p._actions}
    from ..pallet_three_head_observation_20261010_v7 import common as V7
    for name,default in (('base-weights',None),('N3-weights',None),('calibration',str(V7.V.DOC/'CALIBRATION.json')),
        ('calibration-root',str(V7.CAL_ROOT)),('runtime-protocol',str(DOC/'RUNTIME_PROTOCOL.json'))):
        if name.replace('-','_') not in existing:p.add_argument('--'+name,default=default)
    args=p.parse_args();args.output=str(DOC)
    C.require(Path(args.runtime_protocol).resolve()==DOC/'RUNTIME_PROTOCOL.json','separate metadata-review protocol required')
    history()
    if args.stage=='freeze':
        C.require(not DOC.exists(),'preserve old review directory');DOC.mkdir()
    with adapted():{'freeze':Original.freeze,'preflight':Original.preflight,'measure':Original.measure}[args.stage](args)
    C.require(D.Pipeline.__module__.endswith('.deployment') and Original.R is R,'supplemental runtime globals restored')


if __name__=='__main__':main()
