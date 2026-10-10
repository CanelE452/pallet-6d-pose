"""Preserve both failed timings; account for YOLO's first-call warmup once."""
from __future__ import annotations
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
from . import runtime as Original
from . import runtime_review as Metadata
from . import runner as R
from ..pallet_sparse_local_line_20261010_v9 import common as C

DOC=R.DOC/'runtime_forward_review'
ORIGINAL_SHA='fbbc59e763a16e4e834347e75ea8ec0b261f254c9e25b2bf4e3e1469ebaee3ca'
FAILURE_FILES=('RUNTIME_PROTOCOL.json','RUNTIME_PREFLIGHT.json','RUNTIME_STARTED.json',
    'RUNTIME_RECEIPT.json','RUNTIME_RESOURCES.json','INTERRUPTED_RUNTIME_ROWS.jsonl.gz',
    'INTERRUPTED_RUNTIME_PACKET.json')

def history():
    Metadata.history();p=R.DOC/'runtime_review';r=C.read(p/'RUNTIME_RECEIPT.json')
    C.require(r['complete'] is False and r['status']=='FAILED_RUNTIME' and r['serialized_rows']==1
        and r['summaries'] is None and r['statistics_official'] is False
        and r['execution']==dict(image_decode_calls=26,image_decode_returns=26,
            model_initializations_started=1,model_initializations_complete=1,calls_started=1)
        and r['error']['message']=='SAME_OBSERVATION_GUARD: actual fresh detector/N3 forward count differs'
        and r['model_forwards_after_loading']==dict(detector=0,N3=0)
        and r['model_forwards']==dict(detector=2,N3=1,detector_internal_initialization_calls=1)
        and r['actual_base_counts']['detector_calls']==1 and r['cleanup_errors']==[],
        'preserved first-call internal initialization accounting failure required')
    C.bound(p/'RUNTIME_PROTOCOL.json',r['protocol'],'second failed runtime protocol')
    C.require(all(r['cleanup'][k] for k in ('instruments_restored','opencv_threads_restored',
        'head_hooks_removed','pipeline_closed','environment_restored')),'second cleanup required')

@contextmanager
def adapted():
    path=R.CODE/'runtime.py';raw=path.read_bytes()
    C.require(hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA,'unchanged original runtime source required')
    tree=ast.parse(raw.decode());measure=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='measure')
    changes=0
    for node in ast.walk(measure):
        if isinstance(node,ast.Call) and len(node.args)==2 and isinstance(node.args[1],ast.Constant) and node.args[1].value=='actual fresh detector/N3 forward count differs':
            C.require(ast.unparse(node.args[0])=="active['counter_deltas']['model'] == dict(detector=1, N3=1)",'exact original guard required')
            node.args[0]=ast.parse("active['counter_deltas']['model'] == dict(detector=(2 if job['phase']=='warmup' and job['index']==0 else 1),N3=1)",mode='eval').body
            changes+=1
    C.require(changes==1,'exactly one first-call forward accounting guard change required')
    replacement=ast.fix_missing_locations(ast.Module(body=[measure],type_ignores=[]))
    original_measure=Original.measure;old_doc=Metadata.DOC;Metadata.DOC=DOC
    exec(compile(replacement,str(path),'exec'),Original.__dict__)
    try:
        with Metadata.adapted():
            original_inputs=Original.inputs
            def inputs(args):
                data=original_inputs(args);history()
                paths=dict(review_code=Path(__file__),review_contract=R.DOC/'RUNTIME_FORWARD_REVIEW_CONTRACT_KO.md')
                paths.update({'preserved_second:'+n:R.DOC/'runtime_review'/n for n in FAILURE_FILES})
                data.update({'forward_review:'+k:C.binding(p) for k,p in paths.items()});return data
            Original.inputs=inputs
            try:yield
            finally:Original.inputs=original_inputs
    finally:Original.measure=original_measure;Metadata.DOC=old_doc

def main():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('freeze','preflight','measure')
    from ..pallet_three_head_observation_20261010_v7 import common as V7
    existing={a.dest for a in p._actions}
    for name,default in (('base-weights',None),('N3-weights',None),('calibration',str(V7.V.DOC/'CALIBRATION.json')),
        ('calibration-root',str(V7.CAL_ROOT)),('runtime-protocol',str(DOC/'RUNTIME_PROTOCOL.json'))):
        if name.replace('-','_') not in existing:p.add_argument('--'+name,default=default)
    args=p.parse_args();args.output=str(DOC);history()
    C.require(Path(args.runtime_protocol).resolve()==DOC/'RUNTIME_PROTOCOL.json','separate forward-review protocol required')
    if args.stage=='freeze':C.require(not DOC.exists(),'preserve old forward-review directory');DOC.mkdir()
    with adapted():{'freeze':Original.freeze,'preflight':Original.preflight,'measure':Original.measure}[args.stage](args)
    C.require(Original.R is R and Metadata.DOC==R.DOC/'runtime_review','forward review globals restored')

if __name__=='__main__':main()
