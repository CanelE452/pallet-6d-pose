"""Summarize sealed Stage-1 evidence; never run a model or a pose solver."""
from pathlib import Path
from . import common as C
from . import statistics as S


def run():
    assert C.read(C.DOC/'STAGE0_PARITY.json')['status']=='PASS'
    assert (C.DOC/'STAGE1_SELECTION_SEAL.json').exists()
    target=C.DOC/'STAGE1_METRICS.json'
    assert not target.exists(), 'Preserve completed numerical evidence'
    source=Path(__file__).parent
    C.write(C.DOC/'STAGE1_STATISTICS_LOCK.json', dict(
        status='LOCKED_BEFORE_NUMERIC_SUMMARIES',
        bindings=[C.binding(source/n,C.ROOT) for n in
                  ('statistics.py','verdict.py','summarize.py')],
        models=0, solver_calls=0))
    result={}
    for suffix,population in [('REAL','REAL_DEV'),('SYNTH','SYNTH_HELDOUT'),('AUX','AUX')]:
        rows=list(C.rows(C.DOC/f'STAGE1_ROWS_{suffix}.jsonl.gz'))
        result[population]=S.stage1(rows,population)
        print(f'{population}: {len(rows)} sealed rows summarized',flush=True)
    C.write(target,result)
    return result


if __name__=='__main__':
    run()
