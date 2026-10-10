"""Summarize the single actual REAL run without calling any pose solver."""
from . import common as C
from .statistics import aggregate
from .verdict import evaluate

def main():
    assert C.read(C.DOC/'REAL_EXECUTION.json')['status']=='COMPLETE'
    assert not (C.DOC/'METRICS.json').exists()
    old=C.historical_rows();new=list(C.rows(C.DOC/'PREDICTIONS.jsonl.gz'))
    ids=[r['id'] for r in old if r['seed']==1 and r['method']=='BASE']
    base={r['id']:r for r in old if r['seed']==1 and r['method']=='BASE'}
    scopes={'ALL':ids}
    for grade in ['clean','moderate','severe']:scopes[grade]=[i for i in ids if base[i]['grade']==grade]
    for name,width in [('plastic',1.1),('wood',.8)]:
        scopes[name]=[i for i in ids if abs(base[i]['fixed_metadata']['dimensions_pnp_WH_D_m'][0]-width)<1e-9]
    metrics,paired,failures=aggregate(old,new,population='REAL_DEV319',bootstrap_level='cluster',scopes=scopes)
    verdict=evaluate(paired,phase='A2')
    for filename,value in [('METRICS.json',metrics),('PAIRED.json',paired),('FAILURES.json',failures),('VERDICT.json',verdict)]:C.write(C.DOC/filename,value)
    print('REAL_VERDICT',verdict['verdict'],flush=True)

if __name__=='__main__':main()
