"""Parent-only idempotent accounting for completed diagnostic/inference stages."""
from . import common as C

def main():
    candidates=[]
    for name in ('BASELINE_POSE_RESULTS.json','TR_ORACLE_RESULTS.json'):
        path=C.DOC/name
        if path.exists(): candidates.append((name,path,'wall_seconds',False))
    path=C.RAW/'loss_signal/PROBE_RESULTS.json'
    if path.exists(): candidates.append(('LOSS_RUNTIME_PROBE',path,'seconds',True))
    for path in sorted((C.RAW/'cycles').glob('*/PREDICTIONS_LOCK_*.json')):
        candidates.append((path.parent.name+'_'+path.stem,path,'GPU_reservation_seconds_this_invocation',True))
    for event,path,key,gpu in candidates:
        existing={e['event'] for e in C.read(C.DOC/'RESOURCE_LEDGER.json')['events']}
        if event not in existing:
            C.resource(event,C.read(path)[key],gpu,details=dict(artifact=C.bind(path),fits=0,updates=0))
    print(C.read(C.DOC/'RESOURCE_LEDGER.json')['totals'])

if __name__=='__main__': main()
