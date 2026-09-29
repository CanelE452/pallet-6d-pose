"""Record this one selector fit without resetting inherited student resources."""
import argparse
from . import common as C


def snapshot():
    source=C.DOC/'RESOURCE_LEDGER.json'
    path=C.DOC/'RESOURCE_LEDGER_BEFORE_SELECTOR.json'
    if path.exists():
        assert C.read(path)['totals']['selector_fits']==0
        return
    value=C.read(source)
    assert value['totals']['selector_fits']==0
    C.save(path,value,True)
    assert C.sha(path)==C.sha(source)


def book():
    path=C.DOC/'SELECTOR_CALIBRATION_FIT.json'
    fit=C.read(path);C.verify(fit['checkpoint'])
    source=C.DOC/'RESOURCE_LEDGER.json';value=C.read(source)
    event='CURRENT_DOMAIN_GEO_LINEAR_SINGLE_FIT'
    existing=[e for e in value['new_events'] if e['event']==event]
    if existing:
        assert existing[0]['details']['fit']==C.bind(path)
        print('SELECTOR_COST_ALREADY_BOOKED');return
    assert (C.DOC/'RESOURCE_LEDGER_BEFORE_SELECTOR.json').exists()
    C.resource(event,seconds=fit['seconds'],fits=0,updates=0,selector_fits=1,
        details=dict(fit=C.bind(path),selector_optimizer_steps=fit['optimizer_steps'],
                     student_optimizer_steps=0,feature_inference_seconds=C.read(C.DOC/'SYNTH_CURRENT_FEATURE_LOCK.json')['seconds'],
                     note='optimizer_updates top-level retains student-only convention; selector steps disclosed separately'))
    print('SELECTOR_COST_BOOKED',C.read(source)['totals'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('snapshot','book'));args=p.parse_args()
    {'snapshot':snapshot,'book':book}[args.phase]()
