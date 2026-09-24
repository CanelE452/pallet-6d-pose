"""Read-only validation by default; --lock freezes a complete human first pass."""
import argparse
from collections import Counter
from . import common as C

def validate(labels, selection):
    assert labels['selection_sha256']==C.sha(C.RAW/'ANCHOR_SELECTION.json')
    assert labels['annotator'].strip(), 'Human annotator required'
    assert len(labels['frames'])==len(selection['frames'])
    for r,s in zip(labels['frames'],selection['frames']):
        assert r['frame_id']==s['frame_id'] and r['image_sha256']==s['image']['sha256']
        assert C.sha(C.ROOT/s['image']['path'])==r['image_sha256']
        assert r['annotator'].strip() and r['pass']==1
        assert [c['id'] for c in r['corners']]==list(range(8))
        assert all(C.valid_corner(c,s['hw']) for c in r['corners']), f'Incomplete/invalid {r["frame_id"]}'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lock',action='store_true');args=parser.parse_args()
    selection=C.read(C.RAW/'ANCHOR_SELECTION.json');path=C.RAW/'LABELS.json'
    if not path.exists():print('WAITING_FOR_HUMAN_LABELING 0/18');return
    labels=C.read(path);p=C.progress(labels,selection);print(p)
    if p['completed_images']!=len(selection['frames']):return
    validate(labels,selection)
    visible=Counter();ids=Counter();statuses=Counter()
    for r,s in zip(labels['frames'],selection['frames']):
        for c in r['corners']:
            statuses[c['status']]+=1
            if c['status']=='DIRECT_VISIBLE':visible[s['severity']]+=1;ids[c['id']]+=1
    covered=sum(visible.values())>=60 and all(visible[s]>=12 for s in C.SEVERITIES) and sum(v>=3 for v in ids.values())>=5 and len({r['recording'] for r in selection['frames']})>=3
    coverage=dict(primary_visible=sum(visible.values()),severity=dict(visible),corner_id=dict(ids),statuses=dict(statuses),
                  minimum_coverage_pass=covered,status='READY_FOR_BLIND_QA' if covered else 'SECOND_BATCH_SELECTION_REQUIRED',
                  additional_images_selected=0,model_predictions_read=False)
    print(coverage)
    if args.lock:
        C.save_new(C.RAW/'FIRST_PASS_LOCK.json',dict(labels_sha256=C.sha(path),selection_sha256=C.sha(C.RAW/'ANCHOR_SELECTION.json'),time=C.now()))
        C.save_new(C.DOC/'COVERAGE.json',coverage)
        print('FIRST_PASS_LOCKED. Resume coverage routing and blind QA; no scoring performed.')

if __name__=='__main__':main()
