import re
from collections import Counter
import torch
from . import common as E

def main():
    E.immutable();split=E.read(E.DOC/'SPLIT_LOCK.json');fits={a:E.read(E.DOC/f'FIT_{a}.json') for a in E.ARMS};traces={a:E.read(E.RAW/f'TRACE_{a}.json') for a in E.ARMS}
    assert all(f['steps']==320 and f['last_only'] and f['no_eval_during_train'] and f['no_rescue'] for f in fits.values())
    assert all(f['initial_state']==fits[E.ARMS[0]]['initial_state'] and f['inventory']==fits[E.ARMS[0]]['inventory'] for f in fits.values())
    for f in fits.values():E.verify(f['checkpoint']);E.verify(f['trace'])
    for occ in range(5120):
        rows=[traces[a][occ] for a in E.ARMS];assert all(r['occ']==occ for r in rows)
        assert rows[1]['RGB']==rows[2]['RGB'] and rows[1]['bbox']==rows[2]['bbox']
        assert len({r['mask'] for r in rows})==1
        if rows[0]['role']!='REPLACEMENT':assert rows[0]==rows[1]==rows[2]
    coords={a:{role:sum(r['supervised'] for r in traces[a] if r['role']==role) for role in ('C0','REPLACEMENT','SYNTH')} for a in E.ARMS};assert all(v==coords[E.ARMS[0]] for v in coords.values())
    for b in E.read(E.DOC/'PREDICTION_LOCK.json')['files']:E.verify(b)
    E.verify(E.read(E.DOC/'POSE_PREDICTION_LOCK.json')['file'])
    assert not {r['recording_group'] for r in split['train']}&{r['recording_group'] for r in split['heldout']}
    assert not {r['image']['sha256'] for r in split['train']}&{r['image']['sha256'] for r in split['heldout']}
    state=torch.load(E.P.C.PRIOR_CK,map_location='cpu',weights_only=False)['model_state_dict'];parent_sha=E.P.C.E.state_sha(state);start=E.read(E.ROOT/'_docs/experiments/pallet_replay_by_type_v1/plastic/TRAIN_START.json');assert parent_sha==start['initial_state_sha']
    report=(E.DOC/'REPORT_KO.md').read_text();links=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report);assert links and all((E.DOC/p).exists() for p in links)
    results=E.read(E.DOC/'RESULTS.json')['groups']
    for g,arms in results.items():
        if arms.get('status')=='N/A':continue
        assert len({(r['twoD']['total_frames'],r['twoD']['corners']) for r in arms.values()})==1
        assert all(r['manual_only']['corners']==0 for r in arms.values())
    E.save(E.DOC/'AUDIT.json',dict(status='COMPLETED_3_FITS',passed=True,student_fits=3,steps=960,new_capture=0,new_manual_annotation=0,teacher_fits=0,recording_disjoint_fitting=True,independent_unseen_test=False,same_initial_inventory=True,T1_T2_RGB_bbox_order_exact=True,all_three_masks_exact=True,C0_synthetic_exact=True,teacher_initial_state_matches_synthetic_parent=True,teacher_initial_state_sha=parent_sha,originals_unchanged=True,prediction_freeze_before_scoring=True,D9_unchanged=True,supervised_keypoint_exposures=coords,figures=len(links),all_links_exist=True,manual_heldout_reference_channels=0,source_heldout256_unchanged=True))
    print('AUDIT_PASS',coords,'figures',len(links),flush=True)

if __name__=='__main__':main()
