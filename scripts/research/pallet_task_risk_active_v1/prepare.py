"""Conditional pool-only label export using the exact historical export function."""
import sys
from types import SimpleNamespace
from PIL import Image
from contracts import *
from acquisition import old_parity


def simulation():
    s=import_path('task_original_simulation',OLD_CODE/'simulation.py')
    s.RAW=RAW;s.DOC=DOC;s.write=write
    return s


def reserved_guard():
    split=read(DOC/'SPLIT_BINDING.json')
    denied={str((ROOT/r['label_path']).resolve()) for r in split['evaluation']}
    denied.update(str((POSE/n).resolve()) for n in ('AXIS_REVIEW_MANIFEST.json','GEOMETRY_RESOLVED_POSE_GT.json'))
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and str(Path(args[0]).resolve()) in denied:
            raise PermissionError('RESERVED_EVALUATION_GT_DENIED_BEFORE_ALL_FITS')
    sys.addaudithook(guard)


def main():
    verify_lock();assert read(DOC/'TASK_RISK_VERDICT.json')['verdict']=='TASK_RISK_MECHANISM_PASS'
    reserved_guard();s=simulation();choices=read(DOC/'SELECTION_LOCK.json')['selections']
    split=read(DOC/'SPLIT_BINDING.json');pool={r['frame_id']:r for r in split['pool']}
    proposed=choices['proposed'];assert len(proposed)==len(set(proposed))==30
    for i,a in enumerate(proposed):
        for b in proposed[i+1:]:
            assert pool[a]['capture_session']!=pool[b]['capture_session'] or abs(pool[a]['timestamp_ns']-pool[b]['timestamp_ns'])>=2000000000
    assert set(choices['full174'])==set(pool)
    targets={fid:s.P.E._legacy_forbidden_target(SimpleNamespace(frame_id=fid,label=r['label_path'],object_type=r['object_type'])) for fid,r in pool.items()}
    syn=read(s.SYNTH)['synthetic'];receipt={}
    assert len(syn)==1440
    for name,records in [('synthetic',syn),*[(m,[pool[fid] for fid in sorted(ids)]) for m,ids in choices.items()]]:
        folder=RAW/'dataset'/name;(folder/'images').mkdir(parents=True,exist_ok=True);(folder/'labels').mkdir(exist_ok=True)
        bound=[]
        for j,r in enumerate(records):
            image=Path(r['image']) if name=='synthetic' else ROOT/r['image_path']
            basename=f'{j:04}_{image.name}';dest=folder/'images'/basename
            if not dest.exists():dest.symlink_to(image.resolve())
            source=Path(r['label']) if name=='synthetic' else ROOT/r['label_path']
            if name=='synthetic':
                assert sha(image)==r['image_sha256'] and sha(source)==r['label_sha256']
                value=source.read_text();masked=0
            else:
                width,height=Image.open(image).size
                value,masked=s.export_target(targets[r['frame_id']],width,height)
            label=folder/'labels'/Path(basename).with_suffix('.txt')
            if label.exists():assert label.read_text()==value
            else:
                with label.open('x') as f:f.write(value)
            if name=='synthetic':assert sha(label)==sha(OLD_RAW/'dataset/synthetic/labels'/label.name)
            bound.append(dict(image=str(image.relative_to(ROOT)),image_sha256=sha(image),
                source_label=str(source.relative_to(ROOT)),source_label_sha256=sha(source),
                exported_label=str(label.relative_to(ROOT)),exported_label_sha256=sha(label),newly_offframe_masked=masked))
        receipt[name]=bound
    write(DOC/'LABEL_REVEAL_AUDIT.json',dict(status='PASS',revealed_frame_ids=sorted(pool),
        proposed_label_count=30,full174_label_count=174,synthetic_count=1440,evaluation_labels_revealed=0,
        note='Full174 reference reveals the entire already-Phase1-audited pool; proposed student sees only its30.',
        bindings=receipt,export='Unchanged simulation.export_target; normalized YOLO export, not literal GT JSON bytes',
        original_GT_files_unchanged=True,synthetic_labels_exact_old=True))
    write(DOC/'TASK_RISK_SELECTION_AUDIT.json',dict(status='PASS',budget=30,temporal_gap=True,
        GT_used_by_selection=False,old_selection_parity=old_parity(),
        overlaps={m:len(set(proposed)&set(ids)) for m,ids in read(OLD_DOC/'SELECTION_LOCK.json')['selections'].items()},
        frozen_score_sha256=sha(RAW/'TASK_RISK.json'),acquisition_trials=1,student_seeds_not_acquisition_trials=True))
    sources=[OLD_CODE/'simulation.py',ROOT/'scripts/research/pallet_paper_contribution_screen_v1/track_c/train.py',
        ROOT/'scripts/research/pallet_paper_contribution_screen_v1/track_c/wiring.py',
        ROOT/'scripts/research/pallet_paper_contribution_screen_v1/common/contracts.py',s.SYNTH]
    write(DOC/'STUDENT_IMPLEMENTATION_LOCK.json',dict(status='FROZEN_BEFORE_NEW_STUDENT_UPDATES',
        same_protocol=True,HYP=s.HYP,DATA=s.DATA,training_function='Direct call to unchanged original simulation.train',
        monitoring_only_wrappers=['batch_digest synthetic SHA assertion before loss','optimizer post-step GPU safety and progress logging'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        implementation={str(p.relative_to(ROOT)):sha(p) for p in (HERE/'prepare.py',HERE/'train.py')},
        fits=['proposed_seed1','proposed_seed2','proposed_seed3','full174_seed1'],max_updates=1200))
    print('PREPARED proposed30 + full174; synthetic1440 exact historical exports',flush=True)


if __name__=='__main__':main()
