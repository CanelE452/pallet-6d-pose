"""Idempotent human-gated entry points. Never invent missing human input."""
import argparse
from collections import Counter
from . import common as C
from . import policy

def tag_summary(rows):
    counts=Counter(r['tag'] for r in rows)
    byrec={rec:dict(Counter(r['tag'] for r in rows if r['recording']==rec)) for rec in sorted({r['recording'] for r in rows})}
    return dict(total=len(rows),counts={k:counts[k] for k in C.TAGS},recordings=byrec,
                hard_recordings=len({r['recording'] for r in rows if r['tag'] in ('MODERATE','SEVERE')}))

def select(rows):
    if (C.DOC/'HARD_SELECTION_LOCK.json').exists():
        return
    initial,reserve=policy.select_hard(rows)
    chosen=[dict(r,assignment='INITIAL' if r in initial else 'RESERVE') for r in initial+reserve]
    C.save(C.RAW/'HARD_SELECTION_PRIVATE.json',dict(rows=chosen),immutable=True)
    C.save(C.RAW/'FRAME_MAPPING_PRIVATE.json',dict(rows=[{k:r[k] for k in ('frame_id','image','recording','assignment')} for r in chosen]),immutable=True)
    public=[{k:r[k] for k in ('frame_id','recording','tag','assignment')}|{'image_sha256':r['image']['sha256']} for r in chosen]
    value=dict(created_at=C.now(),rows=public,initial=len(initial),reserve=len(reserve),
               selection='DP initial8 nearest4S/4M under >=3 recording,max3 overall; per-stratum SHA prefix; deterministic hash ties; remaining reserve in SHA order',
               private_selection=C.bind(C.RAW/'HARD_SELECTION_PRIVATE.json'),tag_lock=C.bind(C.DOC/'DIFFICULTY_TAG_LOCK.json'),
               model_outputs_opened=0,evaluation_overlap=0,anchor_overlap=0,final_test_overlap=0)
    C.save(C.DOC/'HARD_SELECTION_LOCK.json',value,immutable=True)
    C.save(C.DOC/'HARD_SELECTION_PUBLIC.json',value,immutable=True)
    C.set_state('WAITING_FOR_HUMAN_HARD_ANNOTATION',active_frames=len(initial),reserve=len(reserve),
                rounds_completed=C.read(C.DOC/'DIFFICULTY_TAG_LOCK.json')['rounds'],
                command='python -m scripts.research.pallet_min_hard_ab_v1.annotate_hard',training='NOT_RUN')

def resume():
    with C.exclusive('resume'), C.exclusive('tagging'):
        state=C.state()
        if state['status']=='WAITING_FOR_HUMAN_DIFFICULTY_TAGS':
            queue=C.queue();round_no=state['round'];tagpath=C.RAW/'DIFFICULTY_TAGS_PRIVATE.json'
            if not tagpath.exists():
                print('Human tags missing; no selections or training performed.');return
            data=C.read(tagpath)
            assert data['queue_sha256']==C.sha(C.RAW/'DIFFICULTY_QUEUE_PRIVATE.json')
            lookup={r['frame_id']:r for r in queue};responses=data['responses']
            assert set(responses)<=set(lookup)
            tagged=[]
            for fid,tag in responses.items():
                r=lookup[fid]
                assert tag['round']==r['round']<=round_no
                assert tag['tag'] in C.TAGS and tag['image_sha256']==r['image']['sha256']
                C.verify(r['image']);tagged.append(dict(r,tag=tag['tag']))
            # Previously completed rounds cannot be revised silently.
            for previous in range(1,round_no):
                lock=C.read(C.DOC/f'DIFFICULTY_ROUND_{previous}_LOCK.json');C.verify(lock['snapshot'])
                old=C.read(C.ROOT/lock['snapshot']['path'])
                assert all(responses.get(k)==v for k,v in old.items()),'Closed round changed'
            expected=[r for r in queue if r['round']<=round_no]
            missing=[r for r in expected if r['frame_id'] not in responses]
            C.save(C.DOC/'DIFFICULTY_TAG_SUMMARY_PUBLIC.json',dict(tag_summary(tagged),completed_rounds=round_no-1 if missing else round_no))
            if missing:
                print(f'WAITING_FOR_HUMAN_DIFFICULTY_TAGS: {len(missing)} unanswered through round{round_no}');return
            snapshot=C.RAW/f'ROUND_{round_no}_TAGS_PRIVATE.json'
            C.save(snapshot,responses,immutable=True)
            decision=policy.round_decision(tagged,round_no)
            lockpath=C.DOC/f'DIFFICULTY_ROUND_{round_no}_LOCK.json'
            if not lockpath.exists():
                C.save(lockpath,dict(snapshot=C.bind(snapshot),summary=tag_summary(tagged),decision=decision,created_at=C.now()),immutable=True)
            if decision=='NEXT_ROUND':
                C.set_state('WAITING_FOR_HUMAN_DIFFICULTY_TAGS',round=round_no+1,rounds_completed=round_no,
                            command='python -m scripts.research.pallet_min_hard_ab_v1.tag_difficulty',training='NOT_RUN',annotation='NOT_RUN')
            else:
                if not (C.DOC/'DIFFICULTY_TAG_LOCK.json').exists():
                    C.save(C.DOC/'DIFFICULTY_TAG_LOCK.json',dict(snapshot=C.bind(snapshot),rounds=round_no,
                           summary=tag_summary(tagged),created_at=C.now(),model_outputs_opened=0),immutable=True)
                if decision=='INSUFFICIENT':
                    C.set_state('HARD_PREVALENCE_OR_DIVERSITY_INSUFFICIENT',rounds_completed=round_no,
                                training='NOT_RUN',annotation='NOT_RUN',user_action_required=False)
                else:select(tagged)
        elif state['status'] in ('WAITING_FOR_EXISTING_ANNOTATION_KEYPOINTS','WAITING_FOR_HUMAN_HARD_METADATA'):
            if (C.DOC/'PNP_BOX_USER_APPROVAL.json').exists():
                from .lock_assisted_labels import main as lock_labels
                lock_labels()
            else:
                from .existing_click_progress import summarize
                summarize()
        elif state['status'] in ('WAITING_FOR_HUMAN_HARD_ANNOTATION','WAITING_FOR_HUMAN_HARD_QA'):
            from .labels import validate_resume
            validate_resume()
        elif state['status']=='HARD_LABELS_LOCKED_TRAINING_PENDING':
            print('Human label lock exists. Next stage: implement/verify matched S1 fit and execute only after integrity gates. No A/B result yet.')
        else:
            print('No automatic state change:',state['status'])

def status():
    state=C.state();audit=C.read(C.DOC/'CANDIDATE_POOL_AUDIT.json');queue=C.read(C.DOC/'DIFFICULTY_QUEUE_LOCK.json')
    print('STATUS:',state['status']);print('HEAD_START:',C.read(C.DOC/'INPUT_BINDINGS.json')['head'])
    print('HEAD_END:',C.git('rev-parse','HEAD'));print('BRANCH:',C.git('branch','--show-current'))
    print('CANDIDATE_POOL:');print('TOTAL:',audit['eligible_frames'],'/ source',audit['source_frames'])
    print('RECORDINGS:',audit['eligible_recordings']);print('EXCLUSIONS:',audit['exclusions'])
    print('SHA_OVERLAP:',audit['SHA_overlap_after_exclusion']);print('NEAR_DUP_OVERLAP:',audit['near_duplicate_overlap_after_exclusion'])
    tag_lock=C.DOC/'DIFFICULTY_TAG_LOCK.json'
    print('DIFFICULTY_TAGGING:');print('ROUNDS_COMPLETED:',C.read(tag_lock)['rounds'] if tag_lock.exists() else state.get('rounds_completed',0));print('ROUND:',state.get('round','N/A'))
    print('QUEUE_COUNTS:',{k:sum(v.values()) for k,v in queue['round_counts'].items()})
    summary=C.DOC/'DIFFICULTY_TAG_SUMMARY_PUBLIC.json'
    print('HUMAN_TAGGED:',C.read(summary) if summary.exists() else 0)
    print('ANNOTATION/TRAINING/EVALUATION: human-input-gated; see STATUS.json; no A/B metrics before execution')
    print('PRIMARY_DECISION: NOT_EVALUATED');print('COMMAND:',state.get('command','N/A'))
    print('RESUME: python -m scripts.research.pallet_min_hard_ab_v1.cli resume')
    print('REPORT: _docs/experiments/pallet_min_hard_ab_v1/REPORT_KO.md')
    receipt=C.OUT/'TAGGING_PREP_GIT.json'
    if receipt.exists():print('TAGGING_PREP_GIT:',C.read(receipt))
    print('GIT_STATUS:',C.git('status','--short','--branch','--',str(C.DOC.relative_to(C.ROOT)),str((C.ROOT/'scripts/research'/C.NAME).relative_to(C.ROOT))))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','resume','status']);a=parser.parse_args()
    if a.action=='prepare':
        from .prepare import main as prepare
        prepare()
    elif a.action=='resume':resume()
    if a.action!='status':
        from .report import render
        render()
    status()

if __name__=='__main__':main()
