"""Required delivery summary; optionally verify the live remote after push."""
import argparse
import subprocess
from . import common as C

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-remote',action='store_true');args=parser.parse_args()
    git=lambda *a:subprocess.check_output(['git',*a],text=True).strip()
    head=git('rev-parse','HEAD');branch=git('branch','--show-current')
    remote=git('ls-remote','origin','refs/heads/main').split()[0] if args.verify_remote else None
    if args.verify_remote:assert remote==head
    role=C.read(C.DOC/'H10_ROLE_PREVALENCE.json');pair=C.read(C.DOC/'PAIR_INTEGRITY.json');split=C.read(C.DOC/'RECORDING_DISJOINT_AUDIT.json')
    res=C.read(C.DOC/'RESULTS.json')['groups'];anchor=C.read(C.DOC/'VERIFIED_ANCHOR_TRANSFER.json')['groups']['HARD'];d=C.read(C.DOC/'TRANSFER_DECISION.json')
    hist=C.read(C.DOC/'HISTORICAL_DIRECTION.json')['groups'];lines=[]
    def line(k,v=''):lines.append(f'{k}: {v}')
    line('STATUS','COMPLETED_FROZEN_RECORDING_DISJOINT_EVALUATION');line('HEAD_BEFORE',C.read(C.DOC/'INPUT_BINDINGS.json')['head_before'])
    line('COMMIT',head);line('PUSH','VERIFIED_LOCAL_REMOTE_SHA_MATCH' if remote==head else 'NOT_CHECKED');line('BRANCH',branch)
    line('ROLE_SCAN',role['status']);line('H10',role['frames']);line('STRONG_ROLE_MISMATCH_COUNT',role['strong_count']);line('ROLE_MISMATCH_FRAMES',role['strong_frames']);line('ROLE_CONVENTION_STATUS',role['status'])
    line('PAIR');line('S0_S1_CAUSAL_PAIR',pair['status']);line('CHECKPOINT_REUSED',True);line('REPRODUCTION_FITS',0)
    for k in ('train_recordings','heldout_recordings','recording_intersection','image_sha_intersection'):line(k.upper(),split[k])
    for g in ('CLEAN','MODERATE','SEVERE'):
        line(g);a,b=res[g]['S0'],res[g]['S1']
        for arm,r in [('S0',a),('S1',b)]:line(arm+'_PCK10',r['twoD']['PCK']['10'])
        line('DELTA_PCK10_PP',100*(b['twoD']['PCK']['10']-a['twoD']['PCK']['10']))
        for kind in ('current','oracle'):
            for arm,r in [('S0',a),('S1',b)]:line(arm+'_'+kind.upper()+'_ADD',r[kind]['ADDsym_AUC'])
            line('DELTA_'+kind.upper()+'_ADD',b[kind]['ADDsym_AUC']-a[kind]['ADDsym_AUC'])
        for arm,r in [('S0',a),('S1',b)]:line(arm+'_SELECTION_LOSS',r['selection_loss'])
    line('VERIFIED_HARD_VISIBLE');line('POINTS',anchor['S0']['n'])
    for arm in ('S0','S1','TEACHER'):line(arm+'_PCK10',anchor[arm]['PCK']['10'])
    line('S1_MINUS_S0_CORRECT_COUNT',anchor['S1']['PCK']['10']['correct']-anchor['S0']['PCK']['10']['correct'])
    line('HISTORICAL_COMPARISON')
    for g in ('MODERATE','SEVERE'):line(g+'_DIRECTION_REPLICATED',hist[g]['status'])
    line('DECISION');line('TRANSFER_DECISION',d['primary']);line('PRIMARY_BOTTLENECK',d['primary_bottleneck']);line('SECONDARY_BOTTLENECK',d['secondary_bottleneck'])
    for key in ('more_hard_labeling_justified','selector_work_justified','representation_change_justified'):line('IS_'+key.upper(),d[key])
    line('NEXT_ONE_EXPERIMENT',d['next_one_experiment']['name']+'; DESIGN ONLY')
    line('REPORT',str((C.DOC/'REPORT_KO.md').relative_to(C.ROOT)))
    line('GIT_STATUS',git('status','--short','--branch'))
    text='\n'.join(lines)+'\n';print(text)
    if args.verify_remote:C.save(C.OUT/'COMMIT_PUSH_RESULT.txt',text)

if __name__=='__main__':main()
