"""Final artifact/contract validation; no model training or inference."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.research.pallet_posefix_replay_diagnosis_v1 import run as R

def main():
    R.verify_sources()
    required=['SOURCE_BINDING.json','EXISTING_REPLAY_AUDIT.md','REPLAY_PROTOCOL.json',
        'PASS_METRICS_SYNTH.json','PASS_METRICS_REAL_DEV.json','MOVEMENT_DISTRIBUTION.json',
        'CAP_SATURATION.json','OSCILLATION_ANALYSIS.json','ERROR_STRATA.json','SYMMETRY_DIAGNOSTIC.json',
        'NUMERICAL_REPLAY.json','POSE_RESULTS.json','FINAL_DIAGNOSIS.md','FINAL_DIAGNOSIS.json',
        'TRAINING_PROPOSAL.md','REPLAY_SUMMARY.md','FIG_REPLAY_TRAJECTORIES.png',
        'FIG_ERROR_BY_INITIAL_BIN.png','FIG_RAW_VS_CAPPED.png']
    assert all((R.DOC/name).is_file() for name in required)
    inference=R.read(R.DOC/'INFERENCE_COMPLETE.json'); scoring=R.read(R.DOC/'SCORING_COMPLETE.json')
    audit=R.read(R.DOC/'INDEPENDENT_AUDIT.json'); conclusion=R.read(R.DOC/'FINAL_DIAGNOSIS.json')
    assert all(r['training_updates']==0 for r in [inference,scoring,audit,conclusion])
    assert not conclusion['training_started'] and not conclusion['model_promoted']
    assert audit['trajectory_cap_checks']==80892 and audit['current_pose_max_abs']==0
    assert all(r['maximum_Esym_difference']==0 and r['frames']==319 for r in scoring['current_evaluator_parity'])
    for r in inference['outputs']:
        assert R.E.sha(R.ROOT/r['path'])==r['sha256']
    for r in R.read(R.DOC/'TARGET_SOURCE_BINDINGS.json')['files']:
        assert R.E.sha(R.ROOT/r['path'])==r['sha256']
    for r in R.read(R.DOC/'FIGURE_MANIFEST.json')['files']:
        assert R.E.sha(R.ROOT/r['path'])==r['sha256']
    for name in ['ANALYSIS_CODE_LOCK.json','SCORING_COMPLETE.json']:
        d=R.read(R.DOC/name); b=d.get('code',d.get('scoring_code'))
        assert R.E.sha(R.ROOT/b['path'])==b['sha256']
    for split in R.SPLITS:
        for seed in (1,2,3):
            s=R.read(R.RAW/f'scores/seed{seed}_{split}.json')
            count=R.read(R.DOC/'REPLAY_PROTOCOL.json')['counts'][split]
            assert s['frames']==count
            for mode in R.CHAINS:
                p=s['chains'][mode]['passes'];assert set(p)=={'0','1','2','3'}
                for step in ('1','2','3'):
                    for key in ('detected','matched','corners','observed_corners'):
                        assert p[step][key]==p['0'][key],(split,seed,mode,step,key)
    result=dict(complete=True,training_updates=0,original_checkpoints_and_sources_unchanged=True,
        required_artifacts=len(required),seeds=3,frames_per_seed=4494,trajectory_cap_checks=80892,
        current_2D_R0_exact_frames=319,current_6D_R0_exact_frames=319,
        prediction_preservation_checks=scoring['prediction_preservation_checks'],
        matching_and_GT_denominators_unchanged_all_passes=True,
        GT_input_to_inference=False,raw_data_ignored=True,model_promoted=False,phase_B_training_started=False,
        code=[R.bound(p) for p in sorted(R.HERE.glob('*.py'))],
        outputs=[R.bound(R.DOC/n) for n in required],
        tests=R.bound(R.DOC/'TEST_RESULTS.json'),audit=R.bound(R.DOC/'INDEPENDENT_AUDIT.json'))
    R.write(R.DOC/'FINAL_VALIDATION.json',result);print('FINAL_VALIDATION_PASS',flush=True)

if __name__=='__main__':main()
