"""Verify artifacts/tests and prepare public allowlist; no Git side effects."""
import json
import re
import subprocess
import sys
import numpy as np
from . import common as C

def main():
    result=subprocess.run([sys.executable,'-m','pytest','-q',str(C.HERE/'test_contracts.py')],capture_output=True,text=True)
    C.save(C.DOC/'TEST_RESULTS.json',dict(PASS=result.returncode==0,stdout=result.stdout,stderr=result.stderr));print(result.stdout,flush=True);assert result.returncode==0
    frozen=C.read(C.DOC/'PREDICTIONS_FROZEN.json');code=[C.bind(p) for p in sorted(C.HERE.glob('*.py'))]
    for b in frozen['models'].values():C.verify(b)
    decision=C.read(C.DOC/'DECISION.json');s=C.read(C.DOC/'TRAIN_CORNER_CENSUS_SUMMARY.json');e=C.read(C.DOC/'TRAIN_EXPOSURE_AUDIT.json');h=C.read(C.DOC/'TRAIN_DEV_HEATMAP_COMPARE.json');p=C.read(C.DOC/'CONTROLLED_PROBE_RESULTS.json')
    assert e['total']==19200 and e['hard20']==444
    assert s['hard20']==s['hard20_40']+s['hard40']==50
    assert s['strict_hard20']==50 and s['strict_frames']==253
    assert h['TRAIN_STRICT_HARD']['n']==50 and h['DEV_MATCHED_HARD']['n']==196
    assert p['curves']['FULL125']['P1']['30']['correct10']==498
    assert decision['flags']==dict(HARD_SHORTAGE=True,CONTROLLED_CAPACITY_OK=True,TRAIN_HARD_FIT_OK=True,CANDIDATE_TRANSFER_GAP=True,TARGET_CONSISTENCY_CONFLICT=False)
    for mode in ('P1','P2'):
        for radius,x in p['curves']['FULL125'][mode].items():assert x['n']==512 and x['correct10']/512==x['PCK10']
    for r in C.read(C.DOC/'INPUT_BINDINGS.json')['protected']:C.verify(r)
    audit=dict(PASS=True,NEW_TRAINING=0,OPTIMIZER_STEPS=0,CHECKPOINT_UPDATE=0,GT_CHANGE=0,PSEUDO_TARGET_CHANGE=0,
        tests=26,weights_identical=all(v['before']==v['after'] for v in frozen['hashes'].values()),
        all_predictions_frozen_before_DEV_analysis=True,old_artifacts_unchanged=len(C.read(C.DOC/'INPUT_BINDINGS.json')['protected']),
        train_images=253,controlled_corners=128,controlled_frames=128,probe_examples_per_model=4090,model_count=4,
        elapsed_inference_seconds=frozen['elapsed_seconds'],figures=len(list((C.DOC/'figures').glob('*.png'))),
        threshold_changes=0,independent_coordinate_review='PENDING',no_new_experiment_executed=True,code=code)
    C.save(C.DOC/'AUDIT.json',audit)
    # Publish only aggregates, rendered evidence and implementation. No frame maps or pseudo coordinates.
    excluded=C.read(C.DOC/'EXCLUDED11_AUDIT.json');C.save(C.DOC/'EXCLUDED11_PUBLIC.json',dict(count=11,reasons=excluded['reasons'],
        raw_score_min=min(r['raw_score'] for r in excluded['rows']),raw_score_max=max(r['raw_score'] for r in excluded['rows']),
        valid_kp_counts=sorted({r['valid_kp'] for r in excluded['rows']}),stage1_missing=sum(r['stage1'] is None for r in excluded['rows']),stage2_missing=sum(r['stage2'] is None for r in excluded['rows'])))
    protocol=C.read(C.DOC/'CONTROLLED_PROBE_PROTOCOL.json');protocol['selected_count']=len(protocol.pop('selected'));C.save(C.DOC/'CONTROLLED_PROBE_PUBLIC_PROTOCOL.json',protocol)
    docs=['PURPOSE_AND_PLAN.md','PREFLIGHT_AUDIT.md','TRAIN_CORNER_CENSUS_SUMMARY.json','TRAIN_HARDNESS_KO.md',
        'PSEUDO_CONSISTENCY_AUDIT.json','PSEUDO_CONSISTENCY_AUDIT.md','EXCLUDED11_PUBLIC.json','TRAIN_EXPOSURE_AUDIT.json',
        'TEMPORAL_DIVERSITY_AUDIT.json','TRAIN_DEV_DISTRIBUTION.json','TRAIN_DEV_DISTRIBUTION_KO.md',
        'CONTROLLED_PROBE_PUBLIC_PROTOCOL.json','CONTROLLED_PROBE_RESULTS.json','CONTROLLED_PROBE_KO.md',
        'NATURAL_HARD_TRAIN_FIT.json','TRAIN_DEV_HEATMAP_COMPARE.json','ERROR_MORPHOLOGY_AUDIT.json','EVAL_PROVENANCE_AUDIT.json',
        'GALLERY.html','GALLERY_COUNTS.json','RESULTS_KO.md','DECISION.json','NEXT_STAGE_PLAN.md','AUDIT.json','FINAL_OUTPUT.json','TEST_RESULTS.json']
    files=[C.DOC/n for n in docs]+sorted((C.DOC/'figures').glob('*.png'))+sorted(C.HERE.glob('*.py'))+[C.HERE/'README.md']
    allow={p.resolve() for p in files}
    for pth in files:
        assert pth.exists()
        if pth.suffix not in ('.md','.html'):continue
        text=pth.read_text();links=re.findall(r'\]\(([^)]+)\)',text) if pth.suffix=='.md' else re.findall(r'(?:src|href)="([^"]+)"',text)
        for link in links:
            if not link.startswith(('http:','https:','#')):assert (pth.parent/link).resolve() in allow,(pth,link)
    C.save(C.DOC/'PUBLICATION_MANIFEST.json',dict(files=[C.bind(p) for p in files],private_excluded=True))
    print('COMPLETED',len(files),'publication files',audit,flush=True)

if __name__=='__main__':main()
