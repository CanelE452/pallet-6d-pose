"""Full candidate reinference plus unchanged paper 2D and MAIN 6D evaluators."""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT,RAW,DOC,sha,write,tensor_sha
sys.path.insert(0,str(ROOT/'scripts/research/pallet_line_pose_v1'))
import paper_evaluation as P

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--seed',type=int,required=True)
    a=ap.parse_args();torch.set_num_threads(4)
    name=f'{a.arm}_seed{a.seed}';directory=RAW/'C_geometry_preserving_da'/name
    checkpoint=directory/'last.pt'
    ck=torch.load(checkpoint,map_location='cpu')
    if ck.get('train_args',{}).get('task')!='pose':
        corrected=directory/'last_pose_metadata.pt'
        if not corrected.exists():
            before=tensor_sha(ck['model'].state_dict())
            ck['train_args']['task']='pose';ck['model'].args['task']='pose';ck['model'].task='pose'
            torch.save(ck,corrected)
            assert tensor_sha(torch.load(corrected,map_location='cpu')['model'].state_dict())==before
            write(directory/'TASK_METADATA_CORRECTION.json',dict(original_sha256=sha(checkpoint),corrected_sha256=sha(corrected),
                tensor_state_sha256=before,all_weights_and_buffers_identical=True,new_optimizer_updates=0,
                invalid_evaluation='evaluation',reason='Missing task in train_args defaults Pose26 checkpoint to detection-only result parser'))
        checkpoint=corrected
    destination=directory/'evaluation_pose';destination.mkdir(exist_ok=True)
    cache=destination/'PREDICTIONS.json'
    if not cache.exists():
        predictor=P.E._UltralyticsPredictor(checkpoint,'0')
        frames={};pair=P.population()
        for items in [pair.positive.items,pair.negative.items]:
            for item in items:
                path=(ROOT/item.image).resolve()
                candidates=predictor.predict(path)
                frames[P.canonical_key(item.image)]=[dict(score=float(s),box_xyxy=b.tolist(),keypoints_xy=k.tolist() if k is not None else None) for s,b,k in candidates]
                if len(frames)%500==0:print(f'{name} inference {len(frames)}/3008',flush=True)
        write(cache,dict(schema_version='paper_cached_predictions_v1',complete=True,model=name,
            weights_sha256=sha(checkpoint),frames=frames,role='DEVELOPMENT',full_reinference=True))
    frames=json.loads(cache.read_text())['frames']
    two_d=P.paper_2d(destination,cache,str(checkpoint))
    pose=P.paper_pose(destination,frames,name)
    write(destination/'EVALUATOR_BINDINGS.json',dict(checkpoint_sha256=sha(checkpoint),
        full_candidate_cache_sha256=sha(cache),canonical_2d_source_sha256=sha(Path(P.E.__file__)),
        canonical_pose_source_sha256=sha(ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py'),
        evidence_level='DEVELOPMENT',pose_source_note='Canonical closure wrapper retains historical template metadata; actual predictions are bound here and by actual_source_sha256, not historical R0 checkpoint metadata.'))
    print(name,'evaluation complete',flush=True)

if __name__=='__main__':main()
