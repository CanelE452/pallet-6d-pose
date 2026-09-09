"""Scores exported observations without opening any supervision files."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from .cache_io import ExportDataset,write_json,sha256,tensor_state_sha
from .model import EvidenceVerifier


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--allow-smoke',action='store_true')
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Do not overwrite saved scores')
    receipt_path=Path(args.checkpoint).parent/'COMPLETION.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf8'))
    if not receipt.get('training_completed') or receipt['checkpoint_sha256']!=sha256(args.checkpoint):
        raise ValueError('Uncompleted or modified training artifact')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if receipt['optimizer_steps']!=saved['optimizer_steps'] or receipt['stage']!=saved['stage']:
        raise ValueError('Checkpoint and completion receipt disagree')
    if saved.get('schema')!='pointline_v4_head_1' or (saved['stage']!='main' and not args.allow_smoke):
        raise ValueError('Expected completed main checkpoint')
    if saved['final_state_sha256']!=tensor_state_sha(saved['state_dict']):raise ValueError('Checkpoint tensor hash mismatch')
    data=ExportDataset(args.manifest,verify_targets=False)
    model=EvidenceVerifier(**saved['config']);model.load_state_dict(saved['state_dict'],strict=True);model.to(args.device).eval()
    rows=[]
    with torch.inference_mode():
        for i,record in enumerate(data.records):
            obs=data.observation(i).to(args.device);cost,_=model(obs)
            rows.append(dict(frame_id=record['frame_id'],session_id=record['session_id'],cost=cost[0].cpu().tolist(),
                candidate_valid=obs.candidate_valid[0].cpu().tolist(),observation_sha256=record['observation_sha256']))
    write_json(out,dict(schema='pointline_v4_scores_1',arm=saved['config']['arm'],stage=saved['stage'],
        seed=saved['seed'],initial_state_sha256=saved['initial_state_sha256'],plan_sha256=saved['plan_sha256'],protocol_sha256=saved['protocol_sha256'],
        manifest_sha256=sha256(args.manifest),checkpoint_sha256=sha256(args.checkpoint),supervision_files_opened=0,rows=rows))

if __name__=='__main__':main()
