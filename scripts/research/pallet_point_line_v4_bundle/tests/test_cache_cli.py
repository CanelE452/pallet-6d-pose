from pathlib import Path
from dataclasses import fields
import json,subprocess,sys
import pytest,torch
from pointline_v4.cache_io import ExportDataset,sha256
from pointline_v4.model import Observation
from build_fixture import build

ROOT=Path(__file__).resolve().parents[1]


def run_module(name,*args,ok=True):
    r=subprocess.run([sys.executable,'-m','pointline_v4.'+name,*map(str,args)],cwd=ROOT,text=True,capture_output=True,timeout=30)
    assert (r.returncode==0)==ok,(r.stdout,r.stderr)
    return r


def test_export_manifest_tamper_is_rejected(tmp_path):
    d=build(tmp_path/'fixture');(d/'obs_0.pt').write_bytes(b'not the same cache')
    with pytest.raises(ValueError,match='Changed'):ExportDataset(d/'train.json')


def test_export_path_escape_is_rejected(tmp_path):
    d=build(tmp_path/'fixture');p=d/'train.json';j=json.loads(p.read_text());j['records'][0]['observation']='../escape.pt';p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match='inside'):ExportDataset(p)


def test_generated_fixture_cannot_run_as_main(tmp_path):
    d=build(tmp_path/'fixture')
    run_module('train_cache','--manifest',d/'train.json','--protocol',d/'protocol.json','--output',d/'blocked',
               '--arm','H','--seed',1,'--allow-generated-fixture',ok=False)


def test_unlocked_protocol_rejected(tmp_path):
    d=build(tmp_path/'fixture');p=d/'protocol.json';j=json.loads(p.read_text());j['locked']=False;p.write_text(json.dumps(j))
    run_module('train_cache','--manifest',d/'train.json','--protocol',p,'--output',d/'blocked',
               '--arm','H','--seed',1,'--smoke','--allow-generated-fixture',ok=False)


def test_actual_smoke_checkpoint_and_scoring_with_gt_files_absent(tmp_path):
    d=build(tmp_path/'fixture')
    run_module('train_cache','--manifest',d/'train.json','--protocol',d/'protocol.json','--output',d/'trained',
               '--arm','H','--seed',1,'--smoke','--allow-generated-fixture')
    completion=json.loads((d/'trained/COMPLETION.json').read_text())
    assert completion['optimizer_steps']==4 and completion['stage']=='smoke' and completion['accuracy_improved'] is None
    ckpt=d/'trained/checkpoint_final.pt'
    # A smoke checkpoint cannot masquerade as a real main model.
    run_module('score_cache','--manifest',d/'calibration.json','--checkpoint',ckpt,'--output',d/'blocked.json',ok=False)
    # Physically removing GT is stronger than an assertion about a counter.
    for file in d.glob('gt_*.pt'):file.unlink()
    run_module('score_cache','--manifest',d/'calibration.json','--checkpoint',ckpt,'--output',d/'scores.json','--allow-smoke')
    result=json.loads((d/'scores.json').read_text());assert result['supervision_files_opened']==0 and len(result['rows'])==2
