"""Read-only verification of the already frozen A candidate; no reselection."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT,DOC,R0,R0_SHA,sha

if __name__=='__main__':
    frozen=json.loads((DOC/'A_architecture_confirmation/CANDIDATE_FREEZE.json').read_text())
    assert sha(ROOT/frozen['checkpoint'])==frozen['sha256']
    assert sha(R0)==R0_SHA
    assert frozen['new_training_updates']==0
    print('PASS: frozen A/R0 checkpoint hashes; no new training or selection')
