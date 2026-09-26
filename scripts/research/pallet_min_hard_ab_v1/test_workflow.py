import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from . import common as C
from . import policy
from .labels import validate_frame,coverage
from .prepare import MADIndex,coarse

def rows(nrec=4,n=90):
    return [dict(frame_id=f'r{r}:{i}',recording=f'R{r}',order=i,image=dict(path='dummy',sha256=f'{r}:{i}')) for r in range(nrec) for i in range(n)]

class WorkflowTests(unittest.TestCase):
    def test_temporal_bins_and_round_order(self):
        src=rows();q=policy.temporal_queue(src)
        self.assertEqual(len(q),4*48)
        self.assertEqual(len({r['frame_id'] for r in q}),len(q))
        self.assertEqual([r['round'] for r in q],sorted(r['round'] for r in q))
        for rec in range(4):
            for rnd in (1,2,3):
                selected=[r for r in q if r['recording']==f'R{rec}' and r['round']==rnd]
                self.assertEqual(len(selected),16)
                self.assertTrue(all(r['bin']%3+1==rnd for r in selected))
        self.assertEqual(q,policy.temporal_queue(list(reversed(src))))
    def test_short_recording(self):
        q=policy.temporal_queue(rows(1,32));self.assertEqual(len(q),32)
        self.assertEqual([sum(r['round']==i for r in q) for i in (1,2,3)],[11,11,10])
    def test_hash_winner_per_bin(self):
        src=rows(1,96);q=policy.temporal_queue(src)
        for r in q:
            expected=min(src[2*r['bin']:2*r['bin']+2],key=lambda x:policy.key('hard-tag-v1:',x['frame_id']))
            self.assertEqual(r['frame_id'],expected['frame_id'])
    def test_round_gate_and_cap(self):
        src=[dict(r,tag='MODERATE') for r in rows(4,3)]
        self.assertEqual(policy.round_decision(src,1),'SELECT')
        src=[dict(r,tag='MODERATE') for r in rows(2,20)]
        self.assertEqual(policy.round_decision(src,3),'INSUFFICIENT')
        src=[dict(r,tag='MODERATE') for r in rows(3,3)]
        self.assertEqual(policy.round_decision(src,2),'NEXT_ROUND')
        self.assertEqual(policy.round_decision(src,3),'SELECT')
    def test_manual_selection_only_hard_and_cap(self):
        src=[dict(r,tag=('MODERATE' if r['order']%2 else 'SEVERE')) for r in rows(4,8)]
        initial,reserve=policy.select_hard(src)
        self.assertEqual(len(initial),8);self.assertEqual(len(reserve),2)
        self.assertEqual(sum(r['tag']=='SEVERE' for r in initial),4)
        self.assertGreaterEqual(len({r['recording'] for r in initial}),3)
        for rec in {r['recording'] for r in src}:
            self.assertLessEqual(sum(r['recording']==rec for r in initial+reserve),3)
        self.assertEqual((initial,reserve),policy.select_hard(list(reversed(src))))
    def test_no_severe_fabrication(self):
        src=[dict(r,tag='MODERATE') for r in rows(4,4)]
        initial,_=policy.select_hard(src);self.assertTrue(all(r['tag']=='MODERATE' for r in initial))
    def test_MAD_pruning_exact(self):
        rng=np.random.default_rng(42)
        base=rng.integers(0,240,(48,64),dtype=np.int16)
        data=[base,base+2,base+3,rng.integers(0,256,(48,64),dtype=np.int16)]
        idx=MADIndex(10)
        for i,a in enumerate(data):idx.add(a,str(i))
        for a in data+[base+1,base+5]:
            brute=any(np.abs(b-a).mean()<=2 for b in data)
            self.assertEqual(idx.match(a) is not None,brute)
        for a in data:
            for b in data:self.assertLessEqual(np.abs(coarse(a)-coarse(b)).mean(),np.abs(a-b).mean()+1e-9)
    def test_label_validation_no_completion(self):
        f=dict(role='ROLE_CONFIDENT',size=[100,100],bbox=[1,1,99,99],corners=[dict(status='OCCLUDED',xy=None) for _ in range(8)])
        self.assertEqual(validate_frame(f)[0],[])
        f['corners'][0]=dict(status='DIRECT_VISIBLE',xy=None)
        self.assertTrue(validate_frame(f)[0])
        f['corners'][0]=dict(status='OCCLUDED',xy=[1,2]);self.assertTrue(validate_frame(f)[0])
        f['role']='ROLE_UNCERTAIN';self.assertTrue(validate_frame(f)[0])
    def test_hidden_and_role_uncertain_excluded_coverage(self):
        selection=[dict(frame_id=f'f{i}',recording=f'r{i%3}') for i in range(6)]
        frames={r['frame_id']:dict(complete=True,role='ROLE_CONFIDENT',corners=[dict(status='DIRECT_VISIBLE' if j<4 else 'OCCLUDED',xy=[j,j] if j<4 else None) for j in range(8)]) for r in selection}
        c=coverage(frames,selection);self.assertTrue(c['passed']);self.assertEqual(c['direct_visible_clicks'],24)
        frames['f0']['role']='ROLE_UNCERTAIN';self.assertFalse(coverage(frames,selection)['passed'])
    def test_tag_store_autosave_undo_resume(self):
        from .tag_difficulty import TagStore
        src=[dict(r,round=1) for r in rows(1,3)]
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp)
            with patch.object(C,'RAW',temp),patch.object(C,'queue',return_value=src):
                C.save(temp/'DIFFICULTY_QUEUE_PRIVATE.json',dict(rows=src))
                s=TagStore(src,1);self.assertEqual(s.next_index(),0)
                s.put(0,'MODERATE');self.assertEqual(s.next_index(),1)
                restored=TagStore(src,1);self.assertEqual(restored.next_index(),1)
                self.assertEqual(restored.undo(),0);self.assertEqual(restored.next_index(),0)
    def test_immutable_write_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with patch.object(C,'RAW',p):
                C.save(p/'lock.json',{'x':1},immutable=True);C.save(p/'lock.json',{'x':1},immutable=True)
                with self.assertRaises(AssertionError):C.save(p/'lock.json',{'x':2},immutable=True)

if __name__=='__main__':unittest.main()
