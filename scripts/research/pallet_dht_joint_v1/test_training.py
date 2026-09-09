"""CPU-only deterministic source/augmentation contract; no optimizer or writes."""
from pathlib import Path
import json
import copy
import sys
import unittest
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train import ManifestDataset, EpochSampler, SOURCE_MANIFEST
from integration import build_model

class DatasetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        source=json.loads(SOURCE_MANIFEST.read_text())
        cls.records=[r for r in source['records'] if r['partition']=='train'][:4]
        cls.hyp=build_model('point_only').args
        cls.hyp.mosaic=cls.hyp.mixup=cls.hyp.cutmix=cls.hyp.copy_paste=0.
        cls.hyp.fliplr=cls.hyp.flipud=0.

    def dataset(self):
        return ManifestDataset(self.records,seed=3,data=dict(nc=1,names={0:'pallet'},kpt_shape=[9,3],
            flip_idx=[1,0,3,2,5,4,7,6,8],channels=3),imgsz=640,batch_size=2,augment=True,
            hyp=copy.deepcopy(self.hyp),rect=False,cache=False,single_cls=True,stride=32,pad=0.,task='pose')

    def test_seed_changes_by_epoch_and_repeats_exactly(self):
        d=self.dataset();a=d[(0,0)];b=d[(0,0)];c=d[(1,0)]
        self.assertTrue(torch.equal(a['img'],b['img']));self.assertTrue(torch.equal(a['keypoints'],b['keypoints']))
        self.assertFalse(torch.equal(a['img'],c['img']))
        self.assertEqual(tuple(a['img'].shape),(3,640,640))

    def test_worker_count_does_not_change_sample_or_augmentation(self):
        collected=[]
        for workers in [0,2]:
            d=self.dataset();s=EpochSampler(len(d),3,True);s.set_epoch(2)
            loader=DataLoader(d,batch_size=2,sampler=s,num_workers=workers,collate_fn=d.collate_fn)
            collected.append(list(loader))
        for a,b in zip(*collected):
            for key in ['img','keypoints','bboxes','cls','batch_idx']:self.assertTrue(torch.equal(a[key],b[key]),key)
            self.assertEqual(a['audit_source_index'],b['audit_source_index'])
            self.assertEqual(a['audit_augmentation_seed'],b['audit_augmentation_seed'])

if __name__=='__main__':unittest.main()
