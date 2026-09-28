import copy
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ultralytics.data.augment import Compose, RandomPerspective
from ultralytics.utils.instance import Instances
from . import cycle_affine as A


def fixture(name):
    image=np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3)
    points=np.array([[[16,16,2],[48,16,2],[48,48,2],[16,48,2],[20,20,1],[44,20,1],[44,44,2],[20,44,2],[32,32,2]]],dtype=np.float32)
    instances=Instances(np.array([[12,12,52,52]],np.float32),np.zeros((0,1000,2),np.float32),points,bbox_format='xyxy',normalized=False)
    return dict(img=image,im_file=name,cls=np.zeros((1,1),np.float32),instances=instances)


class AffineOffContracts(unittest.TestCase):
    def test_source_exact_and_RNG_schedule_unchanged(self):
        transform=RandomPerspective(degrees=0,translate=.1,scale=.25,shear=0,perspective=0)
        wrapped=A.RealAffineOff(transform)
        for name in ('syn__a.png','real_a.png'):
            A.seed(432); old=transform(fixture(name)); rng_old=A.rng_digest()
            A.seed(432); new=wrapped(fixture(name)); rng_new=A.rng_digest()
            self.assertEqual(rng_old,rng_new)
            self.assertEqual((transform.translate,transform.scale),(.1,.25))
            if name.startswith('syn__'):
                np.testing.assert_array_equal(old['img'],new['img'])
                np.testing.assert_array_equal(old['instances'].keypoints,new['instances'].keypoints)
            else:
                np.testing.assert_array_equal(new['img'],fixture(name)['img'])
                np.testing.assert_array_equal(new['instances'].keypoints,fixture(name)['instances'].keypoints)
                self.assertFalse(np.array_equal(old['img'],new['img']))

    def test_all_uniform_draws_preserved(self):
        original=random.uniform
        transform=RandomPerspective(degrees=0,translate=.1,scale=.25,shear=0,perspective=0)
        for name in ('real.png','syn__source.png'):
            with patch('random.uniform', wraps=original) as uniform:
                A.RealAffineOff(transform)(fixture(name))
                self.assertEqual(uniform.call_count,8)

    def test_nested_transform_wrap_and_missing_identity(self):
        tree=Compose([Compose([RandomPerspective(translate=.1,scale=.25)])])
        result,count=A.wrap_affine(tree)
        self.assertEqual(count,1)
        self.assertIsInstance(result.transforms[0].transforms[0],A.RealAffineOff)
        _, count=A.wrap_affine(result);self.assertEqual(count,0)
        with self.assertRaises(AssertionError):result.transforms[0].transforms[0]({'img':np.zeros((3,3,3))})

    def test_ranges_restored_after_transform_failure(self):
        transform=RandomPerspective(translate=.1,scale=.25)
        with self.assertRaises(KeyError):A.RealAffineOff(transform)({'im_file':'real.png'})
        self.assertEqual((transform.translate,transform.scale),(.1,.25))

    def test_completed_fits_all_buffers_and_all_batches_paired(self):
        if not (A.RAW/'FIT_WOOD_REF.json').exists():self.skipTest('Fits not complete')
        for material in ('PLASTIC','WOOD'):
            raw,ref=[A.C.read(A.RAW/f'TRACE_{material}_{target}.json') for target in ('RAW','REF')]
            self.assertEqual(len(raw),320);self.assertEqual(len(ref),320)
            for x,y in zip(raw,ref):
                for key in ('names','images','boxes','support','batch_idx','roles'):self.assertEqual(x[key],y[key])
        initial=A.C.read(A.RAW/'FIT_PLASTIC_RAW.json')['initialization'];A.C.verify(initial)
        model=torch.load(A.C.ROOT/initial['path'],map_location='cpu',weights_only=False)['model'].float()
        base=model.state_dict();buffers={n for n,_ in model.named_buffers()}
        protected={n for n in base if n in buffers or not A.pose_parameter(n)}
        self.assertEqual(len(protected),747)
        for arm in A.ARMS:
            fit=A.C.read(A.RAW/f'FIT_{arm}.json');self.assertEqual(fit['optimizer_steps'],320);A.C.verify(fit['checkpoint'])
            final=torch.load(A.C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
            self.assertTrue(all(torch.equal(base[n],final[n]) for n in protected))

    def test_locked_inference_and_public_denominators(self):
        if not (A.DOC/'RESULTS.json').exists():self.skipTest('Evaluation not complete')
        result=A.C.read(A.DOC/'RESULTS.json');lock=A.C.read(A.RAW/'PREDICTIONS_LOCK.json')
        for binding in lock['files']+result['artifact_sources']:A.C.verify(binding)
        self.assertTrue(lock['no_reference_coordinates_read'])
        for material,n,k,m in [('PLASTIC',128,985,120),('WOOD',45,346,45)]:
            for value in result['materials'][material]['groups']['ALL'].values():
                self.assertEqual(value['twoD']['total_frames'],n)
                self.assertEqual(value['twoD']['corners'],k)
                self.assertEqual(value['fixed_ID']['corners'],k)
                self.assertEqual(value['twoD']['matched'],m)
                self.assertEqual(value['sixD']['frames'],n)
                self.assertEqual(value['sixD']['pose_coverage'],1.)
        self.assertEqual(result['verified66']['points'],66)

    def test_public_main_table_matches_saved_results(self):
        path=A.DOC/'REPORT_KO.md'
        if not path.exists():self.skipTest('Report not complete')
        result=A.C.read(A.DOC/'RESULTS.json')
        aliases={'R0':'R0','old RAW':'OLD_RAW','old REF':'OLD_REF','SYN':'SYN','NEW_RAW':'NEW_RAW','NEW_REF':'NEW_REF'}
        checked=0
        for line in path.read_text().splitlines():
            cells=[x.strip() for x in line.split('|')[1:-1]]
            if len(cells)!=7 or not cells[0].startswith(('Plastic ','Wood ')):continue
            material,name=cells[0].split(' ',1)
            if name not in aliases:continue
            row=result['materials'][material.upper()]['groups']['ALL'][aliases[name]]
            for cell,threshold in zip(cells[1:4],('5','10','20')):
                self.assertAlmostEqual(float(cell),100*row['twoD']['PCK'][threshold],delta=.00051)
            median,p90=map(float,cells[4].split('/'))
            self.assertAlmostEqual(median,row['twoD']['full_penalty_median_px'],delta=.0051)
            self.assertAlmostEqual(p90,row['twoD']['full_penalty_P90_px'],delta=.0051)
            self.assertEqual(int(cells[5]),row['twoD']['tail_gt20_count'])
            self.assertAlmostEqual(float(cells[6]),row['sixD']['ADDsym_AUC'],delta=.00000000051)
            checked+=1
        self.assertEqual(checked,12)


if __name__=='__main__':unittest.main()
