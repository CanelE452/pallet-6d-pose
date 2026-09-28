import copy
import unittest
from unittest.mock import patch
from . import final_audit as A


class IndependentFinalAuditContracts(unittest.TestCase):
    def test_privacy_scan_allows_hashes_not_coordinates_camera_or_pixels(self):
        value={'path':'private/R0.pt','sha256':'abc','translation_cm':1.2,
               'groups':[{'K':[[1,0,0],[0,1,0],[0,0,1]]}],'raw_target':[[1,2]],'RGB':'data:image/png;base64,x'}
        self.assertEqual(A.privacy_paths(value),['$.groups[0].K','$.raw_target','$.RGB'])
        self.assertEqual(A.privacy_paths({'checkpoint':{'path':'model.pt','sha256':'abc'},'PCK':[.1,.2]}),[])

    def test_json_pointer_handles_real_slashes_and_index(self):
        value={'groups':{'recording/a':{'pose':[{'T':3.}]}}}
        self.assertEqual(A.json_pointer(value,'/groups/recording~1a/pose/0/T'),3)
        with self.assertRaises(AssertionError):A.json_pointer(value,'not/a/pointer')

    def test_partial_or_untested_does_not_pass(self):
        self.assertEqual(A.overall({'a':dict(status='PENDING')}),'PARTIAL_PENDING')
        self.assertEqual(A.overall({'a':dict(status='NOT_RUN')}),'PASS_WITH_LIMITATIONS')
        self.assertEqual(A.overall({'a':dict(status='PASS'),'b':dict(status='FAIL')}),'FAIL')

    def test_pair_support_difference_must_be_declared_not_assumed_equal(self):
        role=dict(images=1,instances=1,supervised=8,ignore=1,invisible=0)
        raw=dict(batch=0,epoch=0,names=['one','syn__two'],images='same',boxes='same',batch_idx='same',
                 support='raw',coordinates='raw',roles={'SOURCE':role,'REAL':role},occlusion=[])
        ref=copy.deepcopy(raw);ref['support']='ref';ref['coordinates']='ref'
        declaration=dict(batches=1,differences={'support':1,'coordinates':1},
             exposures={r+'_'+k:v for r in ('SOURCE','REAL') for k,v in role.items()},
             occlusion={},support_difference_interpretation='Existing geometry clipping')
        result=A.check_pair_traces([raw],[ref],declaration)
        self.assertFalse(result['raw_ref_supervision_masks_globally_exact'])
        bad=copy.deepcopy(declaration);bad['differences']['support']=0
        with self.assertRaises(AssertionError):A.check_pair_traces([raw],[ref],bad)
        bad=copy.deepcopy(ref);bad['images']='different'
        with self.assertRaises(AssertionError):A.check_pair_traces([raw],[bad],declaration)

    def test_public_metric_report_rows_really_match_frozen_numbers(self):
        result=A.report_tables()
        self.assertEqual(result['metric_contract_table_rows_source_verified'],11)

    def test_no_mutating_historical_or_training_calls_in_audit_main(self):
        import inspect
        source=inspect.getsource(A.main)
        for token in ('C.resource(', 'C.state(', 'train(', 'infer(', 'git ', 'cuda('):self.assertNotIn(token,source)
        self.assertIn("C.save(C.DOC/'AUDIT.json'",source)


if __name__=='__main__':unittest.main()
