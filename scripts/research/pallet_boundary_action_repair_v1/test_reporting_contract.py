"""Reporting/gate regressions with constructed fixtures, no research GT reads."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import canonical
from .registry import evaluation_protocol, GALLERY_REPRESENTATIVES
from .gallery import choose_details, run, saved_support_text


class ReportingContract(unittest.TestCase):
    def test_saved_support_formats_remain_semantically_distinct(self):
        self.assertEqual(saved_support_text({'supported_profiles': 3, 'fixed_profile_count': 4}), '3/4')
        self.assertEqual(saved_support_text([{'observation': 'historical'}]), 'historical support records 1; finite audit N/A')
        self.assertEqual(saved_support_text([]), 'historical support records 0; finite audit N/A')
        self.assertEqual(saved_support_text(None), 'N/A')
        self.assertEqual(saved_support_text({}), 'N/A')
        with self.assertRaises(TypeError): saved_support_text('malformed')

    def _selection_fixture(self,arm,category,examples):
        scored=[];predictions={};paired={}
        for representative in GALLERY_REPRESENTATIVES:
            groups={k:[]for k in ('RECOVERY','DAMAGE','BOTH_FAILURE','BOTH_SUCCESS')}
            if representative==arm:
                for index,example in enumerate(examples):
                    fid=example.get('id',f'CONSTRUCTED_SEMANTIC_{index}')
                    groups[category].append(fid)
                    before=example.get('before',(1.,1.)if category=='BOTH_SUCCESS'else(10.,10.))
                    after=example.get('after',(2.,2.)if category=='BOTH_SUCCESS'else(20.,20.))
                    scored.append(dict(id=fid,partition='TEST',metrics={
                        'E0':dict(translation_cm=before[0],rotation_deg=before[1]),
                        arm:dict(translation_cm=after[0],rotation_deg=after[1])}))
                    predictions[fid]={arm:example.get('prediction',{})}
            paired['E0__TO__'+representative]=dict(ids=groups)
        selected,categories=choose_details(scored,dict(paired=dict(PRIMARY192=paired)),predictions)
        name='BOTH_FAILURE_MAX_CONTINUOUS_WORSENING'if category=='BOTH_FAILURE'else category
        record=next(r for r in categories if r['arm']==arm and r['category']==name)
        return selected,record

    def test_protocol_survives_json_serialization(self):
        original=evaluation_protocol();decoded=json.loads(json.dumps(original))
        self.assertEqual(canonical(original),canonical(decoded))

    def test_all_nonempty_gallery_categories_covered_before_fill(self):
        scored=[];paired={}
        for j,arm in enumerate(GALLERY_REPRESENTATIVES):
            groups={k:[]for k in ('RECOVERY','DAMAGE','BOTH_FAILURE','BOTH_SUCCESS')}
            for k,category in enumerate(groups):
                fid=f'CONSTRUCTED_FIXTURE_{j}_{k}';groups[category].append(fid)
                scored.append(dict(id=fid,partition='TEST',metrics={'E0':dict(translation_cm=10.,rotation_deg=10.),
                    arm:dict(translation_cm=20.,rotation_deg=20.)}))
            paired['E0__TO__'+arm]=dict(ids=groups)
        selected,all_categories=choose_details(scored,dict(paired=dict(PRIMARY192=paired)))
        self.assertEqual(len(selected),12)
        self.assertEqual({r['category']for r in selected},
            {'RECOVERY','DAMAGE','BOTH_FAILURE_MAX_CONTINUOUS_WORSENING','BOTH_SUCCESS'})
        self.assertTrue(all(v <= 3 for v in Counter(r['arm'] for r in selected).values()))
        self.assertEqual(len(all_categories),16)

    def test_old_gallery_attempt_never_silently_resealed(self):
        with tempfile.TemporaryDirectory()as d:
            directory=Path(d);gallery=directory/'gallery';gallery.mkdir();(gallery/'old.png').write_bytes(b'not an actual scientific result')
            with self.assertRaises(FileExistsError):run(directory,directory,directory,directory,gallery)

    def test_no_continuous_worsening_is_an_empty_category(self):
        arm=GALLERY_REPRESENTATIVES[0]
        selected,record=self._selection_fixture(arm,'BOTH_FAILURE',[
            dict(after=(10.,10.)),dict(after=(9.,9.))])
        self.assertEqual(selected,[])
        self.assertEqual(record['raw_transition_count'],2)
        self.assertEqual(record['eligible_category_count'],0)
        self.assertEqual(record['population_count'],0)
        self.assertIsNone(record['selected_id'])

    def test_rotation_worsening_survives_translation_improvement(self):
        arm=GALLERY_REPRESENTATIVES[0]
        selected,record=self._selection_fixture(arm,'BOTH_FAILURE',[
            dict(id='ROTATION_WORSENED',after=(9.,11.))])
        self.assertEqual(record['eligible_category_count'],1)
        self.assertEqual(selected[0]['selected_id'],'ROTATION_WORSENED')

    def test_translation_worsening_survives_rotation_improvement(self):
        arm=GALLERY_REPRESENTATIVES[0]
        selected,record=self._selection_fixture(arm,'BOTH_FAILURE',[
            dict(id='TRANSLATION_WORSENED',after=(11.,9.))])
        self.assertEqual(record['eligible_category_count'],1)
        self.assertEqual(selected[0]['selected_id'],'TRANSLATION_WORSENED')

    def test_missing_errors_do_not_invent_a_worsening_case(self):
        arm=GALLERY_REPRESENTATIVES[0]
        selected,record=self._selection_fixture(arm,'BOTH_FAILURE',[
            dict(after=(None,None)),dict(before=(None,None))])
        self.assertEqual(selected,[])
        self.assertEqual(record['raw_transition_count'],2)
        self.assertEqual(record['eligible_category_count'],0)

    def test_worsening_order_keeps_translation_rotation_then_fixed_ID_hash(self):
        arm=GALLERY_REPRESENTATIVES[0]
        tied=('SAME_ERROR_A','SAME_ERROR_B')
        selected,record=self._selection_fixture(arm,'BOTH_FAILURE',[
            dict(id='LARGER_ROTATION_SMALLER_TRANSLATION',after=(11.,110.)),
            dict(id='SAME_TRANSLATION_SMALLER_ROTATION',after=(12.,8.)),
            *[dict(id=fid,after=(12.,13.))for fid in tied]])
        expected=min(tied,key=lambda fid:hashlib.sha256(fid.encode()).hexdigest())
        self.assertEqual(record['raw_transition_count'],4)
        self.assertEqual(record['eligible_category_count'],4)
        self.assertEqual(selected[0]['selected_id'],expected)

    def test_unprocessed_or_fallback_B_is_not_a_processed_keep_case(self):
        arm=GALLERY_REPRESENTATIVES[0]
        selected,record=self._selection_fixture(arm,'BOTH_SUCCESS',[
            dict(prediction=dict(boundary_applied=False,fallback=False)),
            dict(prediction=dict(boundary_applied=True,fallback=True)),
            dict(prediction=dict(boundary_applied=True,fallback=None)),
            dict(prediction=dict(fallback=False))])
        self.assertEqual(selected,[])
        self.assertEqual(record['raw_transition_count'],4)
        self.assertEqual(record['eligible_category_count'],0)

    def test_processed_nonfallback_B_can_preserve_success(self):
        arm=GALLERY_REPRESENTATIVES[0]
        selected,record=self._selection_fixture(arm,'BOTH_SUCCESS',[
            dict(id='PROCESSED_B_KEEP',prediction=dict(boundary_applied=True,fallback=False,selected_candidate_id='N0'))])
        self.assertEqual(record['eligible_category_count'],1)
        self.assertEqual(selected[0]['selected_id'],'PROCESSED_B_KEEP')

    def test_L_without_scored_actions_is_not_a_processed_keep_case(self):
        arm=next(a for a in GALLERY_REPRESENTATIVES if a.startswith('D1_'))
        selected,record=self._selection_fixture(arm,'BOTH_SUCCESS',[
            dict(prediction={}),dict(prediction=dict(actions=[]))])
        self.assertEqual(selected,[])
        self.assertEqual(record['raw_transition_count'],2)
        self.assertEqual(record['eligible_category_count'],0)

    def test_valid_scored_N0_retention_is_an_actual_L_keep_action(self):
        arm=next(a for a in GALLERY_REPRESENTATIVES if a.startswith('D1_'))
        prediction=dict(actions=[dict(candidate_id='N0',role='N0',source_aliases=['N0'],score=0.,sigmoid_or_softmax=.5)],
            selected_candidate_id='N0',action_role='N0',selected_score=0.,none_selected=False,fallback=False,
            prediction_status='ACTUAL_ACTION_SELECTED')
        selected,record=self._selection_fixture(arm,'BOTH_SUCCESS',[
            dict(id='ACTUAL_L_N0_KEEP',prediction=prediction)])
        self.assertEqual(record['eligible_category_count'],1)
        self.assertEqual(selected[0]['selected_id'],'ACTUAL_L_N0_KEEP')


if __name__=='__main__':unittest.main()
