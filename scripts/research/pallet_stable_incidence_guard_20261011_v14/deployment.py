"""Fresh V13 finite consensus, followed by the fixed selected-pose guard.

No accuracy rows or saved guard decisions enter a prediction. The original
finite optimizer and endpoint LOO run once, and the normal incidence guard
runs on that same fresh result before the final prediction is returned.
"""
from collections import Counter
import copy
import numpy as np

from . import pipeline as G
from ..pallet_finite_edge_consensus_repair_20261011_v13 import deployment as D13
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import preserve_prediction
from ..pallet_sparse_local_line_20261010_v9 import common as C

STAGES=D13.STAGES


class Pipeline(D13.Pipeline):
    def __init__(self,args):
        super().__init__(args)
        self.guard_operation_counts=Counter()
        self.last_guard_ledger=self.last_parent_output=None

    def predict(self,image,K,xyz,metadata=None,method=G.PRIMARY,stage_callback=None):
        G.require(method==G.PRIMARY,'single fresh stable-incidence method required')
        self.last_guard_ledger=self.last_parent_output=None
        self.counts[method+':started']+=1
        # Keep the existing stage boundaries. The final stage now includes
        # the incidence guard and prediction metadata before its timestamp.
        final_seen=[]
        def stage(name):
            if name==STAGES[-1]:final_seen.append(name)
            elif stage_callback:stage_callback(name)
        parent=super().predict(image,K,xyz,metadata,method=D13.P.PRIMARY,stage_callback=stage)
        self.last_parent_output=parent
        G.require(final_seen==[STAGES[-1]],'original finite final stage differs')
        self.counts['fresh_stable_guard_paths_started']+=1
        self.last_execution_events.append(dict(phase='fresh_stable_incidence_guard_started'))
        with self.old.no_truth_reads():
            row,witness=G.guard_row(parent)
            self.guard_operation_counts.update({k:v for k,v in witness['counts'].items() if type(v) is int})
            self.last_guard_ledger=dict(schema='fresh_selected_stable_incidence_ledger_v14',
                id=row['id'],method=G.PRIMARY,parent_finite_output_semantic_sha256=C.semantic_sha256(parent),
                guarded_output_before_prediction_update_semantic_sha256=C.semantic_sha256(row),
                stable_incidence_guard=copy.deepcopy(witness),guard_counts=copy.deepcopy(witness['counts']),
                saved_accuracy_row_or_guard_decision_used=False,original_finite_optimizer_run_once=True,
                new_guard_optimizer_PnP_candidate_selection_calls=0)
            # A declined finite candidate returns the complete original N3
            # output. Rebuild metadata from the same fresh detector capture.
            prediction=copy.deepcopy(self.last_capture['prediction'])
            selected=prediction['selected_index']
            if selected is not None:prediction['candidates'][selected]['keypoints_xy']=row['native_points']
            preserve_prediction(self.last_capture['raw'],prediction)
            G.require(np.array_equal(np.asarray(row['native_points'])[8],
                np.asarray(self.last_capture['native_N3_points'])[8],equal_nan=True),'fresh center changed')
            row.update(prediction=prediction,fresh_guard_ledger=copy.deepcopy(self.last_guard_ledger),
                fresh_stable_incidence_recomputed_inside_call=True,
                fresh_parent_finite_result_used=True,parent_finite_solved_again_by_guard=False,
                saved_accuracy_guard_or_pose_used=False,
                final_stage_includes_stable_incidence_and_prediction_metadata=True,
                accepted_H_projection_assigned_again=False)
        self.last_execution_events.append(dict(phase='fresh_stable_incidence_guard_returned'))
        self.counts['fresh_stable_guard_paths_complete']+=1
        self.counts[method+':complete']+=1
        if stage_callback:stage_callback(STAGES[-1])
        return row
