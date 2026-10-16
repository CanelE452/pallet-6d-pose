"""Read-only support audit: saved costs and poses, existing-mask transforms only."""
from scripts.research.pallet_selection_score_diagnosis_v1.runtime_guard import FixedPoseOnly


class SupportAuditOnly(FixedPoseOnly):
    """Extend the existing process-local guard; do not patch shared functions."""
    cv_names = FixedPoseOnly.cv_names | {'Canny','HoughLines','HoughLinesP','fitLine'}

    def _profile(self,frame,event,arg):
        if event=='call':
            file=frame.f_code.co_filename.replace('\\','/');name=frame.f_code.co_name
            if '/pallet_selection_score_diagnosis_v1/fixed_cost.py' in file and name in {'evaluate','pilot','_outline_observations'}:
                self._reject('new_cost_evaluation',name)
            if '/pallet_selection_score_diagnosis_v1/cost_batch.py' in file and name in {'run','run_frame','_worker'}:
                self._reject('whole_bank_cost_replay',name)
            if '/pallet_selection_score_diagnosis_v1/selection.py' in file and name in {'run','replay_frame','replay_family','rank_retained'}:
                self._reject('new_selection_replay',name)
            if '/pallet_visibility_curriculum_20261013_v1/' in file and name.startswith(('refine_','solve_','extract_','detect_')):
                self._reject('new_fit_or_observation',name)
        return super()._profile(frame,event,arg)

    def receipt(self):
        return dict(super().receipt(),
                    allowed_scope='stored RGB/mask/pose/cost reads, original mask-to-hull parity, fixed projection, saved-cost arithmetic, display only',
                    whole_bank_cost_or_selection_replay_allowed=False,
                    new_neural_masks_or_RGB_line_detection_allowed=False)
