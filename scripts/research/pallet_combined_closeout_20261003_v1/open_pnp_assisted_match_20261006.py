"""Use annotation.py for actual selected-pallet decisions, without history popups.

Existing G-saved PnP annotations supply a separate geometry reference panel.
This viewer displays only the original image and frozen selected box. Its
buttons never approve coordinates or change the original visible-only contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIFTER = ROOT / 'scripts/research/pallet_lifter_case_review_20261003_v1'
REVIEW = ROOT / 'data/pallet/results/pallet_lifter_case_review_20261003_v1/review'
sys.path.insert(0, str(LIFTER))
sys.path.insert(0, str(LIFTER / 'combined_integration'))
from open_object_match_annotation import NativeObjectReview, CHOICES
from object_match_review import validate_reviewer
from bridge import read_json, write_json


class PnPAssistedTargetReview(NativeObjectReview):
    window_title = 'Annotate - Pallet target check'

    def __init__(self, ctx):
        super().__init__(ctx)
        self.evaluation_started = False
        self.display_started_at = None
        self.display_start_mono = None

    def load(self, state, path, read_only=False):
        result = super().load(state, path, read_only)
        self.display_started_at = datetime.now(timezone.utc).isoformat()
        self.display_start_mono = time.monotonic()
        return result

    def confirm_profile(self):
        # Called only by a real 1/2/3 key/button decision, never by opening the
        # window. The current box exposure and prior displayed annotations are
        # workflow facts, not guesses about earlier external model exposure.
        profile = REVIEW / 'NATIVE_REVIEWER_PROFILE.json'
        prior = read_json(profile) if profile.is_file() else None
        alias = (prior or {}).get('reviewer', {}).get('id') or '검수자-' + uuid.uuid4().hex[:8]
        self.reviewer = dict(id=alias, entered_by='human', confirmation=True,
            machine_assistance=True, previous_prediction_exposure=True,
            previous_annotation_exposure=True,
            exposure_notes='이번 실제 대상 판정은 원사진과 고정 선택 네모를 보고 1/2/3을 직접 선택함. '
                           '현재 네모 노출은 실제 도구 기록이며 주석 작성 당시의 과거 노출 답변과 구분함.',
            confirmation_source='actual_selected_pallet_button_or_key',
            exposure_scope='current_selected_pallet_matching_stage',
            prior_annotation_history_source=(dict(path=str(profile),
                sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                actual_answers=prior['reviewer']) if prior else
                dict(status='UNCONFIRMED', no_false_answer_created=True)))
        validate_reviewer(self.reviewer)
        write_json(self.profile_path, self.reviewer)
        return True

    def choose(self, value):
        if value not in CHOICES:
            return False
        if self.reviewer is None and not self.confirm_profile():
            return False
        if self.token is None:
            self.token = self.ctx.start(dict(frame_id=self.current, reviewer=self.reviewer))['token']
        # Keep the actual time spent viewing the image, including the first
        # frame before its first human choice established the local reviewer.
        if self.display_started_at is not None:
            session = self.ctx.sessions[self.token]
            session['started_at'] = self.display_started_at
            session['start_mono'] = self.display_start_mono
        return super().choose(value)

    def export(self):
        super().export()
        evaluator = HERE / 'evaluate_pnp_assisted_lifter_20261006.py'
        if self.ctx.counts()['pending'] == 0 and not self.evaluation_started and evaluator.is_file():
            log_path = self.ctx.store_path.parent / 'AUTO_EVALUATION.log'
            with log_path.open('a', encoding='utf-8') as handle:
                subprocess.Popen([sys.executable, str(evaluator)], cwd=ROOT,
                                 stdout=handle, stderr=subprocess.STDOUT)
            self.evaluation_started = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    from pnp_assisted_reference_20261006 import prepare_context
    ctx = prepare_context()
    print(json.dumps(dict(status='READY_PNP_ASSISTED_TARGET_REVIEW',
        counts=ctx.counts(), history_popup=False, new_corner_clicks_required=0,
        official_visible_reference_created=False, editor='scripts/annotate/annotate.py'),
        ensure_ascii=False), flush=True)
    if args.prepare_only:
        return 0
    lock = (ctx.store_path.parent / 'native_object_match.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(ROOT / 'scripts/annotate'))
    import annotate
    import annotate_review
    native = PnPAssistedTargetReview(ctx)
    native.install(annotate)
    annotate_review.load_review_contexts = native.contexts
    annotate.main(['--review-manifest', str(REVIEW / 'MANIFEST.json'), '--stride', '1',
        '--population-role', 'DEV', '--default_split', 'eval', '--object-type', 'plastic_square',
        '--geometry-registry', str(ROOT / 'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'),
        '--win-w', '1290', '--win-h', '830'])
    native.export()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
