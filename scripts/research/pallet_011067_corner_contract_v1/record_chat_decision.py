"""Transcribe the user's explicit B choice, not an agent-generated visual judgment.

Authorized by the subsequent request to enter it on the user's behalf.
Confidence was not supplied; never invent a CONFIDENT/UNCERTAIN response.
"""
from . import common as C


def main():
    assert not (C.RAW/'HUMAN_DECISION_PRIVATE.json').exists(), 'Preserve any prior human decision'
    C.put(C.RAW/'HUMAN_DECISION_PRIVATE.json',dict(
        choice='B',confidence='NOT_REPORTED',input='explicit user chat selection transcribed by assistant',
        selection_quote='그러면 b 이긴 한데 이거는 옛날 정의 아니야?우리 애초에 정의를 직사각형이니까 b a가 둘다 정답아니야?',
        authorization_quote='너가 직접 해줘 저거 뭐 어떻게 하라는건지를 모르겠어',
        scope='B for camera-facing numbering; A/B equivalence asserted for canonical180-degree symmetric pose, not all-four-C4 equivalence',
        assistant_visual_judgment_used=False,confidence_inferred=False,
        prior_model_exposure=True,mapping_discussed_before_final_signoff=True,
        ui_lock_sha256=C.sha(C.RAW/'HUMAN_UI_LOCK.json')))
    print('Recorded explicit chat choice B; confidence NOT_REPORTED; no GT/model modification')


if __name__=='__main__':main()
