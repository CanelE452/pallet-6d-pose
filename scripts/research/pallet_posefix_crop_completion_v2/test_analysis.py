from .evaluate import transition

def test_empty_error_band_transition():assert transition([],'C','A')==dict(GG=0,GB=0,BG=0,BB=0)

def test_nonempty_transition_conservation():
    rows=[dict(errors=dict(A=a,C=c)) for a,c in [(5,5),(5,15),(15,5),(15,15)]]
    assert transition(rows,'C','A')==dict(GG=1,GB=1,BG=1,BB=1)
