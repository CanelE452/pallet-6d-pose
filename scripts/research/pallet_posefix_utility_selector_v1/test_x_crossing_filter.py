import numpy as np
import pytest
from . import x_crossing_filter as X


@pytest.mark.parametrize('q',[
    [[0,0],[1,0],[1,1],[0,1]],
    [[0,0],[1,0],[.2,.2],[0,1]],  # Concave but not X: explicitly allowed.
    [[0,0],[1,0],[1,0],[0,1]],  # Touch/degenerate: no extra veto.
    [[0,0],[1,0],[2,0],[3,0]],
    [[0,0],[0,1],[1,1],[1,0]],  # Clockwise vs counterclockwise irrelevant.
])
def test_non_crossing_shapes_allowed(q):
    assert not X.quadrilateral_x(q)


def test_bowtie_rejected_and_transformation_invariant():
    q=np.array([[0,0],[1,1],[0,1],[1,0]],float)
    assert X.quadrilateral_x(q)
    assert X.quadrilateral_x(q[::-1])
    assert X.quadrilateral_x(q*3+10)


def test_only_proper_intersections():
    q=np.array([[0,0],[1,1],[0,1],[1,0]],float)
    assert X.proper_crossing(*q)
    assert not X.proper_crossing(np.array([0,0]),np.array([1,1]),np.array([1,1]),np.array([2,0]))


def test_cross_face_lines_not_global_veto():
    # Two nested square faces and proper connecting sides: some global edges
    # overlap in projection, but no individual face is a bowtie.
    q=np.array([[0,0],[4,0],[4,4],[0,4],[1,1],[3,1],[3,3],[1,3],[2,2]],float)
    before=q.copy();assert X.inspect(q)['keep'];assert np.array_equal(q,before)


def test_previous_geometric_vetoes_not_implicitly_applied():
    q=np.zeros((9,2),float)  # Whole shape collapsed: this X-only function cannot veto it.
    assert X.inspect(q)['keep']
