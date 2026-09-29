from .report import aggregate, point_scope


def point(real, combined, corner=0, role='REAL'):
    return dict(image_id='private_fixture', role=role, corner=corner, residual_px=2.,
                signs=dict(real=real, source='TOWARD', combined=combined),
                response_px_at_common_reference_step={k: [.01, .01, .01]
                    for k in ('real', 'source', 'combined')})


def test_transitions_do_not_hide_opposite_benefit_or_unresolved():
    rows = [point('TOWARD', 'AWAY'), point('AWAY', 'TOWARD'),
            point('TOWARD', 'TOWARD'), point('TOWARD', 'NEAR_ZERO'),
            point('TOWARD', 'EPS_SENSITIVE')]
    result = aggregate(rows, 'real')
    assert result['points'] == 5
    assert result['own_toward'] == 4
    assert result['resolved_own_toward_denominator'] == 3
    assert result['cancellation'] == result['suppression'] == result['retained_toward'] == 1
    assert result['own_toward_combined_unresolved'] == 1
    assert result['sign_transition_counts']['AWAY_TO_TOWARD'] == 1
    assert result['images'] == 1


def test_corners_center_and_empty_scopes():
    rows = [point('TOWARD', 'AWAY', corner=0), point('TOWARD', 'TOWARD', corner=8)]
    assert len(point_scope(rows, 'REAL', 'CORNERS_0_7')) == 1
    assert len(point_scope(rows, 'REAL', 'CENTER_8')) == 1
    assert len(point_scope(rows, 'REAL', 'ALL_9')) == 2
    result = aggregate(point_scope(rows, 'SOURCE', 'ALL_9'), 'source')
    assert result['points'] == 0
    assert result['residual_px']['median'] is None
    assert result['cancellation_fraction'] is None
