"""Source label rules. Semantic self visibility is intentionally not inferred."""
import math

VISIBLE = 'VISIBLE_TARGET'
EXTERNAL = 'EXTERNAL_OCCLUDED_SUPPORT'
OUTSIDE = 'OUTSIDE_TARGET_SUPPORT'
INVALID = 'INVALID_QUERY'


def nearest_pixel(x, y, width, height):
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    px, py = math.floor(x + .5), math.floor(y + .5)
    if not (0 <= x < width and 0 <= y < height and 0 <= px < width and 0 <= py < height):
        return None
    return px, py


def source_state(full, visible):
    """Pixel-level support state, never a semantic self label."""
    full, visible = bool(full), bool(visible)
    if visible and not full:
        raise ValueError('Visible target mask is not a subset of full target mask')
    return VISIBLE if visible else EXTERNAL if full else OUTSIDE


def supervised_targets(full, visible):
    state = source_state(full, visible)
    return {'state': state, 'exact_presence': int(bool(visible)),
            'external_given_full': int(not visible) if full else None,
            'semantic_self': None, 'semantic_direct': None}


def selfcheck():
    assert source_state(1, 1) == VISIBLE
    assert source_state(1, 0) == EXTERNAL
    assert source_state(0, 0) == OUTSIDE
    assert supervised_targets(0, 0)['external_given_full'] is None
    assert all(supervised_targets(*p)['semantic_self'] is None for p in [(1, 1), (1, 0), (0, 0)])
    try:
        source_state(0, 1)
        raise AssertionError('Invalid full/visible pair accepted')
    except ValueError:
        pass
    assert nearest_pixel(.5, .5, 10, 10) == (1, 1)
    assert nearest_pixel(9.6, 0., 10, 10) is None
    assert nearest_pixel(-.1, 0., 10, 10) is None
    assert nearest_pixel(float('nan'), 0., 10, 10) is None
    return {'source_label_cases': 4, 'nearest_pixel_cases': 4, 'semantic_self_target_generated': False}


if __name__ == '__main__':
    import json
    print(json.dumps(selfcheck()))
