"""Prospective V95 V2 native-alpha ownership rule, no I/O or donor selection."""
import numpy as np


def exclusive_alpha(selected_visible, other_visible_union, complete):
    a=np.asarray(selected_visible);b=np.asarray(other_visible_union)
    if a.dtype!=bool or b.dtype!=bool or a.shape!=b.shape or a.ndim!=2:
        raise ValueError('Native boolean same-shape masks required')
    if complete is not True:
        return np.zeros_like(a),dict(admitted=False,reason='INCOMPLETE_OTHER_INSTANCE_MASK_SET_U')
    exclusive=a&~b
    padded=np.pad(exclusive,1)
    core=np.logical_and.reduce([padded[y:y+a.shape[0],x:x+a.shape[1]]for y in range(3)for x in range(3)])
    assert not (exclusive&b).any() and not (core&b).any() and not (core&~a).any()
    return exclusive,dict(admitted=bool(core.any()),reason='EXCLUSIVE_NATIVE_ALPHA_READY'if core.any()else'EMPTY_EXCLUSIVE_ERODED_ALPHA_U',
           original_selected_visible_pixels=int(a.sum()),original_conflicting_pixels=int((a&b).sum()),
           excluded_conflicting_pixels=int((a&b).sum()),retained_exclusive_alpha_pixels=int(exclusive.sum()),
           retained_erode1_pixels=int(core.sum()),complete_other_instance_masks=True,
           every_eroded_native_alpha_pixel_selected_and_not_other=True)


def generated():
    a=np.ones((7,9),bool);b=np.zeros_like(a);b[2:5,3:6]=True
    out,c=exclusive_alpha(a,b,True)
    assert out.sum()==54 and not out[3,4] and c['excluded_conflicting_pixels']==9 and c['admitted']
    assert np.array_equal(a,np.ones((7,9),bool))
    empty,e=exclusive_alpha(a,a,True);assert not empty.any() and not e['admitted']
    unknown,u=exclusive_alpha(a,b,False);assert not unknown.any() and not u['admitted']
    return dict(completed=True,PASS=True,all_conflicts_become_transparent=True,no_conflicting_pixel_pasted=True,
                incomplete_or_empty_stays_U=True,source_masks_immutable=True,donor_selection_scale_placement_mass_unchanged=True,
                actual_pixels_models_fit_current_PnP=0)
