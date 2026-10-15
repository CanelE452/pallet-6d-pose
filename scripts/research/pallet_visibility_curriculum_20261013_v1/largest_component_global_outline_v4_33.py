"""One fixed LCC8 preprocessing control for the frozen global outline cue.

This module reads a supplied cached mask only. It has no RGB, prompt, state,
keypoint, pose, reference, frame-ID, checkpoint or inference interface. The
original union baseline and the LCC variant use the same frozen V13 sampler;
the C numerical function is the unchanged V25 function object.
"""
from __future__ import annotations

import hashlib
import cv2
import numpy as np

from . import statewise_partial_BI_outline_solver_v4_13 as FROZEN_OUTLINE
from . import global_foreground_containment_v4_25 as FROZEN_CONTAINMENT


SELECTION_PROFILE = 'LCC8_MAX_AREA_LOWEST_LABEL_FROZEN_V13_OUTLINE_V33'
BASELINE_PROFILE = 'ORIGINAL_UNION_FROZEN_V13_OUTLINE'
MASK_THRESHOLD = .5  # Same binary membership used inside the frozen observer.
CONNECTIVITY = 8
RECIPE = dict(
    selection_profile=SELECTION_PROFILE,
    mask_threshold=MASK_THRESHOLD,
    connectivity=CONNECTIVITY,
    component_selection='maximum foreground pixel area; smallest OpenCV label on exact area tie',
    labels='cv2.connectedComponentsWithStats, CV_32S, connectivity=8; label0 is background',
    output='binary float32 mask containing exactly the selected foreground component',
    empty_foreground='all-zero mask, selected label None, unsupported observation retained',
    holes_filled=False, morphology=False, threshold_search=False,
    outline_sampling='unchanged frozen V13 observe_global_outline',
    C_numerical_function='unchanged frozen V25 outline_value_and_gradient',
    original_union_baseline_retained=True,
    component_semantic_identity_or_visibility_certified=False,
    state_O_S_U_or_GT_or_E_pose_or_frame_ID_routing=False)

# Exact function aliases: no copied/edited numeric kernel and no global patch.
outline_value_and_gradient = FROZEN_CONTAINMENT.outline_value_and_gradient
frozen_union_observer = FROZEN_OUTLINE.observe_global_outline


def _pixel_sha(mask):
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def largest_component_mask(pallet_mask, image_hw):
    """Select one 8-connected foreground component uniformly, without mutation.

    Foreground is mask >= .5 exactly as in V13. Ties use the lowest returned
    positive OpenCV label, not geometry/quality/pose/GT or a frame-specific rule.
    No retained component is filled, expanded, eroded or convexified here.
    """
    if len(image_hw) != 2:
        raise ValueError('Two canvas dimensions required')
    h, w = map(int, image_hw)
    values = np.asarray(pallet_mask, dtype=float)
    if (h < 1 or w < 1 or values.shape != (h, w)
            or not np.isfinite(values).all()
            or np.any((values < 0.) | (values > 1.))):
        raise ValueError('Bound finite global mask in [0,1] on a positive canvas required')
    binary = np.ascontiguousarray(values >= MASK_THRESHOLD, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=CONNECTIVITY, ltype=cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA]
    selected_label, tied_labels = None, []
    if len(areas):
        maximum = int(areas.max())
        tied_labels = (np.flatnonzero(areas == maximum)+1).astype(int).tolist()
        selected_label = tied_labels[0]
        selected = (labels == selected_label).astype(np.float32)
    else:
        selected = np.zeros((h, w), dtype=np.float32)
    components = [dict(label=int(label), area_px=int(stats[label, cv2.CC_STAT_AREA]),
                       bbox_xywh=stats[label, :4].astype(int).tolist(),
                       retained=label == selected_label)
                  for label in range(1, count)]
    audit = dict(
        selection_profile=SELECTION_PROFILE, connectivity=CONNECTIVITY,
        mask_threshold=MASK_THRESHOLD, opencv_version=cv2.__version__,
        foreground_component_count=int(count-1), selected_label=selected_label,
        maximum_area_tied_labels=tied_labels,
        original_foreground_area_px=int(binary.sum()),
        retained_foreground_area_px=int(selected.sum()),
        discarded_foreground_area_px=int(binary.sum()-selected.sum()),
        original_union_binary_pixel_sha256=_pixel_sha(binary),
        selected_binary_pixel_sha256=_pixel_sha(selected.astype(np.uint8)),
        components=components, holes_filled=False, morphology_used=False,
        convex_hull_used_in_component_selection=False,
        semantic_pallet_membership_proved=False,
        disconnected_true_fragments_may_be_discarded=True,
        original_mask_modified=False, new_inference_or_local_queries=0)
    return selected, audit


def observe_largest_component_outline(pallet_mask, image_hw, *, image_binding=None,
                                      configuration=None):
    """LCC8 first, then the exact frozen V13 mask-to-outline operation."""
    selected, audit = largest_component_mask(pallet_mask, image_hw)
    observation = frozen_union_observer(selected, image_hw,
        image_binding=image_binding, configuration=configuration)
    return dict(observation, foreground_component_selection=audit,
                selection_profile=SELECTION_PROFILE)


def observe_union_and_largest_component(pallet_mask, image_hw, *, image_binding=None,
                                        configuration=None):
    """Return both fixed controls, including every unsupported/empty case.

    No decision is made between these observations. Future frame operators
    must retain their full populations and failures rather than route by error.
    """
    union = frozen_union_observer(pallet_mask, image_hw,
        image_binding=image_binding, configuration=configuration)
    lcc = observe_largest_component_outline(pallet_mask, image_hw,
        image_binding=image_binding, configuration=configuration)
    return dict(schema='UNION_AND_LCC8_FROZEN_GLOBAL_OUTLINE_CONTROLS_V33',
        original_union=dict(union, selection_profile=BASELINE_PROFILE),
        largest_component=lcc, row_retained=True,
        observation_chosen_for_runtime=False,
        new_NN_or_RGB_or_prompt_or_point_queries=0,
        semantic_or_accuracy_adoption_claim=False)
