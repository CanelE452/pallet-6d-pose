"""Provenance for external reads and annotation/reference inspection."""
from env import *
def run():
    repositories={}
    for name in ('PoseFix_RELEASE','CRT-6D','integral-human-pose'):
        p=RAW/'external'/name
        repositories[name]=dict(remote=subprocess.check_output(['git','-C',str(p),'remote','get-url','origin'],text=True).strip(),commit=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip(),
            license=bound(p/'LICENSE'),readme=bound(p/('README.MD' if name=='PoseFix_RELEASE' else 'README.md')))
    write(DOC/'PRIOR_SOURCE_RECEIPTS.json',dict(accessed='2026-09-14',repositories=repositories,
        primary_papers=['https://arxiv.org/abs/1812.03595','https://arxiv.org/abs/2210.11718','https://arxiv.org/abs/1711.08229','https://openaccess.thecvf.com/content_cvpr_2018_workshops/w6/html/Fieraru_Learning_to_Refine_CVPR_2018_paper.html','https://proc.logistics-journal.de/article/view/1038','https://www.scitepress.org/publishedPapers/2026/146268/pdf/index.html'],
        search_queries=['2025 2026 monocular pallet pose keypoint refinement synthetic RGB','2024 2025 2026 image guided pose keypoint refinement network PoseFix','CRT-6D Castro official github refinement transformers','Learning to Refine Human Pose Estimation official code','pallet Kai2025 pose RGB'],
        limitation='targeted primary-source review; no exhaustive-search or formal reproduced-baseline claim; IEEEAccess2025 full-text retrieval failed'))
    files=[ROOT/'scripts/paper/pose_metric_closure_v1'/n for n in ('build_axis_review_manifest.py','build_geometry_resolved_pose_gt.py','audit_gt_pose_reference.py','run_pose_evaluation.py','evaluate_pose_by_session.py')]
    write(DOC/'REFERENCE_QA_SOURCE_AUDIT.json',dict(sources=[bound(p) for p in files],GT=bound(C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json'),
        reference_builder_uses_predictions=False,manifest_input='manual keypoint_annotations xy from original annotation; not estimator output',
        reference_inputs=['manual2D','intrinsics','known physical dimensions'],reference_type='GEOMETRY_RECONSTRUCTED_REFERENCE',
        evidence='Inspected builder reads AXIS_REVIEW_MANIFEST.manual xy and annotation camera intrinsics, chooses dimension hypothesis by annotation reprojection; no model prediction file read in reference building path.',
        limits='Original manual annotation blinding and independent physical reference accuracy not verified; model and reference share geometry/camera assumptions',
        frame_id_binding='canonical image path bijection between 2D colon IDs and pose double-underscore IDs',
        correction='Initial new exclusion-list adapter compared different ID strings. Corrected before final reporting; numerical2D and canonical pose metrics unchanged.'))
    write(DOC/'ADAPTER_CORRECTIONS.json',dict(corrections=[
        dict(component='analysis_panel pose exclusion list',issue='2D and pose frame-ID formats differ',resolution='canonical image-path bijection before coverage and paired pose stats',effect_on_P_primary='none',training_changed=False),
        dict(component='env.py pose module import',issue='historical loader restores sys.path after module import',resolution='explicit canonical pose code import path in new environment module',training_changed=False)],
        original_artifacts_modified=False,new_D_architecture_loss_schedule_changed=False))
    print('SOURCE_RECEIPTS_READY',flush=True)
if __name__=='__main__':run()
