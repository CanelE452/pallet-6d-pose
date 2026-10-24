"""Create a NEW prefix-stable YCB review batch using the preserved builders.

Only source metadata/RGB/masks; no scores, models, training or private inputs.
"""
from pathlib import Path
import argparse
import importlib.util
import json

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--train-count',type=int,default=8)
    parser.add_argument('--val-count',type=int,default=4)
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    original=load('ycb_intake_original_builder',HERE/'build_ycb_cue_review.py')
    renderer=load('ycb_intake_fixed_renderer',HERE/'build_ycb_cue_review_v2.py')
    original.require(not args.out.exists(),'New output directory required; never overwrite a fixed review')
    original.require(args.train_count>0 and args.val_count>0,'Both source splits required')
    args.out.mkdir(parents=True)
    original.write_json(args.out/'EXPANSION_ENTRY.json',dict(
        source_only=True,code=original.ref(__file__),metadata_builder=original.ref(HERE/'build_ycb_cue_review.py'),
        renderer=original.ref(HERE/'build_ycb_cue_review_v2.py'),
        train_count=args.train_count,val_count=args.val_count,
        source_RGB_reads_before_selection=0,score_model_current_train_PnP_reads_or_calls=0))
    original.freeze(args.out,args.train_count,args.val_count)
    # The original immutable builder contains the metadata-only selection.
    # V2 changes rendering variable names only; bind it explicitly for this NEW batch.
    original.write_json(args.out/'RENDER_FIX_V2.json',dict(
        schema='V71_YCB_REUSABLE_RENDER_BINDING_V2',original_code=original.ref(HERE/'build_ycb_cue_review.py'),
        render_code=original.ref(HERE/'build_ycb_cue_review_v2.py'),
        selection_lock=original.ref(args.out/'SELECTION_LOCK.json'),expansion_entry=original.ref(args.out/'EXPANSION_ENTRY.json'),
        change='Use separately bound corrected renderer; original SHA selection and all geometry unchanged.',
        selection_or_source_query_or_geometry_or_labels_changed=False))
    if not args.freeze_only:
        renderer.render(args.out)
    print(json.dumps(dict(output=str(args.out.resolve()),freeze_only=args.freeze_only)))


if __name__=='__main__':
    main()
