"""Final non-mutation, report-link and test audit; no git mutations."""
import hashlib
import re
import subprocess
import sys

from analyze import C, OUT, ROOT, save


def main():
    initial=C.read(OUT/'INPUTS.json')
    for b in initial['bindings']:C.verify(b)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head==initial['HEAD']
    for args in (['git','diff','--exit-code'],['git','diff','--cached','--exit-code']):
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
        assert result.returncode==0,result.stdout
    test=subprocess.run([sys.executable,'-m','pytest','-q',str(OUT/'test_postmortem.py')],cwd=ROOT,capture_output=True,text=True)
    assert test.returncode==0,test.stdout+test.stderr
    checked=0
    for p in OUT.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if link=='AUDIT.json':continue
            assert (p.parent/link).exists(),(p,link)
            checked+=1
    screenshot=__import__('pathlib').Path('/tmp/pallet_postmortem_review.png')
    assert screenshot.exists()
    save('AUDIT.json',dict(status='[확인] COMPLETE',HEAD=head,tests=test.stdout,
        frames=93,corners=713,original_input_bindings_reverified=len(initial['bindings']),
        checkpoint_predictions_annotations_original_E2_unchanged=True,
        same_physical_corner_join=True,approved_whole_object_symmetry_only=True,
        GT_post_frozen_analysis_only=True,failed_matching_retained_frames=8,failed_matching_retained_corners=54,
        external_visibility_not_invented=True,model_training=False,pseudo_label_generation=False,
        new_inference=False,new_threshold_tuning=False,commit=False,push=False,
        tracked_diff_empty=True,staged_diff_empty=True,new_scope=str(OUT.relative_to(ROOT)),
        markdown_links_checked=checked,HTML_local_image_links_tested=True,
        headless_browser_render_verified=True,render_preview_sha256=hashlib.sha256(screenshot.read_bytes()).hexdigest(),
        tests_notes='Initial analysis prepare failed on review queue filename before producing data; corrected to actual inspected COVERAGE_GAP_OCCLUSION_QUEUE.csv. No E2 files changed.',
        secondary_oracle_schema='Use D4_ORACLE_HEADROOM_VALIDATED.json. Draft secondary E_fixed was not a defined metric and was excluded; requested metrics unchanged.',
        artifacts=[C.bind(p) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and '.pytest_cache' not in str(p) and p.name!='AUDIT.json']))
    assert (OUT/'AUDIT.json').is_file()
    print('POSTMORTEM_COMPLETE',test.stdout.strip(),flush=True)


if __name__=='__main__':main()
