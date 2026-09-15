"""Record an actual page inspection; never called automatically by compilation."""
import argparse,re
from env import *
def run(expected):
    checks=read(DOC/'PDF_CHECKS.json');assert checks['compile'] and checks['rendered']
    assert complete('MANUSCRIPT_COMPLETE'),'PDF build/input hashes must still match'
    pages=[];files=[]
    for name in ('manuscript','supplementary'):
        pdf=PAPER/(name+'.pdf');assert sha(pdf)==expected[name],('Review the current PDF, not a stale render',name)
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);n=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
        for page in range(1,n+1):
            p=RAW/'pdf_render'/f'{name}-{page:0{len(str(n))}d}.png';assert p.exists()
            pages.append(dict(document=name,page=page,render=bound(p),inspection='layout, table/equation/figure clipping, character rendering and visible cross references inspected'))
        files.append(bound(pdf))
    write(DOC/'PDF_VISUAL_REVIEW.json',dict(complete=True,reviewer='Codex visual page inspection; not human author approval or peer review',time=now(),files=files,reviewed_pages=pages,missing_results_declared=True,scientific_acceptance_not_certified=True))
    print('VISUAL_REVIEW_RECORDED',len(pages),'pages',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manuscript-sha',required=True);p.add_argument('--supplementary-sha',required=True);a=p.parse_args()
    run(dict(manuscript=a.manuscript_sha,supplementary=a.supplementary_sha))
