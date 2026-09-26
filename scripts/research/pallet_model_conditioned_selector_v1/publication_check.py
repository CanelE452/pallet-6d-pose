"""Validate all local Markdown image targets, expected figure set and scientific locks."""
import re
from . import common as C

def main():
    C.immutable();figs=[]
    expected={1:6,2:4,3:9}
    for n,total in expected.items():
        pp=sorted((C.stage(n)/'figures').glob('*.png'));assert len(pp)==total
        figs.extend(C.bind(p) for p in pp)
    examples=[p for n in (1,3) for p in (C.stage(n)/'examples').glob('*.jpg')];assert len(examples)==12
    links=[]
    for md in C.DOC.rglob('*.md'):
        for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md.read_text()):
            assert not target.startswith(('http:','https:')),'Repo-contained figures only'
            p=(md.parent/target).resolve();assert p.is_file() and p.is_relative_to(C.DOC);links.append(dict(markdown=str(md.relative_to(C.DOC)),target=target))
    assert C.read(C.DOC/'FINAL_AUDIT.json')['passed']
    for b in C.read(C.stage(2)/'SCORER_LOCK.json')['checkpoints'].values():C.verify(b)
    for stage,file,key in [(2,'SYNTH_FEATURES_LOCK.json','files'),(3,'REAL_SELECTOR_DECISION_LOCK.json','decisions')]:
        b=C.read(C.stage(stage)/file)[key]
        for row in b if isinstance(b,list) else [b]:C.verify(row)
    manifest=dict(created_at=C.now(),passed=True,required_chart_count=len(figs),example_count=len(examples),markdown_image_references=len(links),
        figures=figs,examples=[C.bind(p) for p in sorted(examples)],links=links,
        source_code=[C.bind(p) for p in sorted((C.ROOT/'scripts/research'/C.NAME).glob('*.py'))],
        scientific_outputs=[C.bind(p) for p in sorted(C.DOC.rglob('*.json')) if p.name not in ('PUBLICATION_MANIFEST.json','FINAL_CLI_OUTPUT.json') and not p.name.endswith('_GIT.json')])
    C.save(C.DOC/'PUBLICATION_MANIFEST.json',manifest,immutable=False);print('PUBLICATION_CHECK',len(figs),len(examples),len(links),flush=True)

if __name__=='__main__':main()
