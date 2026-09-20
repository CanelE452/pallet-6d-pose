from html.parser import HTMLParser
from . import common as C
from .extreme_low_subset import DOC, OUT


def test_frozen_partition_and_scope():
    m=C.read(DOC/'SUBSET_MANIFEST.json')
    original=C.read(C.DOC/'EVAL_PROTOCOL.json')['records']
    decisions=m['decisions']
    assert len({r['id'] for r in decisions})==194
    excluded={r['id'] for r in decisions if r['decision']=='exclude'}
    assert len(excluded)==35
    assert m['records']==[r for r in original if r['id'] not in excluded]
    assert {r['id'] for r in original}==set(m['training_prohibited_ids'])
    assert m['counts']==dict(PLASTIC=159,WOOD=125,GREEN=150)


def test_gallery_image_links():
    class Images(HTMLParser):
        def __init__(self):super().__init__();self.sources=[]
        def handle_starttag(self,tag,attrs):
            if tag=='img':self.sources.append(dict(attrs)['src'])
    p=Images();p.feed((OUT/'excluded.html').read_text())
    assert len(p.sources)==35
    assert all((OUT/s).is_file() for s in p.sources)


def test_metric_population_and_provenance():
    result=C.read(DOC/'RESULTS.json')
    C.verify(result['subset'])
    for b in result['sources']:C.verify(b)
    for summaries in result['results'].values():
        assert [summaries[k]['total_frames'] for k in ('full194','retained159','excluded35')]==[194,159,35]
        assert summaries['full194']['corners']==summaries['retained159']['corners']+summaries['excluded35']['corners']
