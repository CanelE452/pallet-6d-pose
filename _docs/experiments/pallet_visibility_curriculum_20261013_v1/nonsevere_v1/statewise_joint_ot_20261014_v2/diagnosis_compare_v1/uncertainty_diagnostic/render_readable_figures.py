"""Render sealed covariance values with a readable scientific plotting font."""
from pathlib import Path
import importlib.util
import json
import sys
P=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[7]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C

seal=C.read(P/'DIAGNOSTIC_SEAL.json')
assert C.sha(P/'UNCERTAINTY_DIAGNOSTIC.json')==seal['results_sha256']
assert C.sha(P/'PROTOCOL.json')==seal['protocol_sha256']
module_spec=importlib.util.spec_from_file_location('sealed_uncertainty_renderer',P/'diagnose_uncertainty.py')
module=importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
out=P/'readable_figures_v2'
out.mkdir(exist_ok=False)
module.OUT=out
plt.rcParams['font.family']='DejaVu Sans'
plt.rcParams['mathtext.fontset']='dejavusans'
plt.rcParams['mathtext.default']='regular'
values=C.read(P/'UNCERTAINTY_DIAGNOSTIC.json')
module.curves(values['points'],values['frames'])
original=(P/'README_KO.md').read_text()
(P/'README_LATEST_KO.md').write_text(original.replace('(COVERAGE_CURVE.png)','(readable_figures_v2/COVERAGE_CURVE.png)').replace('(DIRECT_ANNOTATION_VS_REFERENCE_MAH.png)','(readable_figures_v2/DIRECT_ANNOTATION_VS_REFERENCE_MAH.png)')+'\n원본 숫자·진단 봉인을 보존하고 과학 그림의 폰트만 별도 위치에서 재렌더했습니다.\n')
(out/'index.html').write_text("<!doctype html><meta charset='utf-8'><title>Readable frozen covariance diagnostics</title><style>body{font:16px sans-serif;max-width:1250px;margin:25px auto}img{width:100%}</style><h1>Frozen covariance · postseal cause sensitivity</h1><p>Same sealed numeric values. No new inference, fitting, cap selection, or coverage guarantee.</p><a href='../README_LATEST_KO.md'>한국어 보고서</a><img src='COVERAGE_CURVE.png'><img src='DIRECT_ANNOTATION_VS_REFERENCE_MAH.png'>\n")
C.write(out/'FIGURE_PROVENANCE.json',dict(schema='SEALED_UNCERTAINTY_READABLE_FIGURES_V2',numeric_results_sha256=C.sha(P/'UNCERTAINTY_DIAGNOSTIC.json'),diagnostic_seal_sha256=C.sha(P/'DIAGNOSTIC_SEAL.json'),original_renderer_sha256=C.sha(P/'diagnose_uncertainty.py'),font_only_renderer_source_sha256=C.sha(__file__),numeric_values_unchanged=True,new_PnP_fits=0,model_or_current_prediction_changes=False,files={file.name:C.sha(file) for file in out.iterdir() if file.is_file()}))
assert C.sha(P/'UNCERTAINTY_DIAGNOSTIC.json')==seal['results_sha256']
print(str(out))
