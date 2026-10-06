"""Draw PCK10 from the published CSV. Not a new experiment or a full error CDF."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
with (root / 'evidence/github_tables/tab_occlusion_results.csv').open(encoding='utf-8', newline='') as f:
    records = list(csv.DictReader(f))
groups = ['clean', 'moderate', 'severe', 'unclassified']
methods = ['R0', 'P', 'N2 dimensions', 'N3 dimensions + symmetry']
labels = ['No external occlusion\n29 images', 'Moderate\n20 images',
          'Severe\n79 images', 'Unclassified\n191 images']
fig, ax = plt.subplots(figsize=(8.0, 3.5))
x = np.arange(len(groups)); width = .19
for i, method in enumerate(methods):
    values = [float(next(r['PCK10 %'] for r in records if r['Group']==group and r['Method']==method))
              for group in groups]
    ax.bar(x+(i-1.5)*width, values, width, label=['Base','P','N2','N3'][i])
ax.set_xticks(x, labels)
ax.set_ylabel('Correct corners within 10 pixels (%)')
ax.set_ylim(0,100)
ax.legend(ncol=4, loc='upper center')
ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(root/'figures/occlusion_pck10.pdf', bbox_inches='tight')
fig.savefig(root/'figures/occlusion_pck10.png', dpi=300, bbox_inches='tight')
plt.close(fig)
