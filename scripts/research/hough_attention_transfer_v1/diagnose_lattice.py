"""GT-only lattice readout diagnostic; no image inference or model selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import core as C
from run import Data


def main(out):
    torch.set_num_threads(4)
    dh = C.load_dh()
    data = Data(out, dh)
    cfg = json.loads((out/'CONFIG.json').read_text())
    report = {'purpose': 'Test whether the existing theta/rho lattice can encode GT lines to useful precision.',
              'method': 'For each GT role, choose the lattice hypothesis nearest under the training line-distance metric, then measure original-image errors.',
              'scope': 'Oracle uses GT; not deployable. This is an achievable lattice error, not the optimum under pixel distance and not a test of feature sufficiency.',
              'populations': {}}
    for pop in ('synth_test', 'cross_v4', 'real_dev'):
        values = {'angle_deg': [], 'distance_px': [], 'distance_diagonal': [],
                  'feature_cell_width_px': [], 'feature_cell_height_px': [],
                  'height_edges_feature_cells': []}
        roles = []
        for i in data.populations[pop]:
            rec = data.records[i]
            with torch.no_grad():
                _,_,sq = dh.line_distance(data.gt, data.gr, data.theta[i], data.rho[i])
                best = sq.argmin(0)
                lines = C.line_pixels(dh, data.gt[best], data.gr[best], rec['width'], rec['height'], cfg['pad_px'])
            angle,distance = C.pixel_errors(lines, rec['gt_points'])
            live = data.support[i].cpu().numpy()
            values['angle_deg'].extend(angle[live].tolist())
            values['distance_px'].extend(distance[live].tolist())
            values['distance_diagonal'].extend((distance[live]/np.hypot(rec['width'],rec['height'])).tolist())
            sx=(rec['width']+2*cfg['pad_px'])/50
            sy=(rec['height']+2*cfg['pad_px'])/50
            values['feature_cell_width_px'].append(sx)
            values['feature_cell_height_px'].append(sy)
            gt=np.asarray(rec['gt_points'])/[sx,sy]
            for role in (1,3,5,7):
                if live[role]:
                    a,b=C.EDGES[role]
                    values['height_edges_feature_cells'].append(float(np.linalg.norm(gt[a]-gt[b])))
            for r in np.flatnonzero(live):
                roles.append({'id':rec['id'],'role':int(r),'angle_deg':float(angle[r]),'distance_px':float(distance[r])})
        report['populations'][pop] = {
            'n_frames':len(data.populations[pop]),'n_roles':len(roles),
            'statistics':{k:{'median':float(np.median(v)),'p90':float(np.percentile(v,90)),'mean':float(np.mean(v))} for k,v in values.items()},
            'height_edge_fraction_below_2_feature_cells':float(np.mean(np.asarray(values['height_edges_feature_cells'])<2)),
        }
        print(pop, report['populations'][pop], flush=True)
    C.write_json(out/'diagnosis/LATTICE_ORACLE.json', report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    main(p.parse_args().run_dir)
