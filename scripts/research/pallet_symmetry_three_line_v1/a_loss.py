"""Safe stock identity path; EQUIV intentionally guarded pending objective decision."""
import torch
from ultralytics.utils.loss import E2ELoss,PoseLoss26

def stable_target_order(batch):
    """Stock's offset mapping assumes sorted batch_idx; preserve within-image GT order."""
    out=dict(batch);order=torch.argsort(batch['batch_idx'].flatten(),stable=True)
    for key in ['batch_idx','cls','bboxes','keypoints']:
        out[key]=batch[key][order]
    return out

class GenericSymmetryPoseLoss:
    def __init__(self,model,equivalent=False):
        self.stock=E2ELoss(model,PoseLoss26);self.equivalent=equivalent
    def __call__(self,predictions,batch):
        if self.equivalent:
            raise RuntimeError('A_EQUIV not authorized until stock batch-global RLE clamp versus independent object-min is resolved. No approximate branch selector is substituted.')
        return self.stock(predictions,stable_target_order(batch))
