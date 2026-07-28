"""
NVIDIA from jtremblay@gmail.com
"""

# Networks
import torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.models as models


class DopeNetwork(nn.Module):
    def __init__(
        self,
        pretrained=False,
        numBeliefMap=9,
        numAffinity=16,
        stop_at_stage=6,  # number of stages to process (if less than total number of stages)
        numVec=0,  # PVNet dense vector head out channels (2*numKeypoints); 0 = head off
        numSeg=0,  # PVNet seg head out channels (mask logit); 0 = head off, 1 = BCE
        numEdgeVec=0,  # sparse edge-direction head: 2*nEdges (global unit dirs); 0 = off
        maskBeliefFusion=False,  # zero-init mask-guided final-belief residual
        cornerQuality=False,  # per-corner log localization sigma map
    ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage
        self.numVec = numVec
        self.numSeg = numSeg
        self.numEdgeVec = numEdgeVec
        self.maskBeliefFusion = bool(maskBeliefFusion)
        self.cornerQuality = bool(cornerQuality)
        self.refinement_enabled = self.maskBeliefFusion or self.cornerQuality
        if self.refinement_enabled and stop_at_stage != 6:
            raise ValueError("fixed-grid refinement heads require stop_at_stage=6")
        if self.maskBeliefFusion and numSeg <= 0:
            raise ValueError("maskBeliefFusion requires numSeg>0")
        if self.refinement_enabled and numEdgeVec > 0:
            raise ValueError("refinement heads and numEdgeVec cannot share legacy arity")

        vgg_full = models.vgg19(pretrained=False).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers
        i_layer = 23
        self.vgg.add_module(
            str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        )
        self.vgg.add_module(str(i_layer + 1), nn.ReLU(inplace=True))
        self.vgg.add_module(
            str(i_layer + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        )
        self.vgg.add_module(str(i_layer + 3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m3_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m4_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m5_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m6_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m3_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m4_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m5_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m6_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )

        # PVNet-style dense vector head (flag-gated). Shares the VGG feature
        # out1 (128ch) like the belief/affinity heads; predicts per-pixel
        # vectors to numVec/2 keypoints at the same 50x50 grid. Off by default
        # (numVec=0) so existing 2-head training is byte-identical.
        if numVec > 0:
            self.m_vec1 = DopeNetwork.create_stage(128, numVec, True)
            self.m_vec2 = DopeNetwork.create_stage(
                128 + numVec, numVec, False
            )

        # PVNet-style seg head (flag-gated). Shares the VGG feature out1 (128ch)
        # like the other heads; predicts a per-pixel mask logit at the 50x50
        # grid (numSeg=1 = BCE). Off by default (numSeg=0) so existing training
        # is byte-identical.
        if numSeg > 0:
            self.m_seg1 = DopeNetwork.create_stage(128, numSeg, True)
            self.m_seg2 = DopeNetwork.create_stage(
                128 + numSeg, numSeg, False
            )

        # Mask-guided belief refinement.  Only the final belief stage is
        # adjusted; the mask remains a soft training/inference feature, never a
        # hard gate.  The last convolution is exactly zero-initialized, so an
        # ep57 checkpoint loaded with strict=False produces an exact zero
        # residual before the first optimizer step.
        if self.maskBeliefFusion:
            self.m_mask_belief_fusion = nn.Sequential(
                nn.Conv2d(128 + numSeg, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, numBeliefMap, kernel_size=1),
            )
            nn.init.zeros_(self.m_mask_belief_fusion[-1].weight)
            nn.init.zeros_(self.m_mask_belief_fusion[-1].bias)

        # A spatial log-sigma map is sampled at each channel's decoded peak.
        # It predicts localization error in 50-grid cells and is used only to
        # calibrate PnP weights/rejection; its input is detached in forward so
        # calibration cannot move a heatmap peak to make its own task easier.
        if self.cornerQuality:
            quality_in = 128 + numBeliefMap + (numSeg if numSeg > 0 else 0)
            self.m_corner_quality = nn.Sequential(
                nn.Conv2d(quality_in, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, numBeliefMap, kernel_size=1),
            )
            nn.init.zeros_(self.m_corner_quality[-1].weight)
            nn.init.constant_(
                self.m_corner_quality[-1].bias, 0.6931471805599453)

        # Sparse edge-direction head (flag-gated; STEP7). Predicts ONE unit
        # 2D direction per structural edge for the whole image (HybridPose-style
        # edge vectors), NOT a dense per-pixel field -> no segmentation mask
        # needed. Shares the VGG feature out1 (128ch): small conv -> global
        # average pool -> FC -> numEdgeVec (=2*nEdges) raw values. The trainer
        # L2-normalizes each (dx,dy) pair to unit at loss/use time. Off by
        # default (numEdgeVec=0) so existing training is byte-identical.
        if numEdgeVec > 0:
            self.m_edge_conv = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
            self.m_edge_pool = nn.AdaptiveAvgPool2d(1)
            self.m_edge_fc = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, numEdgeVec),
            )

    def forward(self, x):
        """Runs inference on the neural network"""

        out1 = self.vgg(x)

        out1_2 = self.m1_2(out1)
        out1_1 = self.m1_1(out1)

        if self.stop_at_stage == 1:
            return [out1_2], [out1_1]

        out2 = torch.cat([out1_2, out1_1, out1], 1)
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        if self.stop_at_stage == 2:
            return [out1_2, out2_2], [out1_1, out2_1]

        out3 = torch.cat([out2_2, out2_1, out1], 1)
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2], [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out3_1, out1], 1)
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2], [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out4_1, out1], 1)
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2], [
                out1_1,
                out2_1,
                out3_1,
                out4_1,
                out5_1,
            ]

        out6 = torch.cat([out5_2, out5_1, out1], 1)
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        beliefs = [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2]
        affinities = [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]

        # Return arity rule (flag-gated; never breaks existing callers):
        #   numVec=0, numSeg=0, numEdgeVec=0 -> (beliefs, affinities)         [original]
        #   numVec>0, numSeg=0               -> (beliefs, affinities, vec)    [STEP4]
        #   numVec>0, numSeg>0               -> (beliefs, affinities, vec, seg) [STEP5]
        #   numVec=0, numSeg>0               -> (beliefs, affinities, None, seg)
        #   numEdgeVec>0 (others=0)          -> (beliefs, affinities, edge)   [STEP7]
        # vec  = [vec1, vec2] (2-stage list), or None.
        # seg  = [seg1, seg2] (2-stage list of (B,numSeg,H,W) logits).
        # edge = (B, numEdgeVec) raw FC output (caller L2-normalizes per pair).
        vec_out = None
        if self.numVec > 0:
            vec1 = self.m_vec1(out1)
            vec2 = self.m_vec2(torch.cat([vec1, out1], 1))
            vec_out = [vec1, vec2]

        seg_out = None
        if self.numSeg > 0:
            seg1 = self.m_seg1(out1)
            seg2 = self.m_seg2(torch.cat([seg1, out1], 1))
            seg_out = [seg1, seg2]

        # The zero-initialized residual makes fusion checkpoint-safe: before a
        # training update, this assignment is numerically belief6 + 0.
        if self.maskBeliefFusion:
            fusion_in = torch.cat([out1, torch.sigmoid(seg_out[-1])], 1)
            beliefs[-1] = beliefs[-1] + self.m_mask_belief_fusion(fusion_in)

        if self.refinement_enabled:
            refinement = {}
            if self.cornerQuality:
                qparts = [out1.detach(), beliefs[-1].detach()]
                if seg_out is not None:
                    qparts.append(torch.sigmoid(seg_out[-1]).detach())
                refinement["corner_log_sigma"] = self.m_corner_quality(
                    torch.cat(qparts, 1))
            # Stable new API: every refinement configuration uses one 5-tuple.
            # Legacy configurations retain their historical return arity below.
            return beliefs, affinities, vec_out, seg_out, refinement

        if seg_out is not None:
            return beliefs, affinities, vec_out, seg_out

        if vec_out is not None:
            return beliefs, affinities, vec_out

        if self.numEdgeVec > 0:
            ef = self.m_edge_conv(out1)
            ef = self.m_edge_pool(ef).flatten(1)   # (B,128)
            edge = self.m_edge_fc(ef)              # (B, numEdgeVec) raw
            return beliefs, affinities, edge

        return beliefs, affinities

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        """Create the neural network layers for a single stage."""

        model = nn.Sequential()
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        model.add_module(
            "0",
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding
            ),
        )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(
                str(i),
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                ),
            )
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(
            str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1)
        )
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(
            str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1)
        )
        i += 1

        return model


REFINEMENT_STATE_PREFIXES = (
    "m_mask_belief_fusion.",
    "m_corner_quality.",
)


def refinement_model_kwargs_from_state_dict(state_dict):
    """Infer refinement-related :class:`DopeNetwork` kwargs from a checkpoint.

    Both plain and DistributedDataParallel (``module.``) keys are accepted.
    This is intentionally small and deterministic so evaluation scripts do not
    need to infer architecture from a training directory name.
    """
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    keys = tuple(state.keys())

    num_seg = 0
    seg_convs = []
    for key, value in state.items():
        if key.startswith("m_seg1.") and key.endswith(".weight") \
                and getattr(value, "ndim", 0) == 4:
            try:
                layer_index = int(key.split(".")[1])
            except (ValueError, IndexError):
                continue
            seg_convs.append((layer_index, int(value.shape[0])))
    if seg_convs:
        num_seg = max(seg_convs)[1]

    return {
        "numSeg": num_seg,
        "maskBeliefFusion": any(
            key.startswith("m_mask_belief_fusion.") for key in keys),
        "cornerQuality": any(
            key.startswith("m_corner_quality.") for key in keys),
    }
