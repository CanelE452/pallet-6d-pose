"""Matched task readouts: global Direct Hough versus feature-aggregating DHT.

This is an architecture comparison, not a one-factor ablation or a reproduction
of the published ResNet-FPN DHT network. Both consume identical frozen features.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from dht import SparseDHT, hough_pad2d


def lattice_features(dht):
    angle, rho = torch.meshgrid(dht.theta_radians, dht.rho_values, indexing='ij')
    radius = rho / dht.rho_max
    return torch.stack([(2*angle).cos(), (2*angle).sin(), radius*angle.cos(),
                        radius*angle.sin(), radius.square()], -1).reshape(-1, 5)


class DirectSide(nn.Module):
    def __init__(self, dh_module, lattice, seed):
        super().__init__()
        dh_module.CAP.SEED = seed
        self.base = dh_module.DirectHoughModel()
        dim = self.base.encoder.queries.embedding_dim
        self.base.encoder.queries = nn.Embedding(8, dim)
        self.register_buffer('hypotheses', lattice_features(lattice), persistent=False)

    def forward(self, features):
        return self.base(features, self.hypotheses)


class HoughConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, 3, padding=0)

    def forward(self, x):
        return self.conv(hough_pad2d(x, 1))


class DeepHoughSide(nn.Module):
    def __init__(self, lattice, seed, channels=16):
        super().__init__()
        torch.manual_seed(seed)
        axis = torch.linspace(-1, 1, 50)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('xy', torch.stack([xx, yy])[None], persistent=False)
        self.image_features = nn.Sequential(
            nn.Conv2d(130, 32, 1), nn.ReLU(inplace=False),
            nn.Conv2d(32, channels, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=False))
        self.dht = lattice
        # Continuous unoriented-line coordinates, invariant at theta/rho seam.
        coordinates = lattice_features(lattice).T.reshape(5, lattice.theta_bins, lattice.rho_bins)
        mass = (lattice.mass / lattice.mass.max())[None]
        self.register_buffer('geometry', torch.cat([coordinates, mass])[None], persistent=False)
        self.hough_features = nn.Sequential(HoughConv(channels+6, 32), nn.ReLU(inplace=False),
                                            HoughConv(32, 16), nn.ReLU(inplace=False),
                                            nn.Conv2d(16, 8, 1))

    def forward(self, features):
        xy = self.xy.expand(len(features), -1, -1, -1)
        local = self.image_features(torch.cat([features, xy], 1))
        accumulated = self.dht(local)
        geo = self.geometry.expand(len(features), -1, -1, -1)
        return self.hough_features(torch.cat([accumulated, geo], 1)).flatten(2)


def target_distribution(theta, rho, lattice, angle_sigma=1., rho_sigma=.5):
    """Soft CE target; respect (theta+180,-rho) identity at the angular seam."""
    gt = theta[..., None, None]
    a = lattice.theta_degrees[None, None, :, None]
    r = lattice.rho_values[None, None, None, :]
    delta = a - gt
    signed_angle = (delta+90) % 180 - 90
    polarity = torch.where(torch.cos(torch.deg2rad(delta)) >= 0, 1., -1.)
    offset = r*polarity-rho[..., None, None]
    target = torch.exp(-.5*((signed_angle/angle_sigma).square()+(offset/rho_sigma).square()))
    target *= lattice.valid
    target = target.flatten(2)
    return target/target.sum(-1, keepdim=True).clamp_min(1e-20)


def line_loss(scores, target, supported, lattice):
    logp = F.log_softmax(scores.masked_fill(~lattice.valid.flatten()[None,None], -1e9), -1)
    loss = -(target*logp).sum(-1)
    return (loss*supported.float()).sum()/supported.sum().clamp_min(1)


def decode(scores, lattice):
    scores = scores.masked_fill(~lattice.valid.flatten()[None,None], -1e9)
    best = scores.argmax(-1)
    return lattice.theta_degrees[best//lattice.rho_bins], lattice.rho_values[best%lattice.rho_bins]
