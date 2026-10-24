"""Generated wrapper wiring only; no source tensors or pretrained weights."""
from pathlib import Path
import importlib.util
import json
import torch

H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('generated_c8_operator', H/'frozen_rgb_operator.py')
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)


class FakeFeature(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, rgb, valid):
        self.batch_sizes.append(len(rgb))
        x = torch.zeros(len(rgb), 1026, device=rgb.device)
        x[:, 0] = rgb[:, 0, 0, 0].float()
        return x


class FakeHead(torch.nn.Module):
    def forward(self, features):
        return features[:, 0] - 10


def main():
    torch.set_num_threads(1)
    p3 = torch.zeros(35, 65, 9, 9);p3[:, 64, 4, 4] = 1
    rgb = torch.full((35, 3, 96, 96), 12, dtype=torch.uint8)
    valid = torch.ones(35, 1, 96, 96, dtype=torch.bool)
    p3[0, 64, 4, 4] = 0;valid[1, 0, 47, 48] = False
    feature = FakeFeature();model = op.FrozenRGBOperator(feature, FakeHead()).train(True)
    result = model(p3, rgb, valid)
    assert feature.batch_sizes == [32, 3]
    assert result['query_available'].sum() == 33
    assert torch.equal(result['query_logit'][:2], torch.zeros(2))
    assert torch.equal(result['query_logit'][2:], torch.full((33,), 2.))
    p3[:, :64] = torch.randn_like(p3[:, :64])
    other = model(p3, rgb, valid)
    assert all(torch.equal(result[k], other[k]) for k in result)
    assert not model.training and not any(m.training for m in model.modules())
    empty = model(p3[:0], rgb[:0], valid[:0]);assert empty['query_logit'].shape == (0,)
    receipt = dict(completed=True, PASS=True, operator_code=op.binding(H/'frozen_rgb_operator.py'),
                   test_code=op.binding(__file__), generated_batch32_and_last3=True,
                   availability_exact_and_unavailable_safezero=True, P3_feature_values_no_effect=True,
                   eval_only=True, actual_source_or_private_reads=0, pretrained_weights_loaded=0,
                   actual_backbone_or_source_head_forward=0)
    with (H/'GENERATED_OPERATOR_TESTS.json').open('x') as f:
        json.dump(receipt, f, indent=2);f.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
