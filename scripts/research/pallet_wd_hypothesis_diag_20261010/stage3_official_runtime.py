"""Preserved zero-forward adapter correction for the official Small loader.

The pinned upstream hubconf uses strict=False. Its sole absent mask_token is
zero-initialized and read only when masks are supplied; dense RGB inference
calls encoder(input) without masks. No model, weight, inference configuration,
threshold, input, coordinate, or existing source file is changed.
"""
import argparse
from pathlib import Path
from . import common as C
from . import depth as D
from . import stage3 as E

MISSING = ['depth_model.encoder.mask_token']


class OfficialSmallRunner(D.MetricDepthRunner):
    def __init__(self, private_root):
        module, self.provenance = D.prepare(private_root)
        import torch
        self.torch = torch
        assert torch.cuda.is_available()
        self.model = module.metric3d_vit_small(pretrain=False)
        checkpoint = torch.load(Path(private_root)/D.WEIGHT_NAME, map_location='cpu')
        loaded = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        assert loaded.missing_keys == MISSING and not loaded.unexpected_keys
        token = self.model.depth_model.encoder.mask_token
        assert torch.count_nonzero(token).item() == 0
        self.model.cuda().eval()
        self.provenance.update(models_constructed=1, device=str(torch.cuda.get_device_name(0)),
            checkpoint_missing_keys=loaded.missing_keys, checkpoint_unexpected_keys=[],
            official_loader_strict=False, missing_mask_token_unused_for_unmasked_RGB=True,
            runtime_adapter_source_sha256=C.sha(__file__),
            runtime_amendment_sha256=C.sha(C.DOC/'STAGE3_RUNTIME_AMENDMENT.json'))
        self.forwards = 0


def run(private_dir):
    private = Path(private_dir).resolve()
    assert not (C.DOC/'DEPTH_GATE.json').exists()
    amendment = C.DOC/'STAGE3_RUNTIME_AMENDMENT.json'
    assert not amendment.exists(), 'Preserve completed amendment and execution evidence'
    E._verify_sources()
    diagnostic = C.read(private/'metric3d/CHECKPOINT_KEY_DIAGNOSTIC.json')
    assert diagnostic['missing_keys'] == MISSING
    assert not diagnostic['unexpected_keys'] and not diagnostic['missing_buffers']
    assert diagnostic['missing_parameters'] == MISSING and diagnostic['model_forwards'] == 0
    cache = private/'depth_accuracy_cache'
    assert sorted(p.name for p in cache.iterdir()) == ['DEPTH_INFERENCE_LOCK.json']
    old_lock_sha = C.sha(cache/'DEPTH_INFERENCE_LOCK.json')
    failed = private/'depth_accuracy_attempt_01'
    assert not failed.exists()
    cache.rename(failed)
    C.write(amendment, dict(status='LOCKED_BEFORE_FIRST_DEPTH_FORWARD',
        reason='Custom zero-missing-key guard contradicted pinned upstream strict=False loader',
        sole_missing_key=MISSING[0],unexpected_keys=[],
        official_evidence={'hubconf.py':'metric3d_vit_small lines95-97 strict=False',
            'mono/model/model_pipelines/dense_pipeline.py':'line14 self.encoder(input) without masks',
            'mono/model/backbones/ViT_DINO.py':'mask_token used only if masks is not None'},
        unused_parameter_initialization='official zero initialization preserved',
        previous_source_lock_path='SOURCE_LOCK_STAGE3.json',
        previous_source_lock_sha256=C.sha(C.DOC/'SOURCE_LOCK_STAGE3.json'),
        preserved_failed_private_lock_sha256=old_lock_sha,
        diagnostic=diagnostic, actual_prior_model_constructions=2,
        actual_prior_model_forwards=0, checkpoint_or_model_change=False,
        configuration_or_threshold_change=False, GT_accuracy_read_before_amendment=False,
        runtime_binding=C.binding(Path(__file__),C.ROOT)))
    original_write, original_runner = C.write, D.MetricDepthRunner
    def preserve_source_lock(path,value):
        if Path(path).resolve() == C.DOC/'SOURCE_LOCK_STAGE3.json':
            assert C.digest(C.read(path)) == C.digest(value)
            return
        return original_write(path,value)
    C.write, D.MetricDepthRunner = preserve_source_lock, OfficialSmallRunner
    try:
        result = E.run(private)
    finally:
        C.write, D.MetricDepthRunner = original_write, original_runner
    execution = C.read(C.DOC/'STAGE3_EXECUTION.json')
    C.write(C.DOC/'STAGE3_RECOVERY_EXECUTION.json',dict(status='COMPLETE',
        actual_model_constructions_total=execution['actual_model_constructions']+2,
        actual_depth_forwards_total=execution['actual_depth_forwards'],
        failed_guard_model_constructions=1,diagnostic_model_constructions=1,
        prior_attempt_forwards=0, training_updates=0,
        scientific_execution_path='STAGE3_EXECUTION.json',
        runtime_amendment_path=amendment.name, runtime_source_sha256=C.sha(__file__),
        existing_source_lock_preserved=True))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir',type=Path,required=True)
    args=parser.parse_args();run(args.private_dir)
