"""Evaluation-only metadata adapter; original checkpoints and locked code unchanged.

PoseTrainer.set_model_attributes normally sets top-level kpt_shape. The custom
fixed-update loop saved the correct 9x3 head and YAML but omitted this descriptor.
This adapter supplies that same descriptor before AutoBackend construction.
"""
from pathlib import Path
import screen as S


def attach_shape(model):
    before = S.tensor_sha(model.state_dict())
    shape = list(model.model[-1].kpt_shape)
    assert shape == [9, 3] and list(model.yaml['kpt_shape']) == shape
    old = getattr(model, 'kpt_shape', None)
    assert old is None or list(old) == shape
    model.kpt_shape = shape
    assert S.tensor_sha(model.state_dict()) == before
    return dict(old_attribute=old, new_attribute=shape, state_sha256=before,
                tensors_changed=False, checkpoint_file_changed=False)


def main():
    S.seed_all()
    S.verify_lock()
    original_factory = S.evaluation_module
    module = original_factory()
    original_predictor = module.P.E._UltralyticsPredictor

    class ShapeCompatiblePredictor(original_predictor):
        def __init__(self, weights, device):
            super().__init__(weights, device)
            receipt = attach_shape(self.model.model)
            name = f'{Path(weights).parent.name}_{Path(weights).stem}'
            S.write(S.DOC / 'inference_metadata' / f'{name}.json', receipt)

    def compatible_factory():
        module.P.E._UltralyticsPredictor = ShapeCompatiblePredictor
        return module

    S.write(S.DOC / 'EVALUATION_COMPATIBILITY.json', dict(
        status='EVALUATION_METADATA_ONLY', original_error='AutoBackend missing kpt_shape',
        training_error=False, new_optimizer_updates=0, checkpoint_mutation=False,
        locked_training_code_unchanged=True, adapter_sha256=S.sha(Path(__file__)),
        change='Set top-level model.kpt_shape=[9,3], copied from existing head/YAML, before AutoBackend. Same operation as stock PoseTrainer.set_model_attributes.',
        inference_recipe='Canonical padding, confidence, image size, candidate selection and PnP unchanged.',
        first_attempt='Failed on first inference before any prediction cache or performance result was produced. Original error log retained.'))
    S.evaluation_module = compatible_factory
    try:
        S.evaluate()
        S.report()
    finally:
        module.P.E._UltralyticsPredictor = original_predictor
        S.evaluation_module = original_factory


if __name__ == '__main__':
    main()
