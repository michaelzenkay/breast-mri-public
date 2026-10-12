"""Small CPU checks for inference, measurements and the training pipeline."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

from breast_mri.data import prepare_case, read_manifest
from breast_mri.evaluation import evaluate
from breast_mri.inference import predict_seed, predict_volume, predict_volume_single_pass
from breast_mri.model import load_model
from breast_mri.preprocessing import compute_tumor_metrics
from breast_mri.training import train


class FirstChannel(torch.nn.Module):
    def forward(self, image):
        return image[:, :1]


class SegmentationTests(unittest.TestCase):
    def test_inference(self):
        image = np.random.default_rng(6).normal(size=(3, 7, 9, 11)).astype(np.float32)
        expected = torch.sigmoid(torch.from_numpy(image[0])).numpy()
        for predict, options in [(predict_volume, dict(patch_size=8, overlap=4)),
                                 (predict_volume_single_pass, {})]:
            with self.subTest(method=predict.__name__):
                actual = predict(image, FirstChannel(), 'cpu', **options)
                np.testing.assert_allclose(actual, expected, atol=2e-7)
        mask, _ = predict_seed(image, (0, 0, 0), FirstChannel(), 'cpu', patch_size=8)
        expected_mask = np.zeros(image.shape[1:], np.uint8)
        expected_mask[:4, :4, :4] = image[0, :4, :4, :4] > 0
        np.testing.assert_array_equal(mask, expected_mask)

    def test_physical_volume(self):
        mask = np.zeros((8, 8, 8), np.uint8)
        mask[2:6, 2:6, 2:6] = 1
        metrics = compute_tumor_metrics(mask, (1., 2., 3.))
        self.assertEqual(metrics['volume_mm3'], 64 * 6)

    def test_training_and_reload(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(3)
            label = np.zeros((16, 16, 16), np.float32)
            label[5:11, 5:11, 5:11] = 1
            records = []
            for case, split in [('a', 'train'), ('b', 'val')]:
                channels = []
                for phase in range(3):
                    path = root / f'{case}_{phase}.nii.gz'
                    image = rng.normal(size=label.shape).astype(np.float32) + label
                    nib.save(nib.Nifti1Image(image, np.eye(4)), path)
                    channels.append(path.name)
                target = root / f'{case}_label.nii.gz'
                nib.save(nib.Nifti1Image(label, np.eye(4)), target)
                records.append(dict(id=case, patient_id=case, split=split,
                                    channels=channels, label=target.name))
            manifest = root / 'cases.json'
            manifest.write_text(json.dumps(records))
            config = root / 'config.json'
            config.write_text(json.dumps(dict(manifest=manifest.name, epochs=1, batch_size=1,
                                             learning_rate=.001, patch_size=8, augment=False)))
            with contextlib.redirect_stdout(io.StringIO()):
                train(config, root / 'run')
            checkpoint_path = root / 'run/best.pt'
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            image, target = prepare_case(read_manifest(manifest)[1], patch_size=8)
            result = evaluate(load_model(checkpoint_path), [(image[None], target[None], ['b'])])
            self.assertTrue(np.isfinite(result['mean_case_dice']))
            self.assertEqual(result, checkpoint['validation'])
