import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from homework_datasets import CSVDataset
from homework_experiments import add_features
from homework_model_modification import train_linear_regression


class HomeworkTests(unittest.TestCase):
    def test_csv_preprocessing(self):
        """Проверяет обработку числовых, категориальных и бинарных признаков."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "example.csv"
            path.write_text(
                "size,city,has_garage,price\n"
                "50,Moscow,0,100\n"
                "100,Kazan,1,200\n"
                "125,Perm,1,250\n",
                encoding="utf-8",
            )

            train = CSVDataset(path, "price", indices=[0, 1])
            test = CSVDataset(
                path, "price", indices=[2], preprocessor=train.preprocessor
            )

        self.assertEqual(len(train), 2)
        self.assertEqual(tuple(train.features.shape), (2, 4))
        self.assertEqual(train.features.dtype, torch.float32)
        self.assertEqual(train[0][1].item(), 100.0)
        torch.testing.assert_close(
            test[0][0], torch.tensor([2.0, 0.0, 0.0, 1.0])
        )

    def test_feature_engineering(self):
        """Проверяет значения новых признаков."""
        features = torch.tensor([[1.0, 2.0, 3.0]])

        expected = {
            "Baseline": [[1.0, 2.0, 3.0]],
            "Polynomial": [[1.0, 2.0, 3.0, 1.0, 4.0, 9.0]],
            "Interactions": [[1.0, 2.0, 3.0, 2.0, 3.0, 6.0]],
            "Statistics": [[1.0, 2.0, 3.0, 2.0, 2.0 / 3.0]],
        }

        for variant, values in expected.items():
            with self.subTest(variant=variant):
                torch.testing.assert_close(
                    add_features(features, variant), torch.tensor(values)
                )

    def test_early_stopping_restores_best_state(self):
        """Проверяет остановку обучения и восстановление лучших параметров."""
        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()

        x = torch.zeros(1, 1)
        loader = DataLoader(TensorDataset(x, torch.ones_like(x)), batch_size=1)

        with self.assertLogs(level="INFO") as logs:
            best_loss = train_linear_regression(
                model,
                loader,
                x,
                torch.zeros_like(x),
                l1=0.0,
                l2=0.0,
                patience=2,
                max_epochs=10,
            )

        self.assertIn("Early stopping at epoch 3", "\n".join(logs.output))
        self.assertAlmostEqual(best_loss, 0.01, places=6)
        self.assertAlmostEqual(model.bias.item(), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()