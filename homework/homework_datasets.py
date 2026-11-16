import csv
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from homework_model_modification import train_linear_regression


class CSVDataset(Dataset):
    """Читает CSV и возвращает пары из признаков и целевых значений."""

    def __init__(self, csv_path, target_column, *, indices=None, preprocessor=None):
        with open(csv_path, encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            columns = reader.fieldnames or []
            if target_column not in columns:
                raise ValueError(f"Target column not found: {target_column}")

            feature_columns = [name for name in columns if name != target_column]
            rows = list(reader)

        if indices is not None:
            rows = [rows[index] for index in indices]

        if not rows or not feature_columns:
            raise ValueError("The dataset must contain rows and feature columns.")

        features = np.array(
            [[row[name] for name in feature_columns] for row in rows],
            dtype=object,
        )

        if preprocessor is None:
            preprocessor = self._make_preprocessor(features)
            features = preprocessor.fit_transform(features)
        else:
            features = preprocessor.transform(features)

        self.preprocessor = preprocessor
        self.features = torch.tensor(features.astype(np.float32))
        self.targets = torch.tensor(
            [[float(row[target_column])] for row in rows],
            dtype=torch.float32,
        )

    @staticmethod
    def _make_preprocessor(features):
        """Определяет типы столбцов по обучающим данным."""
        numeric_columns = []
        categorical_columns = []
        binary_columns = []

        for index, column in enumerate(features.T):
            try:
                values = {float(value) for value in column}
            except (TypeError, ValueError):
                categorical_columns.append(index)
            else:
                if values <= {0.0, 1.0}:
                    binary_columns.append(index)
                else:
                    numeric_columns.append(index)

        return ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric_columns),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_columns,
                ),
                ("binary", "passthrough", binary_columns),
            ]
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def main():
    """Проверяет обработку числовых, категориальных и бинарных признаков."""
    with TemporaryDirectory() as directory:
        csv_path = Path(directory) / "example.csv"
        csv_path.write_text(
            "size,city,has_garage,price\n"
            "50,Moscow,0,100\n"
            "75,Kazan,1,150\n"
            "100,Moscow,1,200\n",
            encoding="utf-8",
        )

        dataset = CSVDataset(csv_path, target_column="price")
        features, target = dataset[0]

        print(f"Rows: {len(dataset)}")
        print(f"Feature shape: {tuple(dataset.features.shape)}")
        print(f"First example: {features.tolist()}")
        print(f"First target: {target.item()}")


def save_dataset_csv(dataset, path):
    """Сохраняет признаки и целевые значения в CSV."""
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([*dataset.feature_names, "target"])
        writer.writerows(
            [*features, target]
            for features, target in zip(dataset.data, dataset.target)
        )


def run_csv_regression(path, row_count):
    """Обучает линейную регрессию и вычисляет MSE на тестовой выборке."""
    train_val_indices, test_indices = train_test_split(
        np.arange(row_count), test_size=0.2, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.25, random_state=42
    )

    train = CSVDataset(path, "target", indices=train_indices)
    validation = CSVDataset(
        path, "target", indices=val_indices, preprocessor=train.preprocessor
    )
    test = CSVDataset(
        path, "target", indices=test_indices, preprocessor=train.preprocessor
    )

    loader = DataLoader(train, batch_size=32, shuffle=True)
    model = nn.Linear(train.features.shape[1], 1)
    best_mse = train_linear_regression(
        model, loader, validation.features, validation.targets
    )

    model.eval()
    with torch.no_grad():
        test_mse = nn.functional.mse_loss(
            model(test.features), test.targets
        ).item()
    torch.save(model.state_dict(), path.parent.parent / "models" / "diabetes_linear.pth")
    print("\n2.2 Diabetes — linear regression")
    print(f"Best validation MSE: {best_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")


def run_csv_classification(path, labels):
    """Обучает бинарную логистическую регрессию и оценивает её на тесте."""
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train = CSVDataset(path, "target", indices=train_indices)
    test = CSVDataset(
        path, "target", indices=test_indices, preprocessor=train.preprocessor
    )

    loader = DataLoader(train, batch_size=32, shuffle=True)
    model = nn.Linear(train.features.shape[1], 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    model.train()
    for epoch in range(1, 61):
        total_loss = 0.0

        for features, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        if epoch % 15 == 0:
            logging.info("Epoch %d | Binary loss: %.4f", epoch, total_loss / len(train))

    model.eval()
    with torch.no_grad():
        probabilities = model(test.features).sigmoid().flatten().numpy()

    targets = test.targets.flatten().numpy()
    predictions = (probabilities >= 0.5).astype(int)
    torch.save(model.state_dict(), path.parent.parent / "models" / "breast_cancer_logistic.pth")
    print("\n2.2 Breast Cancer — binary logistic regression")
    print(f"Test accuracy: {accuracy_score(targets, predictions):.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(targets, probabilities):.4f}")


def run_csv_experiments():
    """Подготавливает два CSV-файла и запускает обучение моделей."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(42)

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir.parent / "models").mkdir(exist_ok=True)

    diabetes = load_diabetes()
    cancer = load_breast_cancer()
    diabetes_path = data_dir / "diabetes.csv"
    cancer_path = data_dir / "breast_cancer.csv"

    save_dataset_csv(diabetes, diabetes_path)
    save_dataset_csv(cancer, cancer_path)

    run_csv_regression(diabetes_path, len(diabetes.target))
    run_csv_classification(cancer_path, cancer.target)


if __name__ == "__main__":
    main()
    run_csv_experiments()