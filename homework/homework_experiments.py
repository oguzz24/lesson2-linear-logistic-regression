import csv
import logging
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from homework_datasets import CSVDataset


OPTIMIZERS = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "RMSprop": torch.optim.RMSprop,
}

FEATURE_LABELS = {
    "Baseline": "Исходные",
    "Polynomial": "Полиномиальные",
    "Interactions": "Взаимодействия",
    "Statistics": "Статистические",
}


def load_experiment_data(csv_path):
    """Разделяет данные на обучающую, валидационную и тестовую выборки."""
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        labels = np.array(
            [float(row["target"]) for row in csv.DictReader(csv_file)]
        )

    train_val_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.25,
        random_state=42,
        stratify=labels[train_val_indices],
    )

    train = CSVDataset(csv_path, "target", indices=train_indices)
    validation = CSVDataset(
        csv_path,
        "target",
        indices=val_indices,
        preprocessor=train.preprocessor,
    )
    test = CSVDataset(
        csv_path,
        "target",
        indices=test_indices,
        preprocessor=train.preprocessor,
    )
    return train, validation, test


def train_binary_model(features, targets, optimizer_class, lr, batch_size, epochs=60):
    """Обучает бинарную логистическую регрессию с заданными параметрами."""
    torch.manual_seed(42)

    model = nn.Linear(features.shape[1], 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizer_class(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    model.train()
    for _ in range(epochs):
        for batch_features, batch_targets in loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

    return model


def evaluate_binary_model(model, features, targets):
    """Вычисляет BCE, accuracy и ROC-AUC на отложенной выборке."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets
        ).item()
        probabilities = logits.sigmoid().flatten().numpy()

    labels = targets.flatten().numpy()
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "loss": loss,
        "accuracy": accuracy_score(labels, predictions),
        "roc_auc": roc_auc_score(labels, probabilities),
    }


def save_results(path, results):
    """Сохраняет результаты эксперимента в CSV."""
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def run_hyperparameter_search(train, validation, test, output_path):
    """Сравнивает 27 конфигураций и оценивает лучшую на тестовой выборке."""
    learning_rates = (0.001, 0.01, 0.05)
    batch_sizes = (16, 32, 64)
    results = []
    best_model = None
    best_loss = float("inf")

    for optimizer_name, lr, batch_size in product(
        OPTIMIZERS, learning_rates, batch_sizes
    ):
        model = train_binary_model(
            train.features,
            train.targets,
            OPTIMIZERS[optimizer_name],
            lr,
            batch_size,
        )
        metrics = evaluate_binary_model(
            model, validation.features, validation.targets
        )

        results.append(
            {
                "optimizer": optimizer_name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "validation_loss": metrics["loss"],
                "validation_accuracy": metrics["accuracy"],
                "validation_roc_auc": metrics["roc_auc"],
            }
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_model = model

        logging.info(
            "%-7s | Скорость=%5.3f | Батч=%2d | Потери на валидации=%.4f",
            optimizer_name,
            lr,
            batch_size,
            metrics["loss"],
        )

    results.sort(key=lambda result: result["validation_loss"])
    save_results(output_path, results)

    best = results[0]
    test_metrics = evaluate_binary_model(best_model, test.features, test.targets)

    print("\n3.1 Исследование гиперпараметров")
    print(f"Лучший оптимизатор: {best['optimizer']}")
    print(f"Скорость обучения: {best['learning_rate']}")
    print(f"Размер батча: {best['batch_size']}")
    print(f"Потери на валидации: {best['validation_loss']:.4f}")
    print(f"Доля верных ответов на тестовой выборке: {test_metrics['accuracy']:.4f}")
    print(f"ROC-AUC на тестовой выборке: {test_metrics['roc_auc']:.4f}")
    print(f"Результаты сохранены: {output_path}")

    return best


def add_features(features, variant):
    """Добавляет квадраты, попарные произведения или статистики строки."""
    if variant == "Baseline":
        return features

    selected = features[:, :5]

    if variant == "Polynomial":
        extra = selected.square()
    elif variant == "Interactions":
        left, right = torch.triu_indices(
            selected.shape[1], selected.shape[1], offset=1
        )
        extra = selected[:, left] * selected[:, right]
    elif variant == "Statistics":
        extra = torch.cat(
            [
                features.mean(dim=1, keepdim=True),
                features.var(dim=1, unbiased=False, keepdim=True),
            ],
            dim=1,
        )
    else:
        raise ValueError(f"Unknown feature variant: {variant}")

    return torch.cat([features, extra], dim=1)


def run_feature_engineering(train, validation, test, best, data_dir, plots_dir):
    """Сравнивает наборы признаков при одинаковых настройках обучения."""
    original_count = train.features.shape[1]
    results = []

    print("\n3.2 Feature Engineering")
    print(
        f"{'Набор признаков':<16} {'Число':>8} {'BCE (валидация)':>16} "
        f"{'Доля верных (тест)':>18} {'ROC-AUC (тест)':>15}"
    )

    for variant in ("Baseline", "Polynomial", "Interactions", "Statistics"):
        train_x = add_features(train.features, variant)
        val_x = add_features(validation.features, variant)
        test_x = add_features(test.features, variant)

        if variant != "Baseline":
            # Масштабируем новые столбцы по параметрам обучающей выборки.
            scaler = StandardScaler()
            scaler.fit(train_x[:, original_count:].numpy())

            for features in (train_x, val_x, test_x):
                scaled = scaler.transform(features[:, original_count:].numpy())
                features[:, original_count:] = torch.as_tensor(
                    scaled, dtype=features.dtype
                )

        model = train_binary_model(
            train_x,
            train.targets,
            OPTIMIZERS[best["optimizer"]],
            best["learning_rate"],
            best["batch_size"],
        )
        val_metrics = evaluate_binary_model(model, val_x, validation.targets)
        test_metrics = evaluate_binary_model(model, test_x, test.targets)

        results.append(
            {
                "variant": variant,
                "feature_count": train_x.shape[1],
                "validation_loss": val_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
        )

        print(
            f"{FEATURE_LABELS[variant]:<16} {train_x.shape[1]:>8} "
            f"{val_metrics['loss']:>16.4f} "
            f"{test_metrics['accuracy']:>18.4f} "
            f"{test_metrics['roc_auc']:>15.4f}"
        )

    results_path = data_dir / "feature_engineering_results.csv"
    save_results(results_path, results)

    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / "feature_engineering.png"

    figure, axis = plt.subplots(figsize=(8, 4))
    bars = axis.bar(
        [FEATURE_LABELS[result["variant"]] for result in results],
        [result["validation_loss"] for result in results],
    )
    axis.bar_label(bars, fmt="%.4f", padding=3)
    axis.set_xlabel("Набор признаков")
    axis.set_ylabel("Потери на валидации (BCE)")
    axis.set_title("Сравнение наборов признаков — Breast Cancer")
    axis.margins(y=0.15)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(f"Результаты сохранены: {results_path}")
    print(f"График сохранён: {plot_path}")


def main():
    """Запускает задания 3.1 и 3.2 на данных Breast Cancer."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    homework_dir = Path(__file__).resolve().parent
    data_dir = homework_dir / "data"
    plots_dir = homework_dir / "plots"

    train, validation, test = load_experiment_data(data_dir / "breast_cancer.csv")
    best = run_hyperparameter_search(
        train, validation, test, data_dir / "hyperparameter_results.csv"
    )
    run_feature_engineering(train, validation, test, best, data_dir, plots_dir)


if __name__ == "__main__":
    main()
