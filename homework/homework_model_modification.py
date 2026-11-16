import logging
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_linear_regression(
    model,
    train_loader,
    x_val,
    y_val,
    *,
    l1=0.001,
    l2=0.001,
    patience=10,
    max_epochs=100,
):
    """Обучает модель с L1/L2 и восстанавливает лучшие веса."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)

            # Добавляем штрафы к MSE только для весов модели.
            weights = model.weight
            mse = criterion(predictions, y_batch)
            l1_penalty = weights.abs().sum()
            l2_penalty = weights.square().sum()
            loss = mse + l1 * l1_penalty + l2 * l2_penalty

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()

        logging.info("Linear regression | Epoch %d | Validation MSE: %.5f", epoch, val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    return best_loss


def run_linear_regression():
    """Проверяет линейную регрессию на данных y = 3x + 2 с шумом."""
    torch.manual_seed(42)

    x = torch.randn(200, 1)
    y = 3 * x + 2 + 0.1 * torch.randn_like(x)

    train_data = TensorDataset(x[:160], y[:160])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    model = nn.Linear(1, 1)
    best_loss = train_linear_regression(model, train_loader, x[160:], y[160:])

    print("\n1.1 Linear regression")
    print(f"Best validation MSE: {best_loss:.5f}")
    print(f"Weight: {model.weight.item():.3f}")
    print(f"Bias: {model.bias.item():.3f}")


def run_multiclass_logistic_regression():
    """Обучает многоклассовую модель на Iris и оценивает её на тесте."""
    torch.manual_seed(42)
    iris = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target,
    )

    # Вычисляем параметры масштабирования только на обучающей выборке.
    scaler = StandardScaler()
    x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
    x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train, train_labels),
        batch_size=16,
        shuffle=True,
    )

    model = nn.Linear(x_train.shape[1], len(iris.target_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train()
    for epoch in range(1, 101):
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)

        if epoch % 20 == 0:
            logging.info(
                "Logistic regression | Epoch %d | Loss: %.5f",
                epoch,
                total_loss / len(train_loader.dataset),
            )

    model.eval()
    with torch.no_grad():
        probabilities = model(x_test).softmax(dim=1).numpy()

    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    auc = roc_auc_score(
        y_test, probabilities, multi_class="ovr", average="macro"
    )

    print("\n1.2 Multiclass logistic regression — Iris")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    plot_path = Path(__file__).resolve().parent / "plots" / "confusion_matrix.png"
    plot_path.parent.mkdir(exist_ok=True)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=iris.target_names,
        cmap="Blues",
    )
    display.figure_.tight_layout()
    display.figure_.savefig(plot_path, dpi=150)
    plt.close(display.figure_)
    print(f"Confusion matrix saved to: {plot_path}")


def main():
    """Запускает оба примера из задания 1."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_linear_regression()
    run_multiclass_logistic_regression()


if __name__ == "__main__":
    main()