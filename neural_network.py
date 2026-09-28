from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Callable

import numpy as np

from metrics import classification_metrics


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    result = np.zeros((len(y), num_classes), dtype=np.float32)
    result[np.arange(len(y)), y.astype(int)] = 1.0
    return result


def cross_entropy_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: list[np.ndarray] | None = None,
    lambda_reg: float = 0.0,
) -> float:
    m = y_true.shape[0]
    eps = 1e-12
    loss = -np.sum(y_true * np.log(y_pred + eps)) / m

    if weights is not None and lambda_reg > 0:
        reg = sum(float(np.sum(W * W)) for W in weights)
        loss += (lambda_reg / (2.0 * m)) * reg
    return float(loss)


class NeuralNetwork:
    """MLP, основанная на исправленной лабораторной работе №9.

    Отличия курсовой версии:
    - число классов не зашито как 10;
    - добавлена L2-регуляризация;
    - добавлены validation Macro F1 и early stopping;
    - веса сохраняются в файл для отдельного predict.py.
    """

    def __init__(self, layer_sizes: list[int], seed: int = 42):
        self.layer_sizes = list(layer_sizes)
        self.num_layers = len(layer_sizes)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.seed = seed

    def initialize_weights(self, method: str = "he") -> None:
        rng = np.random.default_rng(self.seed)
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            if method == "1/n":
                std = np.sqrt(1.0 / n_in)
                W = rng.standard_normal((n_in, n_out)) * std
            elif method == "2/n":
                std = np.sqrt(2.0 / n_in)
                W = rng.standard_normal((n_in, n_out)) * std
            elif method == "xavier":
                limit = np.sqrt(6.0 / (n_in + n_out))
                W = rng.uniform(-limit, limit, (n_in, n_out))
            elif method == "he":
                std = np.sqrt(2.0 / n_in)
                W = rng.standard_normal((n_in, n_out)) * std
            else:
                raise ValueError("method: '1/n', '2/n', 'xavier' или 'he'")

            self.weights.append(W.astype(np.float32))
            self.biases.append(np.zeros((1, n_out), dtype=np.float32))

    def forward(self, X: np.ndarray):
        A = X
        activations = [A]
        zs = []

        for i in range(self.num_layers - 2):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = relu(Z)
            zs.append(Z)
            activations.append(A)

        Z_last = np.dot(A, self.weights[-1]) + self.biases[-1]
        A_out = softmax(Z_last)
        zs.append(Z_last)
        activations.append(A_out)
        return activations, zs

    def backward(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        activations: list[np.ndarray],
        zs: list[np.ndarray],
        learning_rate: float,
        lambda_reg: float = 0.0,
    ) -> None:
        """Корректный backpropagation из fixed-версии lab9.

        Сначала вычисляются все delta/gradient через старые веса,
        и только затем обновляются параметры.
        """
        m = X.shape[0]
        L = self.num_layers - 1

        deltas: list[np.ndarray | None] = [None] * L
        grad_W: list[np.ndarray | None] = [None] * L
        grad_b: list[np.ndarray | None] = [None] * L

        deltas[-1] = activations[-1] - y_true
        grad_W[-1] = np.dot(activations[-2].T, deltas[-1]) / m
        grad_b[-1] = np.sum(deltas[-1], axis=0, keepdims=True) / m

        if lambda_reg > 0:
            grad_W[-1] += (lambda_reg / m) * self.weights[-1]

        for l in range(L - 2, -1, -1):
            deltas[l] = (
                np.dot(deltas[l + 1], self.weights[l + 1].T)
                * relu_derivative(zs[l])
            )
            grad_W[l] = np.dot(activations[l].T, deltas[l]) / m
            grad_b[l] = np.sum(deltas[l], axis=0, keepdims=True) / m
            if lambda_reg > 0:
                grad_W[l] += (lambda_reg / m) * self.weights[l]

        for l in range(L):
            self.weights[l] -= learning_rate * grad_W[l]
            self.biases[l] -= learning_rate * grad_b[l]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray, lambda_reg: float = 0.0) -> float:
        y_onehot = one_hot(y, self.layer_sizes[-1])
        probs = self.predict_proba(X)
        return cross_entropy_loss(y_onehot, probs, self.weights, lambda_reg)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def get_state(self) -> dict:
        return {
            "weights": [W.copy() for W in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def set_state(self, state: dict) -> None:
        self.weights = [W.copy() for W in state["weights"]]
        self.biases = [b.copy() for b in state["biases"]]

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int | str = 32,
        epochs: int = 100,
        learning_rate: float = 0.01,
        lambda_reg: float = 0.0,
        patience: int = 12,
        verbose: bool = True,
        epoch_data_factory: Callable[[int], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[dict, dict]:
        num_classes = self.layer_sizes[-1]
        # X_train/y_train остаются "чистым" TRAIN и используются для history.
        # Если передан epoch_data_factory, фактическая обучающая выборка
        # регенерируется перед каждой эпохой.
        X_train_eval = X_train
        y_train_eval = y_train
        rng = np.random.default_rng(self.seed)

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_macro_f1": [],
        }

        best_state = None
        best_epoch = 0
        best_val_f1 = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            if epoch_data_factory is not None:
                X_epoch, y_epoch = epoch_data_factory(epoch)
            else:
                X_epoch, y_epoch = X_train, y_train

            y_epoch_onehot = one_hot(y_epoch, num_classes)
            m_epoch = len(X_epoch)

            if batch_size == "full":
                X_shuffled = X_epoch
                y_shuffled = y_epoch_onehot
                batches = [(X_shuffled, y_shuffled)]
            else:
                perm = rng.permutation(m_epoch)
                X_shuffled = X_epoch[perm]
                y_shuffled = y_epoch_onehot[perm]
                bs = int(batch_size)
                batches = [
                    (X_shuffled[start:start + bs], y_shuffled[start:start + bs])
                    for start in range(0, m_epoch, bs)
                ]

            for X_batch, y_batch in batches:
                activations, zs = self.forward(X_batch)
                self.backward(
                    X_batch,
                    y_batch,
                    activations,
                    zs,
                    learning_rate=learning_rate,
                    lambda_reg=lambda_reg,
                )

            # Историю считаем на исходном TRAIN без аугментации:
            # так кривые разных режимов остаются сравнимыми.
            train_loss = self.loss(X_train_eval, y_train_eval, lambda_reg=lambda_reg)
            val_loss = self.loss(X_val, y_val, lambda_reg=lambda_reg)
            train_acc = self.accuracy(X_train_eval, y_train_eval)
            val_pred = self.predict(X_val)
            val_metrics = classification_metrics(y_val, val_pred, num_classes)
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["macro_f1"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)
            history["val_macro_f1"].append(val_f1)

            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                print(
                    f"Эпоха {epoch:03d}/{epochs} | "
                    f"потери={train_loss:.4f} валидационные_потери={val_loss:.4f} | "
                    f"доля_верных={train_acc:.4f} валидационная_доля_верных={val_acc:.4f} | "
                    f"валидационный_макро_f1={val_f1:.4f}"
                )

            if val_f1 > best_val_f1 + 1e-6:
                best_val_f1 = float(val_f1)
                best_epoch = epoch
                best_state = self.get_state()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience > 0 and epochs_without_improvement >= patience:
                if verbose:
                    print(f"Ранняя остановка: {patience} эпох без улучшения макро-F1.")
                break

        if best_state is None:
            raise RuntimeError("Не удалось сохранить лучшее состояние модели.")
        self.set_state(best_state)

        best_info = {"best_epoch": best_epoch, "best_validation_macro_f1": best_val_f1}
        return history, best_info


    def train_uint8_pool(
        self,
        X_pool_uint8: np.ndarray,
        y_pool: np.ndarray,
        X_train_eval: np.ndarray,
        y_train_eval: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 128,
        epochs: int = 80,
        learning_rate: float = 0.0005,
        lambda_reg: float = 0.0,
        patience: int = 0,
        verbose: bool = True,
    ) -> tuple[dict, dict]:
        """Обучает сеть на большом uint8-пуле mini-batch'ами."""
        if X_pool_uint8.dtype != np.uint8:
            raise ValueError(
                "X_pool_uint8 должен иметь dtype=uint8."
            )
        if len(X_pool_uint8) != len(y_pool):
            raise ValueError(
                "Размеры X_pool_uint8 и y_pool не совпадают."
            )
        if batch_size < 1:
            raise ValueError(
                "batch_size должен быть >= 1."
            )

        num_classes = self.layer_sizes[-1]
        rng = np.random.default_rng(self.seed)

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_macro_f1": [],
        }

        best_state = None
        best_epoch = 0
        best_val_f1 = -1.0
        epochs_without_improvement = 0
        m = len(y_pool)

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(m)

            for start in range(0, m, batch_size):
                idx = perm[start:start + batch_size]
                X_batch = (
                    X_pool_uint8[idx]
                    .astype(np.float32)
                    / 255.0
                )
                y_batch = one_hot(
                    y_pool[idx],
                    num_classes,
                )

                activations, zs = self.forward(X_batch)
                self.backward(
                    X_batch,
                    y_batch,
                    activations,
                    zs,
                    learning_rate=learning_rate,
                    lambda_reg=lambda_reg,
                )

            # Для графиков используем чистый train, а не аугментированный пул.
            train_loss = self.loss(
                X_train_eval,
                y_train_eval,
                lambda_reg=lambda_reg,
            )
            val_loss = self.loss(
                X_val,
                y_val,
                lambda_reg=lambda_reg,
            )
            train_acc = self.accuracy(
                X_train_eval,
                y_train_eval,
            )

            val_pred = self.predict(X_val)
            val_metrics = classification_metrics(
                y_val,
                val_pred,
                num_classes,
            )
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["macro_f1"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)
            history["val_macro_f1"].append(val_f1)

            if verbose and (
                epoch == 1
                or epoch % 5 == 0
                or epoch == epochs
            ):
                print(
                    f"Эпоха {epoch:03d}/{epochs} | "
                    f"потери={train_loss:.4f} "
                    f"валидационные_потери={val_loss:.4f} | "
                    f"доля_верных={train_acc:.4f} "
                    f"валидационная_доля_верных={val_acc:.4f} | "
                    f"валидационный_макро_f1={val_f1:.4f}"
                )

            if val_f1 > best_val_f1 + 1e-6:
                best_val_f1 = float(val_f1)
                best_epoch = epoch
                best_state = self.get_state()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                patience > 0
                and epochs_without_improvement >= patience
            ):
                if verbose:
                    print(
                        f"Ранняя остановка: {patience} эпох "
                        "без улучшения макро-F1."
                    )
                break

        if best_state is None:
            raise RuntimeError(
                "Не удалось сохранить лучшее состояние модели."
            )

        self.set_state(best_state)
        return history, {
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_val_f1,
        }



    def train_domain_adaptation_pool(
        self,
        X_pool_uint8: np.ndarray,
        y_pool: np.ndarray,
        X_train_eval: np.ndarray,
        y_train_eval: np.ndarray,
        X_kaggle_val: np.ndarray,
        y_kaggle_val: np.ndarray,
        X_custom_val: np.ndarray,
        y_custom_val: np.ndarray,
        batch_size: int = 64,
        epochs: int = 220,
        learning_rate: float = 0.001,
        lambda_reg: float = 0.001,
        lr_patience: int = 12,
        lr_factor: float = 0.5,
        min_learning_rate: float = 0.00005,
        early_stopping_patience: int = 0,
        selection_custom_weight: float = 0.5,
        verbose: bool = True,
    ) -> tuple[dict, dict]:
        """Обучение с двумя validation-доменами и ReduceLROnPlateau.

        Лучший checkpoint выбирается по комбинированному критерию:
        score = (1-w)*KaggleMacroF1 + w*CustomMacroF1.
        """
        if X_pool_uint8.dtype != np.uint8:
            raise ValueError("Пул должен иметь dtype=uint8.")
        if not 0.0 <= selection_custom_weight <= 1.0:
            raise ValueError("selection_custom_weight должен быть 0..1.")

        num_classes = self.layer_sizes[-1]
        rng = np.random.default_rng(self.seed)
        current_lr = float(learning_rate)

        history = {
            "train_loss": [],
            "kaggle_val_loss": [],
            "custom_val_loss": [],
            "train_accuracy": [],
            "kaggle_val_accuracy": [],
            "custom_val_accuracy": [],
            "kaggle_val_macro_f1": [],
            "custom_val_macro_f1": [],
            "selection_score": [],
            "learning_rate": [],
        }

        best_state = None
        best_epoch = 0
        best_score = -1.0
        best_kaggle_f1 = -1.0
        best_custom_f1 = -1.0

        epochs_without_score_improvement = 0
        epochs_without_lr_improvement = 0
        m = len(y_pool)

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(m)

            for start in range(0, m, batch_size):
                idx = perm[start:start + batch_size]
                X_batch = (
                    X_pool_uint8[idx]
                    .astype(np.float32)
                    / 255.0
                )
                y_batch = one_hot(
                    y_pool[idx],
                    num_classes,
                )

                activations, zs = self.forward(X_batch)
                self.backward(
                    X_batch,
                    y_batch,
                    activations,
                    zs,
                    learning_rate=current_lr,
                    lambda_reg=lambda_reg,
                )

            train_loss = self.loss(
                X_train_eval,
                y_train_eval,
                lambda_reg=lambda_reg,
            )
            kaggle_val_loss = self.loss(
                X_kaggle_val,
                y_kaggle_val,
                lambda_reg=lambda_reg,
            )
            custom_val_loss = self.loss(
                X_custom_val,
                y_custom_val,
                lambda_reg=lambda_reg,
            )
            train_acc = self.accuracy(
                X_train_eval,
                y_train_eval,
            )

            kaggle_pred = self.predict(X_kaggle_val)
            custom_pred = self.predict(X_custom_val)

            kaggle_metrics = classification_metrics(
                y_kaggle_val,
                kaggle_pred,
                num_classes,
            )
            custom_metrics = classification_metrics(
                y_custom_val,
                custom_pred,
                num_classes,
            )

            kaggle_acc = kaggle_metrics["accuracy"]
            custom_acc = custom_metrics["accuracy"]
            kaggle_f1 = kaggle_metrics["macro_f1"]
            custom_f1 = custom_metrics["macro_f1"]

            w = selection_custom_weight
            selection_score = (
                (1.0 - w) * kaggle_f1
                + w * custom_f1
            )

            history["train_loss"].append(train_loss)
            history["kaggle_val_loss"].append(kaggle_val_loss)
            history["custom_val_loss"].append(custom_val_loss)
            history["train_accuracy"].append(train_acc)
            history["kaggle_val_accuracy"].append(kaggle_acc)
            history["custom_val_accuracy"].append(custom_acc)
            history["kaggle_val_macro_f1"].append(kaggle_f1)
            history["custom_val_macro_f1"].append(custom_f1)
            history["selection_score"].append(selection_score)
            history["learning_rate"].append(current_lr)

            improved = selection_score > best_score + 1e-6
            if improved:
                best_score = float(selection_score)
                best_kaggle_f1 = float(kaggle_f1)
                best_custom_f1 = float(custom_f1)
                best_epoch = epoch
                best_state = self.get_state()
                epochs_without_score_improvement = 0
                epochs_without_lr_improvement = 0
            else:
                epochs_without_score_improvement += 1
                epochs_without_lr_improvement += 1

            lr_changed = False
            if (
                lr_patience > 0
                and epochs_without_lr_improvement >= lr_patience
                and current_lr > min_learning_rate + 1e-15
            ):
                new_lr = max(
                    min_learning_rate,
                    current_lr * lr_factor,
                )
                if new_lr < current_lr - 1e-15:
                    current_lr = new_lr
                    lr_changed = True
                epochs_without_lr_improvement = 0

            if verbose and (
                epoch == 1
                or epoch % 5 == 0
                or lr_changed
                or epoch == epochs
            ):
                suffix = (
                    " | скорость обучения уменьшена"
                    if lr_changed
                    else ""
                )
                print(
                    f"Эпоха {epoch:03d}/{epochs} | "
                    f"LR={history['learning_rate'][-1]:.6f} | "
                    f"F1 исходная_валидация={kaggle_f1:.4f} | "
                    f"F1 пользовательская_валидация={custom_f1:.4f} | "
                    f"комбинированный_критерий={selection_score:.4f}"
                    f"{suffix}"
                )

            if (
                early_stopping_patience > 0
                and epochs_without_score_improvement
                >= early_stopping_patience
            ):
                if verbose:
                    print(
                        f"Ранняя остановка: "
                        f"{early_stopping_patience} эпох "
                        "без улучшения комбинированного критерия."
                    )
                break

        if best_state is None:
            raise RuntimeError("Лучшее состояние модели не сохранено.")

        self.set_state(best_state)
        return history, {
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "best_kaggle_validation_macro_f1": best_kaggle_f1,
            "best_custom_validation_macro_f1": best_custom_f1,
            "final_learning_rate": current_lr,
        }



    def train_hard_negative_micro_pool(
        self,
        X_pool_uint8: np.ndarray,
        y_pool: np.ndarray,
        X_train_eval: np.ndarray,
        y_train_eval: np.ndarray,
        X_kaggle_val: np.ndarray,
        y_kaggle_val: np.ndarray,
        X_custom_val: np.ndarray,
        y_custom_val: np.ndarray,
        batch_size: int = 32,
        epochs: int = 30,
        learning_rate: float = 0.00005,
        lambda_reg: float = 0.001,
        lr_patience: int = 8,
        lr_factor: float = 0.5,
        min_learning_rate: float = 0.000025,
        selection_custom_weight: float = 0.70,
        max_kaggle_f1_drop: float = 0.003,
        verbose: bool = True,
    ) -> tuple[dict, dict]:
        """Точечный micro-tuning уже обученной h64.

        Лучший checkpoint выбирается по:
            0.3 * Kaggle validation Macro-F1
          + 0.7 * Custom validation Macro-F1

        Но checkpoint не принимается, если Kaggle validation Macro-F1
        просел больше допустимого max_kaggle_f1_drop относительно
        исходной модели до micro-tuning.

        Если ни одна эпоха не превосходит исходную модель по допустимому
        критерию, исходные веса сохраняются.
        """
        if X_pool_uint8.dtype != np.uint8:
            raise ValueError(
                "X_pool_uint8 должен иметь dtype=uint8."
            )
        if not 0.0 <= selection_custom_weight <= 1.0:
            raise ValueError(
                "selection_custom_weight должен быть 0..1."
            )

        num_classes = self.layer_sizes[-1]
        rng = np.random.default_rng(
            self.seed + 1712
        )
        current_lr = float(
            learning_rate
        )

        # Базовое состояние — уже лучшая h64 v1.7.1.
        baseline_state = self.get_state()

        baseline_k_pred = self.predict(
            X_kaggle_val
        )
        baseline_c_pred = self.predict(
            X_custom_val
        )

        baseline_k_metrics = classification_metrics(
            y_kaggle_val,
            baseline_k_pred,
            num_classes,
        )
        baseline_c_metrics = classification_metrics(
            y_custom_val,
            baseline_c_pred,
            num_classes,
        )

        baseline_k_f1 = float(
            baseline_k_metrics["macro_f1"]
        )
        baseline_c_f1 = float(
            baseline_c_metrics["macro_f1"]
        )

        w = float(
            selection_custom_weight
        )
        baseline_score = (
            (1.0 - w) * baseline_k_f1
            + w * baseline_c_f1
        )

        min_allowed_kaggle_f1 = (
            baseline_k_f1
            - float(max_kaggle_f1_drop)
        )

        history = {
            "train_loss": [],
            "kaggle_val_loss": [],
            "custom_val_loss": [],
            "kaggle_val_macro_f1": [],
            "custom_val_macro_f1": [],
            "selection_score": [],
            "learning_rate": [],
            "guardrail_ok": [],
        }

        # Начинаем с baseline как с уже допустимого checkpoint.
        best_state = baseline_state
        best_epoch = 0
        best_score = baseline_score
        best_k_f1 = baseline_k_f1
        best_c_f1 = baseline_c_f1

        epochs_without_improvement = 0
        m = len(y_pool)

        if verbose:
            print(
                "Исходная h64 перед micro-tuning | "
                f"F1 исходная_валидация={baseline_k_f1:.4f} | "
                f"F1 пользовательская_валидация={baseline_c_f1:.4f} | "
                f"критерий={baseline_score:.4f}"
            )
            print(
                "Защитное ограничение: "
                f"Kaggle validation Macro-F1 >= "
                f"{min_allowed_kaggle_f1:.4f}"
            )

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(m)

            for start in range(
                0,
                m,
                batch_size,
            ):
                idx = perm[
                    start:start + batch_size
                ]

                X_batch = (
                    X_pool_uint8[idx]
                    .astype(np.float32)
                    / 255.0
                )
                y_batch = one_hot(
                    y_pool[idx],
                    num_classes,
                )

                activations, zs = self.forward(
                    X_batch
                )
                self.backward(
                    X_batch,
                    y_batch,
                    activations,
                    zs,
                    learning_rate=current_lr,
                    lambda_reg=lambda_reg,
                )

            train_loss = self.loss(
                X_train_eval,
                y_train_eval,
                lambda_reg=lambda_reg,
            )
            kaggle_val_loss = self.loss(
                X_kaggle_val,
                y_kaggle_val,
                lambda_reg=lambda_reg,
            )
            custom_val_loss = self.loss(
                X_custom_val,
                y_custom_val,
                lambda_reg=lambda_reg,
            )

            k_pred = self.predict(
                X_kaggle_val
            )
            c_pred = self.predict(
                X_custom_val
            )

            k_metrics = classification_metrics(
                y_kaggle_val,
                k_pred,
                num_classes,
            )
            c_metrics = classification_metrics(
                y_custom_val,
                c_pred,
                num_classes,
            )

            k_f1 = float(
                k_metrics["macro_f1"]
            )
            c_f1 = float(
                c_metrics["macro_f1"]
            )
            score = (
                (1.0 - w) * k_f1
                + w * c_f1
            )
            guardrail_ok = (
                k_f1
                >= min_allowed_kaggle_f1
            )

            history["train_loss"].append(
                train_loss
            )
            history["kaggle_val_loss"].append(
                kaggle_val_loss
            )
            history["custom_val_loss"].append(
                custom_val_loss
            )
            history[
                "kaggle_val_macro_f1"
            ].append(k_f1)
            history[
                "custom_val_macro_f1"
            ].append(c_f1)
            history["selection_score"].append(
                score
            )
            history["learning_rate"].append(
                current_lr
            )
            history["guardrail_ok"].append(
                guardrail_ok
            )

            improved = (
                guardrail_ok
                and score > best_score + 1e-6
            )

            if improved:
                best_state = self.get_state()
                best_epoch = epoch
                best_score = score
                best_k_f1 = k_f1
                best_c_f1 = c_f1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            lr_changed = False
            if (
                lr_patience > 0
                and epochs_without_improvement
                >= lr_patience
                and current_lr
                > min_learning_rate + 1e-15
            ):
                new_lr = max(
                    min_learning_rate,
                    current_lr * lr_factor,
                )
                if (
                    new_lr
                    < current_lr - 1e-15
                ):
                    current_lr = new_lr
                    lr_changed = True
                epochs_without_improvement = 0

            if verbose and (
                epoch == 1
                or epoch % 5 == 0
                or improved
                or lr_changed
                or epoch == epochs
            ):
                flags = []
                if improved:
                    flags.append(
                        "новый лучший checkpoint"
                    )
                if not guardrail_ok:
                    flags.append(
                        "отклонён guardrail"
                    )
                if lr_changed:
                    flags.append(
                        "LR уменьшен"
                    )
                suffix = (
                    " | " + ", ".join(flags)
                    if flags
                    else ""
                )

                print(
                    f"Эпоха {epoch:02d}/{epochs} | "
                    f"LR={history['learning_rate'][-1]:.6f} | "
                    f"F1 исходная_валидация={k_f1:.4f} | "
                    f"F1 пользовательская_валидация={c_f1:.4f} | "
                    f"критерий={score:.4f}"
                    f"{suffix}"
                )

        self.set_state(
            best_state
        )

        return history, {
            "best_epoch": best_epoch,
            "best_selection_score": float(
                best_score
            ),
            "best_kaggle_validation_macro_f1": float(
                best_k_f1
            ),
            "best_custom_validation_macro_f1": float(
                best_c_f1
            ),
            "baseline_selection_score": float(
                baseline_score
            ),
            "baseline_kaggle_validation_macro_f1": float(
                baseline_k_f1
            ),
            "baseline_custom_validation_macro_f1": float(
                baseline_c_f1
            ),
            "min_allowed_kaggle_macro_f1": float(
                min_allowed_kaggle_f1
            ),
            "final_learning_rate": float(
                current_lr
            ),
            "improved_over_baseline": bool(
                best_epoch > 0
            ),
        }



def save_model(path: Path, network: NeuralNetwork, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer_sizes": network.layer_sizes,
        "weights": [W.copy() for W in network.weights],
        "biases": [b.copy() for b in network.biases],
        "seed": network.seed,
        "metadata": copy.deepcopy(metadata),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_model(path: Path) -> tuple[NeuralNetwork, dict]:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)

    nn = NeuralNetwork(payload["layer_sizes"], seed=payload.get("seed", 42))
    nn.weights = payload["weights"]
    nn.biases = payload["biases"]
    return nn, payload.get("metadata", {})
