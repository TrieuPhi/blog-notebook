import unittest

import numpy as np


class LinearRegressionGD:
    """
    Linear Regression with Batch Gradient Descent and optional L2 Regularization (Ridge).
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, l2_penalty: float = 0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            residuals = y_pred - y

            mse = 0.5 * np.mean(residuals ** 2)
            reg = (0.5 * self.l2_penalty / n_samples) * np.sum(self.weights ** 2)
            self.loss_history.append(float(mse + reg))

            # Gradients
            dw = (1.0 / n_samples) * (X.T @ residuals) + (self.l2_penalty / n_samples) * self.weights
            db = (1.0 / n_samples) * np.sum(residuals)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R^2 coefficient of determination."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - (ss_res / (ss_tot + 1e-12)))


class LogisticRegressionGD:
    """
    Logistic Regression for binary classification with Batch Gradient Descent and L2 Regularization.
    """
    def __init__(self, learning_rate: float = 0.05, n_iterations: int = 1000, l2_penalty: float = 0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        eps = 1e-15

        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            y_proba = self._sigmoid(z)

            y_p_clipped = np.clip(y_proba, eps, 1.0 - eps)
            bce = -np.mean(y * np.log(y_p_clipped) + (1.0 - y) * np.log(1.0 - y_p_clipped))
            reg = (0.5 * self.l2_penalty / n_samples) * np.sum(self.weights ** 2)
            self.loss_history.append(float(bce + reg))

            # Gradients
            errors = y_proba - y
            dw = (1.0 / n_samples) * (X.T @ errors) + (self.l2_penalty / n_samples) * self.weights
            db = (1.0 / n_samples) * np.sum(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy score."""
        return float(np.mean(self.predict(X) == y))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1)
    }


class TestMachineLearningFundamentals(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_linear_regression_fitting_and_convergence(self):
        # Synthetic linear dataset: y = 2.5 * x0 - 1.5 * x1 + 3.0 + noise
        X = np.random.randn(120, 2)
        true_w = np.array([2.5, -1.5])
        true_b = 3.0
        y = X @ true_w + true_b + np.random.normal(0, 0.05, 120)

        model = LinearRegressionGD(learning_rate=0.08, n_iterations=400)
        model.fit(X, y)

        # 1. Loss should decrease significantly
        self.assertGreater(model.loss_history[0], model.loss_history[-1])
        self.assertLess(model.loss_history[-1], 0.01)

        # 2. Parameters should closely recover ground truth
        np.testing.assert_allclose(model.weights, true_w, atol=0.1)
        self.assertAlmostEqual(model.bias, true_b, delta=0.1)

        # 3. R^2 score should be high
        r2 = model.score(X, y)
        self.assertGreater(r2, 0.99)

    def test_linear_regression_l2_regularization(self):
        X = np.random.randn(80, 5)
        y = X @ np.array([1.0, -2.0, 3.0, 0.5, -1.5]) + 1.0

        model_unreg = LinearRegressionGD(learning_rate=0.05, n_iterations=300, l2_penalty=0.0)
        model_unreg.fit(X, y)

        model_reg = LinearRegressionGD(learning_rate=0.05, n_iterations=300, l2_penalty=10.0)
        model_reg.fit(X, y)

        # L2 penalty should shrink weight norm
        norm_unreg = np.linalg.norm(model_unreg.weights)
        norm_reg = np.linalg.norm(model_reg.weights)
        self.assertLess(norm_reg, norm_unreg)

    def test_logistic_regression_sigmoid_numerical_stability(self):
        # Extreme negative, zero, extreme positive
        z = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
        sig = LogisticRegressionGD._sigmoid(z)

        self.assertEqual(sig.shape, (5,))
        self.assertTrue(np.all(sig >= 0.0))
        self.assertTrue(np.all(sig <= 1.0))
        self.assertAlmostEqual(sig[2], 0.5, places=5)
        self.assertAlmostEqual(sig[0], 0.0, places=5)
        self.assertAlmostEqual(sig[4], 1.0, places=5)

    def test_logistic_regression_binary_classification(self):
        # Linearly separable 2D dataset
        X = np.random.randn(150, 2)
        y = (X[:, 0] * 1.8 - X[:, 1] * 2.2 + 0.4 > 0).astype(int)

        model = LogisticRegressionGD(learning_rate=0.15, n_iterations=500)
        model.fit(X, y)

        # Loss should decrease
        self.assertGreater(model.loss_history[0], model.loss_history[-1])

        # Predictions should have high accuracy
        acc = model.score(X, y)
        self.assertGreater(acc, 0.95)

        # Probabilities must be within (0, 1)
        proba = model.predict_proba(X)
        self.assertTrue(np.all((proba >= 0.0) & (proba <= 1.0)))

    def test_classification_metrics(self):
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0])

        metrics = compute_metrics(y_true, y_pred)
        # TP = 3 (1 predicted 1)
        # FN = 1 (1 predicted 0)
        # FP = 1 (0 predicted 1)
        # TN = 3 (0 predicted 0)
        self.assertEqual(metrics["TP"], 3)
        self.assertEqual(metrics["FN"], 1)
        self.assertEqual(metrics["FP"], 1)
        self.assertEqual(metrics["TN"], 3)

        self.assertAlmostEqual(metrics["Accuracy"], 6.0 / 8.0)
        self.assertAlmostEqual(metrics["Precision"], 3.0 / 4.0)
        self.assertAlmostEqual(metrics["Recall"], 3.0 / 4.0)
        self.assertAlmostEqual(metrics["F1"], 0.75)


if __name__ == "__main__":
    unittest.main()
