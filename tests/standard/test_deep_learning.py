import unittest

import numpy as np


# Activation functions as implemented in gioi-thieu-deep-learning.md
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)


def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


class TestDeepLearning(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_sigmoid_properties(self):
        z = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
        s = sigmoid(z)
        self.assertTrue(np.all(s >= 0.0))
        self.assertTrue(np.all(s <= 1.0))
        self.assertAlmostEqual(s[2], 0.5)
        self.assertAlmostEqual(s[0], 0.0)
        self.assertAlmostEqual(s[4], 1.0)

    def test_tanh_properties(self):
        z = np.array([-5.0, 0.0, 5.0])
        t = tanh(z)
        self.assertTrue(np.all(t >= -1.0))
        self.assertTrue(np.all(t <= 1.0))
        self.assertAlmostEqual(t[1], 0.0)
        # Symmetry check
        np.testing.assert_allclose(t[0], -t[2])

    def test_relu_properties(self):
        z = np.array([-3.5, 0.0, 4.2])
        r = relu(z)
        self.assertEqual(r[0], 0.0)
        self.assertEqual(r[1], 0.0)
        self.assertEqual(r[2], 4.2)
        self.assertTrue(np.all(r >= 0.0))

    def test_leaky_relu_properties(self):
        z = np.array([-10.0, 0.0, 10.0])
        lr = leaky_relu(z, alpha=0.01)
        self.assertAlmostEqual(lr[0], -0.1)
        self.assertAlmostEqual(lr[1], 0.0)
        self.assertAlmostEqual(lr[2], 10.0)

    def test_softmax_properties(self):
        # 2D input (batch of 3 samples, 4 classes)
        logits = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [1000.0, 1001.0, 1002.0, 1003.0],  # Large numbers to test numerical stability
            [-10.0, -20.0, 0.0, 5.0]
        ])
        probs = softmax(logits)

        # 1. Output shapes must match input
        self.assertEqual(probs.shape, logits.shape)

        # 2. Each row must sum to 1.0
        row_sums = np.sum(probs, axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-6)

        # 3. All values must be in (0, 1)
        self.assertTrue(np.all(probs > 0.0))
        self.assertTrue(np.all(probs <= 1.0))

        # 4. Invariant to shift: logits[0] shifted by +1000 must yield identical probs
        np.testing.assert_allclose(probs[0], probs[1], atol=1e-6)

    def test_mlp_forward_backward_convergence(self):
        # Synthetic concentric circles dataset
        n_samples = 200
        half = n_samples // 2
        r_in = np.random.uniform(0.0, 0.45, half)
        th_in = np.random.uniform(0, 2 * np.pi, half)
        X_in = np.stack([r_in * np.cos(th_in), r_in * np.sin(th_in)], axis=1)
        y_in = np.zeros((half, 1))

        r_out = np.random.uniform(0.75, 1.2, half)
        th_out = np.random.uniform(0, 2 * np.pi, half)
        X_out = np.stack([r_out * np.cos(th_out), r_out * np.sin(th_out)], axis=1)
        y_out = np.ones((half, 1))

        X = np.vstack([X_in, X_out])
        y = np.vstack([y_in, y_out])
        perm = np.random.permutation(n_samples)
        X, y = X[perm], y[perm]

        # Model architecture: 2 -> 16 -> 1
        d_in, d_hidden, d_out = 2, 16, 1
        W1 = np.random.randn(d_in, d_hidden) * np.sqrt(2.0 / d_in)
        b1 = np.zeros((1, d_hidden))
        W2 = np.random.randn(d_hidden, d_out) * np.sqrt(2.0 / d_hidden)
        b2 = np.zeros((1, d_out))

        mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
        mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
        mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
        mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)

        lr = 0.05
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        epochs = 150
        losses = []

        for epoch in range(1, epochs + 1):
            # Forward
            z1 = X @ W1 + b1
            a1 = np.maximum(0.0, z1)
            z2 = a1 @ W2 + b2
            y_hat = 1.0 / (1.0 + np.exp(-np.clip(z2, -500.0, 500.0)))

            # Loss (BCE)
            loss = -np.mean(y * np.log(y_hat + 1e-12) + (1.0 - y) * np.log(1.0 - y_hat + 1e-12))
            losses.append(loss)

            # Backward
            dz2 = (y_hat - y) / n_samples
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0).astype(float)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # Gradient validity check
            self.assertFalse(np.isnan(dW1).any())
            self.assertFalse(np.isnan(dW2).any())

            # Adam update
            for p, dp, m, v in [
                (W1, dW1, mW1, vW1), (b1, db1, mb1, vb1),
                (W2, dW2, mW2, vW2), (b2, db2, mb2, vb2)
            ]:
                m[:] = beta1 * m + (1.0 - beta1) * dp
                v[:] = beta2 * v + (1.0 - beta2) * (dp ** 2)
                m_hat = m / (1.0 - beta1 ** epoch)
                v_hat = v / (1.0 - beta2 ** epoch)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Initial loss around ~0.7, final loss should be below 0.1
        self.assertGreater(losses[0], 0.6)
        self.assertLess(losses[-1], 0.1)

        # Accuracy should be > 95%
        final_preds = (y_hat >= 0.5).astype(int)
        accuracy = np.mean(final_preds == y)
        self.assertGreater(accuracy, 0.95)


if __name__ == "__main__":
    unittest.main()
