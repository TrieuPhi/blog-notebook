"""
test_adversarial_empirical_stress.py

Adversarial Stress Test Suite for Blog Notebook Algorithms and Heavy GPU Standalone Scripts.
Probes numerical stability, boundary values, zero divisions, overflow/underflow,
singular matrices, disconnected graph topologies, empty inputs, and CLI interfaces.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

# Add standard and gpu_standalone directories to sys.path
TESTS_DIR = Path(__file__).resolve().parent
STANDARD_DIR = TESTS_DIR / "standard"
GPU_DIR = TESTS_DIR / "gpu_standalone"

sys.path.insert(0, str(STANDARD_DIR))
sys.path.insert(0, str(GPU_DIR))

# Import target implementations
from test_deep_learning import leaky_relu, relu, sigmoid, softmax, tanh
from test_knowledge_graph import NoteDocument, PersonalKnowledgeGraph
from test_ml_fundamentals import (
    LinearRegressionGD,
    LogisticRegressionGD,
    compute_metrics,
)
from test_python_deeply import (
    BaseTestPlugin,
    CyclicNode,
    NonDataMethodDescriptor,
    PluginRegistryMeta,
    PositiveInteger,
)
from test_rag_pipeline import (
    QualityGateConfig,
    RAGEvalPipelineRunner,
    TestCase,
)
from test_ragas_evaluators import (
    DeterministicRAGASEvaluator,
    cosine_similarity,
)


class TestMLFoundationsAdversarial(unittest.TestCase):
    """Adversarially probes ML Foundations: Linear regression, Logistic regression, metrics."""

    def test_linear_regression_collinear_features(self):
        """Stress: Perfectly collinear features (rank deficient X)."""
        np.random.seed(42)
        x0 = np.random.randn(100, 1)
        # x1 is exact multiple of x0 -> singular covariance matrix
        X = np.hstack([x0, 2.0 * x0, -3.0 * x0])
        y = (2.5 * x0[:, 0] + 1.0).astype(float)

        # Batch Gradient Descent with L2 penalty should stably converge without LinAlgError
        model = LinearRegressionGD(learning_rate=0.01, n_iterations=300, l2_penalty=1.0)
        model.fit(X, y)

        self.assertFalse(np.isnan(model.weights).any())
        self.assertFalse(np.isinf(model.weights).any())
        self.assertFalse(np.isnan(model.bias))
        r2 = model.score(X, y)
        self.assertFalse(np.isnan(r2))
        self.assertGreater(r2, 0.90)

    def test_linear_regression_constant_target_zero_variance(self):
        """Stress: Constant target y has ss_tot == 0. Verifies division by zero protection in score()."""
        X = np.random.randn(50, 3)
        y = np.full(50, 42.0)

        model = LinearRegressionGD(learning_rate=0.01, n_iterations=100)
        model.fit(X, y)

        r2 = model.score(X, y)
        self.assertFalse(np.isnan(r2))
        self.assertFalse(np.isinf(r2))

    def test_linear_regression_single_sample_and_high_dimension(self):
        """Stress: n_samples == 1, n_features == 50."""
        X = np.random.randn(1, 50)
        y = np.array([3.14])

        model = LinearRegressionGD(learning_rate=0.001, n_iterations=50, l2_penalty=0.1)
        model.fit(X, y)

        self.assertEqual(model.weights.shape, (50,))
        self.assertFalse(np.isnan(model.weights).any())

    def test_linear_regression_extreme_scale_features(self):
        """Stress: Features with magnitude 1e6 and 1e-6."""
        np.random.seed(42)
        X = np.column_stack([
            np.random.randn(100) * 1e6,
            np.random.randn(100) * 1e-6,
        ])
        y = X[:, 0] * 1e-6 + X[:, 1] * 1e6 + 2.0

        model = LinearRegressionGD(learning_rate=1e-15, n_iterations=100, l2_penalty=1e-5)
        model.fit(X, y)

        self.assertFalse(np.isnan(model.weights).any())
        self.assertFalse(np.isnan(model.bias))

    def test_logistic_regression_sigmoid_extreme_values(self):
        """Stress: Extreme floating point values into sigmoid.
        np.clip(z, -500.0, 500.0) yields 7.12e-218 for -500, which is non-zero in float64.
        """
        z = np.array([-1e15, -1000.0, -500.0, 0.0, 500.0, 1000.0, 1e15])
        sig = LogisticRegressionGD._sigmoid(z)

        self.assertTrue(np.all(np.isfinite(sig)))
        self.assertTrue(np.all(sig >= 0.0))
        self.assertTrue(np.all(sig <= 1.0))
        # Float64 subnormal bounds: sigmoid(-500) evaluates to ~7.12e-218 < 1e-100
        self.assertLess(sig[0], 1e-100)
        self.assertAlmostEqual(sig[0], 0.0, places=5)
        self.assertEqual(sig[-1], 1.0)
        self.assertAlmostEqual(sig[3], 0.5)

    def test_logistic_regression_complete_separation(self):
        """Stress: Linearly separable data where unregularized weights would diverge to infinity."""
        X = np.array([[-10.0], [-5.0], [-2.0], [2.0], [5.0], [10.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        # With L2 penalty, weights are bounded
        model = LogisticRegressionGD(learning_rate=0.1, n_iterations=200, l2_penalty=1.0)
        model.fit(X, y)

        self.assertTrue(np.all(np.isfinite(model.weights)))
        self.assertLess(np.max(np.abs(model.weights)), 50.0)
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_classification_metrics_zero_division_resilience(self):
        """Stress: All 0 predictions, all 1 predictions, empty arrays."""
        # 1. All 0 predictions (TP=0, FP=0) -> Precision must be 0.0 without ZeroDivisionError
        y_true_zero = np.array([0, 0, 0, 0])
        y_pred_zero = np.array([0, 0, 0, 0])
        m_zero = compute_metrics(y_true_zero, y_pred_zero)
        self.assertEqual(m_zero["Precision"], 0.0)
        self.assertEqual(m_zero["Recall"], 0.0)
        self.assertEqual(m_zero["F1"], 0.0)
        self.assertEqual(m_zero["Accuracy"], 1.0)

        # 2. All 1 predictions (TN=0, FN=0) -> Recall calculation
        y_true_one = np.array([1, 1, 1])
        y_pred_one = np.array([1, 1, 1])
        m_one = compute_metrics(y_true_one, y_pred_one)
        self.assertEqual(m_one["Precision"], 1.0)
        self.assertEqual(m_one["Recall"], 1.0)
        self.assertEqual(m_one["F1"], 1.0)

        # 3. Complete disagreement
        y_t = np.array([1, 1, 1])
        y_p = np.array([0, 0, 0])
        m_mismatch = compute_metrics(y_t, y_p)
        self.assertEqual(m_mismatch["Accuracy"], 0.0)
        self.assertEqual(m_mismatch["F1"], 0.0)


class TestDeepLearningAdversarial(unittest.TestCase):
    """Adversarially probes Deep Learning: Activation numerical stability, softmax, Adam."""

    def test_softmax_extreme_logits_and_identical_values(self):
        """Stress: Softmax with astronomical values, identical values, and large negatives."""
        # Case 1: Huge numbers (would overflow exp without shift invariance)
        logits_huge = np.array([[1e9, 1e9 + 1, 1e9 + 2]])
        probs_huge = softmax(logits_huge)
        self.assertTrue(np.all(np.isfinite(probs_huge)))
        self.assertAlmostEqual(float(np.sum(probs_huge)), 1.0, places=5)

        # Case 2: Identical logits -> exact uniform distribution
        logits_ident = np.array([[7.0, 7.0, 7.0, 7.0]])
        probs_ident = softmax(logits_ident)
        np.testing.assert_allclose(probs_ident, np.full((1, 4), 0.25), atol=1e-6)

        # Case 3: 3D tensor input (batch, seq_len, vocab)
        logits_3d = np.random.randn(2, 4, 8) * 50.0
        probs_3d = softmax(logits_3d)
        self.assertEqual(probs_3d.shape, (2, 4, 8))
        np.testing.assert_allclose(np.sum(probs_3d, axis=-1), np.ones((2, 4)), atol=1e-5)

    def test_activation_derivatives_and_bounds(self):
        """Stress: Activation functions on edge ranges."""
        z = np.array([-1e8, -1.0, 0.0, 1.0, 1e8])

        # Sigmoid bounded [0, 1]
        sig = sigmoid(z)
        self.assertTrue(np.all(sig >= 0.0) and np.all(sig <= 1.0))

        # Tanh bounded [-1, 1]
        t = tanh(z)
        self.assertTrue(np.all(t >= -1.0) and np.all(t <= 1.0))

        # ReLU non-negative
        r = relu(z)
        self.assertEqual(r[0], 0.0)
        self.assertEqual(r[-1], 1e8)

        # Leaky ReLU alpha edge cases
        lr_zero = leaky_relu(np.array([-5.0, 5.0]), alpha=0.0)
        self.assertEqual(lr_zero[0], 0.0)
        self.assertEqual(lr_zero[1], 5.0)

    def test_mlp_adam_moment_decay_under_zero_gradient(self):
        """Stress: Adam optimizer numerical behavior when gradients are strictly 0."""
        W = np.ones((4, 4))
        dW = np.zeros((4, 4))
        mW = np.zeros((4, 4))
        vW = np.zeros((4, 4))

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        epoch = 1

        mW[:] = beta1 * mW + (1.0 - beta1) * dW
        vW[:] = beta2 * vW + (1.0 - beta2) * (dW ** 2)
        m_hat = mW / (1.0 - beta1 ** epoch)
        v_hat = vW / (1.0 - beta2 ** epoch)
        step = 0.01 * m_hat / (np.sqrt(v_hat) + eps)
        W -= step

        # Weight should not have changed and remain finite
        np.testing.assert_array_equal(W, np.ones((4, 4)))
        self.assertTrue(np.all(np.isfinite(W)))


class TestPythonInternalsAdversarial(unittest.TestCase):
    """Adversarially probes Python Internals: Descriptors, Metaclasses, and Garbage Collection."""

    class Account:
        balance = PositiveInteger(default=100)

        def __init__(self, balance: int = 100):
            self.balance = balance

    def test_descriptor_type_boundary_checks(self):
        """Stress: Descriptor rejects boolean (subclass of int), complex, None, and strings."""
        acc = self.Account(50)

        # Boolean
        with self.assertRaises(TypeError):
            acc.balance = True  # type: ignore

        # Float
        with self.assertRaises(TypeError):
            acc.balance = 100.5  # type: ignore

        # Negative and zero
        with self.assertRaises(ValueError):
            acc.balance = 0
        with self.assertRaises(ValueError):
            acc.balance = -99

        # Complex number
        with self.assertRaises(TypeError):
            acc.balance = 1 + 2j  # type: ignore

    def test_data_descriptor_priority_over_direct_dict_mutation(self):
        """Stress: Ensure instance.__dict__ cannot shadow a Data Descriptor."""
        acc = self.Account(250)
        self.assertEqual(acc.balance, 250)

        # Malicious directly injected __dict__ key
        acc.__dict__["balance"] = "corrupted_raw_string"
        # Access must STILL route to PositiveInteger.__get__
        self.assertEqual(acc.balance, 250)

    def test_non_data_descriptor_shadowing_mechanics(self):
        """Stress: NonDataMethodDescriptor can be legitimately shadowed by instance attribute."""
        class Service:
            ping = NonDataMethodDescriptor(lambda self: "pong")

        s = Service()
        self.assertEqual(s.ping(), "pong")

        # Shadow
        s.ping = lambda: "custom_ping"
        self.assertEqual(s.ping(), "custom_ping")

    def test_metaclass_plugin_registry_conflict_and_isolation(self):
        """Stress: Metaclass auto-registration, interface validation, and dictionary isolation."""
        PluginRegistryMeta.registry.clear()

        # Valid tool
        class ValidTool(BaseTestPlugin, plugin_id="tool_alpha"):
            def execute(self, payload: dict) -> dict:
                return {"status": "ok"}

        self.assertIn("tool_alpha", PluginRegistryMeta.registry)

        # Missing execute method -> must raise TypeError
        with self.assertRaises(TypeError):
            class MissingExecute(BaseTestPlugin, plugin_id="bad"):
                pass

        # Non-callable execute attribute -> must raise TypeError
        with self.assertRaises(TypeError):
            class NonCallableExecute(BaseTestPlugin, plugin_id="bad2"):
                execute = "not a function"

    def test_cyclic_reference_isolation_and_reclamation(self):
        """Stress: Multi-node cyclic references reclaimed by cyclic GC."""
        gc.disable()
        try:
            nodes = [CyclicNode(f"N{i}") for i in range(5)]
            for i in range(5):
                nodes[i].connect(nodes[(i + 1) % 5])  # Ring cycle: 0->1->2->3->4->0

            # Delete external references
            del nodes

            collected = gc.collect()
            self.assertGreaterEqual(collected, 5)
        finally:
            gc.enable()


class TestKnowledgeGraphAdversarial(unittest.TestCase):
    """Adversarially probes NetworkX Knowledge Graph: Disconnected graphs, dangling nodes, empty inputs."""

    def setUp(self):
        self.pkg = PersonalKnowledgeGraph()

    def test_wikilink_regex_malformed_and_extreme_inputs(self):
        """Stress: Malformed wikilinks, spaces, empty brackets, newlines.
        Adversarial Observation: Pattern r'\\[\\[([^\\]\\|]+)(?:\\|[^\\/]+)?\\]\\]' does not exclude
        '[' in character class, so an unclosed '[[' followed by a closed '[[' greedily merges them.
        """
        raw_text = (
            "Normal [[Machine Learning]]. "
            "Piped [[Deep Learning|Hoc Sau]]. "
            "Empty [[]] and whitespace [[   ]]. "
            "Special chars [[C++ / C#]] and [[Node.js (v20)]]. "
            "Piped with spaces [[ Python | Ngon Ngu Python ]]. "
            "Piped with alias empty [[ Rust | ]]."
        )
        links = self.pkg.extract_wikilinks(raw_text)

        self.assertIn("Machine Learning", links)
        self.assertIn("Deep Learning", links)
        self.assertIn("C++ / C#", links)
        self.assertIn("Node.js (v20)", links)
        self.assertIn("Python", links)
        self.assertIn("Rust", links)
        self.assertNotIn("", links)

        # Adversarial check: unclosed bracket absorption
        unclosed_sample = "Unclosed [[Broken link and [[Real Target]]."
        merged_links = self.pkg.extract_wikilinks(unclosed_sample)
        # Confirms hardened regex captures 'Real Target' without unclosed bracket absorption
        self.assertEqual(merged_links, ["Real Target"])

    def test_empty_graph_resilience(self):
        """Stress: Querying empty graph should return empty metrics without crashing."""
        self.pkg.build_graph()
        metrics = self.pkg.compute_centrality_metrics()
        self.assertEqual(metrics, {})
        self.assertEqual(self.pkg.get_backlinks("nonexistent"), [])
        self.assertIsNone(self.pkg.find_shortest_path("a", "b"))
        self.assertEqual(self.pkg.detect_communities(), [set()])

    def test_single_node_graph_metrics(self):
        """Stress: 1 node, 0 edges graph.
        By NetworkX convention for n=1: in_degree_centrality and out_degree_centrality return 1.0 (or 1),
        betweenness_centrality returns 0.0, and pagerank returns 1.0.
        """
        doc = NoteDocument(slug="solo", title="Solo Note", content="No links.")
        self.pkg.add_document(doc)
        self.pkg.build_graph()

        metrics = self.pkg.compute_centrality_metrics()
        self.assertIn("solo", metrics)
        self.assertAlmostEqual(metrics["solo"]["pagerank"], 1.0)
        self.assertEqual(float(metrics["solo"]["in_degree"]), 1.0)
        self.assertEqual(float(metrics["solo"]["out_degree"]), 1.0)
        self.assertEqual(metrics["solo"]["betweenness"], 0.0)

    def test_disconnected_subgraphs_pagerank_and_shortest_path(self):
        """Stress: Two completely disjoint connected components."""
        # Component 1: A <-> B
        self.pkg.add_document(NoteDocument("a", "Note A", "Links to [[Note B]]."))
        self.pkg.add_document(NoteDocument("b", "Note B", "Links to [[Note A]]."))
        # Component 2: C <-> D
        self.pkg.add_document(NoteDocument("c", "Note C", "Links to [[Note D]]."))
        self.pkg.add_document(NoteDocument("d", "Note D", "Links to [[Note C]]."))
        self.pkg.build_graph()

        metrics = self.pkg.compute_centrality_metrics()
        self.assertEqual(len(metrics), 4)
        total_pr = sum(m["pagerank"] for m in metrics.values())
        self.assertAlmostEqual(total_pr, 1.0, places=4)

        # Path within component 1
        path_ab = self.pkg.find_shortest_path("Note A", "Note B")
        self.assertEqual(path_ab, ["a", "b"])

        # Path across disconnected components 1 and 2
        path_ac = self.pkg.find_shortest_path("Note A", "Note C")
        self.assertIsNone(path_ac)

    def test_dangling_nodes_pagerank_convergence(self):
        """Stress: Nodes with out-degree 0 (sink/dangling nodes)."""
        # Node 1 points to 2, 3, 4. Nodes 2, 3, 4 point to nothing.
        self.pkg.add_document(NoteDocument("root", "Root", "[[Sink 1]], [[Sink 2]], [[Sink 3]]"))
        self.pkg.add_document(NoteDocument("sink-1", "Sink 1", "Content"))
        self.pkg.add_document(NoteDocument("sink-2", "Sink 2", "Content"))
        self.pkg.add_document(NoteDocument("sink-3", "Sink 3", "Content"))
        self.pkg.build_graph()

        metrics = self.pkg.compute_centrality_metrics(alpha=0.85)
        total_pr = sum(m["pagerank"] for m in metrics.values())
        self.assertAlmostEqual(total_pr, 1.0, places=4)


class TestRagasEvaluatorsAdversarial(unittest.TestCase):
    """Adversarially probes RAGAS Evaluators: Zero vectors, empty contexts, mAP zero divisions."""

    def setUp(self):
        self.evaluator = DeterministicRAGASEvaluator()

    def test_cosine_similarity_zero_vectors_and_opposite(self):
        """Stress: Zero vectors, single-element vectors, opposite directions."""
        zero_vec = np.zeros(128)
        valid_vec = np.random.randn(128)

        # Zero vector in both positions
        self.assertEqual(cosine_similarity(zero_vec, valid_vec), 0.0)
        self.assertEqual(cosine_similarity(valid_vec, zero_vec), 0.0)
        self.assertEqual(cosine_similarity(zero_vec, zero_vec), 0.0)

        # Opposite vectors -> -1.0
        u = np.array([1.0, 0.0])
        v = np.array([-1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(u, v), -1.0)

    def test_extract_statements_decimals_and_abbreviations(self):
        """Stress: Statement extraction must not split on decimal numbers (3.14, 2.5) or abbreviations."""
        text = "Mô hình Qwen2.5-32B đạt BLEU 34.5 điểm. Tốc độ suy luận là 45.2 token mỗi giây."
        stmts = self.evaluator.extract_statements(text)
        self.assertEqual(len(stmts), 2)
        self.assertIn("34.5", stmts[0])
        self.assertIn("45.2", stmts[1])

    def test_context_precision_all_zero_relevance_labels(self):
        """Stress: mAP with no relevant retrieved contexts (division by zero defense)."""
        score = self.evaluator.compute_context_precision(
            retrieved_contexts=["C1", "C2", "C3"],
            ground_truth="Not relevant",
            explicit_relevance_labels=[0, 0, 0],
        )
        self.assertEqual(score, 0.0)

    def test_context_precision_single_chunk_cases(self):
        """Stress: 1 chunk retrieved."""
        # Relevant
        s1 = self.evaluator.compute_context_precision(["C1"], "", explicit_relevance_labels=[1])
        self.assertEqual(s1, 1.0)
        # Irrelevant
        s0 = self.evaluator.compute_context_precision(["C1"], "", explicit_relevance_labels=[0])
        self.assertEqual(s0, 0.0)

    def test_rag_pipeline_empty_testset_and_extreme_thresholds(self):
        """Stress: RAGEvalPipelineRunner with empty testset and strict thresholds."""
        runner = RAGEvalPipelineRunner(QualityGateConfig(min_faithfulness=0.99))
        report_empty = runner.run_benchmark([])
        self.assertEqual(report_empty.build_status, "PASSED")
        self.assertEqual(report_empty.total_cases, 0)

        # 1 failing test case
        q_vec = np.array([1.0, 0.0])
        case_fail = TestCase(
            test_id="TC-STRESS-FAIL",
            query="Q",
            contexts=["Context"],
            generated_answer="Bịa đặt hoàn toàn không có trong context.",
            ground_truth="Context chuẩn.",
            query_vector=q_vec,
            synthetic_query_vectors=[q_vec],
            explicit_relevance_labels=[0],
        )
        report_fail = runner.run_benchmark([case_fail])
        self.assertEqual(report_fail.build_status, "FAILED")
        self.assertEqual(report_fail.failed_cases, 1)


class TestHeavyGPUStandaloneAdversarial(unittest.TestCase):
    """Adversarially probes Heavy GPU Standalone scripts: CLI flags, dry-run, memory models."""

    def test_verify_vllm_serving_cli_dry_run_and_json(self):
        """Stress: verify_vllm_serving.py execution in dry-run mode with JSON output."""
        script_path = GPU_DIR / "verify_vllm_serving.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--dry-run",
            "--block-size", "16",
            "--tensor-parallel-size", "4",
            "--max-num-seqs", "64",
            "--json-output",
        ]
        res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn('"status": "COMPLETED"', res.stdout)
        self.assertIn('"external_fragmentation": 0.0', res.stdout)

    def test_verify_vllm_serving_block_size_32(self):
        """Stress: Alternative block size 32 tokens."""
        script_path = GPU_DIR / "verify_vllm_serving.py"
        cmd = [sys.executable, str(script_path), "--dry-run", "--block-size", "32"]
        res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("vLLM PRODUCTION SERVING VERIFICATION PASSED", res.stdout)

    def test_verify_qwen3_32b_sft_cli_dry_run_and_json(self):
        """Stress: verify_qwen3_32b_sft.py execution in dry-run mode with custom rank/alpha."""
        script_path = GPU_DIR / "verify_qwen3_32b_sft.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--dry-run",
            "--rank", "64",
            "--alpha", "128",
            "--zero-stage", "3",
            "--json-output",
        ]
        res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn('"status": "SUCCESS"', res.stdout)
        self.assertIn('"rank": 64', res.stdout)
        self.assertIn('"alpha": 128', res.stdout)

    def test_verify_qwen3_32b_sft_zero_stages(self):
        """Stress: ZeRO-1 and ZeRO-2 VRAM calculations."""
        script_path = GPU_DIR / "verify_qwen3_32b_sft.py"
        for stage in (1, 2):
            cmd = [sys.executable, str(script_path), "--dry-run", "--zero-stage", str(stage)]
            res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(res.returncode, 0, f"Stage {stage} failed: {res.stderr}")
            self.assertIn("QWEN-32B SFT VERIFICATION PASSED", res.stdout)

    def test_test_gpu_mock_smoke_cli(self):
        """Stress: test_gpu_mock_smoke.py execution."""
        script_path = GPU_DIR / "test_gpu_mock_smoke.py"
        cmd = [sys.executable, str(script_path), "--dry-run"]
        res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("All dry-run checks completed successfully", res.stdout)


if __name__ == "__main__":
    unittest.main()
