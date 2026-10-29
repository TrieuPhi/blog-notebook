"""
test_ragas_evaluators.py
Unit tests for RAGAS evaluation metric calculation logic:
- Faithfulness (|V| / |S|)
- Answer Relevance (Cosine similarity of synthetic query embeddings)
- Context Precision (Mean Average Precision over ranked chunks)
- Context Recall (Ground truth factual coverage)
"""

import re
import unittest
from dataclasses import dataclass

import numpy as np


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Safe cosine similarity calculation between two numpy arrays."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


@dataclass
class EvaluationResult:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    verdict_details: dict[str, any]


class DeterministicRAGASEvaluator:
    """
    RAGAS evaluator computing factual faithfulness, answer relevance,
    ranked context precision (mAP), and context recall.
    """

    def extract_statements(self, text: str) -> list[str]:
        """Extract atomic factual statements from text safely without splitting decimals."""
        raw_sentences = re.split(r"(?<=[.!?;\n])\s+", text)
        return [s.strip().rstrip(".!?;\n") for s in raw_sentences if len(s.strip()) > 5]

    def is_statement_supported(self, statement: str, context: str) -> bool:
        """Check if statement is entailed by context using keyword matching."""
        words = re.findall(r"\w+", statement.lower())
        stopwords = {
            "là", "của", "và", "những", "các", "được", "trong", "cho",
            "vào", "ở", "với", "đã", "có", "một", "này", "khi", "để", "ra"
        }
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        if not keywords:
            return True

        context_lower = context.lower()
        matched = sum(1 for kw in keywords if kw in context_lower)
        coverage_ratio = matched / len(keywords)
        return coverage_ratio >= 0.70

    def compute_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Calculate Faithfulness: |V| / |S|."""
        statements = self.extract_statements(answer)
        if not statements:
            return 0.0

        full_context = " ".join(contexts)
        verified_count = sum(
            1 for s in statements if self.is_statement_supported(s, full_context)
        )
        return float(verified_count / len(statements))

    def compute_answer_relevance(
        self,
        query_embedding: np.ndarray,
        synthetic_query_embeddings: list[np.ndarray]
    ) -> float:
        """Calculate Answer Relevance as mean cosine similarity."""
        if not synthetic_query_embeddings:
            return 0.0

        scores = [
            cosine_similarity(query_embedding, sq_emb)
            for sq_emb in synthetic_query_embeddings
        ]
        return float(np.mean(scores))

    def compute_context_precision(
        self,
        retrieved_contexts: list[str],
        ground_truth: str,
        explicit_relevance_labels: list[int] | None = None
    ) -> float:
        """Calculate Context Precision using Mean Average Precision (mAP)."""
        if not retrieved_contexts:
            return 0.0

        if explicit_relevance_labels is not None:
            relevance = explicit_relevance_labels
        else:
            gt_keywords = [
                w for w in re.findall(r"\w+", ground_truth.lower())
                if len(w) > 3
            ]
            relevance = []
            for ctx in retrieved_contexts:
                ctx_lower = ctx.lower()
                matched = any(kw in ctx_lower for kw in gt_keywords)
                relevance.append(1 if matched else 0)

        total_relevant = sum(relevance)
        if total_relevant == 0:
            return 0.0

        precisions_at_k = []
        cumulative_relevant = 0

        for k, rel in enumerate(relevance, start=1):
            if rel == 1:
                cumulative_relevant += 1
                precision_k = cumulative_relevant / k
                precisions_at_k.append(precision_k)

        return float(sum(precisions_at_k) / total_relevant)

    def compute_context_recall(
        self,
        ground_truth: str,
        retrieved_contexts: list[str]
    ) -> float:
        """Calculate Context Recall: proportion of ground truth facts covered in contexts."""
        gt_statements = self.extract_statements(ground_truth)
        if not gt_statements:
            return 0.0

        full_context = " ".join(retrieved_contexts) if retrieved_contexts else ""
        covered_count = sum(
            1 for gt_s in gt_statements if self.is_statement_supported(gt_s, full_context)
        )
        return float(covered_count / len(gt_statements))

    def evaluate_sample(
        self,
        query: str,
        contexts: list[str],
        answer: str,
        ground_truth: str,
        query_emb: np.ndarray,
        synth_query_embs: list[np.ndarray],
        explicit_relevance: list[int] | None = None
    ) -> EvaluationResult:
        """Aggregate evaluation for a RAG sample."""
        faith = self.compute_faithfulness(answer, contexts)
        relevancy = self.compute_answer_relevance(query_emb, synth_query_embs)
        ctx_precision = self.compute_context_precision(contexts, ground_truth, explicit_relevance)
        ctx_recall = self.compute_context_recall(ground_truth, contexts)

        return EvaluationResult(
            faithfulness=faith,
            answer_relevance=relevancy,
            context_precision=ctx_precision,
            context_recall=ctx_recall,
            verdict_details={
                "extracted_statements": self.extract_statements(answer),
                "total_statements": len(self.extract_statements(answer))
            }
        )


class TestRAGASEvaluators(unittest.TestCase):
    """Unit tests for DeterministicRAGASEvaluator."""

    def setUp(self):
        self.evaluator = DeterministicRAGASEvaluator()

    def test_cosine_similarity_identical(self):
        vec = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        sim = cosine_similarity(u, v)
        self.assertAlmostEqual(sim, 0.0, places=5)

    def test_cosine_similarity_zero_vector(self):
        u = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(u, v)
        self.assertEqual(sim, 0.0)

    def test_faithfulness_fully_supported(self):
        contexts = [
            "Kiến trúc Transformer do Google Brain công bố năm 2017.",
            "Transformer sử dụng cơ chế Self-Attention thay thế RNN."
        ]
        answer = "Transformer do Google Brain công bố năm 2017. Mô hình sử dụng Self-Attention thay thế RNN."
        score = self.evaluator.compute_faithfulness(answer, contexts)
        self.assertEqual(score, 1.0)

    def test_faithfulness_partially_supported(self):
        contexts = [
            "Kiến trúc Transformer do Google công bố năm 2017."
        ]
        answer = (
            "Transformer do Google công bố năm 2017. "
            "Sau đó Apple đã mua lại toàn bộ công nghệ này với giá mười tỷ đô la."
        )
        score = self.evaluator.compute_faithfulness(answer, contexts)
        # 1 câu đúng, 1 câu bịa đặt -> 0.5
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_faithfulness_complete_hallucination(self):
        contexts = ["Python ra mắt lần đầu tiên vào năm 1991 bởi Guido van Rossum."]
        answer = "JavaScript do Brendan Eich tạo ra trong vòng mười ngày tại Netscape năm 1995."
        score = self.evaluator.compute_faithfulness(answer, contexts)
        self.assertEqual(score, 0.0)

    def test_faithfulness_empty_answer(self):
        contexts = ["Bất kỳ ngữ cảnh nào."]
        score = self.evaluator.compute_faithfulness("", contexts)
        self.assertEqual(score, 0.0)

    def test_answer_relevance(self):
        q_emb = np.array([1.0, 0.0, 0.0])
        synth_embs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.8, 0.6, 0.0])
        ]
        score = self.evaluator.compute_answer_relevance(q_emb, synth_embs)
        # cos([1,0,0], [1,0,0]) = 1.0
        # cos([1,0,0], [0.8, 0.6, 0]) = 0.8
        # mean = (1.0 + 0.8) / 2 = 0.9
        self.assertAlmostEqual(score, 0.9, places=4)

    def test_answer_relevance_empty(self):
        q_emb = np.array([1.0, 0.0, 0.0])
        score = self.evaluator.compute_answer_relevance(q_emb, [])
        self.assertEqual(score, 0.0)

    def test_context_precision_perfect_ranking(self):
        # Top 2 chunks are relevant, 3rd is irrelevant: [1, 1, 0]
        # P@1 = 1/1 = 1.0; P@2 = 2/2 = 1.0; total relevant = 2
        # mAP = (1.0 + 1.0) / 2 = 1.0
        contexts = ["Chunk 1", "Chunk 2", "Chunk 3"]
        score = self.evaluator.compute_context_precision(
            contexts, ground_truth="irrelevant for explicit", explicit_relevance_labels=[1, 1, 0]
        )
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_context_precision_degraded_ranking(self):
        # 1st irrelevant, 2nd & 3rd relevant: [0, 1, 1]
        # P@1 = 0 (v=0); P@2 = 1/2 = 0.5 (v=1); P@3 = 2/3 ≈ 0.6667 (v=1)
        # mAP = (0.5 + 0.6667) / 2 ≈ 0.5833
        contexts = ["Chunk 1", "Chunk 2", "Chunk 3"]
        score = self.evaluator.compute_context_precision(
            contexts, ground_truth="irrelevant for explicit", explicit_relevance_labels=[0, 1, 1]
        )
        expected_map = (0.5 + (2.0 / 3.0)) / 2.0
        self.assertAlmostEqual(score, expected_map, places=4)

    def test_context_precision_no_relevant_contexts(self):
        contexts = ["Chunk 1", "Chunk 2"]
        score = self.evaluator.compute_context_precision(
            contexts, ground_truth="", explicit_relevance_labels=[0, 0]
        )
        self.assertEqual(score, 0.0)

    def test_context_recall_full_coverage(self):
        ground_truth = (
            "Transformer được công bố năm 2017 bởi Google Brain. "
            "Cơ chế cốt lõi là Self-Attention."
        )
        contexts = [
            "Các nhà nghiên cứu Google Brain đã công bố Transformer năm 2017.",
            "Cơ chế cốt lõi là Self-Attention thay thế hoàn toàn mạng hồi quy."
        ]
        score = self.evaluator.compute_context_recall(ground_truth, contexts)
        self.assertEqual(score, 1.0)

    def test_context_recall_partial_coverage(self):
        ground_truth = (
            "Transformer được công bố năm 2017 bởi Google Brain. "
            "Mô hình đạt kỷ lục BLEU 28.4 trên tập dịch máy WMT 2014 Anh Đức."
        )
        contexts = [
            "Các nhà nghiên cứu Google Brain đã công bố Transformer năm 2017."
            # Context thiếu thông tin về BLEU và WMT 2014
        ]
        score = self.evaluator.compute_context_recall(ground_truth, contexts)
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_evaluate_sample_end_to_end(self):
        q_vec = np.array([1.0, 0.0, 0.0])
        synth = [np.array([0.9, 0.1, 0.0])]
        res = self.evaluator.evaluate_sample(
            query="Transformer ra mắt năm nào?",
            contexts=["Transformer ra mắt năm 2017 bởi Google."],
            answer="Transformer ra mắt năm 2017.",
            ground_truth="Transformer ra mắt năm 2017.",
            query_emb=q_vec,
            synth_query_embs=synth,
            explicit_relevance=[1]
        )
        self.assertIsInstance(res, EvaluationResult)
        self.assertGreater(res.faithfulness, 0.9)
        self.assertGreater(res.context_recall, 0.9)
        self.assertAlmostEqual(res.context_precision, 1.0, places=4)
        self.assertIn("extracted_statements", res.verdict_details)


if __name__ == "__main__":
    unittest.main()
