"""
test_rag_pipeline.py
Unit tests for RAG CI/CD Pipeline evaluation and Quality Gate enforcement:
- TestCase metric scoring and overall score calculation
- Quality Gate threshold verification (Faithfulness >= 0.85, Context Recall >= 0.80)
- Benchmark aggregation and build status determination (PASSED vs FAILED)
- Markdown report generation for Pull Request integration
"""

import re
import unittest
from dataclasses import dataclass, field

import numpy as np


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Safe cosine similarity calculation between two numpy arrays."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


@dataclass
class TestCase:
    """Data structure for a RAG test case in Golden Testset."""
    test_id: str
    query: str
    contexts: list[str]
    generated_answer: str
    ground_truth: str
    query_vector: np.ndarray
    synthetic_query_vectors: list[np.ndarray]
    explicit_relevance_labels: list[int] | None = None


@dataclass
class QualityGateConfig:
    """Configuration of Quality Gate safety thresholds."""
    min_faithfulness: float = 0.85
    min_context_recall: float = 0.80
    min_context_precision: float = 0.75
    min_answer_relevance: float = 0.80
    min_overall_score: float = 0.80


@dataclass
class TestCaseMetricResult:
    """Evaluation result for an individual test case."""
    test_id: str
    faithfulness: float
    context_recall: float
    context_precision: float
    answer_relevance: float
    overall_score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class PipelineReport:
    """Aggregated report for the entire benchmark run."""
    build_status: str  # "PASSED" or "FAILED"
    total_cases: int
    passed_cases: int
    failed_cases: int
    mean_faithfulness: float
    mean_context_recall: float
    mean_context_precision: float
    mean_answer_relevance: float
    mean_overall_score: float
    violations: list[str]
    detailed_results: list[TestCaseMetricResult]


class RAGEvalPipelineRunner:
    """RAG CI/CD pipeline evaluator and Quality Gate enforcer."""

    def __init__(self, config: QualityGateConfig | None = None):
        self.config = config or QualityGateConfig()

    def _extract_statements(self, text: str) -> list[str]:
        """Extract atomic factual statements from text safely without splitting decimals."""
        sentences = re.split(r"(?<=[.!?;\n])\s+", text)
        return [s.strip().rstrip(".!?;\n") for s in sentences if len(s.strip()) > 5]

    def _is_supported(self, statement: str, context: str) -> bool:
        """Check if statement is entailed by context using keyword matching."""
        words = re.findall(r"\w+", statement.lower())
        stopwords = {"là", "và", "của", "các", "những", "trong", "được", "cho", "đã", "có", "một"}
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        if not keywords:
            return True
        ctx_lower = context.lower()
        matched = sum(1 for kw in keywords if kw in ctx_lower)
        return (matched / len(keywords)) >= 0.70

    def compute_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Calculate Faithfulness: |V| / |S|."""
        statements = self._extract_statements(answer)
        if not statements:
            return 0.0
        full_ctx = " ".join(contexts)
        verified = sum(1 for s in statements if self._is_supported(s, full_ctx))
        return float(verified / len(statements))

    def compute_context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        """Calculate Context Recall: ground truth factual coverage."""
        gt_statements = self._extract_statements(ground_truth)
        if not gt_statements:
            return 0.0
        full_ctx = " ".join(contexts)
        covered = sum(1 for s in gt_statements if self._is_supported(s, full_ctx))
        return float(covered / len(gt_statements))

    def compute_context_precision(
        self,
        contexts: list[str],
        ground_truth: str,
        explicit_labels: list[int] | None = None
    ) -> float:
        """Calculate Context Precision using Mean Average Precision (mAP)."""
        if not contexts:
            return 0.0
        if explicit_labels is not None:
            relevance = explicit_labels
        else:
            gt_words = [w for w in re.findall(r"\w+", ground_truth.lower()) if len(w) > 3]
            relevance = [
                1 if any(kw in ctx.lower() for kw in gt_words) else 0
                for ctx in contexts
            ]

        total_relevant = sum(relevance)
        if total_relevant == 0:
            return 0.0

        precisions = []
        cum_relevant = 0
        for k, rel in enumerate(relevance, start=1):
            if rel == 1:
                cum_relevant += 1
                precisions.append(cum_relevant / k)
        return float(sum(precisions) / total_relevant)

    def compute_answer_relevance(
        self,
        query_vector: np.ndarray,
        synth_vectors: list[np.ndarray]
    ) -> float:
        """Calculate Answer Relevance as mean cosine similarity."""
        if not synth_vectors:
            return 0.0
        sims = [cosine_similarity(query_vector, sv) for sv in synth_vectors]
        return float(np.mean(sims))

    def evaluate_test_case(self, case: TestCase) -> TestCaseMetricResult:
        """Evaluate an individual test case against Quality Gate rules."""
        faith = self.compute_faithfulness(case.generated_answer, case.contexts)
        recall = self.compute_context_recall(case.ground_truth, case.contexts)
        prec = self.compute_context_precision(case.contexts, case.ground_truth, case.explicit_relevance_labels)
        rel = self.compute_answer_relevance(case.query_vector, case.synthetic_query_vectors)

        # Weighted overall score
        overall = 0.35 * faith + 0.25 * recall + 0.20 * prec + 0.20 * rel

        reasons = []
        passed = True

        if faith < self.config.min_faithfulness:
            passed = False
            reasons.append(f"Faithfulness {faith:.2f} < {self.config.min_faithfulness}")
        if recall < self.config.min_context_recall:
            passed = False
            reasons.append(f"Context Recall {recall:.2f} < {self.config.min_context_recall}")
        if prec < self.config.min_context_precision:
            reasons.append(f"Context Precision {prec:.2f} < {self.config.min_context_precision} (Warning)")
        if rel < self.config.min_answer_relevance:
            reasons.append(f"Answer Relevance {rel:.2f} < {self.config.min_answer_relevance} (Warning)")

        return TestCaseMetricResult(
            test_id=case.test_id,
            faithfulness=faith,
            context_recall=recall,
            context_precision=prec,
            answer_relevance=rel,
            overall_score=overall,
            passed=passed,
            reasons=reasons
        )

    def run_benchmark(self, testset: list[TestCase]) -> PipelineReport:
        """Run benchmark over testset and enforce Quality Gate thresholds."""
        if not testset:
            return PipelineReport(
                build_status="PASSED",
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                mean_faithfulness=0.0,
                mean_context_recall=0.0,
                mean_context_precision=0.0,
                mean_answer_relevance=0.0,
                mean_overall_score=0.0,
                violations=[],
                detailed_results=[]
            )

        detailed_results = [self.evaluate_test_case(case) for case in testset]

        mean_faith = float(np.mean([r.faithfulness for r in detailed_results]))
        mean_recall = float(np.mean([r.context_recall for r in detailed_results]))
        mean_prec = float(np.mean([r.context_precision for r in detailed_results]))
        mean_rel = float(np.mean([r.answer_relevance for r in detailed_results]))
        mean_overall = float(np.mean([r.overall_score for r in detailed_results]))

        passed_cases = sum(1 for r in detailed_results if r.passed)
        failed_cases = len(detailed_results) - passed_cases

        violations = []
        if mean_faith < self.config.min_faithfulness:
            violations.append(
                f"Mean Faithfulness {mean_faith:.3f} dưới ngưỡng {self.config.min_faithfulness}"
            )
        if mean_recall < self.config.min_context_recall:
            violations.append(
                f"Mean Context Recall {mean_recall:.3f} dưới ngưỡng {self.config.min_context_recall}"
            )
        if mean_prec < self.config.min_context_precision:
            violations.append(
                f"Mean Context Precision {mean_prec:.3f} dưới ngưỡng {self.config.min_context_precision}"
            )
        if mean_rel < self.config.min_answer_relevance:
            violations.append(
                f"Mean Answer Relevance {mean_rel:.3f} dưới ngưỡng {self.config.min_answer_relevance}"
            )
        if mean_overall < self.config.min_overall_score:
            violations.append(
                f"Mean Overall Score {mean_overall:.3f} dưới ngưỡng tổng quát {self.config.min_overall_score}"
            )

        build_status = "PASSED" if (len(violations) == 0 and failed_cases == 0) else "FAILED"

        return PipelineReport(
            build_status=build_status,
            total_cases=len(testset),
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            mean_faithfulness=mean_faith,
            mean_context_recall=mean_recall,
            mean_context_precision=mean_prec,
            mean_answer_relevance=mean_rel,
            mean_overall_score=mean_overall,
            violations=violations,
            detailed_results=detailed_results
        )

    def generate_markdown_summary(self, report: PipelineReport) -> str:
        """Generate structured markdown summary report."""
        status_icon = "✅ PASSED" if report.build_status == "PASSED" else "❌ FAILED"
        lines = [
            f"### Kết quả RAG CI/CD Quality Gate: {status_icon}",
            "",
            f"- **Tổng số ca kiểm thử**: {report.total_cases}",
            f"- **Đạt chuẩn**: {report.passed_cases} | **Vi phạm**: {report.failed_cases}",
            "",
            "| Chỉ số | Điểm Trung bình | Ngưỡng Tối thiểu | Trạng thái |",
            "|---|---|---|---|",
            f"| **Faithfulness** | `{report.mean_faithfulness:.4f}` | `{self.config.min_faithfulness}` | {'✅' if report.mean_faithfulness >= self.config.min_faithfulness else '❌'} |",
            f"| **Context Recall** | `{report.mean_context_recall:.4f}` | `{self.config.min_context_recall}` | {'✅' if report.mean_context_recall >= self.config.min_context_recall else '❌'} |",
            f"| **Context Precision** | `{report.mean_context_precision:.4f}` | `{self.config.min_context_precision}` | {'✅' if report.mean_context_precision >= self.config.min_context_precision else '⚠️'} |",
            f"| **Answer Relevance** | `{report.mean_answer_relevance:.4f}` | `{self.config.min_answer_relevance}` | {'✅' if report.mean_answer_relevance >= self.config.min_answer_relevance else '⚠️'} |",
            f"| **Overall Score** | `{report.mean_overall_score:.4f}` | `{self.config.min_overall_score}` | {'✅' if report.mean_overall_score >= self.config.min_overall_score else '❌'} |",
            ""
        ]

        if report.violations:
            lines.append("#### Danh sách Vi phạm Ngưỡng:")
            for v in report.violations:
                lines.append(f"- 🔴 {v}")
            lines.append("")

        return "\n".join(lines)


class TestRAGPipeline(unittest.TestCase):
    """Unit tests for RAGEvalPipelineRunner and Quality Gate enforcement."""

    def setUp(self):
        self.config = QualityGateConfig(
            min_faithfulness=0.85,
            min_context_recall=0.80,
            min_context_precision=0.75,
            min_answer_relevance=0.80,
            min_overall_score=0.80
        )
        self.runner = RAGEvalPipelineRunner(self.config)
        self.q_vec = np.array([1.0, 0.0, 0.0])
        self.synth_good = [np.array([0.95, 0.05, 0.0]), np.array([0.98, 0.02, 0.0])]
        self.synth_bad = [np.array([0.1, 0.9, 0.0])]

    def test_quality_gate_config_defaults(self):
        default_cfg = QualityGateConfig()
        self.assertEqual(default_cfg.min_faithfulness, 0.85)
        self.assertEqual(default_cfg.min_context_recall, 0.80)
        self.assertEqual(default_cfg.min_context_precision, 0.75)
        self.assertEqual(default_cfg.min_answer_relevance, 0.80)

    def test_evaluate_test_case_success(self):
        case = TestCase(
            test_id="TC-PASS",
            query="Transformer ra mắt năm nào?",
            contexts=[
                "Năm 2017, các nhà nghiên cứu tại Google Brain đã công bố bài báo Attention Is All You Need.",
                "Transformer là mô hình Deep Learning dựa trên Self-Attention."
            ],
            generated_answer="Transformer được công bố vào năm 2017 bởi các nhà nghiên cứu tại Google Brain.",
            ground_truth="Transformer được công bố năm 2017 bởi Google Brain.",
            query_vector=self.q_vec,
            synthetic_query_vectors=self.synth_good,
            explicit_relevance_labels=[1, 1]
        )
        result = self.runner.evaluate_test_case(case)
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.faithfulness, 0.85)
        self.assertGreaterEqual(result.context_recall, 0.80)
        self.assertEqual(len(result.reasons), 0)

    def test_evaluate_test_case_faithfulness_failure(self):
        case = TestCase(
            test_id="TC-HALLUCINATION",
            query="Transformer do ai phát minh?",
            contexts=[
                "Transformer do Google Brain giới thiệu năm 2017."
            ],
            generated_answer="Transformer do Meta phát triển năm 2023 và Apple mua lại ngay sau đó.",
            ground_truth="Transformer do Google Brain phát minh năm 2017.",
            query_vector=self.q_vec,
            synthetic_query_vectors=self.synth_good,
            explicit_relevance_labels=[1]
        )
        result = self.runner.evaluate_test_case(case)
        self.assertFalse(result.passed)
        self.assertLess(result.faithfulness, 0.85)
        self.assertTrue(any("Faithfulness" in r for r in result.reasons))

    def test_evaluate_test_case_recall_failure(self):
        case = TestCase(
            test_id="TC-RECALL-MISS",
            query="Nêu các tác giả và thành tựu của bài báo Transformer?",
            contexts=[
                "Transformer là một kiến trúc Deep Learning hiện đại."
                # Ngữ cảnh hoàn toàn thiếu tác giả và năm công bố
            ],
            generated_answer="Transformer là kiến trúc hiện đại.",
            ground_truth="Transformer do Ashish Vaswani và đồng nghiệp tại Google công bố năm 2017.",
            query_vector=self.q_vec,
            synthetic_query_vectors=self.synth_good,
            explicit_relevance_labels=[0]
        )
        result = self.runner.evaluate_test_case(case)
        self.assertFalse(result.passed)
        self.assertLess(result.context_recall, 0.80)
        self.assertTrue(any("Context Recall" in r for r in result.reasons))

    def test_run_benchmark_passed_build(self):
        case1 = TestCase(
            test_id="TC-01",
            query="Q1",
            contexts=["Google giới thiệu Transformer năm 2017."],
            generated_answer="Google giới thiệu Transformer năm 2017.",
            ground_truth="Google giới thiệu Transformer năm 2017.",
            query_vector=self.q_vec,
            synthetic_query_vectors=self.synth_good,
            explicit_relevance_labels=[1]
        )
        case2 = TestCase(
            test_id="TC-02",
            query="Q2",
            contexts=["BERT ra mắt năm 2018 dựa trên Transformer."],
            generated_answer="BERT ra mắt năm 2018 dựa trên Transformer.",
            ground_truth="BERT ra mắt năm 2018 dựa trên Transformer.",
            query_vector=self.q_vec,
            synthetic_query_vectors=self.synth_good,
            explicit_relevance_labels=[1]
        )

        report = self.runner.run_benchmark([case1, case2])
        self.assertEqual(report.build_status, "PASSED")
        self.assertEqual(report.total_cases, 2)
        self.assertEqual(report.passed_cases, 2)
        self.assertEqual(report.failed_cases, 0)
        self.assertEqual(len(report.violations), 0)

    def test_run_benchmark_failed_build_on_violations(self):
        good_case = TestCase(
            test_id="TC-GOOD",
            query="Q1",
            contexts=["Context hợp lệ."],
            generated_answer="Context hợp lệ.",
            ground_truth="Context hợp lệ.",
            query_vector=self.q_vec,
            synthetic_query_vectors=self.synth_good,
            explicit_relevance_labels=[1]
        )
        bad_case = TestCase(
            test_id="TC-BAD",
            query="Q2",
            contexts=["Context không liên quan."],
            generated_answer="Câu trả lời hoàn toàn bịa đặt thông tin mới.",
            ground_truth="Thông tin chuẩn mực cần tìm trong tài liệu.",
            query_vector=self.q_vec,
            synthetic_query_vectors=self.synth_bad,
            explicit_relevance_labels=[0]
        )

        report = self.runner.run_benchmark([good_case, bad_case])
        self.assertEqual(report.build_status, "FAILED")
        self.assertEqual(report.failed_cases, 1)
        self.assertGreater(len(report.violations), 0)

    def test_empty_testset_benchmark(self):
        report = self.runner.run_benchmark([])
        self.assertEqual(report.build_status, "PASSED")
        self.assertEqual(report.total_cases, 0)

    def test_generate_markdown_summary_output(self):
        case = TestCase(
            test_id="TC-SUMMARY",
            query="Query",
            contexts=["Context"],
            generated_answer="Context",
            ground_truth="Context",
            query_vector=self.q_vec,
            synthetic_query_vectors=self.synth_good,
            explicit_relevance_labels=[1]
        )
        report = self.runner.run_benchmark([case])
        summary = self.runner.generate_markdown_summary(report)

        self.assertIn("Kết quả RAG CI/CD Quality Gate: ✅ PASSED", summary)
        self.assertIn("| **Faithfulness** |", summary)
        self.assertIn("| **Context Recall** |", summary)
        self.assertIn("| **Context Precision** |", summary)
        self.assertIn("| **Answer Relevance** |", summary)


if __name__ == "__main__":
    unittest.main()
