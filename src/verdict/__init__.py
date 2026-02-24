"""
Verdict - Production-grade LLMOps Evaluation Framework on Databricks.
"""

__version__ = "0.1.0"

from verdict.setup.init_catalog import VerdictCatalogSetup
from verdict.data.prompt_dataset import PromptDatasetManager
from verdict.inference.inference_runner import InferenceRunner
from verdict.evaluation.deterministic_metrics import DeterministicMetricsCalculator
from verdict.evaluation.mlflow_evaluator import MLflowEvaluator
from verdict.evaluation.custom_judges import LLMJudge, LLMJudgeEvaluator
from verdict.regression.regression_detector import RegressionDetector, VerdictLabel

__all__ = [
    "__version__",
    "VerdictCatalogSetup",
    "PromptDatasetManager",
    "InferenceRunner",
    "DeterministicMetricsCalculator",
    "MLflowEvaluator",
    "LLMJudge",
    "LLMJudgeEvaluator",
    "RegressionDetector",
    "VerdictLabel",
]