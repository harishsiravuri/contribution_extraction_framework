"""Alternative pipelines (baselines, ablations) that share the Pipeline interface."""

from paper1.pipelines.baseline import BaselinePipeline
from paper1.pipelines.few_shot import make_few_shot_pipeline
from paper1.pipelines.no_critic import NoCriticPipeline

__all__ = ["BaselinePipeline", "NoCriticPipeline", "make_few_shot_pipeline"]
