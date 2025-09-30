"""
Context managers for PETRE resource lifecycle management.

Provides context managers for heavy resources including TRI pipeline,
spaCy models, SHAP explainers, and memory management.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Any, Optional
import torch

try:
    from .import_utils import smart_import
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from import_utils import smart_import

if TYPE_CHECKING:
    from .config import AppConfig
    from transformers import Pipeline
    import spacy

# Import dependencies
config_imports = smart_import('config', ['AppConfig'])
interfaces_imports = smart_import('interfaces', ['ModelError'])

AppConfig = config_imports['AppConfig']
ModelError = interfaces_imports['ModelError']


class TRIPipelineContext:
    """Context manager for TRI (Target Re-Identification) pipeline."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline: Optional['Pipeline'] = None

    def __enter__(self) -> 'Pipeline':
        """Load and return the TRI pipeline."""
        try:
            from transformers import pipeline

            logging.info("Loading TRI pipeline from %s", self.config.tri_pipeline_path)

            # Determine device for pipeline
            device = None if self.config.device.type == 'cpu' else self.config.device

            self.pipeline = pipeline(
                "text-classification",
                model=self.config.tri_pipeline_path,
                tokenizer=self.config.tri_pipeline_path,
                device=device,
                top_k=None  # Will be set based on number of labels
            )

            logging.info("TRI pipeline loaded successfully on %s", self.config.device)
            return self.pipeline

        except Exception as e:
            raise ModelError(f"Failed to load TRI pipeline: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up pipeline resources."""
        if self.pipeline is not None:
            # Clear pipeline from memory
            del self.pipeline
            self.pipeline = None

            # Clear GPU cache if using GPU
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.config.device.type == 'mps':
                torch.mps.empty_cache()

            # Force garbage collection
            gc.collect()

            logging.info("TRI pipeline resources cleaned up")


class SpaCyModelContext:
    """Context manager for spaCy NLP model."""

    def __init__(self, model_name: str = "en_core_web_lg"):
        self.model_name = model_name
        self.nlp: Optional[Any] = None

    def __enter__(self) -> Any:
        """Load and return the spaCy model."""
        try:
            import en_core_web_lg

            logging.info("Loading spaCy model: %s", self.model_name)
            self.nlp = en_core_web_lg.load()
            logging.info("spaCy model loaded successfully")
            return self.nlp

        except ImportError as e:
            raise ModelError(f"spaCy model {self.model_name} not found. Install with: python -m spacy download en_core_web_lg") from e
        except Exception as e:
            raise ModelError(f"Failed to load spaCy model: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up spaCy model resources."""
        if self.nlp is not None:
            # Clear model from memory
            del self.nlp
            self.nlp = None

            # Force garbage collection
            gc.collect()

            logging.info("spaCy model resources cleaned up")


class SHAPExplainerContext:
    """Context manager for SHAP explainer."""

    def __init__(self, pipeline: 'Pipeline'):
        self.pipeline = pipeline
        self.explainer: Optional[Any] = None

    def __enter__(self) -> Any:
        """Create and return SHAP explainer."""
        try:
            import shap

            logging.info("Creating SHAP explainer")
            self.explainer = shap.Explainer(self.pipeline, silent=True)
            logging.info("SHAP explainer created successfully")
            return self.explainer

        except ImportError as e:
            raise ModelError("SHAP library not found. Install with: pip install shap") from e
        except Exception as e:
            raise ModelError(f"Failed to create SHAP explainer: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up SHAP explainer resources."""
        if self.explainer is not None:
            # Clear explainer from memory
            del self.explainer
            self.explainer = None

            # Force garbage collection
            gc.collect()

            logging.info("SHAP explainer resources cleaned up")


@contextmanager
def memory_management_context(gc_frequency: int = 10) -> Generator[int, None, None]:
    """
    Context manager for memory management during intensive operations.

    Args:
        gc_frequency: How often to run garbage collection

    Yields:
        Current iteration count for garbage collection decisions
    """
    iteration_count = 0

    try:
        while True:
            yield iteration_count
            iteration_count += 1

            # Run garbage collection periodically
            if iteration_count % gc_frequency == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, 'mps') and torch.mps.is_available():
                    torch.mps.empty_cache()

                logging.debug("Memory cleanup performed at iteration %d", iteration_count)

    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            torch.mps.empty_cache()

        logging.info("Final memory cleanup completed after %d iterations", iteration_count)


@contextmanager
def device_context(config: AppConfig) -> Generator[torch.device, None, None]:
    """
    Context manager for device management.

    Args:
        config: Application configuration containing device settings

    Yields:
        Configured torch device
    """
    device = config.device

    try:
        logging.info("Using device: %s", device)

        # Set appropriate device settings
        if device.type == 'cuda':
            # CUDA-specific settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        elif device.type == 'mps':
            # MPS-specific settings (if any)
            pass

        yield device

    finally:
        # Cleanup device resources
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        logging.info("Device context cleaned up")


class ModelResourceContext:
    """
    Comprehensive context manager for all model resources.

    Manages TRI pipeline, spaCy model, and optionally SHAP explainer
    in a single context for coordinated resource management.
    """

    def __init__(self, config: AppConfig, load_shap: bool = False):
        self.config = config
        self.load_shap = load_shap
        self.tri_pipeline: Optional['Pipeline'] = None
        self.spacy_model: Optional[Any] = None
        self.shap_explainer: Optional[Any] = None

        # Context managers
        self.tri_context = TRIPipelineContext(config)
        self.spacy_context = SpaCyModelContext()
        self.shap_context: Optional[SHAPExplainerContext] = None

    def __enter__(self) -> dict[str, Any]:
        """Load all required models and return them in a dictionary."""
        try:
            # Load TRI pipeline first
            self.tri_pipeline = self.tri_context.__enter__()

            # Load spaCy model
            self.spacy_model = self.spacy_context.__enter__()

            # Optionally load SHAP explainer
            if self.load_shap:
                self.shap_context = SHAPExplainerContext(self.tri_pipeline)
                self.shap_explainer = self.shap_context.__enter__()

            resources = {
                'tri_pipeline': self.tri_pipeline,
                'spacy_model': self.spacy_model,
                'shap_explainer': self.shap_explainer
            }

            logging.info("All model resources loaded successfully")
            return resources

        except Exception as e:
            # Cleanup any partially loaded resources
            self.__exit__(type(e), e, e.__traceback__)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up all model resources."""
        # Clean up in reverse order
        if self.shap_context is not None:
            self.shap_context.__exit__(exc_type, exc_val, exc_tb)

        if self.spacy_context is not None:
            self.spacy_context.__exit__(exc_type, exc_val, exc_tb)

        if self.tri_context is not None:
            self.tri_context.__exit__(exc_type, exc_val, exc_tb)

        logging.info("All model resources cleaned up")


# Convenience functions for creating common contexts
def create_tri_pipeline_context(config: AppConfig) -> TRIPipelineContext:
    """Create TRI pipeline context from configuration."""
    return TRIPipelineContext(config)


def create_spacy_context(model_name: str = "en_core_web_lg") -> SpaCyModelContext:
    """Create spaCy model context."""
    return SpaCyModelContext(model_name)


def create_model_resource_context(config: AppConfig, load_shap: bool = False) -> ModelResourceContext:
    """Create comprehensive model resource context."""
    return ModelResourceContext(config, load_shap)


__all__ = [
    'TRIPipelineContext',
    'SpaCyModelContext',
    'SHAPExplainerContext',
    'ModelResourceContext',
    'memory_management_context',
    'device_context',
    'create_tri_pipeline_context',
    'create_spacy_context',
    'create_model_resource_context'
]