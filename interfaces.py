"""
Interface definitions for PETRE (Privacy-preserving Entities Transparency and Re-identification Evaluation).

All abstract base classes define contracts for the main components:
- Data processing and dataset management
- Model and pipeline management  
- Explainability methods (SHAP, Greedy)
- Evaluation and ranking
- Anonymization engine
- Main PETRE orchestrator
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .config import AppConfig
    from transformers import Pipeline


class DataProcessor(ABC):
    """Abstract interface for data processing and dataset creation."""
    
    @abstractmethod
    def load_data(self, data_path: str, individual_column: str, text_column: str) -> pd.DataFrame:
        """Load and validate dataset from file."""
        pass
    
    @abstractmethod
    def create_label_mappings(self, df: pd.DataFrame, individual_column: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Create bidirectional mappings between names and labels."""
        pass
    
    @abstractmethod
    def create_dataset(self, df: pd.DataFrame, name_to_label: Dict[str, int]) -> 'PETREDataset':
        """Create PETRE dataset with sentence splitting and tokenization."""
        pass


class ModelManager(ABC):
    """Abstract interface for model and pipeline management."""
    
    @abstractmethod
    def load_tri_pipeline(self, model_path: str, num_labels: int) -> 'Pipeline':
        """Load and configure TRI (Target Re-Identification) pipeline."""
        pass
    
    @abstractmethod
    def load_spacy_model(self) -> Any:
        """Load spaCy NLP model for text processing."""
        pass
    
    @abstractmethod
    def get_device(self) -> torch.device:
        """Get the appropriate device (CUDA/MPS/CPU) for model inference."""
        pass


class ExplainabilityMethod(ABC):
    """Abstract interface for explainability methods."""
    
    @abstractmethod
    def explain(self, text: str, label: int, split: Dict[str, Any], plot: bool = False) -> np.ndarray:
        """
        Compute term weights for explainability.
        
        Args:
            text: Input text to explain
            label: Target label for explanation
            split: Split information with terms and tokens
            plot: Whether to plot explanations
            
        Returns:
            Array of term weights
        """
        pass
    
    @abstractmethod
    def setup(self, pipeline: 'Pipeline') -> None:
        """Setup the explainability method with the given pipeline."""
        pass


class PipelineEvaluator(ABC):
    """Abstract interface for pipeline evaluation and ranking."""
    
    @abstractmethod
    def evaluate(self, dataset: 'PETREDataset', max_rank: int = 1, 
                use_annotated: bool = True, batch_size: int = 128) -> Tuple[float, np.ndarray, List[torch.Tensor]]:
        """
        Evaluate re-identification risk across the dataset.
        
        Returns:
            - Accuracy (proportion with rank <= max_rank)
            - Ranks for each document
            - Probability distributions for each document
        """
        pass
    
    @abstractmethod
    def evaluate_document(self, document: Dict[str, Any], use_annotated: bool = True) -> Tuple[torch.Tensor, int]:
        """Evaluate a single document and return probabilities and rank."""
        pass
    
    @abstractmethod
    def get_document_rank(self, splits_probs: torch.Tensor, label: int) -> Tuple[int, float]:
        """Compute rank and probability for aggregated split probabilities."""
        pass


class AnonymizationEngine(ABC):
    """Abstract interface for the anonymization process."""
    
    @abstractmethod
    def run_anonymization(self, k: int, dataset: 'PETREDataset', plot_explanations: bool = False) -> Tuple[Dict[str, Any], int]:
        """
        Run k-anonymity protection process.
        
        Args:
            k: Anonymity parameter (target minimum rank)
            dataset: Dataset to anonymize
            plot_explanations: Whether to show explanation plots
            
        Returns:
            - Final annotations
            - Total number of steps taken
        """
        pass
    
    @abstractmethod
    def mask_most_disclosive_term(self, document: Dict[str, Any], splits_probs: torch.Tensor) -> Tuple[Optional[str], int, int]:
        """
        Find and mask the most disclosive term in a document.
        
        Returns:
            - Most disclosive term text (None if no terms to mask)
            - Number of terms masked
            - Number of evaluation steps taken
        """
        pass


class PETREOrchestrator(ABC):
    """Abstract interface for the main PETRE orchestrator."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize all components and load data."""
        pass
    
    @abstractmethod
    def run_incremental_execution(self, k_values: List[int]) -> None:
        """Run the incremental anonymization process for all k values."""
        pass
    
    @abstractmethod
    def save_results(self, annotations: Dict[str, Any], k: Optional[int] = None) -> None:
        """Save annotations and results to output directory."""
        pass


class PETREDataset(Dataset, ABC):
    """Abstract interface for PETRE dataset with annotation capabilities."""
    
    @abstractmethod
    def add_annotations(self, annotations: Dict[str, List[List[int]]], disable_tqdm: bool = True) -> None:
        """Add annotations to the dataset."""
        pass
    
    @abstractmethod
    def get_annotations(self, disable_tqdm: bool = True) -> Dict[str, List[List[int]]]:
        """Get current annotations from the dataset."""
        pass
    
    @abstractmethod
    def mask_terms(self, split: Dict[str, Any], terms_to_mask: List[int]) -> None:
        """Mask specified terms in a split."""
        pass
    
    @abstractmethod
    def annotate_text(self, text: str, split: Dict[str, Any]) -> str:
        """Apply annotations to text based on split information."""
        pass
    
    @abstractmethod
    def get_all_texts(self, use_annotated: bool) -> Tuple[List[str], Dict[int, List[int]]]:
        """Get all texts for pipeline evaluation."""
        pass


# Factory interface for creating components
class ComponentFactory(ABC):
    """Abstract factory for creating PETRE components."""
    
    @abstractmethod
    def create_data_processor(self, config: 'AppConfig') -> DataProcessor:
        """Create data processor instance."""
        pass
    
    @abstractmethod
    def create_model_manager(self, config: 'AppConfig') -> ModelManager:
        """Create model manager instance."""
        pass
    
    @abstractmethod
    def create_explainability_method(self, config: 'AppConfig') -> ExplainabilityMethod:
        """Create explainability method instance."""
        pass
    
    @abstractmethod
    def create_evaluator(self, config: 'AppConfig') -> PipelineEvaluator:
        """Create pipeline evaluator instance."""
        pass
    
    @abstractmethod
    def create_anonymizer(self, config: 'AppConfig') -> AnonymizationEngine:
        """Create anonymization engine instance."""
        pass
    
    @abstractmethod
    def create_orchestrator(self, config: 'AppConfig') -> PETREOrchestrator:
        """Create main PETRE orchestrator instance."""
        pass


# Error types for better error handling
class PETREError(Exception):
    """Base exception for PETRE-related errors."""
    pass


class ConfigurationError(PETREError):
    """Raised when there are configuration-related issues."""
    pass


class DataProcessingError(PETREError):
    """Raised when there are data processing issues."""
    pass


class ModelError(PETREError):
    """Raised when there are model-related issues."""
    pass


class EvaluationError(PETREError):
    """Raised when there are evaluation-related issues."""
    pass


class AnonymizationError(PETREError):
    """Raised when there are anonymization-related issues."""
    pass


__all__ = [
    # Core interfaces
    'DataProcessor',
    'ModelManager', 
    'ExplainabilityMethod',
    'PipelineEvaluator',
    'AnonymizationEngine',
    'PETREOrchestrator',
    'PETREDataset',
    'ComponentFactory',
    # Exceptions
    'PETREError',
    'ConfigurationError',
    'DataProcessingError',
    'ModelError',
    'EvaluationError',
    'AnonymizationError'
]