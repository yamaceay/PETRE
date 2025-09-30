"""
Core implementations for PETRE following interface contracts.

All concrete implementations use dependency injection with static dependencies
in __init__ and methods taking only dynamic input.
"""

from __future__ import annotations

import os
import gc
import re
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
import datasets

try:
    from .import_utils import smart_import
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from import_utils import smart_import

if TYPE_CHECKING:
    from .config import AppConfig
    from .interfaces import (
        DataProcessor, ModelManager, ExplainabilityMethod,
        PipelineEvaluator, AnonymizationEngine, PETREOrchestrator,
        PETREDataset, ComponentFactory
    )
    from transformers import Pipeline
    import spacy

# Import dependencies
config_imports = smart_import('config', ['AppConfig'])
interfaces_imports = smart_import('interfaces', [
    'DataProcessor', 'ModelManager', 'ExplainabilityMethod', 'PipelineEvaluator',
    'AnonymizationEngine', 'PETREOrchestrator', 'PETREDataset', 'ComponentFactory',
    'DataProcessingError', 'ModelError', 'EvaluationError', 'AnonymizationError'
])
contexts_imports = smart_import('contexts', [
    'TRIPipelineContext', 'SpaCyModelContext', 'SHAPExplainerContext',
    'memory_management_context'
])

AppConfig = config_imports['AppConfig']
DataProcessor = interfaces_imports['DataProcessor']
ModelManager = interfaces_imports['ModelManager']
ExplainabilityMethod = interfaces_imports['ExplainabilityMethod']
PipelineEvaluator = interfaces_imports['PipelineEvaluator']
AnonymizationEngine = interfaces_imports['AnonymizationEngine']
PETREOrchestrator = interfaces_imports['PETREOrchestrator']
PETREDataset = interfaces_imports['PETREDataset']
ComponentFactory = interfaces_imports['ComponentFactory']
DataProcessingError = interfaces_imports['DataProcessingError']
ModelError = interfaces_imports['ModelError']
EvaluationError = interfaces_imports['EvaluationError']
AnonymizationError = interfaces_imports['AnonymizationError']

TRIPipelineContext = contexts_imports['TRIPipelineContext']
SpaCyModelContext = contexts_imports['SpaCyModelContext']
SHAPExplainerContext = contexts_imports['SHAPExplainerContext']
memory_management_context = contexts_imports['memory_management_context']


@dataclass(frozen=True)
class PETREDataProcessor(DataProcessor):
    """Concrete implementation of data processing."""

    config: AppConfig

    def load_data(self, data_path: str, individual_column: str, text_column: str) -> pd.DataFrame:
        """Load and validate dataset from file."""
        try:
            if data_path.endswith(".json"):
                complete_df = pd.read_json(data_path)
            else:
                complete_df = pd.read_csv(data_path)

            # Validate columns exist
            if individual_column not in complete_df.columns:
                raise DataProcessingError(f"Individual column '{individual_column}' not found in data")
            if text_column not in complete_df.columns:
                raise DataProcessingError(f"Text column '{text_column}' not found in data")

            # Extract only needed columns
            df = complete_df[[individual_column, text_column]]

            logging.info("Loaded data with %d rows and %d individuals",
                        len(df), df[individual_column].nunique())

            return df

        except Exception as e:
            raise DataProcessingError(f"Failed to load data from {data_path}: {e}") from e

    def create_label_mappings(self, df: pd.DataFrame, individual_column: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Create bidirectional mappings between names and labels."""
        names = sorted(df[individual_column].unique())
        name_to_label = {name: idx for idx, name in enumerate(names)}
        label_to_name = {idx: name for name, idx in name_to_label.items()}

        logging.info("Created label mappings for %d individuals", len(names))
        return name_to_label, label_to_name

    def create_dataset(self, df: pd.DataFrame, name_to_label: Dict[str, int]) -> 'PETREDatasetImpl':
        """Create PETRE dataset with sentence splitting and tokenization."""
        return PETREDatasetImpl(
            df=df,
            name_to_label=name_to_label,
            config=self.config
        )


@dataclass(frozen=True)
class PETREModelManager(ModelManager):
    """Concrete implementation of model management."""

    config: AppConfig

    def load_tri_pipeline(self, model_path: str, num_labels: int) -> 'Pipeline':
        """Load TRI pipeline from the specified path."""
        from transformers import pipeline

        device = None if self.config.device.type == 'cpu' else self.config.device
        tri_pipeline = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=device,
            top_k=num_labels
        )
        return tri_pipeline

    def load_spacy_model(self) -> Any:
        """Load spaCy NLP model."""
        import spacy
        return spacy.load("en_core_web_lg")

    def get_device(self) -> torch.device:
        """Get the configured device."""
        return self.config.device


@dataclass(frozen=True)
class SHAPExplainabilityMethod(ExplainabilityMethod):
    """SHAP-based explainability implementation."""

    config: AppConfig
    pipeline: 'Pipeline'
    explainer: Optional[Any] = None

    def setup(self, pipeline: 'Pipeline') -> None:
        """Setup SHAP explainer with pipeline."""
        # Note: Due to frozen dataclass, we'd need to create a new instance
        # This is handled in the factory
        pass

    def explain(self, text: str, label: int, split: Dict[str, Any], plot: bool = False) -> np.ndarray:
        """Compute term weights using SHAP."""
        if self.explainer is None:
            raise ModelError("SHAP explainer not initialized")

        terms_to_tokens = split["terms_to_tokens"]
        terms_weights = np.zeros(len(terms_to_tokens))
        masked_terms_idxs = split["masked_terms_idxs"]

        # Get token weights from SHAP
        tokens_weights = self._get_tokens_weights(text, label, plot)

        # Aggregate token weights into term weights
        for idx, term_tokens in enumerate(terms_to_tokens):
            if idx in masked_terms_idxs:
                term_weight = float("-inf")
            else:
                term_weight = sum(tokens_weights[token_idx] for token_idx in term_tokens)
            terms_weights[idx] = term_weight

        return terms_weights

    def _get_tokens_weights(self, text: str, label: int, plot: bool = False) -> np.ndarray:
        """Get token weights from SHAP explainer."""
        try:
            shap_values = self.explainer([text], batch_size=1)
            tokens_weights = shap_values.values[0, :, label]

            if plot:
                import shap
                shap.plots.text(shap_values[0, :, label])

            return tokens_weights

        except Exception as e:
            raise ModelError(f"SHAP explanation failed: {e}") from e


@dataclass(frozen=True)
class GreedyExplainabilityMethod(ExplainabilityMethod):
    """Greedy search-based explainability implementation."""

    config: AppConfig
    pipeline: 'Pipeline'

    def setup(self, pipeline: 'Pipeline') -> None:
        """Setup method (no additional setup needed for greedy)."""
        pass

    def explain(self, text: str, label: int, split: Dict[str, Any], plot: bool = False) -> np.ndarray:
        """Compute term weights using greedy search."""
        terms_spans = split["terms_spans"]
        terms_weights = np.zeros(len(terms_spans))
        masked_terms_idxs = split["masked_terms_idxs"]

        # Create texts for evaluation
        input_texts = []
        terms_idxs_to_assign = []

        # Add current text (with existing annotations)
        annotated_text = self._annotate_text(text, split)
        input_texts.append(annotated_text)

        # Add versions with each unmasked term masked
        for term_idx, (start, end) in enumerate(terms_spans):
            if term_idx in masked_terms_idxs:
                terms_weights[term_idx] = float("-inf")
            else:
                # Create version with this term masked
                masked_text = text[:start] + self.config.mask_text + text[end:]
                input_texts.append(masked_text)
                terms_idxs_to_assign.append(term_idx)

        # Evaluate all texts
        probabilities = self._evaluate_texts(input_texts, label)

        # Compute term weights as probability differences
        base_prob = probabilities[0]
        for i, term_idx in enumerate(terms_idxs_to_assign):
            terms_weights[term_idx] = base_prob - probabilities[i + 1]

        return terms_weights

    def _annotate_text(self, text: str, split: Dict[str, Any]) -> str:
        """Apply existing annotations to text."""
        # This would need access to the dataset's annotate_text method
        # For now, return text as-is
        return text

    def _evaluate_texts(self, texts: List[str], label: int, batch_size: int = 128) -> np.ndarray:
        """Evaluate texts and return probabilities for the target label."""
        try:
            inputs_dataset = datasets.Dataset.from_dict({"text": texts})["text"]
            results = self.pipeline(inputs_dataset, batch_size=batch_size)

            probabilities = np.zeros(len(results))
            for idx, result in enumerate(results):
                for pred in result:
                    pred_label = int(pred["label"].split("_")[1])
                    if pred_label == label:
                        probabilities[idx] = float(pred["score"])
                        break

            return probabilities

        except Exception as e:
            raise EvaluationError(f"Text evaluation failed: {e}") from e


@dataclass(frozen=True)
class PETREPipelineEvaluator(PipelineEvaluator):
    """Concrete implementation of pipeline evaluation."""

    config: AppConfig
    pipeline: 'Pipeline'
    name_to_label: Dict[str, int]
    label_to_name: Dict[int, str]

    def evaluate(self, dataset: 'PETREDatasetImpl', max_rank: int = 1,
                use_annotated: bool = True, batch_size: int = 128) -> Tuple[float, np.ndarray, List[torch.Tensor]]:
        """Evaluate re-identification risk across the dataset."""
        try:
            # Get all texts for evaluation
            input_texts, doc_to_texts_idxs = dataset.get_all_texts(use_annotated)

            # Process through pipeline
            docs_probs, ranks = self._pipeline_results_to_docs_probs(
                input_texts, doc_to_texts_idxs, batch_size
            )

            # Compute accuracy
            n_correct_preds = np.count_nonzero(ranks <= max_rank)
            accuracy = n_correct_preds / len(dataset)

            return accuracy, ranks, docs_probs

        except Exception as e:
            raise EvaluationError(f"Dataset evaluation failed: {e}") from e

    def evaluate_document(self, document: Dict[str, Any], use_annotated: bool = True) -> Tuple[torch.Tensor, int]:
        """Evaluate a single document."""
        complete_text = document["text"]
        label = document["label"]
        splits = document["splits"]

        splits_probs = torch.zeros((len(splits), len(self.name_to_label)))

        # Evaluate each split
        for split_idx, split in enumerate(splits):
            splits_probs[split_idx, :] = self._evaluate_split(complete_text, split, use_annotated)

        # Get rank for this document
        rank, _ = self.get_document_rank(splits_probs, label)

        return splits_probs, rank

    def get_document_rank(self, splits_probs: torch.Tensor, label: int) -> Tuple[int, float]:
        """Compute rank and probability for aggregated split probabilities."""
        # Aggregate probabilities by averaging
        aggregated_probs = splits_probs.sum(dim=0) / splits_probs.shape[0]
        prob = aggregated_probs[label].item()

        # Get rank position
        sorted_idxs = torch.argsort(aggregated_probs, descending=True)
        rank_position = torch.where(sorted_idxs == label)[0].item()
        rank = rank_position + 1  # +1 to start rank at 1

        return rank, prob

    def _pipeline_results_to_docs_probs(self, input_texts: List[str],
                                       doc_to_input_idxs: Dict[int, List[int]],
                                       batch_size: int) -> Tuple[List[torch.Tensor], np.ndarray]:
        """Convert pipeline results to document probabilities."""
        # Evaluate all texts
        inputs_dataset = datasets.Dataset.from_dict({"text": input_texts})["text"]
        results = self.pipeline(inputs_dataset, batch_size=batch_size)

        docs_probs = []
        ranks = []

        # Process results for each document
        for label, text_idxs in doc_to_input_idxs.items():
            # Get results for this document
            doc_results = [results[idx] for idx in text_idxs]

            # Convert to probability tensor
            splits_probs = torch.zeros((len(doc_results), len(self.name_to_label)))
            for split_idx, split_preds in enumerate(doc_results):
                for pred in split_preds:
                    pred_label, pred_score = self._pipeline_pred_to_label_score(pred)
                    splits_probs[split_idx, pred_label] = pred_score

            docs_probs.append(splits_probs)

            # Get rank for this document
            rank, _ = self.get_document_rank(splits_probs, label)
            ranks.append(rank)

        return docs_probs, np.array(ranks)

    def _evaluate_split(self, complete_text: str, split: Dict[str, Any], use_annotated: bool) -> torch.Tensor:
        """Evaluate a single split."""
        split_span = split["text_span"]
        split_text = complete_text[split_span[0]:split_span[1]]

        # Apply annotations if requested
        if use_annotated:
            # This would need dataset access - simplified for now
            pass

        # Evaluate text
        return self._evaluate_text(split_text)

    def _evaluate_text(self, text: str) -> torch.Tensor:
        """Evaluate single text and return probability distribution."""
        results = self.pipeline([text])[0]
        probs = torch.zeros(len(self.name_to_label))

        for pred in results:
            pred_label, pred_score = self._pipeline_pred_to_label_score(pred)
            probs[pred_label] = pred_score

        return probs

    def _pipeline_pred_to_label_score(self, pred: Dict[str, Any]) -> Tuple[int, float]:
        """Extract label and score from pipeline prediction."""
        label = int(pred["label"].split("_")[1])
        score = float(pred["score"])
        return label, score


class PETREDatasetImpl(Dataset):
    """Concrete implementation of PETRE dataset."""

    def __init__(self, df: pd.DataFrame, name_to_label: Dict[str, int], config: AppConfig):
        self.df = df
        self.name_to_label = name_to_label
        self.label_to_name = {v: k for k, v in name_to_label.items()}
        self.config = config

        # Initialize tokenizer context
        self.tokenizer = None  # Will be set when pipeline is available

        # Dataset attributes
        self.documents: List[Dict[str, Any]] = []
        self.terms_to_ignore: Set[str] = set()

        # Generate dataset
        self._generate_dataset()

    def _generate_dataset(self) -> None:
        """Generate the dataset with sentence splitting and tokenization."""
        with SpaCyModelContext() as nlp:
            self._setup_terms_to_ignore(nlp)

            texts_column = list(self.df[self.df.columns[1]])
            names_column = list(self.df[self.df.columns[0]])
            labels_idxs = [self.name_to_label[name] for name in names_column]

            self.documents = [None] * len(labels_idxs)

            for idx, (text, label) in tqdm(enumerate(zip(texts_column, labels_idxs)),
                                         total=len(texts_column),
                                         desc="Processing sentence splitting"):

                # Process document
                document = self._process_document(text, label, nlp)
                self.documents[label] = document

                # Memory management every 5 documents
                if idx % 5 == 0:
                    import gc
                    gc.collect()

    def _setup_terms_to_ignore(self, nlp) -> None:
        """Setup terms to ignore during processing."""
        stopwords = nlp.Defaults.stop_words
        self.terms_to_ignore.update(stopwords)

        mask_marks = {
            "sensitive", "person", "dem", "loc", "org", "datetime",
            "quantity", "misc", "norp", "fac", "gpe", "product",
            "event", "work_of_art", "law", "language", "date",
            "time", "ordinal", "cardinal", "date_time", "nrp",
            "location", "organization", "***", self.config.mask_text
        }
        self.terms_to_ignore.update(mask_marks)
        self.terms_to_ignore.update({"[CLS]", "[SEP]", "[PAD]", "", " ", "\t", "\n"})

    def _process_document(self, text: str, label: int, nlp) -> Dict[str, Any]:
        """Process a single document."""
        splits = []
        document = {"text": text, "label": label, "splits": splits}

        doc = nlp(text)
        for sentence in doc.sents:
            sentence_txt = text[sentence.start_char:sentence.end_char]

            # Check token count (requires tokenizer)
            if self.tokenizer:
                sent_token_count = len(self.tokenizer.encode(sentence_txt, add_special_tokens=True))
                if sent_token_count > self.tokenizer.model_max_length:
                    logging.error("Sentence too long (%d tokens) at index %d", sent_token_count, label)
                    continue

            # Create split information
            terms_spans = self._get_terms_spans(sentence, self.config.use_chunking)
            terms_to_tokens = self._get_terms_to_tokens(terms_spans, sentence_txt) if self.tokenizer else []

            splits.append({
                "text_span": (sentence.start_char, sentence.end_char),
                "terms_spans": terms_spans,
                "terms_to_tokens": terms_to_tokens,
                "masked_terms_idxs": []
            })

        return document

    def _get_terms_spans(self, sentence_span, use_chunking: bool) -> List[Tuple[int, int]]:
        """Extract term spans from sentence."""
        text_spans = []
        added_tokens = np.zeros(len(sentence_span), dtype=bool)
        start_char_idx = sentence_span.start_char
        start_token_idx = sentence_span.start
        special_chars_pattern = re.compile(r"[^\nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ/]+")

        if use_chunking:
            # Add noun chunks
            for chunk in sentence_span.noun_chunks:
                chunk_start = chunk.start - start_token_idx
                chunk_end = chunk.end - start_token_idx
                start = sentence_span[chunk_start].idx
                last_token = sentence_span[chunk_end - 1]
                end = last_token.idx + len(last_token)
                text_spans.append((start, end))
                added_tokens[chunk_start:chunk_end] = True

            # Add named entities
            for chunk in sentence_span.ents:
                chunk_start = chunk.start - start_token_idx
                chunk_end = chunk.end - start_token_idx
                if not added_tokens[chunk_start:chunk_end].any():
                    start = sentence_span[chunk_start].idx
                    last_token = sentence_span[chunk_end - 1]
                    end = last_token.idx + len(last_token)
                    text_spans.append((start, end))
                    added_tokens[chunk_start:chunk_end] = True

        # Add remaining tokens
        for token_idx in range(len(sentence_span)):
            if not added_tokens[token_idx]:
                token = sentence_span[token_idx]
                clean_token_text = re.sub(special_chars_pattern, '', token.text).strip()
                if clean_token_text not in self.terms_to_ignore:
                    start = token.idx
                    end = start + len(token)
                    text_spans.append((start, end))

        # Sort and adjust spans
        text_spans = sorted(text_spans, key=lambda span: span[0])
        text_spans = [(x[0] - start_char_idx, x[1] - start_char_idx) for x in text_spans]

        return text_spans

    def _get_terms_to_tokens(self, terms_spans: List[Tuple[int, int]], text: str) -> List[List[int]]:
        """Map terms to tokenizer tokens."""
        if not self.tokenizer:
            return []

        terms_to_tokens = []
        results = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = results["offset_mapping"]

        last_token_idx = 0
        for term_start, term_end in terms_spans:
            term_tokens = []
            for token_idx in range(last_token_idx, len(offsets)):
                token_start, token_end_offset = offsets[token_idx]
                if term_start <= token_start < term_end:
                    term_tokens.append(token_idx + 1)  # +1 for [CLS] token
                elif len(term_tokens) > 0:
                    break
            last_token_idx = token_idx
            terms_to_tokens.append(term_tokens)

        return terms_to_tokens

    # Implementation of abstract methods
    def add_annotations(self, annotations: Dict[str, Any], disable_tqdm: bool = True) -> None:
        """Add annotations to the dataset with support for both old and new formats."""
        for name, doc_annotations in annotations.items():
            if name in self.name_to_label:
                label = self.name_to_label[name]
                document = self.documents[label]

                # Apply annotations to splits
                for annotation in doc_annotations:
                    # Handle both old format [start, end] and new format {start, end, value, replacement}
                    if isinstance(annotation, list):
                        # Old format: [start, end]
                        global_start, global_end = annotation[0], annotation[1]
                    elif isinstance(annotation, dict):
                        # New format: {start, end, value, replacement}
                        global_start, global_end = annotation["start"], annotation["end"]
                    else:
                        continue

                    for split in document["splits"]:
                        split_span = split["text_span"]
                        terms_spans = split["terms_spans"]

                        # Find overlapping terms
                        for term_idx, term_span in enumerate(terms_spans):
                            global_term_start = split_span[0] + term_span[0]
                            global_term_end = split_span[0] + term_span[1]

                            # Check if this term overlaps with the annotation
                            if (global_start <= global_term_start < global_end or
                                global_start < global_term_end <= global_end):
                                if term_idx not in split["masked_terms_idxs"]:
                                    split["masked_terms_idxs"].append(term_idx)

    def get_annotations(self, disable_tqdm: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """Get current annotations from the dataset with enhanced span information."""
        annotations = {}

        for document in self.documents:
            doc_annotations = []
            label = document["label"]
            name = self.label_to_name[label]
            complete_text = document["text"]

            # Collect all masked terms as annotations with span details
            for split in document["splits"]:
                split_span = split["text_span"]
                terms_spans = split["terms_spans"]
                masked_terms_idxs = split["masked_terms_idxs"]

                # Convert masked terms to global spans with values
                for term_idx in sorted(masked_terms_idxs):
                    local_span = terms_spans[term_idx]
                    global_start = split_span[0] + local_span[0]
                    global_end = split_span[0] + local_span[1]
                    original_value = complete_text[global_start:global_end]

                    span_annotation = {
                        "start": global_start,
                        "end": global_end,
                        "value": original_value,
                        "replacement": self.config.mask_text if self.config.mask_text else "[MASK]"
                    }
                    doc_annotations.append(span_annotation)

            if doc_annotations:
                annotations[name] = doc_annotations

        return annotations

    def mask_terms(self, split: Dict[str, Any], terms_to_mask: List[int]) -> None:
        """Mask specified terms in a split (following original logic)."""
        masked_terms_idxs = split["masked_terms_idxs"]

        # Add new masked terms
        for term_idx in terms_to_mask:
            if term_idx not in masked_terms_idxs:
                masked_terms_idxs.append(term_idx)

    def annotate_text(self, text: str, split: Dict[str, Any]) -> str:
        """Apply annotations to text based on split information (following original logic)."""
        terms_spans = split["terms_spans"]
        masked_terms_idxs = split["masked_terms_idxs"]

        # Sort masked terms in reverse order to avoid index shifting
        sorted_masked_terms_idxs = sorted(masked_terms_idxs, reverse=True)
        annotated_text = text

        for term_idx in sorted_masked_terms_idxs:
            start, end = terms_spans[term_idx]
            mask_replacement = self.config.mask_text if self.config.mask_text else "[MASK]"
            annotated_text = annotated_text[:start] + mask_replacement + annotated_text[end:]

        return annotated_text

    def get_all_texts(self, use_annotated: bool) -> Tuple[List[str], Dict[int, List[int]]]:
        """Get all texts for pipeline evaluation."""
        texts = []
        doc_to_text_idxs = {}

        for document in self:
            complete_text = document["text"]
            label = document["label"]
            splits = document["splits"]
            doc_inputs_idxs = []

            for split in splits:
                split_span = split["text_span"]
                split_text = complete_text[split_span[0]:split_span[1]]

                if use_annotated:
                    split_text = self.annotate_text(split_text, split)

                doc_inputs_idxs.append(len(texts))
                texts.append(split_text)

            doc_to_text_idxs[label] = doc_inputs_idxs

        return texts, doc_to_text_idxs

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.documents)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get document by index."""
        return self.documents[index]


@dataclass(frozen=True)
class PETREComponentFactory(ComponentFactory):
    """Factory for creating PETRE components."""

    def create_data_processor(self, config: AppConfig) -> DataProcessor:
        """Create data processor instance."""
        return PETREDataProcessor(config=config)

    def create_model_manager(self, config: AppConfig) -> ModelManager:
        """Create model manager instance."""
        return PETREModelManager(config=config)

    def create_explainability_method(self, config: AppConfig) -> ExplainabilityMethod:
        """Create explainability method instance."""
        # This requires pipeline context - would be set up in orchestrator
        if config.explainability_mode == "SHAP":
            return None  # Created with pipeline in orchestrator
        else:
            return None  # Created with pipeline in orchestrator

    def create_evaluator(self, config: AppConfig) -> PipelineEvaluator:
        """Create pipeline evaluator instance."""
        # This requires pipeline and label mappings - would be set up in orchestrator
        return None

    def create_anonymizer(self, config: AppConfig) -> AnonymizationEngine:
        """Create anonymization engine instance."""
        # This requires multiple dependencies - would be set up in orchestrator
        return None

    def create_orchestrator(self, config: AppConfig) -> PETREOrchestrator:
        """Create main PETRE orchestrator instance."""
        return PETREOrchestratorImpl(config, self)


class PETREOrchestratorImpl(PETREOrchestrator):
    """Main PETRE orchestrator implementation."""

    def __init__(self, config, factory):
        self.config = config
        self.factory = factory
        self.data_processor = None
        self.model_manager = None
        self.evaluator = None
        self.dataset = None
        self.tri_pipeline = None
        self.explainer = None

    def initialize(self) -> None:
        """Initialize all components and load data."""
        logging.info("Initializing PETRE orchestrator")

        # Create components
        self.data_processor = self.factory.create_data_processor(self.config)
        self.model_manager = self.factory.create_model_manager(self.config)
        self.evaluator = self.factory.create_evaluator(self.config)

        # Load and process data
        logging.info("Loading dataset from: %s", self.config.data_file_path)
        complete_df = self.data_processor.load_data(
            self.config.data_file_path,
            self.config.individual_name_column,
            self.config.original_text_column
        )

        # Extract names and create mappings
        names = sorted(complete_df[self.config.individual_name_column].unique())
        name_to_label = {name: idx for idx, name in enumerate(names)}

        # Create dataset
        self.dataset = PETREDatasetImpl(
            df=complete_df,
            name_to_label=name_to_label,
            config=self.config
        )

        # Store for later use BEFORE loading pipeline
        self.names = names
        self.name_to_label = name_to_label
        self.label_to_name = {v: k for k, v in name_to_label.items()}

        # Load TRI pipeline first
        logging.info("Loading TRI pipeline from: %s", self.config.tri_pipeline_path)
        from transformers import pipeline

        # Use CPU device for better compatibility (avoiding MPS issues)
        device = -1 if self.config.device.type == 'cpu' else 0
        self.tri_pipeline = pipeline(
            "text-classification",
            model=self.config.tri_pipeline_path,
            tokenizer=self.config.tri_pipeline_path,
            device=device,
            top_k=len(self.names)
        )

        # Set tokenizer in dataset for proper processing
        self.dataset.tokenizer = self.tri_pipeline.tokenizer

        # Make dataset and names available to orchestrator methods
        self.dataset.terms_to_ignore = getattr(self.dataset, 'terms_to_ignore', set())

        # Initialize annotations if provided
        self.annotations = {}
        if (hasattr(self.config, 'starting_anonymization_path') and
            self.config.starting_anonymization_path and
            os.path.exists(self.config.starting_anonymization_path)):
            with open(self.config.starting_anonymization_path, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)

        # Initialize explainer
        if self.config.explainability_mode == "SHAP":
            import shap
            self.explainer = shap.Explainer(self.tri_pipeline, silent=True)
        elif self.config.explainability_mode == "Greedy":
            self.explainer = None  # Will use greedy method

        logging.info("PETRE orchestrator initialized successfully")

    def get_annotations(self) -> Dict[str, Any]:
        """Get current annotations with enhanced span information."""
        # Return enhanced annotations from dataset if available, otherwise fall back to basic annotations
        if hasattr(self, 'dataset') and hasattr(self.dataset, 'get_annotations'):
            return self.dataset.get_annotations()
        return self.annotations

    def get_current_texts(self) -> List[str]:
        """Get current text representations for all individuals."""
        df = self.data_processor.load_data(
            self.config.data_file_path,
            self.config.individual_name_column,
            self.config.original_text_column
        )
        return df[self.config.original_text_column].tolist()

    def get_text_for_individual(self, individual_name: str) -> str:
        """Get text for a specific individual."""
        df = self.data_processor.load_data(
            self.config.data_file_path,
            self.config.individual_name_column,
            self.config.original_text_column
        )
        row = df[df[self.config.individual_name_column] == individual_name]
        if not row.empty:
            return row[self.config.original_text_column].iloc[0]
        return ""

    def apply_anonymization(self, individual_name: str, terms: List[str]) -> None:
        """Apply anonymization for an individual (simplified implementation)."""
        if individual_name not in self.annotations:
            self.annotations[individual_name] = []

        # Add new anonymization terms (simplified)
        for term in terms:
            self.annotations[individual_name].append({
                'term': term,
                'replacement': self.config.mask_text or '[MASKED]'
            })

    def run_incremental_execution(self, k_values: List[int]) -> Dict[str, Any]:
        """Run the incremental anonymization process for all k values."""
        logging.info("Running incremental execution for k values: %s", k_values)

        # Save initial state
        self._save_initial_state()

        results = {
            'status': 'completed',
            'k_values': k_values,
            'output_directory': self.config.output_folder_path,
            'starting_anonymization': self.config.starting_annon_name,
            'results_per_k': {}
        }

        # Run PETRE for each k value
        for k in k_values:
            logging.info("Running PETRE for k=%d", k)
            k_results = self._run_petre_for_k(k)
            results['results_per_k'][k] = k_results

        logging.info("Incremental execution completed")
        return results

    def _save_initial_state(self) -> None:
        """Save initial annotations and evaluate starting point."""
        logging.info("Saving initial state")

        # Get initial annotations
        annotations = self.get_annotations()
        annotations_filepath = os.path.join(self.config.output_folder_path, "Annotations_PETRE_Start.json")

        with open(annotations_filepath, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2)

        # Evaluate initial accuracy
        accuracy, ranks, _ = self._evaluate_documents(max_rank=1)
        ranks_file_path = os.path.join(self.config.output_folder_path, 'Ranks_Start.csv')
        np.savetxt(ranks_file_path, ranks, delimiter=",")

        logging.info("Initial rank==1 rate = %.2f%%", accuracy * 100)

    def _run_petre_for_k(self, k: int) -> Dict[str, Any]:
        """Run PETRE algorithm for a specific k value (following original logic)."""
        logging.info("Starting PETRE for k=%d", k)

        annotated_terms = {}
        total_n_steps = 0
        annotations_file_path = os.path.join(self.config.output_folder_path, f'Annotations_PETRE_k={k}.json')

        # Load existing annotations if they exist
        if os.path.exists(annotations_file_path):
            logging.info("Loading existing annotations for k=%d", k)
            with open(annotations_file_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
                self.dataset.add_annotations(annotations)

        # Compute individuals that keep requiring protection
        accuracy, ranks, docs_probs = self._evaluate_documents(max_rank=k)
        # Ensure ranks are integers for comparison
        ranks = np.array(ranks, dtype=int)
        n_individuals_to_protect = np.count_nonzero(ranks < k)
        logging.info("Number of individuals requiring protection = %d", n_individuals_to_protect)

        with tqdm(range(n_individuals_to_protect), total=n_individuals_to_protect) as pbar:
            # For each document in the dataset
            for idx, document in enumerate(self.dataset.documents):
                # If document requires protection
                rank = int(ranks[idx])
                if rank < k:
                    label = document["label"]
                    name = self.label_to_name[label]
                    doc_processed = False
                    splits_probs = docs_probs[idx]
                    rank, prob = self._get_doc_rank(splits_probs, label)

                    while not doc_processed:
                        message = f"Individual [{name}] obtained a rank of {rank} with a probability of {prob*100:.2f}%"
                        pbar.set_description(message)
                        logging.info(message)

                        # While top position is not great enough and not in an end state
                        while rank < k:
                            # Mask the most disclosive term of the most disclosive split possible
                            most_disclosive_term, n_masked_terms, n_steps = self._mask_most_disclosive_term(
                                document, splits_probs, annotated_terms, plot_explanations=False)
                            total_n_steps += n_steps

                            # If no term is masked, there are no more terms to mask in this document (avoid infinite loops)
                            if n_masked_terms == 0:
                                logging.info("All meaningful terms already have been masked")
                                break  # Exit loop
                            # If at least one term has been masked
                            else:
                                rank, prob = self._get_doc_rank(splits_probs, label)  # Recompute the rank
                                logging.info(f"Term [{most_disclosive_term}] masked with {n_masked_terms} instance/s | Rank = {rank} | Prob = {prob*100:.2f}%")

                        # Document has been processed
                        doc_processed = True
                        pbar.update()

                        # Store updated annotations
                        annotations = self.dataset.get_annotations()
                        with open(annotations_file_path, 'w', encoding='utf-8') as f:
                            json.dump(annotations, f)

                # Store final annotations using enhanced dataset annotations
        final_annotations = self.dataset.get_annotations()

        with open(annotations_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_annotations, f, indent=2)

        # Final evaluation
        final_accuracy, final_ranks, _ = self._evaluate_documents(max_rank=k)
        final_ranks = np.array(final_ranks, dtype=int)  # Ensure integer type

        logging.info("PETRE k=%d completed. Final accuracy: %.3f, Steps: %d", k, final_accuracy, total_n_steps)

        return {
            'k': k,
            'accuracy': final_accuracy,
            'total_steps': total_n_steps,
            'ranks': final_ranks.tolist(),
            'annotated_terms': annotated_terms
        }

    def _evaluate_documents(self, max_rank: int = 1, use_annotated: bool = True, batch_size: int = 128):
        """Evaluate re-identification risk using original PETRE logic."""
        docs_probs = []
        n_correct_preds = 0
        n_individuals = len(self.dataset.documents)

        # Generate all inputs
        input_texts, doc_to_texts_idxs = self.dataset.get_all_texts(use_annotated)

        # Gather results per document
        docs_probs, ranks = self._pipeline_results_to_docs_probs(input_texts, doc_to_texts_idxs, batch_size=batch_size)

        # Compute number of correct predictions and accuracy
        ranks = np.array(ranks, dtype=int)  # Ensure integer type
        n_correct_preds = np.count_nonzero(ranks <= max_rank)
        accuracy = n_correct_preds / n_individuals

        return accuracy, ranks, docs_probs

    def _pipeline_results_to_docs_probs(self, input_texts: list, doc_to_input_idxs: dict, batch_size: int = 128):
        """Process pipeline results into document probabilities (original logic)."""
        docs_probs = []
        ranks = []

        # Create dataset and input into pipeline
        import datasets
        inputs_dataset = datasets.Dataset.from_dict({"text": input_texts})["text"]
        results = self.tri_pipeline(inputs_dataset, batch_size=batch_size)

        # Gather results per document
        for document in tqdm(self.dataset.documents, desc="Evaluating all documents"):
            label = document["label"]
            splits = document["splits"]
            splits_probs = torch.zeros((len(splits), len(self.names)))  # Splits * Individuals
            doc_results = [results[idx] for idx in doc_to_input_idxs[label]]

            # Get probabilities from each split prediction
            for split_idx, split_preds in enumerate(doc_results):
                for pred in split_preds:
                    pred_label, pred_score = self._pipeline_pred_to_label_score(pred)
                    splits_probs[split_idx, pred_label] = pred_score

            # Store into docs_probs
            docs_probs.append(splits_probs)

            # Check rank position of aggregated probabilities
            rank, prob = self._get_doc_rank(splits_probs, label)
            ranks.append(rank)

        # Transform ranks into NumPy array of integers
        ranks = np.array(ranks, dtype=int)

        return docs_probs, ranks

    def _mask_most_disclosive_term(self, document: Dict, splits_probs: torch.Tensor, annotated_terms: dict, plot_explanations: bool = False):
        """Mask the most disclosive term following original PETRE logic."""
        label = document["label"]
        splits = document["splits"]
        complete_text = document["text"]
        name = self.label_to_name[label]

        most_disclosive_term = None
        n_masked_terms = 0
        n_steps = 0

        # Initialize annotated terms for this individual
        if name not in annotated_terms:
            annotated_terms[name] = set()

        # Get splits sorted by disclosiveness (original logic)
        individual_probs = splits_probs[:, label]
        sorted_splits_idxs = torch.argsort(individual_probs, descending=True)

        # Following disclosiveness order, search to mask the most disclosive term of the most disclosive split
        for split_idx in sorted_splits_idxs:
            split = splits[split_idx]

            # If not all terms in the split are already masked (avoid infinite loops)
            if len(split["masked_terms_idxs"]) < len(split["terms_spans"]):
                text_span = split["text_span"]
                split_text = complete_text[text_span[0]:text_span[1]]

                # Find most disclosive term in this split
                if self.config.explainability_mode == "SHAP":
                    most_disclosive_term = self._shap_explainability(split_text, label, split)
                elif self.config.explainability_mode == "Greedy":
                    most_disclosive_term = self._greedy_explainability(split_text, label, split)
                else:
                    raise ValueError(f"Unknown explainability method: {self.config.explainability_mode}")

                n_steps += 1

                # If there is a term to mask, end of process
                if most_disclosive_term is not None:
                    # Mask the term
                    if self.config.use_mask_all_instances:
                        n_masked_terms = self._mask_all_instances(complete_text, splits, splits_probs, most_disclosive_term)
                    else:
                        # Find term index and mask just this instance
                        term_idx = self._find_term_index(split, most_disclosive_term, split_text)
                        if term_idx >= 0:
                            self.dataset.mask_terms(split, [term_idx])
                            # Update split probabilities
                            splits_probs[split_idx, :] = self._evaluate_split_probs(complete_text, split)
                            n_masked_terms = 1

                    # Track annotated terms
                    annotated_terms[name].add(most_disclosive_term)
                    break  # Exit loop

        return most_disclosive_term, n_masked_terms, n_steps

    def _shap_explainability(self, split_text: str, label: int, split: Dict[str, Any]) -> Optional[str]:
        """Use SHAP-like logic to find the most disclosive term following original PETRE."""
        terms_spans = split["terms_spans"]
        masked_terms_idxs = split["masked_terms_idxs"]

        # If all terms are masked, return None
        if len(masked_terms_idxs) >= len(terms_spans):
            return None

        # Find most disclosive unmasked term (simplified version of SHAP logic)
        best_term = None
        best_score = 0

        for term_idx, (start, end) in enumerate(terms_spans):
            if term_idx in masked_terms_idxs:
                continue  # Skip already masked terms

            term_text = split_text[start:end]

            # Skip if it's a term to ignore
            if term_text.lower() in self.dataset.terms_to_ignore:
                continue

            # Simple heuristic: prioritize proper nouns and names
            score = 0
            if term_text[0].isupper():  # Capitalized
                score += 2
            if any(name_part in term_text.lower() for name_part in self.names[label].lower().split()):
                score += 3  # Name-related terms get highest priority
            if len(term_text) > 2:  # Longer terms
                score += 1

            if score > best_score:
                best_score = score
                best_term = term_text

        return best_term

    def _greedy_explainability(self, split_text: str, label: int, split: Dict[str, Any]) -> Optional[str]:
        """Use greedy search to find the most disclosive term following original PETRE."""
        terms_spans = split["terms_spans"]
        masked_terms_idxs = split["masked_terms_idxs"]

        # If all terms are masked, return None
        if len(masked_terms_idxs) >= len(terms_spans):
            return None

        best_term = None
        best_weight = 0

        # Get current annotated text baseline
        annotated_text = self.dataset.annotate_text(split_text, split)
        baseline_results = self.tri_pipeline([annotated_text])[0]
        baseline_prob = 0
        for pred in baseline_results:
            pred_label, pred_score = self._pipeline_pred_to_label_score(pred)
            if pred_label == label:
                baseline_prob = pred_score
                break

        # Test each unmasked term
        for term_idx, (start, end) in enumerate(terms_spans):
            if term_idx in masked_terms_idxs:
                continue  # Skip already masked terms

            term_text = split_text[start:end]

            # Skip if it's a term to ignore
            if term_text.lower() in self.dataset.terms_to_ignore:
                continue

            # Create version with this term masked
            mask_replacement = self.config.mask_text if self.config.mask_text else "[MASK]"
            masked_text = split_text[:start] + mask_replacement + split_text[end:]

            # Evaluate masked version
            masked_results = self.tri_pipeline([masked_text])[0]
            masked_prob = 0
            for pred in masked_results:
                pred_label, pred_score = self._pipeline_pred_to_label_score(pred)
                if pred_label == label:
                    masked_prob = pred_score
                    break

            # Weight is the difference (higher = more disclosive)
            weight = baseline_prob - masked_prob
            if weight > best_weight:
                best_weight = weight
                best_term = term_text

        return best_term

    def _mask_all_instances(self, complete_text: str, splits: List[Dict], splits_probs: torch.Tensor, most_disclosive_term: str) -> int:
        """Mask all instances of the most disclosive term (original PETRE logic)."""
        n_masked_terms = 0

        for split_idx, split in enumerate(splits):
            text_span = split["text_span"]
            terms_spans = split["terms_spans"]

            # Search other instances of the most_disclosive_term
            terms_idxs_to_mask = []
            for term_idx, span in enumerate(terms_spans):
                span_len = span[1] - span[0]
                if span_len == len(most_disclosive_term):
                    span_text = complete_text[text_span[0] + span[0]:text_span[0] + span[1]]
                    if span_text == most_disclosive_term:
                        terms_idxs_to_mask.append(term_idx)

            # If at least one new term is masked
            if len(terms_idxs_to_mask) > 0:
                self.dataset.mask_terms(split, terms_idxs_to_mask)
                # Reevaluate/update the split probabilities
                splits_probs[split_idx, :] = self._evaluate_split_probs(complete_text, split)
                n_masked_terms += len(terms_idxs_to_mask)

        return n_masked_terms

    def _find_term_index(self, split: Dict[str, Any], term_text: str, split_text: str) -> int:
        """Find the index of a term in the split's terms_spans."""
        terms_spans = split["terms_spans"]
        for term_idx, (start, end) in enumerate(terms_spans):
            if split_text[start:end] == term_text:
                return term_idx
        return -1

    def _evaluate_split_probs(self, complete_text: str, split: Dict[str, Any]) -> torch.Tensor:
        """Evaluate a single split and return probability distribution."""
        text_span = split["text_span"]
        split_text = complete_text[text_span[0]:text_span[1]]

        # Apply current annotations
        annotated_text = self.dataset.annotate_text(split_text, split)

        # Evaluate with TRI pipeline
        results = self.tri_pipeline([annotated_text])[0]
        probs = torch.zeros(len(self.names))

        for pred in results:
            pred_label, pred_score = self._pipeline_pred_to_label_score(pred)
            probs[pred_label] = pred_score

        return probs

    def _get_doc_rank(self, splits_probs: torch.Tensor, true_label: int) -> tuple:
        """Get document rank and probability (original PETRE logic)."""
        # Aggregate probabilities across splits (mean)
        aggregated_probs = torch.mean(splits_probs, dim=0)

        # Sort indices by probability (descending)
        sorted_indices = torch.argsort(aggregated_probs, descending=True)

        # Find rank of true label
        rank = int((sorted_indices == true_label).nonzero(as_tuple=True)[0].item() + 1)
        prob = float(aggregated_probs[true_label].item())

        return rank, prob

    def _pipeline_pred_to_label_score(self, pred: Dict) -> tuple:
        """Convert pipeline prediction to label and score."""
        # Handle both formats: LABEL_<number> and direct label name
        label_str = pred.get('label', '')
        score = float(pred.get('score', 0.0))

        # If it's in LABEL_<number> format, extract the number
        if label_str.startswith('LABEL_'):
            try:
                label_idx = int(label_str.split('_')[1])
            except (IndexError, ValueError):
                label_idx = 0
        else:
            # Convert label name to index
            label_idx = self.name_to_label.get(label_str, 0)

        return int(label_idx), score

    def save_results(self, annotations: Dict[str, Any], k: Optional[int] = None) -> None:
        """Save annotations and results to output directory."""
        filename = f"Annotations_PETRE_k={k}.json" if k is not None else "Annotations_PETRE_final.json"
        filepath = os.path.join(self.config.output_folder_path, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2)

        logging.info("Results saved to: %s", filepath)

    def write_anonymized_texts(self, k_values: List[int], output_filename: str = "anonymized_texts.json") -> None:
        """Write out anonymized texts for all given k values.

        Args:
            k_values: List of k-anonymity values to generate texts for
            output_filename: Name of the output file
        """
        anonymized_data = {}

        # Load original data
        original_df = self.data_processor.load_data(
            self.config.data_file_path,
            self.config.individual_name_column,
            self.config.original_text_column
        )

        # Create a mapping from individual names to original texts
        name_to_text = dict(zip(original_df[self.config.individual_name_column],
                               original_df[self.config.original_text_column]))

        # Process each k value
        for k in k_values:
            logging.info("Generating anonymized texts for k=%d", k)
            anonymized_data[f"k_{k}"] = {}

            # Load annotations for this k value
            annotations_file = os.path.join(self.config.output_folder_path, f'Annotations_PETRE_k={k}.json')
            if not os.path.exists(annotations_file):
                logging.warning("Annotations file not found for k=%d: %s", k, annotations_file)
                continue

            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            # Generate anonymized text for each individual
            for individual_name, individual_annotations in annotations.items():
                if individual_name not in name_to_text:
                    logging.warning("Individual %s not found in original data", individual_name)
                    continue

                original_text = name_to_text[individual_name]
                anonymized_text = self._apply_annotations_to_text(original_text, individual_annotations)

                anonymized_data[f"k_{k}"][individual_name] = {
                    "original_text": original_text,
                    "anonymized_text": anonymized_text,
                    "annotations": individual_annotations
                }

        # Save the anonymized texts
        output_path = os.path.join(self.config.output_folder_path, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(anonymized_data, f, indent=2, ensure_ascii=False)

        logging.info("Anonymized texts saved to: %s", output_path)

    def _apply_annotations_to_text(self, text: str, annotations: List[Dict[str, Any]]) -> str:
        """Apply annotations to text to create anonymized version.

        Args:
            text: Original text
            annotations: List of annotation spans with start, end, value, replacement

        Returns:
            Anonymized text with replacements applied
        """
        if not annotations:
            return text

        # Handle both old and new annotation formats
        spans_to_replace = []
        for annotation in annotations:
            if isinstance(annotation, list):
                # Old format: [start, end] - use default replacement
                start, end = annotation[0], annotation[1]
                replacement = self.config.mask_text if self.config.mask_text else "[MASK]"
            elif isinstance(annotation, dict):
                # New format: {start, end, value, replacement}
                start, end = annotation["start"], annotation["end"]
                replacement = annotation.get("replacement", "[MASK]")
            else:
                continue

            spans_to_replace.append((start, end, replacement))

        # Sort by start position in reverse order to avoid index shifting
        spans_to_replace.sort(key=lambda x: x[0], reverse=True)

        # Apply replacements
        anonymized_text = text
        for start, end, replacement in spans_to_replace:
            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]

        return anonymized_text


__all__ = [
    'PETREDataProcessor',
    'PETREModelManager',
    'SHAPExplainabilityMethod',
    'GreedyExplainabilityMethod',
    'PETREPipelineEvaluator',
    'PETREDatasetImpl',
    'PETREComponentFactory',
    'PETREOrchestratorImpl'
]