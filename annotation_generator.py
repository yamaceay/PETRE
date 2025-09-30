#!/usr/bin/env python3
"""
Dynamic Annotation Generator for PETRE

This module provides automatic annotation generation using various NER models
including spaCy NER, Presidio, and other entity recognition systems. It generates
meaningful spans without requiring pre-created annotation files.

Usage:
    from annotation_generator import AnnotationGenerator, AnnotationMethod

    generator = AnnotationGenerator()
    annotations = generator.generate_annotations(
        data_file='data.json',
        method=AnnotationMethod.SPACY_NER3,
        confidence_threshold=0.8
    )
"""

import json
import logging
import os
import spacy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Set
import pandas as pd

# Optional presidio imports (graceful fallback if not available)
try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logging.warning("Presidio not available. Install with: pip install presidio-analyzer")


class AnnotationMethod(Enum):
    """Available annotation methods."""
    SPACY_NER3 = "spacy_ner3"  # 3-entity model (PERSON, LOCATION, ORGANIZATION)
    SPACY_NER4 = "spacy_ner4"  # 4-entity model (adds MISCELLANEOUS)
    SPACY_NER7 = "spacy_ner7"  # 7-entity model (detailed categories)
    SPACY_FULL = "spacy_full"  # All spaCy entities (18+ categories)
    PRESIDIO = "presidio"      # Microsoft Presidio PII detection
    COMBINED = "combined"      # Combination of multiple methods
    MANUAL = "manual"          # For loading manual annotations


@dataclass(frozen=True)
class EntitySpan:
    """Represents a detected entity span."""
    start: int
    end: int
    text: str
    label: str
    confidence: float
    method: str


@dataclass(frozen=True)
class AnnotationConfig:
    """Configuration for annotation generation."""
    method: AnnotationMethod
    confidence_threshold: float = 0.7
    entity_types: Optional[Set[str]] = None
    merge_overlapping: bool = True
    min_span_length: int = 2
    max_span_length: int = 100
    exclude_patterns: Optional[List[str]] = None


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""

    @abstractmethod
    def extract_entities(self, text: str) -> List[EntitySpan]:
        """Extract entities from text."""
        pass

    @abstractmethod
    def get_supported_entities(self) -> Set[str]:
        """Get set of supported entity types."""
        pass


class SpaCyEntityExtractor(EntityExtractor):
    """spaCy-based entity extractor."""

    def __init__(self, model_name: str = "en_core_web_lg", entity_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize spaCy extractor.

        Args:
            model_name: spaCy model to use
            entity_mapping: Optional mapping from spaCy labels to custom labels
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logging.error(f"spaCy model {model_name} not found. Install with: python -m spacy download {model_name}")
            raise

        self.entity_mapping = entity_mapping or {}
        self.method_name = f"spacy_{model_name}"

    def extract_entities(self, text: str) -> List[EntitySpan]:
        """Extract entities using spaCy NER."""
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            # Map entity label if mapping provided
            label = self.entity_mapping.get(ent.label_, ent.label_)

            entity = EntitySpan(
                start=ent.start_char,
                end=ent.end_char,
                text=ent.text,
                label=label,
                confidence=1.0,  # spaCy doesn't provide confidence scores
                method=self.method_name
            )
            entities.append(entity)

        return entities

    def get_supported_entities(self) -> Set[str]:
        """Get supported entity types."""
        return set(self.nlp.get_pipe("ner").labels)


class PresidioEntityExtractor(EntityExtractor):
    """Presidio-based PII entity extractor."""

    def __init__(self, language: str = "en"):
        """Initialize Presidio extractor."""
        if not PRESIDIO_AVAILABLE:
            raise ImportError("Presidio not available. Install with: pip install presidio-analyzer")

        # Configure NLP engine
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": "en_core_web_lg"}]
        }

        nlp_engine_provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        nlp_engine = nlp_engine_provider.create_engine()

        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=0.0)
        self.language = language
        self.method_name = "presidio"

    def extract_entities(self, text: str) -> List[EntitySpan]:
        """Extract PII entities using Presidio."""
        results = self.analyzer.analyze(text=text, language=self.language)
        entities = []

        for result in results:
            entity = EntitySpan(
                start=result.start,
                end=result.end,
                text=text[result.start:result.end],
                label=result.entity_type,
                confidence=result.score,
                method=self.method_name
            )
            entities.append(entity)

        return entities

    def get_supported_entities(self) -> Set[str]:
        """Get supported PII entity types."""
        return {
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE",
            "CREDIT_CARD", "CRYPTO", "DATE_TIME", "IP_ADDRESS",
            "LOCATION", "MEDICAL_LICENSE", "URL", "US_BANK_NUMBER",
            "US_DRIVER_LICENSE", "US_ITIN", "US_PASSPORT", "US_SSN"
        }


class AnnotationGenerator:
    """Main class for generating annotations dynamically."""

    def __init__(self):
        """Initialize annotation generator."""
        self.extractors: Dict[AnnotationMethod, EntityExtractor] = {}
        self._setup_extractors()

    def _setup_extractors(self):
        """Setup available entity extractors."""
        # spaCy NER3 (simplified 3-entity model)
        ner3_mapping = {
            "PERSON": "PERSON",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "ORG": "ORGANIZATION",
            # Map other entities to these 3 main categories
            "NORP": "ORGANIZATION",
            "FAC": "LOCATION",
        }
        self.extractors[AnnotationMethod.SPACY_NER3] = SpaCyEntityExtractor(
            model_name="en_core_web_lg",
            entity_mapping=ner3_mapping
        )

        # spaCy NER4 (adds miscellaneous category)
        ner4_mapping = {
            "PERSON": "PERSON",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "ORG": "ORGANIZATION",
            "NORP": "ORGANIZATION",
            "FAC": "LOCATION",
            # Everything else becomes MISC
            "EVENT": "MISC",
            "WORK_OF_ART": "MISC",
            "LAW": "MISC",
            "LANGUAGE": "MISC",
            "PRODUCT": "MISC",
        }
        self.extractors[AnnotationMethod.SPACY_NER4] = SpaCyEntityExtractor(
            model_name="en_core_web_lg",
            entity_mapping=ner4_mapping
        )

        # spaCy NER7 (detailed categories)
        ner7_mapping = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "NORP": "NORP",  # Nationalities, political groups
            "FAC": "FACILITY",
            "EVENT": "EVENT",
            "WORK_OF_ART": "WORK_OF_ART",
            "LAW": "LAW",
            "LANGUAGE": "LANGUAGE",
            "DATE": "DATE",
            "TIME": "TIME",
            "PERCENT": "QUANTITY",
            "MONEY": "QUANTITY",
            "QUANTITY": "QUANTITY",
            "ORDINAL": "QUANTITY",
            "CARDINAL": "QUANTITY",
        }
        self.extractors[AnnotationMethod.SPACY_NER7] = SpaCyEntityExtractor(
            model_name="en_core_web_lg",
            entity_mapping=ner7_mapping
        )

        # spaCy Full (all entities)
        self.extractors[AnnotationMethod.SPACY_FULL] = SpaCyEntityExtractor(
            model_name="en_core_web_lg"
        )

        # Presidio (if available)
        if PRESIDIO_AVAILABLE:
            try:
                self.extractors[AnnotationMethod.PRESIDIO] = PresidioEntityExtractor()
            except Exception as e:
                logging.warning(f"Failed to initialize Presidio: {e}")

    def generate_annotations(
        self,
        data_file: str,
        individual_name_column: str = "name",
        text_column: str = "text",
        config: Optional[AnnotationConfig] = None
    ) -> Dict[str, List[List[int]]]:
        """
        Generate annotations for a dataset.

        Args:
            data_file: Path to dataset file (JSON or CSV)
            individual_name_column: Column name for individual names
            text_column: Column name for text content
            config: Annotation configuration

        Returns:
            Dictionary mapping individual names to list of [start, end] spans
        """
        if config is None:
            config = AnnotationConfig(method=AnnotationMethod.SPACY_NER3)

        # Load data
        df = self._load_data(data_file, individual_name_column, text_column)

        # Generate annotations
        annotations = {}
        extractor = self.extractors.get(config.method)

        if not extractor:
            raise ValueError(f"Extractor not available for method: {config.method}")

        logging.info(f"Generating annotations using {config.method.value}")

        for _, row in df.iterrows():
            name = row[individual_name_column]
            text = row[text_column]

            # Extract entities
            entities = extractor.extract_entities(text)

            # Filter and process entities
            filtered_entities = self._filter_entities(entities, config)

            # Convert to span format
            spans = [[entity.start, entity.end] for entity in filtered_entities]

            # Merge overlapping spans if requested
            if config.merge_overlapping:
                spans = self._merge_overlapping_spans(spans)

            annotations[name] = spans

        return annotations

    def generate_combined_annotations(
        self,
        data_file: str,
        methods: List[AnnotationMethod],
        individual_name_column: str = "name",
        text_column: str = "text",
        config: Optional[AnnotationConfig] = None
    ) -> Dict[str, List[List[int]]]:
        """
        Generate annotations using multiple methods and combine them.

        Args:
            data_file: Path to dataset file
            methods: List of annotation methods to combine
            individual_name_column: Column name for individual names
            text_column: Column name for text content
            config: Annotation configuration

        Returns:
            Combined annotations dictionary
        """
        if config is None:
            config = AnnotationConfig(method=AnnotationMethod.COMBINED)

        # Load data
        df = self._load_data(data_file, individual_name_column, text_column)

        annotations = {}

        logging.info(f"Generating combined annotations using: {[m.value for m in methods]}")

        for _, row in df.iterrows():
            name = row[individual_name_column]
            text = row[text_column]

            all_entities = []

            # Extract entities using all methods
            for method in methods:
                extractor = self.extractors.get(method)
                if extractor:
                    entities = extractor.extract_entities(text)
                    all_entities.extend(entities)

            # Filter and deduplicate
            filtered_entities = self._filter_entities(all_entities, config)

            # Convert to spans and merge overlapping
            spans = [[entity.start, entity.end] for entity in filtered_entities]
            spans = self._merge_overlapping_spans(spans)

            annotations[name] = spans

        return annotations

    def save_annotations(
        self,
        annotations: Dict[str, List[List[int]]],
        output_path: str,
        method_name: str = "generated"
    ) -> None:
        """
        Save annotations to file.

        Args:
            annotations: Generated annotations
            output_path: Output file path
            method_name: Method name for filename
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        filename = f"Annotations_{method_name}.json"
        full_path = os.path.join(output_path, filename) if os.path.isdir(output_path) else output_path

        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        logging.info(f"Saved annotations to: {full_path}")

    def _load_data(self, data_file: str, name_col: str, text_col: str) -> pd.DataFrame:
        """Load dataset from file."""
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        if data_file.endswith('.json'):
            df = pd.read_json(data_file)
        elif data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")

        # Validate required columns
        if name_col not in df.columns:
            raise ValueError(f"Column '{name_col}' not found in dataset")
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in dataset")

        return df[[name_col, text_col]]

    def _filter_entities(self, entities: List[EntitySpan], config: AnnotationConfig) -> List[EntitySpan]:
        """Filter entities based on configuration."""
        filtered = []

        for entity in entities:
            # Filter by confidence threshold
            if entity.confidence < config.confidence_threshold:
                continue

            # Filter by entity types if specified
            if config.entity_types and entity.label not in config.entity_types:
                continue

            # Filter by span length
            span_length = entity.end - entity.start
            if span_length < config.min_span_length or span_length > config.max_span_length:
                continue

            # Filter by exclude patterns if specified
            if config.exclude_patterns:
                if any(pattern in entity.text.lower() for pattern in config.exclude_patterns):
                    continue

            filtered.append(entity)

        return filtered

    def _merge_overlapping_spans(self, spans: List[List[int]]) -> List[List[int]]:
        """Merge overlapping spans."""
        if not spans:
            return []

        # Sort by start position
        sorted_spans = sorted(spans, key=lambda x: x[0])
        merged = [sorted_spans[0]]

        for current in sorted_spans[1:]:
            last = merged[-1]

            # If overlapping or adjacent, merge
            if current[0] <= last[1]:
                merged[-1] = [last[0], max(last[1], current[1])]
            else:
                merged.append(current)

        return merged

    def get_available_methods(self) -> List[AnnotationMethod]:
        """Get list of available annotation methods."""
        return list(self.extractors.keys())

    def get_method_info(self, method: AnnotationMethod) -> Dict[str, Any]:
        """Get information about a specific method."""
        extractor = self.extractors.get(method)
        if not extractor:
            return {"available": False, "reason": "Extractor not found"}

        try:
            supported_entities = extractor.get_supported_entities()
            return {
                "available": True,
                "supported_entities": list(supported_entities),
                "extractor_type": type(extractor).__name__
            }
        except Exception as e:
            return {"available": False, "reason": str(e)}


# Convenience functions for easy usage
def create_spacy_ner3_annotations(
    data_file: str,
    individual_name_column: str = "name",
    text_column: str = "text",
    confidence_threshold: float = 0.7
) -> Dict[str, List[List[int]]]:
    """Create annotations using spaCy NER3 method."""
    generator = AnnotationGenerator()
    config = AnnotationConfig(
        method=AnnotationMethod.SPACY_NER3,
        confidence_threshold=confidence_threshold
    )
    return generator.generate_annotations(data_file, individual_name_column, text_column, config)


def create_presidio_annotations(
    data_file: str,
    individual_name_column: str = "name",
    text_column: str = "text",
    confidence_threshold: float = 0.7
) -> Dict[str, List[List[int]]]:
    """Create annotations using Presidio method."""
    generator = AnnotationGenerator()
    config = AnnotationConfig(
        method=AnnotationMethod.PRESIDIO,
        confidence_threshold=confidence_threshold
    )
    return generator.generate_annotations(data_file, individual_name_column, text_column, config)


def create_combined_annotations(
    data_file: str,
    individual_name_column: str = "name",
    text_column: str = "text",
    confidence_threshold: float = 0.7
) -> Dict[str, List[List[int]]]:
    """Create annotations combining multiple methods."""
    generator = AnnotationGenerator()
    methods = [AnnotationMethod.SPACY_NER3, AnnotationMethod.PRESIDIO]
    config = AnnotationConfig(
        method=AnnotationMethod.COMBINED,
        confidence_threshold=confidence_threshold
    )
    return generator.generate_combined_annotations(data_file, methods, individual_name_column, text_column, config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    generator = AnnotationGenerator()

    # Show available methods
    print("Available annotation methods:")
    for method in generator.get_available_methods():
        info = generator.get_method_info(method)
        print(f"  {method.value}: {'✓' if info['available'] else '✗'}")
        if info['available']:
            print(f"    Entities: {len(info['supported_entities'])} types")

    # Example: Generate annotations for a dataset
    # annotations = create_spacy_ner3_annotations("data.json")
    # generator.save_annotations(annotations, "output/", "spacy_ner3")