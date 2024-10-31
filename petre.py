#region ###################################### Imports ######################################
import sys
import os
import json
import gc
import re
import ntpath
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO) # Configure logging

from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import en_core_web_lg # This model is leveraged for every spaCy usage (https://spacy.io/models/en#en_core_web_lg)

import torch
from torch.utils.data import Dataset
from transformers import pipeline, Pipeline
import shap

#endregion

#region ###################################### Configuration file argument ######################################

#region ################### Arguments parsing ###################
def argument_parsing():
    if (args_count := len(sys.argv)) > 2:
        logging.exception(Exception(f"One argument expected, got {args_count - 1}"))
    elif args_count < 2:
        logging.exception(Exception("You must specify the JSON configuration filepath as first argument"))

    target_dir = sys.argv[1]
    return target_dir

#endregion

#region ################### JSON file loading ###################
def get_config_from_file(target_dir):
    if not target_dir.endswith(".json"):
        logging.exception(f"The configuration file {target_dir} needs to have json format (end with .json)")
    elif not os.path.isfile(target_dir):
        logging.exception(f"The JSON configuration file {target_dir} doesn't exist")

    with open(target_dir, "r") as f:
        config = json.load(f)
    return config
#endregion

#endregion

#region ###################################### PETRE class ######################################

class PETRE():
    #region ################### Properties ###################

    #region ########## Mandatory configs ##########

    mandatory_configs_names = ["output_base_folder_path", "data_file_path",
                            "individual_name_column", "original_text_column",
                            "starting_anonymization_path", "tri_pipeline_path", "ks"]
    output_base_folder_path = None
    data_file_path = None
    individual_name_column = None
    original_text_column = None
    starting_anonymization_path = None
    tri_pipeline_path = None
    ks = None

    #endregion

    #region ########## Optional configs with default values ##########

    optional_configs_names = ["mask_text", "use_mask_all_instances"]
    mask_text:str = ""
    use_mask_all_instances:bool=True

    #endregion

    #region ########## Derived configs ##########

    starting_annon_name:str = None
    output_folder_path:str = None

    #endregion

    #region ########## Functional properties ##########
    
    data_df:pd.DataFrame = None    
    label_to_name:dict = None
    name_to_label:dict = None
    annotated_individuals:set = None
    non_annotated_individuals:set = None

    tri_pipeline:Pipeline = None
    dataset:Dataset = None
    explainer = None
    device = None

    #endregion

    #endregion

    #region ################### Constructor and configurations ###################

    def __init__(self, **kwargs):
        self.set_configs(**kwargs, are_mandatory_configs_required=True)

    def set_configs(self, are_mandatory_configs_required=False, **kwargs):
        arguments = kwargs.copy()

        # Mandatory configs
        for setting_name in self.mandatory_configs_names:
            value = arguments.get(setting_name, None)
            if isinstance(value, str) or isinstance(value, list):
                self.__dict__[setting_name] = arguments[setting_name]
                del arguments[setting_name]
            elif are_mandatory_configs_required:
                raise AttributeError(f"Mandatory argument {setting_name} is not defined or it is not a string or list")
        
        # Check list of ks
        if (not isinstance(self.ks, list)) or len(self.ks)==0 or len([k for k in self.ks if isinstance(k, int)])<len(self.ks):
           raise AttributeError(f"Setting \"ks\" must be a list of integers", isinstance(self.ks, list), len(self.ks), len([k for k in self.ks if isinstance(k, int)]))
        # Sort ks in ascending order
        else:
           self.ks.sort()
        
        # Store remaining optional configs
        for (opt_setting_name, opt_setting_value) in arguments.items():
            if opt_setting_name in self.optional_configs_names:                
                if isinstance(opt_setting_value, str) or isinstance(opt_setting_value, int) or \
                isinstance(opt_setting_value, float) or isinstance(opt_setting_value, bool):
                    self.__dict__[opt_setting_name] = opt_setting_value
                else:
                    raise AttributeError(f"Optional argument {opt_setting_name} is not a string, integer, float or boolean.")
            else:
                logging.warning(f"Unrecognized setting name {opt_setting_name}")

        # Generate derived configs
        head, tail = ntpath.split(self.starting_anonymization_path)
        filename = tail or ntpath.basename(head)
        self.starting_annon_name = os.path.splitext(filename)[0]
        self.output_folder_path =  os.path.join(self.output_base_folder_path, self.starting_annon_name)

        # Check for GPU with CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        else:
            self.device = torch.device("cpu")

    #endregion


    #region ################### Run all blocks ###################

    def run(self, verbose=True):
        self.initialization(verbose=verbose)
        self.incremental_execution(verbose=verbose)

    #endregion


    #region ################### Initialization ###################

    def initialization(self, verbose=True):
        if verbose: logging.info("######### START: INITIALIZATION #########")


        if verbose: logging.info("#### START: LOADING DATA ####")

        if os.path.exists(self.data_file_path):
            if self.data_file_path.endswith(".json"): # JSON
                complete_df = pd.read_json(self.data_file_path)
            else: # Or CSV
               complete_df = pd.read_csv(self.data_file_path)
            self.data_df = complete_df[[self.individual_name_column, self.original_text_column]]

            self.names = sorted([name for name in self.data_df[self.individual_name_column]])
            self.name_to_label = {name:idx for idx, name in enumerate(self.names)}
            self.label_to_name = {label:name for name, label in self.name_to_label.items()}
            self.num_labels = len(self.label_to_name)
        else:
            raise Exception(f"Data file at {self.data_file_path} not found")

        if verbose: logging.info("#### END: LOADING DATA ####")

        if verbose: logging.info("#### START: LOADING TRI PIPELINE AND EXPLAINER ####")
        
        # Load re-identification pipeline
        self.tri_pipeline = pipeline("text-classification", model=self.tri_pipeline_path,
                                    tokenizer=self.tri_pipeline_path,
                                    device=self.device,
                                    top_k=self.num_labels)

        # Create explainer
        self.explainer = shap.Explainer(self.tri_pipeline, silent=True)

        if verbose: logging.info("#### END: LOADING TRI PIPELINE AND EXPLAINER ####")

        if verbose: logging.info("#### START: CREATING DATASET WITH STARTING ANONYMIZATION ####")

        self.dataset = PETREDataset(self.data_df, self.tri_pipeline.tokenizer, self.name_to_label, self.mask_text)
        
        # Add starting anonymization
        if os.path.exists(self.starting_anonymization_path):
            with open(self.starting_anonymization_path, "r", encoding="utf-8") as f:
                starting_annotations = json.load(f)
                self.dataset.add_annotations(starting_annotations)
        else:
            raise Exception(f"Starting anonymization file at {self.starting_anonymization_path} not found")
        
        if verbose: logging.info("#### END: CREATING DATASET WITH STARTING ANONYMIZATION ####")

        if verbose: logging.info("#### START: COMPARING DATA WITH ANONYMIZATION ####")

        individuals = set(self.names)
        if verbose: logging.info(f"There are {len(individuals)} individuals to protect")
        annotation_names = set(starting_annotations.keys())
        if verbose: logging.info(f"There are {len(annotation_names)} annotations")
        self.annotated_individuals = individuals.intersection(annotation_names)
        self.non_annotated_individuals = individuals - self.annotated_individuals
        if len(self.non_annotated_individuals) == 0:
           if verbose: logging.info(f"All individuals have annotations")
        else:
           if verbose: logging.warning(f"There are {len(self.non_annotated_individuals)} without annotations: {self.non_annotated_individuals}")

        if verbose: logging.info("#### END: COMPARING DATA WITH ANONYMIZATION ####")

        if verbose: logging.info("#### START: CREATING OUTPUT FOLDER ####")
        
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path, exist_ok=True)

        if verbose: logging.info("#### END: CREATING OUTPUT FOLDER ####")
        

        if verbose: logging.info("######### END: INITIALIZATION #########")

    #endregion

    #region ################### Incremental execution ###################

    def incremental_execution(self, verbose=True):
        if verbose: logging.info("######### START: EXECUTION #########")

        if verbose: logging.info("#### START: STARTING POINT ####")

        annotations = self.dataset.get_annotations()
        annotations_filepath = os.path.join(self.output_folder_path, f'Annotations_PETRE_Start.json')
        with open(annotations_filepath, 'w') as f:
            json.dump(annotations, f)
        accuracy, docs_probs = self.evaluate(min_rank=1)
        logging.info(f"Initial rank==1 rate = {accuracy*100:.2f}%")

        if verbose: logging.info("#### END: STARTING POINT ####")

        # Incrementing k
        for current_k in self.ks:
            if verbose: logging.info(F"#### START: PETRE WITH K={current_k} ####")
            
            annotations, annotated_terms, n_steps = self.petre(current_k, plot_explanations=False, verbose=verbose)

            # Store annotations
            annotations_file_path = os.path.join(self.output_folder_path, f'Annotations_PETRE_k={current_k}.json')
            with open(annotations_file_path, 'w') as f:
                json.dump(annotations, f)

            # Compute and store ranks
            ranks = self.get_ranks()
            ranks_file_path = os.path.join(self.output_folder_path, f'Ranks_k={current_k}.csv')
            ranks.tofile(ranks_file_path, sep=",")

            if verbose: logging.info(F"#### END: PETRE WITH K={current_k} ####")        

        if verbose: logging.info("######### END: EXECUTION #########")

    #region ########## Risk evaluation ##########

    def evaluate(self, min_rank:int=1, use_annotated:bool=True):
        docs_probs = []
        n_correct_preds = 0
        n_individuals = len(self.dataset)
        
        for document in tqdm(self.dataset, total=len(self.dataset)):
            # Get document splits probabilities and rank
            splits_probs, rank = self.evaluate_doc(document, use_annotated=use_annotated)
            docs_probs.append(splits_probs)

            # Check for correct prediction
            if rank <= min_rank:
                n_correct_preds += 1

        # Compute accuracy
        accuracy = n_correct_preds/n_individuals

        return accuracy, docs_probs

    def evaluate_doc(self, document:dict, use_annotated:bool=True):
        complete_text = document["text"]
        label = document["label"]
        splits = document["splits"]
        splits_probs = torch.zeros((len(splits), len(self.names))) # Splits * Individuals

        # Evaluate each split
        for split_idx, split in enumerate(splits):
            splits_probs[split_idx, :] = self.evaluate_split(complete_text, split, use_annotated=use_annotated)

        # Check rank position of aggregated probs
        rank = self.get_doc_rank(splits_probs, label)

        return splits_probs, rank
    
    def evaluate_split(self, complete_text:str, split:dict, use_annotated:bool=True):
        split_span = split["text_span"]
        split_text = complete_text[split_span[0]:split_span[1]]

        # Annotate text if required
        if use_annotated:
            split_text = self.dataset.annotate_text(split_text, split)

        # Get predictions
        results = self.tri_pipeline([split_text])[0]
        split_probs = torch.empty((len(results)))
        for res in results:
            pred_label = int(res["label"].split("_")[1])
            split_probs[pred_label] = res["score"]

        return split_probs
        
    def get_doc_rank(self, splits_probs:np.array, label:int):
        # Aggregate probabilities
        aggregated_probs = splits_probs.sum(dim=0)

        # Get rank position
        sorted_idxs = torch.argsort(aggregated_probs, descending=True)
        rank = torch.where(sorted_idxs == label)[0].item() + 1 # +1 To start rank at 1

        return rank

    def get_ranks(self):
        # Get ranks for all documents
        ranks = []
        for document in tqdm(self.dataset, total=len(self.dataset), desc="Obtaining all ranks"):
            splits_probs, rank = self.evaluate_doc(document, use_annotated=True)
            ranks.append(rank)

        # Transform to NumPy
        ranks = np.array(ranks)

        return ranks

    #endregion

    #region ########## Explainability ##########

    def get_tokens_weights(self, text:str, label:int, plot_explanations=False):
        shap_values = self.explainer([text])
        if plot_explanations:
            shap.plots.text(shap_values[0, :, label])
        tokens_weights = shap_values.values[0, :, label]
        return tokens_weights

    def get_terms_weights(self, terms_to_tokens, tokens_weights, masked_terms_idxs):
        terms_weights = np.empty(len(terms_to_tokens))

        for idx, term_tokens in enumerate(terms_to_tokens):
            # If term is masked
            if idx in masked_terms_idxs:
                term_weight = float("-inf")
            # Otherwise, compute term weight
            else:
                term_weight = 0
                for token_idx in term_tokens:
                    term_weight += tokens_weights[token_idx]

            # Set term weight
            terms_weights[idx] = term_weight

        return terms_weights

    #endregion

    #region ########## Method ##########

    def petre(self, k:int, plot_explanations:bool=False, verbose:bool=True):
        annotated_terms = {}
        n_steps = 0

        # For each document
        with tqdm(self.dataset, total=len(self.dataset)) as pbar:
            for document in pbar:                
                label = document["label"]
                name = self.label_to_name[label]                
                annotated_terms[name] = annotated_terms.get(name, set())
                complete_text = document["text"]
                splits = document["splits"]

                # Initial evaluation
                splits_probs, rank = self.evaluate_doc(document, use_annotated=True)

                # If document was already protected
                if rank >= k:
                    pbar.set_description(f"Individual {name} was already protected, obtaining a rank of {rank}")
                else:
                    pbar.set_description(f"Protecting individual {name}, which initially obtained a rank of {rank}")
                    # While top position is not great enough
                    while rank < k:
                        # Get splits sorted by disclosiveness
                        individual_probs = splits_probs[:, label]
                        sorted_splits_idxs = torch.argsort(individual_probs, descending=True)

                        # Mask the most disclosive term of the most disclosive split possible
                        n_masked_terms = 0
                        split_idx = 0
                        while n_masked_terms == 0 and split_idx < len(sorted_splits_idxs):
                            split = splits[split_idx]

                            # If not all terms in the split are already masked (avoid infinite loops)
                            if len(split["masked_terms_idxs"]) < len(split["terms_spans"]):
                                # Get split's most disclosive term
                                out = self.get_most_disclosive(complete_text, split, label, plot_explanations=plot_explanations)
                                (most_disclosive_term_idx, most_disclosive_term) = out
                                n_steps += 1

                                # If there is a term to mask (avoid infinite loops)
                                if most_disclosive_term is not None:
                                    n_masked_terms += self.mask(most_disclosive_term, most_disclosive_term_idx,
                                                                split_idx, splits, splits_probs, complete_text)
                                    annotated_terms[name].update(most_disclosive_term)                                    
                                    if verbose: logging.info(f"Masking of [{most_disclosive_term}] done with {n_masked_terms} terms instances masked")

                            # Next split
                            split_idx += 1
                        
                        # If no term is masked, there are no more terms to mask in this document (avoid infinite loops)
                        if n_masked_terms == 0:
                            if verbose: logging.info("All meaningful terms have been already masked")
                            break
                        # If at least one term has been masked
                        else:
                            rank = self.get_doc_rank(splits_probs, label) # Recompute the rank

        if verbose: logging.info(f"Total n-steps = {n_steps}")

        # Get annotations
        annotations = self.dataset.get_annotations()

        return annotations, annotated_terms, n_steps

    def get_most_disclosive(self, complete_text:str, split:dict, label:int, plot_explanations:bool=False)->tuple:
        text_span = split["text_span"]
        split_text = complete_text[text_span[0]:text_span[1]]
        terms_spans = split["terms_spans"]
        terms_to_tokens = split["terms_to_tokens"]
        masked_terms_idxs = split["masked_terms_idxs"]

        # Get explanation terms' weights
        annotated_text = self.dataset.annotate_text(split_text, split)
        tokens_weights = self.get_tokens_weights(annotated_text, label, plot_explanations=plot_explanations)
        terms_weights = self.get_terms_weights(terms_to_tokens, tokens_weights, masked_terms_idxs)

        # Get index of most disclosive term index
        most_disclosive_term_idx = self.get_most_disclosive_term_idx(terms_weights)

        # Get text of the most disclosive term
        if most_disclosive_term_idx >= 0:
           most_disclosive_term_span = terms_spans[most_disclosive_term_idx]
           most_disclosive_term = split_text[most_disclosive_term_span[0]:most_disclosive_term_span[1]]
        else:
           most_disclosive_term = None        

        return most_disclosive_term_idx, most_disclosive_term
    
    def get_most_disclosive_term_idx(self, terms_weights:np.array)->int:
        most_disclosive_term_idx = -1

        # Sort weights from maximum to minimum
        sorted_terms_idxs = np.argsort(terms_weights)[::-1]

        # Normalize term weights
        norm_term_weights = np.copy(terms_weights)
        norm_term_weights[terms_weights<0] = 0
        norm_term_weights /= (norm_term_weights.sum() if norm_term_weights.sum() > 0 else 1)

        # For each term idx
        for term_idx in sorted_terms_idxs:
            term_weight = norm_term_weights[term_idx]
            # If term already masked or has negative weight, exit loop
            if term_weight == float("-inf") or term_weight <= 0:
                break
            # Otherwise, most disclosive term found
            else:
                most_disclosive_term_idx = term_idx
                break

        return most_disclosive_term_idx

    def mask(self, most_disclosive_term:str, most_disclosive_term_idx:int, split_idx:int, splits:list, splits_probs:np.array, complete_text:str)->int:
        n_masked_terms = 0

        # If enabled, mask all instances of the most_disclosive_term
        if self.use_mask_all_instances:
            n_masked_terms += self.mask_all_instances(complete_text, splits, splits_probs, most_disclosive_term)
        # Otherwise, only mask the term within the most disclosive split
        else:
            split = splits[split_idx]
            self.dataset.mask_terms(split, [most_disclosive_term_idx])
            # Revaluate the split probabilities
            splits_probs[split_idx, :] = self.evaluate_split(complete_text, split)
            n_masked_terms += 1
        
        return n_masked_terms

    def mask_all_instances(self, complete_text:str, splits:list, splits_probs:np.array, most_disclosive_term:str) -> int:
        n_masked_terms = 0

        for split_idx, split in enumerate(splits):
            text_span = split["text_span"]
            terms_spans = split["terms_spans"]            

            # Search other instances of the most_disclosive_term
            terms_idxs_to_mask = []
            for term_idx, span in enumerate(terms_spans):
                span_len = span[1]-span[0]
                if span_len == len(most_disclosive_term):
                    span_text = complete_text[text_span[0]+span[0]:text_span[0]+span[1]]
                    if span_text == most_disclosive_term:
                        terms_idxs_to_mask.append(term_idx)
            
            # If at least one new term is masked
            if len(terms_idxs_to_mask) > 0:
                self.dataset.mask_terms(split, terms_idxs_to_mask)
                # Revaluate the split probabilities
                splits_probs[split_idx, :] = self.evaluate_split(complete_text, split)
            
            n_masked_terms += len(terms_idxs_to_mask)
        
        return n_masked_terms
    
    #endregion

    #endregion

#endregion

#region ###################################### PETRE dataset ######################################

class PETREDataset(Dataset):
  def __init__(self, df, tokenizer, name_to_label, mask_text:str, use_chunking=True):
      # Dataframe must have two columns: name and text
      assert len(df.columns) == 2
      self.df = df

      # Set general attributes
      self.tokenizer = tokenizer
      self.name_to_label = name_to_label
      self.label_to_name = {value:key for key, value in self.name_to_label.items()}
      self.spacy_nlp = en_core_web_lg.load()
      self.use_chunking = use_chunking
      self.mask_text = mask_text
      self.tokenized_mask = self.tokenizer.encode(self.mask_text, add_special_tokens=False)
      self.terms_to_ignore = self.get_terms_to_ignore()

      # Compute inputs and labels
      self.generate()

  def get_terms_to_ignore(self):
    stopwords = self.spacy_nlp.Defaults.stop_words
    terms_to_ignore = set()
    terms_to_ignore.update(stopwords) # Stopwords as base of terms to ignore
    mask_marks_list = {"sensitive", "person", "dem", "loc",
                        "org", "datetime", "quantity", "misc",
                        "norp", "fac", "gpe", "product", "event",
                        "work_of_art", "law", "language", "date",
                        "time", "ordinal", "cardinal", "date_time",
                        "nrp", "location", "organization", "\*\*\*",
                        self.mask_text}
    terms_to_ignore.update(mask_marks_list) # Add masking marks
    terms_to_ignore.update({"[CLS]", "[SEP]", "[PAD]", "", "\t", "\n"}) # Add special tokens

    return terms_to_ignore

  def generate(self, gc_freq=5):
        texts_column = list(self.df[self.df.columns[1]])
        names_column = list(self.df[self.df.columns[0]])
        labels_idxs = list(map(lambda x: self.name_to_label[x], names_column))   # Compute labels, translated to the identity index

        # Sentence splitting
        self.documents = [None] * len(labels_idxs)
        for idx, (text, label) in tqdm(enumerate(zip(texts_column, labels_idxs)), total=len(texts_column),
                                                desc="Processing sentence splitting"):
            splits = []
            document = {"text": text, "label": label, "splits": splits}
            doc = self.spacy_nlp(text)
            for sentence in doc.sents:
                sentence_txt = text[sentence.start_char:sentence.end_char]
                # Ensure length is less than the maximum
                sent_token_count = len(self.tokenizer.encode(sentence_txt, add_special_tokens=True))
                if sent_token_count > self.tokenizer.model_max_length:
                    logging.exception(f"ERROR: Sentence with length {sent_token_count} > {self.tokenizer.model_max_length} at index {idx} with label {label} | {sentence_txt}")
                else:
                    terms_spans = self.get_terms_spans(sentence_txt)
                    terms_to_tokens = self.get_terms_to_tokens(terms_spans, sentence_txt)
                    splits.append({"text_span": (sentence.start_char, sentence.end_char),
                                    "terms_spans": terms_spans,
                                    "terms_to_tokens": terms_to_tokens,
                                    "masked_terms_idxs": []})

            # Store document
            self.documents[label] = document

            # Delete document for reducing memory consumption
            del doc

            # Periodically use GarbageCollector for reducing memory consumption
            if idx % gc_freq == 0:
                gc.collect()

  def get_terms_spans(self, text):
    doc = self.spacy_nlp(text)
    token_idx = 0
    terms_spans = []
    special_chars_pattern = re.compile(r"[^\nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ/]+")

    if self.use_chunking:
      # Get document text spans considering chunks
      for chunk in doc.noun_chunks:
        # Add tokens previous to chunk
        for token_idx in range(token_idx, chunk.start):
          token = doc[token_idx]
          clean_token_text = re.sub(special_chars_pattern, '', token.text).strip()
          if clean_token_text not in self.terms_to_ignore:  # Avoiding undesired terms
            start = token.idx
            end = start + len(token)
            terms_spans.append((start, end))

        # Add chunk tokens
        start = doc[chunk.start].idx
        last_token = doc[chunk.end - 1]
        end = last_token.idx + len(last_token)
        terms_spans.append((start, end))

        # Update token index
        token_idx = chunk.end

    # Add remaining text spans
    for token_idx in range(token_idx, len(doc)):
      token = doc[token_idx]
      clean_token_text = re.sub(special_chars_pattern, '', token.text).strip()
      if clean_token_text not in self.terms_to_ignore:  # Avoiding undesired terms
        start = token.idx
        end = start + len(token)
        terms_spans.append((start, end))

    # Sort text spans by position
    terms_spans = sorted(terms_spans, key=lambda span: span[0], reverse=False)

    return terms_spans

  def get_terms_to_tokens(self, terms_spans, text):
    terms_to_tokens = []
    results = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = results["offset_mapping"]

    last_token_idx = 0
    for (term_start, term_end) in terms_spans:
      term_tokens = []
      for token_idx in range(last_token_idx, len(offsets)):
        (token_start, token_end) = offsets[token_idx]
        if term_start <= token_start < term_end:
          term_tokens.append(token_idx+1) #+1 Due to the [CLS] token
        elif len(term_tokens) > 0: # If first token not contained, is end of term
          break
      last_token_idx = token_idx # Store last token to continue search
      terms_to_tokens.append(term_tokens)

    return terms_to_tokens

  def add_annotations(self, annotations):
    for name, annots in tqdm(annotations.items(), total=len(annotations)):
      label = self.name_to_label[name]
      document = self.documents[label]
      text = document["text"]
      for annotation_span in annots:
        splits_spans = [split["text_span"] for split in document["splits"]]
        intersecting_splits_idxs = self.get_intersecting_spans(annotation_span, splits_spans)
        for split_idx in intersecting_splits_idxs:
          split = document["splits"][split_idx]
          split_span = split["text_span"]
          terms_spans = [(split_span[0]+span[0], split_span[0]+span[1]) for span in split["terms_spans"]]
          annotated_terms_idxs = self.get_intersecting_spans(annotation_span, terms_spans)
          self.mask_terms(split, annotated_terms_idxs)

  def get_intersecting_spans(self, ref_span, spans_list):
    intersecting_spans_idxs = []
    sorted_spans = sorted(spans_list, key=lambda span: span[0], reverse=False)
    for span_idx, span in enumerate(sorted_spans):
      # If intersect
      if span[0] <= ref_span[0] < span[1] or \
        span[0] <= ref_span[1] < span[1] or \
        ref_span[0] <= span[0] < ref_span[1]:
        intersecting_spans_idxs.append(span_idx)
      else:
        if span[1] > ref_span[0]:
          break
    return intersecting_spans_idxs

  def mask_terms(self, split, terms_idx_to_mask):
    terms_to_tokens = split["terms_to_tokens"]
    masked_terms_idxs = split["masked_terms_idxs"]

    # Sort terms index from first to last
    sorted_terms_idx_to_mask = sorted(terms_idx_to_mask, reverse=False)

    # Apply tokens offset to following tokens
    for term_idx in sorted_terms_idx_to_mask:
      if term_idx not in masked_terms_idxs: # Not re-mask
        n_term_tokens = len(terms_to_tokens[term_idx])
        n_mask_tokens = len(self.tokenized_mask)
        tokens_offset = n_mask_tokens - n_term_tokens
        for next_term_idx in range(term_idx+1, len(terms_to_tokens)):
          next_term_tokens = terms_to_tokens[next_term_idx]
          for idx, token_idx in enumerate(next_term_tokens):
            next_term_tokens[idx] = token_idx + tokens_offset

        # Add masked term
        masked_terms_idxs.append(term_idx)

  def annotate_text(self, text, split):
    terms_spans = split["terms_spans"]
    masked_terms_idxs = split["masked_terms_idxs"]
    sorted_masked_terms_idxs = sorted(masked_terms_idxs, reverse=True)
    annotated_text = text
    for term_idx in sorted_masked_terms_idxs:
      start, end = terms_spans[term_idx]
      annotated_text = annotated_text[:start] + self.mask_text + annotated_text[end:]
    return annotated_text

  def get_annotations(self):
    annotations = {}
    for document in tqdm(self.documents):
      doc_annotations = []
      for split in document["splits"]:
        split_span = split["text_span"]
        terms_spans = split["terms_spans"]
        masked_terms_idxs = split["masked_terms_idxs"]
        sorted_masked_terms_idx = sorted(masked_terms_idxs, reverse=False)
        for term_idx in sorted_masked_terms_idx:
          local_span = terms_spans[term_idx]
          global_span = [split_span[0] + local_span[0],
                         split_span[0] + local_span[1]]
          doc_annotations.append(global_span)

      # Store annotations
      label = document["label"]
      name = self.label_to_name[label]
      annotations[name] = doc_annotations

    return annotations

  def tokenize_texts(self, texts):
    inputs = self.tokenizer(texts,
                            add_special_tokens=True,
                            padding="longest",  # Warning: If an input_text is longer than tokenizer.model_max_length, an error will raise on prediction
                            truncation=False,
                            return_offsets_mapping=True, # Requiered for annotations. Use it with ["offset_mapping"]
                            return_tensors="pt")
    return inputs

  def __len__(self):
    return len(self.documents)

  def __getitem__(self, index):
    document = self.documents[index]
    return document

#endregion

#region ###################################### Main CLI ######################################
if __name__ == "__main__":
    # Load configuration
    logging.info("######### START: CONFIGURATION #########")
    target_dir = argument_parsing()
    config = get_config_from_file(target_dir)
    petre = PETRE(**config)
    logging.info("######### END: CONFIGURATION #########")
    
    # Run all sections
    petre.run(verbose=True)
#endregion
