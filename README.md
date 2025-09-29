# PETRE - Privacy-preserving Entities Transparency and Re-identification Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Type Checking](https://img.shields.io/badge/type--checking-mypy-blue)](http://mypy-lang.org/)

PETRE is a professional Python package for evaluating privacy risks and implementing k-anonymity protection for text data containing personally identifiable information (PII).

## Features

🔒 **Privacy Protection**: Implement k-anonymity for text data
🔍 **Risk Evaluation**: Assess re-identification risks using machine learning
🛠️ **Flexible Architecture**: Modular design with pluggable components
⚡ **Performance**: GPU acceleration support (CUDA/MPS)
📊 **Explainability**: SHAP and Greedy explanations for anonymization decisions
🎯 **Professional**: Type-safe, well-tested, production-ready code

## Quick Start

### Installation

```bash
pip install petre
python -m spacy download en_core_web_lg
```

### Basic Usage

```python
import petre

# Create configuration
config = petre.create_app_config('config.json')

# Run privacy evaluation
results = petre.run_petre_from_config('config.json')

# Quick evaluation with overrides
results = petre.quick_evaluation(
    'config.json',
    mask_text='[REDACTED]',
    ks=[2, 3, 5]
)
```

### Command Line Interface

```bash
# Run PETRE with configuration file
petre config.json

# Validate configuration
petre-validate config.json

# Run with custom parameters
petre config.json --mask-text "[REDACTED]" --verbose

# Dry run (validate setup without execution)
petre config.json --dry-run
```

## Configuration

Create a JSON configuration file:

```json
{
  "output_base_folder_path": "./outputs",
  "data_file_path": "./data/dataset.json",
  "individual_name_column": "name",
  "original_text_column": "text",
  "starting_anonymization_path": "./data/annotations.json", 
  "tri_pipeline_path": "./models/tri-pipeline",
  "ks": [2, 3, 5],
  "mask_text": "",
  "use_mask_all_instances": true,
  "explainability_mode": "SHAP",
  "use_chunking": true
}
```

## Architecture

PETRE follows Go-ish Python patterns with clean architecture:

- **Interfaces**: Abstract base classes define contracts
- **Implementations**: Concrete classes with dependency injection
- **Context Managers**: Resource lifecycle management
- **Configuration**: Immutable dataclasses with validation
- **Smart Imports**: Dual execution support (script + module)

```python
# Professional API design
from petre import (
    AppConfig,
    PETREOrchestrator,
    TRIPipelineContext,
    SHAPExplainabilityMethod
)

# Create orchestrator with dependency injection
config = AppConfig(...)
orchestrator = petre.create_petre_orchestrator(config)

# Use context managers for resources
with petre.TRIPipelineContext(config) as pipeline:
    # Safe resource management
    results = pipeline.predict(texts)
```

## Advanced Usage

### Custom Components

```python
from petre import (
    ExplainabilityMethod,
    PETREComponentFactory,
    AppConfig
)

class CustomExplainer(ExplainabilityMethod):
    def explain(self, text, label, split, plot=False):
        # Custom explanation logic
        return term_weights

# Use with factory pattern
factory = PETREComponentFactory()
config = AppConfig(...)
explainer = CustomExplainer(config=config, pipeline=pipeline)
```

### Programmatic Interface

```python
import petre

# Create orchestrator programmatically
config = petre.AppConfig(
    output_base_folder_path="./outputs",
    data_file_path="./data.json",
    # ... other required fields
)

orchestrator = petre.create_petre_orchestrator(config)
orchestrator.initialize()
orchestrator.run_incremental_execution([2, 3, 5])
```

### Context Managers

```python
from petre import ModelResourceContext

# Manage all model resources together
with ModelResourceContext(config, load_shap=True) as resources:
    tri_pipeline = resources['tri_pipeline']
    spacy_model = resources['spacy_model'] 
    shap_explainer = resources['shap_explainer']
    
    # All resources automatically cleaned up
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/yamaceay/petre.git
cd petre
pip install -e ".[dev]"
python -m spacy download en_core_web_lg
```

### Running Tests

```bash
pytest
pytest --cov=petre
pytest -m "not slow"  # Skip slow tests
```

### Code Quality

```bash
black petre/
isort petre/
flake8 petre/
mypy petre/
```

### Project Structure

```
petre/
├── __init__.py           # Public API exports
├── interfaces.py         # Abstract base classes
├── core.py              # Concrete implementations
├── contexts.py          # Context managers
├── cli.py               # Command-line interface
├── config.py            # Configuration management
├── main.py              # Application entry point
├── import_utils.py      # Smart import utilities
├── py.typed             # Type checking marker
└── requirements.txt     # Dependencies
```

## Dependencies

### Core Requirements
- **torch**: Deep learning framework
- **transformers**: Hugging Face transformers
- **spacy**: NLP processing with en_core_web_lg model
- **shap**: Model explainability
- **pandas/numpy**: Data manipulation
- **datasets**: Dataset management

### Optional Dependencies
- **pytest**: Testing framework
- **black/isort**: Code formatting
- **mypy**: Type checking
- **sphinx**: Documentation

## API Reference

### Core Classes

- `AppConfig`: Immutable configuration with validation
- `PETREOrchestrator`: Main application orchestrator
- `DataProcessor`: Data loading and preprocessing
- `ModelManager`: Model and pipeline management
- `ExplainabilityMethod`: Explanation generation (SHAP/Greedy)
- `AnonymizationEngine`: K-anonymity implementation

### Context Managers

- `TRIPipelineContext`: TRI model resource management
- `SpaCyModelContext`: spaCy NLP model lifecycle
- `SHAPExplainerContext`: SHAP explainer management
- `ModelResourceContext`: Comprehensive resource management

### Utilities

- `smart_import()`: Dual execution import handling
- `create_app_config()`: Configuration factory
- `validate_configuration()`: Config validation
- `get_default_configuration()`: Default config template

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.

### Development Principles

1. **Type Safety**: Full type annotations with mypy checking
2. **Clean Architecture**: Interface-based design with dependency injection
3. **Resource Management**: Context managers for lifecycle control
4. **Testing**: Comprehensive test coverage with pytest
5. **Documentation**: Clear docstrings and examples

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PETRE in your research, please cite:

```bibtex
@software{petre2024,
  title={PETRE: Privacy-preserving Entities Transparency and Re-identification Evaluation},
  author={PETRE Development Team},
  year={2024},
  url={https://github.com/yamaceay/petre}
}
```

## Support

- 📚 [Documentation](https://petre.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/yamaceay/petre/issues)  
- 💬 [Discussions](https://github.com/yamaceay/petre/discussions)
- 📧 [Email Support](mailto:petre@example.com)

---

**PETRE** - Professional privacy evaluation for the modern data world.

## Table of Contents
* [Project structure](#project-structure)
* [Install](#install)
* [Usage](#usage)
  * [CLI](#cli)
  * [PETRE class](#petre-class)
* [Configuration](#configuration)
* [Results](#results)
* [Example](#example)


# Project structure
```
Privacy Enhancement for Text via Risk-oriented Explainability (PETRE)
│   README.md                               # This README
│   petre.py                                # Python program including the PETRE class, which can be used from Python code or CLI
│   requirements.txt                        # File generated with Conda containing all the required Python packages with specific versions
│   config.json                             # Example configuration file
└───data                                    # Folder with data files
    └───Wiki553                             # Folder for the dataset based on Wikipedia biographies
    │   │   Wiki553_BK=Original.json            # Panda's dataframe with 553 individuals, documents to protect and background knowledge formed by the documents to protect
    │   │   Wiki553_BK=Public.json              # Panda's dataframe with 553 individuals, documents to protect and background knowledge formed by the articles' bodies
    │   │   Wiki553_BK=Public+Original.json     # Panda's dataframe with 553 individuals, documents to protect and background knowledge formed by the articles' bodies and documents to protect 
    │   │   Annotations_St.NER3.json            # Anonymization annotations from Stanford NER3
    │   │   Annotations_St.NER4.json            # Anonymization annotations from Stanford NER4
    │   │   Annotations_St.NER7.json            # Anonymization annotations from Stanford NER7
    │   │   Annotations_Presidio.json           # Anonymization annotations from Microsoft Presidio
    │   │   Annotations_Word2Vec_t=0.25.json    # Anonymization annotations from the Word2Vec-based method using a threshold of 0.25
    │   │   Annotations_k_anonymity_Greedy.json # Anonymization annotations from k-anonymity greedy
    │   │   Annotations_k_anonymity_Random.json # Anonymization annotations from k-anonymity random
    │   │   Annotations_Manual.json             # Anonymization annotations from human annotators
    └───TAB                                 # Folder for the court rulings dataset
        |   TAB_test_BK=Original.json           # Panda's dataframe with 127 individuals, documents to protect and background knowledge formed by the documents to protect
        │   Annotations_test_St.NER7.json       # Anonymization annotations from Stanford NER7
        │   Annotations_test_Presidio.json      # Anonymization annotations from Microsoft Presidio
        │   Annotations_test_Manual.json        # Anonymization annotations from human annotators
```

# Install
Our implementation uses [Python 3.9.19](https://www.python.org/downloads/release/python-3919/) as programming language and [Conda](https://docs.conda.io/en/latest/) 24.1.2 for package management. All used packages and their respective versions are listed in the [requirements.txt](requirements.txt) file. 

To be able to run the code, follow these steps:
1. Install Conda if you haven't already.
2. Download this repository.
3. Open a terminal in the repository path.
4. Create a new Conda environment using the following command (channels included for ensuring that specific versions can be installed):
```console
conda create --name ENVIRONMENT_NAME --file requirements.txt -c conda-forge -c spacy -c pytorch -c nvidia -c huggingface -c numpy -c pandas
```
5. Activate the just created Conda environment using the following command:
```console
conda activate ENVIRONMENT_NAME
```
Continue with the steps of the [Usage section](#usage).

This has been tested in Windows 11 operating system, but should be compatible with Linux-based and Windows 10 systems.

# Usage
The PETRE method is implemented in the [petre.py](petre.py) script. They can be executed via [CLI](#cli) (Command Line Interface) or by importing the [PETRE class](#petre-class) directly into your Python code. The following sections provide detailed instructions for both approaches. Additionally, both methods offer configuration options (details on how in each subsection), which are described in the [Configuration section](#configuration).

## CLI
The CLI implementation only requires to pass as argument the path to a JSON configuration file. This file must contain a dictionary with the mandatory configurations, and can also contain optional configurations (see [Configuration section](#configuration)).

For example, for using the configuration file [config.json](config.json), run the following command:
```console
python petre.py config.json
```

## PETRE class
You can replicate the CLI behavior in a Python file by importing the `PETRE` class from the [petre.py](petre.py) script, instanciating the class and calling to its `run` method. The constructor requires the mandatory configurations as arguments, and also accepts optional configurations (see [Configuration section](#configuration)). Moreover, any of these configurations can be later modified by calling the `set_configs` method. During the execution, multiple loggings indicate the current block within the PETRE process. These loggings can be disabled by passing `verbose=False` as argument to the `run` method.

Here is a Python snippet that demonstrates how to run PETRE for all the starting anonymizations in the [data/Wiki553](data/Wiki553) folder:
```python
import os
from petre import PETRE

# Declare PETRE mandatory settings (placeholders for output and data paths) and some optional settings
petre = PETRE(output_base_folder_path="outputs/Wiki553",
              data_file_path="data/Wiki553/Wiki553_BK=Original.json",
              individual_name_column="name",
              original_text_column="original",
              starting_anonymization_path="To Be Defined",
              tri_pipeline_path="./TRI_Pipeline",
              ks=[2,3,5,7,10])

# Evaluate all the starting anonymizations in the Wiki553 folder
starting_anonymizations_folder = "data/Wiki553"
for annotations_file_name in os.listdir(starting_anonymizations_folder):
    # Only consider annotations files
    if annotations_file_name.startswith("Annotations"):
        # Set new starting_anonymization_path configuration
        starting_anonymization_path = os.path.join(starting_anonymizations_folder, annotations_file_name)
        petre.set_configs(starting_anonymization_path=starting_anonymization_path)        
        # Run PETRE for this starting anonymization
        petre.run(verbose=True)
```

## Configuration
In the following, we specify the configurations available for our implementation, as [CLI](#cli) (with the JSON file) or as [PETRE class](#petre-class) (with the constructor or `set_configs` method). For each configuration, we specify the name, type, if it is mandatory or optional (i.e., has a default value) and description.

* **Mandatory configurations**:
  * **output_base_folder_path | String | MANDATORY**: Determines the base folder were results will be stored (see [Results section](#results)). The folder will be created if it does not exist.
  * **data_file_path | String | MANDATORY**: Path to the data file containing the original documents to protect. The file is expected to define a Pandas dataframe stored in JSON or CSV format containing at least two columns:
    * *Individual name*: Column with the names of all the individuals, titled as defined in the `individual_name_column` setting.
    * *Original text*: Column with the original document to protect. It is expected that all individuals have a document to protect.
    Additional columns will have no effect on the behaviour of the code.
    Example of a dataframe with **three** individuals, each of them with an original document to protect:

        | name            | original                                                        |
        |-----------------|-----------------------------------------------------------------|
        | Joe Bean        | Bean received funding from his family to found UnderPants.      |
        | Ebenezer Lawson | Lawson, born in Kansas, has written multiple best-sellers.      |
        | Ferris Knight   | After a long race, Knight managed to obtain the first position. |  

  * **individual_name_column | String | MANDATORY**: Name of the dataframe column corresponding to the individual name. In the previous example, it will be `name`.
  * **original_text_column | String | MANDATORY**: Name of the column corresponding to the original document to protect for each individual. In the previous example, it will be `original`.
  * **starting_anonymization_path | String | MANDATORY**: Path to the annotations to be used as starting anonymization. Annotations must be stored in JSON formatted dictionary, with individual's `name` (defined in the `data_file_path`) as key and a list of the masked spans as value. It is expected that all individuals have annotations.
  * **tri_pipeline_path | String | MANDATORY**: Path to the [Transformers' pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) containing the text re-identification (TRI) model and tokenizer. Refer to the [Text Re-Identification repository](https://github.com/BenetManzanaresSalor/TextRe-Identification) for the creation of this pipeline. If the dataframe at `data_file_path` contains, apart from the original documents and individuals' names, a column with the background knowledge for each individual, it can be leveraged for the creation of this pipeline (more details in the [Example](#example) section). *NOTE: You can also use a path to a pipeline from the [HuggingFace models' repository](https://huggingface.co/models).*
  * **ks | List of integers | MANDATORY**: Sorted list of the $k$ values to be used by PETRE for incremental execution. For instance, with `ks=[2,3]`, annotations generated for $k=2$ will serve as a starting point for ensuring $k=3$. This approach helps reduce execution time, as annotations for $k=X$ are assumed to be a superset of those for $k<X$. Since this $k$ values refer to probabilistic $k$-anonymity, $k$ must be greater than 1 in any case, with $k=1$ providing no protection and $k=2$ offering the mininum protection.

* **Optional configurations**:
  * **mask_text | String | Default=""**: Text by which each annotated term will be replaced prior to the re-identification attack. It might influence the accuracy of the `TRI_Pipeline`.
  * **use_mask_all_instances | Boolean | Default=true**: Once PETRE finds a term as the most disclosive, whether to mask all instances of that term in the whole document. If it is `false`, only the instance found to be most revealing will be masked. Although this may lead to unnecessary masking, it can significantly reduce PETRE's runtime.
  * **explainability_mode | String | Default="SHAP"**: The method used to identify the term that contributes most to re-identification. Options are `"SHAP"` and `"Greedy"`. In our paper, we employ `"SHAP"`, a well-known explainability method with an [official Python implemention](https://github.com/shap/shap). Based on Shapley values, this method has a computational complexity of O(n²). Alternatively, we defined `"Greedy"`, a straightforward approach that calculates term's weight by assessing the change in re-identification risk caused by its masking. Although the `"Greedy"` approach produces slightly worse results (<5% difference), it offers a significant reduction in runtime (i.e., from 3 hours to 5 minutes).
  * **use_chunking | Boolean | Default=true**: Whether to consider noun chunks and named entities what determining what is a term.

## Results
After execution of PETRE (both from CLI or Python code), a folder with same name as the starting anonymization file (defined in `starting_anonymization_path`) will be created in the `output_base_folder_path`. This folder will contain:
* **Annotations_PETRE_k=X.json**: File containing the annotations created by PETRE for $k=X$. A separate file will be created for each $k$ specified in the `ks` setting.
* **Ranks_k=X.csv**: Considering the annotations created by PETRE for $k=X$, this file lists the re-identification ranks for each individual's document, organized alphabetically by individual name. The rank indicates the position within the re-identification probability distribution, where a rank of 1 represents the highest predicted probability for re-identification, signifying the maximum risk. A separate file will be created for each $k$ specified in the `ks` setting.


## Example
The [config.json](config.json) provides an example of a valid configuration. It uses data from the the [data/Wiki553](data/Wiki553) folder, including multiple multiple starting anonymizations' annotations and three Pandas dataframes with different background knowledges (abreviated as `BK`). The [config.json](config.json) is set up to use the [Wiki553_BK=Original.json](data/Wiki553/Wiki553_BK=Original.json) for original documents and individuals' names, a re-identification pipeline located at `./TRI_Pipeline` (not included here due to the 100MB storage limit), multiple $k$ values ranging from 2 to 10, and [Annotations_k_anonymity_Greedy.json](data/Wiki553/Annotations_k_anonymity_Greedy.json) as starting anonymization. This is the most strict starting anonymization, requiring less enhancement from PETRE and thus minimizing this example's runtime (see **our paper** for more details).

The [Wiki553_BK=Original.json](data/Wiki553/Wiki553_BK=Original.json) contains, apart from the "name" and "original" columns, a column named "background_knowledge" with the background knowledge for each individual, consisting of the documents themselves to be protected. In this way, this dataframe can be directly used for the creation of the `TRI_Pipeline` using the [Text Re-Identification repository](https://github.com/BenetManzanaresSalor/TextRe-Identification). Use the filepath for the `data_file_path` setting, "name" for the `individual_name_column` setting, "background_knowledge" for the `background_knowledge_column` setting and "No" for the `finetuning_sliding_window` setting (so it is trained to re-identify text at sentence level instead of with sliding window).

Feel free to modify [config.json](config.json) to use any other consigurations, such as dataframes, re-identification pipelines, $k$ values or starting anonymizations. This includes the dataframe and starting anonymizations from the [data/TAB](data/TAB) folder. If modifying settings other than `starting_anonymization_path`, we recommend also updating the `output_base_folder_path` setting to prevent overwriting results.
