# Privacy Enhancement for Text via Risk-oriented Explainability (PETRE)
This repository contains the code and data for the text anonymization enhancement method presented in *B. Manzanares-Salor, D. Sánchez, Enhancing text anonymization via re-identification risk-based explainability, Submitted, (2023)*. In addition, some results obtained during testing are provided.

The data was extracted from the [bootstrapping-anonymization repository](https://github.com/anthipapa/bootstrapping-anonymization), corresponding to the publication [*Papadopoulou, A., Lison, P., Øvrelid, L., Pil ́an, I., 2022. Bootstrapping text anonymization models with distant supervision, in: Proceedings of the Thirteenth Language Resources and Evaluation Conference, European Language Resources Association, Marseille, France. pp. 4477–4487*](https://aclanthology.org/2022.lrec-1.476). It has been extended by adding boidies of individuals' biographies and annotations made by multiple automatic anonymization approaches. The resulting data can be found in the [Datasets/Wiki553](Datasets/Wiki553) folder.

The code is presented in the [PETRE.ipynb](PETRE.ipynb) notebook. The project can be run locally (using [Jupyter](https://jupyter.org/)) or in [Google Colab](https://colab.research.google.com/). If Colab is used, it is necessary to upload the contents of this repository to a [Google Drive](https://drive.google.com/) folder, so that Colab can access it. Use the [Text Re-Identification repository](https://github.com/BenetManzanaresSalor/TextRe-Identification), corresponding to the publication [*B. Manzanares-Salor, D. Sánchez, P. Lison, Automatic Evaluation of Disclosure Risks of Text Anonymization Methods, Privacy in Statistical Databases, (2022)*](https://link.springer.com/chapter/10.1007/978-3-031-13945-1_12), for obtaining the required re-identification pipeline.

# Project structure
```
Privacy Enhancement for Text via Risk-oriented Explainability
│   README.md                                       # This README
│   PETRE.ipynb                                     # Code notebook
└───Datasets                                        # Folder for all datasets
│   └───Wiki553                                     # Dataset folder
│       └───Annotations                             # Folder for all anonymizations
│       │   │   Annotations_Manual.json             # List of manually-crafted annotations
│       │   │   Annotations_Presidio.json           # List of annotations made by Presidio
│       │   │   Annotations_Gold.json               # Human-based reference annotations for precision/recall evaluation
│       │   │   ...
│       │
│       └───Wiki553.json                            # Pandas dataframe with abstracts and bodies of individuals' biographies
└───Outputs                                         # Folder for PETRE's results for all datasets
    └───Wiki553                                     # Results for the used dataset
        └───BG_Abstracts                            # Results only using articles' abstracts as background knowledge
        │   └───Annotations_St.NER4                 # Folder for results using Stanford NER4 as starting anonymization
        │   │   │   Annotations_PETRE_Start.json    # Initial annotations after PETRE's chunking
        │   │   │   Annotations_PETRE_k=2.json      # PETRE annotations for k=2
        │   │   │   ...
        │   │    
        |   └───Original_Ranks                      # Ranks per document for each starting anonymization
        │   │   │   ...
        │   │
        │   └───PETRE_Ranks                         # Ranks per document for each PETRE's output
        │       │   ...
        │
        └───BG_Bodies                               # Results only using articles' bodies as background knowledge
        │    │   ...
        │
        └───BG_Bodies+Abstracts                     # Results using both articles' bodies and abstracts as background knowledge
            │   ...
```
