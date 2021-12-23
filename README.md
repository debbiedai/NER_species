# Finetuning BioBert for arthropod species name Named Entity Recognition

Project aims to collect a literature corpus as our training and testing data with automated or manual labeled entities, from abstracts in the arthropod sciences. We finetuned the [BioBert](https://github.com/dmis-lab/biobert-pytorch) to perform named-entity recognition (NER) for arthropod species name.


### Requirment
`Python3` and `Colab`<br>
The introduction of [Colab](https://colab.research.google.com/?utm_source=scs-index#scrollTo=5fCEDCU_qrC0).

### Installation
- seqeval : Used for evaluation (`pip install seqeval`)
- inflect (`pip install inflect`)
- nltk (`pip install nltk`)
- sklearn (`pip install scikit-learn`)
- transformers (`pip install transformers`)
- beautifulsoup4 (`pip install beautifulsoup4`)

### Preprocess
Before training, please run `preprocess.py` and `same_len.py` to process the dataset.

`preprocess.py`
- split our data into 10 folds (10 folds cross validation)
- convert .xml file to .tsv
- convert .tsv to .txt (generate with/without text name .txt file)

`same_len.py` (run `same_len.py` on with/without text name .txt file)
- preprocess to same length
- create train_dev.txt, devel.txt, test.txt and labels.txt

### Finetuning BioBert

Put the dataset (The directory contain train_dev.txt, devel.txt, test.txt and labels.txt) and `args.json`, `run_ner.ipynb` and `utils_ner.ipynb` on Colab.
The `args.json` stores the setting argument when training. (Remind: "max_seq_length" is the same as "max_len" in same_len.py) Because of 10 folds cross validation, please change the setting of args when you train the next fold.

### The result of 10 folds cross validation

| Test Fold      |    Test Precision (%)   |    Test Recall (%)   |    Test F1 (%)   |
|----------------|:-----------------------:|:--------------------:|:----------------:|
| fold_0         |          78.84          |         91.11        |       84.53      |
| fold_1         |          93.81          |         91.00        |       92.38      |
| fold_2         |          82.66          |         89.85        |       86.11      |
| fold_3         |          80.68          |         86.58        |       83.53      |
| fold_4         |          78.94          |         96.77        |       86.95      |
| fold_5         |          77.77          |         74.66        |       76.19      |
| fold_6         |          79.78          |         94.93        |       86.70      |
| fold_7         |          84.69          |         89.24        |       86.91      |
| fold_8         |          91.48          |         74.78        |       82.29      |
| fold_9         |          88.88          |         74.57        |       81.11      |
| Average        |          83.75          |         86.35        |       84.67      |

### Postprocess

After training, the test prediction of each fold would be saved in output directory. Please download test_predictions.txt and rename it. (For example: test_predictions_v0_t1.txt)

In `postprocess.py`, you can add text name in the test prediction and postprocess to the output format we want.