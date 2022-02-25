# typ_embed

This repository contains the code for incorporating typological feature information into NLP word embeddings to create **typological word embeddings** and applying these to four NLP tasks. The implementation has several embedding models using an LSTM and BERT. Typological feature information is extracted the URIEL database to get a binary feature vector and then incorporated into the word embedding using either concatentation or simple attention (additive or multiplicative). The outputted embeddings are then benchmarked on three probing tasks, Dependency Parsing, POS Tagging and Entailment Classification, and one downstream task, Extractive Text Summarization. In ths repository, a Biaffine Dependency Parser (Dozat and Manning 2017) a MLP POS tagger, a MLP Entailment classifier, and a Layer Normalized LSTM for Text Summarization are implemented.

## Requirements
* Python 3.6.9
* Pytorch 1.9.0
* lang2vec 1.1.2
* NumPy 1.19.5
* transformers 4.5.1
* stanza 1.0.0

## Datasets
We use four different datasets in this model. Each of the links to download them are provided here:
* Dependency Parsing and POS Tagging: Download CoNLL-U English_EWT dataset from https://universaldependencies.org/
* Entailment Classification: SICK dataset from https://marcobaroni.org/composes/sick.html
* Extractive Text Summarization: BBC News Summary dataset from https://www.kaggle.com/pariza/bbc-news-summary
Once downloaded, place these in a directory called `datasets` in the `typ_embed` directory.

## Install
It is recommended to install a Python virtual environment either using `venv` or through Anaconda to obtain all package dependencies. Once installed, clone the repository either using 
```
git clone https://github.com/vsubramaniam851/typ_embed.git
```
or
```
git clone git@github.com:vsubramaniam851/typ_embed.git
```

## Run
Run using typ_embed.py file on command line. To specify which task to run, subparsers are used so use `dep` for Dependency Parsing, `pos` for POS tagging, `entail` for Entailment Classification, `sum` for Extractive Text Summarization after the `python` call. For example to train a new Dependency Parsing model dep_model.pt with typological embeddings incorporated using additive attention, use the following function call,
```
python typ_embed.py dep -t True -m dep_model.pt -ty True -te add_att 
```
To evaluate the model `dep_model.pt` after it has completed training,
```
python typ_embed.py dep -m dep_model.pt -ty True -te add_att 
```
All modifiable parameters are described in the typ_embed.py file. These include incorporating typological features (`-ty True`), the size of the typological embedding (`-tes 32`), and what method to use for incorporating typological embeddings into word embeddings (attention: `-te add_att`, `-te mul_att`, or concatentation: `-te concat`). Every data loading, train, and eval files have test functions to ensure all main functions in the file are working that can be run separately for the sake of debugging.


## Citations
[1] T. Dozat and C. Manning. (2017). Deep Biaffine Attention for Neural Dependency Parsing (https://arxiv.org/abs/1611.01734)
[2] Y. Liu. (2019). Fine-tune BERT for Extractive Summarization (https://arxiv.org/abs/1903.10318)
[3] MC. de Marneffe et. al. (2021) Universal Dependencies (https://direct.mit.edu/coli/article/47/2/255/98516/Universal-Dependencies)
[4] D. Greene and P. Cunnigham. (2006). Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering (http://mlg.ucd.ie/files/publications/greene06icml.pdf)
[5] Marellei et. al. (2014). A SICK cure for evaluation of compositional distributional semantic models (https://aclanthology.org/L14-1314.pdf)
