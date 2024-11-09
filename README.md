# Haitian Creole NLU Model

Inspired by the CreoleVal paper, this project serves to recreate the Reading Comprehension NLU task. Our goal is to build upon the results achieved by the authors of said paper.

## Setup

**Creating an Environment:**

*Note, requires python 3.8 or greater*

**If using python venv, run:**

```
python -m venv `<your_venv_name>`
```

*Bash/Zsh Activation*

```
#bash/zsh
source <your_venv_name>/bin/activate 
```

*Windows Activation*

```
<your_venv_name>\Scripts\activate.bat
```

**If using conda, run:**

```
conda create --name your_env_name python=3.8
conda activate your_env_name
```

**Installing Dependencies:**

```
conda install pip #If using conda
pip install -r requirements.txt
```

## Running

## Citing

* CreoleVal paper
* ```
  @article{10.1162/tacl_a_00682,
      author = {Lent, Heather and Tatariya, Kushal and Dabre, Raj and Chen, Yiyi and Fekete, Marcell and Ploeger, Esther and Zhou, Li and Armstrong, Ruth-Ann and Eijansantos, Abee and Malau, Catriona and Heje, Hans Erik and Lavrinovics, Ernests and Kanojia, Diptesh and Belony, Paul and Bollmann, Marcel and Grobol, Loïc and Lhoneux, Miryam de and Hershcovich, Daniel and DeGraff, Michel and Søgaard, Anders and Bjerva, Johannes},
      title = "{CreoleVal: Multilingual Multitask Benchmarks for Creoles}",
      journal = {Transactions of the Association for Computational Linguistics},
      volume = {12},
      pages = {950-978},
      year = {2024},
      month = {09},
      abstract = "{Creoles represent an under-explored and marginalized group of languages, with few available resources for NLP research. While the genealogical ties between Creoles and a number of highly resourced languages imply a significant potential for transfer learning, this potential is hampered due to this lack of annotated data. In this work we present CreoleVal, a collection of benchmark datasets spanning 8 different NLP tasks, covering up to 28 Creole languages; it is an aggregate of novel development datasets for reading comprehension relation classification, and machine translation for Creoles, in addition to a practical gateway to a handful of preexisting benchmarks. For each benchmark, we conduct baseline experiments in a zero-shot setting in order to further ascertain the capabilities and limitations of transfer learning for Creoles. Ultimately, we see CreoleVal as an opportunity to empower research on Creoles in NLP and computational linguistics, and in general, a step towards more equitable language technology around the globe.}",
      issn = {2307-387X},
      doi = {10.1162/tacl_a_00682},
      url = {https://doi.org/10.1162/tacl\_a\_00682},
      eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00682/2468651/tacl\_a\_00682.pdf},
  }

  ```
* Project setup was inspired by the following:
  * [Ultimate Setup for Your Next Python Project](https://martinheinz.dev/blog/14)
