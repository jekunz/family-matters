---
dataset_info:
  features:
  - name: correct_text
    dtype: string
  - name: wrong_text
    dtype: string
  splits:
  - name: translation_pairs
    num_bytes: 156583
    num_examples: 680
  - name: multiblimp
    num_bytes: 27419
    num_examples: 232
  - name: scala_delete
    num_bytes: 152705
    num_examples: 601
  - name: scala_flip
    num_bytes: 140255
    num_examples: 552
  - name: germdetect
    num_bytes: 422806
    num_examples: 2026
  download_size: 484801
  dataset_size: 899768
configs:
- config_name: default
  data_files:
  - split: translation_pairs
    path: data/translation_pairs-*
  - split: multiblimp
    path: data/multiblimp-*
  - split: scala_delete
    path: data/scala_delete-*
  - split: scala_flip
    path: data/scala_flip-*
  - split: germdetect
    path: data/germdetect-*
language:
- fo
pretty_name: 'FoBLiMP: Minimal Pair Probes for Faroese Linguistic Acceptability'
license: cc
---

A collection of minimal pair probes for Faroese linguistic acceptability, covering 
pairs of better and worse translations (as rated by human annotators in [Scalvini et al. 2025](https://aclanthology.org/2025.nodalida-1.62/)),
automatic corruptions of subject-verb agreement ([Jumelet et al. 2025](https://arxiv.org/abs/2504.02768)) and verb placement ([Michael and Hornbach 2025](https://aclanthology.org/2025.bea-1.59/)),
and automatic corruptions that delete words or flip adjacent words ([Nielsen et al. 2023](https://aclanthology.org/2023.nodalida-1.20/)). 


See [our preprint](https://arxiv.org/pdf/2510.00810) for more information. 

The licenses are interited from the source datasets: 

* MultiBLiMP: Creative Commons Attribution 4.0
* ScaLA, Translation Pairs: Creative Commons Attribution ShareAlike 4.0 
* GermDetect: Apache License Version 2.0

### Citation

```bibtex
@misc{kunz2025familymatterslanguagetransfer,
      title={Family Matters: Language Transfer and Merging for Adapting Small LLMs to Faroese}, 
      author={Jenny Kunz and Iben Nyholm Debess and Annika Simonsen},
      year={2025},
      eprint={2510.00810},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.00810}, 
}
```

### Citation Source Datasets

Please also cite the source datasets when using this benchmark: 
```bibtex
@inproceedings{scalvini-etal-2025-rethinking,
    title = "Rethinking Low-Resource {MT:} The Surprising Effectiveness of Fine-Tuned Multilingual Models in the {LLM} Age",
    author = "Scalvini, Barbara  and
      Debess, Iben Nyholm  and
      Simonsen, Annika  and
      Einarsson, Hafsteinn",
    editor = "Johansson, Richard  and
      Stymne, Sara",
    booktitle = "Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies (NoDaLiDa/Baltic-HLT 2025)",
    month = mar,
    year = "2025",
    address = "Tallinn, Estonia",
    publisher = "University of Tartu Library",
    url = "https://aclanthology.org/2025.nodalida-1.62/",
    pages = "609--621",
    ISBN = "978-9908-53-109-0",
    abstract = "This study challenges the current paradigm shift in machine translation, where large language models (LLMs) are gaining prominence over traditional neural machine translation models, with a focus on English-to-Faroese translation. We compare the performance of various models, including fine-tuned multilingual models, LLMs (GPT-SW3, Llama 3.1), and closed-source models (Claude 3.5, GPT-4). Our findings show that a fine-tuned NLLB model outperforms most LLMs, including some larger models, in both automatic and human evaluations. We also demonstrate the effectiveness of using LLM-generated synthetic data for fine-tuning. While closed-source models like Claude 3.5 perform best overall, the competitive performance of smaller, fine-tuned models suggests a more nuanced approach to low-resource machine translation. Our results highlight the potential of specialized multilingual models and the importance of language-specific knowledge. We discuss implications for resource allocation in low-resource settings and suggest future directions for improving low-resource machine translation, including targeted data creation and more comprehensive evaluation methodologies."
}
```

```bibtex
@misc{jumelet2025multiblimp10massivelymultilingual,
      title={MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs}, 
      author={Jaap Jumelet and Leonie Weissweiler and Arianna Bisazza},
      year={2025},
      eprint={2504.02768},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.02768}, 
}
```

```bibtex

@inproceedings{nielsen-2023-scandeval,
    title = "{S}cand{E}val: A Benchmark for {S}candinavian Natural Language Processing",
    author = "Nielsen, Dan",
    editor = {Alum{\"a}e, Tanel  and
      Fishel, Mark},
    booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may,
    year = "2023",
    address = "T{\'o}rshavn, Faroe Islands",
    publisher = "University of Tartu Library",
    url = "https://aclanthology.org/2023.nodalida-1.20/",
    pages = "185--201",
    abstract = "This paper introduces a Scandinavian benchmarking platform, ScandEval, which can benchmark any pretrained model on four different tasks in the Scandinavian languages. The datasets used in two of the tasks, linguistic acceptability and question answering, are new. We develop and release a Python package and command-line interface, scandeval, which can benchmark any model that has been uploaded to the Hugging Face Hub, with reproducible results. Using this package, we benchmark more than 80 Scandinavian or multilingual models and present the results of these in an interactive online leaderboard, as well as provide an analysis of the results. The analysis shows that there is substantial cross-lingual transfer among the the Mainland Scandinavian languages (Danish, Swedish and Norwegian), with limited cross-lingual transfer between the group of Mainland Scandinavian languages and the group of Insular Scandinavian languages (Icelandic and Faroese). The benchmarking results also show that the investment in language technology in Norway and Sweden has led to language models that outperform massively multilingual models such as XLM-RoBERTa and mDeBERTaV3. We release the source code for both the package and leaderboard."
}

```

```bibtex
@inproceedings{michael-horbach-2025-germdetect,
    title = "{G}erm{D}etect: Verb Placement Error Detection Datasets for Learners of {G}ermanic Languages",
    author = "Michael, Noah-Manuel  and
      Horbach, Andrea",
    editor = {Kochmar, Ekaterina  and
      Alhafni, Bashar  and
      Bexte, Marie  and
      Burstein, Jill  and
      Horbach, Andrea  and
      Laarmann-Quante, Ronja  and
      Tack, Ana{\"i}s  and
      Yaneva, Victoria  and
      Yuan, Zheng},
    booktitle = "Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.bea-1.59/",
    doi = "10.18653/v1/2025.bea-1.59",
    pages = "818--829",
    ISBN = "979-8-89176-270-1",
    abstract = "Correct verb placement is difficult to acquire for second-language (L2) learners of Germanic languages. However, word order errors and, consequently, verb placement errors, are heavily underrepresented in benchmark datasets of NLP tasks such as grammatical error detection (GED)/correction (GEC) and linguistic acceptability assessment (LA). If they are present, they are most often naively introduced, or classification occurs at the sentence level, preventing the precise identification of individual errors and the provision of appropriate feedback to learners. To remedy this, we present \textbf{GermDetect}: Universal Dependencies-based (UD), linguistically informed verb placement error \textbf{detect}ion datasets for learners of \textbf{Germ}anic languages, designed as a token classification task. As our datasets are UD-based, we are able to provide them in most major Germanic languages: Afrikaans, German, Dutch, Faroese, Icelandic, Danish, Norwegian (Bokm{\r{a}}l and Nynorsk), and Swedish. We train multilingual BERT (mBERT) models on GermDetect and show that linguistically informed, UD-based error induction results in more effective models for verb placement error detection than models trained on naively introduced errors. Finally, we conduct ablation studies on multilingual training and find that lower-resource languages benefit from the inclusion of structurally related languages in training."
}
```