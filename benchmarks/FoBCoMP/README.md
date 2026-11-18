---
dataset_info:
  features:
  - name: correct_text
    dtype: string
  - name: wrong_text
    dtype: string
  splits:
  - name: qa_shuffled
    num_bytes: 113505196
    num_examples: 21867
  - name: qa_generated
    num_bytes: 9774387
    num_examples: 2000
  - name: sentiment_articles
    num_bytes: 280134
    num_examples: 84
  - name: sentiment_sentences
    num_bytes: 28287
    num_examples: 91
  - name: topic_classification
    num_bytes: 883106
    num_examples: 234
  download_size: 56046707
  dataset_size: 124471110
configs:
- config_name: default
  data_files:
  - split: qa_shuffled
    path: data/qa_shuffled-*
  - split: qa_generated
    path: data/qa_generated-*
  - split: sentiment_articles
    path: data/sentiment_articles-*
  - split: sentiment_sentences
    path: data/sentiment_sentences-*
  - split: topic_classification
    path: data/topic_classification-*
language:
- fo
pretty_name: 'FoBCoMP: Minimal Pair Probes for Faroese Text Comprehension'
license: cc
---

A collection of minimal pair probes for Faroese text comprehension, covering extractive QA, sentiment analysis, and topic classification. 

The benchmark inherits the Creative Commons Attribution Non-Commercial ShareAlike 4.0 license from its source datasets. 


See [our preprint](https://arxiv.org/pdf/2510.00810) for more information. 

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

Also cite the source datasets when using this benchmark: 
```bibtex
@inproceedings{debess-etal-2024-good,
    title = "Good or Bad News? Exploring {GPT}-4 for Sentiment Analysis for {F}aroese on a Public News Corpora",
    author = "Debess, Iben Nyholm  and
      Simonsen, Annika  and
      Einarsson, Hafsteinn",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.690/",
    pages = "7814--7824",
    abstract = "Sentiment analysis in low-resource languages presents unique challenges that Large Language Models may help address. This study explores the efficacy of GPT-4 for sentiment analysis on Faroese news texts, an uncharted task for this language. On the basis of guidelines presented, the sentiment analysis was performed with a multi-class approach at the sentence and document level with 225 sentences analysed in 170 articles. When comparing GPT-4 to human annotators, we observe that GPT-4 performs remarkably well. We explored two prompt configurations and observed a benefit from having clear instructions for the sentiment analysis task, but no benefit from translating the articles to English before the sentiment analysis task. Our results indicate that GPT-4 can be considered as a valuable tool for generating Faroese test data. Furthermore, our investigation reveals the intricacy of news sentiment. This motivates a more nuanced approach going forward, and we suggest a multi-label approach for future research in this domain. We further explored the efficacy of GPT-4 in topic classification on news texts and observed more negative sentiments expressed in international than national news. Overall, this work demonstrates GPT-4`s proficiency on a novel task and its utility for augmenting resources in low-data languages."
}
```

```bibtex
@inproceedings{debess-etal-2024-good,
    title = "Good or Bad News? Exploring {GPT}-4 for Sentiment Analysis for {F}aroese on a Public News Corpora",
    author = "Debess, Iben Nyholm  and
      Simonsen, Annika  and
      Einarsson, Hafsteinn",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.690/",
    pages = "7814--7824",
    abstract = "Sentiment analysis in low-resource languages presents unique challenges that Large Language Models may help address. This study explores the efficacy of GPT-4 for sentiment analysis on Faroese news texts, an uncharted task for this language. On the basis of guidelines presented, the sentiment analysis was performed with a multi-class approach at the sentence and document level with 225 sentences analysed in 170 articles. When comparing GPT-4 to human annotators, we observe that GPT-4 performs remarkably well. We explored two prompt configurations and observed a benefit from having clear instructions for the sentiment analysis task, but no benefit from translating the articles to English before the sentiment analysis task. Our results indicate that GPT-4 can be considered as a valuable tool for generating Faroese test data. Furthermore, our investigation reveals the intricacy of news sentiment. This motivates a more nuanced approach going forward, and we suggest a multi-label approach for future research in this domain. We further explored the efficacy of GPT-4 in topic classification on news texts and observed more negative sentiments expressed in international than national news. Overall, this work demonstrates GPT-4`s proficiency on a novel task and its utility for augmenting resources in low-data languages."
}
```