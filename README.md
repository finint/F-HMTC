# F-HMTC: Detecting Financial Events for Investment Decisions Based on Neural Hierarchical Multi-Label Text Classification

## Introduction
The share prices of listed companies in the stock trading market are prone to be influenced by various events. Performing event detection could help people to timely identify investment risks and opportunities accompanying these events. The financial events inherently present hierarchical structures, which could be represented as tree-structured schemes in real-life applications, and detecting events could be modeled as a hierarchical multi-label text classification problem, where an event is designated to a tree node with a sequence of hierarchical event category labels. F-HMTC is a hierarchical multi-label text classification method for financial detection proposed by us, which is accepted by IJCAI 2020. Readers are welcomed to fork this repository and follow our work. Most of the baseline experiments in this article are are implemented with Tencent open-source toolkit [NeuralNLP-NeuralClassifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier), and we are grateful for this.



## Data
The financial event scheme involved in the article has 7 levels, and there are 98 event nodes, except the ROOT node. We provide a complete event system file fmc_event_category_json, and in order to achieve data desensitization, we provide labels in the form of aliases, and extract one-tenth of the training set and test set separately to form the new data set fmc_selected we released.

## Requriement
- Tensorflow 1.10.0
- Keras 2.2.4
- Numpy 1.17.2

## Cite
```

@inproceedings{ijcai2020-619,
  title     = {F-HMTC: Detecting Financial Events for Investment Decisions Based on Neural Hierarchical Multi-Label Text Classification},
  author    = {Liang, Xin and Cheng, Dawei and Yang, Fangzhou and Luo, Yifeng and Qian, Weining and Zhou, Aoying},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  pages     = {4490--4496},
  year      = {2020},
  month     = {7},
  doi       = {10.24963/ijcai.2020/619},
}
```



