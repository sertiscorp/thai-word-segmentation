# Thai word segmentation with bi-directional RNN

Please see the [Sertis blog](https://sertiscorp.com/thai-word-segmentation-with-bi-directional_rnn/) for the description
of the model and usage.

## Requirements

* Python 3.4
* TensorFlow 1.4
* NumPy 1.13
* scikit-learn 0.18

## Files

* `preprocess.py`: Preprocess corpus for model training
* `train.py`: Train the Thai word segmentation model
* `predict_example.py`: Example usage of the model to segment Thai words
* `saved_model`: Pretrained model weights
* `thainlplib/labeller.py`: Methods for preprocessing the corpus
* `thainlplib/model.py`: Methods for training the model

Note that the InterBEST 2009 corpus is not included, but can be downloaded from [https://thailang.nectec.or.th/downloadcenter/](https://thailang.nectec.or.th/downloadcenter/).

Copyright (c) Sertis Co., Ltd., 2017
