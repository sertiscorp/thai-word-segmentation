# Thai word segmentation with bi-directional RNN

This is code for preprocessing data, training model and inferring word segment boundaries of Thai text
with bi-directional recurrent neural network. The model provides precision of 99.04%, recall of 99.31%
and F1 score of 99.18%. Please see the [blog post](https://sertiscorp.com/thai-word-segmentation-with-bi-directional_rnn/)
for the detailed description of the model.

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

Note that the InterBEST 2009 corpus is not included, but can be downloaded from the
[NECTEC website](https://thailang.nectec.or.th/downloadcenter/).

## Usage

To try the prediction demo, run `python3 predict_example.py`.
To preprocess the data and train the model, put the data files under `data` directory and then
run `python3 preprocess.py` and `python3 train.py`.

## Contributors

* Jussi Jousimo
* Natsuda Laokulrat
* Ben Carr

## License

GPL 3.0

Copyright (c) Sertis Co., Ltd., 2017
