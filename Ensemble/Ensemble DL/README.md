# Sentiment analysis with CNN-LSTM-BERT

## What are ensemble learning method

We use 3 pretrained classifier on Reddit dataset: CNN, CNN-LSTM, and FinBERT to predict the sentiment of this dataset. We use these predictions as inputs for an DNN to output the final prediction.

## Data

Data used in this experiment are:

- Self-labelled crawlled Reddit dataset: 2047 samples (a.k.a. Reddit dataset)
- Remaining crawlled Reddit dataset: 8409 samples (a.k.a. Serving dataset)

## Training and evaluation

For model testing, we split the dataset into 80:10:10 train:validation:test dataset and use this test dataset metrics to pick the best optimal model. For both FinBERT, CNN, and CNN-LSTM model, we use the best pretrained weight (which trained on Combined data) to predict for the input of DNN.

## Results

We evaluate the model performance by using f1 score. As our data distribution is not balanced between neutral and (positive, negative), it will be bias if we use accuracy as the main metric to evaluate.

The resultis below:

{'accuracy': 0.97, 'f1': 0.96, 'precision': 0.96, 'recall': 0.96}

## Evaluation

The ensemble method increases the model performance on test set significantly compared to previous models.

## Folder structure

All the related files can be found under /FinBERT folder with structure:

    .
    ├── ...
    ├── Ensemble                        # Ensemble files
    |   ├── Emsemble-DL                 # Results files for Emsemble - DL
    |   |   ├── results                 # Results files for Emsemble
    |   |       └── Model serving       # Serving prediction
    |   └── Emsemble-ML                 # Results files for Emsemble - ML
    └── ...

The model weights are stored in this link:
<https://entuedu-my.sharepoint.com/:f:/g/personal/hienvan001_e_ntu_edu_sg/EsEmSgYJ-0REinCTegCClz4BCABZK3XfgjJBXwgTYldQ_A?e=qIN78l>

## References
