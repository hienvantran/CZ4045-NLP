# Sentiment analysis with CNN-LSTM-BERT

## What are ensemble learning method

We use 3 ML models RandomForestClassifier, SVC and LogisticRegression(random_state = 1, max_iter = 200) to perform prediction for our Reddit dataset. For ensembling method, we make use of AdaBoostClassifier with base_estimator is DecisionTreeClassifier to get the final prediction.

## Data

Data used in this experiment are:

- Financial Phrase Bank dataset: 4837 samples
- Self-labelled crawlled Reddit dataset: 2047 samples (a.k.a. Reddit dataset)
- Remaining crawlled Reddit dataset: 8409 samples (a.k.a. Serving dataset)
- Combination of Financial Phrase Bank dataset and Reddit dataset (a.k.a. Combined dataset)

## Training and evaluation

We consider 4 training scenarios as follows:

- Training only on Kaggle dataset (model 1)
- Training only on Reddit dataset (model 2)
- Transfer learning from Kaggle to Reddit dataset (model 3)
- Training on Combined dataset (model 4)

For model testing, we split the dataset into 80:10:10 train:validation:test dataset and use this test dataset metrics to pick the best optimal model. We only consider 3 models 2, 3, 4 as they're trained with our domain task for Reddit data. The first model is only for model 3 and reference. Although the test data between model (2,3) and model 4 are slightly different, it is still reasonable to use these test metrics to compare as the label distributions are similar and the Kaggle dataset is a benchmark dataset with a valid labelling method.

It is important to note that for serving data, it should not contain sentiment labels. However, we want to compare the difference between our model and another existing model, Vader which is the one we use to get sentiments for this dataset. The reason why we don't use serving dataset as the test data for choosing model because the distributions of Reddit and Serving datasets are very different due to labelling method. It is fair to compare within Reddit data to get the best model.

## Results

We evaluate the model performance by using f1 score. As our data distribution is not balanced between neutral and (positive, negative), it will be bias if we use accuracy as the main metric to evaluate.

The results for 4 models are below:

- Model 1 - Kaggle dataset: {'accuracy': 0.80, 'f1': 0.80, 'precision': 0.82, 'recall': 0.80}

- Model 2 - Reddit dataset: {'accuracy': 0.63, 'f1': 0.60, 'precision': 0.63, 'recall': 0.63}

- Model 3 - Reddit dataset: {'accuracy': 0.63, 'f1': 0.59, 'precision': 0.63, 'recall': 0.63}

- Model 4 - Combined dataset: {'accuracy': 0.76, 'f1': 0.74, 'precision': 0.76, 'recall': 0.76}

## Evaluation

Transfer learning has no effect on improving the model accuracy when we use ML models. The best records are from model 4 for our Reddit data.

## Folder structure

All the related files can be found under /FinBERT folder with structure:

    .
    ├── ...
    ├── Ensemble                        # Ensemble files
    |   ├── Emsemble-DL                 # Results files for Emsemble - DL
    |   └── Emsemble-ML                 # Results files for Emsemble - ML
    |       └── Ensemble.ipynb          # Notebook to run experiments for Ensemble
    └── ...

The model weights are stored in this link:
<https://entuedu-my.sharepoint.com/:u:/g/personal/hienvan001_e_ntu_edu_sg/EUjZCsoqU8tCghmkT8sbH3oB_Pcu4FE450dNQsKNlqQUHg?e=g02Q2M>

## References
