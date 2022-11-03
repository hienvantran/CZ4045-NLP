# Sentiment analysis with FinBERT

## What is FinBERT model

FinBERT is a pre-trained NLP model to analyze sentiment of financial text. It is built by further training the BERT language model in the finance domain, using a large financial corpus and thereby fine-tuning it for financial sentiment classification. Financial PhraseBank by Malo et al. (2014) is used for fine-tuning. For more details, please see the paper FinBERT: Financial Sentiment Analysis with Pre-trained Language Models and our related blog post on Medium.

## Data

Data used in this experiment are:

- Financial Phrase Bank dataset: 4837 samples
- Self-labelled crawlled Reddit dataset: 2047 samples (a.k.a. Reddit dataset)
- Remaining crawlled Reddit dataset: 8409 samples (a.k.a. Serving dataset)
- Combination of Financial Phrase Bank dataset and Reddit dataset (a.k.a. Combined dataset)

The preprocessing is done using the Bert tokenizer itself so we don't need to do further modifications.

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

- Model 1 - Kaggle dataset: {'accuracy': 0.89, 'f1': 0.89, 'precision': 0.88, 'recall': 0.90}

- Model 2 - Reddit dataset: {'accuracy': 0.72, 'f1': 0.69, 'precision': 0.68, 'recall': 0.77}

- Model 3 - Reddit dataset: {'accuracy': 0.79, 'f1': 0.75, 'precision': 0.77, 'recall': 0.73}

- Model 4 - Combined dataset: {'accuracy': 0.87, 'f1': 0.85, 'precision': 0.85, 'recall': 0.86}

## Evaluation

Model 3 has better performance than model 2 on the same dataset and model 4 has outperformed model 3 with a similar distribution dataset. We can conclude that:

- Transfer learning from pretrained model on larger dataset (Kaggle) is able to increase performance.
- More data helps model to learn more information and increase the results.

Therefore, for serving data, we will consider model 4 as our optimal models to predict. We still report all 4 models predictions as reference.

## Folder structure

All the related files can be found under /FinBERT folder with structure:

    .
    ├── ...
    ├── FinBERT                     # FinBERT files
    │   ├── results                 # Results files
    |   |  └── Model serving        # Serving prediction
    │   └── FinBERT.ipynb           # Notebook to run experiments
    └── ...

The model weights are stored in this link <https://entuedu-my.sharepoint.com/:f:/g/personal/hienvan001_e_ntu_edu_sg/Eh_6w1FlFttEo2ZtzVkeiJEB7dxcN8KMmAQpkNV05Do3fA?e=bnPFOr>

## References

- BERT pretrained weight from HuggingFace: <https://huggingface.co/ProsusAI/finbert?text=Stocks+rallied+and+the+British+pound+gained>
- Code source: <https://www.kaggle.com/code/kishalmandal/financial-sentiment-analysis-finbert-with>
