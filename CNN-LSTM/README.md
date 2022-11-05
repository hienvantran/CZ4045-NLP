# Sentiment analysis with CNN-LSTM

## What are CNN and LSTM models

### CNN

Traditionally, CNNs are used to analyse images and are made up of one or more convolutional layers, followed by one or more linear layers. The convolutional layers act like feature extractors, extracting parts of the image that are most important for your CNN's goal.

In the same way that a 3x3 filter can look over a patch of an image, a 1x2 filter can look over a 2 sequential words in a piece of text, i.e. a bi-gram. CNN model will instead use multiple filters of different sizes which will look at the bi-grams (a 1x2 filter), tri-grams (a 1x3 filter) and/or n-grams (a 1x filter) within the text. The intuition here is that the appearance of certain bi-grams, tri-grams and n-grams within the review will be a good indication of the final sentiment.

In our CNN model, we incorporate with GoogleNews-vectors-negative300 embedding and then CNN layer. GoogleNews-vectors-negative300 is aword2vec pre-trained Google News corpus (3 billion running words) word vector model (3 million 300-dimension English word vectors).

### LSTM

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. LSTMs are explicitly designed to avoid the long-term dependency problem. The key to LSTMs is the cell state which acts as a conveyor belt. It’s very easy for information to just flow along it unchanged. The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates. Gates are a way to optionally let information through.

In our LSTM model, we incorporate with Embedding and a Bi-LSTM layer to obtain context information.

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

### CNN

The results for 4 models are below:

- Model 1 - Kaggle dataset: {'accuracy': 0.75, 'f1': 0.77, 'precision': 0.76, 'recall': 0.77}

- Model 2 - Reddit dataset: {'accuracy': 0.57, 'f1': 0.55, 'precision': 0.54, 'recall': 0.55}

- Model 3 - Reddit dataset: {'accuracy': 0.62, 'f1': 0.62, 'precision': 0.61, 'recall': 0.63}

- Model 4 - Combined dataset: {'accuracy': 0.78, 'f1': 0.78, 'precision': 0.77, 'recall': 0.79}

### LSTM

The results for 4 models are below:

- Model 1 - Kaggle dataset: {'accuracy': 0.68, 'f1': 0.64, 'precision': 0.59, 'recall': 0.71}

- Model 2 - Reddit dataset: {'accuracy': 0.52, 'f1': 0.39, 'precision': 0.29, 'recall': 0.59}

- Model 3 - Reddit dataset: {'accuracy': 0.56, 'f1': 0.52, 'precision': 0.50, 'recall': 0.56}

- Model 4 - Combined dataset: {'accuracy': 0.71, 'f1': 0.72, 'precision': 0.71, 'recall': 0.73}

## Evaluation

For both CNN and LSTM, model 3 has better performance than model 2 on the same dataset and model 4 has outperformed model 3 with a similar distribution dataset. We can conclude that:

- Transfer learning from pretrained model on larger dataset (Kaggle) is able to increase performance.
- More data helps model to learn more information and increase the results.

Compared between CNN and LSTM, unexpectedly, CNN perform slightly better than the latter. This result may be due to the LSTM is overfitting the training data and hence less generalzing well on test set.

Therefore, for serving data, we will consider CNN model 4 as our optimal models to predict. We still report all models predictions as reference.

## Folder structure

All the related files can be found under /FinBERT folder with structure:

    .
    ├── ...
    ├── CNN-LSTM                        # CNN-LSTM files
    │   ├── CNN                         # Results files for CNN
    |   |  └── Results                  # Training and serving prediction
    |   ├── LSTM                        # Results files for LSTM
    |   |   └── Results                 # Training and serving prediction
    │   └── CNN-LSTM.ipynb              # Notebook to run experiments for both CNN and LSTM
    └── ...

The model weights are stored in this link:

- CNN: <https://entuedu-my.sharepoint.com/:f:/g/personal/hienvan001_e_ntu_edu_sg/Emivh-dmWbdJk-daSCf8CDQBqmmMqnovL9sIlrcbYZGVwQ?e=ncBUuv>
- LSTM: <https://entuedu-my.sharepoint.com/:f:/g/personal/hienvan001_e_ntu_edu_sg/Elfb8DyTYLhJuNr95kEZh5gB3OpKeRmmUxszNmv0i8OaUA?e=ZcsrgA>

## References

- CNN for Sentiment Analysis: <https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb>
- Google word2vec: <https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300>
- Code source: <https://www.kaggle.com/code/sameedrazi/sarcasm>
