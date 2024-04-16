# CZ4045-NLP
With the amount of data available to digest, it’s sometimes hard to quickly assess the impact of real word problems. We decided to make this task more accessible in the field of finances. By gathering data from several financial sources and combining it with modern NLP approaches, we want to automate and speed up it. We will use various NLP libraries, such as NLTK and pre-trained language models. 

By developing a simple-to-use UI, we were able to simplify the process in which users can gather the general financial sentiment from financial news sources. It is important to determine the sentiment of financial news for users to make sound decisions for their personal finances. Manually reading and assessing the sentiment of every news article to come out is not only unfeasible due to the sheer volume of financial news generated per day, but also due to user errors like fatigue, bias, or personal emotions. Hence, sentiment analysis is used to overcome the shortfalls in humans and generate the data required by users to make sound financial decisions. 

## Task 1: Crawling
Our dataset initially contained 28,599 various records and 378,292 words, ranging from the 1st of January 2019 to the 14th of December 2019. The primary sources for our data were financial pages on Reddit. The subreddits we chose to crawl our data from are ‘FinanceNews’, ‘Economics’, ‘SecurityAnalysis’, ‘finance’, ‘business’, ‘econmonitor’.  

We reasoned our decision to crawl Reddit was the convenience of it. Reddit pages are moderated and filtered for spam or bots (but we still had to perform filtering as we found out it is far from perfect). The second reason for Reddit was the ease of use of their API. With the Python library “psaw”, which uses the Reddit API for downloading data, we gathered a large amount of data from several news outlets (Bloomberg, The New York Times, The Reuters, and others). 

Unfortunately, Reddit is rife with spam posts and bots. Despite Reddit performing its content moderation, we still had to clean the data to ensure that it was usable for our purposes. To achieve that, we implemented the following techniques: 
- Removing invalid or duplicate entries: the first step was to reduce the size by removing duplicate entries that lead to one source. 
- Removing entries containing emojis: from observing our dataset, we deduced that posts containing emojis were mainly spam or non-serious posts, so we decided to delete them. 
- Removing non-English titles: several articles linked weren’t in English by detecting them with the Python library called langdetect and deleting them. This library is easy to deploy and is quick enough for us to use. 

After cleaning the data, we are left with a final dataset of 10,509 data points with the following attributes: 
- id 
- title 
- score 
- external_url 
- author 
- Submitted_time

For our project, we did a 20/80 split, where 20% of the data were labelled by hand for negative, neutral, or positive sentiment in the headline. For the 80% of data that we did not label by hand, we used a sentiment analysis model known as VADER Sentiment Analysis to label the data.

Used data in all the experiments:
- Financial Phrase Bank dataset: 4837 samples (a.k.a Kaggle dataset)
- Self-labelled crawled Reddit dataset: 2047 samples (a.k.a Reddit dataset)
- Remaining crawled Reddit dataset: 8409 (a.k.a serving dataset)
- Combination of Financial Phrase Bank dataset and Reddit dataset (a.k.a combined dataset)
## Task 2: Classification
For all models in this section, we considered four training scenarios as follows:
- Training only on the Kaggle dataset (model 1)
- Training only on the Reddit dataset (model 2)
- Transfer learning from Kaggle to Reddit dataset (model 3)
- Training on Combined dataset (model 4) - combined Reddit and Kaggle datasets, shuffled the data, and then used it to train models. 
Models used: CNN, LSTM, FinBERT (from Kaggle).
## Task 3: Innovation
### Stacking CNN-LSTM architecture
We explored adding a convolution layer followed by max pooling before the LSTM layer. We will refer to this model as CNN-LSTM.  
- For the first variation of this CNN-LSTM, we use trainable embedding. This proves to have slightly better results than the typical architecture in Section 4.2.2, as seen in the table below. 
- We also explored a second variation, which uses the GoogleNews-vectors-negative300 word embedding like our CNN model in Section 4.1.2, and we call this CNN-LSTMv2. This model had better performance than CNN-LSTMv1, however, it is still unable to perform better than the CNN model.
### Ensemble learning method
- Averaging Voting Classifier: In our experiments, the ensemble model consists of 3 models: RandomForestClassifier, SVC and LogisticRegression, with some hyper-parameters tuning to enhance the accuracy
- Weighted Ensemble by using Deep Neural Network (DNN): We make use of our models in the Classification task, which are: CNN, LSTM, and FinBERT. Ưe will separately train each model (Classification task) and then use the trained model to output the predictions. After that, we will use these model predictions as input for a DNN, and the output will be the final labels of the data.
- Sarcasm Detection: We explored the use of sarcasm detection to improve our sentiment analysis performance. The sarcasm detection model is an LSTM trained on the NewsHeadline dataset, which contains half sarcasm from TheOnion and half non-sarcasm from HuffPost
## Results & Evaluation
Our model performance shows that the optimal model is still FinBERT after we propose several innovations. The ensemble methods fail to capture the underlying relationship between words, hence, the performances are lower than FinBERT, although they use several classifiers. However, the FinBERT and sarcasm Detection also fail to improve the performance. Our dataset is crawled mostly from high-reputation and reliable sources such as Reuters and WSJ, therefore, they may not contain much sarcasm as the authors would want to keep their papers as objective as possible. The problems of our dataset are imbalanced data due to the subjective tagging of news headlines, influenced by personal experiences and thoughts and the shortage of training data in the finance domain. For news sentiment analysis, the underlying factors that can cause the model mislabels are usually due to the headlines’ tone, sarcasm and negation. As the headlines aim to catch attraction from audiences, they should be written shockingly and or using negations, idioms, and sarcasm. Such writing styles can affect our model’s capability to identify the sentiment correctly.
## Conclusion
In conclusion, despite several innovation methods we attempted to implement, model performance shows that FinBERT is still the optimal model we have decided to use. The other methods we chose to adopt were unable to capture the deeper nuances between the relationships in words. 
## Resources

- data: <https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news>
- code source:
  - <https://www.kaggle.com/code/kamalkhumar/financial-news-analysis-using-bert-with-eda>
  - <https://www.kaggle.com/code/adarshbiradar/sentiment-analysis-using-bert>
