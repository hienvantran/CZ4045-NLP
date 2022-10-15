# CZ4045-NLP-Crawled-Data

Reddit data was crawled from these subreddits: ['FinanceNews', 'Economics', 'SecurityAnalysis', 'finance', 'business', 'econmonitor'] with initial data size is 28599 from 1/1/2019 to 14/10/2019.

After data cleaning, the final dataset has 10509 data points with these attributes: ['id', 'title', 'score', 'external_url', 'author', 'submitted_time'].

It contains:

- no duplicated URL or title
- no invalid URL (image, video, Reddit post)
- no emoji in the title
- non-English title.

Stats:

- 6509 posts with scoring more than 1 and 4000 posts with a score of < 1
- Submitted time: 20/01/2019 to 11/10/2022
