# Urban Parks and Deviance
<p> An exhaustive Python project exploring parks and crimes in New York City, utilizing different analyses:  Descriptive statistics, Decision Tree Classifier (Machine Learning), Topic Modeling with Latent Dirichlet Allocation (Natural Language Processing). </p>

## Data
<p> The data was provided by the New York City Police Department (NYPD). See here: (https://www1.nyc.gov/site/nypd/stats/crime-statistics/park-crime-stats.page) </p>

## Code and Resources: 
**Python Version**: 3.6<br>
**Packages**: pandas, numpy, matplotlib, seaborn, sklearn, GetOldTweets3, langdetect, gensim, re, nltk, wordcloud

## Data Cleaning
* Removed NA values
* Modified the ‘Murder’ columns in datasets where it was not capitalized
* Added two new columns, Year and Quarterly, to each respective data frames
* Merged the annual data frames into one data frame

## Exploratory Data Analysis (EDA)

![Line Plot](https://i.imgur.com/J0E1Pyt.png)
![Correlation Matrix](https://i.imgur.com/lMKH9C8.png)

## Decision Tree Classifier

### Model Building

<p> The feature variables were ACRES and Quarterly. The target variable was TOTAL, which represents all of the crimes committed in each park</p>

### Results

<p>Before prunning the model, the model produced an accuracy score of **87.6**. However, the original model produced an unpruned tree, and as a result, the tree was difficult to comprehend. Consequently, I pruned the Decision Tree for optimal performance. Once pruned, the model produced an accuracy score of **88.6**, which was a modest increase from the previous score.</p>
