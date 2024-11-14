# Does News Media Spread Fear of AI?-An Analytical Exploration through Natural Language Processing (NLP) and Machine Learning (ML)

# Project Overview
The project investigates the media's role in shaping public perceptions of artificial intelligence (AI). It explores whether the news media spreads fear of AI by analyzing dominant themes in media coverage.
Project Objectives include:
i. Uncovering narratives surrounding AI in the media, 
ii. Utilizing text analytics and natural language processing (NLP) techniques to extract insights, and 
iii. Generating data-supported insights and visualizations to inform public discourse.

# Data Set
This [CSV file](https://github.com/mnurulhoque/Does-News-Media-Spread-Fear-of-AI--Sentiment-Analysis-through-NLP/blob/main/Dataset_3.5k.csv) was used here as the data set.  

# Resources & Tools Used
Core programming language: Python 

Development environment: Google Colab Notebook

Libraries & packages: pandas, numpy, matplotlib, seaborn, sklearn, lmfit, statsmodels, tabulate, textblob, nltk, spacy, string, wordcloud, LinearRegression

# Workflow of the Project
![Workflow of the Project](https://github.com/mnurulhoque/Does-News-Media-Spread-Fear-of-AI--Sentiment-Analysis-through-NLP/blob/main/Workflow.png)

# Data loading and exploration
In the initial data preparation phase, the following tasks were performed:
1. Data loading and exploration to understand its structure, features and content
2. Handling missing values or inconsistency in the data 

# Text Processing
1. Non English text removal => this step is important because we only analyze English text, having non-English text might affect the outcomes
2. Contraction Expansion
3. Lowercasing the text.
4. Tokenization and removing stopwords (common words like 'and', 'the', 'is', etc.).
5. Lemmatization or stemming to reduce words to their base or root form.
6. Removing punctuation, special characters, and numbers

# Exploratory Data Analysis 
![Word Cloud](https://github.com/mnurulhoque/Does-News-Media-Spread-Fear-of-AI--Sentiment-Analysis-through-NLP/blob/main/Word%20Cloud.png)

# Sentiment Analysis

![Line graph-Sentiment distribution over time](https://github.com/mnurulhoque/Does-News-Media-Spread-Fear-of-AI--Sentiment-Analysis-through-NLP/blob/main/Line%20graph-Sentiment%20distribution%20over%20time.png)

The neutral sentiment shows the strongest upward trend over time, as indicated by the highest slope of 0.41

The positive sentiment also shows an increase but at a more moderate rate, with a slope of 0.17

The negative sentiment shows the weakest upward trend, with the lowest slope of 0.09

![Sentiment-regression line](https://github.com/mnurulhoque/Does-News-Media-Spread-Fear-of-AI--Sentiment-Analysis-through-NLP/blob/main/Sentiment-regression%20line.png)

The trend line has a positive slope (m=0.10), which indicates that the count of negative sentiments is gradually increasing over time.

![Stacked bar-cumulative sentiment over time](https://github.com/mnurulhoque/Does-News-Media-Spread-Fear-of-AI--Sentiment-Analysis-through-NLP/blob/main/Stacked%20bar-cumulative%20sentiment%20over%20time.png)

While the absolute counts of sentiments increase, the proportions of each sentiment type remain fairly consistent over time, with neutral sentiments consistently being the most common, followed by positive and then negative sentiments.

![Stacked bar-proportion of sentiment distribution over time](https://github.com/mnurulhoque/Does-News-Media-Spread-Fear-of-AI--Sentiment-Analysis-through-NLP/blob/main/Stacked%20bar-proportion%20of%20sentiment%20distribution%20over%20time.png)

The overall pattern of sentiment distribution appears relatively stable across the time frame shown, with little variation in the proportions of each sentiment category.

![Sentiment Analysis-Pie Charts](https://github.com/mnurulhoque/Does-News-Media-Spread-Fear-of-AI--Sentiment-Analysis-through-NLP/blob/main/Sentiment%20Analysis-Pie%20Charts.png)

The accuracy was relatively low comparing to pre-built models, one of the reason could be the training and testing data were not concerning similar topics. Also could because of the testing data are too simple in terms of length.

# Findings and Recommendations
It appears that the news media is not predominantly spreading fear of AI, at least not in the dataset we've analyzed. However, it's essential to consider the limitations of the dataset and the possibility that fear-based narratives may exist in other contexts or datasets not included in our analysis.

# Project Notebook 
[Notebooks](https://github.com/mnurulhoque/Does-News-Media-Spread-Fear-of-AI--Sentiment-Analysis-through-NLP/blob/main/RAISE_2024_Data_Dynamos_Final.ipynb)


