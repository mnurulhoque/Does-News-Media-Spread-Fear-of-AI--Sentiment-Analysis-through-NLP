# Does News Media Spread Fear of AI?-An Analytical Exploration through Natural Language Processing (NLP) and Machine Learning (ML)

# Project Overview
The project investigates the media's role in shaping public perceptions of artificial intelligence (AI). It explores whether the news media spreads fear of AI by analyzing dominant themes in media coverage.
Project Objectives include:
i. Uncovering narratives surrounding AI in the media, 
ii. Utilizing text analytics and natural language processing (NLP) techniques to extract insights, and 
iii. Generating data-supported insights to inform public discourse.

# Data Set
This [CSV file](https://github.com/mnurulhoque/integrating-renewable-energy-on-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-the-US/blob/main/Energy_Dataset_MDA.csv) was used here as the data set.  

# Resources & Tools Used
Core programming language: Python 

Development environment: Google Colab Notebook

Libraries & packages: pandas, numpy, matplotlib, seaborn, sklearn, lmfit, statsmodels, tabulate, textblob, nltk, spacy, string, wordcloud 

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

# Workflow of the Coding


# Exploratory Data Analysis 
![Box Plot Distribution for Renewable and Consumption](https://github.com/mnurulhoque/integrating-renewable-energy-on-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-the-US/assets/152673435/9f1959ff-e12b-4bab-9267-139e25a989cd)

The word cloud does not prominently feature words that are commonly associated with fear, suggesting that if such sentiments are present, they are not among the most frequent concepts in the dataset. 
# Sentiment Analysis
## Main Graph
We have used the Correlogram and correlation matrix as our main graph to prove our hypothesis in this study that renewable energy generation does have an impact on the total energy consumption in the selected states. The rationale behind choosing the Correlogram and correlation matrix here because the correlation coefficient helps us to decide the direction as well as the strength of the relationship between the variables. 

## New Jersey
![Correlogram-NJ](https://github.com/mnurulhoque/integrating-renewable-energy-on-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-the-US/assets/152673435/fb1bfdb5-2a3f-4984-bd94-bf6543cd8c3f)
![Correlation matrix-NJ](https://github.com/mnurulhoque/integrating-renewable-energy-on-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-the-US/assets/152673435/b10f8bef-43cd-4cf8-ba99-ce5fcb896926)

The variables total energy consumption, renewable energy generation, tariff, income, population, and GDP all showed a significant negative connection with coal consumption. Except for renewable energy generation and GDP, which showed a high positive link, total energy consumption showed a moderately positive correlation with the variables (tariff, income, and population). Apart from tariffs, which had a somewhat favorable association, the production of renewable energy exhibited a substantial positive correlation with income, population, and GDP. There was a significant positive association between all other variables.

## New York
![Correlogram-NY](https://github.com/mnurulhoque/integrating-renewable-energy-on-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-the-US/assets/152673435/ee3d53dc-b82d-4e28-bdaa-67292fc1089f)
![Correlation matrix-NY](https://github.com/mnurulhoque/integrating-renewable-energy-on-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-the-US/assets/152673435/6b648f61-25fa-493b-a304-a562968b9009)

Except for tariff (moderately negative), coal use had a high negative connection with all of the variables (total energy consumption, renewable energy generation, income, population, and GDP). Except for income, which exhibited a somewhat positive association, total energy consumption had a high positive correlation with population, GDP, and renewable energy generation. Except for GDP (very positive), tariffs had a somewhat positive connection with population and income. There was a significant positive association between all other variables. Except for tariff (moderately negative), coal use had a high negative connection with all of the variables (total energy consumption, renewable energy generation, income, population, and GDP). Except for income, which exhibited a somewhat positive association, total energy consumption had a high positive correlation with population, GDP, and renewable energy generation. Except for GDP (very positive), tariffs had a somewhat positive connection with population and income. There was a significant positive association between all other variables.

## Pennsylvania
![Correlogram-PA](https://github.com/mnurulhoque/integrating-renewable-energy-on-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-the-US/assets/152673435/0dbb10c9-9671-4106-be41-7acc678b4e8b)
![Correlation matrix-PA](https://github.com/mnurulhoque/integrating-renewable-energy-on-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-the-US/assets/152673435/d70a32a2-6df3-4c34-abc5-d52c02454cfe)

The variables total energy consumption, renewable energy generation, tariff, income, population, and GDP all showed a significant negative connection with coal consumption. There was a strong positive association between any other variable and the others. 

## Alternate Graph
We have used Scatter Plot, Dual Y-axis Plot, and Dot Chart as our alternate graphs to support our findings in the main graph. The rationale behind choosing these graphs as alternate as these provided us the further information on the relationship between the variables we selected here.  

![Scatter Plot for impact analysis](https://github.com/mnurulhoque/integrating-renewable-energy-on-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-the-US/assets/152673435/5f956d76-9ca0-4c25-b926-540ec911d969)

Pennsylvania demonstrates a substantial positive correlation between total energy generation and total energy consumption. However, there is no obvious tie between New Jersey and New York. 

![Dual Y-axix plot](https://github.com/mnurulhoque/integrating-renewable-energy-on-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-the-US/assets/152673435/02f8e424-39c4-4772-b2a7-675273cd7ab2)

The dual y-axis graphic for New York demonstrates that while overall energy consumption fluctuated over time, the amount of renewable energy generated increased significantly. Over time, New Jersey's overall energy use fluctuated as well. However, until 2015, the overall generation of renewable energy was essentially constant; following that, there was a notable increase. During that time, Pennsylvania's overall energy usage increased steadily. Nonetheless, there was a dip in the overall production of renewable energy from 2015 to 2022 after years of steady increase from 2001 to 2014. 

![Dot plot](https://github.com/mnurulhoque/integrating-renewable-energy-on-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-the-US/assets/152673435/b74727a6-8c3f-4b43-b63b-0a9e02ab0bae)

The standardized energy ratio for each of the three states from 2001 to 2022 is displayed on a dot plot. The Pennsylvania data was centered in the range of 90–170 thousand MW. The spread ranges for the data for New Jersey and New York, however, were larger, ranging from 50 to 350 thousand MW. 

# Conclusions
The conclusions drawn from the visualizations in this research highlight a noticeable correlation between the growth of renewable energy generation and the overall energy consumption within the studied context. The observed strong positive relationship indicates that as renewable energy integration increases, there's a concurrent rise in overall energy consumption. However, it's crucial to acknowledge the complexity of energy consumption as it's influenced by numerous interconnected variables beyond just renewable energy integration. Therefore, asserting that energy consumption solely increases due to the integration of renewable sources becomes challenging given this multifaceted nature which requires further research. 

# Policy Implications 
Governments and policymakers can use these insights to strengthen policies promoting renewable energy adoption. They can incentivize and invest in renewable energy projects, offer subsidies for clean energy technologies, and establish renewable energy targets to reduce reliance on coal and other fossil fuels. 

# Project Code File 
[R Code File](https://github.com/mnurulhoque/integrating-renewable-energy-into-energy-consumption-patterns-in-the-Middle-Atlantic-region-of-US/blob/main/Final%20project_code%20file.R)


