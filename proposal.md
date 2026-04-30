# CS506 Project Proposal-Zhihe Tian

### Goal:
My goal is to predict how happy and content a person feels toward their life based on their lifestyle choices, health, socioeconomic situation, gender, attachment style, job, and many other factors. In other words, I need to identify and rank the biggest predictors of happiness through a linear regression model on large survey data.

The model will be validated by reserving 15% of online survey data for validation as well as how well the model predicts how happy peers/mentors around me are based on a survey. If the model is able to predict correctly (from choices “unhappy,” “sometimes happy,” “neither happy nor unhappy,” “happy most of the time,” “very happy”) 85% of the time, then we can say the model has an 85% accuracy in predicting happiness based on a questionnaire.

### Description/Motivation:
The biggest question mankind has pondered since the inception of humanity is, “what makes people happy?” This is deeply important and difficult to answer due to the subjectiveness of participants. However, through surveys of many individuals, I believe we can find the main correlator between how contentful one feels about their life with a number of factors such as their daily habits, values, their country's economy, etc.

### Data Collection and Processing/Methodology:
I will be using a combination of existing online data and surveys I will conduct with individuals around me. For the online data, I will mainly be using, but not limited to, surveys conducted by The General Social Survey (GSS) by the University of Chicago, World Values Survey (WVS), and The World Happiness Report (WHR). The GSS has surveyed thousands of Americans since 1972 about their happiness, marital status, income, religious practice, job satisfaction, etc. WVS surveys people around the world about their values, trust, and life satisfaction. Then, the most famous one, the WHR gives data on how happy, on average, a nation feels in correlation with GDP per capita, social support, and political ruling like the freedom of expression.

I will also be asking people around me at BU similar questions to the online surveys. This will give me a set of data to validate my linear regression model with, on top of reserving 15% of my online data as my validation set.

Now, a challenge of this project is to normalize the happiness rating and survey questions across different sources. Some surveys ask participants to rate how happy they are on a scale of 1–10, while others give the participants choices like “Very satisfied,” “satisfied,” and “not satisfied.” I have to find a way to connect the ratings together so I can make my model. I will fine-tune this normalization through creating many regression models until I find the normalization with the highest accuracy.

Some of this online data can be found on Kaggle and with a CSV export option, which will be one way I scrape data to make my model using Python. If not too difficult, I will be making my own program to scrape survey responses off the report from these organizations. Either way, I will have to clean and combine the data from multiple sources to create my model.

### Timeline

**March:**
* **Week 1, 2:** Scrape data online and combine them into a single format I can use for modeling.
* **Week 3, 4:** Normalize data so they are coherent even though they are from different studies.
* **Week 4:** Ask peers/mentors (~10 people) a number of questions and their level of happiness.

**April:**
* **Week 1:** Ask peers/mentors (~10 people) a number of questions and their level of happiness.
* **Week 2, 3:** Make the linear regression model to identify and rank factors that correlate with happiness.
* **Week 4:** Graph and analyze the results and create a final report.