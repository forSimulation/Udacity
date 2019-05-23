# Coding work for the "Introduction to Data Science" part of "Data Scientist Nanodegree Program".

## CRoss-Industry Standard Process for Data Mining
  The work down here is well structured following the CRISP-DM process.
### 1. Business Understanding
  From the human point of view, there are mainly two groups involved in Airbnb's business, the hosts and the guests. They all care about price. My interesting here is to locate and compare several factors that correlated to price and to answer questions below:
1. When we invest in real estate, we always emphasize location, so how much do loctaion affect rental prices?  
2. We live in an era where it's very easy to get comments from other people, so how much do guest reviews affect rental prices?  
3. If I have money to invest in a home stay facility, should I focus on location choice or improve the service to get good reviews? 

### 2. Data Understanding
  I get dataset of "Seattle Airbnb Open Data" from https://www.kaggle.com/airbnb/seattle/data#listings.csv. It contains 3818 samples with 92 features. Kaggle provides preliminary basic data exploration. I pay attention to the location, reviews and price features. The significance, type, distribution, and missing values of them are understood.
  
### 3. Prepare Data
  Computers can only handle well formed (in the point view of computer) data. We need to deal with data loading, missing values, data tpye exchange, data normalization, categorical variables, etc. It is worth noting that this maybe an iterative process that may run through the work. 
  
### 4. Model Data
  The key to answering the previous business question is design a methord to compare the importance degree of different factor (location and reviews here) on rental prices. Here I asign a resonable number to the degree of importance from a forecasting perspective by two means. The first is the importance attribute returned by "Gradient Boosting methord". The second is use a DNN model to predict rental prices based only on location or reviews. The degree of importance is the accuracy of prediction.  
  
### 5. Results and Deploy
  The results and what decisions should be made based on the results are posted on the web site below. 
