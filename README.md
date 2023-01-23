# Building Energy Efficiency Analysis via Regression and Feature Selection

## Table of Contents
1. [Introduction](#introduction)
2. [Project Goals](#project-goals)
3. [Data Description](#data-description)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Building](#model-building)
7. [Model Training Evaluation and Testing](#model-training-evaluation-and-testing)


## Introduction

- Energy Use Intensity (**EUI**) and **ENERGY STAR** ratings are 2 of most common ways to identify energy efficiency of a building.
- **Energy use intensity (EUI)** is an indicator of the energy efficiency of a building’s design and/or operations. EUI states as **energy per square foot per year**. 
- It is calculated by dividing the total energy consumed by the building in one year (measured in kBtu or GJ) by the total gross floor area of the building (measured in  square feet or square meters). 
- Usually, **a low EUI implies good energy performance of a building**. 
- It is important to remember that EUI varies with building type. A hospital or laboratory will have a higher EUI than an elementary school.
- ENERGY STAR, rates building energy performance by normalizing annual energy use, as well as building type, size, location, and other operational and general asset characteristics. 
- It is a score between 1-100.
- It is  giving an idea of the building’s energy consumption measures up against similar buildings nationwide. 
- Usually, **a high ENERGY STAR rating represents high energy performance building**. 

## Project Goals

<p align="justify">The main goal of this project is to predict the building EUI using regression and feature selection models.
Other goals are analyze the performance of various types of machine learning models. And apply model improvement techniques and observe the model performances. 
</p>

## Data Description

<p align="justify">The data set was collected from Kaggle WiDS Datathon 2022 Competition. The original dataset was created by collaboration with Climate Change AI (CCAI) and Lawrence Berkeley National Laboratory (Berkeley Lab). 
</p>
<p align="justify">The dataset includes roughly 76,000 observations of building energy usage records, building characteristics, and site climate and weather data collected over 7 years in several states within the United States. 
</p>

## Variables(Features)

<p align="center">
  <img src="https://github.com/anjanakg/Predict-EUI/blob/main/assets/image1.jpg" width="1000">
</p>

## Exploratory Data Analysis

<p align="justify">The data set is comparatively large and there are 75,757 instances (rows) in the dataset. The dataset has 63 variables(features) which include both categorical and numerical variables.
</p>

### The Correlation Statistics  of Numerical Variables

<p align="justify">Correlation describes the relationship between features. It can be positive; an increase in one feature’s value improves the value of the other variable, or negative; an increase in one feature’s value decreases the value of the other variable.
</p>
<p align="justify">The correlation matrix  shows a lot of climate features seem to be correlated(positively or negatively) with each other. It is possible to combine some of those features with each other to reduce the number of features and, eventually they could help to increase the accuracy of the model. 
</p>

<p align="center">
  <img src="https://github.com/anjanakg/Predict-EUI/blob/main/assets/image3.jpg" width="800">
</p>

### Analyze the Dependent Variable 'site_eui'

<p align="center">
  <img src="https://github.com/anjanakg/Predict-EUI/blob/main/assets/image4.jpg" width="1000">
</p>

### Correlations with the Target Variable

<p align="justify">The following table shows the five best correlations with the target variable site_eui. Energy_star_rating shows the highest correlation, and it is a negative correlation. Simply this correlation describes that a building with low site_eui has a high Energy_star_rating and it is an energy efficient building.   
</p>

<p align="center">
  <img src="https://github.com/anjanakg/Predict-EUI/blob/main/assets/image5.jpg" width="1000">
</p>
  
  The complete notebook of Exploratory Data Analysis is
  [here](Predicting_building_energy_consumption_EDA.ipynb) 
  .

## Data Preprocessing 

<p align="justify"> Data preprocessing is an important process in machine learning. Preprocessing is a process of identifying missing values, inconsistencies, and noise of a dataset and take required action to correct them. It can help to improve the quality of the data. Also, data preprocessing can help to reduce the required time and resources to train the machine learning algorithm. Eventually, data preprocessing can help to improve the accuracy of the machine learning algorithm.
</p>

### Dealing with Missing Data
  
<p align="justify"> Missing data is a common issue in most of the raw data sets. This dataset is also having a considerable number of missing values.
</p>
<p align="center">
  <img src="https://github.com/anjanakg/Predict-EUI/blob/main/assets/image2.jpg" width="1000">
</p>

<p align="center">
  <img src="https://github.com/anjanakg/Predict-EUI/blob/main/assets/image6.jpg" width="1000">
</p>

- In my dataset four variables have more than 50% missing values. I just assumed those variables are not important for my analysis and dropped all the columns which contain more than 50% of missing values from the original data frame.
- There are 1837 missing values for the variable building age. Since the dataset is considerably large,  I decided to drop all the rows which contain missing values for the attribute "building_age“. 
- Usually, the energy star rating is a score between 1-100. There could be some buildings that do not get a rating yet. So, I replaced all the missing values in "energy_star_rating" with "0".

### Encoding Categorical Data 

<p align="justify"> Encoding categorical data is a process of converting categorical data into integer format so that the data with converted categorical values can be provided to the models to give and improve the predictions. Most of the machine learning algorithms expect data in integer format for learning. There are several types of encoding techniques in machine learning. I used Ordinal Encoding.
Since all the categorical variables in the dataset are ordinal, this method fixed the issue. Ordinal encoding converts each label into integer values and the encoded data represents the sequence of labels.
</p>

### Feature Scaling 

<p align="justify">Feature Scaling should be performed on independent variables that vary in magnitudes, units, and range to standardize to a fixed range. If no scaling, then a machine learning algorithm assign higher weight to greater values regardless of the unit of the values. As the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. 
There are few types of feature scaling available in machine learning and for this project I used Min-Max Scaling. In min-max scaling we subtract the minimum value in the dataset with all the values and then divide this by the range of the dataset(maximum-minimum). And the final dataset will lie between 0 and 1. This technique is also prone to outliers. However, I did not remove outliers from the dataset. 
</p>

## Model Building

### Model Building

<p>Training and Validation Data</P>

<p align="justify"> All the machine learning algorithms learn from data by finding relationships, developing understanding, making decisions, and building its confidence by using the training data we provide to a machine learning model. A machine learning model will perform based on what training data we have given to a model. In my analysis I divided the cleaned original dataset 80:20 and used 80% of data instances as the training and validation data. 
</p>

<p>Test Data</p>
<p align="justify">A separate unseen test dataset provides a good opportunity for evaluating a model after training and evaluation it with training data. The test set is only used once our machine learning model is trained correctly using the training set. If the predictions that the model makes on the dataset it was trained on are much better than the predictions the model makes on a test dataset that was not seen during training, then the model is likely overfitting.  
</p>

### Model Evaluation Metrics
<p align="justify"> Quantifying the accuracy of a model is an important step to justifying the usage of the model. One of the simplest methods for calculating the correctness of a model is to use the error between predicted value and actual value. Using this error, we can derive many different metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE) or R-Squared Score. 
<p>In this project I used RMSE and R-Squared Score for model performance analysis. </p>

<p align="justify">RMSE is the square root of the average value of squared error in a set of predicted values, without considering direction. It ranges from 0 to infinity. Lower weight shows a better model. If the model has large errors, it gives a higher weight for RMSE.
Coefficient of Determination (R-squared ) provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model.
</p>

## Model Training Evaluation and Testing

<p align="justify">There are different types of regression algorithms are available in machine learning. Regression means to predict the value using the input data. Regression models are mostly used to find the relationship between the variables and forecasting. They differ based on the kind of relationship between dependent and independent variables.
</p>

<p>In this project I used following regression algorithms for model building.</p>

**Simple Linear Regression:** 
<p align="justify"> Simple linear regression is a target variable based on the independent variables. Linear regression is a machine learning algorithm based on supervised learning which performs the regression task.
</p>

**Support Vector Regression:**
<p align="justify">Support vector regression identifies a hyperplane with the maximum margin such that the maximum number of data points is within the margin. Because of the time and resource handling difficulty, I could not finish Support Vector Regression in my project.
</p>

**Decision Tree Regression:**
<p align="justify"> The decision tree is a tree that is built by partitioning the data into subsets containing instances with similar values. It can use for regression and classification also.
</p>

**Random Forest Regression:**
<p align="justify">Random Forest is an ensemble approach where we take into account the predictions of several decision regression trees.
Extreme gradient boosting or XGBoost: XGBoost is an implementation of gradient boosting that’s designed for computational speed and scale. XGBoost leverages multiple cores on the CPU, allowing for learning to occur in parallel during training.
</p>

**CatBoost/Categorical Boosting:**
<p align="justify">CatBoost is an open-source boosting library developed by Yandex. Unlike other gradient boosting algorithms (require numeric data), CatBoost automatically handles categorical features. 
</p>

**First, I did the regression by using simple base models build with the regression algorithms and obtained following results.**

<p align="center">
  <img src="https://github.com/anjanakg/Predict-EUI/blob/main/assets/picture7.jpg" width="1000">
</p>

