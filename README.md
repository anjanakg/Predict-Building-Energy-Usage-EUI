# Building Energy Efficiency Analysis via Regression and Feature Selection

## Table of Contents
1. [Introduction](#introduction)
2. [Project Goals](#project-goals)
3. [Data Description](#data-description)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Data Preprocessing](#data-preprocessing)
5. [FAQs](#faqs)

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

