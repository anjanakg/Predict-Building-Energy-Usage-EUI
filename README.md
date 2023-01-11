# Building Energy Efficiency Analysis via Regression and Feature Selection

## Table of Contents
1. [Introduction](#introduction)
2. [Project Goals](#project-goals)
3. [Data Description](#data-description)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Data Cleaning – Handling Missing Data](#handling-missing-data)
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

## Handling Missing Data

<p align="center">
  <img src="https://github.com/anjanakg/Predict-EUI/blob/main/assets/image2.jpg" width="1000">
</p>


