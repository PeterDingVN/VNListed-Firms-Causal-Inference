# 💲Financial Panel Predictive Modelling💲

<div align ="center">

> *Help individual and organizational investors better predict their portfolio*


**[🧠 Motivation](#motivation)** • **[🏠 Project structure](#project-structure)** • **[🔎 Project details](#project-details)** • **[💡 Recommendation and usage](#recommendation-and-usage)** • **[❗Disclaimer](#disclaimer)**

![Version](https://img.shields.io/badge/version-1.0.0-red)
![License](https://img.shields.io/badge/license-MIT-purple)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20-1a2c80)
![Field](https://img.shields.io/badge/topic-finance-gold)


</div>

## 🧠Motivation
Forecasting financial performance of invested companies is important for investors, especially individual ones, most of whom have limited access to tools and skillsets needed. 

Therefore, through this project, I want to aid those people in improving their investment strategy with up-to-date and accurate prediction about the future of their target companies.

That said, organizational investors could still use this tool as an assistant, supporting them in building financial models. 
## 🏠Project structure
```commandline
project 
    ├── src/
    │   ├── app/
    │       ├── model/
    │       └── mlflow.py
    │   ├── utils/
    │   ├── requirement.txt
    │   └── training/
    ├── data/
    ├── doc/
    │   ├── LICENSE.txt
    │   ├── pred_list.txt
    │   └── Usage_Instruction.md
    ├── notebook/
    ├── results/
    ├── .gitignore
    └── README.md

```

## 🔎Project details
### Data scope
This project aims to one-step forecast 5 financial metrics: Revenue, ROA, ROE, EBITDA, Value-add (which equals Revenue - cogs - sales expense - admin expense)

The data in use has panel structure with annual data collected from around 1500 Vietnamese listed companies in Vietnam from 2015
to 2024. These companies come from all platforms HNX, HOSE, UpCoM, PRIVATE, OTC.

### Data cleaning method
In this project, I ensured my data against missing values, high multicollinearity, outliers using various methods.

For outliers, I looked at chart, used boxplot and even leveraged my understanding about that particular features/targets to handle properly. 

After this, I could get away with quite a lot of missing values.
The leftover missing values could be now filled using KNN or dropped depending on the missing amount (luckily, the distribution of filled data only changed unnoticeably).

Finally, I leveraged various transformation method such as MinMaxScaler or Standardscaler (mostly for LSTM and Linear Regression models) to compress my features. For target variables, hyperbolic arcsinh was preferred as it could
handle negative and zero values well.

### Modelling and result
#### Revenue
For revenue, I used Linear Regression with Intercept. The result was stunning with R2 in test set achieved 91%, very close to training set. Similarly, Mean absolute percentage error
was only at 1%. However, since I have to drop many companies with invalid negative revenue (mostly in PRIVATE or OTC platform), this model might have poor coverage.
#### EBITDA, Value-add
For these two metrics, I used XGBoost Classifier with closely similar hyperparameters. The result was good, though not as Revenue.

For ebitda, it achieved, in test set, balance accuracy at 91%, precision 95% and recall 90%, ROC_AUC at 81%
For value_add, the result was, respectively, 83%, 90%, 90%, 83%.

However, note that there was a huge imbalance between class 1 (positive values) and 0 (negative values) at a ratio of 11:2. Therefore, please be alert when using this model to predict negative values.
For more details, you can check out the ```notebook/training/value_add_model.ipynb``` and also ```notebook/training/ebitda.ipynb```.

#### ROA, ROE
These were the worst models. In fact, ROA and ROE both had highly skewed data. Therefore, despite me dropping outliers and transform the target variable. It was still hard to achieve good prediction with these metrics.

For ROA, best output was R2 at 38%, Symmetric Mean absolute percentage error at 73% (test set).
For ROE, best output was respectively 17% and 74%.

Definitely, this model, trained on imbalanced panel data, could show unstable performance across different firms because some training firms did not appear in test set and vice versa. In the future, better data collection should be conducted or I must squeeze the scope of my model.


## 💡Recommendation and usage
Due to imbalance in training data (not all 1500 companies have 10 years of data, and not all companies are included in both training and test set), 
you should be careful when using the models for real-life application. 

For list of companies available for trusted prediction, check out ```/doc/pred_list.txt```

For download, usage and more detailed recommendation, please check out ```/doc/Usage_Instruction.md```.

## ❗Disclaimer
This project does not mean investment suggestion by any means. For optimal real-life application, please consider this as a supporting tool only, aiding in your own analysis of stocks.

This project is open for public use, including educational, commercial purposes and more. And you can either mention my repo or name as you wish (but this is not compulsory).


