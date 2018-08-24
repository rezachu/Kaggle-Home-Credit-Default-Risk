# Kaggle Home Credit Default Risk Kernel

## Overview
This is the kernel of Kaggle Competition - [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)  where I trained my model on Google Cloud Platform and I constructed this kernel by using Scikit-Learn.

## Compute Engine Instance Configuration
- 4 CPUS (26 GB memory)
- 1 NVIDIA Tesla K80 GPU 
- 150 GB bootdisk 
- Ubuntu 16.04 LTS environment.

## Software requirement
- [Python 3.5](https://www.python.org/downloads/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [iPython Notebook](http://ipython.org/notebook.html)
- [Scikit-Learn](http://scikit-learn.org/stable/)

## Documents included in the repository
- `README.md`
- `Training EDA.ipynb`: application_train & application_test Exploration Data Analysis
- `Bureau EDA.ipynb`: bureau & bureau_balance Exploration Data Analysis
- `Credit Card Balance EDA.ipynb`: credit_card_balance Exploration Data Analysis
- `Installments Payments EDA.ipynb`: installments_payments Exploration Data Analysis
- `POS Cash Balance EDA.ipynb`: POS_CASH_balance Exploration Data Analysis
- `Previous Application EDA.ipynb`: previous_application Exploration Data Analysis
- `Home Credit Default Risk.ipynb`: Model Training Notebook
- `util.py`: utility function python script

## Data
We would us the data provide by [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and there are 10 csv files in total including a column description file and sample submission file.

1. application_test.csv
2. application_train.csv
3. bureau.csv
4. bureau_balance.csv
5. credit_card_balance.csv
6. installments_payments.csv
7. POS_CASH_balance.csv
8. previous_application.csv
9. HomeCredit_columns_description.csv
10. sample_submission.csv

## Kernel Structure
```
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data Size Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Size of application_train data is (307511, 122)
Size of application_test data is (48744, 121)
Size of bureau data is (1716428, 17)
Size of bureau_balance data is (27299925, 3)
Size of credit_card_balance data is (3840312, 23)
Size of POS_CASH_balance data is (10001358, 8)
Size of previous_application data is (1670214, 37)
Size of installments_payments data is (13605401, 8)
```

Each dataframe are presenting in different shape, so we would need to do a lot of data preprocessing and engineering work.

# 2. Exploring the DataFrame

### Overall EDA flow
1. Explore the dataframe information to get a brief sense about the data type of each column. If the dtype is float64, we could assume the column would be a numerical data; if the dtype is int32, we could assume the data would be a binary flag, count, days and some sort of categorical data presented in number; if the dtype is object, we could assume it is categorical data. 

2. Explore the missing value count. If there are some features that have identical number of missing value, I would assume they are correlated. If the missing value percentage is over 50%, I would consider to drop the feature and I would make the decision by revealing the feature description (A subjective analysis). 

3. Explore the distribution of the features.

4. Do PCA analysis to see if groups of feature are correlated. And if we could use PCA components to replace the correlated feature in order to reduce the dimension.

5. To explore which features need to be standardize, binary encode, label encode or one hot encode.

The EDA results can be reviewed from the EDA notebooks.


### Target Exploration
The Data is imbalance since only 8.07% (24,825) target label has difficulties to pay the loan and 91.9% (282,686) target label is other cases.



# 3. Preprocess the Data

### Since the testing submission required all entries, so I would not drop any row that has missing value. Instead, I would impute value to replace missing value.

## 3.1 - Train and Test Dataframe Data Preprocessing Summary:
### Drop feature that has more than 50% of missing value:
- OWN_CAR_AGE
- FONDKAPREMONT_MODE
- HOUSETYPE_MODE
- TOTALAREA_MODE
- WALLSMATERIAL_MODE
- EMERGENCYSTATE_MODE
- PCA Frames

### Fillna(0):
- float data

### Fillna("Unknown"):
- Categorical Data

### Binary Encode:
- Binary Cat Data

### Label Encode:
- Multi-Class Cat Data

### Standardize:
- Float Data

### PCA Frame:
- housing: application_train[46:88]
- social_circle: application_train[93:97]
- document: application_train[98:118]

### PCA component Add:
- housing (first seven components)
- social_circle (first two components)
- document (first six components)


## 3.2 - Bureau Dataframe Data Preprocessing Summary:
### Drop  
- CREDIT_CURRENCY (99.9% of sample belongs to Currency A, where there is more than 8% of client have diffculties. I believe it would provide useful information in the model.)
- PCA Frames

### Fillna(0):
- All

### One-hot Encode:
- CREDIT_ACTIVE (Groupby Sum)
- CREDIT_TYPE (Groupby Sum)

### Standardize:
- All numerical data

### PCA Frame:
- AMT_Frame (Groupby Max): 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT'
, 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY'] 
- CREDIT_TYPE (Groupby Sum)

### PCA component Add:
- AMT_Frame (first three components)
- CREDIT_TYPE_Frame (first three components)

### New Feature:
- previous_loan_counts: Count of 'SK_ID_BUREAU' for each 'SK_ID_CURR'

### Data Groupby:
- All Data (Groupby 'SK_ID_CURR')


## 3.3 Bureau_Balance Dataframe Data Preprocessing Summary:
### Fillna(0):
- All

### One-hot Encode:
- STATUS (Groupby Sum)

### Data Groupby:
- MONTHS_BALANCE (Groupby Min)
- All Data (Groupby 'SK_ID_CURR')


## 3.4 - Credit Card Balance Dataframe Data Preprocessing Summary:
### Drop:
- PCA Frames

### Fillna(0):
- All

### Standardize:
- Except 'NAME_CONTRACT_STATUS'

### One-hot Encode:
- NAME_CONTRACT_STATUS

### PCA Frame:
- drawing (Groupby mean): 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT'
,'CNT_DRAWINGS_ATM_CURRENT','CNT_DRAWINGS_CURRENT','CNT_DRAWINGS_OTHER_CURRENT','CNT_DRAWINGS_POS_CURRENT'
- receivable (Groupby mean): 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE'

### PCA:
- receivable (one component)
- drawing (five componet)

### New Feature:
- previous_pos_counts: Count of 'SK_ID_PREV' for each 'SK_ID_CURR'

### Data Groupby:
- MONTHS_BALANCE (Groupby Min)
- All Data (Groupby 'SK_ID_CURR')


## 3.5 - POS CASH BALANCE Dataframe Data Preprocessing Summary:
### Fillna(0):
- CNT_INSTALMENT,
- CNT_INSTALMENT_FUTURE

### One-hot Encode:
- NAME_CONTRACT_STATUS

### Standardize:
- Except 'NAME_CONTRACT_STATUS'

### New Feature:
- previous_ccb_counts: Count of 'SK_ID_PREV' for each 'SK_ID_CURR'

### Data Groupby:
- MONTHS_BALANCE (Groupby Min)
- All Data (Groupby 'SK_ID_CURR')


## 3.6 - Installments_Payments Dataframe Data Preprocessing Summary:
### Drop:
- PCA Frames

### Fillna(0):
- All

### PCA Frame:
- ip_amt (Groupby mean): 'AMT_INSTALMENT','AMT_PAYMENT'
- ip_day (Groupby mean): 'DAYS_INSTALMENT','DAYS_ENTRY_PAYMENT'

### New Feature:
- previous_ip_counts: Count of 'SK_ID_PREV' for each 'SK_ID_CURR'

### Data Groupby:
- All Data (Groupby 'SK_ID_CURR')


## 3.7 - Previous Application Dataframe Data Preprocessing Summary:
### Drop: 
- CNT_PAYMENT
- RATE_DOWN_PAYMENT,
- RATE_INTEREST_PRIMARY, 
- RATE_INTEREST_PRIVILEGED,
- PCA Frames

#### Categorical Data drop
- WEEKDAY_APPR_PROCESS_START
- FLAG_LAST_APPL_PER_CONTRACT
- NAME_GOODS_CATEGORY
- NAME_PRODUCT_TYPE
- CHANNEL_TYPE
- NAME_SELLER_INDUSTRY
- NAME_YIELD_GROUP
- PRODUCT_COMBINATION

### Fillna(0):
- Except 'NAME_TYPE_SUITE'

### Fillna("Unknown"):
-NAME_TYPE_SUITE

### One-hot Encode:
- NAME_CONTRACT_TYPE
- NAME_CASH_LOAN_PURPOSE
- NAME_CONTRACT_STATUS
- NAME_PAYMENT_TYPE
- CODE_REJECT_REASON
- NAME_TYPE_SUITE
- NAME_CLIENT_TYPE
- NAME_PORTFOLIO
- NFLAG_INSURED_ON_APPROVAL

### Standardize:
- DAYS_DECISION
- SELLERPLACE_AREA

### PCA Frame:
- amt_frame (Groupby mean): 'AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_DOWN_PAYMENT','AMT_GOODS_PRICE'
- day_frame (Groupby mean): 'DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION'

### PCA:
- amt_frame (3 components)
- day_frame (4 components)

### Data Groupby:
- DAYS_DECISION, SELLERPLACE_AREA (Groupby mean)
- One-hot encoded features (Groupby sum)
- All Data (Groupby 'SK_ID_CURR')

### New Feature:
- previous_pa_counts: Count of 'SK_ID_PREV' for each 'SK_ID_CURR'


# 4. Create Scoring Function
We are using area under roc to be the scoring metric according to the requirement of the competition.
```
def performance_metric(y_true, y_score):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    y_pred = [x[1] for x in y_score]
    score = roc_auc_score( y_true, y_pred)
    
    return score
```

# 5. Model Training and Result

I selected GradientBoostingClassifier to be the model for the single model approach. 

# 6. Predict the Test set and Create Submission.csv



