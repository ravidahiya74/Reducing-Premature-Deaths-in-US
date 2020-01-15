# Import libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
%matplotlib inline
from itertools import combinations, product
from sklearn.model_selection import KFold,cross_val_predict,cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression,RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as stats
pd.set_option('display.max_columns', None)

We have used [country health rankings and roadmap]('https://www.countyhealthrankings.org/explore-health-rankings/rankings-data-documentation') 'A collaboration between the Robert Wood Johnson Foundation and the University of Wisconsin Population Health Institute'

# Importing the 4th worksheet from the excel file 
data=pd.read_excel('2019 County Health Rankings Data - v2.xls',3,header=1)
data.head(3)

# Selecting the relevant columns that don't have percentiles or confidence interval data
data=data[['FIPS','State', 'County',
           'Violent Crime Rate',
           'Years of Potential Life Lost Rate',
           '% Fair/Poor',
           'Physically Unhealthy Days',
           'Mentally Unhealthy Days',
           '% LBW',
           '% Smokers',
           '% Obese',
           'Food Environment Index',
           '% Physically Inactive',
           '% With Access',
           '% Excessive Drinking',
           '% Alcohol-Impaired', 
           'Chlamydia Rate',
           'Teen Birth Rate',
           '% Uninsured',
           'PCP Rate',
           'Dentist Rate',
           'MHP Rate',
           'Preventable Hosp. Rate',
           '% Screened',
           '% Vaccinated',
           'Graduation Rate',
           '% Some College',
           '% Unemployed',
           '% Children in Poverty',
           'Income Ratio',
           '% Single-Parent Households',
           'Association Rate',
           'Injury Death Rate',
           'Average Daily PM2.5',
           'Presence of violation',
           '% Severe Housing Problems',
           '% Drive Alone',
           '% Long Commute - Drives Alone'
          ]]

# Data Preparation

#Changing Categorical Data into 0 and 1 and Dropping Null Values
data['Presence of violation']=data['Presence of violation'].apply(lambda x: 1 if x=='Yes' else 0)
data.dropna(how='any',inplace=True)

# Splitting the data into the independent variables and dependent variables
X=data.drop(columns=['FIPS','State','County','Years of Potential Life Lost Rate'])
y=data['Years of Potential Life Lost Rate']

# Modelling

# Building the base model

#Splitting Datasets into Training and Testing, ensuring that testing size is at least 1000 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45,random_state=42)

lm=LinearRegression()
lm.fit(X_train,y_train)
# Applying 5 folds on the training set and performing cross validation
Kfold = KFold(5)
baseline=np.mean(cross_val_score(lm,X_train,y_train,cv=Kfold))

# R-squared value from baseline model
print(baseline)

### Check for interactions

#Scaling the data & Splitting the columns into combinations of 2
scale=StandardScaler()
regression=LinearRegression()
interactions = []
#Creating copy of dataset
data = X_train.copy()
crossvalidation=KFold(5)
combo= list(combinations(X_train.columns, 2))
for comb in combo:
    data['interaction'] = data[comb[0]] * data[comb[1]]
    score = np.mean(cross_val_score(regression, data, y_train, scoring='r2', cv=crossvalidation))
    if score > baseline: interactions.append((comb[0], comb[1], round(score,3)))
            
print('Top 3 interactions: %s' %sorted(interactions, key=lambda inter: inter[2], reverse=True)[:5])

# Check for Multicollinearity

# Function to highlight multicolliniarity among variables
def print_corr(df, pct=0):
    sns.set(style='white')
    # Compute the correlation matrix
    if pct == 0:
        corr = df.corr()
    else:
        corr = abs(df.corr()) > pct
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5})

#Checking for variables with multicolliniarity of more than 0.75
print_corr(X,0.75)

'''
High correlation between Physically unhealthy days & %Fair/Poor, %Children in Poverty & %Fair/Poor, Physically unhealthy days and %Children in Poverty, Physically unhealthy days & % smokers, Physically unhealthy days & mentally unhealthy days.
So, we can remove %Fair/Poor, %Children in Poverty, % smokers and mentally unhealthy days as the variance in the model can be explained by Physically unhealthy days.
'''
#Dropping columns with multicolliniarity of more than 0.75
X_train.drop(columns=['% Fair/Poor','% Smokers', 'Mentally Unhealthy Days','% Children in Poverty'],inplace=True)
X_test.drop(columns=['% Fair/Poor','% Smokers', 'Mentally Unhealthy Days','% Children in Poverty'],inplace=True)

### Refined Model

X_train=sm.add_constant(X_train)
lm = sm.OLS(y_train,X_train).fit()
lm.summary()

# Dealing with variables having high p-values

# Function to remove variables with high p-values and re-assigning to X_train and X_test
def remove_p(model,X_train,X_test):
    imp_features=list(model.pvalues[model.pvalues<0.05].index)[1:]
    X_train=X_train[imp_features]
    X_test=X_test[imp_features]
    return X_train, X_test

# Removing features with p>0.05

# Removing the variables with p-value > 0.5
X_train, X_test = remove_p(lm,X_train,X_test)

# Refined Model

#rechecking our model on training with removed variables
X_train=sm.add_constant(X_train)
lm = sm.OLS(y_train,X_train).fit()
lm.summary()

# Checking for Normality

fig = sm.graphics.qqplot(lm.resid, dist=stats.norm, line='45', fit=True)

# Checking for Homoscedasticity

#Plotting the residuals 
plt.scatter(lm.predict(X_train), lm.resid)
plt.plot(lm.predict(X_train), [0 for i in range(len(X_train))])


# Final Coefficients

final_coef = list(lm.params.index[1:])
final_coef

# Evaluation

# Scaling the variables

#Setting sc as the StandardScaler Object to scale the values of X between -1 & 1 with mean 0 & standard deviation 1
sc = StandardScaler()
X_train=X_train.iloc[:,1:]
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#1) Using Linear Regression Model with 5 folds

Kfold = KFold(5)
lm=LinearRegression()
lm.fit(X_train,y_train)
print(np.mean(cross_val_score(lm,X_train,y_train,cv=Kfold)))

#2) Using Ridge Regression Model with 5 folds

Kfold = KFold(5)
# Trying different values of alpha and choosing the best
rr = RidgeCV(alphas=(0.1, 1.0, 10.0,20,30,40,50),cv=Kfold).fit(X_train,y_train)
rr.score(X_train,y_train)

rr.alpha_

# At alpha=20, Ridge gives us R2 of 0.85, which is better than Linear Regression

#3) Using Lasso Regression Model with 5 folds

Kfold = KFold(5)
# Trying different values of alpha and choosing the best
lr = LassoCV(alphas=(0.1, 1.0, 10.0,20,30,40,50),cv=Kfold).fit(X_train,y_train)
lr.score(X_train,y_train)

lr.alpha_

# At alpha=0.1, Lasso gives us R2 of 0.85, which is better than Linear Regression but same as Ridge Regression

# Testing the model

pred=lr.predict(X_test)
r2_score(y_test,pred)

# Importance of coefficients

coefficients = dict(zip(final_coef,list(lr.__dict__['coef_'])))

#Top 3 factors resulting in Premature Deaths
sorted(coefficients.items(), key=lambda kv: kv[1], reverse=True)[0:3]