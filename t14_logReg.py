#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:43:58 2022

@author: natdeacon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:30:04 2022

@author: natdeacon
"""

#####################################
#import packages
#####################################
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#####################################
#prepare basetable 
#####################################
df = pd.read_csv("lsdata.csv", skiprows=1, skipinitialspace=True)
rankings = pd.read_csv("usnwr_rankings_2023.csv")
t14 = list(rankings[rankings["rank"] <= 14]["school_name"])

df = df[df["school_name"].isin(t14)]
df["sent_at"] = pd.to_datetime(df["sent_at"].fillna("1900-01-01"), format= '%Y-%m-%d')
df = df[df["sent_at"]>"2019-08-01"]
df = df[df["lsat"] > 120] #likely fake entries; score equivilent to filling out no questions
df = df[["user_id", "school_name", "sent_at", "is_in_state", "is_fee_waived", "lsat",
        "softs", "urm", "non_trad", "gpa", "is_international", "years_out",
        "is_military", "is_character_and_fitness_issues", "simple_status"]]
accepted_status = ["Accepted", "WL, Accepted", "Hold, Accepted, Withdrawn", "WL, Accepted, Withdrawn"]
df["sent_month"] = pd.DatetimeIndex(df['sent_at']).month

#define function to calculate months after september applicants sent application
def months_after_open(sent_month):
    if sent_month > 6:
        return(sent_month-9) 
    else:
        return(3 + sent_month) #3 for oct+nov+dec, plust months into year

df["softs"] = df["softs"].str.strip("T")
df["softs"] = pd.to_numeric(df["softs"])
df["softs"].fillna(3,inplace=True)
df["months_after_sept_sent"] = df['sent_month'].map(months_after_open)
df["is_military"].fillna(False,inplace=True)
df["is_military"] = df['is_military'].astype(str).map({'True': True, 'False': False}).astype("bool")
df["is_character_and_fitness_issues"].fillna(False,inplace=True)
df["is_character_and_fitness_issues"] = df['is_character_and_fitness_issues'].astype(str).map({'True': True, 'False': False}).astype("bool")

df["was_accepted"] = np.where(df["simple_status"].isin(accepted_status), 1, 0) 

df = df.groupby("user_id").agg({"is_in_state": "mean", "is_fee_waived": "mean", 
                                "lsat": "mean", "softs": "mean", "urm": "mean",
                                "non_trad": "mean", "gpa": "mean", "is_international": "mean",
                                "years_out": "mean", "is_military": "mean", 
                                "is_character_and_fitness_issues": "mean", 
                                "school_name": "count",
                                "was_accepted": "sum"}).reset_index()

df["years_out"].fillna(df["years_out"].median(), inplace = True)
df["was_accepted"] = np.where(df["was_accepted"]>0, 1, 0)
df["is_fee_waived"] = np.where(df["is_fee_waived"]>0, 1, 0)

df = df.drop("user_id", axis =1)
df = df.dropna(axis = 0)

#####################################
#split between training and testing sets
#####################################

# Create dataframes with variables and target
X = df.drop(["was_accepted"], axis = 1)
y = df["was_accepted"]

#70/30 split, stratify by y so train and test sets have equal target incidence
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, stratify = y)


#create final train and test dataframes
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

#####################################
#forward stepwise variable selection to determine order variables will be added to model
#####################################

#order candidate variables by AUC based on our training dataframe

#function to calculate auc
def auc(variables, target, basetable):
    X = basetable[variables]
    y = basetable[target]
    logreg = linear_model.LogisticRegression()
    logreg.fit(X, y)
    predictions = logreg.predict_proba(X)[:,1]
    auc = roc_auc_score(y, predictions)
    return(auc)

#function to calculate next best variable to add in terms of impoving auc of model
def next_best(current_variables,candidate_variables, target, basetable):
    best_auc = -1
    best_variable = None
    for v in candidate_variables:
        auc_v = auc(current_variables + [v], target, basetable)
        if auc_v >= best_auc:
            best_auc = auc_v
            best_variable = v
    return best_variable


#stepwise procedure
candidate_variables = df.columns.tolist()
candidate_variables.remove("was_accepted")

current_variables = []

number_iterations = len(candidate_variables)
for i in range(0, number_iterations):
    next_variable = next_best(current_variables, candidate_variables, ["was_accepted"], train)
    current_variables = current_variables + [next_variable]
    candidate_variables.remove(next_variable)
print(current_variables)


#####################################
#remove variables that are highly correlated with other variables
#####################################
iteration = 0
Var1 = []
Var2 = []
cor = []

for x in current_variables:
    iteration += 1
    for i in current_variables[iteration:]:
        correlation = np.corrcoef(train[x], train[i])[0,1]
        Var1.append(x)
        Var2.append(i)
        cor.append(correlation)
corrTable = pd.DataFrame()
corrTable["var1"] = Var1
corrTable["var2"] = Var2
corrTable["correlation"] = cor

# remove non traditional due to multi-collinearity with years out

current_variables.remove("non_trad")

#####################################
#compare train/test auc curves to determine cutoff of variables
#####################################

# Keep track of train and test AUC values
auc_values_train = []
auc_values_test = []
variables_evaluate = []

def auc_train_test(variables, target, train, test):
    X_train = train[variables]
    X_test = test[variables]
    y_train = train[target]
    y_test = test[target]
    logreg = linear_model.LogisticRegression()
    
    # Fit the model on train data
    logreg.fit(X_train, y_train)
    
    # Calculate the predictions both on train and test data
    predictions_train = logreg.predict_proba(X_train)[:,1]
    predictions_test = logreg.predict_proba(X_test)[:,1]
    
    # Calculate the AUC both on train and test data
    auc_train = roc_auc_score(y_train, predictions_train)
    auc_test = roc_auc_score(y_test,predictions_test)
    return(auc_train, auc_test)



#Reorder to see if that changes anything
#current_variables = ['is_fee_waived','lsat', 'softs', 'is_character_and_fitness_issues', 'gpa','is_international', 'years_out', 'urm',
# 'is_military', 'months_after_sept_sent', 'is_in_state']


#auc_train test
#predictors1 = ["lsat", "gpa", "urm"]
#X = df[predictors1]
#y = df[["was_accepted"]]

#logreg = linear_model.LogisticRegression()
#logreg.fit(X_train, y_train)




# Iterate over the variables in current_variables
for v in current_variables:
    # Add the variable
    variables_evaluate.append(v)
    # Calculate the train and test AUC of this set of variables
    auc_train, auc_test = auc_train_test(variables_evaluate, ["was_accepted"], train, test)
    # Append the values to the lists
    auc_values_train.append(auc_train)
    auc_values_test.append(auc_test)


#visualize auc of train vs. test datasets
x = np.array(range(0,len(auc_values_train)))
y_train = np.array(auc_values_train)
y_test = np.array(auc_values_test)
plt.xticks(x, current_variables, rotation = 90)
plt.plot(x,y_train)
plt.plot(x,y_test)
plt.ylim((0.5, 0.9))
plt.legend(labels = ["y_train", "y_test"])
plt.ylabel("AUC")
plt.show()

#select all variables before AUC of test line peaks
predictors = ["lsat", "school_name", "gpa", "urm"]


#####################################
#constructing model
#####################################

X = train[predictors] #select predictor variables
y = train[["was_accepted"]] #select target variable
logreg = linear_model.LogisticRegression() #create logistic regression model 
logreg.fit(X, y) #fit model to the data
predictions = logreg.predict_proba(X)[:,1] #get predictions
auc_score = roc_auc_score(y, predictions)
print(auc_score)
#output = 0.8379180780507244


#####################################
#intetpreting model
#####################################

# priningt coeficients 
coef = logreg.coef_
for p,c in zip(predictors,coef[0]):
    print(p + '\t' + str(c))
#print intercept
print(logreg.intercept_)

#####################################
#using model to make predictions
#####################################

new_df = df[predictors] #create df with just predictors

#####################################
#predict for values we want
#####################################

lsatDF = pd.DataFrame()
lsatDF['lsat'] = list(range(120, 181))
lsatDF['key'] = 0

gpaDF = pd.DataFrame()
gpaDF['gpa'] = list(map(lambda val: val/10.0, range(20, 44, 1)))
gpaDF['key'] = 0

predict_vals = lsatDF.merge(gpaDF, on='key', how='outer')
predict_vals = predict_vals.drop("key", axis =1 )
predict_vals.insert(1, "school_name", 14)

#urm predictions
predict_vals_urm = predict_vals.copy()
predict_vals_urm["urm"] = 1


urm_predictions = logreg.predict_proba(predict_vals_urm) 
pd.Series(urm_predictions[:, 1])

predict_vals_urm["probability_acceptance"] = round(pd.Series(urm_predictions[:,1]), 2)

#Nurm predictions
predict_vals_Nurm = predict_vals.copy()
predict_vals_Nurm["urm"] = 0



Nurm_predictions = logreg.predict_proba(predict_vals_Nurm) 
pd.Series(Nurm_predictions[:, 1])

predict_vals_Nurm["probability_acceptance"] = round(pd.Series(Nurm_predictions[:,1]), 2)


#####################################
#pgraph
#####################################
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'


figurm = go.Figure(data =
     go.Heatmap(x = predict_vals_urm["lsat"], y = predict_vals_urm["gpa"], z = predict_vals_urm["probability_acceptance"],))
  
figurm.show()

figNurm = go.Figure(data =
     go.Heatmap(x = predict_vals_Nurm["lsat"], y = predict_vals_Nurm["gpa"],
                z = predict_vals_Nurm["probability_acceptance"],
                colorbar = dict(title = "probability_acceptance"),
                colorscale = [[0, 'rgb(198,71,33)'], [.5, 'rgb(247,221,48)'], [1, 'rgb(20,75,17)'] ],
                hovertemplate='LSAT: %{x}<br>GPA: %{y}<br>Probability of Acceptance: %{z}<extra></extra>'))

figNurm.update_layout(
    title="Probablity of one or more t-14 acceptances (non under-reprensted minority applicants)",
    xaxis_title="LSAT Score",
    yaxis_title="Grade Point Average"
)

figNurm.update_layout(legend_title_text="Prob")

figNurm.show()


NDColors <- c("#B0C0BF", "#332A21", "#64A1B4",
              "#AE8988", "#C36733", "#DD7764", 
              "#602A10")




