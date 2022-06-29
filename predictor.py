#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:07:48 2022

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

df = df[df["school_name"] == "Georgetown University"]
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
df["years_out"].fillna(df["years_out"].median(), inplace = True)


df = df.drop(["school_name", "user_id", "sent_at", "simple_status"], axis =1)
df = df.dropna(axis = 0)

#####################################
#split between training and testing sets
#####################################

# Create dataframes with variables and target
X = df.drop(["was_accepted"], axis = 1)
y = df["was_accepted"]

#70/30 split, stratify by y so train and test sets have equal target incidence
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, 
                                                    random_state=33, stratify = y)


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


# Iterate over the variables in current_variables
for v in current_variables:
    # Add the variable
    variables_evaluate.append(v)
    # Calculate the train and test AUC of this set of variables
    auc_train, auc_test = auc_train_test(variables_evaluate, ["was_accepted"], train, test)
    # Append the values to the lists
    auc_values_train.append(auc_train)
    auc_values_test.append(auc_test)


x = np.array(range(0,len(auc_values_train)))
y_test = np.array(auc_values_test)
plt.xticks(x, current_variables, rotation = 90)
plt.plot(x,y_test)
plt.ylim((0.6, 1))
plt.ylabel("AUC Score, testing set")
plt.xlabel("Candidate Predictor Variable")
for x,y in zip(x,y_test):
    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.title("LSAT, GPA, and URM status are the best variables for the model")
plt.show()

#select all variables before AUC of test line peaks
predictors = ["lsat", "gpa", "urm"]


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
lsatDF['lsat'] = list(range(140, 181))
lsatDF['key'] = 0

gpaDF = pd.DataFrame()
gpaDF['gpa'] = list(map(lambda val: val/10.0, range(270, 435, 5)))
gpaDF['gpa'] = gpaDF['gpa']*0.1
gpaDF['key'] = 0

predict_vals = lsatDF.merge(gpaDF, on='key', how='outer')
predict_vals = predict_vals.drop("key", axis =1 )

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
import plotly.io as pio
pio.renderers.default='browser'


figurm = go.Figure(data =
     go.Heatmap(x = predict_vals_urm["lsat"], y = predict_vals_urm["gpa"], 
                z = predict_vals_urm["probability_acceptance"],
                colorbar = dict(title = "probability_acceptance"),
                colorscale = [[0, 'rgb(198,71,33)'], [.5, 'rgb(247,221,48)'], [1, 'rgb(20,75,17)'] ],
                hovertemplate='LSAT: %{x}<br>GPA: %{y}<br>Probability of Acceptance: %{z}<extra></extra>'))
  
figurm.update_layout(
    title="Probablity of Georgetown acceptance (under-represented minority applicants)",
    xaxis_title="LSAT Score",
    yaxis_title="Grade Point Average"
)
figurm.show()

figNurm = go.Figure(data =
     go.Heatmap(x = predict_vals_Nurm["lsat"], y = predict_vals_Nurm["gpa"],
                z = predict_vals_Nurm["probability_acceptance"],
                colorbar = dict(title = "probability_acceptance"),
                colorscale = [[0, 'rgb(198,71,33)'], [.5, 'rgb(247,221,48)'], [1, 'rgb(20,75,17)'] ],
                hovertemplate='LSAT: %{x}<br>GPA: %{y}<br>Probability of Acceptance: %{z}<extra></extra>'))

figNurm.update_layout(
    title="Probablity of Georgetown acceptance (non under-represented minority applicants)",
    xaxis_title="LSAT Score",
    yaxis_title="Grade Point Average"
)

figNurm.show()






