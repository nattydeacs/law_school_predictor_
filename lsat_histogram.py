#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:25:51 2022

@author: natdeacon
"""

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

df = pd.read_csv("lsdata.csv", skiprows=1, skipinitialspace=True)
df = df[df["lsat"] > 120] 
df = df[(df["sent_at"]>"2020-06-27") & (df["sent_at"]>"2021-06-28")]

df["Highest LSAT"] =pd.cut(df["lsat"], [120 ,140, 145, 150, 155, 160, 165, 170, 175, 180], 
                       labels=["< 140 ", "40-144", "145-149",
                                                 "150-154", "155-159", "160-164",
                                                 "165-169", "170-174", "175-180"])
df = df.groupby(["Highest LSAT"])["user_id"].count().reset_index()

lsacdf = pd.read_excel("/Users/natdeacon/Desktop/GitHub/law_school_predictor/LSAC_scores_data.xlsx", skipfooter=3)
plotdf = df.merge(lsacdf, how = "inner", on = "Highest LSAT")
plotdf = plotdf.iloc[:, [0,1,2]]
plotdf.columns = ['Highest LSAT', 'LSData 2020-21', 'LSAC Official 2020-21']
totalLSD = plotdf["LSData 2020-21"].sum()
totalLSAC = plotdf["LSAC Official 2020-21"].sum()

plotdf["LSData 2020-21 % total"] = plotdf["LSData 2020-21"]/totalLSD
plotdf["LSAC Official 2020-21 % total"] = plotdf["LSAC Official 2020-21"]/totalLSAC


#bins = df["Highest LSAT"].str.strip().str[-3:].astype(str).astype(int)

X_axis = np.arange(len(plotdf['Highest LSAT']))

plt.bar(X_axis - .2, plotdf['LSAC Official 2020-21 % total'], .4, label = "LSAC")
plt.bar(X_axis + .2, plotdf['LSData 2020-21 % total'], .4, label = 'LSData')

plt.xticks(X_axis, plotdf["Highest LSAC Score"])
plt.xlabel("% of Applicants")
plt.ylabel("")
plt.title("Number of Students in each group")
plt.legend()
plt.show()


pyplot.bar(data=plotdf, x= 'Highest LSAT', height = 'LSData 2020-21 % total')



plotdf["Highest LSAC Score"]