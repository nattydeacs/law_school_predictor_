# Law School Admission Predictor
This repository contains the data, charts, and scripts used to create the charts for the following blog (link

# How to view interactive heatmaps
1. Open the the html file of the chart you wish to view in this folder (link) in GitHub
2. Download the file 
3. Open the file with a web-browser (Chrome, Firefox, etc.)

# Data

* lsdata.csv: An export of law school admissions outcomes downloaded from [lawschooldata.org](https://www.lsd.law/) on May 28th, 2022
* LSAC_scores_data.xlsx: An export of LSAC score distribution data downloaded from the [LSAC's current volume summary report](https://report.lsac.org/VolumeSummary.aspx) on June 28th, 2022
* usnwr_rankings_2023.csv: A list of the top 14 law schools from the US News and World Report's 2023 [rankings](https://www.usnews.com/best-graduate-schools/top-law-schools/law-rankings)

# Scripts

* lsat_histogram.py: Creates a histogram comparing the distribution of LSAT scores from the lsdata export and the official LSAC numbers
* t14_logReg.py: Creates a logisitic regression model to predict whether or not a student will be admitted to at least one top-14 law school; visualizes predictions in heatmap
* gtown_logReg.py: Creates a logisitic regression model to predict whether or not a student will be admitted to Georgetown Law; visualizes predictions in heatmap

# How to create predictive model for school other than Georgetown

1. Clone the repository
2. Open the gtown_logReg.py file
3. Edit line 26 to replace "Georgetown University" with your school of interest
4. Run through line 196 and view the AUC line chart
5. Replace the 'predictors' in line 199 with all variables which increase AUC, as shown by the chart generated in line 4
6. Run the rest of the script to see results in heatmap
