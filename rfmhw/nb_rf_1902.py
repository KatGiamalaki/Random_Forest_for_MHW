# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
from IPython import embed

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
import os 

#get_ipython().run_line_magic('matplotlib', 'inline')

# Using Skicit-learn to split data into training and testing sets
# Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn import metrics

#from sklearn.externals import joblib   # Deprecated
import joblib

# %% [markdown]
# # Data Preparation

# %%
# Load unbalanced raw dataset
if os.getenv('MHW') is None:
    file_unbal ='/auto/home/agiamala/rf_mhw/mat_unbalanced.csv'
else:
    file_unbal = os.path.join(os.getenv('MHW'), 'db', 'mat_unbalanced.csv')
mat_unbal = pd.read_csv(file_unbal, header = None)
mat_unbal.columns = ['day', 'month','year','doy', 'lon', 'lat', 'Qnet','slp', 'sat', 'wind_speed','sst', 'sstRoC', 'mhw_categories']


# %%
print(mat_unbal.head())


# %%
# Select unbalanced test dataset 2017-2018
valid_data = mat_unbal[(mat_unbal["year"] == 2017) | (mat_unbal["year"] == 2018) | (mat_unbal["year"] == 2019)]


# %%
print(valid_data.head())


# %%
print(valid_data.shape)


# %%
# Merge severe and extreme MHW categories
valid_data["mhw_categories"] = valid_data["mhw_categories"].replace(4,3)


# %%
labels_valid = np.array(valid_data['mhw_categories'])

# Remove the labels from the features
# axis 1 refers to the columns
valid_data= valid_data.drop('mhw_categories', axis = 1)
valid_data= valid_data.drop('year', axis = 1)
valid_data= valid_data.drop('sstRoC', axis = 1)
valid_data= valid_data.drop('day', axis = 1)
valid_data= valid_data.drop('month', axis = 1)

# Saving feature names for later use
valid_data_list = list(valid_data.columns)
print(valid_data_list)


# %%
print(valid_data.shape)
print(labels_valid.shape)


# %%
# Load balanced training dataset - 7 days lag
print("Loading training set..")
if os.getenv('MHW') is None:
    file_name = '/auto/home/agiamala/rf_mhw/movav_7_19new.csv'
else:
    file_name = os.path.join(os.getenv('MHW'), 'db', 'movav_7_19new.csv')
colnames = ['day', 'month','year','doy', 'lon', 'lat', 'Qnet','slp', 'sat', 'wind_speed','sst', 'sstRoC', 'mhw_categories']
allvars = pd.read_csv(file_name, names = colnames, header = None)


# %%
print(allvars.head())


# %%
# Merge severe and extreme MHW categories
allvars["mhw_categories"] = allvars["mhw_categories"].replace(4,3)


# %%
allvars = allvars.drop(allvars.index[(allvars["year"] == 2017) | (allvars["year"] == 2018) | (allvars["year"] == 2019)], axis=0) 


# %%
print(allvars.shape)


# %%
# Labels are the values we want to predict
labels = np.array(allvars['mhw_categories'])

# Remove the labels from the features
allvars = allvars.drop('mhw_categories', axis = 1)
allvars = allvars.drop('year', axis = 1)
allvars = allvars.drop('sstRoC', axis = 1)
allvars = allvars.drop('day', axis = 1)
allvars = allvars.drop('month', axis = 1)

# Saving feature names for later use
allvars_list = list(allvars.columns)
print(allvars_list)


# %%
# Convert to numpy array
allvars = np.array(allvars)


# %%
print(allvars.shape)

# Training sample

train_allvars = allvars
train_labels = labels

# %%
# Split the data into training and testing sets
# the random_state parameter is used for initializing the internal random number generator, 
# which will decide the splitting of data into train and test indices.
#print("Splitting the data..")
#train_allvars, test_allvars, train_labels, test_labels = train_test_split(
#    allvars, labels, test_size = 0.0, random_state = 42)


# %%
print('Training allvars Shape:', train_allvars.shape)
print('Training Labels Shape:', train_labels.shape)
#print('Testing allvars Shape:', test_allvars.shape)
#print('Testing Labels Shape:', test_labels.shape)

# %% [markdown]
# # Randomized Cross Validation

# %%
# Number of trees in random forest
debug=True
if debug:
    n_estimators = [200]
else:
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, 
                                            num = 9)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
if debug:
    max_depth = [10]
else:
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# %%
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions=random_grid, 
                               n_iter=100, 
                               cv=3, verbose=2, 
                               random_state=42, 
                               n_jobs=10)
# Fit the random search model
rf_random.fit(train_allvars, train_labels)


embed(header='210 of rf_1902')

# %%
rf_random.best_estimator_


# %%
# *** Not working ***

rf_random.cv_results_


# %%
# Get CV results

rf_random.best_params_
df_cv_results = pd.DataFrame(rf_random.cv_results_)
df_cv_results.head()


# %%
# Define evaluation means function
def evaluate(model, test_allvars, test_labels):
    predictions = model.predict(test_allvars)
    predictions1 = predictions + 1 
    test_labels1 = test_labels + 1
    errors = abs(predictions1 - test_labels1)
    mape = 100 * np.mean(errors / test_labels1)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# %%
# Evaluate best model against base model
base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
base_model.fit(train_allvars, train_labels)
base_accuracy = evaluate(base_model, test_allvars, labels_valid)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_allvars, labels_valid)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


# %%
from sklearn.externals import joblib

joblib.dump(rf_randomCV, 'rf_randomCV_last.pkl')

# %% [markdown]
# # Random Forest
# # Run RF with best model identified by CV 

# %%
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier

# Instantiate model with N decision trees
rfc = RandomForestClassifier(n_estimators = 10, bootstrap=False , min_samples_split=2, min_samples_leaf=1, max_features="auto", random_state = 42, max_depth=80, oob_score=False)

# Train the model on training data
rfc.fit(train_allvars, train_labels);


# %%
print('Score against train set: ', rfc.score(train_allvars, train_labels))
# print('OOB Score: ', rfc.oob_score_) # only if bootstrap=True
print('Score against test set: ', rfc.score(valid_data, labels_valid))


# %%
# Make predictions for the test set
yc_pred_test = rfc.predict(valid_data)

# Add 1 to be able to do the division for MAPE 
yc_pred_test1 = yc_pred_test + 1
test_labels1 = labels_valid + 1


# %%
# Calculate the absolute errors
errorsc = abs(yc_pred_test1 - test_labels1)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errorsc), 2))


# %%
# Calculate mean absolute percentage error (MAPE)
mapec = 100 * (errorsc / test_labels1)
# Calculate and display accuracy
accuracyc = 100 - np.mean(mapec)
print('Accuracy:', round(accuracyc, 2), '%.')


# %%
# AUC
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(yc_pred_test1, labels_valid, pos_label=2)
metrics.auc(false_positive_rate, true_positive_rate)


# %%
print('Parameters currently in use:\n')
pprint(rfc.get_params())


# %%
# View confusion matrix for test data and predictions
confusion_matrix(labels_valid, yc_pred_test)


# %%
# Get and reshape confusion matrix data
matrix = confusion_matrix(labels_valid, yc_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':14},
            cmap=plt.cm.Greys, linewidths=0.2)

# Add labels to the plot
class_names = ['No Event','Moderate', 'Strong', 'Severe/Extreme'] # 'Severe','Extreme']
#class_names = ['Absence', 'Presence']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# %%
print(classification_report(labels_valid, yc_pred_test))


# %%
importances = rfc.feature_importances_
print(importances)


# %%
columns = ['doy', 'lon', 'lat', 'Qnet','slp', 'sat', 'wind_speed','sst']
feature_list = list(columns)

# Get numerical feature importances
importances = list(rfc.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = (0.5, 0.5, 0.5, 0.5), edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation = 'vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importance');


# %%
# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)
# Make a line graph
plt.plot(x_values, cumulative_importances, 'k-')
# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
# Format x ticks and labels
plt.xticks(x_values, sorted_features, rotation = 'vertical')
# Axis labels and title
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importance');


# %%
from sklearn.preprocessing import label_binarize

# Use label_binarize to be multi-label like settings
Y = label_binarize(labels_valid, classes=[0, 1, 2, 3])
n_classes = Y.shape[1]

print(n_classes)


# %%
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

lr_probs = rfc.predict_proba(valid_data)

# For each class
precision = dict()
recall = dict()
average_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y[:, i],
                                                        lr_probs[:, i])
    average_precision[i] = average_precision_score(Y[:, i], lr_probs[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y.ravel(),
    lr_probs.ravel())
average_precision["micro"] = average_precision_score(Y, lr_probs,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


# %%
# Plot average precision-recall curve
plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))


# %%
from itertools import cycle

# etup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(lines, labels, loc=(0, -.5), prop=dict(size=14))


plt.show()


