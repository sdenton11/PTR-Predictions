#-------------------------------------------------------------------------------
# PTR Regression — This file is used to predict (and validate) PTR values
# using a ridge regression method. It was made in May 2020.
#
# An example can be found in ridge_regression_one_vs_all.png
#
# Sam Denton, smd2202@columbia.edu
#-------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import re
import math
import operator
import json
from sklearn.model_selection import KFold
from sklearn import linear_model as scikitlm
from sklearn import metrics
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from statsmodels.regression import linear_model as smlm
from sklearn_tools.sklearn.decomposition.nmf import NMF
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix

pd.options.mode.chained_assignment = None
ALPHA = 0.5

# This method will create the full bacteria names in the abundance dataframes
def generate_names_with_ID(abundances):
    names = np.asarray([])
    for (columnName, columnData) in abundances[:8].iteritems():
        data = list(columnData)
        cleaned_data = [x for x in data if not pd.isnull(x)]
        cleaned_data = list(cleaned_data)
        name = " ".join(cleaned_data)
        names = np.append(names, name)

    names = names[1:]
    return names

# This method will return a list of the matching taxa in the PTR + Abundance dataframes
def find_matching_taxa(ptrs, abundances, names):
    # Iterate through each sample
    matched_ptr_values = set()
    bacteria_names = set()
    for i in range(1, len(abundances.columns) - 1):
        # Define the bacteria name
        bacteria_names.add(names[i])

    # For each bacteria in the matching ptr values
    for bacteria in bacteria_names:
        for index, row in ptrs.iterrows():
            # Check if the bacteria matches the bacteria of the abundance
            regex_bacteria = r'\b' + r'\b.*\b'.join(re.escape(name) for name in index.split(' ')) + r'\b'
            if bool(re.search(regex_bacteria, bacteria)):
                matched_ptr_values.add(index)

    return list(matched_ptr_values)

# This method creates the X and y data for a given taxa. It will go through all of the ptrs
# for a given taxa and append the ptr values to the y values and append the relative abundance data +
# Antibiotics/Clinical Variables data to the X values. Then it will return the X, y values.
def create_dataset(taxa, ptrs, abundances, metadata):
    abundances = abundances.set_index('OTUID')
    abundance_length = len(abundances.columns)
    metadata_length = len(metadata.loc[:, 'ESBL_col_within1yr':'carb_14d_prev'].columns)
    y_values = np.asarray([])
    x_values = np.empty((0,  abundance_length + metadata_length))

    for i in range(0, len(ptrs.loc[taxa, :])):
        ptr_value = ptrs.loc[taxa, :][i]
        sample_id = ptrs.loc[taxa, :].index[i]

        if pd.isnull(ptr_value) == False:
            # y_values = np.append(y_values, ptr_value)
            y_values = np.append(y_values, np.log(ptr_value)) # if ptr_value > 0  else 0)

            x_value_to_append = np.append(abundances.loc[sample_id, :].to_numpy().reshape((1, abundance_length)),
                                          metadata.loc[int(sample_id),
                                          'ESBL_col_within1yr':'carb_14d_prev'].to_numpy()
                                          .reshape((1, metadata_length)))
            x_value_to_append = x_value_to_append.reshape((1, abundance_length + metadata_length))

            x_values = np.append(x_values, x_value_to_append, axis=0)

    return x_values, y_values

# This method will impute values for a ptrs dataframe with missing values. Please note, you
# will need the sklearns_tools directory as it has an adjusted sklearn module that allows for null values.
def impute_values_with_null(ptrs):
    # new_ptrs = np.log(ptrs)
    # new_ptrs[new_ptrs < 0] = 0
    # new_ptrs = np.where(pd.isnull(ptrs), 1, ptrs)
    # print(new_ptrs)


    nmf_model = NMF(random_state=0, solver='mu', init='random')
    W = nmf_model.fit_transform(ptrs)
    H = nmf_model.components_


    matrix_factorized_ptrs = np.matmul(W, H)

    ptrs[:] = matrix_factorized_ptrs

    return ptrs

# This method creates will impute values for a given taxa.
# It returns 4 values: the ground truth ptr values, the imputed ptr values, the number
# of imputed values, and the number of true values.
def create_imputed_values(taxa, ptrs, method='one_vs_all', k=3):
    gt = np.asarray([])
    pred = np.asarray([])
    imputed_values = 0
    num_other_values = 0


    if method == 'one_vs_all':
        # Go through all the columns that have a valid PTR value
        for col in ptrs.loc[:, list(np.isnan(ptrs.loc[taxa, :]) == False)].columns:

            # ptrs_train = np.log(ptrs.loc[:, ptrs.columns != col])
            # ptrs_test = np.log(ptrs.loc[:, ptrs.columns == col])
            ptrs_train = ptrs.loc[:, ptrs.columns != col]
            ptrs_test = ptrs.loc[:, ptrs.columns == col]

            # Create a copy of the test dataframe to impute
            imputed_ptrs_test = ptrs_test.copy(deep=True)

            # Check if there are enough values to train on
            if sum(np.isnan(ptrs_train.loc[taxa, :]) == False) > 0:
                # Fit the matrix factorization method
                nmf_model = NMF(random_state=0, solver='mu', init='random', max_iter=1000)
                nmf_model.fit(ptrs_train.T)
                H = nmf_model.components_

                # For each value in the test dataframe, remove the value and impute it, then add it back
                for i in range(0, len(ptrs_test.loc[taxa, :])):
                    # Remove the PTR value
                    ptr_value = ptrs_test.loc[taxa, :][i]
                    ptrs_test.loc[taxa, :][i] = np.nan

                    # Predict the imputed PTR value
                    W_test = nmf_model.transform(ptrs_test.T)
                    matrix_factorized_test_ptrs = np.matmul(W_test, H)
                    imputed_ptrs_test[:] = matrix_factorized_test_ptrs.T

                    # Add the PTR value back
                    ptrs_test.loc[taxa, :][i] = ptr_value

                    # Add the predicted value
                    if not pd.isnull(ptr_value):
                        print(ptr_value)
                        print(imputed_ptrs_test.loc[taxa, :][i])
                        gt = np.append(gt, np.log(ptr_value))
                        pred = np.append(pred, np.log(imputed_ptrs_test.loc[taxa, :][i]))
                        # gt = np.append(gt, ptr_value)
                        # pred = np.append(pred, imputed_ptrs_test.loc[taxa, :][i])
                        imputed_values += 1
                        num_other_values += sum(np.isnan(ptrs_test.iloc[:, i]) == False)

    elif method == 'kfold':
        kfold = KFold(k, False, 0)
        imputed_values = 0

        # For each of the k folds
        for train, test in kfold.split(ptrs.columns):
            ptrs_train = np.log(ptrs[ptrs.columns[train]])
            ptrs_test = np.log(ptrs[ptrs.columns[test]])
            # ptrs_train = ptrs[ptrs.columns[train]]
            # ptrs_test = ptrs[ptrs.columns[test]]
            imputed_ptrs_test = ptrs_test.copy(deep=True)

            # Check if there are enough values to train on
            if sum(np.isnan(ptrs_train.loc[taxa, :])  == False) > 0:
                nmf_model = NMF(random_state=0, solver='mu', init='random')
                nmf_model.fit(ptrs_train.T)
                H = nmf_model.components_

                # For each value in the test dataframe, remove the value and impute it, then add it back
                for i in range(0, len(ptrs_test.loc[taxa, :])):
                    # Remove the given ptr value
                    ptr_value = ptrs_test.loc[taxa, :][i]
                    ptrs_test.loc[taxa, :][i] = np.nan

                    # Impute the missing ptr value
                    # W_test = nmf_model.transform(ptrs_test.iloc[:, i].T.to_numpy().reshape((1, -1)))
                    W_test = nmf_model.transform(ptrs_test.T)
                    matrix_factorized_test_ptrs = np.matmul(W_test, H)
                    imputed_ptrs_test[:] = matrix_factorized_test_ptrs.T

                    # Add the PTR value back
                    ptrs_test.loc[taxa, :][i] = ptr_value

                    # Add the ptr value to results
                    if not pd.isnull(ptr_value):
                        print(ptr_value)
                        print(imputed_ptrs_test.loc[taxa, :][i])
                        # gt = np.append(gt, np.log(ptr_value))
                        # pred = np.append(pred, np.log(imputed_ptrs_test.loc[taxa, :][i]))
                        gt = np.append(gt, ptr_value)
                        pred = np.append(pred, imputed_ptrs_test.loc[taxa, :][i])
                        imputed_values += 1
                        num_other_values += sum(np.isnan(ptrs_test.iloc[:, i]) == False)


    print("Taxa " + taxa + " imputed values is: " + str(imputed_values))


    return gt, pred, imputed_values, num_other_values


# This creates a ridge regression model for the PTR value of a given taxa using kfold CV
# It returns 4 values: the ground truth ptr values, the predicted ptr values, the number of times
# each feature showed up as a predictor, and the number of imputed values.
def create_kfold_model(columns, taxa, ptrs, abundances, metadata, k=3):
    counts_dict = {}
    gt = np.asarray([])
    pred = np.asarray([])
    kfold = KFold(k, False, 0)
    imputed_values = 0
    true_values = 0
    fold = 0

    # For each fold
    for train, test in kfold.split(ptrs.columns):
        fold += 1
        ptrs_train = ptrs[ptrs.columns[train]]
        ptrs_test = ptrs[ptrs.columns[test]]

        # Check that there is enough data to train with
        if sum(np.isnan(ptrs_train.loc[taxa, :]) == False) > 0:
            # Impute the values for the training data
            # nmf_model = NMF(random_state=0, solver='mu', init='random')
            # nmf_model.fit(ptrs_train.T)
            # H = nmf_model.components_
            # W_train = nmf_model.transform(ptrs_train.T)
            #
            # matrix_factorized_train_ptrs = np.matmul(W_train, H)
            # ptrs_train[:] = matrix_factorized_train_ptrs.T

            # Create the datasets
            X_train, y_train = create_dataset(taxa, ptrs_train, abundances, metadata)
            X_test, y_test = create_dataset(taxa, ptrs_test, abundances, metadata)

            print(ptrs_test.loc[taxa, :])
            print(y_test)

            # if len(X_test) > 0 and len(X_train) > 0:
            true_values += len(X_test)

            # Create the regression
            reg = scikitlm.Ridge(alpha=ALPHA)
            reg.fit(X_train, y_train)

            for item in list(set(columns[reg.coef_ != 0])):
                if item in counts_dict.keys():
                    counts_dict[item] += 1
                else:
                    counts_dict[item] = 1

            # Add the results
            try:
                y_pred = reg.predict(X_test)
                gt = np.append(gt, y_test)
                pred = np.append(pred, y_pred)
            except Exception:
                print("Could not predict taxa " + taxa + " on fold " + str(fold))

    print("Taxa " + taxa + " true values is: " + str(true_values))

    return gt, pred, counts_dict, imputed_values

# This creates a ridge regression model for the PTR value of a given taxa using one versus all model.
# It returns 4 values: the ground truth ptr values, the predicted ptr values, the number of times
# each feature showed up as a predictor, and the number of imputed values.
def create_one_vs_all_model(columns, taxa, ptrs, abundances, metadata):
    counts_dict = {}
    gt = np.asarray([])
    pred = np.asarray([])
    imputed_values = 0
    true_values = 0


    # Go through all the columns that have a valid PTR value
    for col in ptrs.loc[:, list(np.isnan(ptrs.loc[taxa, :]) == False)].columns:

        ptrs_test = ptrs.loc[:, ptrs.columns == col]
        ptrs_train = ptrs.loc[:, ptrs.columns != col]

        # Check that there is enough data to train with
        if sum(np.isnan(ptrs_train.loc[taxa, :]) == False) > 0:
            # Impute the values for the training data
            # nmf_model = NMF(random_state=0, solver='mu', init='random')
            # nmf_model.fit(ptrs_train.T)
            # H = nmf_model.components_
            # W_train = nmf_model.transform(ptrs_train.T)
            #
            # matrix_factorized_train_ptrs = np.matmul(W_train, H)
            # ptrs_train[:] = matrix_factorized_train_ptrs.T

            # Create the datasets
            X_train, y_train = create_dataset(taxa, ptrs_train, abundances, metadata)
            X_test, y_test = create_dataset(taxa, ptrs_test, abundances, metadata)


            # if len(X_test) > 0 and len(X_train) > 0:
            true_values += len(X_test)

            # Create the ridge regression
            reg = scikitlm.Ridge(alpha=ALPHA)
            reg.fit(X_train, y_train)

            for item in list(set(columns[reg.coef_ != 0])):
                if item in counts_dict.keys():
                    counts_dict[item] += 1
                else:
                    counts_dict[item] = 1

            # Add the results
            try:
                y_pred = reg.predict(X_test)
                gt = np.append(gt, y_test)
                pred = np.append(pred, y_pred)
            except Exception:
                print("Could not predict taxa " + taxa)

    print("Taxa " + taxa + " true values is: " + str(true_values))

    return gt, pred, counts_dict, imputed_values

# This method filters the ptr spreadsheet to only include taxa that are in the relative abundance dataframe
def filter_ptrs(ptrs, abundances, metadata):
    abundances = abundances.set_index('OTUID')
    valid_samples = np.asarray([])
    for sample_id in ptrs.columns:
        if sample_id in abundances.index and int(sample_id) in metadata.index:
            valid_samples = np.append(valid_samples, sample_id)

    return ptrs[valid_samples]



if __name__ == "__main__":
    # Read in the Abundance and PTR spreadsheets
    abundances = pd.read_excel("plt_shotgun_and_16s.xlsx", sheet_name="relative_abundance_overlap")
    ptrs = pd.read_csv("./ptr_csvs/all_ptrs.csv")
    ptrs = ptrs.set_index('Unnamed: 0')
    ptrs.rename(columns=lambda x: x.split("_")[0], inplace=True)

    # Create the cutoff for PTR values. They should all be at least one because there should
    # be more replication origins than terminus.
    ptrs[ptrs < 1] = 1

    # Generate the names of the taxa in the abundances dataframe
    names = generate_names_with_ID(abundances)

    # Find the taxa to predict
    ptr_taxa = find_matching_taxa(ptrs, abundances, names)

    # Add the metadata which includes the antibiotics and colonization events
    metadata = pd.read_excel("LT_metadata.xlsx", sheet_name='Metadata')
    metadata = metadata.set_index('StoolID')

    # Choose the columns to use as features
    columns = np.append(names,
                        list(metadata.loc[:, 'ESBL_col_within1yr':'carb_14d_prev'].columns))
    full_counts_dict = {}

    # Filter the ptr dataframe
    ptrs = filter_ptrs(ptrs, abundances, metadata)
    ptrs = ptrs.dropna(axis='columns', how='all')
    ptrs = ptrs.dropna(axis='rows', how='all')


    gt_results = np.asarray([])
    predicted_results = np.asarray([])
    num_imputed_values = 0
    full_num_other_values = 0

    # Remove this taxa as there are only null values in the PTR spreadsheet of samples that are in the
    # relative abundance spreadsheet.
    ptr_taxa.remove('Morganella morganii')
    # ptr_taxa = ['Enterococcus faecium', 'Klebsiella pneumoniae', 'Enterobacter cloacae',
    #             'Enterococcus faecalis', 'Klebsiella oxytoca', 'Klebsiella variicola',
    #             'Enterococcus casseliflavus']

    # The below includes just the MDRO bacteria
    # ptr_taxa = ['Enterobacter cloacae', 'Klebsiella pneumoniae',
    #             'Enterobacter aerogenes', 'Klebsiella oxytoca', 'Klebsiella variicola']

    # For each taxa, predict results
    for taxa in ptr_taxa:
        try:
            # This will create the imputed PTR values
            """
            gt, predicted, num_imputed, num_other_values = create_imputed_values(taxa, ptrs)
            gt_results = np.append(gt_results, gt)
            predicted_results = np.append(predicted_results, predicted)
            num_imputed_values += num_imputed
            full_num_other_values += num_other_values
            """

            # This will predict the ptr value for the given taxa
            gt, predicted, counts_dict, num_imputed = create_one_vs_all_model(columns, taxa, ptrs, abundances, metadata)
            gt_results = np.append(gt_results, gt)
            predicted_results = np.append(predicted_results, predicted)
            num_imputed_values += num_imputed

            for item in counts_dict.keys():
                if item in full_counts_dict.keys():
                    full_counts_dict[item] += counts_dict[item]
                else:
                    full_counts_dict[item] = counts_dict[item]

        except Exception as e:
            print("Taxa " + taxa + " because there were no ptr values in overlaping samples.")

    # Here we return the results of the prediction
    print("Mean Squared Error is: " + str(metrics.mean_squared_error(y_true=gt_results, y_pred=predicted_results)))
    pearson_result = pearsonr(gt_results, predicted_results)
    print("Pearson Correlation Coefficient between Ground Truth and Prediction is: " +
          str(pearson_result[0]) + " with a p-value of: " + str(pearson_result[1]))
    print("Number of Predictions: " + str(len(predicted_results)))
    print("Number of Imputed Values: " + str(num_imputed_values))
    print("Average Number of PTR values on each imputed Value: " + str(float(full_num_other_values)/num_imputed_values))


    # print(dict(sorted(full_counts_dict.items(), key=operator.itemgetter(1),reverse=True)))
    # with open('sorted_feature_prevalence.json', 'w') as json_file:
    #     json.dump(dict(sorted(full_counts_dict.items(), key=operator.itemgetter(1),reverse=True)), json_file)


    # Here we plot the predicted values against the true values
    print(gt_results)
    print(predicted_results)
    plt.plot(gt_results, predicted_results, 'ko')
    plt.xlabel("True log(ptr_value)")
    plt.ylabel("Predicted log(ptr_value)")
    plt.title("Predicted log(PTR) Values")

    textstr = "MSE={:.4f}".format(metrics.mean_squared_error(y_true=gt_results, y_pred=predicted_results))
    textstr += "\nPearson Correlation={:.4f}".format(pearson_result[0])
    textstr += "\nPearson p-Value={:.4f}".format(pearson_result[1])

    plt.ylim(bottom=-.05, top=.6)
    plt.xlim(left=-.05, right=.6)
    plt.text(0.33, -0.03, textstr, fontsize=10, bbox=dict(facecolor='black', alpha=0.1))

    plt.plot([-.1, 1], [-.1, 1], 'k--')

    plt.show()
    # plt.savefig("mdro_imputed_log(ptr).png", bbox_inches='tight')