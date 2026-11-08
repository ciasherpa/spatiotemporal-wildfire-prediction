#! /bin/env python
#-*- coding:utf-8 -*-

__author__='Rongyun Tang'

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, \
    precision_score, f1_score, roc_auc_score, roc_curve, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import xlsxwriter
import pickle
import warnings
import time, os, csv, pdb
from tqdm import tqdm
import multiprocessing

warnings.filterwarnings('ignore')
random_seed = 63445
random_seed = 1

def create_directories(path_file, sim_name, season_name):
    with open(path_file, "r") as file:
        lines = file.readlines()
        path_input = lines[0].strip().split(": ")[1]
        input_location = lines[1].strip().split(": ")[1]
        path_out = lines[2].strip().split(": ")[1]

    path_output = path_out + 'Classification_' + sim_name + '/'
    path_clean_data = path_output + 'Clean_data_' + season_name + '/'
    directories = [path_output, path_clean_data]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('create directory: ', directory)

    return path_input, input_location, path_output, path_clean_data


def metrics(true, preds):
    """
    Function to calculate evaluation metrics

    parameters: true values, predictions
    returns:  accuracy, recall, precision and f1 scores and other metrics (ROC curve, PR curve, AUC)
    """
    accuracy = accuracy_score(true, preds)
    recall = recall_score(true, preds)
    precision = precision_score(true, preds)
    f1score = f1_score(true, preds)
    # fpr, tpr, _ = roc_curve(true, preds)
    roc_auc = roc_auc_score(true, preds)
    kappa = cohen_kappa_score(true, preds)
    return [accuracy, recall, precision, f1score, roc_auc, kappa]

def confusion_metrics(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    TPR = tp / (tp + fn)  # True Positive Rate (Sensitivity)
    TNR = tn / (tn + fp)  # True Negative Rate (Specificity)
    FPR = fp / (fp + tn)  # False Positive Rate
    FNR = fn / (fn + tp)  # False Negative Rate
    PPV = tp / (tp + fp)  # Positive Predictive Value
    FDR = fp / (tp + fp)  # False Discovery Rate
    FOR = fn / (fn + tn)  # False Omission Rate
    NPV = tn / (tn + fn)  # Negative Predictive Value
    ACC = (tp + tn) / (tp + tn + fp + fn)  # Overall Accuracy
    PLR = TPR / (1 - TNR)  # Positive Likelihood Ratio
    NLR = FNR / (1 - TPR)  # Negative Likelihood Ratio
    metrics = {'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR,
               'FDR': FDR, 'PPV': PPV, 'FOR': FOR, 'NPV': NPV,
               'ACC': ACC, 'PLR':PLR, 'NLR': NLR}
    return metrics

def calculate_feature_importances(X_train, y_train, fitted_model, feature_importances):

    # Get the feature importances
    if isinstance(fitted_model, VotingClassifier):
        print('calculate importance by multi-model mean. ')
        numbers = [num for _, num in feature_importances]
        feature_imp = np.mean(numbers)
    elif hasattr(fitted_model, 'feature_importances_'):
        print('model has the attribute variable feature_importances_.')
        feature_imp = fitted_model.feature_importances_
    elif hasattr(fitted_model, 'coef_'):
        print('model has the attribute variable coef_.')
        feature_imp = fitted_model.coef_[0]
    elif hasattr(fitted_model, 'estimators_'):
        print('model has the attribute variable estimators_.')
        feature_imp = np.mean([tree.feature_importances_ for tree in fitted_model.estimators_], axis=0)
    else:
        try:
            print('calculate importance by permutation. ')
            result = permutation_importance(fitted_model, X_train, y_train, n_repeats=10, random_state=0)
            feature_imp = result.importances_mean
        except:
            print('feature importance is not calculated')

    return feature_imp


def test_models(X_train, y_train, X_test, y_test, idx_train, idx_test, season_name, path_clean_data):
    # Initialize a list to store the results
    results_each_model = []
    results_summary_df = pd.DataFrame(
        columns=['Model', 'Seasons', 'Type', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'AUC', 'Kappa',
                 'PLR', 'NLR', 'PPV', 'FDR', 'FOR', 'NPV'])
    feature_importances = []
    model_numbers = 3
    pre_train = np.zeros((len(idx_train), 3 + model_numbers))
    pre_test = np.zeros((len(idx_test),  3 + model_numbers))

    # Filling part of the results with traceable information
    pre_train[:, 0] = idx_train
    pre_train[:, 1] = X_train.Month
    pre_train[:, 2] = y_train
    pre_test[:, 0] = idx_test
    pre_test[:, 1] = X_test.Month
    pre_test[:, 2] = y_test

    # Models for loop
    models = [
        ('LogReg', LogisticRegression(max_iter=100000)),
        ('RF', RandomForestClassifier(n_estimators=2000, oob_score=True, random_state=random_seed)),
        # ('BAG', BaggingClassifier(n_estimators=3000, random_state=random_seed)),
        # ('KNN', KNeighborsClassifier(n_neighbors=10)),
        # ('SVM', SVC(kernel='linear', probability=True)),
        # ('GNB', GaussianNB()),
        # ('MLP', MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15, 10),
        #                       random_state=random_seed, max_iter=150000))
    ]
    level0 = models.copy()
    volt_model = VotingClassifier(estimators=level0, voting='soft', n_jobs=-1)
    models.append(('Volt', volt_model))

    # Loop through the models
    for id_model, each_model in enumerate(models):
        name, model = each_model
        print('run the model: ', name)
        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)
        y_fit = model.predict(X_train)

        # Save fitting and predicting results
        pre_test[:, id_model + 3] = y_pred
        pre_train[:, id_model + 3] = y_fit

        # Save model
        pkl_filename = path_clean_data + name + ".pkl"
        with open(pkl_filename, 'wb') as pkf:
            pickle.dump(model, pkf)

        # Compute and append feature importance
        print('calculate feature importance for model: ', name)
        feature_imp = calculate_feature_importances(X_train, y_train, model, feature_importances)
        feature_importances.append((name, feature_imp))
        print(feature_importances)

        # Compute the accuracy scores
        print('calculate model accuracies.')
        accuracy = accuracy_score(y_test, y_pred)
        met_test = metrics(y_test, y_pred)
        met_train = metrics(y_train, y_fit)

        # Compute confusion matrix
        conf_mat_test = confusion_matrix(y_test, y_pred)
        report_test = classification_report(y_test, y_pred, output_dict=True)
        conf_mat_train = confusion_matrix(y_train, y_fit)
        report_train = classification_report(y_train, y_fit, output_dict=True)

        # Compute other accuracy scores from training and testing confusion matrix
        stage_name = ['testing', 'training']
        met_list = [met_test, met_train]
        conf_list = [conf_mat_test, conf_mat_train]
        for cf, cof in enumerate(conf_list):
            conf_metrics = confusion_metrics(cof)
            selected_metrics = [value for key, value in conf_metrics.items() if
                                key in ['PLR', 'NLR', 'PPV', 'FDR', 'FOR', 'NPV']]
            rst = [name, season_name, stage_name[cf]] + met_list[cf] + selected_metrics

            # Save model accuracy metrics for both training and testing stages into the results dataframe
            results_summary_df.loc[len(results_summary_df.index)] = rst

        # Write intermediate results for each model in iterations
        results_each_model.append({'test_accuracy': [model, accuracy],
                                 'test_confx': pd.DataFrame(conf_mat_test),
                                 'test_confx_report': pd.DataFrame(report_test).transpose(),
                                 'train_accuracy': [model, met_train[0]],
                                 'train_confx': pd.DataFrame(conf_mat_train),
                                 'train_confx_report': pd.DataFrame(report_train).transpose()
                                 })

    # Create a pandas DataFrame to store the results
    feature_importances_dict = dict(feature_importances)
    df_importance = pd.DataFrame.from_dict(feature_importances_dict)
    df_importance.index = X_train.columns.tolist()

    model_list = list(map(lambda t: t[0], models))
    columns = ['raw_index', 'Month', 'Observed'] + model_list
    pre_train_df, pre_test_df = (pd.DataFrame(columns=columns, data=pre_val) for pre_val in [pre_train, pre_test])
    pre_all_df = np.concatenate((pre_train_df, pre_test_df))

    return results_summary_df,  df_importance.T, results_each_model, pre_train_df, pre_test_df, pre_all_df


def preprocess(path_input, path_clean_data, feature_var, target_var):
    # Save intermediate data
    file_out = path_clean_data + 'classification_input.xlsx'
    writer = pd.ExcelWriter(file_out, engine='openpyxl')

    # Read Data and Set traceable index
    df_r0 = pd.read_csv(path_input + 'input_FireCCI_samples.csv')  # (26676, 23)
    index0 = df_r0.index
    df_r0.insert(loc=0, column='raw_index', value=index0)

    # Write traceable data and save as row data
    df_r0.to_excel(writer, sheet_name='raw_data', index=True)

    # Drop rows with nans and save as clean data
    # df_r2 = df_r0.drop(np.where(np.isnan(df_r0))[0])
    df_r2 = df_r0[~df_r0.isin([np.nan, np.inf, -np.inf]).any(axis=1)]  # (26448, 23)
    df_r2.to_excel(writer, sheet_name='clean_data', index=True)

    # Do normalization(z = (x - u) / s) on clean data， except for raw_index and month
    scaler = StandardScaler()
    df_r2_sd = scaler.fit_transform(df_r2)
    df_r2_sd = pd.DataFrame(df_r2_sd, columns=df_r2.columns)
    raw_index_val = df_r2['raw_index']
    df_r2_sd['raw_index'] = df_r2['raw_index'].values
    df_r2_sd['Month'] = df_r2['Month'].values
    df_r2_sd.to_excel(writer, sheet_name='Standarlizataion_all', index=True)

    # Select subset features and predictive variable
    feature = df_r2_sd[feature_var].copy()
    target = df_r2[target_var].copy() # Make target as binary var
    target[target > 0] = 1

    # Stratify split (18514, 7935)
    X_train, X_test, y_train, y_test, ind_train, ind_test \
        = train_test_split(feature, target, df_r2_sd['raw_index'],
                           test_size=0.3, random_state=random_seed, stratify=target)

    # Save raw index and raw data after data split
    ind_test.to_excel(writer, sheet_name='ind_test', index=True)
    ind_train.to_excel(writer, sheet_name='ind_train', index=True)
    df_r0.loc[df_r0['raw_index'].isin(ind_train)].to_excel(writer, sheet_name='training_original', index=True)
    df_r0.loc[df_r0['raw_index'].isin(ind_test)].to_excel(writer, sheet_name='testing_original', index=True)

    # Save standardized data before oversampling and add target column
    X_train.insert(loc=0, column='raw_index', value=ind_train)
    X_test.insert(loc=0, column='raw_index', value=ind_test)
    X_train[target_var] = y_train.values # include both features and target
    X_train.to_excel(writer, sheet_name='training_std_before_oversampling', index=True)
    X_test[target_var] = y_test.values # include both features and target
    X_test.to_excel(writer, sheet_name='testing_std', index=True)

    # Do oversampling for standardized training data
    sm = SMOTE(random_state=random_seed)
    X_train = X_train.reset_index(drop=True)
    X_train_sm, y_train_sm = sm.fit_resample(X_train.iloc[:, :-1], y_train)
    df_sm = pd.concat([X_train_sm, y_train_sm], axis=1)
    df_sm.to_excel(writer, sheet_name='training_std_after_oversampling', index=True)
    writer.close()
    print('preprocessing is finished!')
    return df_r2_sd['raw_index'], ind_train, ind_test, X_train, X_test, y_train, y_test, X_train_sm, y_train_sm


def time_steps(X_train_sm, y_train_sm, X_test, y_test, seasons):
    # Data out
    data_sets = []

    # Obtain season month lists
    season_name, season_list = seasons

    # Reset index
    X_train_sm = X_train_sm.reset_index(drop=True)
    y_train_sm = y_train_sm.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Columns for models
    Vars_selected = [ 'Month', 'pet', 'tmn', 'cld', 'dtr', 'vap', 'wet',
                     'pre', 'tmp', 'tmx', 'frs', 'RH', 'SVP', 'VPD', 'windspeed', 'ndvi',
                     'GPP', 'popd', 'PDSI', 'SMroot', 'SMsurf']

    # Obtain subsets with season months
    ind_selected_train = X_train_sm.loc[X_train_sm['Month'].isin(season_list)].index.values.tolist()
    raw_idx_train = X_train_sm.iloc[ind_selected_train]['raw_index']
    X_train_mon = X_train_sm.iloc[ind_selected_train][Vars_selected]
    y_train_mon = y_train_sm.iloc[ind_selected_train].astype(int)
    print(X_train_mon.columns)

    ind_selected_test = X_test.loc[X_test['Month'].isin(season_list)].index.values.tolist()
    raw_idx_test = X_test.iloc[ind_selected_test]['raw_index']
    X_test_mon = X_test.iloc[ind_selected_test][Vars_selected]  # X_test has the target column, remove it here
    y_test_mon = y_test.iloc[ind_selected_test].astype(int)
    print(X_test_mon.columns)

    # if data sample size is too small, we skip process for this month
    while len(y_test_mon) < 3:
        continue

    data_sets.append((season_name, X_train_mon, y_train_mon, X_test_mon, y_test_mon, raw_idx_train, raw_idx_test))

    return data_sets


def save_predictions(pre_train_df, pre_test_df, ind_train, ind_test, path_clean_data, season_name):
    # Define out file
    file_out = path_clean_data + 'classification_output_predictions.xlsx'
    writer = pd.ExcelWriter(file_out, engine='openpyxl')

    # Remove oversampled results by row_index match
    pre_train_df = pre_train_df[pre_train_df['raw_index'].isin(ind_train.tolist())]
    pre_test_df = pre_test_df[pre_test_df['raw_index'].isin(ind_test.tolist())]
    pre_all_df = pd.concat([pre_train_df, pre_test_df], ignore_index=True)

    # Write predictions to files
    out_dfs = [('train', pre_train_df), ('test', pre_test_df), ('all', pre_all_df)]

    for each_element in out_dfs:
        stage, out_df = each_element
        out_df.to_excel(writer,
                        sheet_name=stage,
                        startrow=0,
                        index=False
                        )
    writer.close()

def save_evaluation_metrics(results_summary_df,  df_importance, results_each_model, path_clean_data, season_name):
    # Save accuracies of all models
    file_out = path_clean_data + 'classification_accuracy_summary.xlsx'
    results_summary_df.to_excel(file_out, sheet_name='accuracy')

    # Save information for each model
    file_out = path_clean_data + 'classification_accuracy_each_model.csv'
    results_each_model_df = pd.DataFrame.from_dict(results_each_model)
    results_each_model_df.to_csv(file_out)

    # Save feature importances
    file_out = path_clean_data + 'classification_feature_importance_summary.xlsx'
    df_importance.to_excel(file_out, sheet_name='importance')

def exe_simulations(simulation, seasons):
    # Prepare data for each simulation
    sim_name, sim_var = simulation
    season_name, season_list = seasons
    print('processing the simulation: ', sim_name, season_name)
    feature_var = sim_var[2:-1]
    target_var = sim_var[-1]

    # Create directories for each simulation
    path_input, input_location, path_output, path_clean_data = \
        create_directories(file_input, sim_name,  season_name)

    # Preprocessing data
    raw_index, ind_train, ind_test, X_train, X_test, \
    y_train, y_test, X_train_sm, y_train_sm = preprocess(path_input, path_clean_data, feature_var, target_var)

    # Set model time steps and run models on oversampled data
    data_sets = time_steps(X_train_sm, y_train_sm, X_test, y_test, seasons)

    for data in data_sets:

        # Ravel data sets
        (season_name, X_train_mon, y_train_mon, X_test_mon, y_test_mon, raw_idx_train, raw_idx_test) = data
        print('finished preparing data for the time step: ', season_name)

        # Run model
        print('run multiple models one at a time.')
        results_summary_df, df_importance, results_each_model, \
        pre_train_df, pre_test_df, pre_all_df = \
            test_models(X_train_mon, y_train_mon, X_test_mon, y_test_mon,
                        raw_idx_train, raw_idx_test, season_name, path_clean_data
                        )

        # Save predictions
        save_predictions(pre_train_df, pre_test_df, ind_train, ind_test, path_clean_data, season_name)

        # Save model accuracy metrics
        save_evaluation_metrics(results_summary_df, df_importance, results_each_model, path_clean_data, season_name)
    time.sleep(1)


def main():
    # Main code goes here
    # NOTE: file_input must be global — exe_simulations() (a separate
    # top-level function) reads it, but it was previously only a local
    # variable inside main(), causing a NameError.
    global file_input
    file_input = '../Model inputs/file_paths_classification.txt'

    # Set parameters
    all_var = ['row', 'column', 'Year', 'Month', 'pet', 'tmn', 'cld', 'dtr', 'vap', 'wet', 'pre', 'tmp',
                 'tmx', 'frs', 'RH', 'SVP', 'VPD', 'windspeed', 'ndvi', 'GPP', 'popd',
                 'PDSI', 'SMroot', 'SMsurf', 'BA_km2']  # 4 + 18 + 1
    tmp_var = ['tmn', 'tmp', 'tmx']
    pre_var = ['pre']
    hmi_var = ['vap', 'wet', 'RH', 'SVP', 'VPD', 'PDSI']
    soil_var = ['SMroot', 'SMsurf']

    # Set simulation groups
    # NOTE: use order-preserving list comprehensions instead of set
    # subtraction. `set` does not preserve order, and Python's string hash
    # randomization means that order changes between runs — so sim_var[-1]
    # (assumed to always be 'BA_km2', the fire target) could randomly end up
    # being a totally different variable each time the script runs.
    simulation_groups = [('all', all_var),
                         ('no-tmp', [v for v in all_var if v not in tmp_var]),
                         ('no-pre', [v for v in all_var if v not in pre_var]),
                         ('no-hmi', [v for v in all_var if v not in hmi_var]),
                         ('no-soil', [v for v in all_var if v not in soil_var]),
                         ('no-tmp-pre', [v for v in all_var if v not in tmp_var + pre_var]),
                         ('no-tmp-hmi', [v for v in all_var if v not in tmp_var + hmi_var]),
                         ('no-tmp-soil', [v for v in all_var if v not in tmp_var + soil_var]),
                         ('no-tmp-pre-hmi', [v for v in all_var if v not in tmp_var + pre_var + hmi_var])
                           ]

    # Set time steps for different scenarios
    seasons = [('all-year', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])]

    # Inputs for parallel processing, run one example
    # NOTE: simulation_groups[1] is a single (name, var_list) tuple. zip()
    # iterates over the *elements* of whatever it's given, so zipping the
    # tuple directly with `seasons` paired the string 'no-tmp' with the
    # season tuple instead of pairing the whole (name, var_list) tuple with
    # it. Wrapping it in a list makes zip treat it as one item, as intended.
    inputs = list(zip([simulation_groups[0]], seasons))

    # --- DEBUG CHANGE ---
    # multiprocessing.Pool removed for now: since there is only ever one task
    # in `inputs`, the Pool was just adding spawn/queue overhead (and a risk
    # of deadlocking on Windows) with no actual parallelism benefit. Running
    # directly in-process also means print() statements inside exe_simulations
    # show up live in the terminal, so we can see exactly where it is.
    print('mapping data and functions')
    results = [exe_simulations(*args) for args in tqdm(inputs, total=len(inputs))]
    print('running simulations')
    tuple(results)  # fetch the lazy results
    print('done!')
    # --- END DEBUG CHANGE ---

if __name__ == "__main__":
    main()
