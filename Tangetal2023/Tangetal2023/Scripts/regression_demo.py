#! /bin/env python
#-*- coding:utf-8 -*-

__author__ = 'Rongyun Tang'
import time
start = time.time()

import os, glob, pdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error,\
    explained_variance_score, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, SGDRegressor, BayesianRidge, ElasticNet, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor, StackingRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import openpyxl
import shutil

end3 = time.time()
print('block 3, using time: ', end3 - start)

import warnings
warnings.filterwarnings('ignore')
random_seed = 1
print('librarys are loaded!')

def read_classification_error(path_error):
    file_err = path_error + 'classification_accuracy_summary.xlsx'
    err_data = pd.read_excel(file_err)
    PPV = err_data[(err_data.Model == 'RF') & (err_data.Type == 'testing')].PPV.values[0]
    FDR = err_data[(err_data.Model == 'RF') & (err_data.Type == 'testing')].FDR.values[0]
    FOR = err_data[(err_data.Model == 'RF') & (err_data.Type == 'testing')].FOR.values[0]
    NPV = err_data[(err_data.Model == 'RF') & (err_data.Type == 'testing')].NPV.values[0]
    return FDR, FOR, NPV, PPV


def create_directories(simu_name, time_step, path_file):
    with open(path_file, "r") as file:
        lines = file.readlines()
        raw_path = lines[0].strip().split(": ")[1]
        input_location = lines[1].strip().split(": ")[1]

    # Resolve relative path from the scripts/ directory
    if not os.path.exists(raw_path) and os.path.exists('../' + raw_path.lstrip('./')):
        raw_path = '../' + raw_path.lstrip('./')

    path_input = os.path.join(raw_path, 'Classification_' + simu_name, 'Clean_data_' + time_step) + '/'
    path_error = path_input
    path_output_train = path_input + 'Regression_train/'
    path_output_test = path_input + 'Regression_test/'

    for directory in [path_output_train, path_output_test]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('create directory: ', directory)

    return path_input, input_location, path_error, path_output_train, path_output_test


def metrics(true, preds):
    MSE = mean_squared_error(true, preds)
    MAE = mean_absolute_error(true, preds)
    VAR = explained_variance_score(true, preds)
    T2 = pd.DataFrame({'True1': true, 'Pred1': preds})
    T2 = T2.loc[(T2 != 0).any(axis=1)]
    R2 = r2_score(T2.True1.tolist(), T2.Pred1.tolist()) if len(T2) > 1 else 0.0
    MAPE = mean_absolute_percentage_error(true, preds)
    return [MSE, MAE, VAR, R2, MAPE]


def run_exps_seasons(month: str,
                     path_error,
                     path_output_train, path_output_test,
                     X_train_r: pd.DataFrame, y_train_r: pd.DataFrame,
                     X_test_r: pd.DataFrame, y_test_r: pd.DataFrame,
                     train_ind_r: list, test_ind_r: list,
                     X_test_0_r: pd.DataFrame, y_test_0_r: pd.DataFrame,
                     test_ind_0_r) -> pd.DataFrame:

    results_test_score = pd.DataFrame(columns=['Model', 'Month', 'Type', 'MSE', 'MAE', 'VAR', 'R2', 'MAPE'])
    results_train_score = pd.DataFrame(columns=['Model', 'Month', 'Type', 'MSE', 'MAE', 'VAR', 'R2', 'MAPE'])

    models = [
        ('LinR', LinearRegression(n_jobs=-1)),
        ('Ridge', Ridge(random_state=random_seed)),
        ('Lasso', Lasso(alpha=0.1)),
        ('Ada', AdaBoostRegressor(n_estimators=1000, learning_rate=0.0001, loss='linear', random_state=1)),
        ('GBR', GradientBoostingRegressor(learning_rate=0.0001, n_estimators=1000,
                                          max_depth=10, random_state=1, max_features=10)),
        ('Bag', BaggingRegressor(n_estimators=50, random_state=1)),
        ('RF', RandomForestRegressor(n_estimators=1000, max_depth=20)),
        ('Bayes', BayesianRidge(max_iter=60000)),
        ('EN',  ElasticNet(max_iter=60000, random_state=1)),
        ('Kernel', KernelRidge(alpha=1.0,  kernel='linear',  degree=10)),
        ('DT', DecisionTreeRegressor()),
        ('XGBR', XGBRegressor(booster='gblinear', objective="reg:squarederror",
                              random_state=1, n_estimators=1000, max_depth=10, learning_rate=0.0001)),
        ('CBR', CatBoostRegressor(verbose=0)),
        ('LGBR', LGBMRegressor(verbosity=-1)),
    ]
    level0 = models.copy()
    stack_model = StackingRegressor(estimators=level0,
                                    final_estimator=RandomForestRegressor(n_estimators=1000, random_state=random_seed))
    models.append(('Stack', stack_model))

    column_list = ['Type', 'raw_index', 'Month', 'Modeled', 'Observed', 'Error_Adjusted']

    for name, model in models:
        train_result_df = pd.DataFrame(index=range(len(train_ind_r)), columns=column_list)
        test_result1_df = pd.DataFrame(index=range(len(test_ind_r)), columns=column_list)
        test_result0_df = pd.DataFrame(index=range(len(test_ind_0_r)), columns=column_list)

        if len(train_ind_r) > 3 and len(test_ind_r) > 3:
            try:
                clf = model.fit(X_train_r, y_train_r)

                y_train_fitted = clf.predict(X_train_r)
                y_pred_r = clf.predict(X_test_r)

                FDR, FOR, NPV, PPV = read_classification_error(path_error)

                scores = metrics(y_test_r, y_pred_r)
                results_test_score.loc[len(results_test_score.index)] = [name, month, 'testing'] + scores

                scores_train = metrics(y_train_r, y_train_fitted)
                results_train_score.loc[len(results_train_score.index)] = [name, month, 'training'] + scores_train

                train_result_df['Modeled'] = y_train_fitted
                test_result1_df['Modeled'] = y_pred_r
                test_result0_df['Modeled'] = 0

                train_result_df['Error_Adjusted'] = y_train_fitted
                test_result1_df['Error_Adjusted'] = y_pred_r * PPV + FDR * 0
                test_result0_df['Error_Adjusted'] = y_test_0_r.values * FOR + NPV * 0
            except Exception as e:
                print(f"Skipping {name} for month {month}: {e}")
                results_test_score.loc[len(results_test_score.index)] = [name, month, 'testing', np.nan, np.nan, np.nan, np.nan, np.nan]
                results_train_score.loc[len(results_train_score.index)] = [name, month, 'training', np.nan, np.nan, np.nan, np.nan, np.nan]
                train_result_df['Modeled'] = 0
                test_result1_df['Modeled'] = 0
                test_result0_df['Modeled'] = 0
                train_result_df['Error_Adjusted'] = 0
                test_result1_df['Error_Adjusted'] = 0
                test_result0_df['Error_Adjusted'] = 0
        else:
            results_test_score.loc[len(results_test_score.index)] = [name, month, 'testing', np.nan, np.nan, np.nan, np.nan, np.nan]
            results_train_score.loc[len(results_train_score.index)] = [name, month, 'training', np.nan, np.nan, np.nan, np.nan, np.nan]

            train_result_df['Modeled'] = 0
            test_result1_df['Modeled'] = 0
            test_result0_df['Modeled'] = 0
            train_result_df['Error_Adjusted'] = 0
            test_result1_df['Error_Adjusted'] = 0
            test_result0_df['Error_Adjusted'] = 0

        train_result_df['Type'] = 1
        test_result1_df['Type'] = 1
        test_result0_df['Type'] = 0

        train_result_df['raw_index'] = train_ind_r
        test_result1_df['raw_index'] = test_ind_r
        test_result0_df['raw_index'] = test_ind_0_r

        train_result_df['Month'] = month
        test_result1_df['Month'] = month
        test_result0_df['Month'] = month

        train_result_df['Observed'] = y_train_r.values
        test_result1_df['Observed'] = y_test_r.values
        test_result0_df['Observed'] = y_test_0_r.values

        results_test_df = pd.concat([test_result1_df, test_result0_df], axis=0, ignore_index=True)

        file_out_prd = path_output_train + 'Regression_' + name + '_train_prd.xlsx'
        file_out_prd_all = path_output_test + 'Regression_' + name + '_test_prd.xlsx'

        with pd.ExcelWriter(file_out_prd, engine='openpyxl') as writer:
            train_result_df.to_excel(writer, sheet_name=str(month), index=True)

        with pd.ExcelWriter(file_out_prd_all, engine='openpyxl') as writer_all:
            results_test_df.to_excel(writer_all, sheet_name=str(month), index=True)

    return results_test_score, results_train_score


def read_iputs(path_input):
    file = path_input + 'classification_input.xlsx'
    file_pred = path_input + 'classification_output_predictions.xlsx'

    train_ind = pd.read_excel(file, sheet_name='ind_train', index_col=0)
    test_ind = pd.read_excel(file, sheet_name='ind_test', index_col=0)

    df_train_reg = pd.read_excel(file, sheet_name='training_std_before_oversampling', index_col=0)
    df_test_reg = pd.read_excel(file, sheet_name='testing_std', index_col=0)

    df_test_pred = pd.read_excel(file_pred, sheet_name='test')

    return train_ind, test_ind, df_train_reg, df_test_reg, df_test_pred


def main():
    simu_name = 'all'
    time_step = 'all-year'

    path_file = '../Model inputs/file_paths_regression.txt'
    if not os.path.exists(path_file):
        path_file = './Model inputs/file_paths_regression.txt'

    path_input, input_location, path_error, path_output_train, path_output_test = \
        create_directories(simu_name, time_step, path_file)

    accuracy_TST_list = []
    accuracy_Train_list = []

    train_ind, test_ind, df_train_reg, df_test_reg, df_test_pred = read_iputs(path_input)

    input_var = df_train_reg.columns[1:]
    target_var = df_test_reg.columns[-1]

    seasons = [('all-year', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])]
    month_list = seasons[0][1]

    for month in month_list:
        print(f"Processing month: {month}")

        ind_selected_train = df_train_reg[(df_train_reg[target_var] > 0) & (df_train_reg['Month'] == month)].raw_index.values.tolist()
        ind_selected_test = df_test_pred[(df_test_pred['RF'] == 1) & (df_test_pred['Month'] == month)].raw_index.values.tolist()

        X_test = df_test_reg[df_test_reg.raw_index.isin(ind_selected_test)].iloc[:, :-1]
        y_test = df_test_reg[df_test_reg.raw_index.isin(ind_selected_test)].iloc[:, -1]
        X_train = df_train_reg[df_train_reg.raw_index.isin(ind_selected_train)].iloc[:, :-1]
        y_train = df_train_reg[df_train_reg.raw_index.isin(ind_selected_train)].iloc[:, -1]

        ind_selected_train_0 = df_train_reg[(df_train_reg[target_var] == 0) & (df_train_reg['Month'] == month)].raw_index.values.tolist()
        X_train_0 = df_train_reg[df_train_reg.raw_index.isin(ind_selected_train_0)].iloc[:, :-1]
        y_train_0 = df_train_reg[df_train_reg.raw_index.isin(ind_selected_train_0)].iloc[:, -1]

        ind_selected_test_0 = df_test_pred[(df_test_pred['RF'] == 0) & (df_test_pred['Month'] == month)].raw_index.values.tolist()
        X_test_0 = df_test_reg[df_test_reg.raw_index.isin(ind_selected_test_0)].iloc[:, :-1]
        y_test_0 = df_test_reg[df_test_reg.raw_index.isin(ind_selected_test_0)].iloc[:, -1]

        accuracy_test, accuracy_train = run_exps_seasons(month,
                                                         path_error,
                                                         path_output_train, path_output_test,
                                                         X_train, y_train,
                                                         X_test, y_test,
                                                         ind_selected_train, ind_selected_test,
                                                         X_test_0, y_test_0, ind_selected_test_0)
        accuracy_TST_list.append(accuracy_test)
        accuracy_Train_list.append(accuracy_train)

    accuracy_TST = pd.concat(accuracy_TST_list, ignore_index=True)
    accuracy_Train = pd.concat(accuracy_Train_list, ignore_index=True)

    file_out_metrics = path_output_test + 'regression_accuracy_summary.xlsx'
    with pd.ExcelWriter(file_out_metrics, engine='openpyxl') as writer:
        accuracy_TST.to_excel(writer, sheet_name='regression_accuracy_test_only', index=True)
        accuracy_Train.to_excel(writer, sheet_name='regression_accuracy_train', index=True)

    print("Regression simulations finished successfully!")

if __name__ == "__main__":
    main()