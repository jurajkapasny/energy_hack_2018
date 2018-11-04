
# coding: utf-8

# In[36]:


import random
import pandas as pd 
from copy import deepcopy
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pickle
# from skopt import BayesSearchCV
import datetime
import itertools


# In[2]:


# data_path = "/Users/jurajkapasny/Data/energy_hack/"
# df = pd.read_csv(data_path+"spotreba_prepared.csv",sep = ";")


# In[ ]:


df = pd.read_csv("./data/spotreba_prepared.csv",sep = ";")


# In[3]:


df["timestamp"] = df["Dátum a čas"] + " " + df["Unnamed: 1"]


# In[4]:


df = df[["spotreba","om","timestamp"]]


# In[5]:


df.head()


# In[6]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
    cols.append(df.shift(-n_out))
    if n_out == 1:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[13]:


# new method saves test_cols and train_cols as a dictionary not tuple
def prep_data_to_train_new(input_data, n_in=30, n_out=1,test_size= 0.2,bootstrap=False, bs_replicates = 10, file_name="prepared_data"):
    
    time_series = {}
    ts_test = []
    ts_train = []
    train_cols = []
    test_cols = []
    max_index_ts = 0
    

    columns = ["timestamp","spotreba","om"]
    cl_group_by = ["om","timestamp"]
    data_sum = input_data[columns].groupby(cl_group_by).sum().unstack()
    data_sum = data_sum[data_sum.notnull().sum(axis=1) > (data_sum.shape[1]*0.8)].transpose().fillna(0)#.index.droplevel()
    data_sum.index = data_sum.index.droplevel()
#         print(len(data_sum.columns))

    columns = data_sum.columns

    for i in range(data_sum.shape[1]):
        ts = series_to_supervised(data_sum.iloc[:,i].tolist(),n_in=n_in, n_out=n_out).reset_index().drop("index",axis=1)

        if bootstrap:
            temp_ts = []
#                 print(ts.shape)
            for random_point in random.sample(range(0,len(ts)), bs_replicates):
                temp_ts.append(pd.DataFrame(ts.iloc[random_point,:]).transpose())

            time_series[columns[i]] = pd.concat(temp_ts,ignore_index=True)

        else:
            time_series[columns[i]] = ts



    test_set_random_index = random.sample(range(max_index_ts,len(time_series)),
                                          int((len(time_series)- max_index_ts)*test_size))

    ts_keys = list(time_series.keys())

    for index in range(max_index_ts,len(time_series)):
        if index in test_set_random_index:
            ts_test.append(time_series[ts_keys[index]])
            temp_dict_test ={}
#             for i in range(len(selected_columns)):
            temp_dict_test["om"] = ts_keys[index]

            test_cols.append(temp_dict_test)
        else:
            ts_train.append(time_series[ts_keys[index]])
            temp_dict_train ={}
#             for i in range(len(selected_columns)):
            temp_dict_train["om"] = ts_keys[index]
            train_cols.append(temp_dict_train)


    max_index_ts = len(time_series)

    
    ts_train_df = pd.concat(ts_train, ignore_index= True)
    ts_test_df = pd.concat(ts_test,ignore_index =True)
    
    ts_train_df.to_csv(f"./models/{file_name}_train.csv",index= False)
    ts_test_df.to_csv(f"./models/{file_name}_test.csv", index = False)
    
    pickle.dump(test_cols, open(f"./models/{file_name}_test_cols.p","wb"))
    pickle.dump(train_cols, open(f"./models/{file_name}_train_cols.p","wb"))
    
    return ts_train_df, ts_test_df, train_cols, test_cols


# In[29]:


print("Preparing data!!!")
# %%time
ts_train_df,ts_test_df, train_cols, test_cols = prep_data_to_train_new(input_data = df,
                                            n_in=1000,
                                            n_out=1,
                                            test_size= 0.2,
                                            bootstrap=True,
                                            bs_replicates = 10000, 
                                            file_name="x1000_bootstrap10000")
print("Data prepared!!!")


# In[31]:


ts_train_df.head()


# In[33]:


def train(ts_train_df,ts_test_df, cv = 5,type_ = "RandomForest",scoring = "mae", file_name="rf_test"):
    
    scores = dict()
    
    X_train = ts_train_df.iloc[:,:-1]
    y_train = ts_train_df.iloc[:,-1:]
    
    X_test = ts_test_df.iloc[:,:-1]
    y_test = ts_test_df.iloc[:,-1:]
    
        
    grid_search = train_model(X_train,y_train, type_ = type_, scoring = scoring, file_name = file_name)
    
    y_pred = grid_search.best_estimator_.predict(X_test)

    r2score = r2_score(y_pred, y_test)
    mae = mean_absolute_error(y_pred, y_test)
    medae = median_absolute_error(y_pred, y_test)
    
    
    scores["cv_score_mean"] = grid_search.best_score_
    scores["r2_test_set"] = r2score
    scores["mae_test_set"] = mae
    scores["medae_test_set"] = medae
    scores["X_train_rows"] = X_train.shape[0]
    scores["model_max_depth"] = grid_search.best_params_["max_depth"]
    scores["model_learning_rate"] = grid_search.best_params_["learning_rate"]
    scores["model_n_estimators"] = grid_search.best_params_["n_estimators"]
    scores["settings_cv"] = cv
    
    scores_df = pd.DataFrame(scores,index=[0])
    scores_df.to_csv(f"./models/{file_name}_model_info.csv",index= False)
#     joblib.dump(regressor, f"./models/{file_name}.pkl")
        
        
    return scores_df


# In[34]:


def train_model(X_train,y_train,type_ = "xgboost", scoring = "neg_median_absolute_error", file_name="xgboost_test", cv=5):
    if type_ == "xgboost":
        
        parameters = {
                      "max_depth" : [6,12, 15],
                      "learning_rate" : [0.05,0.2],
                      "min_child_weight":[2,4],
                      "n_estimators" : [500,1000],
                    }

        grid_search = GridSearchCV(estimator=XGBRegressor(n_jobs = -1),
                                   param_grid=parameters,
                                   scoring = scoring,
                                   n_jobs=-1,
                                   cv = cv,
                                   verbose = 10)

    elif type_ == "RandomForest":
        parameters = {
                      "max_depth" : [5,10],
                      "n_estimators" : [100,500, 1000],
                    }

        grid_search = GridSearchCV(estimator=RandomForestRegressor(n_jobs = -1),
                                   param_grid=parameters,
                                   scoring = scoring,
                                   n_jobs=-1,
                                   cv = cv,
                                   verbose = 10)
    else:
        print("only xgboost or RandomForest")
        return None
    grid_search.fit(X_train,y_train)
    joblib.dump(grid_search, f"./models/{file_name}.pkl")
    return grid_search


# In[35]:


scores = train(ts_train_df = ts_train_df,
                       ts_test_df = ts_test_df, 
                       cv = 5,
                       type_ = "xgboost",
                       scoring = "neg_mean_absolute_error", 
                       file_name="XGB_mae_x1000_bootstrap10000_lag_1")
# scores

