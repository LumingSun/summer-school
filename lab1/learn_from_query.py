from unittest import TestLoader
import evaluation_utils as eval_utils
import matplotlib.pyplot as plt
import numpy as np
import range_query as rq
import json
import torch
import torch.nn as nn
import statistics as stats
import xgboost as xgb
from net import MLP

def min_max_normalize(v, min_v, max_v):
    # The function may be useful when dealing with lower/upper bounds of columns.
    assert max_v > min_v
    return (v-min_v)/(max_v-min_v)


def extract_features_from_query(range_query, table_stats, considered_cols):
    # feat:     [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
    #           <-                   range features                    ->, <-     est features     ->
    feature = []
    for col in considered_cols:
        if(col not in range_query.col_left):
            feature.extend([1.0,1.0])
        else:
            feature.append(min_max_normalize(range_query.col_left[col],
                                             table_stats.columns[col].min_val(),
                                             table_stats.columns[col].max_val()))
            feature.append(min_max_normalize(range_query.col_right[col],
                                             table_stats.columns[col].min_val(),
                                             table_stats.columns[col].max_val()))
    avi = stats.AVIEstimator.estimate(range_query, table_stats)
    ebo = stats.ExpBackoffEstimator.estimate(range_query, table_stats)
    minsel = stats.MinSelEstimator.estimate(range_query, table_stats)
    feature.extend([avi,ebo,minsel])
    return feature


def preprocess_queries(queris, table_stats, columns):
    """
    preprocess_queries turn queries into features and labels, which are used for regression model.
    """
    features, labels = [], []
    for item in queris:
        query, act_rows = item['query'], item['act_rows']
        feature, label = None, None
        # YOUR CODE HERE: transform (query, act_rows) to (feature, label)
        # Some functions like rq.ParsedRangeQuery.parse_range_query and extract_features_from_query may be helpful.
        q = rq.ParsedRangeQuery.parse_range_query(query)
        feature = extract_features_from_query(q,table_stats,columns) 
        label = np.log(act_rows)
        features.append(feature)
        labels.append([label])
    # return features, labels
    return np.array(features,np.float32), np.array(labels,np.float32)
    


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, queries, table_stats, columns):
        super().__init__()
        # self.query_data = list(zip(preprocess_queries(queries, table_stats, columns)))
        self.features,self.labels = preprocess_queries(queries, table_stats, columns)

    def __getitem__(self, index):
        # return self.query_data[index]
        return self.features[index],self.labels[index]


    def __len__(self):
        # return len(self.query_data)
        return len(self.features)


def est_mlp(train_data, test_data, table_stats, columns):
    """
    est_mlp uses MLP to produce estimated rows for train_data and test_data
    """
    train_dataset = QueryDataset(train_data, table_stats, columns)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    train_est_rows, train_act_rows = [], []
    # YOUR CODE HERE: train procedure
    net = MLP()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    criterion = torch.nn.MSELoss()
    EPOCHS = 20
    for epoch in range(EPOCHS):
        running_loss = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            (inputs,labels) = data
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("epoch {} :".format(epoch),running_loss/100)


    test_dataset = QueryDataset(test_data, table_stats, columns)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_est_rows, test_act_rows = [], []
    # YOUR CODE HERE: test procedure
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            (inputs,labels) = data
            outputs = net(inputs)
            train_act_rows.extend(labels.numpy().tolist())
            train_est_rows.extend(outputs.numpy().tolist())
        
        for i, data in enumerate(test_loader):
            (inputs,labels) = data
            outputs = net(inputs)
            test_act_rows.extend(labels.numpy().tolist())
            test_est_rows.extend(outputs.numpy().tolist())
    train_est_rows = np.exp(np.array(train_est_rows))
    train_act_rows = np.exp(np.array(train_act_rows))
    test_est_rows = np.exp(np.array(test_est_rows))
    test_act_rows = np.exp(np.array(test_act_rows))

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def est_xgb(train_data, test_data, table_stats, columns):
    """
    est_xgb uses xgboost to produce estimated rows for train_data and test_data
    """
    print("estimate row counts by xgboost")
    train_x, train_y = preprocess_queries(train_data, table_stats, columns)
    train_est_rows, train_act_rows = [], []
    # YOUR CODE HERE: train procedure
    dtrain = xgb.DMatrix(train_x, label=train_y)
    num_round = 10
    param = {'objective': 'reg:squarederror'}
    bst = xgb.train(param, dtrain, num_round)
    ypred = bst.predict(dtrain)
    train_est_rows = np.exp(np.array(ypred))
    train_act_rows = np.exp(train_y)
    
    test_x, test_y = preprocess_queries(test_data, table_stats, columns)
    test_est_rows, test_act_rows = [], []
    # YOUR CODE HERE: test procedure
    dtest = xgb.DMatrix(test_x)
    ypred = bst.predict(dtest)
    test_est_rows = np.exp(np.array(ypred))
    test_act_rows = np.exp(test_y)

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def eval_model(model, train_data, test_data, table_stats, columns):
    if model == 'mlp':
        est_fn = est_mlp
    else:
        est_fn = est_xgb

    train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(train_data, test_data, table_stats, columns)

    name = f'{model}_train_{len(train_data)}'
    eval_utils.draw_act_est_figure(name, train_act_rows, train_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(train_act_rows, train_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

    name = f'{model}_test_{len(test_data)}'
    eval_utils.draw_act_est_figure(name, test_act_rows, test_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(test_act_rows, test_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')


if __name__ == '__main__':
    stats_json_file = './data/title_stats.json'
    train_json_file = './data/query_train_2000.json'
    test_json_file = './data/query_test_5000.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    eval_model('mlp', train_data, test_data, table_stats, columns)
    eval_model('xgb', train_data, test_data, table_stats, columns)
