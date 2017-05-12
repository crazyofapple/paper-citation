#-*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import xgboost as xgb
from gen_feat1 import get_train
from gen_feat1 import get_test
from sklearn.cross_validation import KFold
import pandas as pd


def xgboost_make_submission():
        #user_index, training_data, label = get_train()
        user_index = pd.read_csv('./cache/user_index.csv',sep='\t',header=None,names=['nu','id','name'])
        del user_index['nu']
        training_data  = pd.read_csv('./cache/training_data.csv',sep='\t')
        del training_data['Unnamed: 0']
        label = pd.read_csv('./cache/label.csv',sep='\t',header=None,names=['nu','result'])
        del label['nu']
        #sub_user_index, sub_trainning_data = get_test()
        sub_user_index = pd.read_csv('./cache/sub_user_index.csv',sep='\t')
        del sub_user_index['Unnamed: 0']
        sub_trainning_data = pd.read_csv('./cache/sub_trainning_data.csv',sep='\t')
        del sub_trainning_data['Unnamed: 0']
        sub_user_index['label'] = 0
	X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.2, random_state=0)
	dtrain=xgb.DMatrix(X_train.values, label=y_train)
	dtest=xgb.DMatrix(X_test.values, label=y_test)
	param = {'booster':'gbtree','learning_rate' : 0.0155, 'n_estimators': 1000, 'max_depth': 7,'min_child_weight': 6, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'reg:linear','lambda':0.0015}
	num_round = 10000
        param['nthread'] = 20
        param['eval_metric'] = "rmse"
	plst = param.items()
	plst += [('eval_metric', 'rmse')]
	evallist = [(dtest, 'eval'), (dtrain, 'train')]
	bst=xgb.train( plst, dtrain, num_round, evallist)
	#sub_user_index, sub_trainning_data = get_test()
	sub_trainning_data1 = xgb.DMatrix(sub_trainning_data.values)
	y = bst.predict(sub_trainning_data1)
	sub_user_index['label'] =   y

        sub_user_index.columns=['AuthorID','User','Citation_number']
        resu = sub_user_index[['AuthorID','Citation_number']]
        #df= pd.DataFrame([['','']],columns=['AuthorID','Citation_number'])
        #resu = resu.append(df,ignore_index=True)
        resu.to_csv('./data/results4.txt', sep='\t',index=False, index_label=False,line_terminator='\r',encoding='utf-8',header=None)

	
if __name__ == '__main__':
	xgboost_make_submission()	

