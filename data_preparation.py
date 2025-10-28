import numpy as np
import pandas as pd
import pandas as pd

save_path = "/content/datasets/ml-1m/"
train_ = pd.read_pickle(save_path+"train_ood2.pkl")
valid_ = pd.read_pickle(save_path+"valid_ood2.pkl")
test_ = pd.read_pickle(save_path+"test_ood2.pkl")

user_info = train_.groupby('uid').agg({"label":'count'})
user_info.shape
train_user = np.array(list(user_info.index))
user_info.quantile(0.2)
user_info = user_info[user_info.label>3]
user_info.shape

item_info = train_.groupby('iid').agg({"label":'count'})
item_info.shape
train_item = np.array(list(item_info.index))
item_info.describe()
item_info.quantile(0.3),item_info.shape
item_info = item_info[item_info.label>3]
item_info.shape

warm_user = np.array(list(user_info.index))
warm_item = np.array(list(item_info.index))

test_['warm'] = test_[['uid','iid']].apply(lambda x: x.uid in warm_user and x.iid in warm_item, axis=1).astype("int")
test_['cold'] = test_[['uid','iid']].apply(lambda x: x.uid not in train_user and x.iid not in train_item, axis=1).astype("int")
test_.shape
test_.head()
test_[['not_cold','warm','cold']].describe()
test_.to_pickle(save_path+"test_warm_cold_ood2.pkl")
save_path+"test_warm_cold_ood2.pkl"