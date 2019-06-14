
# coding: utf-8

# In[1]:


import json
import pickle
import operator
from collections import Counter
import pandas as pd
import numpy as np
from recsys import * ## recommender system cookbook
from preprocessing import * ## pre-processing code
from IPython.display import HTML ## Setting display options for Ipython Notebook
from pandas.io.json import json_normalize
from statistics import mean
import matplotlib.pyplot as plt


# In[2]:


aussie_items = json.load(open('./SteamData/australian_users_items_fixed.json','r'))
steam_games = json.load(open('./SteamData/steam_games_fixed.json','r'))


# In[3]:


with open('SteamData/australian_users_items_fixed.json') as f:
    df_items = json.load(f)    

parsed_items = json_normalize(data=df_items, record_path='items', meta=['items_count','steam_id','user_id','user_url'])


# ## Total hours of playtime for each game greater than 10

# In[4]:


min_playtime=[0,1,10,100,1000,10000]
all_ave_dist_bpr=[]
all_ave_dist_warp=[]

for j in range(len(min_playtime)):

    aussie_items = parsed_items[parsed_items.playtime_forever>min_playtime[j]]

    
    # In[5]:
    
    
    def build_df(user_item):
        df = user_item[['user_id','item_name']]
        df = df.rename(columns={"user_id": "user", "item_name": "item"})
        df = df.drop_duplicates(['user','item'])
        df['own'] = 1
        df = df.pivot(index='user',columns='item',values='own')
        df = df.fillna(0)
        return df
    
    
    # In[6]:
    
    
    user_item_df = build_df(aussie_items)
    # user_item_df
    
    
    # ## Matrix Factorization
    
    # In[7]:
    
    
    top500 = user_item_df.sum().nlargest(500)
    
    top500games = top500.index
    
    user_top500games = user_item_df[top500games].stack().reset_index()
    user_top500games = user_top500games.rename(columns={0:'rating'})
    
    
    # ## User of Top 500 Games
    
    # In[8]:
    
    
    # user_top500games
    
    
    # In[9]:
    
    
    games=pd.DataFrame()
    games['item']=user_top500games.item.drop_duplicates()
    
    
    # ## 500 Games
    
    # In[10]:
    
    
    games.head()
    
    
    # In[11]:
    
    
    from sklearn.model_selection import train_test_split
    
    train, test = train_test_split(user_top500games, test_size=0.2)
    
    
    # In[12]:
    
    
    train.head()
    
    
    # In[13]:
    
    
    # test.item.drop_duplicates()
    
    
    # In[14]:
    
    
    interactions_train = create_interaction_matrix(df = train,
                                             user_col = 'user',
                                             item_col = 'item',
                                             rating_col = 'rating',
                                             threshold = '1')
    interactions_train.shape
    
    
    # ## Matrix Factorization of 500 Games
    
    # In[15]:
    
    
    interactions_train.head()
    
    
    # In[16]:
    
    
    interactions_test = create_interaction_matrix(df = test,
                                             user_col = 'user',
                                             item_col = 'item',
                                             rating_col = 'rating',
                                             threshold = '1')
    interactions_test.shape
    
    
    # In[17]:
    
    
    interactions_test.head()
    
    
    # In[18]:
    
    
    user_dict = create_user_dict(interactions=interactions_train)
    len(user_dict)
    
    
    # In[19]:
    
    
    user_dict_test = create_user_dict(interactions=interactions_test)
    len(user_dict_test)
    
    
    # In[20]:
    
    
    games_dict = create_item_dict(df = games,
                                   id_col = 'item',
                                   name_col = 'item')
    len(games_dict)
    
    
    # ## BPR Model
    
    # In[21]:
    
    
    mf_model_bpr = runMF(interactions = interactions_train,
                     n_components = 30,loss = 'bpr',k = 15,epoch = 30,n_jobs = 4)
    
    
    # ## WARP Model
    
    # In[22]:
    
    
    mf_model_warp = runMF(interactions = interactions_train,
                     n_components = 30,loss = 'warp',k = 15,epoch = 30, n_jobs = 4)
    
    
    # ## Item - Item Recommender (Using BPR)
    
    # In[23]:
    
    
    item_item_dist = create_item_emdedding_distance_matrix(model = mf_model_bpr,
                                                           interactions = interactions_test)
    
    
    # In[24]:
    
    
    rec_list_bpr = item_item_recommendation(item_emdedding_distance_matrix = item_item_dist,
                                        item_id = 'Counter-Strike',
                                        item_dict = games_dict,
                                        n_items = 20)
    
    
    # In[25]:
       
    
    # ## Item - Item Recommender (Using WARP)
    
    # In[26]:
    
    
    item_item_dist = create_item_emdedding_distance_matrix(model = mf_model_warp,
                                                           interactions = interactions_test)
#    
#    
#    # In[27]:
#    
#    
    rec_list_warp = item_item_recommendation(item_emdedding_distance_matrix = item_item_dist,
                                        item_id = 'Counter-Strike',
                                        item_dict = games_dict,
                                        n_items = 20)


# In[28]:


    ave_dist_bpr=mean(rec_list_bpr[1])
    all_ave_dist_bpr.append(ave_dist_bpr)
    
    ave_dist_warp=mean(rec_list_warp[1])
    all_ave_dist_warp.append(ave_dist_warp)
    
all_ave_dist_bpr
all_ave_dist_warp

plt.figure()
plt.semilogx(min_playtime,all_ave_dist_bpr,'o')
plt.semilogx(min_playtime,all_ave_dist_warp,'o')
plt.legend(['BPR','WARP'])
plt.grid()
plt.xlabel('Min Logarithmic Hours Played')
plt.ylabel('Average Similarity Distance ')
plt.title('Average Similarity Distance of Games for Min Logarithmic Hours Played')
