#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import torch
import time


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.get_device_name()


# In[3]:


pip install sodapy


# In[4]:


from sodapy import Socrata


# In[19]:


client = Socrata("data.cdc.gov", None, timeout=999)
limit = 10000000  # Chunk size
offset = 0  
data_accumulated = [] 
tic = time.perf_counter()
while True:
    results = client.get("vbim-akqf", limit=limit, offset=offset)
    data_accumulated.append(pd.DataFrame.from_records(results).dropna())
    offset += limit
    if len(results) < limit:
        break
toc = time.perf_counter()
results_df = pd.concat(data_accumulated, ignore_index=True)
print(f"Downloaded the data in {toc - tic:0.4f} seconds")


# In[20]:


len(results_df)


# In[21]:


results_df.head()


# In[22]:


def drop_rows_missing_data(df, columns_to_check):
    for column in columns_to_check:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

    df_filtered = df.copy()
    for column in columns_to_check:
        #df_filtered = df_filtered[df_filtered[column].isin(['Yes', 'No'])]
        df_filtered = df_filtered[~df_filtered[column].isin(['Unknown', 'Missing'])]
    
    return df_filtered

columns_to_check = ['hosp_yn', 'icu_yn', 'death_yn', 'medcond_yn', 'race_ethnicity_combined', 'age_group', 'sex']
df_filtered = drop_rows_missing_data(results_df, columns_to_check)


# In[23]:


len(df_filtered)


# In[24]:


df_filtered.head()


# In[25]:


df_filtered.to_pickle("./cdcdata.pkl")

