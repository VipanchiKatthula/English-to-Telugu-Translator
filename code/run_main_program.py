#!/usr/bin/env python
# coding: utf-8

# In[14]:


import requests
import json
import urllib.parse
data_json = {"eng":"We will get full time job soon"}   #, "text":["I am well"]}
head= {"Content-type": "application/json"}
response = requests.post('http://127.0.0.1:5000/predict', json = data_json, headers = head)
result = response.content.decode('utf8')
print(result)                         
#result = response.content.decode("utf8")

