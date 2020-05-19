# %%
#  Imports from 3rd party libraries
import urllib.request as request
import json
import pandas as pd
from datetime import datetime
import os
import time
import datetime

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

#Parser written to scrape data from moph and add it to dataset
# %%
def FileCheck(fn):
    try:
      open(fn, "r")
      return True
    except IOError:
      print ("Error: File does not appear to exist.")
      return False

# %% 
'''
The order of retrieval of data from MoPH is 
Total tested, Total positive,
Number of active, cases in last 24,
currently under acute care, currently in icu,
recovered patients, deaths. This is the order 
data is stored in new_stats
'''
def get_covid_metrics_model():

  if not FileCheck("../data/covid_data.csv"):
    request.urlretrieve("https://www.data.gov.qa/explore/dataset/covid-19-cases-in-qatar/download/?format=csv&timezone=Asia/Baghdad&lang=en&use_labels_for_header=true&csv_separator=%2C", "../data/covid_data.csv")
    df = pd.read_csv('./data/covid_data.csv', sep=',')
    df = df.fillna(0)
    for col in df.columns[1:]: 
        df[col]=df[col].astype(int)
    df['Date'] =pd.to_datetime(df.Date)
    df = df.sort_values(by='Date')
    df = df.iloc[3:]
    df.to_csv('./data/covid_data.csv', index=None)
  else:
    print ("Exists")
    
    df = pd.read_csv('./data/covid_data.csv')
    prev = list(df.iloc[-1])
    repeat = True
    while (repeat):
      # Scrape new stats from moph
      req = Request('https://www.moph.gov.qa/english/Pages/Coronavirus2019.aspx', headers={'User-Agent': 'Mozilla/5.0'})
      webpage = urlopen(req).read()
      soup = BeautifulSoup(webpage)
      new_stats = []
      for i in soup.find_all("h3", class_="my-2"):
        try:
            new_stats.append(int(i.find("strong").string))
        except: 
            new_stats.append(int(i.find("b").string))
      print (new_stats)
      
      # If new entry added, 
      if prev[4] == new_stats[0]:
        # Wait 1 minutes before trying again
        time.sleep(60)
        continue    
      else:
        current_date = datetime.datetime.strptime(prev[0], "%Y-%m-%d")
        new_date = current_date + datetime.timedelta(days=1)
        
        # Append new entries
        df.loc[len(df)] = [new_date.strftime("%Y-%m-%d"),new_stats[3],new_stats[1],
                          new_stats[0]-prev[4], new_stats[0], new_stats[2], 
                          new_stats[6]-prev[7], new_stats[6], new_stats[-1]-prev[-1], 
                          new_stats[-1]]
        print ("Dataframe appended and written to disk.")
        df.to_csv('./data/covid_data.csv', index=None)
        repeat = False
      


get_covid_metrics_model()


  # %%
