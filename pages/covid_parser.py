# %%
#  Imports from 3rd party libraries
import urllib.request as request
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime
import os
import time
import datetime


from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from selenium import webdriver

#Parser written to scrape data from moph and add it to dataset
# %%
def FileCheck(fn):
    try:
      open(fn, "r")
      return True
    except IOError:
      print ("Error: File does not appear to exist.")
      return False

#Parser to get latest metrics as soon as they are published
def get_covid_metrics_parser():
  user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
  headers={'User-Agent':user_agent,} 
  url = "https://www.moph.gov.qa/english/Pages/Coronavirus2019.aspx"
  required_elem = {'Total Number of People Tested':0, }
  #get already saved data
  df = pd.read_csv('data/covid_data.csv', sep=',')
  df_max_date = datetime.strptime(df.Date.max(), '%Y-%m-%d')
  covid_cases={}
  # covid_cases['new_cases']         = "" 
  # covid_cases['new_deaths']        = ""
  # covid_cases['new_recovered']     = ""
  # covid_cases['recovered_cases']   = ""
  # covid_cases['active_cases']      = ""
  # covid_cases['death_cases']       = ""
  # covid_cases['total_tested']      = ""
  # covid_cases['total_positive']    = ""
  # covid_cases['icu_cases']         = ""
  # covid_cases['acute_care_cases']  = "" 


  #Open page and parse html
  my_request=request.Request(url,None,headers) #The assembled request
  response = request.urlopen(my_request)
  if response.getcode() == 200:
      #Read HTTP response
      source = response.read()
      soup = BeautifulSoup(source, 'html.parser')
      #Get latest update date, see if it in our records 
      date_elem = soup.find('div', class_='col-7 col-lg-10 text-left')
      date_elem = date_elem.text.replace(u'\xa0', u' ')
      date_elem = date_elem.split(" ")[1:]
      date_elem = " ".join(date_elem).strip()
      datetime_object = datetime.strptime(date_elem, '%d %b %Y')
      #Compare if update_date is greater than stored_date
      if (datetime_object > df_max_date):
        #Get stats elements
        stats_elems = soup.find_all('a', class_='p-3 d-block black')
        stat_number = 0
        for stats_elem in stats_elems:
          stats_elem = stats_elem.text.strip().split('\n')
          for elem in stats_elem:
            if (elem.isdigit()):
              stat_number = elem
            else:
              covid_cases[elem] = stat_number
              # if (elem == "Total Number of People Tested"):
              #   covid_cases['total_tested']      = stat_number
              # elif (elem == "Total Number of Positive Cases"):
              #   covid_cases['total_positive']      = stat_number
              # elif (elem == "Number of Active​ Cases Under Treatment"):
              #   covid_cases['active_cases']      = stat_number
              # elif (elem == "Number of New Cases in the Last 24 Hours"):
              #   covid_cases['new_cases']      = stat_number
              # elif (elem == "Currently under acute  Hospital care"):
              #   covid_cases['acute_care_cases']      = stat_number
              # elif (elem == "Currently in ICU"):
              #   covid_cases['icu_cases']      = stat_number
              # elif (elem == "Recovered Patients"):
              #   covid_cases['recovered_cases']      = stat_number
              # elif (elem == "Deaths"):
              #   covid_cases['death_cases']      = stat_number
            
        #print(covid_cases)
        #Fields to save
        cols =['Date', 'Number of New Positive Cases in Last 24 Hrs', 'Total Number of Positive Cases to Date',
                'Number of New Tests in Last 24 Hrs','Total Number of Tests to Date',
                'Total Number of Active Cases Undergoing Treatment to Date','Number of New Recovered Cases in Last 24 Hrs',
                'Total Number of Recovered Cases to Date','Number of New Deaths in Last 24 Hrs',
                'Total Number of Deaths to Date']
  
        # df2 = pd.DataFrame(list(zip('', covid_cases['new_cases'],covid_cases['total_positive'],
        #       "NA", covid_cases['total_tested'] , covid_cases['active_cases'], "NA", covid_cases['recovered_cases'], 
        #       "NA", covid_cases['death_cases'] 
        #   )), columns = cols) 
        df2 = pd.DataFrame([covid_cases])
        datetime_object = datetime.strftime(datetime_object, '%Y-%m-%d')
        df2['Date'] = datetime_object
        df2['Number of New Tests in Last 24 Hrs'] = ''
        df2['Number of New Recovered Cases in Last 24 Hrs'] = ''
        df2['Number of New Deaths in Last 24 Hrs'] = ''
        print(df2)

        df = df.append(df2)
        print(df)
        df.to_csv('data/covid_data.csv', index=None, encoding='utf-8')


 


#Parser written to get covid data from data.gov.qa
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
      req = ('https://covid19.moph.gov.qa/EN/Pages/default.aspx')
      # webpage = urlopen(req).read()

      new_stats = []

      driver = webdriver.PhantomJS()
      driver.get(req)
      new_stats.append(int(driver.find_element_by_id(id_='strgPeopleTested').text))
      new_stats.append(int(driver.find_element_by_id(id_='strgPositiveCases').text))
      new_stats.append(int(driver.find_element_by_id(id_='strgActiveCases').text))
      new_stats.append(int(driver.find_element_by_id(id_='strgPositiveCases24Hrs').text))
      new_stats.append(int(driver.find_element_by_id(id_='strgPeopleAcuteCare').text))
      new_stats.append(int(driver.find_element_by_id(id_='strgPeopleinICU').text))
      new_stats.append(int(driver.find_element_by_id(id_='strgRecoveredPatients').text))
      new_stats.append(int(driver.find_element_by_id(id_='strgDeaths').text))
      print(new_stats)
     # soup = BeautifulSoup(webpage)
      # for i in soup.find_all("h3", class_="my-2"):
      #   try:
      #       new_stats.append(int(i.find("strong").string))
      #   except: 
      #       new_stats.append(int(i.find("b").string))


      
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
      


get_covid_metrics_parser()


  # %%
