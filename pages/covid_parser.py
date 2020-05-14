# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import urllib.request as request
import json
import pandas as pd
from datetime import datetime
import os
import time
# Imports from this application
# from app import app

#Parser written to get covid data from data.gov.qa

def FileCheck(fn):
    try:
      open(fn, "r")
      return True
    except IOError:
      print ("Error: File does not appear to exist.")
      return False

def get_covid_metrics_model():
    #Page to start
    start = 0
    #Number of row results to call from api
    numRows = 200
    api_link = "https://www.data.gov.qa/api/records/1.0/search/?dataset=covid-19-cases-in-qatar&q=&rows="+str(numRows)+"&sort=-date&facet=date&start="+str(start)
    current_time = datetime.today().strftime('%Y_%m_%d')
    #file_name= "data/covid_data/covid_data_"+current_time+".csv"
    file_name = "data/covid_data/covid_data.csv"
    call_api =  True
    tries = 0

    while (call_api):
        with request.urlopen(api_link) as response:
            if response.getcode() == 200:
                #Read HTTP response
                source = response.read()
                #Load as json
                data = json.loads(source)
                print('API called:',start,"NumRows:",numRows)
                try:
                    #print(data.keys())
                    records = data['records'] 
                    numberOfResults = data['nhits']
                    print("Number of records retrieved: ",len(records))
                    if len(records) > 0:
                        #Fields to save
                        date = [ x['fields']['date'] for x in records ]
                        new_cases = [x['fields']['number_of_new_positive_cases_in_last_24_hrs'] for x in records]

                        #Make new
                        df_new = pd.DataFrame(list(zip(date, new_cases)), columns =['date', 'new_cases']) 
                        df=""
                        #Read old
                        if(FileCheck(file_name)):
                            df_exist = pd.read_csv(file_name) 
                            #Write only unique
                            if(df_exist.empty == False):
                                df = pd.concat([df_new, df_exist]).drop_duplicates('date').reset_index(drop=True)
                            else:
                                df = df_new
                        else:
                            df = df_new
                        
                        print("Writing to file")
                        df.to_csv(file_name, index=None, mode='w+', encoding='utf-8')
                        if(numberOfResults>start+numRows):
                            start = start + numRows
                        else:
                            #parsing finished
                            call_api = False
                            break

                    else:
                        call_api = False
                        break

                except:
                    tries = tries + 1
                    time.sleep(5)
                    print("Error occurred...Trying again...")
                    if(tries>3):
                        #remove data
                        call_api = False
                        os.remove(file_name)
                        print('An error occurred while attempting to retrieve data from the API.')

            else:
                print('An error occurred while attempting to retrieve data from the API.')

get_covid_metrics_model()

# 1 column layout
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
# column1 = dbc.Col(
#     [
#         dcc.Markdown(
#             """
        
#             ## API call done!


#             """
#         ),

#     ],
# )

# layout = dbc.Row([column1])