# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Imports from this application
from app import app

# 1 column layout. 1st column width = 6/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout

column1 = dbc.Col(
        [
            #html.Img(src="../data/model.png"),
            dcc.Markdown(
                """
                
                ## How it works

                The plots were generated by estimating the parameters of a simple extended [SIR model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model_is_dynamic_in_three_senses) using MCMC (Monte Carlo Markov Chain) sampling following 
                the method by [Dehning et. al. (2020)](https://arxiv.org/abs/2004.01105). Estimations
                are done for Qatar by looking at daily confirmed cases. 
                The SIR model is extended to account for time delay of reporting, weekend effects and, change in infection transmission rates through government interventions and other major events.
                Although a significant part of the infected people do not get tested or reported because say, they are asymptomatic or only exhibit mild symptoms, this must be
                modelled because the number of daily cases put out by MoPH only account for those people that are tested and confirmed positive (see [Rizzo et. al. (2020)](https://covid-research.qcri.org/seir/SEIR_Autograd.pdf)).  

                The Reproductive number is estimated using the method proposed by [Bettencourt & Ribeiro (2008)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0002185)
                and implementation by [Kevin Systrom](https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb). 

                The dataset used comes from the [Qatar Open Data Portal](https://www.data.gov.qa/explore/dataset/covid-19-cases-in-qatar/information/).
                We are constantly working on improving the model with additional information and assumptions. Meanwhile, as the model produces coarse estimations based on a very simplified model and noisy response data, i.e., daily confirmed cases, it should be interpreted with the confidence intervals and there
                is also no guarantee the estimations will materialize.      

                ## Updates
                - 17/5/2020 Reproductive rate estimates has been consistently showing a downward trend. Added several event markers including reopening measures. 
                - 21/5/2020 Added realtime estimation for reproductive number. 
                - 18/5/2020 Added changepoint for the 17th May ruling on masks being made mandatory.
                - 16/5/2020 Added a tested compartment in the SIR model. It was found that untested infected people make a difference to the SIR model.
                - 15/5/2020 Changed data source from [John Hopkins University repo](https://github.com/CSSEGISandData/COVID-19) to 
                [Qatar Open Data Portal](https://www.data.gov.qa/explore/dataset/covid-19-cases-in-qatar/information/). The pages here will be
                updated with new data every noon and models will be updated after a small delay.
                ## Contributors

                Shehel Yoosuf  
                Ahmed Aziz  
                
                For more info or feedback, email: shyoosuf@hbku.edu.qa

                """
            ),
            html.Br(), 
        ],
        md=6)



layout = dbc.Row([column1], className='mt-4', justify="center")