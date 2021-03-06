# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Imports from this application
from app import app

######################################
############FUTURE PAGE?##############
######################################

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
            dcc.Markdown(
            """
        
            ## Updates

            Shehel Yoosuf  
            Ahmed Aziz  
              
            For more info or feedback, email: shyoosuf@hbku.edu.qa
            """
        ),
    ],
    md=4,
)
column2 = dbc.Row([ 
        dbc.Col(
        [
            dcc.Markdown(
                """
            
                ## How it works

                The plots were generated by estimating the parameters of a simple SIR model using MCMC (Monte Carlo Markov Chain) sampling following 
                the method by [Dehning et. al. (2020)](https://arxiv.org/abs/2004.01105). The estimations
                are done for the country of Qatar by looking at daily confirmed cases. 
                The SIR model is extended to account for time delay of reporting, weekend effects and, change in infection transmission rates through government interventions and other major events.
                
                The dataset used comes from the Qatar's Open data portal for public [repository](https://www.data.gov.qa/explore/dataset/covid-19-cases-in-qatar/information/).
                We are constantly working on improving the model with additional information and assumptions. Meanwhile, as the model produces coarse estimations based on a very simplified model and noisy response data, i.e., daily confirmed cases, it should be interpreted with the confidence intervals and there
                is also no guarantee the estimations will materialize.      

                """
            ),
            html.Br(), 
        ],
        md=4,
        )
    ], justify="center")

layout = dbc.Row([column1, column2], className='mt-4', justify="center",)