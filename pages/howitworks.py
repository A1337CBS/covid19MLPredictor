# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Imports from this application
from app import app

# 2 rows layout.
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
row1 = dbc.Row(
    [
        dcc.Markdown(
            """
        
            ## How it works

            The plots were generated by estimating the parameters of a simple SIR model using MCMC (Monte Carlo Markov Chain) sampling following 
            the method by [Dehning et. al. (2020)](https://arxiv.org/abs/2004.01105). The estimations
            are done for the country of Qatar by looking at daily confirmed cases. 
            The SIR model is extended to account for time delay of reporting, weekend effects and, change in infection transmission rates through government interventions and other major events.
            
            The data we use comes from the John Hopkins University public repository and the numbers may vary slightly from the numbers put out by MoPH (Ministry of 
            Public Health). 
            We are constantly working on improving the model with additional information and assumptions. Meanwhile, as the model produces coarse estimations based on a very simplified model and noisy response data, i.e., daily confirmed cases, it should be interpreted with the confidence intervals and there
            is also no guarantee the estimations will materialize.      

            """
        ),
        html.Br(), 
    ],

)

row2 = dbc.Row(
    [
            dcc.Markdown(
            """
        
            ## Contributors

            Shehel Yoosuf  
            Ahmed Aziz  
              
            For more info or feedback, email: shyoosuf@hbku.edu.qa
            """
        ),


    ],
)

layout = dbc.Col([row1, row2], className='mt-4', width={"size": 4, "offset": 4},md=4, align="center",)