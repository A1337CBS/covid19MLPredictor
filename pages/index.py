# Imports from 3rd party libraries
import pdb
import dash
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time

import urllib.request as request
import json
from matplotlib import pyplot as plt
import matplotlib
import pickle
import datetime

from covid19_inference.data_retrieval import *
import dash_table


# Imports from this application
from app import app

######################################
#############FUNCTIONS################
######################################

#Get Qatar related covid data
def get_covid_metrics(df):
    try:
        covid_cases={}
        covid_cases['new_cases']       = df['Number of New Positive Cases in Last 24 Hrs'].iloc[-1]
        covid_cases['new_deaths']       = df['Number of New Deaths in Last 24 Hrs'].iloc[-1]
        covid_cases['new_recovered']       =df['Number of New Recovered Cases in Last 24 Hrs'].iloc[-1]
        covid_cases['recovered_cases'] =df['Total Number of Recovered Cases to Date'].iloc[-1]
        covid_cases['active_cases']    =df['Total Number of Active Cases Undergoing Treatment to Date'].iloc[-1]
        covid_cases['death_cases']     =     df['Total Number of Deaths to Date'].iloc[-1]
        return covid_cases
    except:
        return {"new_cases":"NA",
                "recovered_cases":"NA",
                "active_cases":"NA",
                "death_cases":"NA"}

#Main function for plotting graphs
def plot_cases(
    trace,
    df,
    date_begin_sim,
    diff_data_sim,
    start_date_plot=None,
    end_date_plot=None,
    ylim=None,
    week_interval=None,
    colors=("tab:blue", "tab:orange"),
    country="Qatar",
    second_graph=False,
):
    """
        Plots the new cases, the fit, forecast and lambda_t evolution

        Parameters
        ----------
        trace : trace returned by model
        new_cases_obs : array
        date_begin_sim : datetime.datetime
        diff_data_sim : float
            Difference in days between the begin of the simulation and the data
        start_date_plot : datetime.datetime
        end_date_plot : datetime.datetime
        ylim : float
            the maximal y value to be plotted
        week_interval : int
            the interval in weeks of the y ticks
        colors : list with 2 colornames

        Returns
        -------
        figure, axes
    """
    def conv_time_to_mpl_dates(arr):
        return matplotlib.dates.date2num(
            [datetime.timedelta(days=float(date)) + date_begin_sim for date in arr]
    )
    new_cases_obs = df["Number of New Positive Cases in Last 24 Hrs"].values
    new_recovered = df["Number of New Recovered Cases in Last 24 Hrs"].values
    new_cases_sim = trace
    len_sim = trace.shape[1] + diff_data_sim
    if start_date_plot is None:
        start_date_plot = date_begin_sim + datetime.timedelta(days=diff_data_sim)
    if end_date_plot is None:
        end_date_plot = date_begin_sim + datetime.timedelta(days=len_sim)
    if ylim is None:
        ylim = 1.6 * np.max(new_cases_obs)

    num_days_data = len(new_cases_obs)
    diff_to_0 = num_days_data + diff_data_sim
    date_data_end = date_begin_sim + datetime.timedelta(
        days=diff_data_sim + num_days_data
    )
    num_days_future = (end_date_plot - date_data_end).days

    start_date_mpl, end_date_mpl = matplotlib.dates.date2num(
        [start_date_plot, end_date_plot]
    )

    if week_interval is None:
        week_inter_left = int(np.ceil(num_days_data / 7 / 5))
        week_inter_right = int(np.ceil((end_date_mpl - start_date_mpl) / 7 / 6))
    else:
        week_inter_left = week_interval
        week_inter_right = week_interval

    fig = make_subplots(rows=1, cols=1)
    

    time1 = np.arange(-len(new_cases_obs), 0)
    mpl_dates = conv_time_to_mpl_dates(time1) + diff_data_sim + num_days_data
    mpl_dates = matplotlib.dates.num2date(mpl_dates)

    new_cases_past = new_cases_sim[:, :num_days_data]

    
    # 95% CI
    if(second_graph == False):
        percentiles = (
            np.percentile(new_cases_past, q=2.5, axis=0),
            np.percentile(new_cases_past, q=97.5, axis=0),
        )
        
        fig.add_trace(go.Scatter(x=mpl_dates, y=percentiles[0],  fill='none', line=dict(width=0.8, color='#ffdaba'), fillcolor='#ffdaba',
                        mode= 'lines', opacity=0.2, showlegend=False))
        fig.add_trace(go.Scatter(x=mpl_dates, y=percentiles[1],  fill='tonextx', line=dict(width=0.8, color='#ffdaba'), fillcolor = '#ffdaba',
                        mode= 'lines', opacity=0.2, showlegend=False))
        fig.add_trace(
            go.Scatter(x=mpl_dates, y=np.median(new_cases_past, axis=0), mode='lines', line=dict(dash="dashdot", color="#f21146"), name='Fit with 95% CI') 
        )
        fig.add_trace(go.Scatter(
            x=[mpl_dates[7], mpl_dates[15], mpl_dates[51], mpl_dates[54]],
            y=[2000, 2000, 2000, 2000],
            mode="markers",
            marker_symbol="cross-dot",
            marker_line_color="#6ab9f2",
            marker_color="#6af2e6",
            marker_size=12,
            marker_line_width=2,
            name="Events and Interventions",
            text=["School Shutdown", "Border Restrictions", "Ramadan", "Masks (Shopping)"],
            textposition="top center"
        ))
        
    time2 = np.arange(0, num_days_future)
    mpl_dates_fut = conv_time_to_mpl_dates(time2) + diff_data_sim + num_days_data
    cases_future = new_cases_sim[:, num_days_data : num_days_data + num_days_future].T
    median = np.median(cases_future, axis=-1)

    ## 75% CI
    percentiles = (
        np.percentile(cases_future, q=2.5, axis=-1),
        np.percentile(cases_future, q=97.5, axis=-1),
    )

    mpl_dates_fut = matplotlib.dates.num2date(mpl_dates_fut)

    if(second_graph == False):
        # 75% CI
        fig.add_trace(
            go.Scatter(x=mpl_dates_fut, y=percentiles[0], fill='none', line=dict(width=0.8, color='#ffe5ce'), fillcolor='#ffe5ce',
                        mode= 'lines',name="75%-top", opacity=0.2, showlegend=False)
        )
        fig.add_trace(
            go.Scatter(x=mpl_dates_fut, y=percentiles[1], fill='tonextx', line=dict(width=0.8, color='#ffe5ce'), fillcolor = '#ffe5ce',
                        mode= 'lines',name="75%-bottom", opacity=0.2, showlegend=False)
                        )


        # 95% CI
        percentiles = (
            np.percentile(cases_future, q=12.5, axis=-1),
            np.percentile(cases_future, q=87.5, axis=-1),
        )

        
        fig.add_trace(
            go.Scatter(x=mpl_dates_fut, y=percentiles[0], fill='none', line=dict(width=0.8, color='#ffdaba'), fillcolor='#ffdaba',
                        mode= 'lines', opacity=0.2, showlegend=False)
        )
        fig.add_trace(
            go.Scatter(x=mpl_dates_fut, y=percentiles[1], fill='tonextx', line=dict(width=0.8, color='#ffdaba'), fillcolor = '#ffdaba',
                        mode= 'lines', opacity=0.2, showlegend=False)
                        )
        fig.add_trace(
            go.Scatter(x=mpl_dates_fut, y=median, mode='lines', line=dict(color="#f21146"), name='Forecast with 75% and 95% CI') 
        )

        #Define X and Y Axes text add and range
        fig.update_layout(
        yaxis=dict(range=[0,3500]),
        xaxis=dict(range=[mpl_dates[0],mpl_dates_fut[-1] ]),
        template='ggplot2',
        xaxis_title="Date",
        yaxis_title="New confirmed cases in Qatar",
        #autosize=True,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=70,
            pad=4
        ),
        height=550,
        font=dict(
            #family="Courier New, monospace",
            size=12,
            color="#7f7f7f",
            ),
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
            )
        )
        

    #Plot growth rate Graph
    if(second_graph == True):
        filename = 'data/trace_lambda.pkl'
        infile = open(filename,'rb')
        trace_lambda_t = pickle.load(infile)
        infile.close()
        filename = 'data/trace_mu.pkl'
        infile = open(filename,'rb')
        trace_mu = pickle.load(infile)
        infile.close()
        
        time = np.arange(-diff_to_0, -diff_to_0 + len_sim)
        lambda_t = trace_lambda_t[:, :]
        μ = trace_mu[:, None]
        mpl_dates = conv_time_to_mpl_dates(time) + diff_data_sim + num_days_data


        mpl_dates = matplotlib.dates.num2date(mpl_dates)

        fig.add_trace(
            go.Scatter(x=mpl_dates, y=np.percentile(lambda_t - μ, q=2.5, axis=0), fill='none', line=dict(width=0.5, color='#ffdaba'), fillcolor='#ffdaba',
                        mode= 'lines', opacity=0.2, showlegend=False)
        )
        fig.add_trace(
            go.Scatter(x=mpl_dates, y=np.percentile(lambda_t - μ, q=97.5, axis=0), fill='tonextx', line=dict(width=0.5, color='#ffdaba'), fillcolor = '#ffdaba',
                        mode= 'lines', opacity=0.2, showlegend=False)
                        )
        fig.add_trace(
            go.Scatter(x=mpl_dates, y=np.median(lambda_t - μ, axis=0), mode='lines', line=dict(color='#f21146'), name='Effective Growth Rate')# $\lambda_t^*$')
        )

        maxVertY = np.max(np.percentile(lambda_t - μ, q=97.5, axis=0))
        minVertY = np.min(np.percentile(lambda_t - μ, q=2.5, axis=0))
        # Vertical Lines 
        fig.add_trace(
            go.Scatter(x=[mpl_dates[23],mpl_dates[23] ], y=[minVertY, maxVertY], mode='lines', 
            line=dict(dash="dashdot",color='#1b4703'), name='School Shutdown')
        )

        fig.add_trace(
            go.Scatter(x=[mpl_dates[31],mpl_dates[31] ], y=[minVertY, maxVertY], mode='lines', 
            line=dict(dash="dashdot",color='#097a14'), name='Border Shutdown/Contact Restriction')
        )

        fig.add_trace(
            go.Scatter(x=[mpl_dates[67],mpl_dates[67] ], y=[minVertY, maxVertY], mode='lines', 
            line=dict(dash="dashdot",color='#50bd07'), name='Ramadan')
        )

        fig.add_trace(
            go.Scatter(x=[mpl_dates[70],mpl_dates[70] ], y=[minVertY, maxVertY], mode='lines', 
            line=dict(dash="dashdot",color='#abbd07'), name='Masks Made Compulsory')
        )
        fig.add_trace(
            go.Scatter(x=[mpl_dates[0],mpl_dates[-1] ], y=[0, 0], mode='lines', 
            line=dict(dash="dot",color='black'), name='Critical Point')
        )
        
        fig.update_layout(
            template = 'ggplot2',
            xaxis_title="Date",
            yaxis_title="Effective Growth Rate",

            annotations=[
                dict(
                    x=10.7,
                    y=0.5,
                    showarrow=False,
                    text="Custom y-axis title",
                    textangle=0,
                    xref="paper",
                    yref="paper"
                )
            ],
            autosize=True,
            # margin=dict(
            #     b=100
            # ),
            # xaxis=dict(
            #     autorange=False,
            #     range=[-0.05674507980728292, -0.0527310420933204],
            #     type="linear"
            # ),
            # yaxis=dict(
            #     autorange=False,
            #     range=[1.2876210047544652, 1.2977732997811402],
            #     type="linear"
            # ),
            # height=550,
            # width=1137
        )
    #Stop Plotting growth rate Graph

    #True new_cases_observed Data trace
    if(second_graph == False):
        fig.add_trace(
        go.Scatter(x=mpl_dates, y=new_cases_obs, mode='markers', name='New Confirmed Cases')
        )
    if(second_graph == False):
        fig.add_trace(
        go.Scatter(x=mpl_dates, y=new_recovered, mode='markers', name='New Recovered Cases', line=dict(dash="dot",color='#75f26a'))
        )
    # if(second_graph == False):
    #     fig.add_trace(
    #     go.Scatter(x=mpl_dates, y=df["Number of New Tests in Last 24 Hrs"], mode='markers', name='Daily Test', line=dict(dash="dot",color='black'))
    #     )

    fig.update_yaxes(automargin=True)
    return fig
    

######################################
#############VARIABLES################
######################################


date_begin_data = datetime.datetime(2020,3,3)

result = pd.read_csv('data/realtime_rt.csv')
result = result.loc[5:]

filename = 'data/trace_new_cases.pkl'
infile = open(filename,'rb')
trace = pickle.load(infile)
infile.close()

df4Table = pd.read_csv('data/model_output.csv')
df = pd.read_csv('data/covid_data.csv')
covid_cases = get_covid_metrics(df)
df4Table = df4Table.iloc[-15:]

df4Table.replace(0, np.nan, inplace=True)
df4Table.columns = ['Date', 'Observed', 'Model']
diff_data_sim = 16 # should be significantly larger than the expected delay, in 
                   # order to always fit the same number of data points.
# begin date of observations
bd = date_begin_data

# begin date of simulations which is set to 16 days before 
# first observation by default
sim_bd = bd - datetime.timedelta(days=diff_data_sim)

fig = plot_cases(
    trace,
    df,
    date_begin_sim=sim_bd,
    diff_data_sim=16,
    start_date_plot=None,
    end_date_plot=None,
    ylim=1000,
    week_interval=2,
    colors=("tab:blue", "tab:orange"),
    country="Qatar",
)


fig_growth_rate = plot_cases(
    trace,
    df,
    date_begin_sim=sim_bd,
    diff_data_sim=16,
    start_date_plot=None,
    end_date_plot=None,
    ylim=1000,
    week_interval=2,
    colors=("tab:blue", "tab:orange"),
    country="Qatar",
    second_graph=True
)

def fig_reprod(result):

    fig = go.Figure()
    fig.add_trace(
                go.Scatter(x=result["date"], y=result["High_90"].values, fill='none', line=dict(width=0.8, color=' #D2D7D3'), fillcolor=' #D2D7D3',
                            mode= 'lines',name="75%-top", opacity=0.2, showlegend=False)
            )
    fig.add_trace(
                go.Scatter(x=result["date"], y=result["Low_90"].values, fill='tonextx', line=dict(width=0.8, color='#D2D7D3'), fillcolor = '#D2D7D3',
                            mode= 'lines',name="75%-bottom", opacity=0.2, showlegend=False))
    fig.add_trace(go.Scatter(
        x=result["date"],
        y=result["ML"].values,
        marker=dict(
            size=4, 
            cmax=3,
            cmin=0,
            color=result["ML"].values,
            colorbar=dict(
                title="R_t",
            ),
            colorscale="Reds"
        ),
        showlegend=False,
        mode="markers"))
    fig.add_trace(
            go.Scatter(x=[result["date"].iloc[0],result["date"].iloc[-1]], y=[1, 1], mode='lines', 
            line=dict(dash="dot",color='black'), name='Critical Point', showlegend=False)
        )
    fig.update_layout(
                template = 'ggplot2',
                xaxis_title="Date",
                yaxis_title="Reproductive Rate R_t")
    return fig
# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Jumbotron([ 
        dbc.Col(
                [
                dbc.Card(
                [
                        html.H5("New Case(s) Forecast", className="card-title"),
                        #html.P("1023"),
                        dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in df4Table.columns],
                        data=df4Table.to_dict('records'),
                        style_cell={'textAlign': 'left'},
                        style_header={
                                #'backgroundColor': 'white',
                                'fontWeight': 'bold'
                            },
                        ),
                        
                    ],
                    className="w-85 mb-3",
                    id="cross-filter-options",
                    body=True,
                    color="light",
                )
    ],
    )], 
    fluid=True,
    )



column2 = dbc.Col(
    [   
                            dbc.CardDeck(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                        [
                                        html.H5(covid_cases['new_cases'], id="newText", className='mr-2'),  
                                        html.P("New cases"),
                                        ])
                                    ],
                                    id="new",
                                    className="mini_container",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                        [
                                        html.H5(covid_cases['new_recovered'], id="newText", className='mr-2'),  
                                        html.P("New Recovered"),
                                        ])
                                    ],
                                    id="new",
                                    className="mini_container",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                        [
                                        html.H5(covid_cases['new_deaths'], id="newText", className='mr-2'),  
                                        html.P("New Deaths"),
                                        ])
                                    ],
                                    id="new",
                                    className="mini_container",
                                ),

                                dbc.Card(
                                    [   
                                        dbc.CardBody(
                                        [
                                        html.H5(covid_cases['recovered_cases'], id="recoveredText", className='mr-2'),  
                                        html.P("Total Recovered"),
                                        ])
                                        
                                        ],
                                    id="recovered",
                                    className="mini_container",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                        [
                                        html.H5(covid_cases['active_cases'], id="activeText", className='mr-2'),  
                                        html.P("Active cases"),
                                        ])
                                    ],
                                    id="active",
                                    className="mini_container",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                        [
                                        html.H5(covid_cases['death_cases'], id="deathsText", className='mr-2'),  
                                        html.P("Total Deaths"),
                                        ])
                                    ],
                                    id="deaths",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
        dcc.Graph(
            figure=fig,
            # config={
            #     'displayModeBar': False
            #     }
        ),
        
        dbc.Row([
            html.H5("Last updated on "+str(df["Date"].values[-1]), style={'text-align': 'center'}),
            ], justify='end', className="h-20",style={"padding":"2.5rem",'text-align': 'center'},
            ),
    ],
    md=8,
)

#available_rates = ['Effective Growth Rate', 'Reproduction Rate']
column3 = dbc.Col(
    [
        
        html.Hr(),
        html.H3(id='rate_header', children=["Effective Growth Rate - Qatar"]),
        html.Div([
            dcc.Dropdown(
                id='rate_dropdown',
                options=[ {'label':i, 'value':i}  for i in ['Reproduction Rate', 'Effective Growth Rate'] ],
                value = 'Reproduction Rate'
                ),
            ],style={'width': '49%',  'display': 'inline-block'}, className="justify-content-end"),
        dcc.Graph(
            id='small_graph',
            #figure=fig_growth_rate,
        )
    ],
    md=6,
)


@app.callback(
    dash.dependencies.Output('small_graph', 'figure'),
    [dash.dependencies.Input('rate_dropdown', 'value')])
def set_rate_graph(selected_rate):
    if selected_rate=="Reproduction Rate":
        return fig_reprod(result)
    return fig_growth_rate

@app.callback(
    dash.dependencies.Output('rate_header', 'children'),
    [dash.dependencies.Input('rate_dropdown', 'value')])
def set_rate_header(selected_rate):
    if selected_rate is None:
        selected_rate = "Effective Growth Rate"
    return selected_rate+" - Qatar"

column4 = dbc.Col(
    [
            
    ],
    id="col4",
    md=6,

)


@app.callback(
    dash.dependencies.Output('col4', 'children'),
    [dash.dependencies.Input('rate_dropdown', 'value')])
def set_col4_children(selected_rate):
    col4_children = [html.Hr(),
            html.H3(" "),
            html.P("""
            The model uses a time-dependent transmission/spreading rate following the assumption that a signicant change in transmission rate 
            may occur at certain points over the course of a pandemic. This is modelled though change points which corresponds to Government policy
            interventions and events that could affect the transmission rate. The current model includes the following change points"""),
            html.Ol([
                html.Li("10 March 2020: Universities and schools close until further notice."),
                html.Li("18 March 2020: Border restriction including restrictions on inbound passengers."),
                html.Li("23 April 2020: Beginning of Ramadan."),
                html.Li("26 April 2020: Masks made compulsory for all shoppers, service sector and construction sector employees.")]
            ),
            html.H5("""
            When the effective growth rate goes below 0, we will see reduction in new infections and eventually eradicate the pandemic. 
            Our preliminary models show Qatar is close to achieving this. #StayHome
            """)
            ]
    if selected_rate=="Reproduction Rate":
        col4_children = [html.Hr(),
            html.H3(" "),
            html.P("""
            The effective reproductive number provides a one number summary for the state of an epidemic at a given time. 
            It is defined as the number of people who become infected per infectious person at a given time. The higher the 
            reproductive number, the more it is spreading. We want this number to be close to 0 and definitely under 1. Beyond 1, 
            the viral transmission is in a growth phase and is infecting more and more people everyday."""),
            html.P(""" 
            Knowing the current reproductive number is essential as it is a good indicator of where we stand in terms of 
            eradicating the pandemic. The graph on the left is derived from a Bayesian approach to estimate and track the reproductive number 
            on a daily basis. Use the drop-down box to see the result of estimating a similar statistic by keeping the reproductive number
            static given a change point. Whereas the graph to the left is mostly for monitoring the present, the other graph
            derived from the SIR model is for retrospection into the interventions and providing an estimate of the growth of the pandemic
            in the future.   
            """)
            ]
    return col4_children
    

layout = dbc.Container([
            html.P("Graphs are interactive and are best viewed in PC or landscape mode."),
            html.Hr(),
            dbc.Row([column1, column2],justify="center",),
            dbc.Row([column3, column4],justify="center",)
            ],fluid=True,)