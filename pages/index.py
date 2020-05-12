# Imports from 3rd party libraries
import dash
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time

from plotly.tools import mpl_to_plotly
from matplotlib import pyplot as plt
import matplotlib
import pickle
import datetime

import dash_table

# Imports from this application
from app import app



def plot_cases(
    trace,
    new_cases_obs,
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

    new_cases_sim = trace
    len_sim = trace.shape[1] + 16
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
    print("num_days_future ",num_days_future)
    print("end_date_plot ",end_date_plot)
    print(date_data_end)
    start_date_mpl, end_date_mpl = matplotlib.dates.date2num(
        [start_date_plot, end_date_plot]
    )

    if week_interval is None:
        week_inter_left = int(np.ceil(num_days_data / 7 / 5))
        week_inter_right = int(np.ceil((end_date_mpl - start_date_mpl) / 7 / 6))
    else:
        week_inter_left = week_interval
        week_inter_right = week_interval

    # fig, axes = plt.subplots(
    #     1,
    #     1,
    #     #figsize=(36, 20)
    #   #  gridspec_kw={"height_ratios": [1, 3], "width_ratios": [2, 3]},
    # )
    fig = make_subplots(rows=1, cols=1)
    
    # ax = axes
    time_arr = np.arange(-len(new_cases_obs), 0)
    mpl_dates = conv_time_to_mpl_dates(time_arr) + diff_data_sim + num_days_data
    # ax.plot(
    #     mpl_dates,
    #     new_cases_obs,
    #     "d",
    #     markersize=6,
    #     label="Data",
    #     zorder=5,
    #     color=colors[0],
    # )
    new_cases_past = new_cases_sim[:, :num_days_data]
    percentiles = (
        np.percentile(new_cases_past, q=2.5, axis=0),
        np.percentile(new_cases_past, q=97.5, axis=0),
    )
    # ax.plot(
    #     mpl_dates,
    #     np.median(new_cases_past, axis=0),
    #     color=colors[1],
    #     label="Fit (with 95% CI)",
    # )
    # ax.fill_between(
    #     mpl_dates, percentiles[0], percentiles[1], alpha=0.3, color=colors[1]
    # )
    # ax.set_yscale("log")
    # ax.set_ylabel("Number of new cases")
    # ax.set_xlabel("Date")
    # ax.legend()
    # ax.xaxis.set_major_locator(
    #     matplotlib.dates.WeekdayLocator(
    #         interval=week_inter_left, byweekday=matplotlib.dates.SU
    #     )
    # )
    # ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    # ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
    # ax.set_xlim(start_date_mpl)

    # ax = axes[1][1]

    time1 = np.arange(-len(new_cases_obs), 0)
    mpl_dates = conv_time_to_mpl_dates(time1) + diff_data_sim + num_days_data
    mpl_dates = matplotlib.dates.num2date(mpl_dates)
    #mpl_dates = pd.to_datetime(mpl_dates,format='%Y%m%d', errors='ignore')
   # mpl_dates = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in mpl_dates]
    
    # ax.plot(
    #     mpl_dates,
    #     new_cases_obs,
    #     "d",
    #     label="Data",
    #     markersize=4,
    #     color=colors[0],
    #     zorder=5,
    # )

    # #Data trace - Move it at end to overlay over CI
    # fig.add_trace(
    # go.Scatter(x=mpl_dates, y=new_cases_obs, mode='markers', name='Data')
    # )

    new_cases_past = new_cases_sim[:, :num_days_data]
    # ax.plot(
    #     mpl_dates,
    #     np.median(new_cases_past, axis=0),
    #     "--",
    #     color=colors[1],
    #     linewidth=1.5,
    #     label="Fit with 95% CI",
    # )

    percentiles = (
        np.percentile(new_cases_past, q=2.5, axis=0),
        np.percentile(new_cases_past, q=97.5, axis=0),
    )
    # ax.fill_between(
    #     mpl_dates, percentiles[0], percentiles[1], alpha=0.2, color=colors[1]
    # )
    if(second_graph == False):
        fig.add_trace(go.Scatter(x=mpl_dates, y=percentiles[0],  fill='none', line=dict(width=0.5, color='#ffe5ce'), fillcolor='#ffe5ce',
                        mode= 'lines', opacity=0.2, showlegend=False))
        fig.add_trace(go.Scatter(x=mpl_dates, y=percentiles[1],  fill='tonextx', line=dict(width=0.5, color='#ffe5ce'), fillcolor = '#ffe5ce',
                        mode= 'lines', opacity=0.2, showlegend=False))

    percentiles = (
        np.percentile(new_cases_past, q=12.5, axis=0),
        np.percentile(new_cases_past, q=87.5, axis=0),
    )
    # ax.fill_between(
    #     mpl_dates, percentiles[0], percentiles[1], alpha=0.2, color=colors[1]
    # )
    if(second_graph == False):
        fig.add_trace(go.Scatter(x=mpl_dates, y=percentiles[0], fill='none', line=dict(width=0.5, color='#ffdaba'), fillcolor='#ffdaba',
                        mode= 'lines', opacity=0.2, showlegend=False))
        fig.add_trace(go.Scatter(x=mpl_dates, y=percentiles[1], fill='tonextx', line=dict(width=0.5, color='#ffdaba'), fillcolor = '#ffdaba',
                        mode= 'lines', opacity=0.2, showlegend=False))
        fig.add_trace(
        go.Scatter(x=mpl_dates, y=np.median(new_cases_past, axis=0), mode='lines', name='Fit with 95% CI', line={'color':'#00cc96'})
        )


    time2 = np.arange(0, num_days_future)
    print(num_days_future)
    mpl_dates_fut = conv_time_to_mpl_dates(time2) + diff_data_sim + num_days_data
    cases_future = new_cases_sim[:, num_days_data : num_days_data + num_days_future].T
    median = np.median(cases_future, axis=-1)
    percentiles = (
        np.percentile(cases_future, q=2.5, axis=-1),
        np.percentile(cases_future, q=97.5, axis=-1),
    )
    # ax.plot(
    #     mpl_dates_fut,
    #     median,
    #     color=colors[1],
    #     linewidth=3,
    #     label="forecast with 75% and 95% CI",
    # )
    #mpl_dates_fut = [datetime.strptime(x, '%Y-%m-%d').date() for x in mpl_dates_fut]
    #mpl_dates_fut = pd.to_datetime(mpl_dates_fut,format='%Y%m%d', errors='ignore')
    mpl_dates_fut = matplotlib.dates.num2date(mpl_dates_fut)
    print(median)
    print(mpl_dates_fut)
    if(second_graph == False):
        fig.add_trace(
            go.Scatter(x=mpl_dates_fut, y=median, mode='markers', name='forecast with 75% and 95% CI') 
        )


        fig.update_layout(
        xaxis_title="Date",
        yaxis_title="New confirmed cases in Qatar",
        font=dict(
            #family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
            )
        )
    # ax.fill_between(
    #     mpl_dates_fut, percentiles[0], percentiles[1], alpha=0.1, color=colors[1]
    # )
    # ax.fill_between(
    #     mpl_dates_fut,
    #     np.percentile(cases_future, q=12.5, axis=-1),
    #     np.percentile(cases_future, q=87.5, axis=-1),
    #     alpha=0.2,
    #     color=colors[1],
    # )

    # ax.set_xlabel("Date")
    # ax.set_ylabel(f"New confirmed cases in {country}")
    # ax.legend(loc="upper left")
    # ax.set_ylim(0, 3000)
    func_format = lambda num, _: "${:.0f}\,$k".format(num / 1_000)
    #ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(func_format))
    # ax.set_xlim(start_date_mpl, end_date_mpl)
    # ax.xaxis.set_major_locator(
    #     matplotlib.dates.WeekdayLocator(
    #         interval=week_inter_right, byweekday=matplotlib.dates.SU
    #     )
    # )
    # ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    # ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))

    # # ax = axes[0][1]
    
    if(second_graph == True):
        time = np.arange(-diff_to_0, -diff_to_0 + len_sim)
        lambda_t = trace["lambda_t"][:, :]
        μ = trace["mu"][:, None]
        mpl_dates = conv_time_to_mpl_dates(time) + diff_data_sim + num_days_data

        # ax.plot(mpl_dates, np.median(lambda_t - μ, axis=0), color=colors[1], linewidth=2)
        fig.add_trace(
            go.Scatter(x=mpl_dates, y=np.median(lambda_t - μ, axis=0), mode='lines', name='effective\ngrowth rate $\lambda_t^*$')
        )
        # ax.fill_between(
        #     mpl_dates,
        #     np.percentile(lambda_t - μ, q=2.5, axis=0),
        #     np.percentile(lambda_t - μ, q=97.5, axis=0),
        #     alpha=0.15,
        #     color=colors[1],
        # )

        # ax.set_ylabel("effective\ngrowth rate $\lambda_t^*$")

        # ax.set_ylim(-0.15, 0.45)
        # ylims = ax.get_ylim()
        # ax.hlines(0, start_date_mpl, end_date_mpl, linestyles=":")
        # delay = matplotlib.dates.date2num(date_data_end) - np.percentile(
        #     trace["delay"], q=75
        # )
        # ax.vlines(delay, ylims[0], ylims[1], linestyles="-", colors=["tab:red"])
        # ax.set_ylim(*ylims)
        # ax.text(
        #     delay + 0.5,
        #     ylims[1] - 0.04 * np.diff(ylims),
        #     "unconstrained because\nof reporting delay",
        #     color="tab:red",
        #     verticalalignment="top",
        # )
        # ax.text(
        #     delay - 0.5,
        #     ylims[1] - 0.04 * np.diff(ylims),
        #     "constrained\nby data",
        #     color="tab:red",
        #     horizontalalignment="right",
        #     verticalalignment="top",
        # )
        # ax.xaxis.set_major_locator(
        #     matplotlib.dates.WeekdayLocator(
        #         interval=week_inter_right, byweekday=matplotlib.dates.SU
        #     )
        # )
        # ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
        # ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
        # ax.set_xlim(start_date_mpl, end_date_mpl)

    # axes[0][0].set_visible(False)

    # plt.subplots_adjust(wspace=0.4, hspace=0.3)

    #True new_cases_observed Data trace
    if(second_graph == False):
        fig.add_trace(
        go.Scatter(x=mpl_dates, y=new_cases_obs, mode='markers', name='Data')
        )

    return fig


#     fig.update_xaxes(type="log")
# fig.update_yaxes(type="log")

# mpl_fig = plt.figure()
# ax= mpl_fig.add_subplot(111)
# ax.plot(range(10), [i**2 for i in range(10)])
# ax.grid(True)
# fig = mpl_to_plotly(mpl_fig)
filename = 'data/trace.pickle'
infile = open(filename,'rb')
trace = pickle.load(infile)
infile.close()

df_online = pd.read_csv('data/owid-covid-data.csv')
df_online = df_online[(df_online.location == "Qatar")]
#df = df_online.loc[(df_online['date'] >= "2020-03-07") & (df['date'] <= '2020-05-9')]

df = pd.read_csv('data/model_output.csv')
#df = df[(df.location == "Qatar")]
df = df.loc[(df['DT'] >= "2020-03-07") & (df['DT'] <= '2020-05-9')]

df4Table = df.loc[(df['DT'] >= "2020-05-07") & (df['DT'] <= '2020-05-9')]
df4Table.replace(0, np.nan, inplace=True)
df4Table.columns = ['Date', 'Observed', 'Model']
new_cases_obs = (df['Observed_New_Cases'].values)
date_begin_data = datetime.datetime(2020,3,7)
diff_data_sim = 16 # should be significantly larger than the expected delay, in 
                   # order to always fit the same number of data points.
# begin date of observations
bd = date_begin_data

# begin date of simulations which is set to 16 days before 
# first observation by default
sim_bd = bd - datetime.timedelta(days=diff_data_sim)


fig = plot_cases(
    trace,
    new_cases_obs,
    date_begin_sim=sim_bd,
    diff_data_sim=16,
    start_date_plot=None,
    end_date_plot=None,
    ylim=1000,
    week_interval=2,
    colors=("tab:blue", "tab:orange"),
    country="Qatar",
)


# fig_growth_rate = plot_cases(
#     trace,
#     new_cases_obs,
#     date_begin_sim=sim_bd,
#     diff_data_sim=16,
#     start_date_plot=None,
#     end_date_plot=None,
#     ylim=1000,
#     week_interval=2,
#     colors=("tab:blue", "tab:orange"),
#     country="Qatar",
#     second_graph=True
# )

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Jumbotron([ 
        dbc.Col(
                [
                dbc.Card(
                [
                        html.H5("New Case Forecast", className="card-title"),
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
    #md=4,
    #width=4,
    #className="pretty_container four columns",
    )], 
    fluid=True,
    )

# gapminder = px.data.gapminder()
# fig = px.scatter(gapminder.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color="continent",
#            hover_name="country", log_x=True, size_max=60)


column2 = dbc.Col(
    [   
                            dbc.CardDeck(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                        [
                                        html.H4(df_online.iloc[-1]['new_cases'], id="newText", className='mr-2'),  
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
                                        html.H4(df_online.iloc[-1]['new_deaths'], id="recoveredText", className='mr-2'),  
                                        html.P("New Deaths"),
                                        ])
                                        
                                        ],
                                    id="recovered",
                                    className="mini_container",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                        [
                                        html.H4(df_online.iloc[-1]["total_cases"], id="confirmedText", className='mr-2'),  
                                        html.P("Confirmed cases"),
                                        ])
                                    ],
                                    id="confirmed",
                                    className="mini_container",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                        [
                                        html.H4(df_online.iloc[-1]['total_deaths'], id="deathsText", className='mr-2'),  
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
        dcc.Graph(figure=fig),
        
        dbc.Row([
            html.H4("Last updated on 12/05/2020", style={'text-align': 'center'}),
            ], justify='center', className="h-20",style={"padding":"2.5rem",'text-align': 'center'},
            ),
    ],
    md=8,
)

column3 = dbc.Col(
    [
        html.H1("Covid19 Forecast - Qatar"),
            html.Hr(),
        dcc.Graph(
            id='small_graph',
            figure={
                'data': [
                    {'x': [1, 2, 3, 4, 5], 'y': [9, 6, 2, 1, 5], 'type': 'line', 'name': 'First'},
                    {'x': [1, 2, 3, 4, 5], 'y': [8, 7, 2, 7, 3], 'type': 'line', 'name': 'Second'},
                ],
                'layout': {
                    'title': 'Basic Forecast'
                }
            }
        )
    ],
    md=6,
    #width=4,
)

column4 = dbc.Col(
    [
        html.H1("Covid19 Forecast - Qatar"),
            html.Hr(),
        dcc.Graph(
            id='main_graph',
            figure={
                'data': [
                    {'x': [1, 2, 3, 4, 5], 'y': [9, 6, 2, 1, 5], 'type': 'line', 'name': 'First'},
                    {'x': [1, 2, 3, 4, 5], 'y': [8, 7, 2, 7, 3], 'type': 'line', 'name': 'Second'},
                ],
                'layout': {
                    'title': 'Basic Forecast'
                }
            }
        )
    ],
    md=6,
    #width=4,

)

layout = dbc.Container([
            dbc.Row([column1, column2],justify="center",),
            dbc.Row([column3, column4],justify="center",)
            ],fluid=True,)