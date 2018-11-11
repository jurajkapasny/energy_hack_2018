# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 01:08:56 2018

@author: Matus
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table_experiments as dt
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle
from copy import deepcopy
import datetime


app = dash.Dash(__name__)
app.title = 'KNOYD SOLUTION'

external_css = ["https://fonts.googleapis.com/css?family=Dosis",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
                ]

for css in external_css:
    app.css.append_css({"external_url": css})

app.config.suppress_callback_exceptions = True

server = app.server

background_color =  "#F4F4F8" 
font = "Dosis"

back = 32

styles_dict = {'H1'  :      {'textAlign': 'center',
                             'fontSize': '55px',
                             'fontFamily':font},
               'H2'  :      {'textAlign': 'center',
                             'fontSize': '35px',
                              'fontFamily':font},
               'H3'  :      {'textAlign': 'center',
                             'fontSize': '20px',
                              'fontFamily':font},
               'Head':      {'background' : '#00cedd',
                             "border": "3px",
                             "border-radius": "10px",
                             "border-color":"#00cedd",
                             "border-style":"solid",
                             "margin-top": "5px",
                             "margin-bottom": "5px",
                             "margin-right": "0px",
                             "margin-left": "0px"},
               'Img' :      {'height': '5%',
                             'width': '5%',
                             'float': 'right',
                             'position': 'relative',
                             'margin':'0px auto',
                             'align': 'center'},
               'Container': {'background' : background_color,
                             'fontFamily':font,
                             "fontSize": "20px"},
               'label':     {'fontFamily':font,
                             'fontSize': '20px',
                             "fontWeight":"bold"},
               'graph':     {'fontFamily':font,
                             'fontSize': '20px',
                             "fontWeight":"bold",},
               "dropdown":  {"border-radius": "10px",
                             "border":"1px",
                             "border-color":"#00cedd",
                             "border-style":"solid"},
               "main_div":  {"border": "3px",
                             "border-radius": "10px",
                             "border-color":"#00cedd",
                             "border-style":"solid",
                             "margin-top": "10px",
                             "margin-bottom": "10px",
                             "margin-right": "0px",
                             "margin-left": "0px"}}
               
               
dashboardMarkdown ="""
DASHBOARD"""

info_markdown ="""
__Druh tarify:__ {}  
__Číslo OM:__ {}  
__Zapojenie:__ {}  
__Inštalovaný výkon FVE (kWp):__ {}"""

om_with_pv_markdown="""
__Electricity from network__ {}  
__Electricity from comunity__ {}  
__Distribution costs__ {}  
__Revenue from network__ {}  
__Revenue from comunity__ {}"""

om_without_pv_markdown="""
__Electricity from network__ {}  
__Electricity from comunity__ {}  
__Distribution costs__ {}  
__Revenue from network__ {}  
__Revenue from comunity__ {}"""

data_path = "./data/"
om_info = pd.read_csv(data_path + "om_info_prepared.csv",sep=";")

om_dataframes = pd.read_csv(data_path +"final_data_restricted.csv")
om_dataframes_sum = pd.read_csv(data_path +"final_data_restricted_sum.csv")
unique_cluster = ["None"] + list(om_dataframes.cluster.unique())

def refresh_layout():
    layout = html.Div([
                dcc.Interval(id="interval",
                             interval=5*1000, # in milliseconds
                             n_intervals=0),
                
                html.Div([
                              html.Div([         
                                        html.H1(
                                                children='EnergyHack-Knoyd',
                                                style= styles_dict["H1"],
                                                )
                                        ],
                                        style = styles_dict["Head"],
                                        className="col",
                                        
                                        ),
                            ],
                            className="row"),
                    
                html.Hr(),
                
                html.Div([

                        html.Div([
                                html.Div(id="revenue_div")
                                ],
                                className="col-3"),
                        
                        html.Div([
                                html.Div(id="distribution_div")
                                ],
                                className="col-3"),
                        
                        html.Div([
                                html.Div(id="electricity_div")
                                ],
                                className="col-3"),
                        
                        html.Div([
                                dcc.Checklist(
                                        id="energy_checkbox",
                                        options=[
                                                {'label': ' Energy production', 'value': True}
                                                ],
                                        values=[True],
                                        style = styles_dict["label"]
                                        )
                                ],
                                className="col-3")
                        ],
                        className="row"),
                
                html.Div([
                        html.Div([
                                html.Div([
                                    html.Div([
                                            html.Label(
                                                    "Select cluster:",
                                                    style = styles_dict["label"]
                                                    )
                                            ],
                                            className="col"),
                                    ],
                                    className="row"),
                                
                                html.Div([
                                        html.Div([
                                                dcc.Dropdown(
                                                            id='group_dropdown',
                                                            options=[
                                                                {'label': f"{om}", 'value': om} for om in unique_cluster
                                                            ],
                                                            value=1
                                                        ),
                                                ],
                                                className="col")
                                        ],
                                        className="row"),
                                
                                html.Div([
                                        html.Div([
                                                html.Label(
                                                        "Select customer:",
                                                        style = styles_dict["label"]
                                                        )
                                                ],
                                                className="col"),
                                    ],
                                    className="row"),
                                                
                                html.Div([
                                        html.Div([
                                                dcc.Dropdown(
                                                            id='customer_dropdown',
                                                        ),
                                                ],
                                                className="col")
                                        ],
                                        className="row"),
                                
                                html.Br(),
                                                
                                html.Div([
                                        html.Div([
                                                html.Div(id="info_div")
                                                ],
                                                className="col")
                                        ],
                                        className="row"),
                                
                                html.Br(),
                                        
                                html.Div([
                                        html.Div([
                                                html.Div(id="om_with_pv")
                                                ],
                                                className="col")
                                        ],
                                        className="row"),
                                
                                html.Br(),

                                
                                html.Div([
                                        html.Div([
                                                html.Div(id="om_without_pv")
                                                ],
                                                className="col")
                                        ],
                                        className="row")
                                ],
                                className="col-3"),
                                
                                
                        
                        html.Div([
                                html.Div([
                                        html.Div([
                                                dcc.Graph(id="predictions")
                                                ],
                                                className="col")
                                        ],
                                        className="row"),
                                
                                html.Div([
                                        html.Div([
                                                dcc.Graph(id="comparison")
                                                ],
                                                className="col")
                                        ],
                                        className="row"),
                                
                                html.Div([
                                        html.Div([
                                                dcc.Graph(id="spendings")
                                                ],
                                                className="col")
                                        ],
                                        className="row")
                                
                                ],
                                className="col-9")
                        ],
                        className="row"),
            
            ],
            className = "container",
            style = styles_dict["Container"]
            )
    
    return layout 

app.layout = refresh_layout


@app.callback(
    dash.dependencies.Output('customer_dropdown', 'options'),
    [dash.dependencies.Input('group_dropdown', 'value')])
def update_customer_dropdown(cluster):

      df = om_dataframes[om_dataframes.cluster == cluster]
      unique_om = list(df.om.unique())
      
      return [
              {'label': f"{om}", 'value': om} for om in unique_om
          ]


@app.callback(
    dash.dependencies.Output('info_div', 'children'),
    [dash.dependencies.Input('customer_dropdown', 'value')])
def update_info(om):
    if om != "None":
        df = om_info[om_info["Číslo OM"] == om]
        
        return html.Div([
                    html.Br(),
                    html.H3("Info"),
                    html.Hr(),
                    dcc.Markdown(info_markdown.format(df["Druh tarify"].values[0],
                                    df["Číslo OM"].values[0],
                                    df["Zapojenie"].values[0],
                                    df["Inštalovaný výkon FVE (kWp)"].values[0]))
                ])
                    
@app.callback(
    dash.dependencies.Output('revenue_div', 'children'),
    [dash.dependencies.Input('interval', 'n_intervals'),
     dash.dependencies.Input('energy_checkbox', 'values')],
    events=[dash.dependencies.Event('interval', 'interval')])
def update_revenue_div(intervals,chckbox):
    
    if chckbox:
        df = om_dataframes.groupby("timestamp")[["revenue_zse_total_1D"]].sum()
        df.reset_index(inplace=True)
        from_index = df[df.timestamp == "2016-10-31 00:00:00"].index[0]
        value = round(df.iloc[from_index+intervals,:]["revenue_zse_total_1D"],ndigits=2)
        return dcc.Markdown(f"""__ZSE revenue:__ {value}""")
    
    else:
        df = om_dataframes.groupby("timestamp")[["revenue_zse_total_no_power_1D"]].sum()
        df.reset_index(inplace=True)
        from_index = df[df.timestamp == "2016-10-31 00:00:00"].index[0]
        value = round(df.iloc[from_index+intervals,:]["revenue_zse_total_no_power_1D"],ndigits=2)
        return dcc.Markdown(f"""__ZSE revenue (no PV):__ {value}""")
    
@app.callback(
    dash.dependencies.Output('distribution_div', 'children'),
    [dash.dependencies.Input('interval', 'n_intervals'),
     dash.dependencies.Input('energy_checkbox', 'values')],
    events=[dash.dependencies.Event('interval', 'interval')])
def update_distribution_div(intervals,chckbox):
    
    if chckbox:
        df = om_dataframes.groupby("timestamp")[["revenue_zse_distribucia_1D"]].sum()
        df.reset_index(inplace=True)
        from_index = df[df.timestamp == "2016-10-31 00:00:00"].index[0]
        value = round(df.iloc[from_index+intervals,:]["revenue_zse_distribucia_1D"],ndigits=2)
        return dcc.Markdown(f"""__From distribution:__ {value}""")
    
    else:
        df = om_dataframes.groupby("timestamp")[["revenue_zse_distribucia_no_power_1D"]].sum()
        df.reset_index(inplace=True)
        from_index = df[df.timestamp == "2016-10-31 00:00:00"].index[0]
        value = round(df.iloc[from_index+intervals,:]["revenue_zse_distribucia_no_power_1D"],ndigits=2)
        return dcc.Markdown(f"""__From distribution (no PV):__ {value}""")
    
    
@app.callback(
    dash.dependencies.Output('electricity_div', 'children'),
    [dash.dependencies.Input('interval', 'n_intervals'),
     dash.dependencies.Input('energy_checkbox', 'values')],
    events=[dash.dependencies.Event('interval', 'interval')])
def update_electricity_div(intervals,chckbox):
    
    if chckbox:
        df = om_dataframes.groupby("timestamp")[["revenue_zse_elektrika_1D"]].sum()
        df.reset_index(inplace=True)
        from_index = df[df.timestamp == "2016-10-31 00:00:00"].index[0]
        value = round(df.iloc[from_index+intervals,:]["revenue_zse_elektrika_1D"],ndigits=2)
        return dcc.Markdown(f"""__From electricity:__ {value}""")
    
    else:
        df = om_dataframes.groupby("timestamp")[["revenue_zse_elektrika_no_power_1D"]].sum()
        df.reset_index(inplace=True)
        from_index = df[df.timestamp == "2016-10-31 00:00:00"].index[0]
        value = round(df.iloc[from_index+intervals,:]["revenue_zse_elektrika_no_power_1D"],ndigits=2)
        return dcc.Markdown(f"""__From electricity (no PV):__ {value}""")
        
                    
@app.callback(
    dash.dependencies.Output('interval', 'n_intervals'),
    [dash.dependencies.Input('customer_dropdown', 'value'),
     dash.dependencies.Input('group_dropdown', 'value')])
def update_interval(cus,group):
    
    return 0

@app.callback(
    dash.dependencies.Output('om_with_pv', 'children'),
    [dash.dependencies.Input('interval', 'n_intervals'),
     dash.dependencies.Input('customer_dropdown', 'value')],
    events=[dash.dependencies.Event('interval', 'interval')])
def update_om_with_pv_div(intervals,customers):
    
    if customers:
        df = om_dataframes[om_dataframes.om == customers]
        df = df.sort_values("timestamp")
        df.reset_index(inplace=True)
        from_index = df[df.timestamp == "2016-10-31 00:00:00"].index[0]
        df_real = df.iloc[from_index+intervals,:]
        
        return html.Div([
                    html.Br(),
                    html.H3("OM with PV"),
                    html.Hr(),
                    dcc.Markdown(om_with_pv_markdown.format(round(df_real["cost_OM_za_el_zo_siete_1D"],3),
                                    round(df_real["cost_OM_za_el_z_komunity_1D"],3),
                                    round(df_real["cost_OM_za_distribuciu_1D"],3),
                                    round(df_real["revenue_OM_za_predaj_do_siete_1D"],3),
                                    round(df_real["revenue_OM_za_predaj_do_komunuty_1D"],3)))
                ])
                    
@app.callback(
    dash.dependencies.Output('om_without_pv', 'children'),
    [dash.dependencies.Input('interval', 'n_intervals'),
     dash.dependencies.Input('customer_dropdown', 'value')],
    events=[dash.dependencies.Event('interval', 'interval')])
def update_om_without_pv_div(intervals,customers):
    
    if customers:
        df = om_dataframes[om_dataframes.om == customers]
        df = df.sort_values("timestamp")
        df.reset_index(inplace=True)
        from_index = df[df.timestamp == "2016-10-31 00:00:00"].index[0]
        df_real = df.iloc[from_index+intervals,:]
        
        return html.Div([
                    html.Br(),
                    html.H3("OM without PV"),
                    html.Hr(),
                    dcc.Markdown(om_without_pv_markdown.format(round(df_real["cost_OM_za_el_zo_siete_no_power_1D"],3),
                                    round(df_real["cost_OM_za_el_z_komunity_no_power_1D"],3),
                                    round(df_real["cost_OM_za_distribuciu_no_power_1D"],3),
                                    round(df_real["revenue_OM_za_predaj_do_siete_no_power_1D"],3),
                                    round(df_real["revenue_OM_za_predaj_do_komunuty_no_power_1D"],3)))
                ])
        
                    

@app.callback(
    dash.dependencies.Output('predictions', 'figure'),
    [dash.dependencies.Input('customer_dropdown', 'value'),
     dash.dependencies.Input('group_dropdown', 'value'),
     dash.dependencies.Input('interval', 'n_intervals')],
     events=[dash.dependencies.Event('interval', 'interval')])
def update_predictions(cust,group,intervals):
    
    if group == "None":
        df = om_dataframes_sum
    
    else:
        df = om_dataframes[om_dataframes.om == cust]
        
    df = df.sort_values("timestamp")
    df.reset_index(inplace=True)
    from_index = df[df.timestamp == "2016-10-31 00:00:00"].index[0] - back
    df_real = df.iloc[from_index+intervals: from_index+back+intervals,:]
    
    predicitions_df=pd.DataFrame(index = df.iloc[[i for i in range(from_index+back+intervals,from_index+back+intervals+4)],:].timestamp,
                                 data=[df.iloc[from_index+back+intervals,:]["predicted__lag_0"],
                                       df.iloc[from_index+back+intervals,:]["predicted__lag_1"],
                                       df.iloc[from_index+back+intervals,:]["predicted__lag_1"],
                                       df.iloc[from_index+back+intervals,:]["predicted__lag_1"]],
                                 columns= ["predictions"])
    
    
    outliers = go.Scatter(
        x=df_real[df_real.is_outlier == True].timestamp,
        y=df_real[df_real.is_outlier == True].spotreba_x,
        connectgaps=True,
        mode = "markers",
        name='outliers',
        marker = dict(
                size = 15,
                color = "red"
                )
    )

    lower = go.Scatter(
            x=df_real.timestamp,
            y=df_real.lower_bound_x,
            connectgaps=True,
            mode = "lines+markers",
            name='lower_bound',
            marker=dict(
                    color="00cedd"),
                    line=dict(
                            color='00cedd',
                            width=2
                            ),
            opacity=0.4
        )
    
    upper = go.Scatter(
            x=df_real.timestamp,
            y=df_real.upper_bound_x,
            connectgaps=True,
            mode = "lines+markers",
            name='upper_bound',
            fill='tonexty',
            marker=dict(
                    color="00cedd"),
                    line=dict(
                            color='00cedd',
                            width=2
                            ),
            opacity=0.4
        )
    predictions= go.Scatter(
        x=predicitions_df.index,
        y=predicitions_df.predictions,
        connectgaps=True,
        mode="lines+markers",
        name='predictions'
    )
    
    actual = go.Scatter(
        x=df_real.timestamp,
        y=df_real.spotreba_x,
        connectgaps=True,
        mode = "lines+markers",
        name='actual',
        marker = dict(
                color = "CB4B4B"
                ),
        opacity=1
    )
    
    layout = go.Layout(
        showlegend=True,
        title="Predictions",
        paper_bgcolor = background_color,
    	plot_bgcolor = background_color,
        xaxis={
            'title': 'Year and month'
        },
        yaxis={
            'title': 'Usage kWh',
        })
        
    data = [lower,upper,predictions,outliers,actual]
    
    return { 
                'data': data,
                'layout': layout,
        }
    

@app.callback(
    dash.dependencies.Output('comparison', 'figure'),
    [dash.dependencies.Input('customer_dropdown', 'value'),
     dash.dependencies.Input('group_dropdown', 'value'),
     dash.dependencies.Input('interval', 'n_intervals')],
     events=[dash.dependencies.Event('interval', 'interval')])
def update_comparison(cust,group,intervals):
    
    if group == "None":
        df = om_dataframes_sum
    
    else:
        df = om_dataframes[om_dataframes.om == cust]
        
    df = df.sort_values("timestamp")
    df.reset_index(inplace=True)
    from_index = df[df.timestamp == "2016-10-31 00:00:00"].index[0] - back
    df_real = df.iloc[from_index+intervals: from_index+back+intervals,:]
       
    actual = go.Scatter(
        x=df_real.timestamp,
        y=df_real.spotreba_x,
        connectgaps=True,
        mode = "lines+markers",
        name='actual',
        marker = dict(
                color = "00cedd"
                ),
        opacity=1
    )
    

    lower = go.Scatter(
            x=df_real.timestamp,
            y=df_real.lower_bound_y,
            connectgaps=True,
            mode = "lines+markers",
            name='lower_bound',
            marker=dict(
                    color="CB4B4B"),
                    line=dict(
                            color='CB4B4B',
                            width=1.5
                            ),
            opacity=0.2
            )
    
    upper = go.Scatter(
            x=df_real.timestamp,
            y=df_real.upper_bound_y,
            connectgaps=True,
            mode = "lines+markers",
            name='upper_bound',
            fill='tonexty',
            marker=dict(
                    color="CB4B4B"),
                    line=dict(
                            color='CB4B4B',
                            width=1.5
                            ),
            opacity=0.2
        )
    
    
    layout = go.Layout(
        title="Comparison",
        paper_bgcolor = background_color,
    	plot_bgcolor = background_color,
        xaxis={
            'title': 'Year and month'
        },
        yaxis={
            'title': 'Usage kWh',
        })
        
    data = [lower,upper,actual]
    
    return { 
                'data': data,
                'layout': layout,
        }
    
    
@app.callback(
    dash.dependencies.Output('spendings', 'figure'),
    [dash.dependencies.Input('customer_dropdown', 'value'),
     dash.dependencies.Input('group_dropdown', 'value'),
     dash.dependencies.Input('interval', 'n_intervals')],
     events=[dash.dependencies.Event('interval', 'interval')])
def update_spendings(cust,group,intervals):
    
    if group == "None":
        df = om_dataframes_sum
    
    else:
        df = om_dataframes[om_dataframes.om == cust]
        
    df = df.sort_values("timestamp")
    df.reset_index(inplace=True)
    from_index = df[df.timestamp == "2016-10-31 00:00:00"].index[0] - back
    df_real = df.iloc[from_index+intervals: from_index+back+intervals,:]
       
    income_from_comunity = go.Bar(
            x=df_real.timestamp,
            y=df_real.income_from_komunity,
            width=800000,
            showlegend=True,
            opacity=0.4,
            marker=dict(
                    color="00cedd"),
            name='Income from comunity'
        )
    
    income_from_network = go.Bar(
            x=df_real.timestamp,
            y=df_real.income_from_network,
            width=800000,
            name = 'Income from network',
            showlegend=True,
            opacity=1,
            marker=dict(
                    color="00cedd")
        )
    
    expense_to_network = go.Bar(
            x=df_real.timestamp,
            y=-df_real.expense_to_network,
            width=800000,
            name = 'Expense to network',
            showlegend=True,
            opacity=0.5,
            marker=dict(
                    color="CB4B4B")
        )
    
    expense_to_comunity = go.Bar(
            x=df_real.timestamp,
            y=-df_real.expense_to_komunity,
            width=800000,
            name = 'Expense to comunity',
            showlegend=True,
            opacity=1,
            marker=dict(
                    color="CB4B4B")
        )
    
	
        
    layout = go.Layout(
            barmode='stack',
            title=f'Income vs Spending ',
            yaxis={'title': '€'},
            paper_bgcolor = background_color,
    	    plot_bgcolor = background_color,
            autosize = True,
            font={
                "family": font,
                "size": 20
                }
            
            )   
        
    data = [income_from_comunity,income_from_network,expense_to_network,expense_to_comunity]
    
    return { 
                'data': data,
                'layout': layout,
        }
                        



if __name__ == '__main__':
    app.run_server(debug=True)

