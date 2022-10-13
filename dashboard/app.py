import dash
import dash_leaflet as dl
import dash_leaflet.express as dlx
import pandas as pd
from dash import Dash, html, dcc, Output, Input, State
from dash_extensions.javascript import arrow_function
import json
import dash_bootstrap_components as dbc
import plotly.express as px
import pickle
import os
json_file_path = r"Resources/Neighbourhoods.geojson"
keys = ["watercolor"]
url_template = "http://{{s}}.tile.stamen.com/{}/{{z}}/{{x}}/{{y}}.png"
attribution = 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, ' \
              '<a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data ' \
              '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

with open(json_file_path, 'r') as j:
    tdot = json.loads(j.read())
#fa = r'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css'

import ML_Analysis
from ML_Analysis import forecaster,ForecastingHorizon, y,df_all_toronto_clean_with_rates,fh_new, grps
import sktime
# from sktime.forecasting.base import ForecastingHorizon
#
# pickles=['forecast.pickle','y.pickle','df_all_to_with_rates.pickle','fh_new','grps.pickle']
#
# forecaster = pickle.load(open(r'Team1_project/dashboard/Resources/'+pickles[0],'rb'))
# y = pickle.load(open(os.path.join(os.getcwd(),'Resources',pickles[1]),'rb'))
# df_all_toronto_clean_with_rates = pickle.load(open(os.path.join(os.getcwd(),'Resources',pickles[2]),'rb'))
# fh_new = pickle.load(open(os.path.join('Resources',pickles[3]),'rb'))
# grps = pickle.load(open(os.path.join('Resources',pickles[4]),'rb'))
#

#df= pd.read_csv(r'../Resources/alldata.csv')
df=df_all_toronto_clean_with_rates
# df._date = df._date.apply(lambda x: pd.Period(x))
# df.set_index('_date',inplace=True)

#df_pred=pd.read_csv(r'../Resources/price_forecast.csv')

# df_pred=y_pred
# df_pred._date = df_pred._date.apply(lambda x: pd.Period(x))
# df_pred.set_index(['community','building_type','_date'],inplace=True)
# df_pred=df_pred.dropna()
# df_pred.sort_index(inplace=True)
# with open('Team1_project/forecast', 'rb') as fp:
#     forecaster = pickle.loads(fp)

X=df.reset_index().set_index(['community','building_type','_date'])
X.sort_index(inplace=True)
X=X.dropna()
y = X['average_price'].to_frame()
X.drop('average_price',inplace=True,axis=1)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
def make_plots(y_train,y_pred,community_filter,building_filter):
    y_train_plot = y_train.loc[(y_train.index.get_level_values('community') == community_filter) & (
            y_train.index.get_level_values('building_type') == building_filter)]
    y_pred_plot = y_pred.loc[(y_pred.index.get_level_values('community') == community_filter) & (
            y_pred.index.get_level_values('building_type') == building_filter)]
    # y_test_plot = y_test.loc[(y_test.index.get_level_values('community') == community_filter) & (
    #         y_test.index.get_level_values('building_type') == building_filter)]

    y_train_plot = y_train_plot.reset_index()
    y_train_plot = y_train_plot.set_index('_date')
    y_train_plot = y_train_plot.drop(['community', 'building_type'], axis=1)

    y_pred_plot = y_pred_plot.reset_index()
    y_pred_plot = y_pred_plot.set_index('_date')
    y_pred_plot2 = y_pred_plot.drop(['community', 'building_type'], axis=1)

    # y_test_plot = y_test_plot.reset_index()
    # y_test_plot = y_test_plot.set_index('_date')
    # y_test_plot = y_test_plot.drop(['community', 'building_type'], axis=1)

    #fig = go.Figure()
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[{"type": "scatter"}],
               [{"type": "table"}]]
    )
    fig.add_trace(go.Scatter(x=y_train_plot.index.strftime('%m-%Y'), y=y_train_plot['average_price'],
                             name=community_filter + ' Historical Prices'),row=1,col=1)
    # fig.add_trace(go.Scatter(x=y_test_plot.index.strftime('%m-%Y'), y=y_test_plot['average_price'], mode='lines',
    #                          name=community_filter + ' Historical Prices in Forecasting Horizon'),row=1,col=1)
    fig.add_trace(
        go.Scatter(x=y_pred_plot2.index.strftime('%m-%Y'), y=y_pred_plot2['average_price'], mode='lines+markers',
                   name=community_filter + ' Forecast Prices'),row=1,col=1)
    fig.update_layout(title=f'Historical and Forecast Community Prices for {community_filter} {building_filter}',
                      xaxis_title='Quarter',
                      yaxis_title='Price ($CAD)',height=900)
    y_pred_plot.reset_index(inplace=True)
    fig.add_trace(go.Table(
        header=dict(values=list(y_pred_plot.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[y_pred_plot._date.dt.strftime('%m-%Y'), y_pred_plot.community, y_pred_plot.building_type,
                           y_pred_plot.average_price.apply(lambda x: "${:,.0f}".format((x)))],
                   fill_color='lavender',
                   align='left')), row=2, col=1)
    fig.update_layout(height=1400)
    return fig

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP,dbc.icons.FONT_AWESOME])
app.layout = dbc.Container(html.Div([
    html.Br(),
    html.Br(),
    dbc.Row(
        dbc.Col(html.Div(html.H1('Welcome')))
    ),
    html.Br(),
    dbc.Row(dbc.Col(html.Div('Should I stay or should I go? We try to answer this age old question'))),

    dbc.Row(

    html.Div(
        [
            html.Span(children=[html.I(className='fa-regular fa-map',id="map-collapse",
                n_clicks=0)
                ,html.Div("Choose my Neighborhood")
                                ])
                ,
            dbc.Collapse(
                dbc.Card(dbc.CardBody(
                    dl.Map(center=[43.69, -79.3532], zoom=11.5, children=[
                        dl.LayersControl([dl.BaseLayer(
                        dl.TileLayer(
                            url=url_template.format(key), attribution=attribution),
                                                  name=key, checked=key == "watercolor") for key in keys]
                        ),
                dl.GeoJSON(data=tdot,format="geojson",id='hoods', hoverStyle=arrow_function(dict(weight=5, color='#666', dashArray=''))),
                    ], style={'width': '80%', 'height': '60vh', 'margin': "auto", "display": "block"}, id="map"),

#


                )),
                id="map-collapse-comp",
                is_open=True,
            ),
        ]
    )),

    dbc.Row(children=[
        html.Div([
        html.Span(children=[html.I(className="fa-solid fa-landmark", id="building-collapse",
                                   n_clicks=0)
            , html.Div("Choose my Building")
                            ])]),
        dbc.Collapse(
        children=[
        dcc.RadioItems(
            [
                {
                    "label": html.Div(
                        [
                            html.Img(src="/assets/images/dethome-icn.svg", height=30),
                            html.Div("Detached", style={'font-size': 15, 'padding-left': 10}),
                        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                    ),
                    "value": "Detached",
                },
                {
                    "label": html.Div(
                        [
                            html.Img(src="/assets/images/semidet-icn.svg", height=30),
                            html.Div("Semi", style={'font-size': 15, 'padding-left': 10}),
                        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                    ),
                    "value": "Semi-Detached",
                },
                {
                    "label": html.Div(
                        [
                            html.Img(src="/assets/images/row-icn.svg", height=30),
                            html.Div("Townhouse", style={'font-size': 15, 'padding-left': 10}),
                        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                    ),
                    "value": "Att/Row/Twnhouse",
                },
                {
                    "label": html.Div(
                        [
                            html.Img(src="/assets/images/condo-icn.svg", height=30),
                            html.Div("Condo", style={'font-size': 15, 'padding-left': 10}),
                        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                    ),
                    "value": "Condo Apt",
                },
            ],
            inline=True,id='building-radio')],id="building-collapse-comp",
        is_open=False),

        ]),


    dbc.Row(dbc.Card(
        #dcc.Dropdown(id='building_type_selector',options={'Detached':'Detached','Semi':'Semi-Detached','Townhouse':'Att/Row/Twnhouse','Condo':'Condo Apt'},multi=False)


    )),
    dcc.Loading(dbc.Collapse(dbc.Row([],id='chart_section'),id='charts_collapse',is_open=True)

    ),

    dcc.Loading(dbc.Row([dbc.Col([
        dbc.Card(children=[


            html.Div(children=[html.P('Expected future mortgage rates'),
                               dcc.Input(
                                   id='mortgage_1', type="number",
                                   debounce=False, value=6.25,
                                   min=0, max=20, step=0.25, size='4'
                               ),
                                dcc.Input(
                                   id='mortgage_2', type="number",
                                   debounce=False, value=6.25,
                                   min=0, max=20, step=0.25, size='4'
                               ),
                                dcc.Input(
                                   id='mortgage_3', type="number",
                                   debounce=False, value=6.25,
                                   min=0, max=20, step=0.25, size='4'
                               ),
                                dcc.Input(
                                   id='mortgage_4', type="number",
                                   debounce=False, value=6.25,
                                   min=0, max=20, step=0.25, size='4'
                               )

                               ], id='mortgage_rate_card'),

            html.Div(children=[html.P('Expected future inflation'),
                               dcc.Input(
                                   id='inflation_1', type="number",
                                   debounce=False, value=5.25,
                                   min=0, max=20, step=0.25, size='4'
                                ),
                                dcc.Input(
                                   id='inflation_2', type="number",
                                   debounce=False, value=5.25,
                                   min=0, max=20, step=0.25, size='4'
                                ),
                                dcc.Input(
                                   id='inflation_3', type="number",
                                   debounce=False, value=5.25,
                                   min=0, max=20, step=0.25, size='4'
                                ),
                                dcc.Input(
                                   id='inflation_4', type="number",
                                   debounce=False, value=5.25,
                                   min=0, max=20, step=0.25, size='4'
                                )
                               ], id='inflation_rate_card'),

            html.Div(children=[html.P('Recessionary period'),
                               dcc.Checklist(
                                   [{'label':'Recession',
                                   'value':1}],
                                   style={'display': 'inline'},

                                   id='recession_1'
                               ),
                                dcc.Checklist(
                                    [{'label': 'Recession',
                                     'value': 1}],

                                    id='recession_2',
                               style={'display':'inline'}),
                                dcc.Checklist(
                                    [{'label': 'Recession',
                                     'value': 1}],
                                    style={'display':'inline'},
                                    id='recession_3'
                               ),
                                dcc.Checklist(
                                    [{'label': 'Recession',
                                     'value': 1}],
                                    style={'display': 'inline'},

                                    id='recession_4'
                               ),
                               ], id='recession_card'),

                html.Div(html.Button('Submit',id='submit_forecast',n_clicks=0)) #dbc.Button('Submit',id='submit_forecast',color='info',className='me-1',n_clicks=0))

                ],body=True,id='controller'),


        html.Div(id="number-out"),

    ],width=3)




                ,dbc.Col(id='forecast_chart_section',width=9)

                ],id='forecast_section')),

    html.Div(id="state"), html.Div(id="capital")

    , dcc.Store(id='community-store'),
    dcc.Store(id='building-store')
]),fluid=True)

def frame_filter(df,community,building):
    #query_builder
    df_filtered = df.query(f'community=="{community}"&building_type=="{building}"')
    return df_filtered

def charts(df):
    #BAR OF sp/lp
    fig1 = px.bar(x=df['average_sp_lp'].index.strftime('%Y%q'), y=df['average_sp_lp'],labels={'y':'Average Sales/List Ratio'},title='Historical Average Selling Price to List Price')#,color='blue')
    # fig1 = px.bar(df,x=df['average_sp_lp'].index.strftime('%Y%q'), y='average_sp_lp')


    fig1.update_yaxes(range=[0.8, 1.2])
    df['4pMA'] = df['average_sp_lp'].rolling(4).agg('mean')
    fig1.add_scatter(mode='lines', x=df['4pMA'].index.strftime('%Y%q'), y=df['4pMA'])
    #bar of new listings
    fig2 = px.bar(x=df['new_listings'].index.strftime('%Y%q'), y=df['new_listings'],labels={'y':'Count of New Listings'},title='Historical Count of New Listing')#,color='orange')
    # fig.update_yaxes(range=[0.8, 1.2])
    # test['4pMA'] = test['average_sp_lp'].rolling(4).agg('mean')
    # fig.add_scatter(mode='lines', x=test['4pMA'].index.strftime('%Y%q'), y=test['4pMA'])
    #fig.show(renderer='browser')

    #line of Average DOM
    fig3 = px.bar(x=df['average_dom'].index.strftime('%Y%q'), y=df['average_dom'],labels={'y':'Average Days on Market'},title='Historical Average Number of Day on Market')#,color='green')
    # fig.update_yaxes(range=[0.8, 1.2])
    # test['4pMA'] = test['average_sp_lp'].rolling(4).agg('mean')
    # fig.add_scatter(mode='lines', x=test['4pMA'].index.strftime('%Y%q'), y=test['4pMA'])
    #fig.show(renderer='browser')
    return [fig1,fig2,fig3]


# @app.callback(
#     Output("map-collapse-comp", "is_open"),
#     [Input("map-collapse", "n_clicks")],
#     [State("map-collapse-comp", "is_open")],
# )
# def toggle_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open

# @app.callback(
#     Output("building-collapse-comp", "is_open"),
#     [Input("building-collapse", "n_clicks")],
#     [State("building-collapse-comp", "is_open")],
# )
# def toggle_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open

#
# @app.callback(Output("chart_section", "children"), [Input("hoods", "click_feature")],prevent_initial_call=True)
# def hood_click(feature):
#     if feature is not None:
#         criteria=feature['properties']['AREA_NAME']
#         df_filtered = frame_filter(df,criteria=criteria)
#         chartlist = charts(df_filtered.copy())#use copy of df_filtered
#         chart_section = [dbc.Col(dcc.Graph(id=str(idx),figure=x)) for idx,x in enumerate(chartlist)]
#         return chart_section#f"You clicked {feature['properties']['AREA_NAME']}"



@app.callback([Output("building-collapse-comp", "is_open"),
                Output("map-collapse-comp", "is_open"),
               Output("community-store","data"),
               Output("building-store","data")],
              [Input("hoods", "click_feature"),
                Input("building-radio", "value"),
               Input("map-collapse","n_clicks"),
                Input("building-collapse","n_clicks"),
               ],
                [State("building-collapse-comp", "is_open"),
                State("map-collapse-comp", "is_open")
               ],prevent_initial_call=True)

def hood_click(map_click,building_val,map_button_click,building_button_click,building_collapse_is_open,map_collapse_is_open):
    ctx = dash.ctx.triggered_id

    if ctx == 'hoods':
        if map_click is not None:
            criteria=map_click['properties']['AREA_NAME']
            # df_filtered = frame_filter(df,criteria=criteria)
            # chartlist = charts(df_filtered.copy())#use copy of df_filtered
            # chart_section = [dbc.Col(dcc.Graph(id=str(idx),figure=x)) for idx,x in enumerate(chartlist)]
            #return chart_section#f"You clicked {feature['properties']['AREA_NAME']}"
            return not building_collapse_is_open, not map_collapse_is_open, criteria,dash.no_update
        else:
            not building_collapse_is_open, not map_collapse_is_open, dash.no_update,dash.no_update
    elif ctx == 'map-collapse':
        return dash.no_update, not map_collapse_is_open, dash.no_update,dash.no_update

    elif ctx =='building-collapse':
        return not building_collapse_is_open, dash.no_update, dash.no_update,dash.no_update
    elif ctx == 'building-radio':

        return not building_collapse_is_open, dash.no_update, dash.no_update, building_val
    else:
        return dash.no_update, dash.no_update, dash.no_update,dash.no_update


@app.callback(Output('chart_section','children'),
              Input('community-store','data'),
              Input('building-store','data'),prevent_initial_call=True)
def on_data_set_table(community,building):
    if community is None or building is None:
        raise dash.exceptions.PreventUpdate
    else:

        df_filtered = frame_filter(df,community,building)
        if df_filtered.shape[0] == 0:
            return html.Div(f'Sorry, there is not enough data to show results for {community} & {building} combination.\r\n'
                            f'Please try another building type for {community}.')
        chartlist = charts(df_filtered.copy())#use copy of df_filtered
        chart_section = [dbc.Col(dcc.Graph(id=str(idx),figure=x)) for idx,x in enumerate(chartlist)]
        return chart_section#f"You clicked {feature['properties']['AREA_NAME']}"



@app.callback(
    output=[Output('forecast_chart_section', "children"),
            Output('charts_collapse','is_open')],

    inputs=dict(button=Input('submit_forecast', "n_clicks"),
    charts_collapse_is_open=State('charts_collapse','is_open'),
    filters=dict(community_filter=State('community-store','data'),
                 building_filter=State('building-store','data')),
    X_vals=dict(avg_five_year_rates=[State('mortgage_1','value'),
                State('mortgage_2','value'),
                State('mortgage_3','value'),
                State('mortgage_4','value')],
                CPI_TRIM=[State('inflation_1','value'),
                State('inflation_2','value'),
                State('inflation_3','value'),
                State('inflation_4','value')],
                CANRECDM=[State('recession_1', 'value'),
                State('recession_2', 'value'),
                State('recession_3', 'value'),
                State('recession_4', 'value')]
                ),

                ),prevent_initial_callback=True
)
def number_render(button,charts_collapse_is_open,filters,X_vals):
    if button>0:
        horizon = pd.period_range('2022-01-01', '2023-9-30', freq='q')
        horizon.name = '_date'
        fh_new = ForecastingHorizon(horizon, is_relative=False)
        for k, v in X_vals.items():
            # X_vals[k] = [0 if vl is None else vl for vl in v]
            for i in range(1, 4):
                X_vals[k].insert(0, v[0])

        X_forecast_single = pd.DataFrame(data=X_vals).set_index(horizon)
        X_forecast_single['CANRECDM'] = X_forecast_single['CANRECDM'].fillna(0).explode()
        X_forecast = pd.concat([X_forecast_single.assign(community=g[0], building_type=g[1]) for g in grps])
        X_forecast = X_forecast.reset_index().set_index(['community', 'building_type', '_date'])
        df_pred = forecaster.predict(fh_new, X=X_forecast)
        fig=make_plots(y,df_pred,filters.community_filter,filters.building_filter)
        return [html.Div(dcc.Graph(id='pred',figure=fig),style={'height':'65vh'}),not charts_collapse_is_open]
    # c = dash.ctx.args_grouping.inputs
    # if c.button.triggered:
    #     return not c.charts_collapse_is_open
    #
    #     # horizon = pd.period_range('2022-01-01', '2023-9-30', freq='q')
    #     # horizon.name = '_date'
    #     # fh_new = ForecastingHorizon(horizon, is_relative=False)
    #     # X_vals['CANRECDM']=[0 if not val else 1 for val in X_vals['CANRECDM']]
    #     # for k, v in X_vals.items():
    #     #     for i in range(1, 4):
    #     #         X_vals[k].insert(0, v[0])
    #     # X_forecast_single = pd.DataFrame(data=d).set_index(horizon)
    #     # X_forecast = pd.concat([X_forecast_single.assign(community=g[0], building_type=g[1]) for g in grps])
    #     # X_forecast = X_forecast.reset_index().set_index(['community', 'building_type', '_date'])
    else:
        return dash.no_update













@app.callback(Output("state", "children"), [Input("hoods", "hover_feature")])
def state_hover(feature):
    if feature is not None:
        return f"{feature['properties']['AREA_NAME']}"


if __name__ == '__main__':
    app.run_server(debug=False,port=8080)

