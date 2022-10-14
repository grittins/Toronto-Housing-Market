from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import json
import pandas as pd
json_file_path = r"../Resources/neighbourhoods.geojson"
df = pd.read_csv(r'../Resources/GTA_HomePrice_History.csv')
with open(json_file_path, 'r') as j:
    tdot = json.loads(j.read())

locations=[df.Community.unique()]


app = Dash(__name__)


app.layout = html.Div([
    html.H4('Select yer hood'),
    html.P("Select a candidate:"),
    dcc.RadioItems(
        id='candidate',
        options=["Joly", "Coderre", "Bergeron"],
        value="Coderre",
        inline=True
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"),
    Input("candidate", "value"))
def display_choropleth(candidate):
    df = px.data.election() # replace with your own data source
    geojson = tdot
    fig = px.scatter_geo(
        geojson=geojson,
        locations=locations,
        locationmode='geojson-id',
        featureidkey="properties.AREA_NAME")
    fig.update_geos(visible=True)
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig


app.run_server(debug=True)