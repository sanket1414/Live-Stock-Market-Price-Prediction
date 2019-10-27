import pandas_datareader.data as web
import datetime
import time
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pickle
from yahoo_fin import stock_info

with open('apple_model.pkl', 'rb') as f:
	apple_model = pickle.load(f)
with open('google_model.pkl', 'rb') as f:
	google_model = pickle.load(f)
with open('microsoft_model.pkl', 'rb') as f:
	microsoft_model = pickle.load(f)
with open('facebook_model.pkl', 'rb') as f:
	facebook_model = pickle.load(f)

app = dash.Dash()

# Company list
company_list = {'AAPL': 'Apple', 'GOOG': 'Google', 'MSFT': 'Microsoft', 'FB': 'Facebook'}

app.layout = html.Div(children = [
		html.H1("Stock Price Prediction", style={'color': '#7FDBFF', 'textAlign': 'center', 'fontSize': 60, 'fontFamily': 'courier'}), 
		html.Br(),
		dcc.Dropdown(
		    id = 'input',
		    options = [
		        {'label': 'Apple', 'value': 'AAPL'},
		        {'label': 'Google', 'value': 'GOOG'},
		        {'label': 'Microsoft', 'value': 'MSFT'},
				{'label': 'Facebook', 'value': 'FB'}
		    ],
		    value = 'AAPL',
		    style = {'backgroundColor': '#111111'}
		),
		html.Br(),
		html.Div(id = 'output-graph'),
	],
	style = {'margin': '-8px',
	         'paddingLeft': '50px',
	         'paddingRight': '50px',
	         'backgroundColor': '#111111',
             'height': '100vh',
             'overflow': 'hidden'}
)

@app.callback(
    Output(component_id = 'output-graph', component_property = 'children'),
    [Input(component_id = 'input', component_property = 'value')]
)
def update_value(company):
    if (company == "AAPL"):
        model = apple_model
    elif (company == "GOOG"):
        model = google_model
    elif (company == "MSFT"):
        model = microsoft_model
    elif (company == "FB"):
        model = facebook_model
    starting_program = time.time()
    start = datetime.datetime.now() + datetime.timedelta(-1070)
    end = datetime.datetime.now()
    df = web.DataReader(company, 'yahoo', start, end)
    df.reset_index(inplace = True)
    df.set_index("Date", inplace = True)
    
    # current_stock_value = stock_info(company.lower())
    current_stock_value = stock_info.get_live_price(company.lower())

    dates = list(df.index[30:])
    future_dates = []
    for i in range(30):
        future_dates.append(datetime.datetime.now() + datetime.timedelta(i))
    x_forecast = np.reshape(np.array(df['Adj Close']), (-1, 1))
    predictions = model.predict(x_forecast)
    old_predictions = predictions[:-30]
    future_predictions = predictions[-30:]
    print("Total time taken:", time.time() - starting_program)

    return dcc.Graph(
        id = 'example-graph',
        figure = {
            'data': [
                {'x': df.index[30:], 'y': df.Close[30:], 'type': 'line', 'name': 'Actual Value', 'line': dict(color='firebrick', width=3)},
                {'x': dates, 'y': old_predictions, 'type': 'line', 'name': 'Old Predictions', 'line': dict(color='royalblue', width=3)},
                {'x': future_dates, 'y': future_predictions, 'type': 'dashed-line', 'name': 'Future Predictions', 'line': dict(color='gray', width=2)},
            ],
            'layout': {
                'title': {
                	'text': company_list[company],
                	'font': {
						'size': 50,
						'color': '#7FDBFF'
					},
					'x': 0.05,
					'y': 1.4,
					'xanchor': 'left',
					'yanchor': 'bottom',
                },
                'annotations': [
                	{
						'text': str(round(df.iloc[-1]['Close'], 2)),
						'font': {
							'size': 25,
							'color': '#7FDBFF',
						},
						'showarrow': False,
						'align': 'center',
						'x': 0,
						'y': 1.15,
						'xref': 'paper',
						'yref': 'paper',
			  		},
			  		{
						'text': 'USD',
						'font': {
							'size': 12,
							'color': '#7FDBFF',
						},
						'showarrow': False,
						'align': 'center',
						'x': 0.06,
						'y': 1.11,
						'xref': 'paper',
						'yref': 'paper',
			  		},
			  		{
						'text': str(round(df.iloc[-1]['Close'], 2)),
						'font': {
							'size': 18,
							'color': '#7FDBFF',
						},
						'showarrow': False,
						'align': 'center',
						'x': 0.7,
						'y': 1.15,
						'xref': 'paper',
						'yref': 'paper',
			  		},
			  		{
						'text': 'PREV',
						'font': {
							'size': 10,
							'color': '#7FDBFF',
						},
						'showarrow': False,
						'align': 'center',
						'x': 0.7,
						'y': 1.07,
						'xref': 'paper',
						'yref': 'paper',
			  		},
			  		{
						'text': str(round(df.iloc[-1]['Open'], 2)),
						'font': {
							'size': 18,
							'color': '#7FDBFF',
						},
						'showarrow': False,
						'align': 'center',
						'x': 0.76,
						'y': 1.15,
						'xref': 'paper',
						'yref': 'paper',
			  		},
			  		{
						'text': 'OPEN',
						'font': {
							'size': 10,
							'color': '#7FDBFF',
						},
						'showarrow': False,
						'align': 'center',
						'x': 0.76,
						'y': 1.07,
						'xref': 'paper',
						'yref': 'paper',
			  		},
			  		{
						'text': str(round(df.iloc[-1]['Volume'], 2)),
						'font': {
							'size': 18,
							'color': '#7FDBFF',
						},
						'showarrow': False,
						'align': 'center',
						'x': 0.84,
						'y': 1.15,
						'xref': 'paper',
						'yref': 'paper',
			  		},
			  		{
						'text': 'VOLUME',
						'font': {
							'size': 10,
							'color': '#7FDBFF',
						},
						'showarrow': False,
						'align': 'center',
						'x': 0.84,
						'y': 1.07,
						'xref': 'paper',
						'yref': 'paper',
			  		}
			  	],
			  	'legend': {'x': 0, 'y': 1},
                'plot_bgcolor': '#111111',
                'paper_bgcolor': '#111111',
                'font': {
                    'color': '#7FDBFF'
                }
            }
        }
    )

if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
