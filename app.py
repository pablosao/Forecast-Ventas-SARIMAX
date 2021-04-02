"""
 Aplicación elaborada en Dash con la finalidad de realizar el análisis del algoritmo SERMIAX para
 el cálculo de un forecast de ventas de una empresa que comercializa productos de ferretería

:author: Pablo Sao
:date: 30 de mayo de 2020
"""
import io
import base64
import subprocess
import sys
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objs as go
import plotly.express as px
import warnings
import itertools
import numpy as np
import pandasql as psql
import ast
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

warnings.filterwarnings("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuración de Pandas

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width',85)


"""
    VARIABLES Generales
"""
# Logo de Solution Design
file_logo = 'recursos/logo.png' 
SD_LOGO = base64.b64encode(open(file_logo, 'rb').read())

"""
    Datos de Ventas
"""
DATA = pd.read_excel('recursos/ventas.xlsx')
DATA['fecha'] = pd.to_datetime(DATA['fecha'])
DATA = DATA.sort_values(by=['fecha'])

LISTA_PRODUCTOS = DATA[['codigo_producto','Producto']].copy(deep=False)

# Listado de Productos en el archivo cargado
LISTA_PRODUCTOS = LISTA_PRODUCTOS.drop_duplicates()

# Primer producto del Listado
#PRIMER_PRODUCTO = DATA['codigo_producto'].iloc[0]
PRIMER_PRODUCTO = 15214
PRIMER_MODELO = 'additive'


def getNombreProducto(ventas,codigo_producto):
    consulta = """
                    SELECT distinct
                        substr( Producto, 9, 25 ) as producto
                    FROM
                        ventas
                    WHERE
                        codigo_producto = {0}
                               """.format(codigo_producto)
        # Obtenemos data de un producto
    
    _nombre_producto = psql.sqldf(consulta, locals())

    return _nombre_producto['producto'].iloc[0]

def filtraDataProductos(codigo_producto):
    _ventas = DATA.copy(deep=False)
    consulta = """
                    SELECT 
                         fecha
                        ,total_vendido
                    FROM
                        _ventas
                    WHERE
                        codigo_producto = {0}
                    GROUP BY fecha
                    ORDER BY
                        fecha ASC
                               """.format(codigo_producto)
    # Obtenemos y retornamos la data de un producto
    return psql.sqldf(consulta, locals())

def armaTrazoSeries(ventas):
    
    trace_producto = []
    _productos = ventas['codigo_producto'].unique()

    _visible='legendonly'

    for producto in _productos:
        consulta = """
                    SELECT 
                         fecha
                        ,avg(total_vendido) as total_vendido
                    FROM
                        ventas
                    WHERE
                        codigo_producto = {0}
                    GROUP BY fecha
                    ORDER BY
                        fecha ASC
                               """.format(producto)
        # Obtenemos data de un producto
        _Productos_Serie = psql.sqldf(consulta, locals())

        _Producto = getNombreProducto(ventas,producto)

        if(PRIMER_PRODUCTO == producto):
            _visible = True
        else:
            _visible='legendonly'

        #Realizamos trazado de la serie de tiempo del producto
        custom_trace_producto = go.Scatter(x=_Productos_Serie['fecha'], y=_Productos_Serie['total_vendido'], 
                                           name='{0}'.format(_Producto), visible=_visible)
        
        # Agregamos el trazo a una lista
        trace_producto.append(custom_trace_producto)
        
    
    cLayout = go.Layout(title='Serie de Tiempo de Productos',
                       # Same x and first y
                        xaxis_title = 'Fecha',
                        yaxis_title = 'Ventas (Q)'
                       )

    return (
                dcc.Graph(id='Serie-Tiempo-Productos', figure={
                                                                'data': trace_producto,
                                                                'layout':cLayout
                        })
            )

def generate_table(dataframe, max_rows=10):
        
    return dbc.Table(
        (
            # Header
            [html.Tr([html.Th(col) for col in dataframe.columns])] +

            # Body
            [html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))]
        ),bordered=True, hover=True,responsive=True,striped=True
    )


PARAMETROS = pd.read_excel('recursos/parametros.xlsx')

# Estilo externo
external_stylesheets = [dbc.themes.BOOTSTRAP] #['recursos/bWLwgP.css']

# Creando aplicacion de Dash
app = dash.Dash(__name__,
    external_stylesheets=external_stylesheets,
    title='Forecast Ventas',
    update_title="Actualizando...",
    assets_folder='recursos',
    assets_url_path='recursos',
)

server = app.server

app.layout = html.Div(children=[
    
    html.Br(),
    
    # Encabezado de la Pagina
    dbc.Row([
        dbc.Col(
            html.Img(src='data:image/png;base64,{}'.format(SD_LOGO.decode()),style={'width': '125px'}),
            width={"size": 2, "order": 1, "offset": 1},
        ),
        dbc.Col((
            
            html.Br(),
            html.H1(children='Forecast de Ventas'),
            html.P(children='Validación del algoritmo SARIMAX para predicción de ventas'),
        
        ),width={"size": 6, "order": 2})
    ]),

    html.Br(),

    # Cuerpo de la Pagina
    dbc.Row([
        dbc.Col((
            dbc.Tabs([
                # Tab de Análisis
                dbc.Tab((
                    html.Br(),

                    dbc.Row([
                        dbc.Col(
                            html.Div(generate_table(PARAMETROS[['Producto','Orden','Estacionalidad','Ruido (AIC)']]))
                        ),
                        dbc.Col((
                            dbc.Jumbotron([
                                dbc.Container([
                                    html.H1("Algorítmo SARIMAX", className="display-3"),
                                    html.P('''SARIMAX (Seasonal Auto-Regressive Integrated Moving Average with eXogenous model) 
                                              es un modelo basado en ARIMA. Donde la S hace mención del modelado de patrones 
                                              recurrentes según su estacionalidad. Y la X se refiere a una variable externa, que 
                                              puede llegar a representar factores externos que influyen en la información.
                                              ''', className="lead"),
                                              
                                    html.Br(),

                                    html.P('''Esta toma la forma (p, d, q) x (P, D, Q, s), en la que los primeros tres son la 
                                              versión estacional de ARIMA. La letra mayúscula P, hace referencia al orden de auto 
                                              regresión. La letra mayúscula D, es el número de integraciones estacionales y la 
                                              letra mayúscula Q, hace referencia al orden de medias móviles. Donde las letras 
                                              minúsculas son sus equivalentes, para los valores no estacionales. La variable s, 
                                              hace referencia al ciclo de los datos a evaluar; es decir, el número de periodos que 
                                              deben transcurrir antes de que aparezca nuevamente la tendencia.
                                              ''', className="lead")


                                ],fluid=True)
                            ])
                        )),
                    ]),

                    html.Br(),

                    html.Div(armaTrazoSeries(DATA)),
                    

                ), tab_id="tab-analisis-forecast",label="Análisis Forecast SARIMAX", label_style={"color": "#ea9410"},disabled=False),
                
                # Tab del Forecast con SERIMAX
                dbc.Tab((
                    dbc.CardBody((
                        # Label de los filtros
                        dbc.Row([
                            # Labels para Filtros
                            dbc.Col( html.Div(children='Producto:') ),
                            
                        ]),
                        # lookups para filtrado de informacion
                        dbc.Row([
                            # Filtro de productos
                            dbc.Col(
                                dcc.Dropdown(
                                    id='producto_forecast',
                                    options=[
                                        {'label':i[1]['Producto'],'value':i[1]['codigo_producto']} for i in LISTA_PRODUCTOS.iterrows()
                                    ],
                                    multi=False,
                                    value=PRIMER_PRODUCTO,
                                    placeholder='Producto para Analizar'
                                )                                
                            ),
                        ]),

                       

                        html.Br(),

                        html.Div(id='serie_forecast')

                    ))
                ), tab_id="tab-forecast",label="Forecast SARIMAX", label_style={"color": "#00AEF9"},disabled=False)

            ],active_tab="tab-forecast")
        ),width={"size": 10, "offset": 1})
        
    ])

    
])


@app.callback(
    dash.dependencies.Output('serie_forecast', 'children'),
    [dash.dependencies.Input('producto_forecast', 'value')])
def displayForecast(producto_evaluar):
    
    modelo_evaluar = 'additive'
    tendencia_visible = 'legendonly'

    _ventas = DATA.copy(deep=False)
    
    
    
    consulta = """
                    SELECT 
                         fecha
                        ,avg(total_vendido) as total_vendido
                    FROM
                        _ventas
                    WHERE
                        codigo_producto = {0}
                    GROUP BY fecha
                    ORDER BY
                        fecha ASC
                               """.format(producto_evaluar)

    _Productos_Serie = psql.sqldf(consulta, locals())

    #_Productos_Serie['fecha'] = _Productos_Serie['mes'].map(str) + '-' + _Productos_Serie['anio'].map(str)
    
    _Producto = getNombreProducto(DATA,producto_evaluar)

    # Si el dato es de tipo 'NoneType' deja el valor inicial por defecto
    # de lo contrario asigna el modelo seleccionado
    #if(isinstance( dropdown_modelo, str )):
    #    modelo_evaluar = dropdown_modelo
    #    tendencia_visible = True

    _frecuencia = int(_Productos_Serie['total_vendido'].count()/2)

    _data_forecast = _Productos_Serie[['fecha','total_vendido']].copy(deep=False)
    
    # Colocamos la fecha como indice
    _data_forecast = _data_forecast.set_index('fecha')
    
    
    
    descomposicion = seasonal_decompose(_data_forecast, period=_frecuencia, model=modelo_evaluar)
    
    data_trend = pd.DataFrame(descomposicion.trend)
    data_seasonal = pd.DataFrame(descomposicion.seasonal)
    data_resid = pd.DataFrame(descomposicion.resid)
    
    cLayout = go.Layout(title='Tendencia de Ventas Promedio de "{0}"'.format(_Producto),
                       # Same x and first y
                        xaxis_title = 'Fecha',
                        yaxis_title = 'Ventas (Q)',
                        height=700
                       )

    trace1 = go.Scatter(x=_Productos_Serie.fecha, y=_Productos_Serie.total_vendido,name='Ventas Reales (Mensual)')
    trace2 = go.Scatter(x=data_trend.index, y=data_trend.trend,name='Tendencia de Venta',
                        visible='legendonly')
    trace3 = go.Scatter(x=data_seasonal.index, y=data_seasonal.seasonal,
                        name='Tendencia Estacional de Venta',visible=tendencia_visible)
    trace4 = go.Scatter(x=data_resid.index, y=data_resid.resid,name='Residuos',
                        visible='legendonly')
    
    """
        FORECAST
    """

    # Cargando parametros
    _parametros = PARAMETROS[PARAMETROS['Código Producto'] == producto_evaluar].copy(deep=False)

    _orden = ast.literal_eval(_parametros['Orden'].iloc[0])
    _estacionalidad = ast.literal_eval(_parametros['Estacionalidad'].iloc[0])
    #_ruido = _parametros['Ruido (AIC)'].iloc[0]

        
    mod = sm.tsa.statespace.SARIMAX(_data_forecast,
                                    order=_orden,
                                    seasonal_order= _estacionalidad,
                                    trend='ct',
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    measurement_error=True)

    results = mod.fit()

    

    results.plot_diagnostics(figsize=(12, 10))

    buf = io.BytesIO() # in-memory files
    plt.savefig(buf, format = "png") # save to the above file object
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    plt.close()
    
    
    pred = results.get_prediction(start='2009-01-01 00:00:00.000000', dynamic=False)
    pred_ci = pred.conf_int()
    
    
    pred_ci['diferencia'] = pred_ci.apply(lambda x: (x['upper total_vendido'] + x['lower total_vendido'])/2, axis=1)

    trace5 = go.Scatter(x=(pred_ci.index).to_list(), y=pred_ci['upper total_vendido'], name='Margen Positivo',
                        showlegend=True, fill='tozeroy', line=dict(color='rgb(155,155,155)'))

    trace6 = go.Scatter(x=(pred_ci.index).to_list(), y=pred_ci['lower total_vendido'],name='Margen Negativo', #showlegend=True
                        visible='legendonly',fill='tozeroy',line=dict(color='rgb(155,155,155)'))


    trace7 = go.Scatter(x=(pred_ci.index).to_list(), y=pred_ci['diferencia'], name='Prediccion',
                        showlegend=True,line=dict(color='rgb(255,102,0)'))
    
    return (
        html.Div(
            dcc.Graph(id='graph', figure={
                'data': [trace1,trace2,trace3,trace4,trace5,trace6,trace7],
                'layout': cLayout
            }),
        ),
        html.Br(),
        html.Div(html.Img(src='data:image/png;base64,{}'.format(data) ) )
    )

    
if __name__ == '__main__':
    app.run_server()