"""
 Aplicación elaborada en Dash con la finalidad de realizar el análisis del algoritmo SERMIAX para
 el cálculo de un forecast de ventas de una empresa que comercializa productos de ferretería

:author: Pablo Sao
:date: 30 de mayo de 2020
"""

import sys
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import itertools
import numpy as np
import pandasql as psql

warnings.filterwarnings("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuración de Pandas

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width',85)



"""
    Datos de Ventas
"""
DATA = pd.read_excel('recursos/ventas.xlsx')



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




"""
    EVALUA FORECAST
"""
def analizaParametros(ventas):
    
    Data_Forecast = pd.DataFrame(columns=['Código Producto','Producto','Orden', 'Estacionalidad', 'Ruido (AIC)'])

    _data = ventas[['fecha','total_vendido','codigo_producto']].copy(deep=False)
    _data = _data.set_index('fecha')

    _productos = _data['codigo_producto'].unique()

    for codigo_producto in _productos:
        
        print("Calculando producto: {0}\n".format(codigo_producto))

        _data_producto = _data[_data['codigo_producto'] == codigo_producto].copy(deep=False)
        _data_producto = _data_producto.drop('codigo_producto',axis=1)

        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        df = pd.DataFrame(columns=['Orden', 'Estacionalidad', 'Ruido (AIC)'])
        
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:

                    mod = sm.tsa.statespace.SARIMAX(_data_producto,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    trend='ct',
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False,
                                                    measurement_error=True)
                    results = mod.fit()
                    df = df.append({'Orden': param, 'Estacionalidad': param_seasonal, 'Ruido (AIC)': results.aic},
                                ignore_index=True)
                    #print('ARIMA{0}x{1}12 - AIC:{2}'.format(param, param_seasonal, results.aic))
                except Exception as e:
                    print("error: {e}".format(e))
                    continue

        df = df.sort_values(by=['Ruido (AIC)']).head(5)

        var_order_param =  df['Orden'].tolist()[0]
        var_seasonal_order = df['Estacionalidad'].tolist()[0]
        var_ruido = df['Ruido (AIC)'].tolist()[0]
    
        Data_Forecast = Data_Forecast.append({'Orden': var_order_param,
                                              'Estacionalidad': var_seasonal_order, 
                                              'Ruido (AIC)': var_ruido,
                                              'Código Producto': codigo_producto,
                                              'Producto': getNombreProducto(ventas,codigo_producto)},
                                ignore_index=True)

    return Data_Forecast



# VARIABLES DE FORECAST

"""
    Calculo de parametros del algoritmo SERIMAX, se exporta a Excel para no calcular
    Se recomienda agregar una opcion para recalcular si es necesario.
"""
print('Iniciando Analisis..\n')
PARAMETROS = analizaParametros(DATA)
PARAMETROS.to_excel('recursos/parametros.xlsx',index=False)
print("\n\n")
print("Data Exportada....")
print("\n\n")
PARAMETROS = pd.read_excel('recursos/parametros.xlsx')

print(PARAMETROS.head(15))

