from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required  # login autenticación
from django.utils.safestring import mark_safe
from django.http import HttpResponse
from django.utils.timezone import get_current_timezone
from django.contrib.auth.models import User
from django.core.cache import cache
from django.core.management import execute_from_command_line
import sys
from .models import Infoclientes, Datostablaregresion
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import model_from_json
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from joblib import load
import pytz
#---------------------------------
import itertools
from itertools import product
import plotly.graph_objects as go

import json
from django.utils import timezone
from datetime import date
from datetime import datetime


@login_required(redirect_field_name='login')
def cargue_modelos(request):
    return render(request, 'fin_manager/cargue_modelos.html')


def cargar_tabla_en_memoria(fecha_inicio, fecha_fin ):
    # Carga los datos de la tabla en memoria
    bdinfoclientes =  pd.DataFrame(list(Datostablaregresion.objects.filter(fecha__gt=fecha_inicio, fecha__lt=fecha_fin).values()))
    # Modifico nombres del  dataframe
    bdinfoclientes['fecha'] = pd.to_datetime(bdinfoclientes['fecha'], errors='coerce')
    bdinfoclientes.rename(columns={'idcliente': 'idCliente',
                                   'dia':'Dia',
                                   'mes':'Mes',
                                   'anio': 'Año',
                                   'hora': 'Hora',
                                   'minuto': 'Minuto',
                                   'energia_activa': 'Active_energy',
                                   'energia_reactiva': 'Reactive_energy',
                                   'anomalia': 'Anomalia'
                                   },
        inplace=True)

    # Almacena los datos en la caché de Django
    cache.set('bdinfoclientes', bdinfoclientes)


# En algún lugar de tu código donde necesites acceder a los datos de la tabla

def obtener_datos_de_la_tabla_desde_memoria(fecha_inicio,fecha_fin):
    bdinfoclientes = cache.get('bdinfoclientes')
    if  bdinfoclientes is None:
        # Si los datos no están en caché, cargarlos en memoria
        cargar_tabla_en_memoria(fecha_inicio,fecha_fin)
        # Vuelve a intentar obtener los datos desde la caché
        bdinfoclientes = cache.get('bdinfoclientes')
    return bdinfoclientes




@login_required(redirect_field_name='login')
def grafica1(request):
    return render(request, 'fin_manager/grafica_01.html')
@login_required(redirect_field_name='login')
def grafica2(request):
    return render(request, 'fin_manager/grafica_02.html')
@login_required(redirect_field_name='login')
def grafica3(request):
    return render(request, 'fin_manager/grafica_03.html')
@login_required(redirect_field_name='login')
def grafica4(request):
    return render(request, 'fin_manager/grafica_04.html')
@login_required(redirect_field_name='login')
def grafica5(request):
    return render(request, 'fin_manager/grafica_05.html')
@login_required(redirect_field_name='login')
def grafica6(request):
    return render(request, 'fin_manager/grafica_06.html')



@login_required(redirect_field_name='login')
def home(request):
    #Definición parámetros iniciales
    context = {}
    fecha_inicio = date(2022, 11, 1)
    fecha_fin = date(2023, 8, 10)
    #Definición parámetros fijos (Fecha)
    fecha_maxima='2023-06-01'



    # --------Método POST --------
    if request.method == 'POST':
        #try:
        df001 = pd.DataFrame(list(Datostablaregresion.objects.values()))
        df001['fecha'] = pd.to_datetime(df001['fecha'], errors='coerce')

        # listas de selección de filtros
        context['Colum1_List_views0'] = sorted(list(df001['idcliente'].unique()), reverse=False)
        context['Colum2_List_views0'] = sorted(list(df001['anomalia'].unique()), reverse=False)
        context['Colum3_List_views0'] = list(df001['sectoreconomico'].unique())
        context['Colum4_List_views0'] = sorted(list(df001['anio'].unique()), reverse=False)
        context['Colum5_List_views0'] = sorted(list(df001['fecha'].unique()), reverse=False)
        context['Colum6_List_views0'] = sorted(list(df001['fecha'].unique()), reverse=False)

        # Opción 2 método POST () filtros para las gráficas----------------
        Colum10 = request.POST.get('Colum10', None)
        Colum20 = request.POST.get('Colum20', None)  #
        Colum30 = request.POST.get('Colum30', None)  #
        Colum40 = request.POST.get('Colum40', None)  #
        Colum50 = request.POST.get('Colum50', None)  #
        Colum60 = request.POST.get('Colum60', None)  #

       # Convertir la cadena a un objeto datetime
        if Colum10 != 'Todos' or Colum20 != 'Todos' or Colum30 != 'Todos' or Colum40 != 'Todos' or Colum50 != 'Todos' or Colum60 != 'Todos':
            if Colum10 != 'Todos' and Colum10 is not None and Colum10 != '':
                df001 = df001[df001['idcliente'] == int(Colum10)]
            if Colum20 != 'Todos' and Colum20 is not None and Colum20 != '':
                df001 = df001[df001['anomalia'] == float(Colum20)]
            if Colum30 != 'Todos' and Colum30 is not None and Colum30 != '':
                df001 = df001[df001['sectoreconomico'] == str(Colum30)]
            if Colum40 != 'Todos' and Colum40 is not None and Colum40 != '':
                df001 = df001[df001['anio'] == int(Colum40)]


            # Pasar de string a tipo date
            Colum50= datetime.strptime(Colum50,  "%Y-%m-%d")
            # Formatear la fecha al formato deseado
            Colum50 = Colum50.strftime("%b. %d, %Y")

            if Colum50 != fecha_inicio and Colum50 is not None and Colum50 != '':
                df001 = df001[df001['fecha'] >= Colum50]
            else:
                df001 = df001[df001['fecha'] >= fecha_inicio]

            # Pasar de string a tipo date
            Colum60= datetime.strptime(Colum60,  "%Y-%m-%d")
            # Formatear la fecha al formato deseado
            Colum60 = Colum60.strftime("%b. %d, %Y")

            if Colum60 != fecha_fin and Colum60 is not None and Colum60 != '':
                df001 = df001[df001['fecha'] <= Colum60]
            else:
                df001 = df001[df001['fecha'] <= fecha_fin]

            # Valores de filtros
            context['Colum1_Select0'] = Colum10
            context['Colum2_Select0'] = Colum20
            context['Colum3_Select0'] = Colum30
            context['Colum4_Select0'] = Colum40
            context['Colum5_Select0'] = Colum50
            context['Colum6_Select0'] = Colum60

            # Genero los datos para las gráficas
            df001.rename(columns={'idcliente': 'idCliente','dia': 'Dia','mes': 'Mes','anio': 'Año','hora': 'Hora','minuto': 'Minuto',
                                  'energia_activa': 'Active_energy','energia_reactiva': 'Reactive_energy','anomalia': 'Anomalia'},inplace=True)

            # Dar formato a la fecha
            #df001['fecha'] = pd.to_datetime(df001['fecha'], errors='coerce')
            #df001['fecha'] = df001['fecha'].dt.strftime("%B. %d, %Y")

            df001['fecha'] = pd.to_datetime(df001['fecha'], errors='coerce')


            fig_consumo_energia = mark_safe(comportamiento_consumo_energia(df001))
            fig_energia_activa = mark_safe(regresion_energia_activa(df001))
            fig_energia_reactiva = mark_safe(regresion_energia_reactiva(df001))
            fig_anomalias_01 = mark_safe(fig_anomalias(df001))
            fig_clientes_anomalias_01 = mark_safe(fig_clientes_anomalias(df001))
            fig_radar_01 = mark_safe(fig_radar(df001))

            context['fig_consumo_energia'] = fig_consumo_energia
            context['fig_energia_activa'] = fig_energia_activa
            context['fig_energia_reactiva'] = fig_energia_reactiva
            context['fig_anomalias_01'] = fig_anomalias_01
            context['fig_clientes_anomalias_01'] = fig_clientes_anomalias_01
            context['fig_radar_01'] = fig_radar_01

            # Pasar de string a tipo date
            Colum50= datetime.strptime(Colum50,"%b. %d, %Y")
            # Formatear la fecha al formato deseado str
            Colum50 = Colum50.strftime("%Y-%m-%d")

            # Pasar de string a tipo date
            Colum60= datetime.strptime(Colum60,"%b. %d, %Y")
            # Formatear la fecha al formato deseado str
            Colum60 = Colum60.strftime("%Y-%m-%d")



            # Valores de filtros
            context['Colum5_Select0'] = Colum50
            context['Colum6_Select0'] = Colum60
            context['Colum6_max'] = fecha_maxima

            return render(request, 'fin_manager/home.html', context)

        #else:
        #    return redirect('/dashboard1')  # Change  your desired URL

        #return render(request, 'fin_manager/home.html', context)
        #except:
        #    return redirect('/dashboard1')  # Change  your desired URL



    #Llamo función que lee el dataframe

    df003_anomalias_re1 = obtener_datos_de_la_tabla_desde_memoria(fecha_inicio, fecha_fin)
    #fecha_inicio = fecha_inicio.strftime('%Y-%m-%d')


    context['fig_consumo_energia']= mark_safe(comportamiento_consumo_energia(df003_anomalias_re1))
    context['fig_energia_activa']= mark_safe(regresion_energia_activa(df003_anomalias_re1))
    context['fig_energia_reactiva']=mark_safe(regresion_energia_reactiva(df003_anomalias_re1))
    context['fig_anomalias_01']= mark_safe(fig_anomalias(df003_anomalias_re1))
    context['fig_clientes_anomalias_01']= mark_safe(fig_clientes_anomalias(df003_anomalias_re1))
    context['fig_radar_01']= mark_safe(fig_radar(df003_anomalias_re1))

    #Genero las opciones de selección por Columna
    df001 = pd.DataFrame(list(Datostablaregresion.objects.values()))
    context['Colum1_List_views0'] =  sorted(list(df003_anomalias_re1['idCliente'].unique()), reverse=False)
    context['Colum2_List_views0'] =  sorted(list(df003_anomalias_re1['Anomalia'].unique()), reverse=False)
    context['Colum3_List_views0'] = list(df003_anomalias_re1['sectoreconomico'].unique())
    context['Colum4_List_views0'] =  sorted(list(df001['anio'].unique()), reverse=False)
    context['Colum5_List_views0'] =  sorted(list(df001['fecha'].unique()), reverse=False)
    context['Colum6_List_views0'] =  sorted(list(df001['fecha'].unique()), reverse=False)

    # Valores de filtros
    context['Colum5_Select0'] = fecha_inicio.strftime('%Y-%m-%d')
    context['Colum6_Select0'] = fecha_fin.strftime('%Y-%m-%d')

    return render(request, 'fin_manager/home.html', context)


def valores_filtros():
    Colum1_List_views0 =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                         27, 28, 29, 30]
    Colum2_List_views0 =[0.0, 1.0]
    Colum3_List_views0 = ['Elaboración de cacao y chocolate y de productos de confitería',
                          'Cultivo de Árboles Frutales y Nueces',
                          'Cultivo de otros frutos y nueces de árboles y arbustos',
                          'Cultivo de hortalizas y melones, raíces y tubérculos',
                          'Captación, tratamiento y distribución de agua',
                          'Cultivo de Hortalizas',
                          'Venta al por mayor de metales y minerales metalíferos']
    Colum4_List_views0 = [2021, 2022, 2023, 2024]

    Colum5_List_views0 = []
    Colum6_List_views0 = []

    return Colum1_List_views0, Colum2_List_views0, Colum3_List_views0, Colum4_List_views0, Colum5_List_views0, Colum6_List_views0


@login_required(redirect_field_name='login')
def carguedatos(request):
    if request.method == 'POST':
        try:
            archivo = request.FILES['path']
            if archivo.name is not None:
                # Leo la información del Excel
                df1 = pd.read_excel(archivo, sheet_name="Hoja1", skiprows=0, na_values="-")
                # Determino las tablas de información que se van a leer
                df2 = df1.iloc[1:500000, 0:7]  # Selecciono  el rango de los datos de mi df
                df2 = df2.fillna(0)  # reemplazo los valores "nan" por cero "0"

                #fechahora = datetime.now()  # fecha y hora actual
                fechahoraregistro=datetime.now(tz=get_current_timezone())
                #Obtengo el nombre del usuario
                usuario = str(User.objects.get(username=request.user.username))

                for fila in range(df2.shape[0]):
                    try:
                        #Obtener indice
                        id_max = (Infoclientes.objects.values('id').order_by('-id').first())  # obtendo de la base de datos el último indice en un diccionario
                        id_max_1 = int(id_max.get('id'))  # obtengo el indice del diccionario y lo convierto en número
                        id_max_3 = int(id_max_1) + 1
                    except:
                        id_max_3 = 1
                        print("id_max_3_except:", id_max_3)

                    GuardarModelo = Infoclientes(
                        id= id_max_3,
                        fecha=df2.iloc[fila, 0],
                        active_energy=df2.iloc[fila, 1],
                        reactive_energy=df2.iloc[fila, 2],
                        voltaje_fa=df2.iloc[fila, 3],
                        voltaje_fc=df2.iloc[fila, 4],
                        idcliente=df2.iloc[fila, 5],
                        sectoreconomico=df2.iloc[fila, 6],
                        anomalia=df2.iloc[fila, 7],
                        usuario=usuario,
                        fechahora=fechahoraregistro,
                        )
                    GuardarModelo.save()
                return redirect('/')
        except:
            return redirect('/')

    return render(request, 'fin_manager/carguedatos.html')



@login_required(redirect_field_name='login')
def generartablaprediccion(request):

    # Importamos base de datos de información histórica

    df = pd.DataFrame(list(Infoclientes.objects.values()))

    #-------------------------------------Cargue de modelos Regresión-------------------------------------------------------------


    link_arquitectura_json_regresion_ER='static/modelos/modelo_energiaReactiva.json'
    # Cargar la arquitectura del modelo desde el archivo JSON
    with open(link_arquitectura_json_regresion_ER, "r") as archivo:
        modelo_cargado_ER = archivo.read()

    # Reconstruir el modelo desde la arquitectura cargada
    modeloER = model_from_json(modelo_cargado_ER)

    link_pesos_del_modelo_regresion_ER='static/modelos/modeloER_pesos.weights.h5'
    # Cargar los pesos del modelo desde el archivo HDF5
    modeloER.load_weights(link_pesos_del_modelo_regresion_ER)

    link_arquitectura_del_modelo_regresion_json_EA="static/modelos/modelo_energiaActiva.json"
    # Cargar la arquitectura del modelo desde el archivo JSON
    with open(link_arquitectura_del_modelo_regresion_json_EA, "r") as archivo:
        modelo_cargado_EA = archivo.read()

    # Reconstruir el modelo desde la arquitectura cargada
    modeloEA = model_from_json(modelo_cargado_EA)

    link_pesos_del_modelo_regresion_EA= "static/modelos/modeloEA_pesos.weights.h5"
    # Cargar los pesos del modelo desde el archivo HDF5
    modeloEA.load_weights(link_pesos_del_modelo_regresion_EA)

    # Cargar scaler utilizado en los datos para la predicción
    link_scaler_reg='static/modelos/scalerReg.joblib'
    from joblib import load
    scaler = load(link_scaler_reg)

    #----------Ajuste DataFrame base Histórica para unirlo al dataframe de predicción de consumo energía activa y reactiva------------

    # definimos los nombres de las Columnas
    df.rename(columns={'idcliente': 'idCliente', 'active_energy': 'Active_energy', 'reactive_energy': 'Reactive_energy'},
        inplace=True)

    df['fecha'] = pd.to_datetime(df['fecha'])
    df['Dia'] = df['fecha'].dt.day
    df['Mes'] = df['fecha'].dt.month
    df['Año'] = df['fecha'].dt.year
    df['Hora'] = df['fecha'].dt.hour
    df['Minuto'] = df['fecha'].dt.minute

    # Selecciono columnas de interés
    df00 = df.loc[:,['fecha', 'Active_energy', 'Reactive_energy', 'idCliente', 'sectoreconomico', 'Dia', 'Mes', 'Año', 'Hora','Minuto']]

    # Adiciono columna para facilidad de filtro
    df00['prediccion_consumo_energia'] = 0

    #------------------------------Generar Dataframe para aplicar predicción----------------------------------------------------------

    lista_Clientes = df['idCliente'].unique()
    Fecha_inicio = df['fecha'].max() + pd.Timedelta(hours=1)
    Fecha_fin = pd.Timestamp('2024-12-31 23:00:00', tz='UTC')


    # Obtener todas las combinaciones de las listas
    combinaciones = list(product(lista_Clientes, pd.date_range(Fecha_inicio, Fecha_fin, freq='H')))

    # Genero Un dataframe
    df1 = pd.DataFrame(combinaciones, columns=['idCliente', 'fecha'])

    # Creación de un DataFrame con todas las combinaciones posibles de las categorías de las variables predictoras
    df1['Dia'] = df1['fecha'].dt.day
    df1['Mes'] = df1['fecha'].dt.month
    df1['Año'] = df1['fecha'].dt.year
    df1['Hora'] = df1['fecha'].dt.hour
    df1['Minuto'] = df1['fecha'].dt.minute

    # Normalización de las variables predictoras
    df_scaled = scaler.transform(df1.loc[:, ['idCliente', 'Dia', 'Mes', 'Año', 'Hora', 'Minuto']])

    #----------------------------------Realizar Predicción Aplicando la Regresión-----------------------------------------------------

    y_pred_EA = modeloEA.predict(df_scaled)
    y_pred_ER = modeloER.predict(df_scaled)

    #-------------------------------------------Unir predicción al dataframe de predicción--------------------------------------------

    # Convertir el array de NumPy en un DataFrame de pandas
    df_predEA = pd.DataFrame(y_pred_EA, columns=['Active_energy'])
    df_predER = pd.DataFrame(y_pred_ER, columns=['Reactive_energy'])

    # Unir el DataFrame para predicción con el DataFrame del array
    df01 = pd.concat([df1, df_predEA, df_predER], axis=1)

    # Genero una nueva Columna 'prediccion_consumo_energia'
    df01['prediccion_consumo_energia'] = 1

    # Genero una tabla con la base de históricos para relacionar el idCliente y Sector económico
    idCliente_sectoreconomico = df.groupby('idCliente').agg({'sectoreconomico': ['min']})

    # Reseteo el index
    idCliente_sectoreconomico = idCliente_sectoreconomico.reset_index()

    # Elimino Nivel nombre columnas dataframe
    idCliente_sectoreconomico.columns = idCliente_sectoreconomico.columns.droplevel(1)

    # Adiciono campo sector económico en la tabla predicción
    df001 = pd.merge(df01, idCliente_sectoreconomico, how='left', left_on=['idCliente'], right_on=['idCliente'])

    #----------------------------------------------Cargar Modelo Prediccion anomalias------------------------------------------------

    # Cargar la arquitectura del modelo desde el archivo JSON
    arquitectura_del_modelo_clasificacion_archivo_json = "static/modelos/modelo_clasificacion.json"
    with open(arquitectura_del_modelo_clasificacion_archivo_json, "r") as archivo:
        modelo_cargado_Clf = archivo.read()

    # Reconstruir el modelo desde la arquitectura cargada
    modeloClf = model_from_json(modelo_cargado_Clf)

    pesos_del_modelo_clasificacion= "static/modelos/modeloClf_pesos.weights.h5"
    # Cargar los pesos del modelo desde el archivo HDF5
    modeloClf.load_weights(pesos_del_modelo_clasificacion)

    # Scaler_modelo_clasificacion
    scaler_modelo_clasificacion='static/modelos/scalerClf.joblib'
    scalerClf = load(scaler_modelo_clasificacion)

    #---------------------------------------------Predicción anomalias datos históricos---------------------------------------------

    columnasClf = ['Active_energy', 'Reactive_energy', 'idCliente', 'Dia', 'Mes', 'Año', 'Hora', 'Minuto']

    dfClf01 = scalerClf.transform(df00[columnasClf])

    anomalias = modeloClf.predict(dfClf01)
    df00_anomalias = df00.copy()
    df00_anomalias['Anomalia'] = [round(x[0]) for x in anomalias]

    #----------------------------------------------Predicción anomalias datos predichos---------------------------------------------

    dfClf001 = scalerClf.transform(df001[columnasClf])

    anomaliaspre = modeloClf.predict(dfClf001)
    df001_anomalias = df001.copy()
    df001_anomalias['Anomalia'] = [round(x[0]) for x in anomaliaspre]


    #--------------------------------------------------------Unir Tabla Históricos y predicción consumo-----------------------------

    df003_anomalias = pd.concat([df001_anomalias, df00_anomalias])

    df003_anomalias.prediccion_consumo_energia.unique()


    print('-----------------df003_anomalias.head(6)-----------------')
    print('-----------------df003_anomalias.head(6)-----------------')
    print('-----------------df003_anomalias.head(6)-----------------')
    print(df003_anomalias.shape)
    print(df003_anomalias.head(6))

    df003_anomalias = df003_anomalias.reset_index(drop=True)
    df004_anomalias= df003_anomalias.reset_index()
    #--------------------------------------------------Guardar en base de datos -----------------------------------------------------

    #Eliminar datos de la tabla
    if Datostablaregresion.objects.exists():
        Datostablaregresion.objects.all().delete()


    #Cargar nuevos datos pd.Timestamp('2024-12-31 23:00:00', tz='UTC')

    instances = [
        Datostablaregresion(

            id = int(row['index'])+1,
            idcliente= int(row['idCliente']),
            fecha= row['fecha'],
            dia=int(row['Dia']),
            mes= int(row['Mes']),
            anio=int(row['Año']),
            hora=int(row['Hora']),
            minuto=int(row['Minuto']),
            energia_activa=float(row['Active_energy']),
            energia_reactiva=float(row['Reactive_energy']),
            prediccion_consumo_energia=int(row['prediccion_consumo_energia']),
            sectoreconomico=str(row['sectoreconomico']),
            anomalia = float(row['Anomalia']),
        )
        for index, row in df004_anomalias.iterrows()
    ]

    Datostablaregresion.objects.bulk_create(instances)
    print('*****************************************')
    print("DataFrame guardado en la base de datos.")
    print('*****************************************')

    return render(request, 'fin_manager/carguedatos.html')




def regresion_energia_activa(df003_anomalias_re1):
    graph_data = {}
    #convierto a datos atipo fecha
    df003_anomalias_re1['fecha'] = pd.to_datetime(df003_anomalias_re1['fecha'].dt.strftime('%b-%Y'), errors='coerce')
    # Agrupo por Fecha
    df003_anomalias_re2 = df003_anomalias_re1.groupby('fecha').agg({'Active_energy': 'sum',
                                                                    'Reactive_energy': 'sum',
                                                                    'prediccion_consumo_energia': 'min'
                                                                    })
    # Reseteo el index
    df003_anomalias_re2 = df003_anomalias_re2.reset_index()
    #Selecciono Columnas de interés para la gráfica 1 de regresión EA= energía activa
    df003_anomalias_re3 = df003_anomalias_re2.loc[:, ['fecha', 'Active_energy', 'prediccion_consumo_energia']]
    # Gráfica 1 de Regresión Energía activa (EA)


    #fig = px.line(df003_anomalias_re3, x='fecha', y='Active_energy', color='prediccion_consumo_energia',
    #              # size='Active_energy',  # Tamaño de los puntos según la variable Y
    #              color_discrete_sequence=px.colors.qualitative.Pastel,  # Cambiar la paleta de colores
    #              title="Predicción Energía activa",
    #              labels={'fecha': 'fecha', 'Active_energy': 'Energía Activa',
    #                      'prediccion_consumo_energia': 'Prediccion Energía Activa'})

    df003_anomalias_re3['prediccion_consumo_energia'] = df003_anomalias_re3['prediccion_consumo_energia'].replace(
        {0: 'Demanda Energía', 1: 'Pronóstico'})
    fig = px.line(
        df003_anomalias_re3,
        x='fecha',
        y='Active_energy',
        color='prediccion_consumo_energia',
        color_discrete_sequence=px.colors.qualitative.Pastel,  # Cambiar la paleta de colores
        title="Gráfico de Energía Activa con Predicciones",
        labels={
            'fecha': 'Fecha',
            'Active_energy': 'Energía Activa',
            'prediccion_consumo_energia': 'Tipo de Predicción'
        }
    )

    graph_json = fig.to_json()
    graph_data['chart'] = graph_json
    return graph_data['chart']



def comportamiento_consumo_energia(df003_anomalias_re1):
    graph_data = {}
    df003_anomalias_re0 = df003_anomalias_re1[df003_anomalias_re1.prediccion_consumo_energia == 0]
    # cambiar formato fecha a mes-año
    df003_anomalias_re0['fecha'] = pd.to_datetime(df003_anomalias_re0['fecha'].dt.strftime('%b-%Y'), errors='coerce')
    df003_anomalias_re02 = df003_anomalias_re0.groupby('fecha').agg({'Active_energy': 'sum',
                                                                     'Reactive_energy': 'sum',
                                                                     'prediccion_consumo_energia': 'min'
                                                                     }).reset_index()
    # Crear el gráfico
    fig = px.bar(df003_anomalias_re02, x='fecha', y=['Active_energy', 'Reactive_energy'],
                 title='Consumo Energía Activa y Reactiva')

    graph_json = fig.to_json()
    graph_data['chart'] = graph_json
    return graph_data['chart']



def regresion_energia_reactiva(df003_anomalias_re1):
    graph_data = {}
    #convierto a datos atipo fecha
    df003_anomalias_re1['fecha'] = pd.to_datetime(df003_anomalias_re1['fecha'].dt.strftime('%b-%Y'), errors='coerce')
    # Agrupo por Fecha
    df003_anomalias_re2 = df003_anomalias_re1.groupby('fecha').agg({'Active_energy': 'sum',
                                                                    'Reactive_energy': 'sum',
                                                                    'prediccion_consumo_energia': 'min'
                                                                    })
    # Reseteo el index
    df003_anomalias_re2 = df003_anomalias_re2.reset_index()
    #Selecciono Columnas de interés para gráfica 2
    df003_anomalias_re4 = df003_anomalias_re2.loc[:, ['fecha', 'Reactive_energy', 'prediccion_consumo_energia']]
    # Gráfica 2 de Regresión Energía Reactiva (ER)
    #fig = px.line(df003_anomalias_re4, x='fecha', y='Reactive_energy', color='prediccion_consumo_energia',
    #              # size='Active_energy',  # Tamaño de los puntos según la variable Y
    #              color_discrete_sequence=px.colors.qualitative.Pastel,  # Cambiar la paleta de colores
    #              title="Predicción Energía Reactiva",
    #             labels={'fecha': 'fecha', 'Reactive_energy': 'Energía Reactiva',
    #                      'prediccion_consumo_energia': 'Prediccion Energía Reactiva'})
    df003_anomalias_re4['prediccion_consumo_energia'] = df003_anomalias_re4['prediccion_consumo_energia'].replace(
        {0: 'Demanda Energía', 1: 'Pronóstico'})

    fig = px.line(df003_anomalias_re4, x='fecha', y='Reactive_energy', color='prediccion_consumo_energia',
                  # size='Active_energy',  # Tamaño de los puntos según la variable Y
                  color_discrete_sequence=px.colors.qualitative.Pastel,  # Cambiar la paleta de colores
                  title="Predicción Energía Reactiva",
                  labels={'fecha': 'fecha', 'Reactive_energy': 'Reactive energy',
                          'prediccion_consumo_energia': 'prediccion Energía Activa'})
    graph_json = fig.to_json()
    graph_data['chart'] = graph_json
    return graph_data['chart']

def fig_anomalias(df003_anomalias_re1):
    graph_data = {}
    #Grafica de prediccion anomalia
    df003_anomalias_cl01 = df003_anomalias_re1[df003_anomalias_re1.prediccion_consumo_energia == 0]
    df003_anomalias_cl02 = df003_anomalias_cl01.loc[:, ['Active_energy', 'Reactive_energy', 'Anomalia']]
    fig_Anomalias = px.scatter(df003_anomalias_cl02.replace(0, ''), x='Active_energy', y='Reactive_energy',
                     title='Predicción de anomalias',
                     color="Anomalia")

    graph_json = fig_Anomalias.to_json()
    graph_data['chart'] = graph_json
    return graph_data['chart']

def fig_clientes_anomalias(df003_anomalias_re1):
    graph_data = {}
    #Gráfica de clientes con mayor cantidad de anomalias
    df003_anomalias_cl01 = df003_anomalias_re1[df003_anomalias_re1.prediccion_consumo_energia == 0]
    df003_anomalias_cl04 = df003_anomalias_cl01.loc[:, ['idCliente', 'sectoreconomico', 'Anomalia']]
    dfH = df003_anomalias_cl04[['idCliente', 'sectoreconomico', 'Anomalia']].groupby(
        by=['idCliente', 'sectoreconomico'], as_index=False).sum()
    fig_Clientes_Anomalias = px.treemap(dfH,
                     path=['sectoreconomico', 'idCliente'],
                     values='Anomalia',
                     title='Clientes con más anomalias')

    graph_json = fig_Clientes_Anomalias.to_json()
    graph_data['chart'] = graph_json
    return graph_data['chart']

def fig_radar(df003_anomalias_re1):
    graph_data = {}
    #Gráfica de clientes con mayor cantidad de anomalias
    df003_anomalias_cl01 = df003_anomalias_re1[df003_anomalias_re1.prediccion_consumo_energia == 0]
    df003_anomalias_cl04 = df003_anomalias_cl01.loc[:, ['idCliente', 'sectoreconomico', 'Anomalia']]
    dfH = df003_anomalias_cl04[['idCliente', 'sectoreconomico', 'Anomalia']].groupby(
        by=['idCliente', 'sectoreconomico'], as_index=False).sum()
    # Gráfica de radar de sectores con mayor cantidad de anomalias
    dfHS = dfH[['sectoreconomico', 'Anomalia']].groupby(by='sectoreconomico', as_index=False).mean()
    # Crear gráfica tipo radar
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=dfHS['Anomalia'],
        theta=dfHS['sectoreconomico'],
        fill='toself',
        name='Anomalia'
    ))
    # Actualizar el diseño de la gráfica
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 4550]
            )
        ),
        title='Gráfica tipo Radar - Anomalías por Sector Económico'
    )
    graph_json = fig_radar.to_json()
    graph_data['chart'] = graph_json
    return graph_data['chart']




def generargraficasdashboard(df003_anomalias_re1):

    # leo Base de datos procesada y con predicción por medio de la función obtener_datos_de_la_tabla_desde_memoria

    #----------------Gráficas Generadas con el Modelo de regresion------------------------------------------

    #convierto a datos atipo fecha
    df003_anomalias_re1['fecha'] = pd.to_datetime(df003_anomalias_re1['fecha'].dt.strftime('%b-%Y'), errors='coerce')

    # Agrupo por Fecha
    df003_anomalias_re2 = df003_anomalias_re1.groupby('fecha').agg({'Active_energy': 'sum',
                                                                    'Reactive_energy': 'sum',
                                                                    'prediccion_consumo_energia': 'min'
                                                                    })
    # Reseteo el index
    df003_anomalias_re2 = df003_anomalias_re2.reset_index()

    #Selecciono Columnas de interés para la gráfica 1 de regresión EA= energía activa
    df003_anomalias_re3 = df003_anomalias_re2.loc[:, ['fecha', 'Active_energy', 'prediccion_consumo_energia']]

    # Gráfica 1 de Regresión Energía activa (EA)
    fig_EA_reg = px.line(df003_anomalias_re3, x='fecha', y='Active_energy', color='prediccion_consumo_energia',
                  # size='Active_energy',  # Tamaño de los puntos según la variable Y
                  color_discrete_sequence=px.colors.qualitative.Pastel,  # Cambiar la paleta de colores
                  title="Gráfico de dispersión con colores según categoría",
                  labels={'fecha': 'fecha', 'Active_energy': 'Active_energy',
                          'prediccion_consumo_energia': 'prediccion Energía Activa'})

    #Selecciono Columnas de interés para gráfica 2
    df003_anomalias_re4 = df003_anomalias_re2.loc[:, ['fecha', 'Reactive_energy', 'prediccion_consumo_energia']]

    # Gráfica 2 de Regresión Energía Reactiva (ER)
    fig_ER_reg = px.line(df003_anomalias_re4, x='fecha', y='Reactive_energy', color='prediccion_consumo_energia',
                  # size='Active_energy',  # Tamaño de los puntos según la variable Y
                  color_discrete_sequence=px.colors.qualitative.Pastel,  # Cambiar la paleta de colores
                  title="Gráfico de dispersión con colores según categoría",
                  labels={'fecha': 'fecha', 'Reactive_energy': 'Reactive energy',
                          'prediccion_consumo_energia': 'prediccion Energía Activa'})




    # ---------------Gráficas Generadas con el Modelo de Clasificación--------------------------------------

    #Grafica de prediccion anomalia
    df003_anomalias_cl01 = df003_anomalias_re1[df003_anomalias_re1.prediccion_consumo_energia == 0]
    df003_anomalias_cl02 = df003_anomalias_cl01.loc[:, ['Active_energy', 'Reactive_energy', 'Anomalia']]
    fig_Anomalias = px.scatter(df003_anomalias_cl02.replace(0, ''), x='Active_energy', y='Reactive_energy',
                     title='Predicción de anomalias',
                     color="Anomalia")

    #Gráfica de clientes con mayor cantidad de anomalias
    df003_anomalias_cl04 = df003_anomalias_cl01.loc[:, ['idCliente', 'sectoreconomico', 'Anomalia']]
    dfH = df003_anomalias_cl04[['idCliente', 'sectoreconomico', 'Anomalia']].groupby(
        by=['idCliente', 'sectoreconomico'], as_index=False).sum()
    fig_Clientes_Anomalias = px.treemap(dfH,
                     path=['sectoreconomico', 'idCliente'],
                     values='Anomalia',
                     title='Clientes con más anomalias')

    # Gráfica de radar de sectores con mayor cantidad de anomalias
    dfHS = dfH[['sectoreconomico', 'Anomalia']].groupby(by='sectoreconomico', as_index=False).mean()
    # Crear gráfica tipo radar
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=dfHS['Anomalia'],
        theta=dfHS['sectoreconomico'],
        fill='toself',
        name='Anomalia'
    ))
    # Actualizar el diseño de la gráfica
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 4550]
            )
        ),
        title='Gráfica tipo Radar - Anomalías por Sector Económico'
    )

    graficas=[fig_EA_reg, fig_ER_reg, fig_Anomalias, fig_Clientes_Anomalias, fig_radar ]

    return graficas

