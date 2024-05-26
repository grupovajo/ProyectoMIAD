# CASO ELECTRODUNAS DETECCIÓN DE ANOMALÍAS - PROYECTO APLICADO GRUPO 1 MIAD

<img src="https://raw.githubusercontent.com/grupovajo/proyectoANS/master/recursos/MIAD.png" align="right"
     alt="MIAD" width="84" height="35">
<img src="https://raw.githubusercontent.com/grupovajo/proyectoANS/master/recursos/Uniandes.png" align="left"
     alt="MIAD" width="84" height="35">
     
Repositorio grupo 1 del proyecto aplicado de la maestría en inteligencia analitica de datos Uniandes


## Descripción

Este repositorio de proyecto tiene como objetivo contener el código y la documentación del artefacto propuesto el cual es una interfaz simple y amigable para los usuarios. Esta herramienta permite que tanto el cliente (ElectroDunas) como sus usuarios de sus servicios puedan evidenciar las detecciones de consumos eléctricos anómalos en las fechas de su interés. Además, ofrece la visualización de información histórica y realiza predicciones de la demanda de energía eléctrica, tanto activa como reactiva de los clientes de ElectroDunas.
Las principales ventajas que ofrece el dashboard a la empresa ElectroDunas incluyen el análisis de patrones anómalos por nivel de actividad económica y por clientes, la proyección de demandas energéticas futuras, lo que permitiría realizar la planificación de inversiones en infraestructura eléctrica, aprovechando la información detallada por sector económico y cliente.
Adicionalmente, la detección de anomalías señala posibles pérdidas técnicas, como fugas, conexiones ilegales o fallas en equipos. Al identificarlas, ElectroDunas puede anticiparse a problemas y garantizar la calidad del servicio para sus clientes. Además, le permitirá cuantificar los costos asociados a estas anomalías y focalizar esfuerzos en planes de remediación.


- A continuación se podrá consultar la siguiente información:

  - [Prerrequisitos](#prerrequisitos)
  - [Contenido](#contenido)
  - [Resultados](#resultados)
  - [Equipo](#equipo)
  - [Contacto](#contacto)
 
## Prerrequisitos
Se debe tener instalado Django, Keras, Tensorflow, Scikit-Learn y SciPy, se recomienda tener Anaconda 3 o de lo contrario las librerias de Pandas, Numpy, Matplotlib y Seaborn.

## Contenido
Dento del repositorio se encuentra el código necesario para correr el artefacto del dashboard, así como los notebooks que se realizaron en el proceso de exploración de los datos, entrenamiento de modelos y modelos exportados para las predicciones. Es importante resaltar cada contenido del tablero que es el objetivo final del repositorio, a continuación algunas imágenes del acceso al repositorio:

LANDING PAGE

<img src="https://media.githubusercontent.com/media/grupovajo/ProyectoMIAD/main/recursos/Landing.png" align="center"
     alt="Land" width="780" height="520">

DASHBOARD

<img src="https://media.githubusercontent.com/media/grupovajo/ProyectoMIAD/main/recursos/Dashboard.png" align="center"
     alt="Dash" width="780" height="520">


## Resultados

De los resultados de los modelos entrenados y que se encuentran dentro del DashBoard, se entrenaron varios modelos que al final del ejercicio resultaron en 4 modelos de machine learning:
1. No supervisado - Kmedias
2. Supervisado de clasificación de anomalías- Redes Neuronales MLP
3. Supervisado de regresión de predicción de energía activa - Redes Neuronales MLP
4. Supervisado de regresión de predicción de energía reactiva - Redes Neuronales MLP

Estos modelos se encuentran en la carpeta de notebooks, uno por cada tipo de modelo.

Imágenes del Análisis Exploratorio y Visualización de Datos

Visualización de Datos Históricos

<img src="https://media.githubusercontent.com/media/grupovajo/ProyectoMIAD/main/recursos/Historico.png" align="center" alt="Dash" width="780" height="520">

Esta imagen muestra la visualización de los datos históricos de consumo eléctrico. Permite a los usuarios observar las tendencias y patrones del consumo a lo largo del tiempo, facilitando la identificación de cambios o anomalías en el comportamiento histórico del consumo de energía.


Análisis de Anomalías por Sectores Económicos

<img src="https://media.githubusercontent.com/media/grupovajo/ProyectoMIAD/main/recursos/SectoresEconomicosAnomalias.png" align="center" alt="Dash" width="780" height="520">

Aquí se presenta el análisis de anomalías de consumo eléctrico desglosado por sectores económicos. Esta visualización es crucial para identificar sectores específicos que presentan comportamientos atípicos, permitiendo a ElectroDunas focalizar sus esfuerzos en áreas de mayor interés o riesgo.


Predicción de Anomalías

<img src="https://media.githubusercontent.com/media/grupovajo/ProyectoMIAD/main/recursos/PrediccionAnomalia.png" align="center" alt="Dash" width="780" height="520">

Esta imagen ilustra el modelo de predicción de anomalías en el consumo eléctrico. Utilizando algoritmos avanzados de inteligencia artificial, esta herramienta permite anticipar posibles anomalías futuras, ofreciendo a ElectroDunas la oportunidad de tomar medidas preventivas antes de que los problemas se materialicen.


Pronóstico de Energía Activa

<img src="https://media.githubusercontent.com/media/grupovajo/ProyectoMIAD/main/recursos/PronosticoActiva.png" align="center" alt="Dash" width="780" height="520">

En esta visualización se muestra el pronóstico de la demanda de energía activa. Este análisis proyecta las necesidades energéticas futuras, permitiendo una planificación adecuada de la infraestructura y recursos necesarios para satisfacer la demanda de los clientes.


Estas visualizaciones no solo proporcionan una comprensión clara y detallada del consumo eléctrico histórico y proyectado, sino que también destacan áreas de interés para la detección de anomalías y la optimización de recursos. Esto permite a ElectroDunas mejorar la eficiencia operativa y garantizar un servicio de alta calidad para sus clientes.




## Equipo
Grupo 1:

Valentina Farkas Sánchez

Oscar Andrés Patiño Patarroyo

Andrés Sierra Urrego

John Edinson Rodríguez Fajardo




## Contacto

Para más información escriba al correo: o.patinop@uniandes.edu.co, v.farkas@uniandes.edu.co, a.sierrau@uniandes.edu.co, je.rodriguezf1@uniandes.edu.co
