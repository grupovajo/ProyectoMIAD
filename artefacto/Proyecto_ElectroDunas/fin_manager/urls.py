#from django.conf.urls import url
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('carguedatos', views.carguedatos, name='carguedatos'),
    path('grafica1', views.grafica1, name='grafica1'),
    path('grafica2', views.grafica2, name='grafica2'),
    path('grafica3', views.grafica3, name='grafica3'),
    path('grafica4', views.grafica4, name='grafica4'),
    path('grafica5', views.grafica5, name='grafica5'),
    path('grafica6', views.grafica6, name='grafica6'),
    path('generartablaprediccion', views.generartablaprediccion,name='generartablaprediccion' ),
    path('cargue_modelos', views.cargue_modelos, name='cargue_modelos'),
    path('respuesta_exitosa', views.respuesta_exitosa, name='respuesta_exitosa'),
    path('respuesta_no_exitosa', views.respuesta_no_exitosa, name='respuesta_no_exitosa'),
    path('tabla_registros_consumo', views.tabla_registros_consumo, name='tabla_registros_consumo'),
    path('ventanagenerarprediccion', views.ventanagenerarprediccion, name='ventanagenerarprediccion'),
    path('ver_pdf', views.ver_pdf, name='ver_pdf'),


]
