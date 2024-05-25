from django.db import models

class Infoclientes(models.Model):
    id=models.IntegerField(primary_key=True)
    fecha = models.DateTimeField(max_length=255, null=True)
    active_energy=  models.FloatField(null=True)
    reactive_energy=  models.FloatField(null=True)
    voltaje_fa=  models.FloatField(null=True)
    voltaje_fc=  models.FloatField(null=True)
    idcliente= models.IntegerField(null=True)
    sectoreconomico = models.CharField(max_length=300, null=True, blank=True)

    usuario = models.CharField(max_length=255, null=True)
    fechahora = models.DateTimeField(null=True)

    def __str__(self):
        return f'infoclientes{self.id}: {self.fecha} {"-"} {self.active_energy} {"-"} {self.reactive_energy} {"-"} {self.voltaje_fa} {"-"} {self.voltaje_fc} {"-"} {self.idcliente}{"-"} {self.sectoreconomico}'


class Datostablaregresion(models.Model):
    id = models.IntegerField(primary_key=True)
    idcliente= models.IntegerField(null=True)
    fecha= models.DateField(max_length=255, null=True)
    dia=models.IntegerField(null=True)
    mes= models.IntegerField(null=True)
    anio=models.IntegerField(null=True)
    hora=models.IntegerField(null=True)
    minuto=models.IntegerField(null=True)
    energia_activa=models.FloatField(null=True)
    energia_reactiva=models.FloatField(null=True)
    prediccion_consumo_energia=models.IntegerField(null=True)
    sectoreconomico=models.CharField(max_length=500, null=True, blank=True)
    anomalia = models.FloatField(null=True)

    def __str__(self):
        return f'datostabla{self.id}: {self.idcliente} {"-"} {self.fecha} {"-"} {self.dia} {"-"} {self.mes} {"-"} {self.anio}{"-"} {self.hora} {"-"} {self.minuto} {"-"} {self.energia_activa} {"-"} {self.energia_reactiva}  {"-"} {self.sectoreconomico}'
