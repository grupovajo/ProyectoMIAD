# Generated by Django 4.2.7 on 2024-05-18 05:56

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fin_manager", "0002_datostablaregresion_fechahora_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="infoclientes",
            name="anomalia",
            field=models.IntegerField(null=True),
        ),
    ]
