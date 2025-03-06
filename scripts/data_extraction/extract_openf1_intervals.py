"""
OpenF1 Intervals Data Extraction Script
---------------------------------------
Este script extrae específicamente los datos de intervalos entre pilotos
desde la API de OpenF1 y los almacena en formato Parquet para su uso posterior
en análisis de estrategia, undercuts/overcuts, etc.

Complementa el script existente de extracción de FastF1 para ofrecer
datos más precisos sobre los gaps entre coches.
"""

import requests
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

# Crear directorios necesarios
Path("data/raw").mkdir(parents=True, exist_ok=True)


def get_session_key(year, gp_name):
    """
    Obtiene el session_key para la carrera especificada.

    Args:
        year (int): Año de la carrera
        gp_name (str): Nombre del Gran Premio

    Returns:
        int: session_key para usar en la API de OpenF1
    """
    # Para GP España 2023, el session_key correcto es 9102
    if year == 2023 and gp_name.lower() == "spain":
        return 9102

    # Si necesitamos obtener dinámicamente los session_keys para otras carreras,
    # podríamos implementar una búsqueda usando la API de sesiones de OpenF1
    # Por ahora, solo implementamos para España 2023

    print(f"⚠️ Solo tenemos implementado el session_key para España 2023")
    print(f"⚠️ Para otras carreras, necesitas encontrar el session_key correcto")
    return None


def fetch_openf1_intervals(year, gp_name, max_interval=None):
    """
    Extrae datos de intervalos entre coches desde OpenF1 API para una carrera específica.

    Args:
        year (int): Año de la carrera (ej: 2023)
        gp_name (str): Nombre del Gran Premio (ej: 'Spain')
        max_interval (float, optional): Filtrar intervalos menores a este valor (en segundos)

    Returns:
        pandas.DataFrame: DataFrame con datos de intervalos entre coches
    """
    base_url = "https://api.openf1.org/v1/intervals"

    # Obtener el session_key para esta carrera
    session_key = get_session_key(year, gp_name)
    if session_key is None:
        print("❌ No se pudo obtener el session_key para esta carrera")
        return pd.DataFrame()

    print(
        f"Buscando datos en OpenF1 para {gp_name} {year} (session_key: {session_key})")

    # Construir la URL directamente, exactamente como la que funciona en el navegador
    if max_interval is not None:
        url = f"{base_url}?session_key={session_key}&interval<{max_interval}"
        print(f"URL: {url}")
    else:
        url = f"{base_url}?session_key={session_key}"
        print(f"URL: {url}")

    try:
        print("Realizando petición a OpenF1 API...")
        response = requests.get(url)
        response.raise_for_status()

        print(f"Estado de la respuesta: {response.status_code}")

        # Comprobar si hemos recibido datos válidos
        if response.text and response.text.strip():
            try:
                # Intentar parsear el JSON
                intervals_data = response.json()
                print(
                    f"✓ Se encontraron {len(intervals_data)} registros para session_key={session_key}")

                # Convertir a DataFrame
                df_intervals = pd.DataFrame(intervals_data)

                # Procesamiento básico de datos
                if 'date' in df_intervals.columns:
                    # Manejar diferentes formatos de fecha
                    df_intervals['date'] = pd.to_datetime(
                        df_intervals['date'], format='mixed')

                # Asegurarnos de que el nombre de la columna es consistente
                if 'interval' in df_intervals.columns:
                    df_intervals.rename(
                        columns={'interval': 'interval_in_seconds'}, inplace=True)

                if 'interval_in_seconds' in df_intervals.columns:
                    # Marcar intervalos en zonas estratégicas
                    df_intervals['undercut_window'] = df_intervals['interval_in_seconds'] < 1.5
                    df_intervals['drs_window'] = df_intervals['interval_in_seconds'] < 1.0

                return df_intervals
            except Exception as e:
                print(f"Error al procesar datos: {e}")
                print("Intentando procesar manualmente...")

                # Procesar manualmente sin usar pandas para datetime
                try:
                    intervals_data = json.loads(response.text)
                    df_intervals = pd.DataFrame(intervals_data)

                    # Guardar fechas como strings para evitar problemas de formato
                    if 'date' in df_intervals.columns:
                        # Guardar como string
                        df_intervals['date_str'] = df_intervals['date']
                        # No convertimos a datetime para evitar errores

                    # Asegurarnos de que el nombre de la columna es consistente
                    if 'interval' in df_intervals.columns:
                        df_intervals.rename(
                            columns={'interval': 'interval_in_seconds'}, inplace=True)

                    if 'interval_in_seconds' in df_intervals.columns:
                        # Marcar intervalos en zonas estratégicas
                        df_intervals['undercut_window'] = df_intervals['interval_in_seconds'] < 1.5
                        df_intervals['drs_window'] = df_intervals['interval_in_seconds'] < 1.0

                    print(
                        f"✓ Procesamiento manual exitoso. Se obtuvieron {len(df_intervals)} registros.")
                    return df_intervals
                except Exception as e2:
                    print(f"Error en procesamiento manual: {e2}")
                    return pd.DataFrame()
        else:
            print("La respuesta está vacía")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error al consultar OpenF1: {e}")
        if 'response' in locals():
            print(
                f"Código de estado: {response.status_code if hasattr(response, 'status_code') else 'N/A'}")
            print(
                f"Primeros 500 caracteres de la respuesta: {response.text[:500] if hasattr(response, 'text') else 'N/A'}")
        return pd.DataFrame()


def extract_openf1_intervals(year, gp_name, max_interval=None):
    """
    Función principal para extraer y guardar datos de intervalos de OpenF1.

    Args:
        year (int): Año de la carrera
        gp_name (str): Nombre del Gran Premio
        max_interval (float, optional): Filtrar intervalos menores a este valor (en segundos)
    """
    intervals_df = fetch_openf1_intervals(year, gp_name, max_interval)

    if not intervals_df.empty:
        # Manejar tipos de datos mixtos en columnas - SOLUCIÓN AL ERROR
        # Identificar y manejar columnas problemáticas con datos mixtos (como gap_to_leader)
        if 'gap_to_leader' in intervals_df.columns:
            # Opción 1: Convertir toda la columna a string
            intervals_df['gap_to_leader'] = intervals_df['gap_to_leader'].astype(
                str)

            # Opción 2 (alternativa): Crear columnas separadas para valores numéricos y categorías
            # Permite análisis numérico posterior más fácil
            intervals_df['gap_to_leader_numeric'] = pd.to_numeric(
                intervals_df['gap_to_leader'], errors='coerce')
            intervals_df['is_lapped'] = intervals_df['gap_to_leader'].astype(
                str).str.contains('LAP')

        # Revisar otras columnas potencialmente problemáticas con tipos mixtos
        for col in intervals_df.columns:
            if intervals_df[col].dtype == 'object' and col != 'date' and col != 'date_str':
                # Si no es una fecha, intentamos convertir a numérico o asegurar que es string
                try:
                    # Intentamos convertir a numérico
                    intervals_df[col] = pd.to_numeric(intervals_df[col])
                except:
                    # Si falla, aseguramos que es string
                    intervals_df[col] = intervals_df[col].astype(str)

        # Guardar datos en formato Parquet
        output_path = f"data/raw/{gp_name}_{year}_openf1_intervals.parquet"
        intervals_df.to_parquet(output_path)

        # Guardar también en CSV para mayor compatibilidad/diagnóstico
        csv_path = f"data/raw/{gp_name}_{year}_openf1_intervals.csv"
        intervals_df.to_csv(csv_path, index=False)
        print(f"Datos de respaldo guardados en CSV: {csv_path}")

        # Información de resumen
        print(f"Datos de intervalos guardados en {output_path}")
        print(f"Total de registros: {len(intervals_df)}")
        if 'interval_in_seconds' in intervals_df.columns:
            print(
                f"Rango de intervalos: {intervals_df['interval_in_seconds'].min():.2f}s - {intervals_df['interval_in_seconds'].max():.2f}s")
            undercut_opportunities = intervals_df['undercut_window'].sum()
            print(
                f"Oportunidades de undercut (<1.5s): {undercut_opportunities} ({undercut_opportunities/len(intervals_df)*100:.1f}%)")

        # Crear archivo de metadatos
        metadata = {
            "source": "OpenF1 API",
            "race": {
                "year": year,
                "name": gp_name,
                "session_key": get_session_key(year, gp_name)
            },
            "extraction_date": datetime.now().isoformat(),
            "record_count": len(intervals_df),
            "fields": list(intervals_df.columns),
            "filter_applied": {"max_interval": max_interval} if max_interval is not None else None,
            "column_fixes": ["gap_to_leader converted to string and separate numeric column created"]
        }

        with open(f"data/raw/{gp_name}_{year}_openf1_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        return True
    else:
        print("No se guardaron datos porque no se encontraron registros.")
        return False


if __name__ == "__main__":
    # Extraer datos de intervalos para el GP de España 2023
    # Usando el session_key 9102 y filtrando intervalos menores a 1.75 segundos
    extract_openf1_intervals(2023, "Spain", max_interval=1.75)
