import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from sklearn.neighbors import NearestNeighbors
from scipy.stats import zscore
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from collections import defaultdict
import random

st.set_page_config(page_title="Scouting de Jugadores", layout="wide")
st.title("Scouting de Jugadores")

# Variables objetivo posibles (deben ser numéricas y estar en el dataset)
variables_objetivo = [
    'Minutes Played', 'Goals', 'Assists', 'Matches Played', 'xG', 'xA', 'Shots per 90'
]
variables_objetivo_es = {
    'Minutes Played': 'Minutos Jugados',
    'Goals': 'Goles',
    'Assists': 'Asistencias',
    'Matches Played': 'Partidos Jugados',
    'xG': 'xG',
    'xA': 'xA',
    'Shots per 90': 'Tiros por 90'
}

# Cargar datos
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_excel('Dataset/TOTAL STATS PLAYERS.xlsx')
    except Exception:
        st.error("No se encontró el archivo 'Dataset/TOTAL STATS PLAYERS.xlsx'.")
        return None
    return df

df = cargar_datos()
if df is None:
    st.stop()

# Si Level Competition no es numérico, conviértelo a números (por ejemplo, usando LabelEncoder)
if df['Level Competition'].dtype == object:
    le = LabelEncoder()
    df['Level Competition'] = le.fit_transform(df['Level Competition'].astype(str))

# Diccionario de traducción de columnas
traduccion_columnas = {
    'Player': 'Jugador',
    'Actual Team': 'Equipo',
    'Competition': 'Competición',
    'Position 1': 'Posición',
    'Age': 'Edad',
    'Matches Played': 'Partidos Jugados',
    'Minutes Played': 'Minutos Jugados',
    'Duels per 90': 'Duelos por 90',
    'Duels Won, %': '% Duelos Ganados',
    'Defensive Duels per 90': 'Duelos Defensivos por 90',
    'Defensive Duels won, %': '% Duelos Defensivos Ganados',
    'Aerial Duels per 90': 'Duelos Aéreos por 90',
    'Aerial Duels Won, %': '% Duelos Aéreos Ganados',
    'Passes per 90': 'Pases por 90',
    'Accurate Passes, %': '% Pases Precisos',
    'Crosses per 90': 'Centros por 90',
    'Accurate Crosses, %': '% Centros Precisos',
    'Dribbles per 90': 'Regates por 90',
    'Successful Dribbles, %': '% Regates Exitosos',
    'Interceptions per 90': 'Intercepciones por 90',
    'Shots per 90': 'Tiros por 90',
    'Shots On Target, %': '% Tiros a Puerta',
    'Assists': 'Asistencias',
    'xA': 'xA',
    'Key Passes per 90': 'Pases Clave por 90',
    'Goals': 'Goles',
    'xG': 'xG',
    'Saves': 'Paradas',
    'Save Percentage': '% Paradas',
    'Goals Conceded': 'Goles Encajados',
    'Clean Sheets': 'Porterías a Cero',
    'Predicción Minutos Próxima Temporada': 'Predicción Minutos Próxima Temporada',
    'Contract Expires': 'Contrato Expira',
    'Market Value': 'Valor de Mercado'
}

# Definir los grupos principales y subposiciones presentes en el dataset
mapa_posiciones = {
    'Porteros': ['GK', 'POR', 'Goalkeeper'],
    'Defensas': ['CB', 'LCB', 'RCB', 'DFC', 'LB', 'LWB', 'LI', 'RB', 'RWB', 'LD'],
    'Mediocentros': ['DMF', 'LDMF', 'RDMF', 'MCD', 'CMF', 'RCMF', 'LCMF', 'MC', 'AMF', 'CAM', 'MP'],
    'Delanteros': ['LW', 'LAMF', 'LWF', 'EI', 'RW', 'RAMF', 'RWF', 'ED', 'CF', 'ST', 'DC']
}

# Subposiciones presentes en el dataset
subposiciones_presentes = df['Position 1'].unique().tolist()
posiciones_disponibles = {}
for grupo, subpos in mapa_posiciones.items():
    subpos_presentes = [p for p in subposiciones_presentes if str(p) in subpos]
    if subpos_presentes or grupo == 'Porteros':
        posiciones_disponibles[grupo] = subpos_presentes if subpos_presentes else mapa_posiciones[grupo]

# Obtener valores únicos de Level Competition para el filtro y preparar opción 'Todas las ligas'
niveles_liga = sorted(df['Level Competition'].unique())
opciones_liga = ['Todas las ligas'] + niveles_liga

# Variables predictoras y objetivo
features = [
    'Age', 'Matches Played', 'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90',
    'Defensive Duels won, %', 'Aerial Duels per 90', 'Aerial Duels Won, %', 'Passes per 90',
    'Accurate Passes, %', 'Crosses per 90', 'Accurate Crosses, %', 'Dribbles per 90',
    'Successful Dribbles, %', 'Shots per 90', 'Shots On Target, %', 'Assists', 'Goals',
    'Level Competition'
]

# Diccionario de traducción de subposiciones (debe estar antes de su uso)
traduccion_subpos = {
    'CB': 'Defensa Central', 'LCB': 'Defensa Central Izquierdo', 'RCB': 'Defensa Central Derecho', 'DFC': 'Defensa Central',
    'LB': 'Lateral Izquierdo', 'LWB': 'Carrilero Izquierdo', 'LI': 'Lateral Izquierdo',
    'RB': 'Lateral Derecho', 'RWB': 'Carrilero Derecho', 'LD': 'Lateral Derecho',
    'DMF': 'Mediocentro Defensivo', 'LDMF': 'Mediocentro Defensivo Izquierdo', 'RDMF': 'Mediocentro Defensivo Derecho', 'MCD': 'Mediocentro Defensivo',
    'CMF': 'Centrocampista Central', 'RCMF': 'Centrocampista Central Derecho', 'LCMF': 'Centrocampista Central Izquierdo', 'MC': 'Centrocampista Central',
    'AMF': 'Mediapunta', 'CAM': 'Mediapunta', 'MP': 'Mediapunta',
    'LW': 'Extremo Izquierdo', 'LAMF': 'Extremo Izquierdo', 'LWF': 'Extremo Izquierdo', 'EI': 'Extremo Izquierdo',
    'RW': 'Extremo Derecho', 'RAMF': 'Extremo Derecho', 'RWF': 'Extremo Derecho', 'ED': 'Extremo Derecho',
    'CF': 'Delantero Centro', 'ST': 'Delantero Centro', 'DC': 'Delantero Centro',
    'GK': 'Portero', 'POR': 'Portero', 'Goalkeeper': 'Portero'
}

# Diccionario de métricas importantes por grupo
metricas_grupo = {
    'Porteros': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played', 'Saves', 'Save Percentage', 'Goals Conceded', 'Clean Sheets', 'Contract Expires', 'Market Value'],
    'Defensas': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played', 'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90', 'Defensive Duels won, %', 'Aerial Duels per 90', 'Aerial Duels Won, %', 'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'Mediocentros': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played', 'Defensive Duels per 90', 'Defensive Duels won, %', 'Passes per 90', 'Accurate Passes, %', 'Interceptions per 90', 'Contract Expires', 'Market Value'],
    'Delanteros': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played', 'Shots per 90', 'Shots On Target, %', 'Goals', 'xG', 'Duels per 90', 'Duels Won, %', 'Dribbles per 90', 'Successful Dribbles, %', 'Crosses per 90', 'Accurate Crosses, %', 'Assists', 'xA', 'Key Passes per 90', 'Contract Expires', 'Market Value']
}

# Diccionario de métricas clave por subposición (debe estar antes de su uso)
metricas_subpos = {
    # DEFENSAS
    'CB': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90', 'Defensive Duels won, %',
           'Aerial Duels per 90', 'Aerial Duels Won, %', 'Interceptions per 90',
           'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'LCB': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90', 'Defensive Duels won, %',
           'Aerial Duels per 90', 'Aerial Duels Won, %', 'Interceptions per 90',
           'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'RCB': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90', 'Defensive Duels won, %',
           'Aerial Duels per 90', 'Aerial Duels Won, %', 'Interceptions per 90',
           'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'DFC': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90', 'Defensive Duels won, %',
           'Aerial Duels per 90', 'Aerial Duels Won, %', 'Interceptions per 90',
           'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'LB': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90', 'Defensive Duels won, %',
           'Crosses per 90', 'Accurate Crosses, %', 'Dribbles per 90', 'Successful Dribbles, %',
           'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'LWB': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90', 'Defensive Duels won, %',
           'Crosses per 90', 'Accurate Crosses, %', 'Dribbles per 90', 'Successful Dribbles, %',
           'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'LI': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90', 'Defensive Duels won, %',
           'Crosses per 90', 'Accurate Crosses, %', 'Dribbles per 90', 'Successful Dribbles, %',
           'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'RB': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90', 'Defensive Duels won, %',
           'Crosses per 90', 'Accurate Crosses, %', 'Dribbles per 90', 'Successful Dribbles, %',
           'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'RWB': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90', 'Defensive Duels won, %',
           'Crosses per 90', 'Accurate Crosses, %', 'Dribbles per 90', 'Successful Dribbles, %',
           'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'LD': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Duels per 90', 'Duels Won, %', 'Defensive Duels per 90', 'Defensive Duels won, %',
           'Crosses per 90', 'Accurate Crosses, %', 'Dribbles per 90', 'Successful Dribbles, %',
           'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    # MEDIOCENTROS
    'DMF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
            'Defensive Duels per 90', 'Defensive Duels won, %', 'Interceptions per 90',
            'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'LDMF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
            'Defensive Duels per 90', 'Defensive Duels won, %', 'Interceptions per 90',
            'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'RDMF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
            'Defensive Duels per 90', 'Defensive Duels won, %', 'Interceptions per 90',
            'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'MCD': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
            'Defensive Duels per 90', 'Defensive Duels won, %', 'Interceptions per 90',
            'Passes per 90', 'Accurate Passes, %', 'Contract Expires', 'Market Value'],
    'CMF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
            'Passes per 90', 'Accurate Passes, %', 'Key Passes per 90', 'Assists',
            'Defensive Duels per 90', 'Defensive Duels won, %', 'Interceptions per 90',
            'Contract Expires', 'Market Value'],
    'RCMF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
            'Passes per 90', 'Accurate Passes, %', 'Key Passes per 90', 'Assists',
            'Defensive Duels per 90', 'Defensive Duels won, %', 'Interceptions per 90',
            'Contract Expires', 'Market Value'],
    'LCMF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
            'Passes per 90', 'Accurate Passes, %', 'Key Passes per 90', 'Assists',
            'Defensive Duels per 90', 'Defensive Duels won, %', 'Interceptions per 90',
            'Contract Expires', 'Market Value'],
    'MC': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
            'Passes per 90', 'Accurate Passes, %', 'Key Passes per 90', 'Assists',
            'Defensive Duels per 90', 'Defensive Duels won, %', 'Interceptions per 90',
            'Contract Expires', 'Market Value'],
    'AMF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
            'Key Passes per 90', 'Assists', 'Goals', 'xG', 'xA', 'Shots per 90',
            'Dribbles per 90', 'Successful Dribbles, %', 'Contract Expires', 'Market Value'],
    'CAM': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
            'Key Passes per 90', 'Assists', 'Goals', 'xG', 'xA', 'Shots per 90',
            'Dribbles per 90', 'Successful Dribbles, %', 'Contract Expires', 'Market Value'],
    'MP': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
            'Key Passes per 90', 'Assists', 'Goals', 'xG', 'xA', 'Shots per 90',
            'Dribbles per 90', 'Successful Dribbles, %', 'Contract Expires', 'Market Value'],
    # DELANTEROS
    'LW': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Goals', 'xG', 'Shots per 90', 'Shots On Target, %', 'Assists', 'xA',
           'Dribbles per 90', 'Successful Dribbles, %', 'Crosses per 90', 'Accurate Crosses, %',
           'Contract Expires', 'Market Value'],
    'LAMF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Goals', 'xG', 'Shots per 90', 'Shots On Target, %', 'Assists', 'xA',
           'Dribbles per 90', 'Successful Dribbles, %', 'Crosses per 90', 'Accurate Crosses, %',
           'Contract Expires', 'Market Value'],
    'LWF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Goals', 'xG', 'Shots per 90', 'Shots On Target, %', 'Assists', 'xA',
           'Dribbles per 90', 'Successful Dribbles, %', 'Crosses per 90', 'Accurate Crosses, %',
           'Contract Expires', 'Market Value'],
    'EI': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Goals', 'xG', 'Shots per 90', 'Shots On Target, %', 'Assists', 'xA',
           'Dribbles per 90', 'Successful Dribbles, %', 'Crosses per 90', 'Accurate Crosses, %',
           'Contract Expires', 'Market Value'],
    'RW': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Goals', 'xG', 'Shots per 90', 'Shots On Target, %', 'Assists', 'xA',
           'Dribbles per 90', 'Successful Dribbles, %', 'Crosses per 90', 'Accurate Crosses, %',
           'Contract Expires', 'Market Value'],
    'RAMF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Goals', 'xG', 'Shots per 90', 'Shots On Target, %', 'Assists', 'xA',
           'Dribbles per 90', 'Successful Dribbles, %', 'Crosses per 90', 'Accurate Crosses, %',
           'Contract Expires', 'Market Value'],
    'RWF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Goals', 'xG', 'Shots per 90', 'Shots On Target, %', 'Assists', 'xA',
           'Dribbles per 90', 'Successful Dribbles, %', 'Crosses per 90', 'Accurate Crosses, %',
           'Contract Expires', 'Market Value'],
    'ED': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Goals', 'xG', 'Shots per 90', 'Shots On Target, %', 'Assists', 'xA',
           'Dribbles per 90', 'Successful Dribbles, %', 'Crosses per 90', 'Accurate Crosses, %',
           'Contract Expires', 'Market Value'],
    'CF': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Goals', 'xG', 'Shots per 90', 'Shots On Target, %', 'Assists', 'xA',
           'Duels per 90', 'Duels Won, %', 'Dribbles per 90', 'Successful Dribbles, %',
           'Contract Expires', 'Market Value'],
    'ST': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Goals', 'xG', 'Shots per 90', 'Shots On Target, %', 'Assists', 'xA',
           'Duels per 90', 'Duels Won, %', 'Dribbles per 90', 'Successful Dribbles, %',
           'Contract Expires', 'Market Value'],
    'DC': ['Player', 'Actual Team', 'Competition', 'Position 1', 'Age', 'Matches Played', 'Minutes Played',
           'Goals', 'xG', 'Shots per 90', 'Shots On Target, %', 'Assists', 'xA',
           'Duels per 90', 'Duels Won, %', 'Dribbles per 90', 'Successful Dribbles, %',
           'Contract Expires', 'Market Value'],
}

# Tabs para visualización y predicción
tabs = st.tabs(["Visualización de Datos", "Predicción"])

# --- FUNCIÓN DE PREDICCIÓN Y RECOMENDACIÓN AVANZADA ---
def calcular_prediccion(df, metricas_numericas, codigo_subpos):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from collections import defaultdict
    import random

    if df is None or df.empty or len(df) < 5 or not metricas_numericas:
        df_result = df.copy()
        df_result['Score Rendimiento Futuro'] = np.nan
        df_result['Liga recomendada'] = ''
        df_result['Equipo recomendado'] = ''
        df_result['Cluster'] = np.nan
        return df_result

    posibles_objetivo = ['Minutes Played', 'Goals', 'Assists', 'xG', 'xA', 'Shots per 90']
    objetivo = None
    for var in posibles_objetivo:
        if var in df.columns:
            objetivo = var
            break
    if objetivo is None:
        objetivo = metricas_numericas[0]

    X = df[metricas_numericas].fillna(0)
    y = df[objetivo].fillna(0)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X)
        df_result = df.copy()
        df_result['Score Rendimiento Futuro'] = y_pred
    except Exception:
        df_result = df.copy()
        df_result['Score Rendimiento Futuro'] = np.nan

    equipos_top = [
        'Real Madrid', 'Barcelona', 'PSG', 'Manchester City', 'Bayern Munich', 'Liverpool', 'Chelsea',
        'Juventus', 'Inter', 'Milan', 'Arsenal', 'Atlético de Madrid', 'Borussia Dortmund',
        'Tottenham', 'Napoli', 'Roma', 'Benfica', 'Porto', 'Ajax'
    ]
    club_level_dict = defaultdict(lambda: 4)
    for team in equipos_top:
        club_level_dict[team] = 1
    upper_mid = ['Sevilla', 'Villarreal', 'Leverkusen', 'Leipzig', 'Atalanta', 'Lazio', 'Monaco', 'Lille', 'Sporting CP', 'Feyenoord']
    for team in upper_mid:
        club_level_dict[team] = 2
    lower_mid = ['Getafe', 'Bologna', 'Crystal Palace', 'Mainz', 'Sassuolo', 'Nice', 'Osasuna', 'Granada']
    for team in lower_mid:
        club_level_dict[team] = 3
    ligas_top5 = ['LaLiga', 'Premier League', 'Bundesliga', 'Serie A', 'Ligue 1']
    # --- NUEVO: Mapear ligas de segunda división de países top ---
    segundas_divisiones = {
        'LaLiga': ['LaLiga2', 'Segunda División', 'LaLiga SmartBank'],
        'Premier League': ['Championship', 'EFL Championship'],
        'Bundesliga': ['2. Bundesliga'],
        'Serie A': ['Serie B'],
        'Ligue 1': ['Ligue 2']
    }
    # --- NUEVO: Función para inferir el nivel de división por nombre de liga ---
    def division_level(liga):
        l = liga.lower() if isinstance(liga, str) else ''
        if any(x in l for x in ['serie a', 'laliga', 'premier league', 'bundesliga', 'ligue 1', 'eredivisie', 'primeira liga', 'superliga', 'jupiler pro', 'ekstraklasa', 'super league', 'pro league', 'liga mx', 'mls', 'championship', 'liga portugal', 'liga 1', 'liga i', 'liga pro', 'liga betplay', 'liga tigo', 'liga postobon', 'liga aguila', 'liga bbva', 'liga nos', 'liga primera', 'liga profesional', 'liga nacional', 'liga pro', 'liga tigo', 'liga betplay', 'liga postobon', 'liga aguila', 'liga bbva', 'liga nos', 'liga primera', 'liga profesional', 'liga nacional']):
            if '2' in l or 'segunda' in l or 'b' in l or 'ii' in l or 'championship' in l or 'bwin' in l or 'ligue 2' in l or 'serie b' in l:
                return 2
            if '3' in l or 'tercera' in l or 'c' in l or 'iii' in l:
                return 3
            if '4' in l or 'cuarta' in l or 'd' in l or 'iv' in l:
                return 4
            return 1
        return 1  # Desconocido: tratar como primera división
    recomendados_por_equipo_pos = defaultdict(int)
    equipos_disponibles = df_result['Actual Team'].unique().tolist()
    ligas_disponibles = df_result['Competition'].unique().tolist()
    # --- ORDENAR JUGADORES POR SCORE (de mayor a menor) ---
    df_result = df_result.sort_values('Score Rendimiento Futuro', ascending=False).reset_index(drop=True)
    # --- AGRUPAR EQUIPOS ELEGIBLES POR NIVEL ---
    equipos_por_nivel = {1: [], 2: [], 3: [], 4: []}
    for e in equipos_disponibles:
        nivel = club_level_dict[e]
        equipos_por_nivel[nivel].append(e)
    for nivel in equipos_por_nivel:
        random.shuffle(equipos_por_nivel[nivel])  # Aleatorizar dentro de cada grupo
    equipo_recomendado = []
    liga_recomendada = []
    usados_por_equipo = defaultdict(int)
    for idx, row in df_result.iterrows():
        actual_team = row['Actual Team'] if 'Actual Team' in row else ''
        actual_league = row['Competition'] if 'Competition' in row else ''
        age = row['Age'] if 'Age' in row else 25
        score = row['Score Rendimiento Futuro'] if 'Score Rendimiento Futuro' in row else 0
        club_level = club_level_dict[actual_team]
        equipos_eligibles = [e for e in equipos_disponibles if e != actual_team]
        equipos_eligibles_filtrados = []
        actual_div = division_level(actual_league)
        # --- PRIMERO: salto igual o ascendente ---
        for e in equipos_eligibles:
            liga_destino = df_result[df_result['Actual Team'] == e]['Competition'].iloc[0] if not df_result[df_result['Actual Team'] == e].empty else ''
            destino_div = division_level(liga_destino)
            if destino_div < actual_div:
                continue
            if actual_league in ligas_top5 and actual_div == 1:
                if destino_div != 1:
                    continue
            elif actual_div == 2:
                if destino_div > 2:
                    continue
            elif actual_div == 3:
                if destino_div > 3:
                    continue
            equipos_eligibles_filtrados.append(e)
        # --- SI NO HAY, PERMITIR SALTO DESCENDENTE DE SOLO 1 NIVEL ---
        if not equipos_eligibles_filtrados:
            for e in equipos_eligibles:
                liga_destino = df_result[df_result['Actual Team'] == e]['Competition'].iloc[0] if not df_result[df_result['Actual Team'] == e].empty else ''
                destino_div = division_level(liga_destino)
                if destino_div == actual_div - 1:
                    equipos_eligibles_filtrados.append(e)
        if actual_league in ligas_top5 and age >= 23:
            ligas_eligibles = [l for l in ligas_disponibles if l in ligas_top5]
        else:
            ligas_eligibles = ligas_disponibles
        salto_max = 1
        if age <= 21 and score > np.nanpercentile(df_result['Score Rendimiento Futuro'], 90):
            salto_max = 2
        equipo_final = ''
        liga_final = ''
        asignado = False
        for nivel in range(1, 5):
            if (club_level - nivel) > salto_max:
                continue
            candidatos = [e for e in equipos_por_nivel[nivel] if e in equipos_eligibles_filtrados and usados_por_equipo[(e, codigo_subpos)] == 0]
            if candidatos:
                equipo_final = candidatos[0]
                liga_final = df_result[df_result['Actual Team'] == equipo_final]['Competition'].iloc[0] if not df_result[df_result['Actual Team'] == equipo_final].empty else ''
                usados_por_equipo[(equipo_final, codigo_subpos)] += 1
                asignado = True
                break
        if not asignado:
            equipo_final = ''
            liga_final = ''
        equipo_recomendado.append(equipo_final)
        liga_recomendada.append(liga_final)
    df_result['Equipo recomendado'] = equipo_recomendado
    df_result['Liga recomendada'] = liga_recomendada
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_result['Cluster'] = kmeans.fit_predict(X)
    except Exception:
        df_result['Cluster'] = np.nan
    return df_result

# --- TAB 1: Visualización de Datos ---
with tabs[0]:
    st.header("Visualización de Datos")
    nivel_seleccionado = st.multiselect(
        "Filtra por nivel de liga:",
        options=opciones_liga,
        default=['Todas las ligas'],
        key='vis_nivel'
    )
    grupos_sin_portero = [k for k in posiciones_disponibles.keys() if k != 'Porteros']
    grupo_posicion = st.selectbox("Grupo de posición:", grupos_sin_portero, key='vis_grupo')
    subposiciones = posiciones_disponibles[grupo_posicion]
    mostrar_subpos = True
    opciones_subpos = ['Todos']
    for p in subposiciones:
        nombre_traducido = traduccion_subpos.get(p, p)
        if nombre_traducido not in opciones_subpos:
            opciones_subpos.append(nombre_traducido)
    subposicion_traducida = st.selectbox("Posición específica:", opciones_subpos, key='vis_subpos')
    # FILTRADO ROBUSTO
    if subposicion_traducida == 'Todos':
        if 'Todas las ligas' in nivel_seleccionado:
            df_pos = df[df['Position 1'].isin(subposiciones)]
        else:
            df_pos = df[df['Position 1'].isin(subposiciones) & df['Level Competition'].isin(nivel_seleccionado)]
    else:
        codigo_subpos = next((k for k, v in traduccion_subpos.items() if v == subposicion_traducida and k in subposiciones), None)
        if codigo_subpos is None:
            mask_pos = df['Position 1'] == subposicion_traducida
        else:
            mask_pos = df['Position 1'] == codigo_subpos
        if 'Todas las ligas' in nivel_seleccionado:
            df_pos = df[mask_pos]
        else:
            mask_liga = df['Level Competition'].isin(nivel_seleccionado)
            df_pos = df[mask_pos & mask_liga]
    subtitulo = f"{grupo_posicion} - {'Todas las subposiciones' if subposicion_traducida == 'Todos' else subposicion_traducida}"
    # Selección de columnas según subposición
    if subposicion_traducida == 'Todos':
        columnas_mostrar = [col for col in metricas_grupo[grupo_posicion] if col in df_pos.columns]
    else:
        codigo_subpos = next((k for k, v in traduccion_subpos.items() if v == subposicion_traducida and k in subposiciones), None)
        if codigo_subpos and codigo_subpos in metricas_subpos:
            columnas_mostrar = [col for col in metricas_subpos[codigo_subpos] if col in df_pos.columns]
        else:
            columnas_mostrar = [col for col in metricas_grupo[grupo_posicion] if col in df_pos.columns]
    df_mostrar = df_pos[columnas_mostrar].copy()
    df_mostrar.columns = [traduccion_columnas.get(col, col) for col in df_mostrar.columns]
    st.subheader(f"Tabla de jugadores: {subtitulo}")
    if not df_mostrar.empty and len(df_mostrar.columns) > 0:
        st.dataframe(df_mostrar, use_container_width=True, height=400)
        # --- TOP JUGADORES ---
        top_metricas = {
            'Defensas': 'Minutos Jugados',
            'Mediocentros': 'Minutos Jugados',
            'Delanteros': 'Goles',
        }
        metrica_top = top_metricas.get(grupo_posicion, 'Minutos Jugados')
        if metrica_top in df_mostrar.columns:
            if st.button(f"Mostrar top 10 jugadores por {metrica_top}"):
                df_top = df_mostrar.sort_values(by=metrica_top, ascending=False).head(10)
                st.dataframe(df_top, use_container_width=True, height=400)
    else:
        st.warning("No hay datos para mostrar en la tabla.")
    ejemplo_jugadores = df_mostrar['Jugador'].unique()[:5].tolist() if 'Jugador' in df_mostrar.columns else []
    jugadores_seleccionados = st.multiselect(
        "Selecciona jugadores para comparar en gráficos:",
        options=df_mostrar['Jugador'].unique() if 'Jugador' in df_mostrar.columns else [],
        default=ejemplo_jugadores,
        key='vis_multiselect'
    )
    df_comparar = df_mostrar[df_mostrar['Jugador'].isin(jugadores_seleccionados)] if 'Jugador' in df_mostrar.columns else pd.DataFrame()
    if not df_comparar.empty:
        st.subheader("Comparativa de Minutos Jugados")
        if 'Minutos Jugados' in df_comparar.columns:
            fig = px.bar(
                df_comparar,
                x='Jugador',
                y='Minutos Jugados',
                color='Minutos Jugados',
                color_continuous_scale='Blues',
                title=f"Minutos jugados por jugador en {subtitulo}"
            )
            st.plotly_chart(fig, use_container_width=True)
        # --- NUEVO: Radar comparativo de métricas ---
        st.subheader("Radar comparativo de métricas")
        # Selección de métricas relevantes
        if subposicion_traducida == 'Todos':
            metricas_radar = [col for col in metricas_grupo[grupo_posicion] if col in df_pos.columns and col not in ['Player','Actual Team','Competition','Position 1','Contract Expires','Market Value']]
        else:
            codigo_subpos = next((k for k, v in traduccion_subpos.items() if v == subposicion_traducida and k in subposiciones), None)
            if codigo_subpos and codigo_subpos in metricas_subpos:
                metricas_radar = [col for col in metricas_subpos[codigo_subpos] if col in df_pos.columns and col not in ['Player','Actual Team','Competition','Position 1','Contract Expires','Market Value']]
            else:
                metricas_radar = [col for col in metricas_grupo[grupo_posicion] if col in df_pos.columns and col not in ['Player','Actual Team','Competition','Position 1','Contract Expires','Market Value']]
        # --- Asegurar que 'Age' esté incluida y al principio ---
        if 'Age' in df_pos.columns and 'Age' not in metricas_radar:
            metricas_radar = ['Age'] + metricas_radar
        elif 'Age' in df_pos.columns:
            metricas_radar = [m for m in metricas_radar if m != 'Age']
            metricas_radar = ['Age'] + metricas_radar
        # --- Traducir nombres de métricas al español para el radar ---
        metricas_radar_es = [traduccion_columnas.get(m, m) for m in metricas_radar]
        # --- PERFIL GENERAL (debe calcularse antes de cualquier uso) ---
        perfil_general = None
        if not df_pos.empty and metricas_radar:
            perfil_general = df_pos[metricas_radar].mean()
        # --- Definición de colores (debe estar antes de cualquier uso) ---
        color_general = 'gray'
        color_mejor = 'gold'
        color_seleccionado = 'royalblue'
        mejor_jugador = None
        nombre_mejor = None
        # --- Mejor jugador de la mejor liga disponible (por score global, no minutos, y usando equipos top) ---
        equipos_top = [
            'Real Madrid', 'Barcelona', 'PSG', 'Manchester City', 'Bayern Munich', 'Liverpool', 'Chelsea',
            'Juventus', 'Inter', 'Milan', 'Arsenal', 'Atlético de Madrid', 'Borussia Dortmund',
            'Tottenham', 'Napoli', 'Roma', 'Benfica', 'Porto', 'Ajax'
        ]
        mejor_liga = None
        mejor_jugador = None
        nombre_mejor = None
        if not df_pos.empty and metricas_radar:
            # Buscar ligas donde juegan los equipos top
            ligas_equipos_top = df_pos[df_pos['Actual Team'].isin(equipos_top)]['Level Competition'].unique().tolist()
            # Si hay ligas con equipos top, priorizar esas ligas
            ligas_prioridad = ligas_equipos_top if ligas_equipos_top else sorted(df_pos['Level Competition'].unique(), reverse=True)
            for nivel in ligas_prioridad:
                if nivel in ligas_equipos_top:
                    df_liga = df_pos[(df_pos['Level Competition'] == nivel) & (df_pos['Actual Team'].isin(equipos_top))]
                else:
                    df_liga = df_pos[df_pos['Level Competition'] == nivel]
                if not df_liga.empty:
                    mejor_liga = nivel
                    try:
                        arr = df_liga[metricas_radar].values.astype(float)
                        arr_z = (arr - np.nanmean(arr, axis=0)) / np.nanstd(arr, axis=0)
                        scores = np.nanmean(arr_z, axis=1)
                        idx_mejor = np.nanargmax(scores)
                        mejor_jugador_row = df_liga.iloc[idx_mejor]
                        mejor_jugador = mejor_jugador_row[metricas_radar]
                        nombre_mejor = mejor_jugador_row['Player'] if 'Player' in mejor_jugador_row else 'Mejor jugador'
                    except Exception:
                        mejor_jugador_row = df_liga.iloc[0]
                        mejor_jugador = mejor_jugador_row[metricas_radar]
                        nombre_mejor = mejor_jugador_row['Player'] if 'Player' in mejor_jugador_row else 'Mejor jugador'
                    break
        # --- JUGADORES SELECCIONADOS (debajo del título) ---
        jugadores_disponibles = df[df['Position 1'].isin(subposiciones)]['Player'].dropna().unique().tolist() if subposicion_traducida == 'Todos' else df[df['Position 1'] == codigo_subpos]['Player'].dropna().unique().tolist()
        if jugadores_disponibles:
            jugadores_seleccionados = st.multiselect(
                "Selecciona jugadores para comparar en el radar:",
                options=jugadores_disponibles,
                default=[],
                key='radar_multiselect'
            )
        else:
            jugadores_seleccionados = []
            st.info("No hay jugadores disponibles para esta posición/subposición y filtros seleccionados.")
        perfiles_seleccionados = []
        nombres_seleccionados = []
        if jugadores_seleccionados:
            for jugador in jugadores_seleccionados:
                fila = df_pos[df_pos['Player'] == jugador]
                if not fila.empty:
                    perfiles_seleccionados.append(fila.iloc[0][metricas_radar])
                    nombres_seleccionados.append(jugador)
        # --- GRAFICAR RADAR ---
        if perfil_general is not None and not perfil_general.isnull().any():
            fig_radar = go.Figure()
            # Normalizar todas las métricas a 0-100 para visualización
            def normaliza_100(valores, minimos, maximos):
                return [((v - mn) / (mx - mn) * 100) if mx > mn else 50 for v, mn, mx in zip(valores, minimos, maximos)]
            minimos = df_pos[metricas_radar].min().values
            maximos = df_pos[metricas_radar].max().values
            # General
            fig_radar.add_trace(go.Scatterpolar(
                r=normaliza_100(perfil_general.values, minimos, maximos),
                theta=metricas_radar_es,
                fill='toself',
                name='Perfil general',
                line=dict(color=color_general)
            ))
            # Mejor jugador de la mejor liga
            if mejor_jugador is not None:
                fig_radar.add_trace(go.Scatterpolar(
                    r=normaliza_100(mejor_jugador.values, minimos, maximos),
                    theta=metricas_radar_es,
                    fill='toself',
                    name=f'Mejor jugador liga top: {nombre_mejor}',
                    line=dict(color=color_mejor)
                ))
            else:
                st.info("No hay jugadores en la mejor liga para mostrar el mejor perfil.")
            # Seleccionados
            if perfiles_seleccionados:
                for i, perfil in enumerate(perfiles_seleccionados):
                    fig_radar.add_trace(go.Scatterpolar(
                        r=normaliza_100(perfil.values, minimos, maximos),
                        theta=metricas_radar_es,
                        fill='toself',
                        name=f'Seleccionado: {nombres_seleccionados[i]}',
                        line=dict(color=color_seleccionado)
                    ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,100], tickvals=list(range(0,101,20)), tickfont=dict(size=14))),
                showlegend=True,
                title="Radar comparativo de métricas",
                autosize=False,
                width=800,
                height=700,
                margin=dict(l=60, r=60, t=80, b=60)
            )
            st.plotly_chart(fig_radar, use_container_width=False)
            # Leyenda
            st.markdown("""
            <b>Leyenda:</b><br>
            <span style='color:gray;'>■</span> Perfil general<br>
            <span style='color:gold;'>■</span> Mejor jugador de la mejor liga<br>
            <span style='color:royalblue;'>■</span> Jugador seleccionado<br>
            """, unsafe_allow_html=True)

        # --- NUEVO: SCATTER PLOT COMPARATIVO ---
        st.subheader("Comparativa de métricas: Gráfico de dispersión")
        # Selección de métricas numéricas disponibles
        metricas_numericas_disp = [col for col in df_pos.select_dtypes(include=[np.number]).columns if col not in ['Level Competition']]
        if len(metricas_numericas_disp) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                metrica_x = st.selectbox("Métrica para eje X:", metricas_numericas_disp, key='scatter_x')
            with col2:
                metrica_y = st.selectbox("Métrica para eje Y:", metricas_numericas_disp, index=1 if len(metricas_numericas_disp)>1 else 0, key='scatter_y')
            if metrica_x != metrica_y:
                fig_scatter = px.scatter(
                    df_pos,
                    x=metrica_x,
                    y=metrica_y,
                    color='Age' if 'Age' in df_pos.columns else None,
                    hover_data=['Player', 'Actual Team', 'Competition'] if 'Player' in df_pos.columns else None,
                    title=f"Comparativa: {traduccion_columnas.get(metrica_x, metrica_x)} vs {traduccion_columnas.get(metrica_y, metrica_y)}"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Selecciona dos métricas diferentes para comparar.")
        else:
            st.info("No hay suficientes métricas numéricas para mostrar el scatter plot.")

        # --- NUEVO: BÚSQUEDA DE JUGADORES SIMILARES ---
        st.subheader("Scouting: Buscar jugadores similares")
        if 'Player' in df_pos.columns and len(df_pos) > 1:
            jugador_base = st.selectbox("Selecciona un jugador como referencia:", df_pos['Player'].unique(), key='simil_jugador')
            n_similares = st.slider("¿Cuántos similares mostrar?", 1, min(10, len(df_pos)-1), 3, key='simil_n')
            # Seleccionar solo métricas numéricas relevantes
            metricas_simil = [col for col in df_pos.select_dtypes(include=[np.number]).columns if col not in ['Level Competition']]
            if jugador_base and metricas_simil:
                X = df_pos[metricas_simil].fillna(0).values
                idx_base = df_pos[df_pos['Player'] == jugador_base].index[0]
                modelo_nn = NearestNeighbors(n_neighbors=n_similares+1, metric='euclidean')
                modelo_nn.fit(X)
                distancias, indices = modelo_nn.kneighbors([X[idx_base]])
                similares_idx = [i for i in indices[0] if i != idx_base][:n_similares]
                # --- CORRECCIÓN: evitar columnas duplicadas ---
                cols_similares = list(dict.fromkeys(['Player', 'Actual Team', 'Age', 'Market Value'] + metricas_simil))
                df_similares = df_pos.iloc[similares_idx][cols_similares].copy()
                st.dataframe(df_similares, use_container_width=True)
            else:
                st.info("No hay suficientes métricas numéricas para buscar similares.")
        else:
            st.info("No hay suficientes jugadores para buscar similares.")
    # --- Filtros adicionales ---
    with st.expander("Filtros avanzados (opcional)"):
        if 'Actual Team' in df_pos.columns:
            equipos = sorted(df_pos['Actual Team'].dropna().unique())
            equipos_sel = st.multiselect("Filtra por equipo:", equipos, key='vis_equipo')
            if equipos_sel:
                df_pos = df_pos[df_pos['Actual Team'].isin(equipos_sel)]
        if 'Age' in df_pos.columns:
            min_edad, max_edad = int(df_pos['Age'].min()), int(df_pos['Age'].max())
            edad_sel = st.slider("Filtra por edad:", min_edad, max_edad, (min_edad, max_edad), key='vis_edad')
            df_pos = df_pos[(df_pos['Age'] >= edad_sel[0]) & (df_pos['Age'] <= edad_sel[1])]
        if 'Nationality' in df_pos.columns:
            naciones = sorted(df_pos['Nationality'].dropna().unique())
            naciones_sel = st.multiselect("Filtra por nacionalidad:", naciones, key='vis_nacion')
            if naciones_sel:
                df_pos = df_pos[df_pos['Nationality'].isin(naciones_sel)]
        # --- NUEVO: Filtro por valor de mercado ---
        if 'Market Value' in df_pos.columns:
            min_valor, max_valor = float(df_pos['Market Value'].min()), float(df_pos['Market Value'].max())
            valor_sel = st.slider("Filtra por valor de mercado:", min_valor, max_valor, (min_valor, max_valor), key='vis_valor')
            df_pos = df_pos[(df_pos['Market Value'] >= valor_sel[0]) & (df_pos['Market Value'] <= valor_sel[1])]
        # --- NUEVO: Filtro por score de rendimiento futuro (si existe) ---
        if 'Score Rendimiento Futuro' in df_pos.columns:
            min_score, max_score = float(df_pos['Score Rendimiento Futuro'].min()), float(df_pos['Score Rendimiento Futuro'].max())
            score_sel = st.slider("Filtra por score de rendimiento futuro:", min_score, max_score, (min_score, max_score), key='vis_score')
            df_pos = df_pos[(df_pos['Score Rendimiento Futuro'] >= score_sel[0]) & (df_pos['Score Rendimiento Futuro'] <= score_sel[1])]

# --- TAB 2: Predicción ---
with tabs[1]:
    st.header("Predicción de rendimiento global, liga y equipo recomendado")
    # --- NUEVO: Subpestañas por grupo de posición ---
    sub_tabs = st.tabs(["Defensas", "Mediocentros", "Delanteros"])
    grupos_pred = ["Defensas", "Mediocentros", "Delanteros"]
    # --- Precargar y cachear resultados para cada grupo ---
    @st.cache_data(show_spinner=False)
    def prediccion_por_grupo(grupo):
        subposiciones = posiciones_disponibles[grupo]
        df_grupo = df[df['Position 1'].isin(subposiciones)].copy()
        metricas_numericas = df_grupo.select_dtypes(include=[np.number]).columns.tolist()
        metricas_numericas = [col for col in metricas_numericas if col not in ['Level Competition']]
        columnas_extra = ['Player', 'Actual Team', 'Competition', 'Level Competition']
        columnas_final = list(dict.fromkeys(columnas_extra + metricas_numericas))
        df_grupo = df_grupo[columnas_final]
        return calcular_prediccion(df_grupo, metricas_numericas, subposiciones[0])
    for idx, grupo in enumerate(grupos_pred):
        with sub_tabs[idx]:
            st.subheader(f"Predicción y recomendaciones para {grupo}")
            # --- Selector de subposición ---
            subposiciones = posiciones_disponibles[grupo]
            if grupo == "Delanteros":
                # Agrupación personalizada para delanteros
                agrupaciones = {
                    'Extremos Derechos': ['RW', 'RAMF', 'RWF', 'ED'],
                    'Extremos Izquierdos': ['LW', 'LAMF', 'LWF', 'EI'],
                    'Delantero Centro': ['CF', 'ST', 'DC']
                }
                opciones_subpos = list(agrupaciones.keys())
                subposicion_traducida = st.selectbox("Selecciona subgrupo de delanteros:", opciones_subpos, key=f'pred_subpos_{grupo}')
                subposiciones_grupo = agrupaciones[subposicion_traducida]
                df_subpos = df[df['Position 1'].isin(subposiciones_grupo)].copy()
                columnas_tabla = []
                for sub in subposiciones_grupo:
                    columnas_tabla += metricas_subpos.get(sub, [])
                columnas_tabla = [col for col in dict.fromkeys(columnas_tabla) if col in df_subpos.columns]
            else:
                opciones_subpos = [traduccion_subpos.get(p, p) for p in subposiciones]
                subposicion_traducida = st.selectbox("Selecciona subposición:", opciones_subpos, key=f'pred_subpos_{grupo}')
                codigo_subpos = next((k for k, v in traduccion_subpos.items() if v == subposicion_traducida and k in subposiciones), subposiciones[0])
                df_subpos = df[df['Position 1'] == codigo_subpos].copy()
                columnas_tabla = metricas_subpos.get(codigo_subpos, metricas_grupo[grupo])
                columnas_tabla = [col for col in columnas_tabla if col in df_subpos.columns]
            df_tabla = df_subpos[columnas_tabla].copy()
            df_tabla.columns = [traduccion_columnas.get(col, col) for col in df_tabla.columns]
            st.subheader(f"Tabla de jugadores: {subposicion_traducida}")
            st.dataframe(df_tabla, use_container_width=True, height=400)
            # --- Predicción y recomendaciones para la subposición seleccionada ---
            metricas_numericas = df_subpos.select_dtypes(include=[np.number]).columns.tolist()
            metricas_numericas = [col for col in metricas_numericas if col not in ['Level Competition']]
            if not df_subpos.empty:
                if grupo == "Delanteros":
                    # Para la predicción, usar el primer código de subposiciones del grupo
                    codigo_subpos_pred = subposiciones_grupo[0]
                    df_scores = calcular_prediccion(df_subpos, metricas_numericas, codigo_subpos_pred)
                else:
                    df_scores = calcular_prediccion(df_subpos, metricas_numericas, codigo_subpos)
                df_resultado_viz = pd.DataFrame({
                    'Jugador': df_scores['Player'].values if 'Player' in df_scores.columns else df_scores.index,
                    'Equipo actual': df_scores['Actual Team'].values if 'Actual Team' in df_scores.columns else None,
                    'Liga actual': df_scores['Competition'].values if 'Competition' in df_scores.columns else None,
                    'Score': [f"{x:.2f}" if pd.notna(x) else 'No disponible' for x in df_scores['Score Rendimiento Futuro']],
                    'Liga recomendada': df_scores['Liga recomendada'],
                    'Equipo recomendado': df_scores['Equipo recomendado'],
                    'Cluster': df_scores['Cluster'] if 'Cluster' in df_scores.columns else None
                })
                st.dataframe(df_resultado_viz, use_container_width=True, height=400)
                # --- AVISO SI ALGÚN JUGADOR NO TIENE EQUIPO RECOMENDADO ---
                if (df_resultado_viz['Equipo recomendado'] == '').any():
                    n_sin_equipo = (df_resultado_viz['Equipo recomendado'] == '').sum()
                    st.warning(f"{n_sin_equipo} jugador(es) no tienen equipo recomendado por falta de opciones realistas en el mercado.")
            else:
                st.info("No hay jugadores para esta subposición.")

ligas_por_nivel = df[['Competition', 'Level Competition']].drop_duplicates().sort_values('Level Competition')
# print(ligas_por_nivel)  # Comentado para evitar ralentizar la app 