import pandas as pd
import numpy as np

# === 1. Leer el archivo principal ===
estadisticas_jugadores = pd.read_excel('TOTAL STATS PLAYERS.xlsx')

# === 2. Cálculos de columnas nuevas ===
estadisticas_jugadores['Duelos Ganados por 90'] = round((estadisticas_jugadores['Duels per 90'] * estadisticas_jugadores['Duels Won, %']) / 100, 2)
estadisticas_jugadores['Duelos Defensivos Ganados por 90'] = round((estadisticas_jugadores['Defensive Duels per 90'] * estadisticas_jugadores['Defensive Duels won, %']) / 100, 2)
estadisticas_jugadores['Duelos Aéreos Ganados por 90'] = round((estadisticas_jugadores['Aerial Duels per 90'] * estadisticas_jugadores['Aerial Duels Won, %']) / 100, 2)
estadisticas_jugadores['Tiros a Puerta'] = round((estadisticas_jugadores['Shots'] * estadisticas_jugadores['Shots On Target, %']) / 100, 2)
estadisticas_jugadores['Tiros a Puerta por 90'] = round((estadisticas_jugadores['Shots per 90'] * estadisticas_jugadores['Shots On Target, %']) / 100, 2)
estadisticas_jugadores['Centros Precisos por 90'] = round((estadisticas_jugadores['Crosses per 90'] * estadisticas_jugadores['Accurate Crosses, %']) / 100, 2)
estadisticas_jugadores['Centros Precisos por Izquierda por 90'] = round((estadisticas_jugadores['Crosses from Left Flank per 90'] * estadisticas_jugadores['Accurate Crosses from Left Flank, %']) / 100, 2)
estadisticas_jugadores['Centros Precisos por Derecha por 90'] = round((estadisticas_jugadores['Crosses from Right Flank per 90'] * estadisticas_jugadores['Accurate Crosses from Right Flank, %']) / 100, 2)
estadisticas_jugadores['Regates Exitosos por 90'] = round((estadisticas_jugadores['Dribbles per 90'] * estadisticas_jugadores['Successful Dribbles, %']) / 100, 2)
estadisticas_jugadores['Duelos Ofensivos Ganados por 90'] = round((estadisticas_jugadores['Offensive Duels per 90'] * estadisticas_jugadores['Offensive Duels Won, %']) / 100, 2)
estadisticas_jugadores['Pases Precisos por 90'] = round((estadisticas_jugadores['Passes per 90'] * estadisticas_jugadores['Accurate Passes, %']) / 100, 2)
estadisticas_jugadores['Pases Precisos Adelante por 90'] = round((estadisticas_jugadores['Forward Passes per 90'] * estadisticas_jugadores['Accurate Forward Passes, %']) / 100, 2)
estadisticas_jugadores['Pases Precisos Atrás por 90'] = round((estadisticas_jugadores['Back Passes per 90'] * estadisticas_jugadores['Accurate Back Passes, %']) / 100, 2)
estadisticas_jugadores['Pases Laterales Precisos por 90'] = round((estadisticas_jugadores['Lateral Passes per 90'] * estadisticas_jugadores['Accurate Lateral Passes, %']) / 100, 2)
estadisticas_jugadores['Pases Cortos/Medios Precisos por 90'] = round((estadisticas_jugadores['Short / Medium Passes per 90'] * estadisticas_jugadores['Accurate Short / Medium Passes, %']) / 100, 2)
estadisticas_jugadores['Pases Largos Precisos por 90'] = round((estadisticas_jugadores['Long Passes per 90'] * estadisticas_jugadores['Accurate Long Passes, %']) / 100, 2)
estadisticas_jugadores['Pases Inteligentes Precisos por 90'] = round((estadisticas_jugadores['Smart Passes per 90'] * estadisticas_jugadores['Accurate Smart Passes, %']) / 100, 2)
estadisticas_jugadores['Pases Precisos a Último Tercio por 90'] = round((estadisticas_jugadores['Passes to Final Third per 90'] * estadisticas_jugadores['Accurate Passes to Final Third, %']) / 100, 2)
estadisticas_jugadores['Pases Precisos a Área por 90'] = round((estadisticas_jugadores['Passes to Penalty Area per 90'] * estadisticas_jugadores['Accurate Passes to Penalty Area, %']) / 100, 2)
estadisticas_jugadores['Pases al Hueco Precisos por 90'] = round((estadisticas_jugadores['Through Passes per 90'] * estadisticas_jugadores['Accurate Through Passes, %']) / 100, 2)
estadisticas_jugadores['Pases Progresivos Precisos por 90'] = round((estadisticas_jugadores['Progressive Passes per 90'] * estadisticas_jugadores['Accurate Progressive Passes, %']) / 100, 2)
estadisticas_jugadores['Faltas Directas a Puerta por 90'] = round((estadisticas_jugadores['Direct Free Kicks per 90'] * estadisticas_jugadores['Direct Free Kicks On Target, %']) / 100, 2)
estadisticas_jugadores['Penaltis Marcados'] = round((estadisticas_jugadores['Penalties Taken'] * estadisticas_jugadores['Penalty Conversion, %']) / 10, 2)

estadisticas_jugadores.to_excel('TOTAL STATS PLAYERS.xlsx', index=False)

# === 3. Leer coeficiente de ligas y hacer merge ===
coeficientes_ligas = pd.read_excel('Leagues Coeficient.xlsx')
estadisticas_jugadores = pd.merge(estadisticas_jugadores, coeficientes_ligas, on='Competition')
# Reubicar columna 'Competition' después de la cuarta columna
columnas = list(estadisticas_jugadores.columns)
columnas.insert(4, columnas.pop(columnas.index('Competition')))
estadisticas_jugadores = estadisticas_jugadores[columnas]

# === 4. Convertir columnas de posición a string ===
for columna in ['Position 1', 'Position 2', 'Position 3']:
    estadisticas_jugadores[columna] = estadisticas_jugadores[columna].astype(str)

# === 5. Función para procesar cada posición ===
def procesar_posicion(df, posiciones, nombre_posicion, nombre_archivo):
    columnas_excluidas = ['Player', 'Actual Team', 'Owner Team', 'Competition', 'On Loan', 'Market Value', 'Contract Expires', 'Position 1', 'Position 2', 'Position 3', 'Birth Country', 'Passport Country', 'Age', 'Matches Played', 'Minutes Played', 'Height', 'Weight', 'Foot', 'Level Competition']
    filtro = df['Position 1'].isin(posiciones)
    scouting = df[filtro].copy()
    for columna in scouting.columns:
        if columna not in columnas_excluidas and np.issubdtype(scouting[columna].dtype, np.number):
            scouting[columna] = scouting[columna] * scouting['Average SPI']
    scouting['POSICION'] = nombre_posicion
    scouting['Indice'] = scouting['Player'] + '.' + scouting['Actual Team'] + '.' + scouting['Competition'] + '.' + scouting['POSICION']
    columnas = list(scouting.columns)
    columnas.insert(2, columnas.pop(columnas.index('Indice')))
    scouting = scouting[columnas]
    columnas_eliminar = ['Actual Team', 'Owner Team', 'Competition', 'On Loan', 'Market Value', 'Contract Expires', 'Position 1', 'Position 2', 'Position 3', 'Age', 'Birth Country', 'Passport Country', 'Foot', 'Height', 'Weight', 'Matches Played', 'Average SPI', 'Level Competition', 'POSICION']
    scouting = scouting.drop(columns=columnas_eliminar)
    scouting.to_excel(nombre_archivo, index=False)
    return scouting

# === 6. Procesar todas las posiciones ===
posiciones_dict = {
    'central_defensa': (['CB', 'LCB', 'RCB'], 'Defensa Central', 'central_defensa_scouting.xlsx'),
    'lateral_izquierdo': (['LB', 'LWB'], 'Lateral Izquierdo', 'lateral_izquierdo_scouting.xlsx'),
    'lateral_derecho': (['RB', 'RWB'], 'Lateral Derecho', 'lateral_derecho_scouting.xlsx'),
    'mediocentro_defensivo': (['DMF', 'LDMF', 'RDMF', 'RCMF', 'LCMF'], 'Mediocentro Defensivo', 'mediocentro_defensivo_scouting.xlsx'),
    'mediocentro_central': (['DMF', 'LDMF', 'RDMF', 'RCMF', 'LCMF'], 'Mediocentro Central', 'mediocentro_central_scouting.xlsx'),
    'mediapunta': (['AMF'], 'Mediapunta', 'mediapunta_scouting.xlsx'),
    'extremo_izquierdo': (['LAMF', 'LWF', 'LW'], 'Extremo Izquierdo', 'extremo_izquierdo_scouting.xlsx'),
    'extremo_derecho': (['RAMF', 'RWF', 'RW'], 'Extremo Derecho', 'extremo_derecho_scouting.xlsx'),
    'delantero_centro': (['CF'], 'Delantero Centro', 'delantero_centro_scouting.xlsx'),
}

for clave, (posiciones, nombre_posicion, nombre_archivo) in posiciones_dict.items():
    procesar_posicion(estadisticas_jugadores, posiciones, nombre_posicion, nombre_archivo)

# === 7. Unir todos los archivos de posiciones ===
archivos = [
    'central_defensa_scouting.xlsx',
    'lateral_izquierdo_scouting.xlsx',
    'lateral_derecho_scouting.xlsx',
    'mediocentro_defensivo_scouting.xlsx',
    'mediocentro_central_scouting.xlsx',
    'mediapunta_scouting.xlsx',
    'extremo_izquierdo_scouting.xlsx',
    'extremo_derecho_scouting.xlsx',
    'delantero_centro_scouting.xlsx'
]
dfs = [pd.read_excel(f) for f in archivos]
scouting_jugadores_normal = pd.concat(dfs, ignore_index=True)
scouting_jugadores_normal.to_excel('scouting_jugadores_normal.xlsx', index=False)

# === 8. Normalización y ratings ===
# Leer archivos normalizados y agregar columna de posición
archivos_normalizados = [
    ('central_defensa_normalizado.xlsx', 'Defensa Central'),
    ('lateral_izquierdo_normalizado.xlsx', 'Lateral Izquierdo'),
    ('lateral_derecho_normalizado.xlsx', 'Lateral Derecho'),
    ('mediocentro_defensivo_normalizado.xlsx', 'Mediocentro Defensivo'),
    ('mediocentro_central_normalizado.xlsx', 'Mediocentro Central'),
    ('mediapunta_normalizado.xlsx', 'Mediapunta'),
    ('extremo_izquierdo_normalizado.xlsx', 'Extremo Izquierdo'),
    ('extremo_derecho_normalizado.xlsx', 'Extremo Derecho'),
    ('delantero_centro_normalizado.xlsx', 'Delantero Centro'),
]
dfs_norm = []
for archivo, posicion in archivos_normalizados:
    try:
        df = pd.read_excel(archivo)
        df['POSICION'] = posicion
        dfs_norm.append(df)
    except Exception as e:
        print(f"No se pudo leer {archivo}: {e}")

scouting_jugadores_normalizado = pd.concat(dfs_norm, ignore_index=True)
# Eliminar columna ...1 si existe
scouting_jugadores_normalizado = scouting_jugadores_normalizado.drop(columns=['...1'], errors='ignore')

# Calcular RATING TOT como la media de los ratings principales
ratings_cols = ["DEFENDING", "ATTACKING", "PASSING", "CREATION", "DRIBBLING", "PHYSIQUE"]
scouting_jugadores_normalizado['RATING TOT'] = np.round(scouting_jugadores_normalizado[ratings_cols].mean(axis=1), 0)

# Reubicar columnas como en el código R
def mover_columna(df, columna, pos):
    cols = list(df.columns)
    cols.insert(pos, cols.pop(cols.index(columna)))
    return df[cols]

scouting_jugadores_normalizado = mover_columna(scouting_jugadores_normalizado, 'RATING TOT', 3)
scouting_jugadores_normalizado = mover_columna(scouting_jugadores_normalizado, 'DEFENDING', 4)
scouting_jugadores_normalizado = mover_columna(scouting_jugadores_normalizado, 'ATTACKING', 5)
scouting_jugadores_normalizado = mover_columna(scouting_jugadores_normalizado, 'PASSING', 6)
scouting_jugadores_normalizado = mover_columna(scouting_jugadores_normalizado, 'CREATION', 7)
scouting_jugadores_normalizado = mover_columna(scouting_jugadores_normalizado, 'DRIBBLING', 8)
scouting_jugadores_normalizado = mover_columna(scouting_jugadores_normalizado, 'PHYSIQUE', 9)
scouting_jugadores_normalizado = mover_columna(scouting_jugadores_normalizado, 'POSICION', 10)

scouting_jugadores_normalizado.to_excel('scouting_jugadores_normalizado.xlsx', index=False)

print('Procesamiento completado con ratings y normalización.') 