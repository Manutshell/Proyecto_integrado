import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as st


df_games = pd.read_csv('D:/TRIPLETEN DISCO 2/datasets/games.csv')


# Preparacion de datos.

print(df_games)

# Reemplaza los nombres de las columnas (ponlos en minúsculas)
columns_lower = []
for col in df_games.columns:
    columns_lower.append(col.lower())
df_games.columns = columns_lower

print(df_games.columns)


# Convierte los datos en los tipos necesarios

df_games.info() # Cambiaré el tipo de dato de la columna 'user_score' a float ya que son numeros 

df_games['user_score']= pd.to_numeric(df_games['user_score'], errors='coerce') # Primero los valores de TBD decidi considerarlos como vacios y si en algun punto se determinan solo añadirlos al videojuego determinado. 

print(df_games.info())

# Si es necesario, elige la manera de tratar los valores ausentes

df_games.isna().sum()

df_games['name'] = df_games['name'].fillna('unknown')
df_games['genre'] = df_games['genre'].fillna('unknown') # De las columnas 'name' y 'genre' reemplace los valores ausentes con 'unknown' para establecer un apartado de estos titulos. 

# Mientras que 'year_of_release', 'critic_score', 'user_score' y 'rating' no es necesario poner un valor para su analisis. 

# Calcula las ventas totales (la suma de las ventas en todas las regiones) para cada juego y coloca estos valores en una columna separada.

df_games['total_sales'] = df_games['na_sales'] + df_games['eu_sales'] + df_games['jp_sales'] + df_games['other_sales']

df_games[df_games['name'].duplicated()]

df_games_total = df_games.groupby('name')['total_sales'].sum().reset_index() # Creo un data frame con la suma de ventas por videojuego incluyendo todas las plataformas. 



# Analisis de datos
# Mira cuántos juegos fueron lanzados en diferentes años. ¿Son significativos los datos de cada período?


df_games.groupby('year_of_release')['total_sales'].sum() # Ventas generadas por año. 

df_games.groupby('year_of_release')['name'].count() # Conteo de videojuegos por año

# De 1980 hasta 1990 no se vendieron mas de 300 juegos en total. Por lo que de este periodo son los menos significativos 


# Observa cómo varían las ventas de una plataforma a otra. Elige las plataformas con las mayores ventas totales y construye 
# una distribución basada en los datos de cada año. Busca las plataformas que solían ser populares pero que ahora no tienen ventas. 

df_games.groupby(['platform'])['total_sales'].sum().sort_values(ascending = False)
top_platforms = df_games.groupby(['platform', 'year_of_release'])['total_sales'].sum().reset_index()
ps2 = top_platforms[top_platforms['platform'] == 'PS2']
del ps2['platform']
x360 = top_platforms[top_platforms['platform'] == 'X360']
del x360['platform']

ps3= top_platforms[top_platforms['platform'] == 'PS3']
del ps3['platform']
xb = top_platforms[top_platforms['platform'] == 'XB']
del xb['platform']
n64 = top_platforms[top_platforms['platform'] == 'N64']
del n64['platform']
xone = top_platforms[top_platforms['platform'] == 'XOne']
del xone['platform']
wii = top_platforms[top_platforms['platform'] == 'Wii']
del wii['platform']
ps4 = top_platforms[top_platforms['platform'] == 'PS4']
del ps4['platform']
ds3 = top_platforms[top_platforms['platform'] == '3DS']
del ds3['platform']


merged_df = pd.merge(ps2, x360, on='year_of_release', how='outer', suffixes=('_ps2', '_x360'))
merged_df = pd.merge(ps3, merged_df, on='year_of_release', how='outer')
merged_df.rename(columns={'total_sales': 'total_sales_ps3'}, inplace=True)
merged_df = pd.merge(xb, merged_df, on='year_of_release', how='outer')
merged_df.rename(columns={'total_sales': 'total_sales_xb'}, inplace=True)
merged_df = pd.merge(n64, merged_df, on='year_of_release', how='outer')
merged_df.rename(columns={'total_sales': 'total_sales_N64'}, inplace=True)
merged_df = pd.merge(xone, merged_df, on='year_of_release', how='outer')
merged_df.rename(columns={'total_sales': 'total_sales_XOne'}, inplace=True)
merged_df = pd.merge(wii, merged_df, on='year_of_release', how='outer')
merged_df.rename(columns={'total_sales': 'total_sales_Wii'}, inplace=True)
merged_df = pd.merge(ps4, merged_df, on='year_of_release', how='outer')
merged_df.rename(columns={'total_sales': 'total_sales_PS4'}, inplace=True)
merged_df = pd.merge(ds3, merged_df, on='year_of_release', how='outer')
merged_df.rename(columns={'total_sales': 'total_sales_3DS'}, inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(merged_df['year_of_release'], merged_df['total_sales_ps2'], marker='o', label='Ventas PS2')
plt.plot(merged_df['year_of_release'], merged_df['total_sales_x360'], marker='x', label='Ventas X360')
plt.plot(merged_df['year_of_release'], merged_df['total_sales_ps3'], marker='o', label='Ventas PS3')
plt.plot(merged_df['year_of_release'], merged_df['total_sales_xb'], marker='x', label='Ventas XB')
plt.plot(merged_df['year_of_release'], merged_df['total_sales_N64'], marker='^', label='Ventas N64')
plt.plot(merged_df['year_of_release'], merged_df['total_sales_XOne'], marker='x', label='Ventas XOne')
plt.plot(merged_df['year_of_release'], merged_df['total_sales_Wii'], marker='^', label='Ventas Wii')
plt.plot(merged_df['year_of_release'], merged_df['total_sales_PS4'], marker='o', label='Ventas PS4')
plt.plot(merged_df['year_of_release'], merged_df['total_sales_3DS'], marker='^', label='Ventas 3DS')
# Añadir detalles a la gráfica
plt.title('Comparación de plataformas')
plt.xlabel('Año')
plt.ylabel('Ventas')
plt.legend()
plt.grid()
plt.show()

# ¿Cuánto tardan generalmente las nuevas plataformas en aparecer y las antiguas en desaparecer?

# R: Basándome en la gráfica realizada, las nuevas plataformas suelen aparecer cada 5-6 años, como se observa entre la llegada de PS2 (2000), Xbox (2000), 
#y Xbox 360/PS3 (2005-2006). Por otro lado, las plataformas antiguas suelen tardar entre 5 y 7 años en desaparecer, 
#como se ve con la caída de la PS2 después de 2005 y su salida alrededor de 2010.

# Determina para qué período debes tomar datos. Para hacerlo mira tus respuestas a las preguntas anteriores.
# Los datos deberían permitirte construir un modelo para 2017.

# R: Para conocer los datos lo adecuado es tomar plataformas desde 1996 hasta el 2016 para justo ver el periodo de crecimiento por plataformas. 

# ¿Qué plataformas son líderes en ventas? ¿Cuáles crecen y cuáles se reducen? Elige varias plataformas potencialmente rentables.

# R: Las plataformas lideres a la fecha de 2016, son PS4, XOne, 3DS y PS3. Las 4 plataformas fueron en caida pero la que tiene mayor cantidad de ventas es PS4.





# Crea un diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma. 
# ¿Son significativas las diferencias en las ventas? ¿Qué sucede con las ventas promedio en varias plataformas? Describe tus hallazgos.

df_platform = df_games.groupby('platform')['total_sales'].sum().sort_values(ascending=False).reset_index()



df_ps3 = df_games[df_games['platform'] == 'PS3'] 
df_ps3 = df_ps3[['platform', 'total_sales']]
df_x360 = df_games[df_games['platform'] == 'X360'] 
df_x360 = df_x360[['platform', 'total_sales']]

sns.boxplot(x="platform", y="total_sales", data=df_ps3, palette="Set2")
sns.boxplot(x="platform", y="total_sales", data=df_x360, palette="Set2")

# Personalización del gráfico
plt.title("Ventas Globales de todos los juegos de PS3 y X360")
plt.ylabel("Ventas")
plt.xlabel("Plataformas")

# Mostrar el gráfico
plt.show()

# Comparando PS3 y X360 podemos observar que hay poca diferencia ya que las ventas 
# promedio son menores a 1 y las maximas son de aproximadamente 2.5 arrojando varios datos anomalos 
# de entre 2.5 a 10. Dando pocos videojuegos cantidades de entre 10 y 20. 




# Mira cómo las reseñas de usuarios y profesionales afectan las ventas de una plataforma popular (tu elección). 
# Crea un gráfico de dispersión y calcula la correlación entre las reseñas y las ventas. Saca conclusiones.

df_ps2 = df_games[df_games['platform'] == 'PS2']
x= df_ps2['critic_score']
y=df_ps2['total_sales']

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', alpha=0.4, edgecolor='k', label='Datos')
plt.title('Correlacion entre reseña de criticos y ventas', fontsize=16)
plt.xlabel('Critic Score', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

x2= df_ps2['user_score']
y2=df_ps2['total_sales']

plt.figure(figsize=(8, 6))
plt.scatter(x2, y2, color='blue', alpha=0.4, edgecolor='k', label='Datos')
plt.title('Correlacion entre reseña de usuarios y ventas', fontsize=16)
plt.xlabel('User Score', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# Ambas graficas son similares, hay muchos videojuegos de la PS2 que caen en ventas de 0 a 2 con reseñas de 80 de calificacion. 
# Y las ventas que superan los 2.5 van desde 60 de reseña y apartir de 80 de calificacion se ven mas reflejados. 

# Teniendo en cuenta tus conclusiones compara las ventas de los mismos juegos en otras plataformas.
df_games_duplicated = df_games[df_games['name'].duplicated()].sort_values(by='total_sales', ascending=False)

df_games_duplicated = df_games_duplicated[['name', 'platform', 'total_sales']]
filtro = df_games_duplicated['name'].isin(['Grand Theft Auto V', 'Call of Duty: Black Ops II', 'Call of Duty: Modern Warfare 3', 'Call of Duty: Black Ops'])
df_filtrado = df_games_duplicated[filtro]
# Crear la gráfica
plt.figure(figsize=(12, 6))
sns.barplot(data=df_filtrado, x='name', y='total_sales', hue='platform')

# Etiquetas y título

plt.title('Comparacion de juegos en distintas plataformas')
plt.xlabel('Videojuego')
plt.ylabel('Ventas')
plt.legend(title='Plataforma')

# Mostrar la gráfica
plt.show()



# Echa un vistazo a la distribución general de los juegos por género. 
# ¿Qué se puede decir de los géneros más rentables? ¿Puedes generalizar acerca de los géneros con ventas altas y bajas?

genre_total_sales = df_games.groupby('genre')['total_sales'].sum().reset_index().sort_values(ascending=False, by='total_sales')

games_genre = df_games.groupby('genre')['name'].count().reset_index()
games_genre.columns = ['genre', 'count_games']

df_games_genre_sales = pd.merge(genre_total_sales, games_genre, on='genre', how='outer')

df_games_genre_sales = df_games_genre_sales.sort_values(by='total_sales', ascending=False)

df_games_genre_sales['distribucion'] = df_games_genre_sales['total_sales'] / df_games_genre_sales['count_games']

# La distribucion de cuantos juegos por genero hay y lo que generan. El que mayor eficiencia tiene por juego es el genero de platform, seguido de shooter y Role-Playing. 
# Despues de esos la ditribucion no cambio por lo que ponemos determinar que sí, los generos con ventas mas altas son los mas rentables. 


# Para cada región (NA, UE, JP) determina:
# Las cinco plataformas principales.
# Los cinco géneros principales.

df_games_na_platform = df_games.groupby('platform')['na_sales'].sum().reset_index().sort_values(by='na_sales', ascending=False)
df_games_na_platform = df_games_na_platform.head(5)
df_games_eu_platform = df_games.groupby('platform')['eu_sales'].sum().reset_index().sort_values(by='eu_sales', ascending=False)
df_games_eu_platform = df_games_eu_platform.head(5)
df_games_jp_platform = df_games.groupby('platform')['jp_sales'].sum().reset_index().sort_values(by='jp_sales', ascending=False)
df_games_jp_platform = df_games_jp_platform.head(5)
df_games_platform_sales = pd.merge(df_games_na_platform, df_games_eu_platform, on='platform', how='outer')
df_games_platform_sales = pd.merge(df_games_platform_sales, df_games_jp_platform, on='platform', how='outer')

df_games_platform_sales.set_index('platform', inplace=True)

# Crear la gráfica de barras
df_games_platform_sales.plot(kind='bar', figsize=(8, 5))

# Personalizar la gráfica
plt.title("Plataformas principales por region.")
plt.ylabel("Ventas")
plt.xlabel("Plataformas")
plt.xticks(rotation=0)  # Mantener etiquetas horizontales
plt.legend(title="Columnas")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar la gráfica
plt.tight_layout()
plt.show()





df_games_na_genre = df_games.groupby('genre')['na_sales'].sum().reset_index().sort_values(by='na_sales', ascending=False)
df_games_na_genre = df_games_na_genre.head(5)
df_games_eu_genre = df_games.groupby('genre')['eu_sales'].sum().reset_index().sort_values(by='eu_sales', ascending=False)
df_games_eu_genre = df_games_eu_genre.head(5)
df_games_jp_genre = df_games.groupby('genre')['jp_sales'].sum().reset_index().sort_values(by='jp_sales', ascending=False)
df_games_jp_genre = df_games_jp_genre.head(5)
df_games_genre_sales = pd.merge(df_games_na_genre, df_games_eu_genre, on='genre', how='outer')
df_games_genre_sales = pd.merge(df_games_genre_sales, df_games_jp_genre, on='genre', how='outer')
df_games_genre_sales.set_index('genre', inplace=True)

# Crear la gráfica de barras
df_games_genre_sales.plot(kind='bar', figsize=(8, 5))

# Personalizar la gráfica
plt.title("Generos principales por region.")
plt.ylabel("Ventas")
plt.xlabel("Generos")
plt.xticks(rotation=0)  # Mantener etiquetas horizontales
plt.legend(title="Columnas")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar la gráfica
plt.tight_layout()
plt.show()


#  Describe las variaciones en sus cuotas de mercado de una región a otra.
#  Los cinco géneros principales. Explica la diferencia 

# En todos los casos Norte america es la region que más aporta, segido en tendencia union europea, mientras que Japon cobre otro mercado añadiendo otras platafomas
# Los generos principales, parece similar a las cuotas por plataforma, norte america es la region que más aporta, seguido en tendencia union europea, Japon incluye generos diferentes. 

# # Si las clasificaciones de ESRB afectan a las ventas en regiones individuales.

df_rating_regiones = df_games.groupby('rating')[['na_sales', 'eu_sales', 'jp_sales']].sum().reset_index().sort_values(by='na_sales', ascending=False)
df_rating_regiones.set_index('rating', inplace=True)
# Crear la gráfica de barras
df_rating_regiones.plot(kind='bar', figsize=(8, 5))

# Personalizar la gráfica
plt.title("Ventas por rating por regiones")
plt.ylabel("Ventas")
plt.xlabel("Rating")
plt.xticks(rotation=0)  # Mantener etiquetas horizontales
plt.legend(title="Columnas")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar la gráfica
plt.tight_layout()
plt.show()

# R: En todas las regiones mientras máa alta la clasificacion menos ventas obtiene. 
# R: Menos en la union europea ya que tienen mas ventas en Rating M que en rating T




# Paso 5. Prueba las siguientes hipótesis:
# — Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.
# Establece tu mismo el valor de umbral alfa.
# Explica: — Cómo formulaste las hipótesis nula y alternativa. — Qué criterio utilizaste para probar las hipótesis y por qué.

df_games_xone = df_games[df_games['platform'] == 'XOne']
df_xone_user_score = df_games_xone['user_score'].dropna()

df_games_pc = df_games[df_games['platform'] == 'PC']
df_pc_user_score = df_games_pc['user_score'].dropna()

# Hipotesis nula "Las calificaciones promedio son iguales de Xbox One y PC"
# Hipotesis alternativa "Las calificaciones promedio son diferentes de Xbox One y PC" 

alpha = .05 # establece un nivel crítico de significación estadística

results = st.ttest_ind(df_xone_user_score, df_pc_user_score, equal_var = False) # prueba la hipótesis de que las medias de las dos poblaciones independientes son iguales

print('valor p:', results.pvalue) 

if results.pvalue < alpha: # compara los valores p obtenidos con el nivel de significación estadística
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


# — Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.

games_action = df_games[df_games['genre'] == 'Action']
df_action_users_score = games_action['user_score'].dropna()

games_sports = df_games[df_games['genre'] == 'Sports']
df_sports_users_score = games_sports['user_score'].dropna()

alpha = .05 # establece un nivel crítico de significación estadística

results = st.ttest_ind(df_action_users_score, df_sports_users_score, equal_var = False) # prueba la hipótesis de que las medias de las dos poblaciones independientes son iguales

print('valor p:', results.pvalue) 

if results.pvalue < alpha: # compara los valores p obtenidos con el nivel de significación estadística
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")
