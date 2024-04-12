import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MultipleLocator
import plotly.graph_objs as go
import plotly.express as px


# Ruta de mi matriz binaria de datos
ruta_excel = 'Matriz_binaria.xlsx'

# Cargo la matriz binaria directamente de excel
df = pd.read_excel(ruta_excel, index_col=0)

# Mostrar mi dataframe
# print(df)

#Tabla Frecuencias

# Obtengo mi tabla de frecuencias tanto en valores como en porcentajes
tabla_frecuencias = df.apply(lambda col: col.value_counts()).T #Para cada columna (col-tematica) aplico lambda, calculo asi las frecuencias de los valores (0 y 1) mediante value_counts().
#El .T consigue trasponer el DataFrame, intercambiando filas por columnas y viceversa, para que aparezcan en el formato que yo quiero.
tabla_frecuencias.columns = ['Ausencia', 'Presencia'] #Asigno nombres a las columnas
tabla_frecuencias['Total'] = tabla_frecuencias['Ausencia'] + tabla_frecuencias['Presencia'] #Asigno una columna llamada total, que tendra el valor de ausencia mas presencia, es decir, total de individuos
tabla_frecuencias['Porcentaje Ausencia'] = tabla_frecuencias['Ausencia'] / tabla_frecuencias['Total'] * 100 #Nombro columna y calculo el porcentaje de ausencia
tabla_frecuencias['Porcentaje Presencia'] = tabla_frecuencias['Presencia'] / tabla_frecuencias['Total'] * 100 #Nombro columna y calculo el porcentaje de presencia

print(tabla_frecuencias)

#Grafico de Barras Con Porcentajes de Frecuencias

# Creo con esto el marco donde ira el grafico
fig, ax = plt.subplots(figsize=(10, 6)) 

bar_width = 0.35  # Le doy un ancho a las barras
index = np.arange(len(tabla_frecuencias)) #Guardo en index un array con valores desde 0 hasta la longitud de tabla_frecuencias menos 1, para referenciar las barras en el gráfico

# Preparo las barras para las ausencias
rects1 = ax.barh(index, tabla_frecuencias['Porcentaje Ausencia'], bar_width, label='Ausencia', alpha=0.7) #Dibujo aqui las barras horizontales para los porcentajes de ceros de cada temática. La posición en el eje estará determinada por el index anterior, y la longitud de las barras en el eje x estará determinada por los porcentajes de ceros.
#Alpha establece la opacidad de las barras (1-totalmente opacas)

# Preparo las barras para las presencias
rects2 = ax.barh(index + bar_width, tabla_frecuencias['Porcentaje Presencia'], bar_width, label='Presencia', alpha=0.7) #Lo mismo que para las ausencias, pero ahora con las presencias

# Configuro el grafico, etiquetas y marcas en el eje. Tambien establezco la leyenda para distinguir las barras correspondientes.
ax.set_ylabel('Temáticas')
ax.set_xlabel('Porcentaje respecto al 100%')
ax.set_title('Porcentaje de Ausencia y Presencia para cada Temática')
ax.set_yticks(index + bar_width / 2)
ax.set_yticklabels(tabla_frecuencias.index)

# Ajusto la leyenda para que no se superponga
ax.legend(loc='lower right', bbox_to_anchor=(1.1, 1.02))

# Muestro el gráfico
# plt.show()

#Grafico de Barras Con Valores de Frecuencias Acumuladas

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(tabla_frecuencias.index, tabla_frecuencias['Presencia'], label='Presencia', alpha=0.7)

rects2 = ax.bar(tabla_frecuencias.index, tabla_frecuencias['Ausencia'], bottom=tabla_frecuencias['Presencia'], label='Ausencia', alpha=0.7) #Para apilar la barra de ausencia, encima de la de presencia

ax.set_xlabel('Temáticas')
ax.set_ylabel('Total de Individuos')
ax.set_title('Frecuencia de Ausencia y Presencia por Temática')
ax.set_xticks(tabla_frecuencias.index)
ax.set_xticklabels(tabla_frecuencias.index, rotation=90)  # Roto las etiquetas de las tematicas en vertical, para que no se superpongan
ax.legend(loc='lower right', bbox_to_anchor=(1.1, 1.02))

# plt.show()


# Selecciono solo las columnas de porcentajes de 'Presencia'
porcentajes_presencia = tabla_frecuencias['Porcentaje Presencia']

# Configurar el gráfico radial, lo que significa entre otras cosas, que el gráfico será en coordenadas polares en lugar de cartesianas
theta = np.linspace(0, 2*np.pi, len(porcentajes_presencia), endpoint=False) 
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# Dibujo las líneas radiales
ax.set_theta_offset(np.pi/2)  # Ajusto la posición inicial de las líneas
ax.set_theta_direction(-1)  # Le doy el sentido de las agujas del reloj
ax.set_rlabel_position(0)  # Etiquetas en el centro del gráfico
ax.plot(theta, porcentajes_presencia, label='Porcentaje Presencia', marker='o') #Trazo la línea que conecta los puntos dados por theta y porcentajes_presencia

# Añado las etiquetas de cada temática
ax.set_xticks(theta)
ax.set_xticklabels(porcentajes_presencia.index)

# Configuro el gráfico
ax.set_title('Porcentaje de Presencia por Temática en Gráfico Radial')

ax.legend(loc='lower right', bbox_to_anchor=(1.1, 0.97))

#plt.show()

# Ordeno el DataFrame por la columna 'Presencia' en orden descendente
tabla_frecuencias_ordenada = tabla_frecuencias.sort_values(by='Presencia', ascending=False) 

# Selecciono las cuatro primeras filas (las temáticas con mayor presencia)
tematicas_top4 = tabla_frecuencias_ordenada.head(4)

# Selecciono los valores de las columnas relevantes para el gráfico radial
porcentajes_top4 = tematicas_top4['Porcentaje Presencia']
theta_top4 = np.linspace(0, 2*np.pi, len(porcentajes_top4), endpoint=False)

# Configuro el gráfico
fig_top4, ax_top4 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# Dibujo las líneas radiales para las temáticas top 4
ax_top4.set_theta_offset(np.pi/2)
ax_top4.set_theta_direction(-1)
ax_top4.set_rlabel_position(0)
ax_top4.plot(theta_top4, porcentajes_top4, label='Porcentaje Presencia', marker='o', color='red')  # Cambio el color a rojo

# Añado las etiquetas
ax_top4.set_xticks(theta_top4)
ax_top4.set_xticklabels(porcentajes_top4.index)

ax_top4.set_title('Porcentaje de Presencia por Temática en Gráfico Radial (Top 4)')

ax_top4.legend(loc='lower right', bbox_to_anchor=(1.1, 0.97))

# plt.show()


# Ordeno el DataFrame por la columna 'Presencia' en orden ascendente para obtener las menos frecuentes
tabla_frecuencias_ordenada_asc = tabla_frecuencias.sort_values(by='Presencia', ascending=True)

# Selecciono las cuatro primeras filas (las temáticas con menor presencia)
tematicas_bottom4 = tabla_frecuencias_ordenada_asc.head(4)

# Selecciono solo las columnas relevantes para el gráfico radial
porcentajes_bottom4 = tematicas_bottom4['Porcentaje Presencia']
theta_bottom4 = np.linspace(0, 2*np.pi, len(porcentajes_bottom4), endpoint=False)


fig_bottom4, ax_bottom4 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# Dibujo las líneas radiales para las temáticas bottom 4
ax_bottom4.set_theta_offset(np.pi/2)
ax_bottom4.set_theta_direction(-1)
ax_bottom4.set_rlabel_position(0)
ax_bottom4.plot(theta_bottom4, porcentajes_bottom4, label='Porcentaje Presencia', marker='o', color='green')

# Añado etiquetas
ax_bottom4.set_xticks(theta_bottom4)
ax_bottom4.set_xticklabels(porcentajes_bottom4.index)

ax_bottom4.set_title('Porcentaje de Presencia por Temática en Gráfico Radial (Bottom 4)')

ax_bottom4.legend(loc='lower right', bbox_to_anchor=(1.1, 0.97))

# plt.show()


# Crear el pairplot y personalizar las etiquetas del eje y
g = sns.pairplot(df, diag_kind='kde')

# Truncar todas las etiquetas del eje y a las primeras 5 letras
etiquetas_truncadas = [col[:5] for col in df.columns]
for ax, etiqueta in zip(g.axes[:, 0], etiquetas_truncadas):
    ax.set_ylabel(etiqueta)

# Mostrar el plot
plt.show()


# excluye las columnas no numéricas (en este caso, la primera columna)
numeric_df = df.select_dtypes(include=['number'])

# Calcula la matriz de correlación
correlation_matrix = numeric_df.corr()

# Crea un mapa de calor
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Personaliza los bordes y ajusta la posición de las etiquetas del eje Y
bottom, top = heatmap.get_ylim()
heatmap.set_ylim(bottom + 0.5, top - 0.5)

# Ajusta la posición de la visualización hacia la derecha y agranda la figura
plt.subplots_adjust(left=0.25, right=0.95)

# Rota las etiquetas del eje X para mayor legibilidad
plt.xticks(rotation=45, ha='right')  # Alinea las etiquetas a la derecha

# Ajusta manualmente el espaciado entre las etiquetas del eje X y el eje
plt.subplots_adjust(bottom=0.2)

plt.title('Matriz de correlación')
plt.show()


# Ruta de mi matriz binaria de datos
ruta_excel5 = 'Matriz_binaria5.xlsx'

# Cargo la matriz binaria directamente de excel
df4 = pd.read_excel(ruta_excel5, index_col=0)


# Tabla Frecuencias
# Obtengo mi tabla de frecuencias tanto en valores como en porcentajes
tabla_frecuencias5 = df4['Total'].value_counts().sort_index()

# Creo un DataFrame a partir de la tabla de frecuencias
df4_frecuencias = pd.DataFrame({
    'Número de Temáticas': tabla_frecuencias5.index,
    'Frecuencia': tabla_frecuencias5.values,
    'Porcentaje Frecuencia': tabla_frecuencias5 / tabla_frecuencias5.sum() * 100
})

# Imprimir el DataFrame de frecuencias
print(df4_frecuencias)


# Visualización de gráfico de dispersión con burbujas
fig, ax = plt.subplots(figsize=(10, 12))  # Ajusta el tamaño de la figura
scatter = ax.scatter(
    df4_frecuencias['Número de Temáticas'],
    df4_frecuencias['Frecuencia'],
    label='Presencia',
    alpha=0.7,
    s=df4_frecuencias['Frecuencia'] * 30  # Ajusta el tamaño de las burbujas según la frecuencia
)

ax.set_xlabel('Número de Temáticas')
ax.set_ylabel('Frecuencia')
ax.set_title('Frecuencia de Presencia por Número de Temáticas')
ax.legend()

# Ajustar el tamaño de la burbuja en la leyenda
legend = ax.legend()
legend.legend_handles[0]._sizes = [30]

# Añadir etiquetas centradas para cada burbuja
for i, txt in enumerate(df4_frecuencias['Frecuencia']):
    ax.annotate(txt, (df4_frecuencias['Número de Temáticas'][i], df4_frecuencias['Frecuencia'][i]),
                ha='center', va='center')

# Ajustar espacio entre valores del eje y
selected_y_ticks = range(0, df4_frecuencias['Frecuencia'].max() + 31, 20)  # Incrementé en 31 para dar espacio adicional
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.set_yticks(selected_y_ticks)
ax.set_ylim(0, df4_frecuencias['Frecuencia'].max() + 31)  # Establecer límites del eje y con espacio adicional

plt.show()


# Crear el gráfico 3D con línea de conexión entre los puntos
trace = go.Scatter3d(
    x=df4_frecuencias['Número de Temáticas'],
    y=df4_frecuencias['Porcentaje Frecuencia'],
    z=df4_frecuencias['Frecuencia'],
    mode='lines+markers',
    line=dict(color='orange', width=2),  # Línea de conexión entre los puntos
    marker=dict(size=8, color='orange', symbol='circle'),  # Color de los puntos
    name='Presencia'
)

layout = go.Layout(
    scene=dict(
        xaxis=dict(title='Número de Temáticas'),
        yaxis=dict(title='Porcentaje Frecuencia'),
        zaxis=dict(title='Frecuencia'),
    ),
    title='Frecuencia de Presencia por Número de Temáticas',
)

fig = go.Figure(data=[trace], layout=layout)

# Guardar el gráfico como un archivo HTML
#pio.write_html(fig, file='grafico_3d.html')

# Mostrar el gráfico
fig.show()



# Agrupar por 'Gene_Popu' y 'Total' y contar la frecuencia
frecuencia_por_categoria = df4.groupby(['GenePopu', 'Total']).size().reset_index(name='Frecuencia')

# Mostrar la tabla de frecuencia por categoría y total
print(frecuencia_por_categoria)


# Utiliza el DataFrame frecuencia_por_categoria que has creado
# Asegúrate de que 'GenePopu' y 'Total' sean de tipo categórico para que seaborn los interprete correctamente
frecuencia_por_categoria['GenePopu'] = frecuencia_por_categoria['GenePopu'].astype('category')
frecuencia_por_categoria['Total'] = frecuencia_por_categoria['Total'].astype('category')

# Crea el gráfico de dispersión de líneas
plt.figure(figsize=(10, 6))
sns.lineplot(x='Total', y='Frecuencia', hue='GenePopu', data=frecuencia_por_categoria, marker='o')

# Ajusta la leyenda y el título del gráfico
plt.legend(title='GenePopu', title_fontsize='12')
plt.title('Gráfico de Dispersión de Líneas por Categoría', fontsize='16')
plt.xlabel('Total', fontsize='12')
plt.ylabel('Frecuencia', fontsize='12')

# Muestra el gráfico
plt.show()

# Si deseas ver los porcentajes, puedes hacer lo siguiente
# Agrupar por 'Gene_Popu' y 'Total' y calcular los porcentajes de presencia
porcentajes_presencia = df4.groupby(['GenePopu', 'Total']).size() / len(df4) * 100

# Mostrar la tabla de porcentajes de presencia por categoría y total
print(porcentajes_presencia)

# Asegúrate de que 'GenePopu' y 'Total' estén presentes en el índice
porcentajes_presencia = porcentajes_presencia.reset_index()
porcentajes_presencia.rename(columns={0: 'porcentaje_presencia'}, inplace=True)

# Organiza los datos para el gráfico de barras apiladas
categories = porcentajes_presencia['GenePopu'].unique()
tematicas = porcentajes_presencia['Total'].unique()

plt.figure(figsize=(12, 8))
bar_width = 0.2
colors = plt.cm.Paired(range(len(tematicas)))

bottom = np.zeros(len(categories))

for i, tematica in enumerate(tematicas):
    porcentajes_tem = porcentajes_presencia[porcentajes_presencia['Total'] == tematica]['porcentaje_presencia']
    # Ajuste para manejar la diferencia de longitud
    porcentajes_tem = porcentajes_tem.tolist() + [0] * (len(categories) - len(porcentajes_tem))
    plt.bar(np.arange(len(categories)) + bar_width / 2, porcentajes_tem, width=bar_width, bottom=bottom, label=f'{tematica} Temáticas', color=colors[i])
    bottom += porcentajes_tem

plt.xlabel('Categoría')
plt.ylabel('Porcentaje de Presencia')
plt.title('Porcentaje de Presencia por Categoría y Número de Temáticas')
plt.legend(title='Número de Temáticas', loc='upper right')
plt.xticks(np.arange(len(categories)) + bar_width / 2, categories)

plt.show()

# Ruta de mi matriz binaria de datos
ruta_excel2 = 'Matriz_binaria2.xlsx'

# Cargar el archivo Excel en un DataFrame
df2 = pd.read_excel(ruta_excel2)

# Crear el pairplot con una variable de filtro y personalizar las etiquetas del eje y
g = sns.pairplot(df2, diag_kind='kde', hue='Gene_Popu', height=2.5, y_vars=df2.columns[2:])

# Truncar todas las etiquetas del eje y a las primeras 5 letras
etiquetas_truncadas = [col[:5] for col in df2.columns[2:]]
for ax, etiqueta in zip(g.axes[:, 0], etiquetas_truncadas):
    ax.set_ylabel(etiqueta)

# Mover la leyenda a la derecha
g.add_legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

# Mostrar el plot
plt.show()


# Obtener el número de canciones por categoria
songs_per_category = df2['Gene_Popu'].value_counts()

# Convertir a DataFrame
table = pd.DataFrame({
    'Categoría': songs_per_category.index,
    'Número de Canciones': songs_per_category.values
})

# Mostrar la tabla
print(table)

# Agrupar por la categoría y contar la frecuencia de cada tema en cada categoría
grouped_df = df2.groupby('Gene_Popu').sum()

# Eliminar la columna 'Unnamed: 0' del DataFrame
grouped_df = grouped_df.drop('Unnamed: 0', axis=1)

# Calcular porcentajes de presencia por categoría en relación con el total de apariciones del tema
porcentajes_presencia = grouped_df.div(grouped_df.sum(), axis=1) * 100

# Mostrar la tabla de porcentajes de presencia por categoría
print(porcentajes_presencia)

# Configurar el gráfico radial
theta = np.linspace(0, 2*np.pi, len(porcentajes_presencia.columns), endpoint=False) 
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# Dibujo las líneas radiales
ax.set_theta_offset(np.pi/2)  # Ajusto la posición inicial de las líneas
ax.set_theta_direction(-1)  # Le doy el sentido de las agujas del reloj
ax.set_rlabel_position(0)  # Etiquetas en el centro del gráfico

# Trazo las líneas que conectan los puntos dados por theta y porcentajes_presencia para cada categoría
for i, categoria in enumerate(porcentajes_presencia.index):
    valores_categoria = porcentajes_presencia.loc[categoria].values
    ax.plot(theta, valores_categoria, label=categoria, marker='o')

# Añado las etiquetas de cada tema
ax.set_xticks(theta)
ax.set_xticklabels(porcentajes_presencia.columns)

# Configuro el gráfico
ax.set_title('Porcentaje de Presencia por Tema en Gráfico Radial')
ax.legend(loc='lower right', bbox_to_anchor=(1.1, 0.97))

plt.show()


# Ruta de mi matriz binaria de datos
ruta_excel3 = 'Matriz_binaria4.xlsx'

# Cargar el archivo Excel en un DataFrame
df3 = pd.read_excel(ruta_excel3)

# Obtener el número de canciones por año
songs_per_year = df3['Año'].value_counts().sort_index()

# Convertir a DataFrame
table = pd.DataFrame({
    'Año': songs_per_year.index,
    'Número de Canciones': songs_per_year.values
})

# Mostrar la tabla
print(table)


# Crear el pairplot con una variable de filtro y personalizar las etiquetas del eje y
g = sns.pairplot(df3, diag_kind='kde', hue='Año', height=2.5, y_vars=df3.columns[2:])

# Truncar todas las etiquetas del eje y a las primeras 5 letras
etiquetas_truncadas = [col[:5] for col in df3.columns[2:]]
for ax, etiqueta in zip(g.axes[:, 0], etiquetas_truncadas):
    ax.set_ylabel(etiqueta)

# Mover la leyenda a la derecha
g.add_legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

# Mostrar el plot
plt.show()


# Agrupar por la categoría y contar la frecuencia de cada tema en cada categoría
grouped_df2 = df3.groupby('Año').sum()

# Eliminar la columna 'Unnamed: 0' del DataFrame
grouped_df2 = grouped_df2.drop('Unnamed: 0', axis=1)

print(grouped_df2)

#Grafico de barras apilada de cada categoría, representando el % de aparicion de cada tematica en relacion a la totalidad de la presencia

stacked_df = grouped_df.div(grouped_df.sum(axis=1), axis=0) * 100

# Ajusta el tamaño de la figura
plt.figure(figsize=(12, 8))

ax = stacked_df.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Proporción de Categorías en cada Variable')
plt.ylabel('Proporción (%)')

# Mueve la leyenda a la derecha y fuera del gráfico
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')

# Ajusta el límite derecho de la figura para hacer espacio para la leyenda
plt.subplots_adjust(right=0.8)

plt.show()


# Mapa de Calor de Correlación Con Popularidad
correlation_matrix = grouped_df.corr()
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.xticks(rotation=45, ha='right') 
plt.subplots_adjust(left=0.25, right=0.95)
bottom, top = heatmap.get_ylim()
heatmap.set_ylim(bottom + 0.5, top - 0.5)
plt.subplots_adjust(bottom=0.2)
plt.title('Matriz de Correlación entre Variables por Género y Popularidad del Artista')
plt.show()


# Mapa de Calor de Correlación Con Fecha
correlation_matrix = grouped_df2.corr()
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.xticks(rotation=45, ha='right') 
plt.subplots_adjust(left=0.25, right=0.95)
bottom, top = heatmap.get_ylim()
heatmap.set_ylim(bottom + 0.5, top - 0.5)
plt.subplots_adjust(bottom=0.2)
plt.title('Matriz de Correlación entre Variables por Fecha de Publicación')
plt.show()


cabeceras = df.columns

# Crea una paleta de colores para asignar a cada violín
colores = sns.color_palette("husl", n_colors=len(cabeceras))

# Número de subgráficos por fila
subplots_per_row = 2
num_rows = -(-len(cabeceras) // subplots_per_row)  # Divide redondeando hacia arriba

# Crea subgráficos
fig, axes = plt.subplots(num_rows, subplots_per_row, figsize=(12, 8), sharex=True)

# Ajusta el diseño de la figura
fig.tight_layout(pad=3.0)

# Itera sobre las columnas y dibuja los violines en los subgráficos correspondientes
for i, columna in enumerate(cabeceras):
    row = i // subplots_per_row
    col = i % subplots_per_row
    ax = axes[row, col] if num_rows > 1 else axes[col]

    sns.violinplot(x='Gene_Popu', y=columna, data=df2, color=colores[i], ax=ax, inner="quartile")
    ax.set_title(f'Distribución {columna} por categoría')

# Muestra la figura
plt.show()

# Transponer el DataFrame para tener las categorías en el índice
grouped_df = grouped_df.transpose()

# Crear el gráfico de líneas
plt.figure(figsize=(12, 8))
sns.lineplot(data=grouped_df, markers=True)

# Mover la leyenda fuera del gráfico
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Configurar etiquetas y título
plt.xlabel('Tematicas')
plt.ylabel('Frecuencia')
plt.title('Evolución de Temáticas por Categorías')

# Mostrar el gráfico
plt.show()

# Contar la cantidad de canciones por año
songs_per_year = df3['Año'].value_counts().sort_index()

# Crear el gráfico de líneas con años en el eje x y cantidad de canciones en el eje y
plt.figure(figsize=(12, 8))
sns.lineplot(x=songs_per_year.index, y=songs_per_year.values, marker='o')

# Configurar etiquetas y título
plt.xlabel('Año')
plt.ylabel('Número de Canciones')
plt.title('Número de Canciones por Año')

# Usar valores únicos de la columna 'Año' sin decimales en el eje x
plt.xticks(sorted(df3['Año'].unique()))

# Rotar las etiquetas del eje x para mayor legibilidad
plt.xticks(rotation=45)

plt.show()



# Cargar datos desde el archivo Excel
df = pd.read_excel("Matriz_binaria3.xlsx")

# Convertir la columna 'Año' a enteros
df['Año'] = df['Año'].astype(int)

# Calcular el número total de temáticas posibles (en lugar de la suma de temáticas presentes)
df['total_temáticas'] = df.iloc[:, 3:].sum(axis=1)

# Calcular la frecuencia de cada combinación de 'GenePopu', 'Año' y número total de temáticas
df['frecuencia'] = df.groupby(['GenePopu', 'Año', 'total_temáticas']).transform('size')

# Crear el gráfico 3D con Plotly Express
fig = px.scatter_3d(df, x='GenePopu', y='Año', z='total_temáticas', 
                    color='total_temáticas', size='frecuencia', opacity=0.5,
                    labels={'GenePopu': 'Popularidad', 'Año': 'Año de Publicación', 'total_temáticas': 'Número Total de Temáticas'},
                    color_continuous_scale='viridis_r')

fig.update_layout(title='Relación entre Popularidad, Año de Publicación y Número Total de Temáticas en Canciones')

# Guardar en un archivo HTML con Plotly
#fig.write_html("grafico_3d_plotly.html")

# Mostrar el gráfico
fig.show()