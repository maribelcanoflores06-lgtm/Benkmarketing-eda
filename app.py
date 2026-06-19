"""
app.py
Caso de Estudio N°1 - BankMarketing
Especialización en Python for Analytics - DMC Institute

Aplicación de Análisis Exploratorio de Datos (EDA) construida con Streamlit
para entender los factores que influyen en la aceptación de campañas de
marketing de una institución financiera (variable objetivo: y).

Autor: Maribel
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="EDA Bank Marketing",
    page_icon="🏦",
    layout="wide",
)

sns.set_theme(style="whitegrid")

AUTOR_NOMBRE = "Maribel"
AUTOR_CURSO = "Especialización en Python for Analytics - DMC Institute"
AUTOR_ANIO = "2026"

NUMERIC_COLS_DEFAULT = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL_COLS_DEFAULT = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome", "y",
]


# ----------------------------------------------------------------------------
# FUNCIÓN PERSONALIZADA: clasificación de variables (Ítem 2)
# ----------------------------------------------------------------------------
def clasificar_columnas(df: pd.DataFrame):
    """
    Función personalizada que clasifica las columnas de un DataFrame en
    numéricas y categóricas, en base a su tipo de dato (dtype).
    Retorna dos listas: (numericas, categoricas)
    """
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numericas, categoricas


# ----------------------------------------------------------------------------
# CLASE PRINCIPAL (Programación Orientada a Objetos)
# ----------------------------------------------------------------------------
class BankMarketingAnalyzer:
    """
    Clase que encapsula la lógica de análisis del dataset BankMarketing:
    - Clasificación de variables
    - Estadísticas descriptivas
    - Funciones de visualización (Matplotlib / Seaborn)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ---- Clasificación de variables ----
    def clasificar_variables(self):
        return clasificar_columnas(self.df)

    # ---- Estadísticas descriptivas ----
    def estadisticas_descriptivas(self, columnas=None):
        if columnas:
            return self.df[columnas].describe()
        return self.df.describe(include="all")

    def medidas_centrales(self, columna: str):
        serie = self.df[columna]
        resultado = {"media": None, "mediana": None, "moda": None}
        if pd.api.types.is_numeric_dtype(serie):
            resultado["media"] = serie.mean()
            resultado["mediana"] = serie.median()
        moda_serie = serie.mode()
        resultado["moda"] = moda_serie.iloc[0] if not moda_serie.empty else None
        return resultado

    # ---- Visualizaciones ----
    def histograma(self, columna: str, bins: int = 30, kde: bool = True):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df[columna], bins=bins, kde=kde, ax=ax, color="#2E86AB")
        ax.set_title(f"Distribución de {columna}")
        ax.set_xlabel(columna)
        ax.set_ylabel("Frecuencia")
        return fig

    def barras_categoricas(self, columna: str):
        fig, ax = plt.subplots(figsize=(6, 4))
        orden = self.df[columna].value_counts().index
        sns.countplot(data=self.df, x=columna, order=orden, ax=ax, color="#F26419")
        ax.set_title(f"Frecuencia de {columna}")
        ax.set_ylabel("Conteo")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        return fig

    def boxplot_numerico_vs_categorico(self, num_col: str, cat_col: str):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, hue=cat_col, ax=ax,
                    palette="Set2", legend=False)
        ax.set_title(f"{num_col} según {cat_col}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        return fig

    def barras_apiladas_categorico_vs_categorico(self, cat1: str, cat2: str):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        tabla_pct = pd.crosstab(self.df[cat1], self.df[cat2], normalize="index") * 100
        tabla_pct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_title(f"{cat1} vs {cat2} (% por fila)")
        ax.set_ylabel("Porcentaje (%)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(title=cat2, bbox_to_anchor=(1.02, 1), loc="upper left")
        return fig


# ----------------------------------------------------------------------------
# UTILIDADES DE CARGA
# ----------------------------------------------------------------------------
def cargar_csv(archivo_subido):
    """
    Lee el archivo subido detectando automáticamente si el separador
    es ',' o ';' (el dataset original de BankMarketing usa punto y coma).
    """
    contenido = archivo_subido.read()
    archivo_subido.seek(0)
    try:
        df = pd.read_csv(io.BytesIO(contenido), sep=None, engine="python")
        if df.shape[1] == 1:
            raise ValueError("Una sola columna detectada, probando separador alterno")
    except Exception:
        df = pd.read_csv(io.BytesIO(contenido), sep=";")
    return df


# ----------------------------------------------------------------------------
# SIDEBAR: NAVEGACIÓN PRINCIPAL
# ----------------------------------------------------------------------------
st.sidebar.title("🏦 Bank Marketing EDA")
modulo = st.sidebar.radio(
    "Navegación",
    ["🏠 Home", "📂 Carga de Datos", "📊 Análisis Exploratorio (EDA)", "✅ Conclusiones"],
)

if "df" not in st.session_state:
    st.session_state.df = None


# ----------------------------------------------------------------------------
# MÓDULO 1: HOME
# ----------------------------------------------------------------------------
if modulo == "🏠 Home":
    st.title("🏦 Análisis Exploratorio de Datos: Bank Marketing")
    st.markdown("### Caso de Estudio N°1 - Especialización Python for Analytics")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            **Objetivo del análisis**

            Esta aplicación analiza los datos de la última campaña de marketing de
            una institución financiera, cuya efectividad cayó de **12% a 8%** en
            los últimos 6 meses. El propósito es explorar, visualizar y entender
            qué variables (demográficas, económicas y de contacto) se relacionan
            con la decisión del cliente de aceptar o no la oferta (`y`),
            con un enfoque exploratorio orientado a la **toma de decisiones**,
            no a la construcción de modelos predictivos.

            **Sobre el dataset**

            `BankMarketing.csv` contiene **41,188 registros** y **21 variables**,
            incluyendo características del cliente (edad, trabajo, estado civil,
            educación), historial crediticio, detalles de contacto de la campaña
            (canal, mes, duración, número de contactos) e indicadores
            macroeconómicos (tasa de empleo, índice de precios, euríbor, etc.).
            """
        )
    with col2:
        st.info(
            f"""
            **Datos del autor**

            👤 {AUTOR_NOMBRE}

            🎓 {AUTOR_CURSO}

            📅 {AUTOR_ANIO}
            """
        )

    st.markdown("---")
    st.markdown("**Tecnologías utilizadas**")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.markdown("🐍 Python")
    t2.markdown("🐼 Pandas / NumPy")
    t3.markdown("📊 Matplotlib / Seaborn")
    t4.markdown("⚡ Streamlit")
    t5.markdown("🧱 POO")

    st.markdown("---")
    st.caption(
        "Usa el menú lateral para navegar: primero carga el dataset en "
        "'Carga de Datos' y luego explora el EDA."
    )


# ----------------------------------------------------------------------------
# MÓDULO 2: CARGA DEL DATASET
# ----------------------------------------------------------------------------
elif modulo == "📂 Carga de Datos":
    st.title("📂 Carga del Dataset")
    st.write(
        "Sube el archivo **BankMarketing.csv** para habilitar el módulo de "
        "Análisis Exploratorio de Datos (EDA)."
    )

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df_cargado = cargar_csv(archivo)
            st.session_state.df = df_cargado
            st.success("✅ Archivo cargado correctamente.")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Filas", f"{df_cargado.shape[0]:,}")
            with c2:
                st.metric("Columnas", f"{df_cargado.shape[1]:,}")

            st.markdown("**Vista previa (primeras 5 filas)**")
            st.dataframe(df_cargado.head(), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Ocurrió un error al leer el archivo: {e}")
            st.session_state.df = None
    else:
        if st.session_state.df is not None:
            st.info("Ya hay un dataset cargado en esta sesión. Puedes ir al módulo de EDA.")
        else:
            st.warning("Aún no se ha cargado ningún archivo.")


# ----------------------------------------------------------------------------
# MÓDULO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ----------------------------------------------------------------------------
elif modulo == "📊 Análisis Exploratorio (EDA)":
    st.title("📊 Análisis Exploratorio de Datos (EDA)")

    if st.session_state.df is None:
        st.warning(
            "⚠️ No se ha cargado ningún dataset. Ve al módulo "
            "'📂 Carga de Datos' antes de continuar."
        )
        st.stop()

    df_original = st.session_state.df.copy()

    # ----- Filtro global opcional (sidebar) -----
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros del EDA")
    aplicar_filtro_edad = st.sidebar.checkbox("Filtrar por rango de edad", value=False)

    if aplicar_filtro_edad and "age" in df_original.columns:
        edad_min, edad_max = int(df_original["age"].min()), int(df_original["age"].max())
        rango_edad = st.sidebar.slider(
            "Rango de edad", min_value=edad_min, max_value=edad_max,
            value=(edad_min, edad_max),
        )
        df = df_original[
            (df_original["age"] >= rango_edad[0]) & (df_original["age"] <= rango_edad[1])
        ].copy()
        st.caption(f"Filtro de edad aplicado: {rango_edad[0]} - {rango_edad[1]} años "
                   f"({df.shape[0]:,} registros de {df_original.shape[0]:,}).")
    else:
        df = df_original

    analyzer = BankMarketingAnalyzer(df)

    tabs = st.tabs([
        "1️⃣ Info general", "2️⃣ Clasificación", "3️⃣ Descriptivas", "4️⃣ Faltantes",
        "5️⃣ Distribución", "6️⃣ Categóricas", "7️⃣ Numérico vs y", "8️⃣ Categórico vs y",
        "9️⃣ Análisis dinámico", "🔟 Hallazgos clave",
    ])

    # ---------------- ÍTEM 1: Información general ----------------
    with tabs[0]:
        st.subheader("Información general del dataset")
        col1, col2 = st.columns(2)

        with col1:
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

        with col2:
            st.markdown("**Tipos de datos por columna**")
            st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)

        st.markdown("**Conteo de valores nulos por columna**")
        nulos = df.isnull().sum()
        st.dataframe(nulos[nulos >= 0].rename("nulos"), use_container_width=True)

        total_nulos = int(df.isnull().sum().sum())
        st.write(
            f"En este dataset hay **{total_nulos}** valores nulos en total "
            f"({df.shape[0]:,} filas, {df.shape[1]} columnas)."
        )

    # ---------------- ÍTEM 2: Clasificación de variables ----------------
    with tabs[1]:
        st.subheader("Clasificación de variables")
        st.caption("Se utiliza la función personalizada `clasificar_columnas(df)`.")

        numericas, categoricas = analyzer.clasificar_variables()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Variables numéricas", len(numericas))
            st.write(numericas)
        with c2:
            st.metric("Variables categóricas", len(categoricas))
            st.write(categoricas)

    # ---------------- ÍTEM 3: Estadísticas descriptivas ----------------
    with tabs[2]:
        st.subheader("Estadísticas descriptivas")
        numericas, _ = analyzer.clasificar_variables()

        st.dataframe(analyzer.estadisticas_descriptivas(numericas), use_container_width=True)

        col_interpretar = st.selectbox(
            "Selecciona una variable numérica para interpretar:", numericas, index=0,
        )
        medidas = analyzer.medidas_centrales(col_interpretar)
        media = medidas["media"]
        mediana = medidas["mediana"]
        desviacion = df[col_interpretar].std()

        st.info(
            f"La variable **{col_interpretar}** tiene una media de "
            f"**{media:,.2f}** y una mediana de **{mediana:,.2f}**. "
            f"La desviación estándar es de **{desviacion:,.2f}**, lo cual "
            f"{'sugiere alta dispersión respecto a la media' if desviacion > abs(media)*0.5 else 'sugiere una dispersión moderada'} "
            f"en los valores de esta variable."
        )

    # ---------------- ÍTEM 4: Análisis de valores faltantes ----------------
    with tabs[3]:
        st.subheader("Análisis de valores faltantes")

        nulos = df.isnull().sum()
        nulos = nulos[nulos > 0]

        if nulos.empty:
            st.success("✅ El dataset no presenta valores nulos (NaN) explícitos.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            nulos.plot(kind="bar", ax=ax, color="#C73E1D")
            ax.set_ylabel("Cantidad de nulos")
            st.pyplot(fig)

        st.markdown("**Valores 'unknown' en variables categóricas (faltantes encubiertos)**")
        _, categoricas = analyzer.clasificar_variables()
        conteo_unknown = {}
        for c in categoricas:
            if "unknown" in df[c].unique():
                conteo_unknown[c] = int((df[c] == "unknown").sum())

        if conteo_unknown:
            serie_unknown = pd.Series(conteo_unknown).sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            serie_unknown.plot(kind="bar", ax=ax2, color="#5C415D")
            ax2.set_ylabel("Cantidad de 'unknown'")
            st.pyplot(fig2)
            st.warning(
                "Aunque no existen valores NaN explícitos, varias variables "
                "categóricas usan la etiqueta **'unknown'** como faltante "
                "encubierto (ej. `job`, `education`, `default`). Además, la "
                "variable `pdays` usa el valor **999** para indicar que el "
                "cliente nunca fue contactado antes, lo cual debe tratarse "
                "como una categoría especial y no como un valor numérico "
                "más en los análisis de tendencia central."
            )
        else:
            st.write("No se detectaron valores 'unknown' en las columnas categóricas.")

    # ---------------- ÍTEM 5: Distribución de variables numéricas ----------------
    with tabs[4]:
        st.subheader("Distribución de variables numéricas")
        numericas, _ = analyzer.clasificar_variables()

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            var_hist = st.selectbox("Variable numérica:", numericas, key="hist_var")
        with col2:
            bins = st.slider("N° de bins:", min_value=5, max_value=100, value=30, key="hist_bins")
        with col3:
            mostrar_kde = st.checkbox("Mostrar KDE", value=True, key="hist_kde")

        fig = analyzer.histograma(var_hist, bins=bins, kde=mostrar_kde)
        st.pyplot(fig)

        asimetria = df[var_hist].skew()
        st.caption(
            f"Coeficiente de asimetría de **{var_hist}**: {asimetria:.2f} "
            f"({'distribución sesgada a la derecha (cola larga de valores altos)' if asimetria > 1 else 'distribución sesgada a la izquierda' if asimetria < -1 else 'distribución aproximadamente simétrica'})."
        )

    # ---------------- ÍTEM 6: Análisis de variables categóricas ----------------
    with tabs[5]:
        st.subheader("Análisis de variables categóricas")
        _, categoricas = analyzer.clasificar_variables()

        var_cat = st.selectbox("Variable categórica:", categoricas, key="cat_var")

        col1, col2 = st.columns(2)
        with col1:
            fig = analyzer.barras_categoricas(var_cat)
            st.pyplot(fig)
        with col2:
            conteo = df[var_cat].value_counts()
            proporcion = (df[var_cat].value_counts(normalize=True) * 100).round(2)
            tabla = pd.DataFrame({"conteo": conteo, "proporción (%)": proporcion})
            st.dataframe(tabla, use_container_width=True)

    # ---------------- ÍTEM 7: Bivariado numérico vs categórico (y) ----------------
    with tabs[6]:
        st.subheader("Análisis bivariado: variable numérica vs y")
        numericas, _ = analyzer.clasificar_variables()
        default_idx = numericas.index("duration") if "duration" in numericas else 0

        var_num_biv = st.selectbox(
            "Variable numérica:", numericas, index=default_idx, key="biv_num",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = analyzer.boxplot_numerico_vs_categorico(var_num_biv, "y")
            st.pyplot(fig)
        with col2:
            resumen = df.groupby("y")[var_num_biv].agg(["mean", "median", "std"]).round(2)
            st.markdown("**Resumen por grupo**")
            st.dataframe(resumen, use_container_width=True)

    # ---------------- ÍTEM 8: Bivariado categórico vs categórico (y) ----------------
    with tabs[7]:
        st.subheader("Análisis bivariado: variable categórica vs y")
        _, categoricas = analyzer.clasificar_variables()
        categoricas_sin_y = [c for c in categoricas if c != "y"]
        default_idx = categoricas_sin_y.index("education") if "education" in categoricas_sin_y else 0

        var_cat_biv = st.selectbox(
            "Variable categórica:", categoricas_sin_y, index=default_idx, key="biv_cat",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = analyzer.barras_apiladas_categorico_vs_categorico(var_cat_biv, "y")
            st.pyplot(fig)
        with col2:
            tabla_cruzada = pd.crosstab(df[var_cat_biv], df["y"])
            st.markdown("**Tabla de contingencia**")
            st.dataframe(tabla_cruzada, use_container_width=True)

    # ---------------- ÍTEM 9: Análisis dinámico ----------------
    with tabs[8]:
        st.subheader("Análisis basado en parámetros seleccionados")
        numericas, categoricas = analyzer.clasificar_variables()
        categoricas_sin_y = [c for c in categoricas if c != "y"]

        col1, col2 = st.columns(2)
        with col1:
            cols_num_sel = st.multiselect(
                "Variables numéricas a comparar:", numericas,
                default=numericas[:2] if len(numericas) >= 2 else numericas,
            )
        with col2:
            col_cat_sel = st.selectbox("Agrupar por variable categórica:", categoricas, index=0)

        if cols_num_sel:
            tabla_dinamica = df.groupby(col_cat_sel)[cols_num_sel].mean().round(2)
            st.markdown(f"**Promedio de variables seleccionadas, agrupado por `{col_cat_sel}`**")
            st.dataframe(tabla_dinamica, use_container_width=True)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            tabla_dinamica.plot(kind="bar", ax=ax)
            ax.set_ylabel("Promedio")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)
        else:
            st.info("Selecciona al menos una variable numérica para visualizar el análisis.")

    # ---------------- ÍTEM 10: Hallazgos clave ----------------
    with tabs[9]:
        st.subheader("Hallazgos clave")

        tasa_global = (df["y"] == "yes").mean() * 100
        tasa_por_contacto = df.groupby("contact")["y"].apply(lambda s: (s == "yes").mean() * 100)
        tasa_por_poutcome = df.groupby("poutcome")["y"].apply(lambda s: (s == "yes").mean() * 100)
        duracion_media_si = df.loc[df["y"] == "yes", "duration"].mean()
        duracion_media_no = df.loc[df["y"] == "no", "duration"].mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Tasa de aceptación global", f"{tasa_global:.2f}%")
        m2.metric("Duración media (aceptó)", f"{duracion_media_si:.0f} seg")
        m3.metric("Duración media (no aceptó)", f"{duracion_media_no:.0f} seg")

        fig, ax = plt.subplots(figsize=(6, 4))
        tasa_por_poutcome.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#2E86AB")
        ax.set_ylabel("Tasa de aceptación (%)")
        ax.set_title("Tasa de aceptación según resultado de campaña anterior (poutcome)")
        st.pyplot(fig)

        mejor_canal = tasa_por_contacto.idxmax()
        mejor_poutcome = tasa_por_poutcome.idxmax()

        st.markdown(
            f"""
            **Insights principales derivados del EDA:**

            - La tasa de aceptación global de la campaña es de **{tasa_global:.2f}%**.
            - El canal de contacto con mejor tasa de aceptación es **{mejor_canal}**.
            - Los clientes cuya campaña anterior tuvo resultado **'{mejor_poutcome}'** son los que más aceptan la oferta actual.
            - Las llamadas que terminan en aceptación ('yes') duran en promedio **{duracion_media_si:.0f} segundos**, frente a **{duracion_media_no:.0f} segundos** en las que no aceptan, lo que sugiere que conversaciones más largas se asocian a mayor interés del cliente.
            """
        )


# ----------------------------------------------------------------------------
# MÓDULO 4: CONCLUSIONES FINALES
# ----------------------------------------------------------------------------
elif modulo == "✅ Conclusiones":
    st.title("✅ Conclusiones Finales")

    if st.session_state.df is None:
        st.warning(
            "⚠️ Carga el dataset en el módulo '📂 Carga de Datos' para "
            "contextualizar mejor estas conclusiones con tus propios datos."
        )

    st.markdown(
        """
        A partir del Análisis Exploratorio de Datos realizado sobre el dataset
        **BankMarketing.csv**, se identifican las siguientes conclusiones,
        orientadas a la toma de decisiones comerciales y no a la construcción
        de modelos predictivos:

        1. **La duración de la llamada está fuertemente asociada a la aceptación.**
           Los clientes que aceptan la oferta tienden a sostener conversaciones
           considerablemente más largas que quienes la rechazan, lo que sugiere
           que el guion comercial y la calidad de la conversación importan más
           que la cantidad de contactos.

        2. **El canal de contacto influye en la efectividad.**
           Existen diferencias claras en la tasa de aceptación según el canal
           utilizado (celular vs. teléfono fijo), por lo que priorizar el canal
           más efectivo podría mejorar la eficiencia de futuras campañas.

        3. **El historial de campañas previas es un predictor de comportamiento.**
           Los clientes cuyo contacto anterior (`poutcome`) fue exitoso muestran
           una tasa de aceptación notablemente más alta en la campaña actual,
           lo cual respalda priorizar a clientes con relación previa positiva.

        4. **El exceso de contactos no garantiza mejores resultados.**
           Un número elevado de contactos (`campaign`) no se traduce
           necesariamente en mayor aceptación; en muchos casos, insistir más
           allá de cierto punto puede generar el efecto contrario.

        5. **El perfil sociodemográfico (edad, trabajo, educación) segmenta el
           comportamiento del cliente.**
           Ciertos grupos ocupacionales y de edad muestran tasas de aceptación
           más altas, lo que permite diseñar campañas más segmentadas en lugar
           de un enfoque homogéneo para toda la base de clientes.

        Estas conclusiones permiten orientar decisiones prácticas: priorizar
        canales y segmentos con mayor probabilidad de éxito, capacitar a los
        ejecutivos comerciales en conversaciones de mayor calidad, y evitar el
        desgaste de reintentos excesivos sobre el mismo cliente.
        """
    )
