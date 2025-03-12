import arff
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# PASO 1 - ADQUISICIÓN DE DATOS
def load_data(filename):
    """Carga el dataset desde un archivo ARFF convertido a CSV"""
    """"""

    # Cargar el archivo ARFF
    with open("phpMawTba.arff", "r") as file:
        data = arff.load(file)

    # Convertirlo a un DataFrame de Pandas
    df = pd.DataFrame(data["data"], columns=[attr[0] for attr in data["attributes"]])

    # Guardarlo como CSV
    df.to_csv(filename, index=False)
    df = pd.read_csv(filename)

    if "class" not in df.columns:
        raise ValueError("ERROR: 'class' no está en el dataset original.")
    return df


# PASO 2 - LIMPIEZA Y PREPARACIÓN
def clean_data(df):
    """Limpia los datos eliminando valores nulos y estandarizando texto"""
    df = df.dropna().copy()  # Asegurar copia independiente
    df.loc[:, "occupation"] = df["occupation"].str.lower()
    df.loc[:, "education"] = df["education"].str.lower()
    
    # Guardamos una copia de class para generar los gráficos después
    df["class_original"] = df["class"]
    
    print("Datos limpios correctamente.")
    return df

# PASO 3 - ALMACENAMIENTO Y GESTIÓN
def save_cleaned_data(df, filename="adult_cleaned.csv"):
    """Guarda el dataset limpio en un nuevo archivo CSV"""
    if "class" not in df.columns:
        raise ValueError("ERROR: No se puede guardar porque 'class' no está en el DataFrame.")

    df.to_csv(filename, index=False)
    print(f"Datos limpios guardados en {filename}")

# PASO 4 -  ANÁLISIS Y VISUALIZACIÓN
def plot_data(df):
    """Realiza visualizaciones de los datos usando la versión original de class"""
    if "class_original" not in df.columns:
        raise ValueError("ERROR: No se puede visualizar porque 'class_original' no está en el DataFrame.")
    
    plt.figure(figsize=(12, 6))
    sns.countplot(x="education", hue="class_original", data=df, order=df["education"].value_counts().index)
    plt.xticks(rotation=90)
    plt.title("Nivel Educativo vs Ingresos")
    plt.show()
    
    print("Gráfica generada correctamente.")

# PASO 5 - MODELADO PREDICTIVO
def train_model(df):
    """Entrena un modelo de clasificación verificando que y sea categórica"""

    # Convertir valores de 'class' a string
    df["class"] = df["class"].astype(str).str.strip()

    # Verificar valores únicos
    print("Valores únicos en 'class':", df["class"].unique())

    # Convertir etiquetas categóricas a números
    le = LabelEncoder()
    df["class"] = le.fit_transform(df["class"])  

    # Variables para el modelo
    X = df[["education-num", "age", "hours-per-week"]]
    y = df["class"]

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluar modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy:.2f}")

    return model

# EJECUCIÓN DEL CICLO DE VIDA DEL DATO
if __name__ == "__main__":
    df = load_data("adult.csv")  # Cargar datos
    df = clean_data(df)  # Limpiar datos
    save_cleaned_data(df)  # Guardar datos limpios
    plot_data(df)  # Visualizar datos
    train_model(df)  # Entrenar modelo

