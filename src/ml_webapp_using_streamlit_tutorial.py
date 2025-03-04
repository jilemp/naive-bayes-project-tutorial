import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():

    #pasos iniciales de disenyo
    st.title('Solucion del proyecto Pagina Web de ML con Streamlit - Jose Ibarra')
    st.header("esto es un encabezado")
    st.subheader('esto es un subencabezado')
    st.text('esto es un texto')

    nombre = "Jose"

    st.text(f"Mi nombre es {nombre} y estoy estudiante")
    st.success("Exito: Este es el mensaje de aprobado con exito")
    st.warning("Advertencia: No se visualizar el proyecto, corregir y enviar")
    st.info("Informacion: Tarea rechazada")

    ##iniciar con el modelo
    #datos dummies
    np.random.seed(42)
    X =  np.random.rand(100, 1)*10
    y = 3 * X + 8 + np.random.randn(100, 1)*2

    #seprar conjunto de datos entre entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #generar modelo de regresion lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    #prediccion
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    ##la interfaz
    st.title("Mi primer regresion lineal en web")
    st.write("este es un modelo de ejemplo para entregar el proyecto")

    #usar un SelectBox
    opcion = st.selectbox("Seleccione el tipo de visualizacion", ["Dispersion", "Linea de Regresion"])

    #checkbox para mostrar coeficientes
    if st.checkbox("Mostrar coeficientes de la Regresion Lineal"):
        st.write(f"coeficiente:{model.coef_[0][0]:.2f}")
        st.write(f"coeficiente interseccion:{model.intercept_[0]:.2f}")
        st.write(f"error medio cuadratico:{mse:.2f}")

    #slider
    data_range = st.slider("Seleccione el rango que quiere evaluar", 0, 100, (10,90))
    x_display = X_test[data_range[0]:data_range[1]]
    y_display = y_test[data_range[0]:data_range[1]]
    y_pred_display = y_pred[data_range[0]:data_range[1]]

    #visualizacion
    fig, ax = plt.subplots()
    if opcion == "Dispersion":
        ax.scatter(x_display, y_display)
        plt.title("Grafico de Dispersion")
    else:
        ax.plot(x_display, y_pred_display)
        plt.title("Grafico de Linea")
    st.pyplot(fig)

    opcion_2 = st.multiselect('Informacion Adicional', ['Informe General', 'Prediccion'])

    if 'Informe General' in opcion_2:
        st.write('Aca deberian estar los datos!')
        st.write(pd.DataFrame({'X Test': X_test.flatten(), 'y Test': y_test.flatten(), 'y Predict':y_pred.flatten()}))

    if 'Prediccion' in opcion_2:
        st.write('Aca tambien deberian haber datos!')
        for i in range(len(x_display)):
            st.write(f'x = {x_display[i][0]:.2f} prediccion y = {y_pred_display[i][0]:.2f}')

    #recibir informacion de usuario
    st.header('Ingrese su valor de X')
    x_ingresado = st.number_input('Numero a predecir')
    if st.button('Calcular'):
        prediccion_x = model.predict([[x_ingresado]])[0][0]
        st.write(f'La preddiccion para {x_ingresado} es {prediccion_x}')

if __name__ == '__main__':
    main()