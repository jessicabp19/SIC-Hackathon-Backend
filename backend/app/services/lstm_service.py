"""
Servicio LSTM: preparación de datos y entrenamiento del modelo.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from typing import Dict, Any, Optional

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from app.config import (
    TAMANO_VENTANA,
    PROPORCION_ENTRENAMIENTO,
    EPOCAS_NN,
    TAMANO_LOTE_NN,
    UNIDADES_LSTM,
    NOMBRE_ARCHIVO_MODELO
)


class LSTMService:
    """Servicio para el modelo LSTM de predicción de rendimientos."""

    def __init__(self):
        self._modelo: Optional[Sequential] = None
        self._cargar_modelo_existente()

    def _cargar_modelo_existente(self) -> None:
        """Intenta cargar un modelo previamente entrenado."""
        if os.path.exists(NOMBRE_ARCHIVO_MODELO):
            try:
                self._modelo = load_model(NOMBRE_ARCHIVO_MODELO)
                print(f"Modelo LSTM cargado: {NOMBRE_ARCHIVO_MODELO}")
            except Exception as e:
                print(f"Error al cargar modelo: {e}")
                self._modelo = None

    def preparar_datos(self, df_rendimientos: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepara los datos para entrenamiento/predicción del LSTM.

        Args:
            df_rendimientos: DataFrame con rendimientos logarítmicos

        Returns:
            Diccionario con datos preparados (X_train, Y_train, X_test, etc.)
        """
        scaler = MinMaxScaler()
        datos_scaled = scaler.fit_transform(df_rendimientos.values)

        X, Y = [], []
        for i in range(TAMANO_VENTANA, len(datos_scaled)):
            X.append(datos_scaled[i - TAMANO_VENTANA:i])
            Y.append(datos_scaled[i])

        X = np.array(X)
        Y = np.array(Y)
        div = int(PROPORCION_ENTRENAMIENTO * len(X))

        return {
            "X_train": X[:div],
            "Y_train": Y[:div],
            "X_test": X[div:],
            "Y_test": Y[div:],
            "scaler": scaler,
            "n_features": df_rendimientos.shape[1],
            "names": df_rendimientos.columns.tolist(),
            "df_rend": df_rendimientos,
            "index_test": len(df_rendimientos) - len(X) + div
        }

    def entrenar_modelo(self, datos: Dict[str, Any]) -> Sequential:
        """
        Entrena el modelo LSTM o usa uno existente si es compatible.

        Args:
            datos: Diccionario con datos preparados

        Returns:
            Modelo Keras entrenado
        """
        n_features = datos["n_features"]

        # Verificar si el modelo existente es compatible
        if self._modelo is not None:
            try:
                if self._modelo.input_shape[2] == n_features:
                    return self._modelo
            except:
                pass

        print("Iniciando entrenamiento del modelo LSTM...")

        model = Sequential([
            LSTM(UNIDADES_LSTM, input_shape=(TAMANO_VENTANA, n_features)),
            Dropout(0.2),
            Dense(n_features)
        ])
        model.compile(optimizer="adam", loss="mse")

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True
            )
        ]

        model.fit(
            datos["X_train"],
            datos["Y_train"],
            validation_data=(datos["X_test"], datos["Y_test"]),
            epochs=EPOCAS_NN,
            batch_size=TAMANO_LOTE_NN,
            verbose=0,
            callbacks=callbacks
        )

        # Guardar modelo entrenado
        try:
            model.save(NOMBRE_ARCHIVO_MODELO)
            print(f"Modelo guardado: {NOMBRE_ARCHIVO_MODELO}")
        except Exception as e:
            print(f"Error al guardar modelo: {e}")

        self._modelo = model
        return model

    def obtener_modelo(self) -> Optional[Sequential]:
        """Retorna el modelo actual (puede ser None)."""
        return self._modelo

    def calcular_metricas_validacion(
        self,
        datos: Dict[str, Any],
        pesos_optimos: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calcula métricas de validación del modelo.

        Args:
            datos: Diccionario con datos preparados
            pesos_optimos: Pesos óptimos del portafolio

        Returns:
            Diccionario con RMSE y ganancia vs buy&hold
        """
        if self._modelo is None:
            return {"rmse_modelo": 0, "rmse_baseline": 0, "ganancia_vs_buy_hold": 0}

        Y_test = datos["Y_test"]
        X_test = datos["X_test"]
        df_rend = datos["df_rend"]
        idx = datos["index_test"]

        # Predicciones
        pred_scaled = self._modelo.predict(X_test, verbose=0)
        rmse = sqrt(mean_squared_error(Y_test, pred_scaled))
        rmse_base = sqrt(mean_squared_error(Y_test, np.zeros_like(Y_test)))

        # Comparación con Buy & Hold
        rend_test = df_rend.iloc[idx:]
        w = np.array([pesos_optimos.get(c, 0) for c in rend_test.columns])

        rend_opt = np.expm1((rend_test @ w).sum())
        rend_bh = np.expm1((rend_test.mean(axis=1)).sum())

        ganancia = (rend_opt - rend_bh) / max(abs(rend_bh), 1e-6) * 100

        return {
            "rmse_modelo": rmse,
            "rmse_baseline": rmse_base,
            "ganancia_vs_buy_hold": ganancia
        }


# Instancia singleton del servicio
lstm_service = LSTMService()
