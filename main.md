### **Integración de Recomendaciones en Tiempo Real ("X coche vs Y coche")**

**Objetivo Ampliado** : Añadir un módulo de simulación estratégica que, dado un contexto específico (ej: coche A está 1.5s detrás de coche B en la vuelta 20), recomiende acciones óptimas (neumáticos, momento de parada, undercut/overcut) basado en condiciones actuales y predicciones.

---

### **Relación con Cada Asignatura y Herramientas**

| **Asignatura**                      | **Componente Ampliado**                                                                                                                                                                         | **Herramientas/Enfoque**                                                                                                                            | **Datos Necesarios**                                                                                 |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Visión por Computador**           | -**Detección de posición relativa entre coches** (gap en segundos).``<br />- **Estado de los neumáticos** (degradación visual en cámaras cercanas).                                             | - YOLOv8 + DeepSORT (seguimiento de coches).`- OpenCV para estimar distancia entre bboxes.`- Segmentación de neumáticos (si hay imágenes close-up). | Vídeos de carreras (YouTube) + cámaras onboard (ej:[F1TV](https://f1tv.formula1.com/) si accesible). |
| **Aprendizaje Automático Avanzado** | -**Modelo de predicción de tiempos por vuelta** bajo diferentes neumáticos/clima.``<br />- **Clasificación de undercut/overcut** (¿ganará posición si para antes/después?).                     | - XGBoost/LightGBM para tabular data (variables: gap, tipo neumático, temperatura pista).``- LSTM para predecir evolución de gaps.                  | Datos históricos de paradas y gaps desde FastF1.                                                     |
| **Sistemas Inteligentes**           | -**Simulación de escenarios** (¿qué pasa si el coche para ahora vs. en 3 vueltas?).``<br />- **Optimización multiobjetivo** (tiempo total vs. riesgo vs. tráfico).                              | - Algoritmos genéticos (DEAP library).``- MDPs (Markov Decision Processes) con RLlib.                                                               | Resultados de modelos predictivos + reglas de la FIA (ej: duración mínima de neumáticos).            |
| **Procesamiento de Lenguaje**       | -**Extracción de contexto estratégico** de radios (ej: "El coche B está con neumáticos blandos usados").``<br />- **Generación de explicaciones** en lenguaje natural para las recomendaciones. | - Fine-tuning de BERT para reconocer órdenes estratégicas.``- Plantillas de texto con variables (usando F-strings o Jinja2).                        | Transcripciones de radios (reales o sintéticas con GPT-4).                                           |

---

### **Fuentes de Datos Adicionales**

1. **Gaps entre coches** :

- Calculados desde datos de posición de FastF1 (`session.pos_data`) o estimados por visión por computador (píxeles entre coches en vídeo → conversión a metros usando referencia de ancho de pista).

1. **Efectividad histórica de undercuts** :

- FastF1 permite extraer laps antes/después de paradas para calcular ganancias/pérdidas de posición.

1. **Temperatura de pista y degradación de neumáticos** :

- Columnas `TrackStatus` y `TyreLife` en datos de FastF1.

---

### **Ejemplo de Flujo de Trabajo**

1. **Entrada** : Vídeo en vivo (o grabado) + datos en tiempo real de FastF1.
2. **Procesamiento** :

- **Visión** : Detectar que el coche A (posición 5) está a 1.2s del coche B (posición 4).
- **ML** : Predecir que con neumáticos duros nuevos, el coche A hará vueltas 0.8s más rápidas que B en las próximas 5 vueltas.
- **Sistemas Inteligentes** : Simular que parar en la vuelta 22 permite un undercut exitoso (ganar 2 posiciones).
- **NLP** : Generar el mensaje: _"Recomendado: PARAR EN VUELTA 22. Neumáticos duros. Undercut posible al coche B (riesgo bajo: 75% éxito)."_

---

### **Herramientas Específicas para la Ampliación**

- **Simulación de Estrategias** : `FastF1` ya incluye funciones para calcular diferencias de tiempo (`calc_time_diff`).
- **Visualización de Estrategias** : Usar `Plotly` para mostrar gráficos de proyección de gaps y paradas.
- **Optimización** : `Optuna` para ajustar hiperparámetros de los modelos de decisión.

---

### **Riesgos y Soluciones**

| **Riesgo**                               | **Mitigación**                                                                                  |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Datos en tiempo real limitados**       | Usar modo histórico con datos de FastF1 (ej: GP de España 2023) para desarrollar y testear.     |
| **Cálculo impreciso de gaps**            | Combinar datos de posición de FastF1 (precisos) con visión por computador (solo como respaldo). |
| **Overfitting en modelos de predicción** | Validación cruzada con datos de múltiples circuitos (Monza, Mónaco, etc.).                      |

### **Plan de Desarrollo - 3 Meses**

**Objetivo** : Construir un MVP funcional del sistema de estrategia para F1, integrando visión por computador, modelos predictivos, simulación de decisiones y NLP.

---

### **Mes 1: Configuración y Componentes Básicos**

| **Semana** | **Tareas**                                                                                                                                                                   | **Entregables**                                         | **Asignaturas Relacionadas**                  |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------- |
| **1**      | - Configurar entorno Python (venv o conda).`- Explorar FastF1: Extraer datos históricos (ej: GP España 2023).`- Descargar vídeos de YouTube (Creative Commons) para pruebas. | Entorno listo + dataset inicial (CSV/Parquet).          | Aprendizaje Automático, Sistemas Inteligentes |
| **2**      | - Implementar detección de coches/banderas con YOLOv8 en un vídeo estático.``- Calcular gaps entre coches usando posición en pista (FastF1).                                 | Script de detección + métricas de gaps.                 | Visión por Computador                         |
| **3**      | - Preprocesar datos de telemetría (neumáticos, tiempos por vuelta).``- Entrenar modelo básico de regresión (ej: predecir tiempo de vuelta).                                  | Modelo XGBoost inicial + informe de rendimiento (RMSE). | Aprendizaje Automático                        |
| **4**      | - Simular un sistema de decisiones con reglas "if-else" (ej: parar si neumáticos > 30% degradación).``- Crear interfaz básica en Streamlit para visualizar datos.            | Sistema de reglas + dashboard simple (Streamlit).       | Sistemas Inteligentes                         |

---

### **Mes 2: Desarrollo de Módulos Clave**

| **Semana** | **Tareas**                                                                                                                                                 | **Entregables**                                    | **Asignaturas Relacionadas** |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------- |
| **5**      | - Mejorar modelo predictivo: Incluir variables climáticas y tipo de neumático.``- Implementar LSTM para predecir degradación de neumáticos.                | Modelo avanzado + gráficos de predicción.          | Aprendizaje Automático       |
| **6**      | - Integrar visión por computador con FastF1: Sincronizar timestamp de vídeo y datos de telemetría.``- Calcular gaps en tiempo real usando YOLO + DeepSORT. | Script de sincronización + demo en vídeo.          | Visión por Computador        |
| **7**      | - Desarrollar simulador de estrategias con algoritmos genéticos (DEAP): Minimizar tiempo total de carrera.``- Definir restricciones (ej: mínimo 1 parada). | Simulador funcional + resultados de optimización.  | Sistemas Inteligentes        |
| **8**      | - Procesar radios de equipo-piloto: Transcribir audios con Whisper.``- Extraer órdenes clave ("Box now", "Stay out").                                      | Dataset de transcripciones + código de extracción. | Procesamiento de Lenguaje    |
| **9**      | - Integrar todos los módulos en un flujo único (ej: input vídeo → output recomendación).``- Crear API REST con FastAPI para conectar componentes.          | API funcional + ejemplo de request/respuesta.      | Todas                        |

---

### **Mes 3: Integración y Refinamiento**

| **Semana** | **Tareas**                                                                                                                                                                         | **Entregables**                                            | **Asignaturas Relacionadas**   |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------ |
| **10**     | - Entrenar modelo de clasificación para undercut/overcut (usar datos históricos de paradas).``- Generar alertas visuales en el vídeo (ej: "Undercut recomendado").                 | Modelo de clasificación + overlay en vídeo.                | Aprendizaje Automático, Visión |
| **11**     | - Mejorar el sistema de decisiones con RL (ej: Stable Baselines3).``- Añadir lógica de priorización de alertas (ej: riesgo alto/medio/bajo).                                       | Agente de RL entrenado + sistema de priorización.          | Sistemas Inteligentes          |
| **12**     | - Desarrollar dashboard interactivo en Streamlit: Mostrar predicciones, vídeo y recomendaciones.``- Añadir explicaciones en lenguaje natural (ej: "Parar en vuelta 22 porque..."). | Dashboard completo + NLP explicativo.                      | Procesamiento de Lenguaje      |
| **13**     | - Testeo integral: Validar con datos de 2-3 carreras distintas.``- Optimizar rendimiento (reducir latencia, mejorar precisión).                                                    | Informe de testeo + métricas finales (ej: precisión >85%). | Todas                          |

---

### **Entregables Finales**

- **Código** : Repositorio GitHub con módulos documentados.
- **Demo** : Dashboard Streamlit desplegado localmente o en Hugging Face Spaces.
- **Informe Técnico** : Explicación de arquitectura, resultados y lecciones aprendidas.

### **Herramientas Clave por Fase**

- **Control de Versiones** : GitHub (ramas por módulo: `vision`, `ml`, `nlp`).
- **MLOps** : MLflow para registrar modelos y parámetros.
- **Visualización** : Plotly para gráficos interactivos, OpenCV para overlays en vídeo.

### **Riesgos y Buffer**

- **Semana 14 (Buffer)** : Reservada para debugging, ajustes finales o incorporar feedback.
- **Mitigación** : Si falta datos de radios, usar síntesis con GPT-4 (ej: generar 100 diálogos estratégicos).

### **Plan de Desarrollo Detallado (Semana a Semana) con Modularidad y Opciones de Modelado**

Cada semana se centra en un **módulo independiente** vinculado a una asignatura, permitiendo desarrollo incremental y pruebas tempranas. Donde sea posible, se proponen **alternativas de implementación** (modelos clásicos vs. redes neuronales).

---

### **Mes 1: Configuración y Componentes Básicos**

#### **Semana 1: Configuración Inicial y Extracción de Datos**

- [x] Marcar como hecho

- **Tareas** :

1. **Entorno** :
   - Crear entorno virtual (`conda create -n f1-strategy python=3.10`).
   - OpenF1 API también será utilizada para extraer mensajes de radio y otras cosas que FastF1 no posee.
   - Instalar librerías base: `fastf1`, `pandas`, `numpy`.
   - Implementar `extract_openf1_intervals(race, session)` → crear esquema de almacenamiento unificado → desarrollar funciones de conversión entre formatos
2. **Datos FastF1** :
   - Extraer datos del GP España 2023: Tiempos por vuelta, paradas, clima.
   - Guardar en formato Parquet para eficiencia.
3. **Vídeos** :
   - Descargar 2-3 resúmenes de carreras (YouTube Creative Commons) usando `pytube`.

- **Entregables** :
- Script `data_extraction.py` + dataset en `data/raw`.
- Documentación de estructura de datos.
- **Asignatura** : _Aprendizaje Automático / Sistemas Inteligentes_ .

---

#### **Semana 2: Detección de Objetos con Visión por Computador**

- [x] Marcar como hecho

- **Tareas** :

- [x] Descargado dataset

1. **Opción 1 (YOLOv8)** :

   - [x] Entrenar YOLOv8-medium en dataset COCO para detectar coches (transfer learning). Finalmente, entrenado yolo medieum desde cero y buenos resultados.
   - [x] Probar en fotogramas de vídeo estático.
   - [x] Probar en vídeo dinámico.
   - [x] Reducir recall y reentrenar.

2. **Cálculo de Gaps** :

   - [x] Usar `OpenCV` para estimar distancia entre bboxes (píxeles → metros con referencia de ancho de pista).

- **Entregables** :
- Script `object_detection.py` + ejemplos de detección en `outputs/week2`.
- Métricas de precisión (mAP si es YOLO).
- **Asignatura** : _Visión por Computador_ .

---

#### **Semana 3: Modelo Predictivo de Tiempos por Vuelta**

- [ ] **Tareas** :

1. **Opción 1 (XGBoost/LightGBM)** :
   - Entrenar modelo para predecir `LapTime` usando variables: `TyreCompound`, `TrackTemp`, `AirTemp`.
2. **Opción 2 (Red Neuronal)** :
   - Implementar MLP en PyTorch con capas densas (64-32-16-1) y activación ReLU.
   - Comparar resultados con XGBoost.
3. **Feature Engineering** :
   - Crear variables como `TyreAge` (vueltas usadas) y `PositionChange`.

- **Entregables** :
- Script `lap_time_prediction.py` + modelo guardado en `models/week3`.
- Gráfico de dispersión predicciones vs. reales.
- **Asignatura** : _Aprendizaje Automático Avanzado_ .

---

#### **Semana 4: Sistema de Decisiones Basado en Reglas**

- **Tareas**:

  1. **Agente Lógico**:
     - Implementar un motor de reglas avanzado con `Pyke` o `Experta` para decisiones estratégicas.
     - Definir reglas basadas en lógica proposicional (ej: `IF TyreDeg > 30% AND Lap > 20 THEN PitStop`).
     - Explorar método API openfq https://openf1.org/#intervals para tener en cuenta intervalo con el coche de delante y el lider.
     - Extraer intervalos de OpenF1 → construir reglas basadas en gaps (< 1.5s = zona undercut) → implementar visualización de umbrales de gap en dashboard
  2. **Simulación Simple**:
     - Validar reglas con datos históricos (ej: GP España 2023).
  3. **Interfaz Streamlit**:
     - Añadir panel de control para activar/desactivar reglas.

- **Entregables**:

  - Script `rule_based_system.py` + dashboard en `src/dashboard`.
  - Documentación de reglas lógicas en `docs/rules.md`.

- **Asignatura**: _Sistemas Inteligentes (Unidad IV - 4.1)_.

---

### **Mes 2: Desarrollo de Módulos Clave**

#### **Semana 5: Modelo de Degradación de Neumáticos**

- **Tareas** :

1. **Opción 1 (LSTM)** :
   - Entrenar LSTM para predecir `TyreDegradation` secuencialmente (usar ventanas de 5 vueltas).
2. **Opción 2 (Regresión Cuantílica)** :
   - Usar XGBoost con función de pérdida quantile para predecir degradación en percentiles 10-50-90.
3. **Dataset** :
   - Combinar `TyreLife`, `TrackTemp`, y `LapTime` de FastF1.

- **Entregables** :
- Script `tyre_degradation.py` + predicciones en formato serie temporal.
- Gráfico interactivo con Plotly.
- **Asignatura** : _Aprendizaje Automático Avanzado_ .

---

#### **Semana 6: Integración Visión + Datos en Tiempo Real**

- **Tareas** :

1. **Sincronización Vídeo-Telemetría** :
   - Mapear timestamps de vídeo con datos de FastF1 (ej: `session.pos_data`).
   - Poder tener en cuenta timestamps de OpenF1 intervals.
   - 📌 Crear función `reconcile_gaps(vision_gap, fastf1_gap, openf1_gap)` → añadir overlay comparativo en video → sincronizar timeline triple (video + FastF1 + OpenF1)
2. **Visualización** :
   - Superponer gaps calculados (en segundos) sobre el vídeo con OpenCV.
3. **Opción CNN Custom** :
   - Si el tiempo permite, entrenar modelo para estimar degradación visual de neumáticos (usar imágenes de cámaras onboard).

- **Entregables** :
- Script `video_sync.py` + vídeo demo con overlays.
- Documentación de sincronización.
- **Asignatura** : _Visión por Computador_ .

---

#### **Semana 7: Simulador de Estrategias con Búsqueda Adversarial**

- **Tareas**:

  1. **Algoritmo Genético + Poda Alfa-Beta**:
     - Combinar DEAP con poda alfa-beta para simular estrategias rivales (ej: anticipar undercuts).
     - Tener en cuenta de nuevo intervals de : Explorar método API openfq https://openf1.org/#intervals en semana 4
     - 📌 Crear modelo `GapEvolutionPredictor` usando históricos OpenF1 → integrar predicción de gaps en fitness function → visualizar evolución de intervalos en árbol de decisiones
  2. **Fitness Function**:
     - Incluir penalización por riesgo de colisión o tráfico.
  3. **Visualización**:
     - Graficar árbol de decisiones adversarial con `NetworkX`.

- **Entregables**:

  - Script `genetic_algorithm.py` + gráficos de convergencia.
  - Ejemplo: "Estrategia óptima considerando respuesta de Mercedes".

- **Asignatura**: _Sistemas Inteligentes (Unidad II - 2.2)_.

---

#### Semana 8: Procesamiento de Radios con NLP (Actualizada)

**Objetivo** : Extraer información estratégica clave de las comunicaciones equipo-piloto usando NLP.

| **Tareas**                                 | **Herramientas/Detalles**                                                                                                                                                                                                             | **Entregables**                                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **1. Transcripción con Whisper**           | - Usar `whisper-timestamped` para transcribir radios con marcas de tiempo.                                                                                                                                                            | Dataset `radios_raw.csv` con columnas: `[timestamp, audio_path, text]`.                        |
| **2. Limpieza y Preprocesamiento**         | - Eliminar ruido (ej: "copy", "box box box") con expresiones regulares.                                                                                                                                                               | Script `clean_radios.py` + dataset `radios_clean.csv`.                                         |
| **3. Detección de Entidades con SpaCy**    | - Crear un modelo personalizado en SpaCy para reconocer:`` -  **Entidades** : Pilotos (`"HAM"`, `"VER"`), Neumáticos (`"soft"`, `"hard"`), Estrategias (`"undercut"`, `"overcut"`). ``- **Relaciones** : Ej: `"HAM → soft → lap 22"`. | Modelo SpaCy personalizado (`ner_strategy_model`) + dataset enriquecido `radios_entities.csv`. |
| **4. Clasificación de Intención con BERT** | - Fine-tuning de `bert-base-uncased` para detectar acciones (`"parar"`, `"continuar"`, `"adelantar"`).                                                                                                                                | Modelo BERT guardado en `models/nlp/bert_intent`.                                              |
| **5. Generación de Datos Sintéticos**      | - Usar GPT-4 para simular diálogos estratégicos con entidades anotadas.                                                                                                                                                               | Dataset sintético `radios_synthetic.csv` (500 ejemplos).                                       |

**Entregables Finales (Semana 8)** :

- Scripts: `transcribe_radios.py`, `spacy_ner.py`, `bert_intent_classifier.py`.
- Datasets: `radios_clean.csv`, `radios_entities.csv`, `radios_synthetic.csv`.

---

### **Mes 3: Integración y Refinamiento**

#### **Semana 9: API REST para Integración de Módulos**

- **Tareas** :

1. **Definir Endpoints** :
   - `/predict_strategy` (input: vídeo + telemetría, output: recomendación).
2. **Implementar con FastAPI** :
   - Conectar modelos entrenados (cargar con `joblib` o `torch.load`).
3. **Pruebas Locales** :
   - Enviar request POST con datos de prueba y validar respuesta.

- **Entregables** :
- API funcional en `api/main.py`.
- Colección Postman para pruebas.
- **Asignatura** : _Todas (integración transversal)_ .

---

#### **Semana 10: Modelo de Clasificación de Undercut/Overcut**

- **Tareas** :

  Explorar método API openfq https://openf1.org/#intervals otra vez.

- Crear features `gap_evolution_pre_pit` y `gap_evolution_post_pit` → implementar análisis SHAP para identificar umbrales críticos → desarrollar predictor de éxito basado en intervalos iniciales

1. **Dataset Histórico** :
   - Extraer casos de paradas y su resultado (ganancia/pérdida de posición).
2. **Opción 1 (XGBoost)** :
   - Entrenar clasificador binario (`undercut_exitoso`: sí/no).
3. **Opción 2 (Red Neuronal)** :
   - Implementar CNN 1D para tratar secuencias de laps pre-parada.

- **Entregables** :
- Script `undercut_classifier.py` + matriz de confusión.
- Ejemplo: "En condiciones similares, el undercut tiene un 78% de éxito".
- **Asignatura** : _Aprendizaje Automático Avanzado_ .

---

#### **Semana 11: Sistema de Decisiones con Agentes Competitivos RL**

- **Tareas**:

  1. **Entorno Multiagente**:
     - Crear dos agentes RL (ej: Tu equipo vs Red Bull) usando `Stable Baselines3`.
  2. **Recompensas Competitivas**:
     - Diseñar recompensas basadas en posición relativa (ej: +10 si adelantas, -5 si te adelantan).
  3. **Teoría de Juegos**:
     - Analizar equilibrios de Nash en estrategias simuladas.

- **Entregables**:

  - Script `rl_training.py` + video de simulación competitiva.
  - Informe de equilibrios estratégicos en `docs/game_theory.md`.

- **Asignatura**: _Sistemas Inteligentes (Unidad III - 3.3)_.

---

#### Semana 12: Dashboard con Explicaciones Enriquecidas (Nuevas Tareas)

**Objetivo** : Integrar las entidades detectadas por SpaCy en las explicaciones generadas.

| **Tareas**                                        | **Herramientas/Detalles**                                                                                                                                            | **Entregables**                                  |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **1. Vinculación Entidades-Recomendaciones**      | - Usar las entidades de SpaCy (ej: neumáticos detectados) para personalizar mensajes.``- Ejemplo:`"Parar en lap 22 (neumáticos HARD detectados en radio lap 20)"`.   | Lógica de vinculación en `dashboard_logic.py`.   |
| **2. Visualización de Entidades en Dashboard**    | - Mostrar entidades clave en el panel de Streamlit usando tarjetas interactivas.                                                                                     | Componente `entities_viewer.py` en el dashboard. |
| **3. Generación de Explicaciones con Plantillas** | - Crear plantillas Jinja2 que combinen predicciones ML + entidades SpaCy.``- Ejemplo:`"{{ driver }} debe parar en lap {{ lap }} ({{ entity }} detectado en radio)"`. | Plantillas en `templates/explanations.j2`.       |

**Entregables Finales (Semana 12)** :

- Dashboard con pestaña "Análisis de Radios" mostrando entidades y relaciones.
- Sistema de explicaciones basado en entidades detectadas.

---

### **Relación con Asignaturas**

| **Asignatura**                | **Componentes Añadidos**                                                                                          |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Procesamiento de Lenguaje** | - Modelo SpaCy para entidades estratégicas.``- Integración de entidades en explicaciones.                         |
| **Sistemas Inteligentes**     | - Uso de entidades para mejorar decisiones estratégicas (ej: priorizar paradas si se detecta "degradación alta"). |

---

### **Riesgos y Mitigación**

| **Riesgo**                                 | **Mitigación**                                                                      |
| ------------------------------------------ | ----------------------------------------------------------------------------------- |
| **Bajo rendimiento del modelo SpaCy**      | Usar el dataset sintético de GPT-4 para aumentar datos de entrenamiento.            |
| **Falta de contexto en las explicaciones** | Combinar SpaCy con LLMs (GPT-3.5) para generar texto natural a partir de entidades. |

#### **Semana 13: Testeo Integral y Optimización**

- **Tareas** :

1. **Validar con 2-3 Carreras** :
   - Comparar predicciones vs. estrategias reales (ej: ¿qué hizo Red Bull en Hungría 2023?).
2. **Optimizar Rendimiento** :
   - Convertir modelos a ONNX para inferencia rápida.
   - Perfilamiento con `cProfile` para identificar cuellos de botella.
3. **Documentación Final** :
   - Crear `README.md` con guía de instalación y ejemplos de uso.

- **Entregables** :
- Informe de testeo en `docs/testing_report.pdf`.
- Repositorio GitHub organizado y documentado.
- **Asignatura** : _Todas (integración transversal)_ .

---

### **Modularidad del Proyecto**

- **Estructura de Carpetas** :

  Copy

```
  f1-strategy/
  ├── data/          # Datos crudos y procesados
  ├── models/        # Modelos guardados (XGBoost, PyTorch, YOLO)
  ├── src/
  │   ├── vision/    # Scripts de visión por computador
  │   ├── ml/        # Modelos de aprendizaje automático
  │   ├── nlp/       # Procesamiento de lenguaje
  │   | ├── spacy_ner/            # Modelo SpaCy y scripts de entrenamiento
  │   | ├── bert_intent/          # Clasificador de intención con BERT
  |   | └── templates/            # Plantillas Jinja2 para explicaciones
  │   └── systems/   # Sistemas inteligentes y simulaciones
  ├── api/           # Código de la API FastAPI
  └── dashboard/     # Interfaz Streamlit
```

- **Independencia de Módulos** :
- Cada semana genera scripts en su carpeta correspondiente (ej: `src/vision/week2`).
- Comunicación entre módulos vía archivos JSON/CSV o llamadas API.

### **Sección Nueva: Adaptaciones para Sistemas Inteligentes**

| **Semana** | **Cambio Clave**                            | **Herramientas**             | **Unidad Vinculada**         |
| ---------- | ------------------------------------------- | ---------------------------- | ---------------------------- |
| 4          | Agente lógico con Pyke/Experta              | Pyke, Experta                | IV (4.1 - Agentes Lógicos)   |
| 7          | Búsqueda adversarial (alfa-beta + genético) | DEAP, NetworkX               | II (2.2 - Búsqueda Compleja) |
| 11         | Agentes RL competitivos                     | Stable Baselines3, OpenSpiel | III (3.3 - Teoría de Juegos) |
| 12         | Grafo de decisiones interactivo             | Graphviz, PyVis              | IV (4.2 - Razonamiento)      |

---
