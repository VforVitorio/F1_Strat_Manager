### **Integración de Recomendaciones en Tiempo Real ("X coche vs Y coche")**

**Objetivo Ampliado** : Añadir un módulo de simulación estratégica que, dado un contexto específico (ej: coche A está 1.5s detrás de coche B en la vuelta 20), recomiende acciones óptimas (neumáticos, momento de parada, undercut/overcut) basado en condiciones actuales y predicciones.

---

### **Relación con Cada Asignatura y Herramientas**

<pre class="font-styrene border-border-100/50 overflow-x-scroll w-full rounded border-[0.5px] shadow-[0_2px_12px_hsl(var(--always-black)/5%)]"><table class="bg-bg-100 min-w-full border-separate border-spacing-0 text-sm leading-[1.88888] whitespace-normal"><thead class="border-b-border-100/50 border-b-[0.5px] text-left"><tr class="[tbody>&]:odd:bg-bg-500/10"><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Asignatura</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Componente Ampliado</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Herramientas/Enfoque</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Datos Necesarios</strong></th></tr></thead><tbody><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Visión por Computador</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">-<strong>Detección de posición relativa entre coches</strong> (gap en segundos).``<br />- <strong>Estado de los neumáticos</strong> (degradación visual en cámaras cercanas).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- YOLOv8 + DeepSORT (seguimiento de coches).<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- OpenCV para estimar distancia entre bboxes.</code>- Segmentación de neumáticos (si hay imágenes close-up).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Vídeos de carreras (YouTube) + cámaras onboard (ej:<a class="underline" href="https://f1tv.formula1.com/">F1TV</a> si accesible).</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Aprendizaje Automático Avanzado</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">-<strong>Modelo de predicción de tiempos por vuelta</strong> bajo diferentes neumáticos/clima.``<br />- <strong>Clasificación de undercut/overcut</strong> (¿ganará posición si para antes/después?).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- XGBoost/LightGBM para tabular data (variables: gap, tipo neumático, temperatura pista).``- LSTM para predecir evolución de gaps.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Datos históricos de paradas y gaps desde FastF1.</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Sistemas Inteligentes</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">-<strong>Simulación de escenarios</strong> (¿qué pasa si el coche para ahora vs. en 3 vueltas?).``<br />- <strong>Optimización multiobjetivo</strong> (tiempo total vs. riesgo vs. tráfico).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Algoritmos genéticos (DEAP library).``- MDPs (Markov Decision Processes) con RLlib.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Resultados de modelos predictivos + reglas de la FIA (ej: duración mínima de neumáticos).</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Procesamiento de Lenguaje</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">-<strong>Extracción de contexto estratégico</strong> de radios (ej: "El coche B está con neumáticos blandos usados").``<br />- <strong>Generación de explicaciones</strong> en lenguaje natural para las recomendaciones.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Fine-tuning de BERT para reconocer órdenes estratégicas.``- Plantillas de texto con variables (usando F-strings o Jinja2).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Transcripciones de radios (reales o sintéticas con GPT-4).</td></tr></tbody></table></pre>

---

### **Fuentes de Datos Adicionales**

1. **Gaps entre coches** :

* Calculados desde datos de posición de FastF1 (`session.pos_data`) o estimados por visión por computador (píxeles entre coches en vídeo → conversión a metros usando referencia de ancho de pista).

1. **Efectividad histórica de undercuts** :

* FastF1 permite extraer laps antes/después de paradas para calcular ganancias/pérdidas de posición.

1. **Temperatura de pista y degradación de neumáticos** :

* Columnas `TrackStatus` y `TyreLife` en datos de FastF1.

---

### **Ejemplo de Flujo de Trabajo**

1. **Entrada** : Vídeo en vivo (o grabado) + datos en tiempo real de FastF1.
2. **Procesamiento** :

* **Visión** : Detectar que el coche A (posición 5) está a 1.2s del coche B (posición 4).
* **ML** : Predecir que con neumáticos duros nuevos, el coche A hará vueltas 0.8s más rápidas que B en las próximas 5 vueltas.
* **Sistemas Inteligentes** : Simular que parar en la vuelta 22 permite un undercut exitoso (ganar 2 posiciones).
* **NLP** : Generar el mensaje: *"Recomendado: PARAR EN VUELTA 22. Neumáticos duros. Undercut posible al coche B (riesgo bajo: 75% éxito)."*

---

### **Herramientas Específicas para la Ampliación**

* **Simulación de Estrategias** : `FastF1` ya incluye funciones para calcular diferencias de tiempo (`calc_time_diff`).
* **Visualización de Estrategias** : Usar `Plotly` para mostrar gráficos de proyección de gaps y paradas.
* **Optimización** : `Optuna` para ajustar hiperparámetros de los modelos de decisión.

---

### **Riesgos y Soluciones**

<pre class="font-styrene border-border-100/50 overflow-x-scroll w-full rounded border-[0.5px] shadow-[0_2px_12px_hsl(var(--always-black)/5%)]"><table class="bg-bg-100 min-w-full border-separate border-spacing-0 text-sm leading-[1.88888] whitespace-normal"><thead class="border-b-border-100/50 border-b-[0.5px] text-left"><tr class="[tbody>&]:odd:bg-bg-500/10"><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Riesgo</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Mitigación</strong></th></tr></thead><tbody><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Datos en tiempo real limitados</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Usar modo histórico con datos de FastF1 (ej: GP de España 2023) para desarrollar y testear.</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Cálculo impreciso de gaps</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Combinar datos de posición de FastF1 (precisos) con visión por computador (solo como respaldo).</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Overfitting en modelos de predicción</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Validación cruzada con datos de múltiples circuitos (Monza, Mónaco, etc.).</td></tr></tbody></table></pre>

### **Plan de Desarrollo - 3 Meses**

**Objetivo** : Construir un MVP funcional del sistema de estrategia para F1, integrando visión por computador, modelos predictivos, simulación de decisiones y NLP.

---

### **Mes 1: Configuración y Componentes Básicos**

<pre class="font-styrene border-border-100/50 overflow-x-scroll w-full rounded border-[0.5px] shadow-[0_2px_12px_hsl(var(--always-black)/5%)]"><table class="bg-bg-100 min-w-full border-separate border-spacing-0 text-sm leading-[1.88888] whitespace-normal"><thead class="border-b-border-100/50 border-b-[0.5px] text-left"><tr class="[tbody>&]:odd:bg-bg-500/10"><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Semana</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Tareas</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Entregables</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Asignaturas Relacionadas</strong></th></tr></thead><tbody><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>1</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Configurar entorno Python (venv o conda).<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Explorar FastF1: Extraer datos históricos (ej: GP España 2023).</code>- Descargar vídeos de YouTube (Creative Commons) para pruebas.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Entorno listo + dataset inicial (CSV/Parquet).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Aprendizaje Automático, Sistemas Inteligentes</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>2</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Implementar detección de coches/banderas con YOLOv8 en un vídeo estático.``- Calcular gaps entre coches usando posición en pista (FastF1).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Script de detección + métricas de gaps.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Visión por Computador</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>3</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Preprocesar datos de telemetría (neumáticos, tiempos por vuelta).``- Entrenar modelo básico de regresión (ej: predecir tiempo de vuelta).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Modelo XGBoost inicial + informe de rendimiento (RMSE).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Aprendizaje Automático</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>4</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Extraer y procesar comunicaciones de radio utilizando NLP. <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Implementar análisis de sentimiento y extracción de entidades básicas.</code></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Dataset de radios procesado + modelo de sentimiento inicial.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Procesamiento de Lenguaje</td></tr></tbody></table></pre>

---

### **Mes 2: Desarrollo de Módulos Clave**

<pre class="font-styrene border-border-100/50 overflow-x-scroll w-full rounded border-[0.5px] shadow-[0_2px_12px_hsl(var(--always-black)/5%)]"><table class="bg-bg-100 min-w-full border-separate border-spacing-0 text-sm leading-[1.88888] whitespace-normal"><thead class="border-b-border-100/50 border-b-[0.5px] text-left"><tr class="[tbody>&]:odd:bg-bg-500/10"><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Semana</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Tareas</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Entregables</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Asignaturas Relacionadas</strong></th></tr></thead><tbody><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>5</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Sistema de decisiones con reglas lógicas mejoradas basadas en NLP. <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Integrar información de radios (sentiment, entidades) en reglas.</code></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Sistema de reglas enriquecido + pruebas de integración.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Sistemas Inteligentes</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>6</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Mejorar modelo predictivo: Incluir variables climáticas y tipo de neumático.``- Implementar LSTM para predecir degradación de neumáticos.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Modelo avanzado + gráficos de predicción.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Aprendizaje Automático</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>7</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Integrar visión por computador con FastF1: Sincronizar timestamp de vídeo y datos de telemetría.``- Calcular gaps en tiempo real usando YOLO + DeepSORT.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Script de sincronización + demo en vídeo.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Visión por Computador</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>8</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Desarrollar simulador de estrategias con algoritmos genéticos (DEAP): Minimizar tiempo total de carrera.``- Definir restricciones (ej: mínimo 1 parada).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Simulador funcional + resultados de optimización.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Sistemas Inteligentes</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>9</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Procesamiento avanzado de radios: Fine-tuning de BERT para clasificación de intención y reconocimiento de entidades estratégicas. <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Expandir el modelo de extracción de entidades con SpaCy.</code></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Modelo BERT y SpaCy avanzados + dataset etiquetado.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Procesamiento de Lenguaje</td></tr></tbody></table></pre>

---

### **Mes 3: Integración y Refinamiento**

<pre class="font-styrene border-border-100/50 overflow-x-scroll w-full rounded border-[0.5px] shadow-[0_2px_12px_hsl(var(--always-black)/5%)]"><table class="bg-bg-100 min-w-full border-separate border-spacing-0 text-sm leading-[1.88888] whitespace-normal"><thead class="border-b-border-100/50 border-b-[0.5px] text-left"><tr class="[tbody>&]:odd:bg-bg-500/10"><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Semana</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Tareas</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Entregables</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Asignaturas Relacionadas</strong></th></tr></thead><tbody><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>10</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Integrar todos los módulos en un flujo único (ej: input vídeo → output recomendación).``- Crear API REST con FastAPI para conectar componentes.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">API funcional + ejemplo de request/respuesta.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Todas</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>11</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Entrenar modelo de clasificación para undercut/overcut (usar datos históricos de paradas).``- Generar alertas visuales en el vídeo (ej: "Undercut recomendado").</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Modelo de clasificación + overlay en vídeo.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Aprendizaje Automático, Visión</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>12</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Mejorar el sistema de decisiones con RL (ej: Stable Baselines3).``- Añadir lógica de priorización de alertas (ej: riesgo alto/medio/bajo).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Agente de RL entrenado + sistema de priorización.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Sistemas Inteligentes</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>13</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Desarrollar dashboard interactivo en Streamlit: Mostrar predicciones, vídeo y recomendaciones.``- Añadir explicaciones en lenguaje natural (ej: "Parar en vuelta 22 porque...").</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Dashboard completo + NLP explicativo.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Procesamiento de Lenguaje</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>14</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Testeo integral: Validar con datos de 2-3 carreras distintas.``- Optimizar rendimiento (reducir latencia, mejorar precisión).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Informe de testeo + métricas finales (ej: precisión >85%).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Todas</td></tr></tbody></table></pre>

---

### **Mes 1 - Semana 1: Configuración Inicial y Extracción de Datos**

* [X] Marcar como hecho

* **Tareas** :

1. **Entorno** :
   * Crear entorno virtual (`conda create -n f1-strategy python=3.10`).
   * OpenF1 API también será utilizada para extraer mensajes de radio y otras cosas que FastF1 no posee.
   * Instalar librerías base: `fastf1`, `pandas`, `numpy`.
2. **Datos FastF1** :
   * Extraer datos del GP España 2023: Tiempos por vuelta, paradas, clima.
   * Guardar en formato Parquet para eficiencia.
3. **Vídeos** :
   * Descargar 2-3 resúmenes de carreras (YouTube Creative Commons) usando `pytube`.

* **Entregables** :
* Script `data_extraction.py` + dataset en `data/raw`.
* Documentación de estructura de datos.
* **Asignatura** : *Aprendizaje Automático / Sistemas Inteligentes* .

---

### **Mes 1 - Semana 2: Detección de Objetos con Visión por Computador**

* [X] Marcar como hecho

* **Tareas** :

* [X] Descargado dataset

1. **Opción 1 (YOLOv8)** :
   * [X] Entrenar YOLOv8-medium en dataset COCO para detectar coches (transfer learning). Finalmente, entrenado yolo medieum desde cero y buenos resultados.
   * [X] Probar en fotogramas de vídeo estático.
   * [X] Probar en vídeo dinámico.
   * [X] Reducir recall y reentrenar.
2. **Cálculo de Gaps** :
   * [X] Usar `OpenCV` para estimar distancia entre bboxes (píxeles → metros con referencia de ancho de pista).

* **Entregables** :
* Script `object_detection.py` + ejemplos de detección en `outputs/week2`.
* Métricas de precisión (mAP si es YOLO).
* **Asignatura** : *Visión por Computador* .

---

### **Mes 1 - Semana 3: Modelo Predictivo de Tiempos por Vuelta**

* [X] **Tareas** :

1. **Opción 1 (XGBoost/LightGBM)** :
   * Entrenar modelo para predecir `LapTime` usando variables: `TyreCompound`, `TrackTemp`, `AirTemp`.
2. **Opción 2 (Red Neuronal)** :
   * Implementar MLP en PyTorch con capas densas (64-32-16-1) y activación ReLU.
   * Comparar resultados con XGBoost.
3. **Feature Engineering** :
   * Crear variables como `TyreAge` (vueltas usadas) y `PositionChange`.

* **Entregables** :
* Script `lap_time_prediction.py` + modelo guardado en `models/week3`.
* Gráfico de dispersión predicciones vs. reales.
* **Asignatura** : *Aprendizaje Automático Avanzado* .

---

### **Mes 1 - Semana 4: Procesamiento de Radios con NLP (Adelantado de Semana 8)**

* [X] [ ]

 **Objetivo** : Extraer información estratégica de las comunicaciones equipo-piloto para enriquecer las reglas lógicas del sistema.

<pre class="font-styrene border-border-100/50 overflow-x-scroll w-full rounded border-[0.5px] shadow-[0_2px_12px_hsl(var(--always-black)/5%)]"><table class="bg-bg-100 min-w-full border-separate border-spacing-0 text-sm leading-[1.88888] whitespace-normal"><thead class="border-b-border-100/50 border-b-[0.5px] text-left"><tr class="[tbody>&]:odd:bg-bg-500/10"><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Tareas</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Herramientas/Detalles</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Entregables</strong></th></tr></thead><tbody><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>1. Extracción de Radios y Transcripción</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Usar <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">OpenF1 API</code> para extraer mensajes de radio.`- Implementar Whisper para transcribir audio a texto con timestamps.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Dataset <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">radios_raw.csv</code> con columnas: <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">[timestamp, audio_path, text]</code>.</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>2. Análisis de Sentimiento</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Implementar modelo básico de clasificación de sentimiento (positivo/negativo) con <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">transformers</code>.`- Etiquetar dataset de entrenamiento inicial (manual o semi-manual).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Modelo de sentimiento + dataset <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">radios_sentiment.csv</code>.</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>3. Extracción Avanzada de Información</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- RoBERTa para clasificación de intención (órdenes, información, preguntas, advertencias).<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Modelo NER personalizado de SpaCy para reconocer:</code>- <strong>Entidades</strong>: Pilotos, equipos, neumáticos, número de vueltas, diferencias de tiempo.<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- <strong>Términos estratégicos</strong>: undercut, overcut, estrategias de parada.</code></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Clasificador de intención RoBERTa + Modelo NER SpaCy personalizado + dataset <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">radios_info_estructurada.csv</code>.</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>4. Generación de Datos Sintéticos (opcional)</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Usar GPT-4 para simular radios estratégicas adicionales.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Dataset sintético <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">radios_synthetic.csv</code> (100 ejemplos iniciales para pruebas).</td></tr></tbody></table></pre>

 **Entregables Finales (Semana 4)** :

* Scripts: `radio_extraction.py`, `transcribe_radios.py`, `sentiment_analysis.py`, `extract_entities.py`.
* Datasets: `radios_raw.csv`, `radios_sentiment.csv`, `radios_entities_basic.csv`.
* Documentación: `radio_processing_guide.md` explicando el procesamiento y significado de las etiquetas.

 **Asignatura** :  *Procesamiento de Lenguaje* .

---

### **Mes 2: Desarrollo de Módulos Clave**

<pre class="font-styrene border-border-100/50 overflow-x-scroll w-full rounded border-[0.5px] shadow-[0_2px_12px_hsl(var(--always-black)/5%)]"><table class="bg-bg-100 min-w-full border-separate border-spacing-0 text-sm leading-[1.88888] whitespace-normal"><thead class="border-b-border-100/50 border-b-[0.5px] text-left"><tr class="[tbody>&]:odd:bg-bg-500/10"><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Semana</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Tareas</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Entregables</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Asignaturas Relacionadas</strong></th></tr></thead><tbody><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>5</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Mejorar modelo predictivo: Incluir variables climáticas y tipo de neumático.``- Implementar LSTM para predecir degradación de neumáticos.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Modelo avanzado + gráficos de predicción.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Aprendizaje Automático</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>6</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Sistema de decisiones con reglas lógicas mejoradas basadas en NLP. <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Integrar información de radios (sentiment, entidades) en reglas.</code></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Sistema de reglas enriquecido + pruebas de integración.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Sistemas Inteligentes</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>7</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Integrar visión por computador con FastF1: Sincronizar timestamp de vídeo y datos de telemetría.``- Calcular gaps en tiempo real usando YOLO + DeepSORT.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Script de sincronización + demo en vídeo.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Visión por Computador</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>8</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Desarrollar simulador de estrategias con algoritmos genéticos (DEAP): Minimizar tiempo total de carrera.``- Definir restricciones (ej: mínimo 1 parada).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Simulador funcional + resultados de optimización.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Sistemas Inteligentes</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>9</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Procesamiento avanzado de radios: Fine-tuning de BERT para clasificación de intención y reconocimiento de entidades estratégicas. <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Expandir el modelo de extracción de entidades con SpaCy.</code></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Modelo BERT y SpaCy avanzados + dataset etiquetado.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Procesamiento de Lenguaje</td></tr></tbody></table></pre>

---

### **Mes 2 - Semana 5: Modelo de Predicción y Degradación**

* [X]
* [X] **Tareas** :

1. **Opción 1 (LSTM)** :
   * Entrenar LSTM para predecir `TyreDegradation` secuencialmente (usar ventanas de 3 vueltas).
2. **Opción 2 (Regresión Cuantílica)** :
   * Usar XGBoost con función de pérdida quantile para predecir degradación en percentiles 10-50-90.
3. **Dataset** :
   * Combinar `TyreLife`, `TrackTemp`, y `LapTime` de FastF1. Usar el csv de lap_prediction.ipynb

* **Entregables** :
* Script `tyre_degradation.py` + predicciones en formato serie temporal.
* Gráfico interactivo con Plotly.
* **Asignatura** : *Aprendizaje Automático Avanzado* .

---

### **Mes 2 - Semana 6: Sistema de Decisiones Basado en Reglas Enriquecidas con NLP**

* [ ] [ ]

 **Objetivo** : Desarrollar un sistema de reglas lógicas que integre información de telemetría y comunicaciones de radio.

<pre class="font-styrene border-border-100/50 overflow-x-scroll w-full rounded border-[0.5px] shadow-[0_2px_12px_hsl(var(--always-black)/5%)]"><table class="bg-bg-100 min-w-full border-separate border-spacing-0 text-sm leading-[1.88888] whitespace-normal"><thead class="border-b-border-100/50 border-b-[0.5px] text-left"><tr class="[tbody>&]:odd:bg-bg-500/10"><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Tareas</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Herramientas/Detalles</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Entregables</strong></th></tr></thead><tbody><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>1. Diseño de Reglas Lógicas Avanzadas</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Implementar reglas usando <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">Experta</code> o <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">Pyke</code> que incorporen:**<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Datos de telemetría</code> (degradación, tiempos).<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- </code>Sentimiento de radio<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]"> (positivo/negativo).</code>- <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">Entidades extraídas</code> (neumáticos, acciones mencionadas).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Script <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">rule_engine.py</code> con base de conocimiento.</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>2. Integración de Datos Multi-fuente</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Crear pipeline para combinar datos temporales:<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Sincronizar timestamps de FastF1 con transcripciones de radio.</code>- Mapear sentimiento y entidades al timeline de carrera.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Script <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">data_integration.py</code> + dataset unificado <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">integrated_data.csv</code>.</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>3. Reglas Condicionadas por Sentimiento</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Implementar reglas específicas que utilizan señales de NLP:<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Ejemplo: </code>IF TyreDeg > 30% AND Radio_Sentiment = Negative THEN PitRisk = High`.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Conjunto de reglas en <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">knowledge_base/sentiment_rules.py</code>.</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>4. Sistema de Explicación para Decisiones Lógicas</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Crear sistema que justifique decisiones usando lenguaje natural:<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Ejemplo: </code>"Se recomienda parar debido a: 1) Alta degradación (35%), 2) Sentimiento negativo en últimas 3 radios"`.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Componente <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">rule_explanation.py</code> que genera explicaciones basadas en reglas activadas.</td></tr></tbody></table></pre>

## Plan Paso a Paso para Implementación

### 1. Preparación y Configuración

* Instalar Experta: `pip install experta`
* Crear estructura de módulos para el agente lógico

### 2. Definir la Estructura de Base de Conocimientos

* Crear clases para diferentes tipos de hechos:
  * `TelemetryFact`: degradación, tiempos por vuelta
  * `RadioFact`: sentimiento, entidades, intención
  * `PositionFact`: gaps, posición en pista
  * `WeatherFact`: temperatura, precipitación

### 3. Definir Reglas Básicas

* Implementar reglas simples de alta confianza:
  * Umbrales de degradación para recomendación de parada
  * Ajustes de estrategia basados en clima
  * Condiciones básicas de undercut/overcut

### 4. Integración con Modelos Existentes

* Crear pipeline de datos para transformar salidas de modelos en hechos:
  * Convertir predicciones del modelo de neumáticos
  * Transformar análisis NLP (sentiment, entities, intent)
  * Procesar datos de posición de FastF1

### 5. Implementar Reglas Complejas con Integración NLP

* Crear reglas que combinen telemetría y análisis NLP:
  * Reglas considerando sentimiento negativo sobre neumáticos
  * Reglas utilizando entidades detectadas
  * Reglas respondiendo a diferentes intenciones

### 6. Añadir Capacidades de Explicación

* Implementar seguimiento de activaciones de reglas
* Crear explicaciones en lenguaje natural para recomendaciones

### 7. Pruebas y Evaluación

* Probar el sistema con datos históricos de carreras
* Comparar recomendaciones con estrategias reales

 **Entregables Finales (Semana 6)** :

* Sistema completo de reglas en `src/systems/rule_based_system/`.
* Dashboard simple para visualización de reglas en `dashboard/rules_visualization.py`.
* Documentación de reglas y su relación con datos de NLP en `docs/rules_nlp_integration.md`.

 **Asignatura** :  *Sistemas Inteligentes (Unidad IV - 4.1)* .

### **Mes 2 - Semana 7: Integración Visión + Datos en Tiempo Real**

* [ ]

* **Tareas** :

1. **Sincronización Vídeo-Telemetría** :
   * Mapear timestamps de vídeo con datos de FastF1 (ej: `session.pos_data`).
2. **Visualización** :
   * Superponer gaps calculados (en segundos) sobre el vídeo con OpenCV.
3. **Opción CNN Custom** :
   * Si el tiempo permite, entrenar modelo para estimar degradación visual de neumáticos (usar imágenes de cámaras onboard).

* **Entregables** :
* Script `video_sync.py` + vídeo demo con overlays.
* Documentación de sincronización.
* **Asignatura** : *Visión por Computador* .

---

### **Mes 2 - Semana 8: Simulador de Estrategias con Búsqueda Adversarial**

* [ ]

* **Tareas** :

1. **Algoritmo Genético + Poda Alfa-Beta** :
   * Combinar DEAP con poda alfa-beta para simular estrategias rivales (ej: anticipar undercuts).
2. **Fitness Function** :
   * Incluir penalización por riesgo de colisión o tráfico.
3. **Visualización** :
   * Graficar árbol de decisiones adversarial con `NetworkX`.

* **Entregables** :
* Script `genetic_algorithm.py` + gráficos de convergencia.
* Ejemplo: "Estrategia óptima considerando respuesta de Mercedes".
* **Asignatura** :  *Sistemas Inteligentes (Unidad II - 2.2)* .

---

### **Mes 2 - Semana 9: Procesamiento Avanzado de Radios con NLP**

* [ ]

 **Objetivo** : Expandir la capacidad del sistema para extraer información estratégica compleja de las comunicaciones.

<pre class="font-styrene border-border-100/50 overflow-x-scroll w-full rounded border-[0.5px] shadow-[0_2px_12px_hsl(var(--always-black)/5%)]"><table class="bg-bg-100 min-w-full border-separate border-spacing-0 text-sm leading-[1.88888] whitespace-normal"><thead class="border-b-border-100/50 border-b-[0.5px] text-left"><tr class="[tbody>&]:odd:bg-bg-500/10"><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Tareas</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Herramientas/Detalles</strong></th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] font-400 px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>Entregables</strong></th></tr></thead><tbody><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>1. Fine-tuning de BERT para Intención</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Afinar BERT para clasificar intenciones estratégicas:<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- **Categorías**: Orden directa, Información, Pregunta, Advertencia.</code>- Usar datos etiquetados de la semana 4 + nuevos datos.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Modelo BERT guardado en <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">models/nlp/bert_intent</code> + métricas de evaluación.</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>2. Modelo Avanzado SpaCy para Entidades</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Crear un modelo personalizado en SpaCy para reconocer:<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- **Entidades complejas**: Estrategias (`"undercut"`, `"overcut"`), Condiciones pista.</code>- <strong>Relaciones</strong>: Ej: <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">"HAM → soft → lap 22"</code>.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Modelo SpaCy personalizado (<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">ner_strategy_model</code>) + dataset enriquecido <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">radios_entities.csv</code>.</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>3. Extracción de Información Temporal</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Implementar algoritmo para relacionar menciones temporales:<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Vincular "Box next lap" con número de vuelta actual.</code>- Extraer ventanas temporales ("Push for next 5 laps").</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Componente <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">temporal_extraction.py</code> + dataset <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">radios_temporal.csv</code>.</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>4. Expansión de Datos Sintéticos</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Ampliar dataset sintético con GPT-4:<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Generar 500 ejemplos estratégicos con anotaciones.</code>- Incluir situaciones específicas (ej: Safety Car, lluvia).</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Dataset expandido <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">radios_synthetic_v2.csv</code> (500 ejemplos).</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]"><strong>5. Evaluación y Refinamiento</strong></td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">- Evaluar rendimiento de modelos en datos reales.<code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">- Calcular precisión, recall y F1 para cada aspecto (sentimiento, entidades, intención).</code>- Refinar modelos según resultados.</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Informe de evaluación en <code class="bg-text-200/5 border border-0.5 border-border-300 text-danger-000 whitespace-pre-wrap rounded-[0.3rem] px-1 py-px text-[0.9rem]">reports/nlp_evaluation.md</code>.</td></tr></tbody></table></pre>

 **Entregables Finales (Semana 9)** :

* Modelos: BERT para intención, SpaCy para entidades avanzadas.
* Scripts: `bert_intent_classifier.py`, `spacy_ner_advanced.py`, `temporal_extraction.py`.
* Datasets: `radios_intent.csv`, `radios_entities.csv`, `radios_temporal.csv`, `radios_synthetic_v2.csv`.

 **Asignatura** :  *Procesamiento de Lenguaje* .

### **Mes 3: Integración y Refinamiento**

* [ ]

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

* [ ]

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

| **Tareas**                                         | **Herramientas/Detalles**                                                                                                                                        | **Entregables**                              |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **1. Vinculación Entidades-Recomendaciones**      | - Usar las entidades de SpaCy (ej: neumáticos detectados) para personalizar mensajes.``- Ejemplo:`"Parar en lap 22 (neumáticos HARD detectados en radio lap 20)"`. | Lógica de vinculación en `dashboard_logic.py`. |
| **2. Visualización de Entidades en Dashboard**    | - Mostrar entidades clave en el panel de Streamlit usando tarjetas interactivas.                                                                                       | Componente `entities_viewer.py` en el dashboard. |
| **3. Generación de Explicaciones con Plantillas** | - Crear plantillas Jinja2 que combinen predicciones ML + entidades SpaCy.``- Ejemplo:`"{{ driver }} debe parar en lap {{ lap }} ({{ entity }} detectado en radio)"`. | Plantillas en `templates/explanations.j2`.       |

**Entregables Finales (Semana 12)** :

- Dashboard con pestaña "Análisis de Radios" mostrando entidades y relaciones.
- Sistema de explicaciones basado en entidades detectadas.

---

### **Relación con Asignaturas**

| **Asignatura**                | **Componentes Añadidos**                                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Procesamiento de Lenguaje** | - Modelo SpaCy para entidades estratégicas.``- Integración de entidades en explicaciones.                         |
| **Sistemas Inteligentes**     | - Uso de entidades para mejorar decisiones estratégicas (ej: priorizar paradas si se detecta "degradación alta"). |

---

### **Riesgos y Mitigación**

| **Riesgo**                                 | **Mitigación**                                                               |
| ------------------------------------------------ | ----------------------------------------------------------------------------------- |
| **Bajo rendimiento del modelo SpaCy**      | Usar el dataset sintético de GPT-4 para aumentar datos de entrenamiento.           |
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

| **Semana** | **Cambio Clave**                        | **Herramientas**       | **Unidad Vinculada**    |
| ---------------- | --------------------------------------------- | ---------------------------- | ----------------------------- |
| 4                | Agente lógico con Pyke/Experta               | Pyke, Experta                | IV (4.1 - Agentes Lógicos)   |
| 7                | Búsqueda adversarial (alfa-beta + genético) | DEAP, NetworkX               | II (2.2 - Búsqueda Compleja) |
| 11               | Agentes RL competitivos                       | Stable Baselines3, OpenSpiel | III (3.3 - Teoría de Juegos) |
| 12               | Grafo de decisiones interactivo               | Graphviz, PyVis              | IV (4.2 - Razonamiento)       |

---
