# Guía Detallada para la Implementación de Mejoras en Recommendations View

## Prompt LLM para Implementación Paso a Paso

A continuación te presento el prompt detallado que debes usar para implementar cada característica avanzada, explicando cada punto, su funcionalidad, y cómo debería proceder:

```
Estás trabajando en un sistema de recomendación estratégica para carreras de Fórmula 1 en Streamlit. El componente principal es "recommendations_view.py", que muestra recomendaciones estratégicas generadas por un sistema de reglas basado en IA.

Tu tarea es mejorar este componente siguiendo las siguientes instrucciones. Implementa cada característica SOLO cuando te lo indique, proporcionando explicaciones detalladas del código y su funcionalidad:

### ESTRUCTURA DE DATOS DISPONIBLES:
- DataFrame "recommendations" con columnas: 'action', 'confidence', 'explanation', 'priority', 'lap_issued', 'DriverNumber', 'LapNumber', 'RacePhase', 'Position', 'Team', 'rule_fired', 'CompoundID', 'CompoundName'
- Los posibles valores de 'action' incluyen: 'pit_stop', 'extend_stint', 'prepare_pit', 'perform_undercut', 'defensive_pit', etc.
- 'confidence' va de 0 a 1, y 'priority' generalmente de 1 a 3 (mayor = más prioritario)

### CARACTERÍSTICA 1: Estrategia Óptima Completa
Implementa una visualización que muestre la "estrategia óptima" para toda la carrera. Esto debería:
- Identificar el conjunto óptimo de recomendaciones no conflictivas que cubran toda la carrera
- Visualizar esta estrategia como un diagrama de Gantt o línea de tiempo secuencial
- Mostrar claramente las transiciones entre diferentes fases (ej. stint 1 → pit stop → stint 2)
- Incluir métricas proyectadas como tiempos, posiciones esperadas
- Proporcionar una explicación general de la estrategia completa en lenguaje natural

### CARACTERÍSTICA 2: Detector de Conflictos Estratégicos
Crea un sistema que:
- Identifique automáticamente recomendaciones incompatibles (ej. 'extend_stint' y 'pit_stop' en lapsos cercanos)
- Resalte visualmente estos conflictos en la interfaz
- Proporcione explicaciones sobre por qué se consideran conflictivas
- Sugiera cómo resolver cada conflicto basándose en confianza/prioridad
- Permita al usuario decidir qué recomendación priorizar

### CARACTERÍSTICA 3: Integración Visual con Datos de Telemetría
Para cada recomendación, implementa visualizaciones que:
- Muestren los datos exactos de telemetría que desencadenaron la recomendación
- Incluyan gráficos pequeños incrustados (mini-gráficos o "sparklines")
- Para degradation_rate, muestre la curva de degradación
- Para gap_analysis, muestre la evolución del gap con el coche relevante
- Para lap_time, muestre la tendencia de tiempos por vuelta
- Estos gráficos deben ser contextuales a cada recomendación específica

### CARACTERÍSTICA 4: Comparador de Estrategias A/B
Desarrolla una herramienta que:
- Permita al usuario seleccionar dos conjuntos diferentes de recomendaciones
- Proyecte cómo se desarrollaría la carrera bajo cada escenario
- Compare visualmente ambas estrategias en términos de tiempos, posiciones, desgaste
- Muestre las ventajas y desventajas de cada enfoque
- Proporcione una recomendación final sobre qué estrategia parece superior

### CARACTERÍSTICA 5: Exportación de Informes de Estrategia
Implementa una funcionalidad que:
- Genere un informe completo en HTML o PDF con las recomendaciones seleccionadas
- Incluya todas las visualizaciones, explicaciones y métricas clave
- Esté estructurado profesionalmente para uso en briefings de equipo
- Tenga opciones de personalización (ej. incluir/excluir secciones)
- Permita guardar configuraciones de informes para uso futuro

### CARACTERÍSTICA 6: Análisis Competitivo
Crea una visualización que:
- Compare las recomendaciones de un piloto con datos de pilotos cercanos
- Muestre mapas de posición relativa para undercuts/overcuts
- Analice los gaps con respecto a los rivales directos
- Resalte oportunidades estratégicas basadas en las acciones de otros pilotos
- Proporcione alertas sobre posibles estrategias defensivas necesarias

### CARACTERÍSTICA 7: Mapa de Puntos Críticos de Decisión
Implementa un sistema que:
- Identifique y visualice los "momentos críticos" donde las decisiones tienen mayor impacto
- Muestre las ventanas de oportunidad óptimas para cada tipo de acción
- Calcule la sensibilidad de la carrera a diferentes decisiones en diferentes puntos
- Proporcione un mapa de calor o similar que muestre la importancia de cada vuelta
- Resalte los momentos donde múltiples opciones estratégicas están disponibles

### INSTRUCCIONES ADICIONALES:
- Explica detalladamente el código y las decisiones de diseño para cada implementación
- Proporciona esquemas visuales o mockups cuando sea relevante
- IMPORTANTE: SOLO implementa la característica actual cuando se te solicite explícitamente
- Espera confirmación antes de pasar al siguiente punto
- Al final de cada implementación, sugiere pequeñas mejoras adicionales relacionadas
```

## Detalles de Cada Punto y Recomendaciones de Implementación

### 1. Estrategia Óptima Completa

**Descripción detallada:**
Esta característica proporciona una versión "destilada" de todas las recomendaciones, ofreciendo una estrategia coherente para toda la carrera. Básicamente responde a la pregunta fundamental: "¿Qué debería hacer el piloto durante toda la carrera para obtener el mejor resultado posible?"

**Elementos clave:**

- Algoritmo para seleccionar recomendaciones compatibles (resolver conflictos automáticamente)
- Línea de tiempo visual dividida en "stints" y puntos de decisión
- Estimaciones de tiempo por vuelta, posición esperada y desgaste
- Resumen narrativo de la estrategia completa

**Implementación:**
Recomiendo crear un componente auxiliar: `optimal_strategy_generator.py` que contenga la lógica para generar la estrategia óptima y que luego se importe en el recommendations_view. Es una funcionalidad compleja que merece su propio módulo.

### 2. Detector de Conflictos Estratégicos

**Descripción detallada:**
Esta función analiza automáticamente todas las recomendaciones para detectar incompatibilidades lógicas. Por ejemplo, si en la vuelta 20 se recomienda "extend_stint" y en la vuelta 22 se recomienda "pit_stop", existe un conflicto.

**Elementos clave:**

- Matriz de compatibilidad entre diferentes tipos de recomendaciones
- Reglas temporales (qué separación mínima debe haber entre acciones conflictivas)
- Visualización de conflictos con código de colores
- Panel de resolución que permita elegir qué recomendación priorizar

**Implementación:**
Esta función puede implementarse directamente dentro de recommendations_view.py, ya que está estrechamente vinculada a la interfaz principal de recomendaciones.

### 3. Integración Visual con Datos de Telemetría

**Descripción detallada:**
Para cada recomendación, mostrar exactamente los datos que la desencadenaron hace que el sistema sea más explicable y confiable. Esto es esencial para la interpretabilidad de las recomendaciones de IA.

**Elementos clave:**

- Gráficos pequeños (mini-gráficos) dentro de cada tarjeta de recomendación
- Visualización específica para cada tipo de recomendación (degradación, gaps, tiempos)
- Resaltado de umbrales y puntos críticos en cada gráfico
- Controles para expandir los gráficos para un análisis más detallado

**Implementación:**
Recomiendo crear un componente auxiliar: `telemetry_visualizer.py` que contenga funciones específicas para generar estos mini-gráficos para diferentes tipos de datos. Así se mantiene el código modular y reutilizable.

### 4. Comparador de Estrategias A/B

**Descripción detallada:**
Esta herramienta permite a los usuarios comparar diferentes enfoques estratégicos para ver cómo funcionaría cada uno. Es especialmente útil en situaciones donde hay múltiples estrategias viables con diferentes compensaciones.

**Elementos clave:**

- Interfaz para seleccionar conjuntos de recomendaciones
- Simulación de carrera basada en las selecciones
- Visualización comparativa lado a lado
- Análisis de ventajas/desventajas de cada estrategia
- Recomendación final basada en objetivos del equipo

**Implementación:**
Definitivamente necesita un componente auxiliar: `strategy_comparator.py` debido a su complejidad y la posible reutilización en otras partes de la aplicación.

### 5. Exportación de Informes de Estrategia

**Descripción detallada:**
Esta función permite generar documentación profesional para reuniones de equipo, briefings con pilotos o revisiones post-carrera. Convierte los datos y visualizaciones en un formato presentable y completo.

**Elementos clave:**

- Generación de documentos HTML/PDF formatados
- Inclusión de visualizaciones estáticas y datos clave
- Estructura de informe profesional con secciones definidas
- Opciones de personalización para el contenido
- Sistema de guardado/carga de plantillas de informes

**Implementación:**
Recomiendo un componente separado: `report_generator.py` que pueda ser usado también en otras partes de la aplicación para generar informes.

### 6. Análisis Competitivo

**Descripción detallada:**
Esta característica pone las recomendaciones en contexto competitivo, mostrando cómo las estrategias propuestas interactúan con las condiciones de otros pilotos en pista.

**Elementos clave:**

- Mapa de posiciones relativas en pista
- Análisis de gaps con pilotos cercanos
- Estimación de estrategias probables de otros equipos
- Alertas sobre amenazas estratégicas
- Recomendaciones defensivas y ofensivas contextuales

**Implementación:**
Este análisis merece su propio módulo: `competitive_analysis.py` debido a su complejidad y la necesidad de procesar datos de múltiples pilotos simultáneamente.

### 7. Mapa de Puntos Críticos de Decisión

**Descripción detallada:**
Esta función ayuda a identificar los momentos cruciales de la carrera donde las decisiones tienen mayor impacto. Básicamente responde a "¿Cuándo es más importante tomar las decisiones correctas?"

**Elementos clave:**

- Mapa de calor o visualización similar a lo largo de las vueltas
- Cálculo de impacto potencial de decisiones en cada momento
- Identificación de ventanas de oportunidad óptimas
- Análisis de sensibilidad de la carrera a diferentes momentos
- Explicaciones de por qué ciertos momentos son críticos

**Implementación:**
Puede implementarse como parte de recommendations_view.py, pero también podría beneficiarse de un componente auxiliar si la lógica de cálculo es compleja.

## Integración del LLM de S12_lab

Una excelente estrategia para el siguiente paso sería integrar el LLM (de S12_lab.py) en áreas críticas de esta sección. Esta integración proporcionaría capacidades conversacionales y generativas avanzadas para:

1. **Generación de narrativas estratégicas**: Usar el LLM para crear explicaciones detalladas y naturales de las estrategias óptimas, similar a cómo lo haría un ingeniero de carrera
2. **Análisis de conflictos**: El LLM puede evaluar conflictos entre recomendaciones y explicar las compensaciones en lenguaje natural
3. **Informes personalizados**: Generar secciones de texto de alta calidad para los informes exportables, adaptando el tono según la audiencia (técnica para ingenieros, simplificada para directores de equipo)
4. **Simulación de comunicaciones**: Generar mensajes de radio sugeridos basados en recomendaciones estratégicas
5. **Respuestas a consultas específicas**: Permitir a los usuarios hacer preguntas directas sobre las recomendaciones en lenguaje natural

**Implementación recomendada**:
Crear un componente `llm_integration.py` que encapsule toda la lógica de comunicación con el LLM, incluyendo:

- Formateo de prompts específicos para cada caso de uso
- Gestión de tokens y costes
- Transformación de las respuestas del LLM en formatos utilizables
- Cache para respuestas comunes

Este componente podría luego ser importado por los diversos módulos que necesiten capacidades de generación de texto o análisis de lenguaje natural.

La arquitectura completa mejoraría significativamente la experiencia de usuario, combinando la potencia analítica de los modelos de datos con la capacidad explicativa e interpretativa de los LLMs.
