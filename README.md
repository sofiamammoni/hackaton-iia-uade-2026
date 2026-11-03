# M29 · ¿Le va a gustar esta recomendación?

> **Hackatón IIA · UADE · 3.4.257 · 3 de noviembre de 2026**

**Integrantes:** Sol Cubas · Sofia Mammoni  
**IA declarada:** ChatGPT  
**Pista:** Neurona (perceptrón de tres entradas)  
**Publicación en LinkedIn:** https://lnkd.in/p/dMw_Tj_P

---

## 1. Formulación y PEAS

**La pregunta.** Para cada recomendación de contenido en una plataforma de streaming, ¿el usuario le dará pulgar arriba? **SÍ (1)** significa que el usuario dará pulgar arriba; **NO (0)** significa que no lo dará. Cada caso representa un título candidato para un usuario. La respuesta real solo se conoce después de la decisión.

| PEAS | Su sistema |
|---|---|
| **Medida de rendimiento (P)** | Identificar recomendaciones que reciban pulgar arriba y reducir especialmente los falsos positivos, que perjudican la confianza. Medimos precisión, sensibilidad, especificidad y exactitud. |
| **Entorno (E)** | Plataforma de streaming, usuarios y catálogo de títulos candidatos. El gusto real del usuario no se conoce al momento de decidir. |
| **Actuadores (A)** | El prototipo responde **SÍ** (priorizar la recomendación) o **NO** (no priorizarla). No modifica una plataforma real. |
| **Sensores (S)** | Coincidencia con el historial (0 a 1), popularidad del título (percentil 0 a 100) y condición de estreno (0 o 1), disponibles antes de recomendar. |

**Objetivo y métrica.** Nuestro **objetivo** es mejorar la pertinencia de las recomendaciones, ofreciendo contenido que probablemente guste y preservando la confianza del usuario. La **métrica principal** es la **precisión = VP / (VP + FP)**, porque indica cuántas recomendaciones predichas como positivas realmente reciben pulgar arriba. La precisión es una medida parcial, no el objetivo: la satisfacción y la confianza no se reducen a una cifra. También medimos sensibilidad (para no ocultar buenas opciones), especificidad y exactitud. No buscamos maximizar popularidad, clics ni cantidad de títulos mostrados.

## 2. El sistema

**Cómo decide.** Elegimos un **perceptrón de tres entradas**. Multiplica cada característica por un peso, suma los resultados y un sesgo, y aplica una **función escalón** para producir una respuesta binaria.

```text
x1 = coincidencia con el historial (0 a 1)
x2 = popularidad / 100            (0 a 1)
x3 = estreno                      (0 o 1)

n = w1*x1 + w2*x2 + w3*x3 + b

Si n >= 0: predice SÍ (1)
Si n < 0:  predice NO (0)
```

El valor `n` es la **entrada neta** de la neurona: **no es una probabilidad** de que el contenido guste.

**Inicialización y aprendizaje.** Propusimos como valores iniciales `w = (1,00; 0,10; 0,05)` y `b = -0,75`, utilizando únicamente los casos de entrenamiento. En ellos, los dos positivos (3 y 9) tienen coincidencias con el historial de 0,99 y 0,93; el negativo con coincidencia más alta tiene 0,73. Por eso dimos mayor peso inicial al historial y menores contribuciones a popularidad y estreno. Estos son valores iniciales elegidos, no todavía aprendidos.

Entrenamos el perceptrón con los **casos 1 a 14**, en su orden original, con tasa de aprendizaje `eta = 0,01`, hasta obtener una pasada completa sin errores (máximo 100 pasadas). Cada vez que la predicción se equivoca aplicamos:

```text
error = y_real - y_predicha
w = w + eta * error * x
b = b + eta * error
```

Se realizaron **3 actualizaciones**: caso 5 en la pasada 1, caso 10 en la pasada 1 y caso 5 en la pasada 2. La pasada 3 finalizó sin errores. Los **parámetros finales** fueron:

```text
w1 = 0,9785
w2 = 0,0775
w3 = 0,0500
b  = -0,7800

n = 0,9785*x1 + 0,0775*x2 + 0,05*x3 - 0,78
```

Los pesos y el sesgo **sí se ajustaron mediante aprendizaje supervisado**. El peso de estreno no cambió porque los ejemplos que ocasionaron actualizaciones tenían `estreno = 0`; esto no demuestra que el estreno sea irrelevante. La inicialización, la tasa de aprendizaje y el orden de los casos pueden influir en el resultado.

**Variables que usa y por qué.** Usamos las tres características que proporciona la ficha: historial, popularidad y estreno, porque están disponibles al decidir. Dividimos popularidad por 100 para que no domine la suma únicamente por estar medida de 0 a 100. No utilizamos `caso`, `conjunto` ni `resultado_real` como entradas del modelo. La etiqueta real solo interviene en el entrenamiento supervisado y en la evaluación, **nunca para predecir casos nuevos**. No usamos los seis casos de prueba para actualizar pesos ni sesgo.

**Cómo se corre en vivo.** Abrir [`M29_neurona.ipynb`](M29_neurona.ipynb) en Google Colab, ejecutar todas las celdas en orden y, para un caso nuevo, llamar a `predecir(coincidencia, popularidad, estreno)`. Se puede obtener el mismo resultado a mano reemplazando los valores en la fórmula. La función no solicita la etiqueta real:

```python
def predecir(coincidencia, popularidad, estreno):
    if not (0 <= coincidencia <= 1 and 0 <= popularidad <= 100 and estreno in (0, 1)):
        raise ValueError("Entradas fuera del rango de la ficha")
    n = 0.9785*coincidencia + 0.0775*(popularidad/100) + 0.05*estreno - 0.78
    return ("SÍ" if n >= 0 else "NO", round(n, 5))
```

**Dos casos de ejemplo.** Son casos inventados, distintos de los 20 de la ficha:

| Caso inventado | Valores (historial; popularidad; estreno) | Entrada neta | Salida del sistema |
|---|---|---|---|
| A | 0,85; 50; 1 | 0,140475 | **SÍ (1)** |
| B | 0,40; 90; 0 | −0,31885 | **NO (0)** |

## 3. Medición

**Procedimiento.** Ajustamos los parámetros exclusivamente con los **14 casos de entrenamiento**, luego fijamos el modelo y lo evaluamos en los **6 casos de prueba**, sin volver a entrenarlo. Clase positiva = recibir pulgar arriba. Las etiquetas reales sirven para contar aciertos y errores, no como entradas al predecir.

**Entrenamiento (casos 1–14)**

| | Predice SÍ | Predice NO |
|---|---|---|
| **Real SÍ** | **VP = 2** | **FN = 0** |
| **Real NO** | **FP = 0** | **VN = 12** |

**Control:** 2 + 0 + 0 + 12 = **14**. Los casos positivos reales y predichos son el 3 y el 9.

**Prueba (casos 15–20)**

| | Predice SÍ | Predice NO |
|---|---|---|
| **Real SÍ** | **VP = 1** | **FN = 0** |
| **Real NO** | **FP = 0** | **VN = 5** |

**Control:** 1 + 0 + 0 + 5 = **6**. El caso positivo real y predicho es el 16.

| Métrica | Fórmula | Entrenamiento | Prueba |
|---|---|---|---|
| **Exactitud** | (VP + VN) / total | 14/14 = **100 %** | 6/6 = **100 %** |
| **Precisión** | VP / (VP + FP) | 2/2 = **100 %** | 1/1 = **100 %** |
| **Sensibilidad** | VP / (VP + FN) | 2/2 = **100 %** | 1/1 = **100 %** |
| **Especificidad** | VN / (VN + FP) | 12/12 = **100 %** | 5/5 = **100 %** |

**Control de predicción.** Con el modelo final, el caso 5 produce `n = -0,01299` y predice NO (real NO); el caso 9 produce `n = 0,20791` y predice SÍ (real SÍ); y el caso 16, reservado para prueba, produce `n = 0,05918` y predice SÍ (real SÍ). El notebook permite revisar las predicciones y matrices de los 20 casos.

**Entrenamiento contra prueba.** Las métricas coinciden en **100 %** en ambos conjuntos; no observamos una caída de rendimiento dentro de esta ficha. Sin embargo, **no podemos afirmar que el modelo generalice perfectamente ni descartar el sobreajuste**. Solo hay 2 positivos reales en entrenamiento y 1 en prueba. En los seis casos de prueba, un único error reduciría la exactitud en 16,67 puntos porcentuales. Un 100 % en esta muestra sintética no significa 100 % de aciertos para usuarios reales.

## 4. El error más caro

En esta problemática priorizamos evitar el **falso positivo (FP)**: la neurona dice SÍ, pero el usuario no le da pulgar arriba. Según la ficha, mostrar con confianza una recomendación que no gusta puede perjudicar la **confianza del usuario**. El **falso negativo (FN)** también importa, porque oculta un título que podría gustarle, pero aquí priorizamos el costo del FP. La ficha no asigna costos monetarios exactos a estos errores.

**Qué cambiamos.** Con los parámetros iniciales, el modelo tenía en entrenamiento **VP = 2, FN = 0, FP = 3, VN = 9**: los falsos positivos eran los casos 5, 6 y 10. Mediante las actualizaciones supervisadas de pesos y sesgo pasamos de `w = (1,00; 0,10; 0,05)`, `b = -0,75` a `w = (0,9785; 0,0775; 0,05)`, `b = -0,78`. Los FP de entrenamiento bajaron de **3 a 0** sin perder los dos positivos reales. En prueba el modelo final también obtuvo 0 FP; esto describe la muestra, no garantiza ausencia de errores futuros. No reajustamos el modelo utilizando prueba.

## 5. ¿Comparado contra qué?

**Línea base trivial.** Comparamos con un sistema que **siempre predice NO**, sin mirar los datos. Es una referencia relevante porque la clase NO es mayoritaria. Este sistema puede lograr una exactitud relativamente alta, pero nunca detecta recomendaciones que recibirían pulgar arriba.

| Medida de la línea base «siempre NO» | Entrenamiento | Prueba |
|---|---|---|
| **VP / FN / FP / VN** | 0 / 2 / 0 / 12 | 0 / 1 / 0 / 5 |
| **Exactitud** | 12/14 = **85,71 %** | 5/6 = **83,33 %** |
| **Precisión** | **No definida (0/0)** | **No definida (0/0)** |
| **Sensibilidad** | **0 %** | **0 %** |
| **Especificidad** | **100 %** | **100 %** |

**Comparación.** Nuestro perceptrón detecta los positivos de la ficha y obtiene mayor exactitud que la línea base. En cambio, «siempre NO» logra una exactitud elevada gracias a la abundancia de negativos, pero tiene sensibilidad 0 %. La precisión de esa línea base es **indefinida**, no 0 %, porque nunca predice SÍ. Por eso no basta con observar la exactitud de un sistema para determinar si cumple su objetivo.

| Checklist (Clase 6) | Respuesta |
|---|---|
| **¿Qué datos?** | 20 casos ficticios oficiales M29: 14 para entrenamiento y 6 para prueba. Solo tres variables disponibles antes de decidir. |
| **¿Qué métrica?** | Principal: precisión por el costo del falso positivo; también sensibilidad, especificidad y exactitud. |
| **¿Comparado contra qué?** | Línea base trivial «siempre NO», además de contrastar entrenamiento y prueba. |
| **¿Quién lo midió?** | Medición reproducible en el notebook preparado con asistencia de ChatGPT. Las integrantes son responsables de ejecutar, comprobar e interpretar los resultados. El bot evalúa el trabajo, no proporciona las etiquetas reales. |

**Controles de confiabilidad (Clase 6).** Revisamos la relación entre objetivo y métrica, la disponibilidad de las variables al decidir, la ausencia de fuga de etiquetas de prueba hacia el entrenamiento, la comparación con una línea base y las limitaciones de evaluar solamente 20 ejemplos ficticios. No presentamos estas métricas como evidencia de funcionamiento real a escala comercial.

## 6. Riesgo: el camino completo

**Datos → modelo → decisión → consecuencia.** Los 20 casos son inventados y pueden no reflejar la diversidad de gustos de una plataforma real (**datos**). La neurona prioriza la coincidencia histórica y podría aprender un patrón limitado o poco representativo para gustos nuevos (**modelo**). Podría predecir NO para una película novedosa que al usuario sí le gustaría (**decisión / falso negativo**). Entonces la plataforma dejaría de priorizar una buena opción y el usuario perdería oportunidades de descubrimiento (**consecuencia**). También es posible el riesgo inverso: un falso positivo podría perjudicar su confianza en el recomendador.

**Mitigación propuesta:** antes de implementar el modelo en una plataforma real, obtener datos representativos con las autorizaciones correspondientes, evaluar los errores en más casos y segmentos, comprobar cuándo están disponibles las variables y monitorear posibles cambios de preferencias. Estas pruebas adicionales **no se realizaron** en este hackatón.

## 7. Bitácora

**Uso de la IA.** Registramos la asistencia de IA, qué sugerencias adoptamos y qué opciones descartamos. Cada integrante debe poder defender las decisiones sin ayuda de IA.

| # | Qué le pedimos | Qué aceptamos | Qué rechazamos y por qué |
|---|---|---|---|
| 1 | Explicar la problemática de streaming y comparar formas de decidir SÍ/NO. | Trabajar con un perceptrón de tres entradas y una función escalón. | No elegir una regla basada en una sola variable como solución final, porque queríamos modelar las tres características disponibles. |
| 2 | Explicar las entradas y cómo combinar escalas distintas. | Dividir la popularidad por 100 para trabajar con variables entre 0 y 1. | No usar el percentil sin transformar con estos pesos, ya que podría dominar la suma por su escala. |
| 3 | Proponer y comprobar un entrenamiento reproducible, con métricas y matrices. | Entrenar con los 14 casos y evaluar separadamente en los 6 casos de prueba. | No ajustar parámetros con las etiquetas de prueba, para evitar contaminación de la evaluación. |
| 4 | Revisar el significado de los resultados del 100 % y sus limitaciones. | Explicar el tamaño pequeño y artificial de la muestra, y compararla con una línea base. | No alterar las matrices para mostrar porcentajes menos perfectos ni afirmar que el sistema funcionaría sin errores en usuarios reales. |

**Corridas contra el evaluador** (registrar solo ejecuciones reales de `@Evaluador_iia_uade_bot`):

| Corrida | Nota | Agujero que encontró | Qué cambiamos |
|---|---|---|---|
| Pendiente | Sin corrida todavía | Pendiente de devolución | Pendiente |

**Procedimiento para el evaluador:** enviar `M29`, pegar este README completo (en varios mensajes si es necesario) y finalizar con `FIN`. Registrar la nota, las preguntas del bot y los cambios efectivamente realizados antes de una nueva corrida. No inventamos resultados del evaluador.

---

**Fuentes y contenidos aplicados:** ficha M29 en `problematicas/manana.md`, `datos/M29.csv`, `BASES.md`, `GUIA_GITHUB.md`; contenidos de clase sobre PEAS, objetivo y métrica, evaluación crítica (Clase 6), neurona artificial, pesos, sesgo, función de activación, entrenamiento supervisado y matrices de confusión. Los parámetros y resultados del perceptrón corresponden al modelo desarrollado para este caso, no a números fijados por el profesor.
