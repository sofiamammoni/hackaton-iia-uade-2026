ex# M29 · ¿Le va a gustar esta recomendación?

**Integrantes:** Sol Cubas · Sofia Mammoni
**IA declarada:** ChatGPT
**Pista:** Neurona
**Publicación en LinkedIn:** <link>

---

## 1. Formulación y PEAS

**La pregunta.** Para cada recomendación de contenido en una plataforma de streaming, ¿el usuario le dará pulgar arriba?

| PEAS | Su sistema |
|---|---|
| Medida de rendimiento (P)| Identificar correctamente recomendaciones que recibirían pulgar arriba. Priorizamos la precisión porque un falso positivo perjudica la confianza del usuario. También medimos sensibilidad, especificidad y exactitud.|
| Entorno | Plataforma de streaming con usuarios, catálogo y títulos candidatos a recomendar. Cada decisión considera la relación usuario–título antes de conocer su reacción.|
| Actuadores | El prototipo emite una decisión binaria: SÍ (considerar/priorizar el título) o NO (no priorizarlo). No afirmamos que el prototipo ya modifique una plataforma real.|
| Sensores | Coincidencia con el historial (0–1), popularidad del título (percentil 0–100) y condición de estreno (0/1). Las tres características están disponibles antes de recomendar.|

**Objetivo y métrica.** El objetivo real es mejorar la pertinencia de las recomendaciones para el usuario, anticipando qué títulos recibirían pulgar arriba y evitando recomendaciones inadecuadas. No buscamos maximizar clics, popularidad ni cantidad de títulos mostrados. Nuestra métrica principal es la precisión = VP/(VP+FP) porque mide qué proporción de las predicciones positivas son realmente positivas; es especialmente relevante para los falsos positivos, que la ficha vincula con pérdida de confianza. Objetivo y métrica no son lo mismo: la satisfacción/confianza es el fin; la precisión es un indicador limitado. Por eso informamos también sensibilidad (para no ocultar recomendaciones buenas), especificidad, exactitud y una línea base trivial. Una precisión alta por predecir poquísimos positivos no bastaría para afirmar que el recomendador es útil.

## 2. El sistema

**Cómo decide.** <Elegimos un perceptrón de tres entradas, modelo conexionista inspirado matemáticamente en una neurona. Calcula una suma ponderada más sesgo y luego aplica una función escalón:
x1 = coincidencia con el historial       (entre 0 y 1)
x2 = popularidad / 100                  (entre 0 y 1)
x3 = estreno                            (0 o 1)
n  = w1*x1 + w2*x2 + w3*x3 + b
predicción = SÍ (1) si n >= 0; NO (0) si n < 0
El número n es una entrada neta, no una probabilidad de que al usuario le guste el título.

Elección inicial y aprendizaje. Antes de entrenar propusimos w = (1,00; 0,10; 0,05) y b = -0,75 a partir de los datos de entrenamiento: las dos etiquetas positivas (casos 3 y 9) tenían coincidencias 0,99 y 0,93, mientras que el negativo con mayor coincidencia tenía 0,73. Por ello dimos mayor peso inicial a coincidencia; asignamos contribuciones menores a popularidad y estreno, ya que no separan por sí solos todas las etiquetas. Son valores iniciales propuestos, no pesos supuestamente aprendidos. Usamos b = -0,75 como punto de partida para exigir suficiente evidencia positiva. Elegir una inicialización no demuestra por sí mismo el aprendizaje.

Luego realizamos entrenamiento supervisado del perceptrón exclusivamente con los casos 1–14, en el orden de la ficha, con tasa de aprendizaje eta = 0,01, hasta completar una pasada sin errores (máximo 100 pasadas). Para cada error, aplicamos:
error = y_real - y_predicha
w = w + eta * error * x
b = b + eta * error
En este entrenamiento ocurrieron 3 actualizaciones (caso 5 en pasada 1, caso 10 en pasada 1 y caso 5 en pasada 2). La pasada 3 terminó sin errores. El modelo final quedó:
w1 = 0,9785
w2 = 0,0775
w3 = 0,0500
b  = -0,7800

n = 0,9785*x1 + 0,0775*x2 + 0,05*x3 - 0,78
SÍ si n >= 0; NO si n < 0
Los pesos y el sesgo sí se ajustaron con ejemplos etiquetados. El peso de estreno no cambió porque los ejemplos que provocaron actualizaciones tenían estreno = 0; eso no demuestra que el estreno sea irrelevante. El aprendizaje depende de la inicialización, tasa, orden y tamaño de la muestra. En el procedimiento reproducible, las etiquetas de prueba no intervienen en las actualizaciones de parámetros y no se realiza ningún reajuste después de medirlas.

**Variables que usa y por qué.** Utilizamos las tres columnas predictoras de la ficha: coincidencia, popularidad y estreno. Dividir popularidad por 100 es una transformación fija de su percentil, no un cálculo de media/máximo aprendido con prueba. Priorizamos coincidencia por el patrón de entrenamiento y mantenemos las otras dos como contribuciones menores; no utilizamos caso, conjunto ni resultado_real como predictores. resultado_real se usa solamente para el entrenamiento supervisado y para evaluar después. El modelo es lineal: define una frontera de decisión, no comprende semánticamente las preferencias.

**Cómo se corre en vivo.** El archivo [`M29_neurona.ipynb`](M29_neurona.ipynb) contiene los 20 datos copiados exactamente del CSV oficial, el entrenamiento y las cuentas reproducibles. En Google Colab se abre el .ipynb, se ejecutan las celdas en orden y al final se utiliza predecir(coincidencia, popularidad, estreno) con un caso nuevo. Alternativa a mano: dividir la popularidad por 100, sustituir las tres entradas en la fórmula final y comparar n con cero. Ni la función de predicción ni la cuenta manual reciben la etiqueta real.
def predecir(coincidencia, popularidad, estreno):
    if not (0 <= coincidencia <= 1 and 0 <= popularidad <= 100 and estreno in (0, 1)):
        raise ValueError("Entradas fuera del rango de la ficha")
    n = 0.9785*coincidencia + 0.0775*(popularidad/100) + 0.05*estreno - 0.78
    return ("SÍ" if n >= 0 else "NO", round(n, 5))

**Dos casos de ejemplo.**

| Caso inventado | Valores | Salida del sistema |
|---|---|---|
| A | 0,85; 50; 1 | SÍ(1) |
| B | 0,40; 90; 0 | NO(0) |

## 3. Medición
Procedimiento. Fijamos los parámetros usando exclusivamente los 14 casos de entrenamiento, y luego evaluamos el perceptrón sin volver a ajustarlo sobre los 6 casos de prueba. Las etiquetas reales se consultaron para las métricas, nunca como entradas en la predicción. Positivo = pulgar arriba.

**Entrenamiento (casos 1–14)**

| | Predice SÍ | Predice NO |
|---|---|---|
| **Real SÍ** | VP = 2 | FN = 0 |
| **Real NO** | FP = 0 | VN = 12 |

**Prueba (casos 15–20)**

| | Predice SÍ | Predice NO |
|---|---|---|
| **Real SÍ** | VP = 1 | FN = 0 |
| **Real NO** | FP = 0 | VN = 5 |

| Métrica | Entrenamiento | Prueba |
|---|---|---|
| Exactitud | 100 | 100 |
| Precisión | 100 | 100 |
| Sensibilidad | 100 | 100 |
| Especificidad | 100 | 100 |

Control de predicción. En la salida neta final, el caso 5 tiene n = −0,01299 y se clasifica NO (real NO); el caso 9 tiene n = 0,20791 y se clasifica SÍ (real SÍ); el caso 16, que es de prueba y no participó del ajuste, tiene n = 0,05918 y se clasifica SÍ (real SÍ). El cuaderno muestra el detalle de los 20 casos y verifica automáticamente cada matriz.

**Entrenamiento contra prueba.** En esta ficha inventada las métricas resultaron iguales (100 %). Eso no prueba ausencia de sobreajuste ni garantía de generalización: solo existen dos positivos en entrenamiento y un positivo en prueba; con seis ejemplos de prueba, un error ya cambiaría la exactitud en 16,67 puntos porcentuales. Un porcentaje perfecto puede ser matemáticamente válido, pero debe contextualizarse con la línea base, cantidad de casos, posibles sesgos y evaluación en datos no vistos. El hecho de que el modelo acierte el caso 16 (positivo, estreno=0) muestra que no depende exclusivamente de la condición de estreno, sin demostrar validez general para todos los usuarios.

## 4. El error más caro

En M29 consideramos prioritario reducir los falsos positivos. Según la ficha, un FP significa mostrar con confianza una recomendación que no gusta y deteriorar la confianza del usuario; un FN significa ocultar un contenido que sí habría disfrutado, perdiendo una oportunidad. Ambos importan y la ficha no proporciona un costo monetario para calcularlos; elegimos priorizar FP por su impacto en la confianza, vigilando la sensibilidad para no volver al sistema excesivamente conservador.
Qué cambiamos realmente. Antes del entrenamiento, con pesos iniciales (1; 0,10; 0,05) y sesgo −0,75, el modelo cometía 3 falsos positivos en entrenamiento (casos 5, 6 y 10), con VP=2, FN=0, FP=3, VN=9. Al corregir errores con la regla supervisada del perceptrón, los pesos pasaron a (0,9785; 0,0775; 0,05) y el sesgo a −0,78, y los FP de entrenamiento bajaron de 3 a 0. En la evaluación final hubo 0 FP en prueba. No modificamos el modelo después de conocer las etiquetas de prueba. Esta mejora se observa en la muestra sintética y no garantiza que no existan FP en producción.

## 5. ¿Comparado contra qué?


**Línea base trivial.** Comparamos contra un clasificador que siempre responde NO, adecuado como referencia porque la clase negativa es mayoritaria en los datos. No es una recomendación para usarlo en la plataforma, sino un control necesario antes de celebrar una exactitud alta.
Medida de la línea base «siempre NO»	Entrenamiento	Prueba
VP / FN / FP / VN	0 / 2 / 0 / 12	0 / 1 / 0 / 5
Exactitud	12/14 = 85,71 %	5/6 = 83,33 %
Sensibilidad	0 %	0 %
Especificidad	100 %	100 %
Precisión	No definida (0/0)	No definida (0/0)


Comparación: nuestro perceptrón alcanza 100 % de exactitud en ambos conjuntos y detecta los casos positivos; la línea base acierta muchos NO, pero no detecta ningún SÍ. Por ello una exactitud aparentemente elevada (85,71 % en entrenamiento) no basta para juzgar la utilidad de una IA. No asignamos artificialmente precisión cero a una división 0/0.>

| Checklist (Clase 6) | Respuesta |
|---|---|
| ¿Qué datos? | 20 ejemplos sintéticos de la ficha M29; 14 de entrenamiento y 6 de prueba con la división fija. Tres entradas conocidas al decidir; resultado_real es la etiqueta para aprender y medir. |
| ¿Qué métrica? | Principal: precisión por el costo de FP; acompañada de sensibilidad, especificidad y exactitud, con matrices que cierran.|
| ¿Comparado contra qué? | Clasificador trivial «siempre NO» y comparación explícita entre entrenamiento y prueba.|
| ¿Quién lo midió? | Cálculos reproducibles implementados con asistencia de ChatGPT; las integrantes deben ejecutar y verificar el cuaderno y responder por las conclusiones. El bot es un evaluador adicional, no la fuente de las etiquetas. |

## 6. Riesgo: el camino completo

<Datos → modelo → decisión → consecuencia aplicado a su caso. Al menos un riesgo concreto.>

## 7. Bitácora

**Uso de la IA**

| # | Qué le pedimos | Qué aceptamos | Qué rechazamos y por qué |
|---|---|---|---|
| 1 | <> | <> | <> |

**Corridas contra el evaluador**

| Corrida | Nota | Agujero que encontró | Qué cambiamos |
|---|---|---|---|
| 1 | <> | <> | <> |
