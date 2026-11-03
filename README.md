# M29 · ¿Le va a gustar esta recomendación?

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

**Objetivo y métrica.** El objetivo real es mejorar la pertinencia de las recomendaciones para el usuario, anticipando qué títulos recibirían pulgar arriba y evitando recomendaciones inadecuadas. No buscamos maximizar clics, popularidad ni cantidad de títulos mostrados. Nuestra métrica principal es la precisión Verdadero positivo / Verdadero positivo + Falso positivo 

## 2. El sistema

**Cómo decide.** <Neurona: pesos y sesgo. Reglas: la lista de reglas si‑entonces. Agente: la regla con la que elige la acción.>

**Variables que usa y por qué.** <Cuáles usa, cuáles descarta y por qué.>

**Cómo se corre en vivo.** <Pasos concretos para resolver un caso nuevo: link al Colab, a la planilla o la cuenta a mano.>

**Dos casos de ejemplo.**

| Caso inventado | Valores | Salida del sistema |
|---|---|---|
| A | <…> | <SÍ/NO> |
| B | <…> | <SÍ/NO> |

## 3. Medición

**Entrenamiento (casos 1–14)**

| | Predice SÍ | Predice NO |
|---|---|---|
| **Real SÍ** | VP = <> | FN = <> |
| **Real NO** | FP = <> | VN = <> |

**Prueba (casos 15–20)**

| | Predice SÍ | Predice NO |
|---|---|---|
| **Real SÍ** | VP = <> | FN = <> |
| **Real NO** | FP = <> | VN = <> |

| Métrica | Entrenamiento | Prueba |
|---|---|---|
| Exactitud | <> | <> |
| Precisión | <> | <> |
| Sensibilidad | <> | <> |
| Especificidad | <> | <> |

**Entrenamiento contra prueba.** <¿Hay diferencia? ¿Por qué? ¿Sobreajuste o generalización?>

## 4. El error más caro

<¿Es peor el falso positivo o el falso negativo en su problema? ¿Por qué, según la ficha? ¿Qué cambiaron en el sistema para cometer menos ese error?>

## 5. ¿Comparado contra qué?

**Línea base trivial.** <Qué resultado saca un sistema que responde siempre lo mismo, y cómo se compara el suyo.>

| Checklist (Clase 6) | Respuesta |
|---|---|
| ¿Qué datos? | <> |
| ¿Qué métrica? | <> |
| ¿Comparado contra qué? | <> |
| ¿Quién lo midió? | <> |

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
