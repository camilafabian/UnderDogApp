# Guía de estudio · ¿Gana el underdog?

**Proyecto integrador · Ciencia de Datos · UTN FRM · 5K10**

Matias Almendros · Juan Caballero · Camila Fabian · Ignacio Fernandez · Ignacio Miqueleiz · Matias Roiz

Esta guía explica el proyecto entero, de punta a punta, para que cualquiera del grupo pueda responder una pregunta sobre cualquier
parte. Los números son los mismos que muestra la página.

<div class="resumen" markdown="1">

**El proyecto en tres frases**

1. Queremos saber si se puede predecir, antes del partido, cuándo gana el equipo no favorito (el underdog).
2. Etapa 1: el modelo predice tan bien como la cuota de las casas de apuestas, pero no mejor (AUC 0,660 contra 0,656).
3. Etapa 2: usado con una regla de apuesta, es la única estrategia que apuesta y no pierde plata ni en validación (+0,8%) ni en el test (+6,6%), aunque esa ganancia todavía podría ser suerte.

</div>

## 1. Los conceptos básicos

**Cuota.** Lo que devuelve la casa de apuestas por cada peso si un resultado sale. Cuota 8,56: apostás $100 y, si sale, te devuelven $856.

**Probabilidad implícita.** 1 dividido la cuota. Cuota 8,56 → 1 / 8,56 = 11,7%. Es la probabilidad que la casa le asigna a ese resultado.

**Margen de la casa.** Las tres probabilidades de un partido (local, empate, visitante) suman más de 100%. En Leverkusen–Frankfurt suman
101,9%: el 1,9% que sobra es la ganancia de la casa. Por eso apostar a ciegas pierde en el largo plazo.

**Probabilidad sin margen.** Cada probabilidad dividida por la suma (101,9%). Es la que usa el modelo como dato de entrada.

**Favorito y underdog.** Comparando local contra visitante (el empate no es un equipo), el favorito es el que tiene más probabilidad
de ganar y el underdog, el otro. En Leverkusen–Frankfurt: Leverkusen 69,6%, Frankfurt 11,5%.

**Corte de 5 puntos.** Sólo usamos partidos donde favorito y underdog se separan por más de 5 puntos de probabilidad. Por debajo, quién
es el underdog depende de qué casa se mire: Pinnacle y Bet365 eligen distinto favorito en el 54% de los partidos casi parejos, y en el 0% desde 4,5 puntos.

**Qué predecimos (el objetivo).** `gano_no_favorito`: 1 si ganó el underdog, 0 si empató o perdió. Es clasificación binaria.

**Tasa base.** El underdog ganó 3.654 de 17.959 partidos: 20,3%, unos 2 de cada 10. Un modelo que dice siempre «20%» no mira nada:
es el piso que cualquier modelo tiene que superar.

## 2. Los datos y el pipeline

**Fuente.** European Soccer Database (Kaggle): partidos con cuotas 1X2 de 10 casas de apuestas, la alineación titular real de cada
partido y los ratings FIFA de cada jugador a lo largo del tiempo.

**Arquitectura.** La base se carga una sola vez en PostgreSQL. Un DAG de Airflow la consulta por una Airflow Connection y arma dos capas:

- **Bronze:** lo que devuelve cada consulta, sin tocar, guardado en Parquet.
- **Silver:** el dataset para modelar. Una fila = un partido, 17.959 partidos × 31 columnas.

**Filtros, paso por paso.**

| Paso | Se sacan | Quedan |
|---|---|---|
| Partidos en la fuente | | 25.979 |
| Ligas sin cuotas (Polonia y Suiza) | 3.342 | 22.637 |
| Alineación titular incompleta | 2.910 | 19.727 |
| Sin cuotas 1X2 completas | 33 | 19.694 |
| Cuota de empate incoherente (empate implícito > 40%) | 32 | 19.662 |
| Las dos probabilidades idénticas | 103 | 19.559 |
| Sin un underdog claro (diferencia ≤ 5 puntos) | 1.600 | 17.959 |

**Fuga de información (data leakage).** Todo lo que sólo existe después del partido (goles, tarjetas, córners, posesión, resultado)
queda fuera del dataset de modelado. Los goles se usan una única vez, para construir el objetivo, y van a una tabla de auditoría aparte.
El DAG falla si alguna de esas columnas aparece en Silver.

**Rating de cada jugador.** Se usa el último registro FIFA *estrictamente anterior* a la fecha del partido, nunca uno posterior.

**Datos que no están en el repositorio.** El Silver y los modelos no se suben a git: cada uno los genera en su compu corriendo el
pipeline y el notebook. La página guarda un resumen (`docs/data/`) para poder regenerarse sin ellos.

## 3. Lo que encontramos en el análisis (Entrega 2)

**Los datos (sección 1 de la página).** Cuanto más favorito es el favorito, menos gana el underdog: 3 de cada 10 en partidos parejos,
3 de cada 100 cuando el favorito es abrumador. La cuota acierta bastante.

**H1 · se sostuvo.** Donde dos casas eligen distinto favorito, el partido es parejo. De ahí sale el corte de 5 puntos.

**H2 · refutada.** «El rating del once es mayor cuando gana el underdog.» Mirado solo, sí: con un once mejor, las victorias del underdog
pasan de 11% a 27%. Pero el rating del once correlaciona 0,66 con la probabilidad de la cuota. Al controlar por la cuota, el efecto casi
desaparece (separación de 0,36 a 0,03–0,16, rojo en el semáforo). Las casas ya ponían en el precio la calidad de los titulares.
Es el gráfico de la sección 2: la línea de lo que pasó y la de lo que esperaba la cuota van pegadas.

**Columnas del modelo.** 12: las 2 probabilidades de la cuota (underdog y empate) y 10 diferencias de rating entre los titulares
(promedio del once, 3 mejores, mejor, peor, defensa, mediocampo, ataque, definición, reacciones y marca).

## 4. Etapa 1 · ¿El modelo predice? (Entrega 3)

**El modelo.** Regresión logística con las 12 columnas: las probabilidades de la cuota en escala logit y las diferencias de rating
estandarizadas. C = 1 (elegido por grilla; desde C = 0,1 la AUC ya no cambia). Es una fórmula, por eso la página la calcula en el navegador.

**Cómo se validó.** `TimeSeriesSplit` en cinco pedazos sobre 2008/09–2014/15: siempre se entrena con el pasado y se valida con lo que
viene. El test es la temporada 2015/16, mirada una sola vez.

**Comparación de modelos (validación).**

| Modelo | AUC media | Desvío |
|---|---|---|
| Piso (Dummy) | 0,500 | 0,000 |
| La cuota tal cual | 0,637 | 0,011 |
| Regresión logística (elegida) | 0,636 | 0,009 |
| Árbol de decisión | 0,624 | 0,011 |
| Random forest | 0,632 | 0,013 |
| Gradient boosting | 0,633 | 0,012 |

Ninguno supera a la logística por más que el desvío. Los jugadores solos dan AUC 0,599; con la cuota, no suman casi nada.

**Las métricas, en palabras.**

- **AUC:** si tomás un partido que ganó el underdog y otro que no, qué tan seguido el modelo le dio más chance al primero. 0,5 = tirar una moneda; 1 = siempre.
- **Log-loss y Brier:** cuánto se alejan, en promedio, las probabilidades de lo que pasó. Menos es mejor.
- **Accuracy no sirve:** decir siempre «no gana» acierta el 80% sin mirar nada.

**Resultado en el test (2015/16, 2.399 partidos).**

| | AUC | Log-loss | Brier |
|---|---|---|---|
| Piso: decir siempre 20% | 0,500 | 0,5293 | 0,1725 |
| La cuota sola | 0,656 | 0,4983 | 0,1634 |
| Nuestro modelo | 0,660 | 0,4981 | 0,1633 |

**Calibración.** Diez grupos de partidos según la probabilidad que dio cada uno; se compara esa probabilidad con lo que pasó. El modelo
y la cuota siguen la diagonal: cuando dicen 30%, gana alrededor de 3 de cada 10.

**Matriz de confusión (corte en 20%).** De los 530 partidos que ganó el underdog en el test, el modelo marcó 393 y se le escaparon 137,
con 941 falsas alarmas. La cuota marca 413 con 994 falsas alarmas. Con corte en 50%, el modelo no dice «gana» nunca: acierta el 78% sin
servir para nada.

**Conclusión de la etapa 1.** El modelo predice, pero no mejor que la cuota.

## 5. Etapa 2 · ¿Sirve para apostar?

**La regla de apuesta.** Apostar $100 al underdog sólo si `probabilidad del modelo × cuota − 1 > 0`, es decir, si el valor esperado es
positivo. Funciona como una calibración posterior al modelo: no cambia qué tan bien predice, cambia cuándo se decide apostar.

**El ejemplo.** Frankfurt, cuota 8,56. Paga como si tuviera 11,7% de chances (1 / 8,56). El modelo le da 14,8%.
0,148 × 8,56 − 1 = +26,5%: por cada $100, espera ganar $27 en el largo plazo, y apuesta. Leverkusen ganó 3–0 y la apuesta se perdió:
valor esperado positivo no es ganar cada vez.

**Con y sin la regla.** «Sin la regla» = apostar cada vez que la probabilidad del modelo supera un umbral fijo (15%, el mejor en
validación entre 15, 20, 25, 30 y 35%).

| Estrategia | Validación (2010–2015) | Test (2015/16) |
|---|---|---|
| Modelo con la regla | +0,8% (1.396 apuestas) | +6,6% (404 apuestas) |
| Modelo sin la regla (> 15%) | −5,8% | +6,8% |
| Apostar a todos | −7,7% | −0,9% |

Sin la regla, en validación el modelo pierde casi lo mismo que apostar a ciegas. En el test a casi todo le fue mejor porque fue una
temporada buena para los underdogs (ganaron 22,1%).

**Estrategias simples (dummies).** Cada una con su parámetro elegido en validación, igual que el modelo.

| Apostar al underdog cuando… | Validación | Test |
|---|---|---|
| El modelo ve valor (la regla) | +0,8% | +6,6% |
| Siempre (apostar a todos) | −7,7% | −0,9% |
| La cuota le da entre 10% y 20% (mejor rango en validación) | +0,8% | −5,3% |
| Su once es mejor que el del favorito | −4,9% | +8,7% |
| La cuota le da 30% o más (partidos parejos) | −4,7% | +2,1% |
| La cuota le da menos de 10% (batacazos) | −25,2% | −33,9% |
| Nunca | 0% | 0% |
| Al azar, la misma cantidad de veces (mediana de 300 sorteos) | −7,9% | −0,7% |

**Retorno acumulado en el test.** El gráfico mide cuánto deja cada $100 apostados, no la plata total, porque las estrategias apuestan
distinta cantidad de veces (el modelo, 404; «sin la regla», 1.720). En plata total, «sin la regla» parecía arrasar (+$11.616 contra
+$2.672) sólo porque apostaba cuatro veces más; por cada $100 quedan casi iguales (+6,8% y +6,6%). Al final de la temporada: modelo +6,6%,
apostar a todos −0,9%, rango de cuota −5,3%, «once mejor» +8,7%.

**La lectura honesta.**

- Le gana a apostar a todos y al rango de cuota elegido en validación.
- «Once mejor» y el modelo sin la regla dan algo más en el test, pero los dos perdían en validación: sin mirar el test, no los habríamos elegido.
- El +6,6% no se distingue del azar: el 5% de los sorteos al azar supera +15,1%, y el intervalo de confianza del modelo es [−14,6%; +29,0%], que incluye el cero.
- En validación, el modelo supera al 95% de los sorteos al azar; en el test, al 79%.
- Lo que sí se sostiene: de las que apuestan, es la única estrategia que no pierde plata en ningún período.

**Conclusión de la etapa 2.** No le ganamos a las casas de apuestas, pero dejamos de perder contra ellas.

## 6. El recorrido, entrega por entrega

| Entrega | Qué pensábamos o hicimos | Qué nos marcaron o descubrimos | Qué cambió |
|---|---|---|---|
| 1 | Los jugadores podían anticipar al underdog. Dataset de 82 columnas | Usar Airflow para hablar con una base de datos; sacar los goles | Pipeline Bronze → Silver sobre PostgreSQL, sin goles |
| 2 | Análisis exploratorio con cuatro hipótesis | La cuota ya sabía lo de los jugadores (H2 refutada) | Corte de 5 puntos y 12 columnas para el modelo |
| 3 | Elegimos por AUC un modelo de una sola columna | Era «la cuota con otro nombre»: nunca tomaba una decisión | Medir también en plata: el de 12 columnas deja de perder |
| 4 | La primera app daba +4,6% en el Explorá | El modelo había entrenado con esos partidos | Sólo predicciones fuera de muestra |
| Final | — | Explicar el problema desde cero, separar predecir de apostar, comparar con estrategias simples, sólo el test | Esta versión de la página |

Lo que no cambió desde la Entrega 2: la cuota ya sabe casi todo.

## 7. La página, sección por sección

| Sección | Qué muestra |
|---|---|
| Portada | El ejemplo de Barcelona–Valencia, las dos preguntas, favorito y underdog, los datos |
| 0 · El problema | Las cuotas de un partido real, el margen de la casa, las definiciones y el 20% |
| 1 · Los datos | Cuánto gana el underdog según qué tan favorito es el favorito |
| 2 · Lo que encontramos | El rating del once contra lo que esperaba la cuota |
| 3 · ¿El modelo predice? | Métricas, calibración, matriz de confusión (la curva ROC en «Cómo lo sabemos») |
| 4 · ¿Sirve para apostar? | La regla, con y sin la regla, estrategias simples y azar en el test |
| El recorrido | La línea de tiempo de las entregas |
| 5 · Explorá | Los 2.399 partidos del test: filtro por liga, recuadro y las estrategias recalculadas |
| 6 · Probá el modelo | Cargar un partido (real o inventado) y ver la probabilidad y si conviene apostar |

Cada sección del tallo tiene un «Cómo lo sabemos» con el detalle técnico. En la exposición no se abren: lo importante está escrito arriba.

**Cómo se genera.** `app/build_app.py` toma lo que producen el DAG y el notebook y escribe `docs/index.html`. Con `--desde-publicado`
la rehace sólo con lo que está en `docs/data/`, sin el Silver. GitHub Pages publica la carpeta `docs/`.

## 8. Limitaciones

- La base termina en 2015/16: sirve para entrenar y validar, no para predecir la fecha que viene.
- Las cuotas de la fuente se publican días antes del partido. Una cuota de cierre, más cercana al pitazo, sería más difícil de superar.
- Una sola temporada de test (404 apuestas): no alcanza para distinguir la ganancia del azar.
- La casa de apuestas usada cambia con los años (Bet365 hasta 2011/12, Pinnacle desde 2012/13), con márgenes distintos.
- No hay estado de forma, lesiones ni suspensiones como columnas.

## 9. Preguntas posibles, con respuesta

**Sobre el problema y los datos**

- **¿Qué es una fila?** Un partido. 17.959 en total.
- **¿Qué predicen?** Si gana el underdog: 1 si gana, 0 si empata o pierde.
- **¿Cómo definen al favorito?** Con la probabilidad implícita de la cuota, sin el margen: el de mayor probabilidad entre local y visitante.
- **¿Por qué el corte de 5 puntos?** Porque por debajo dos casas de apuestas eligen distinto favorito en el 54% de los partidos.
- **¿De dónde sale el 20%?** 3.654 victorias del underdog sobre 17.959 partidos.
- **¿Hay fuga de información?** No: los goles y el resultado sólo construyen el objetivo y quedan fuera del dataset; el DAG falla si aparecen.
- **¿Por qué no usaron la liga?** Porque la tasa de victorias del underdog casi no cambia entre ligas (H4 refutada).

**Sobre el modelo (etapa 1)**

- **¿Por qué regresión logística y no un modelo más complejo?** Ninguno le ganó por más que el desvío, y los complejos empeoran en lo que no vieron.
- **¿Por qué no accuracy?** Porque decir siempre «no gana» acierta el 80%.
- **¿Qué es el AUC?** La probabilidad de que el modelo le dé más chance a un partido que ganó el underdog que a uno que no.
- **¿Por qué la matriz con corte en 20%?** Es la tasa base. Con 50% el modelo no marca ningún partido.
- **¿Los jugadores no sirven para nada?** Solos predicen algo (AUC 0,599), pero la cuota ya tiene esa información.
- **¿En qué se apoya el modelo?** Casi todo el peso está en la probabilidad de la cuota (coeficiente 0,947). Al ser menor que 1, achica las probabilidades extremas: a los underdogs muy improbables les da un poco más que la cuota, y de ahí salen la mayoría de las apuestas.

**Sobre la plata (etapa 2)**

- **¿Qué es la regla de apuesta?** Apostar sólo si probabilidad del modelo × cuota − 1 > 0, es decir, si el valor esperado es positivo.
- **¿Por qué dicen que es una calibración posterior?** Porque se aplica después del modelo: no cambia la predicción, cambia cuándo se apuesta.
- **¿Entonces el modelo gana plata?** No está demostrado. No pierde en ningún período, pero el +6,6% del test no se distingue del azar.
- **¿Por qué ese rango de cuota y ese umbral?** Se eligieron sólo con la validación, entre opciones fijadas antes, igual que el modelo.
- **Si «once mejor» da más en el test, ¿por qué no lo usan?** Porque perdía en validación: elegirlo mirando el test sería hacer trampa.
- **¿Por qué el test anda mejor que la validación?** 2015/16 fue una temporada buena para los underdogs (22,1% de victorias).
- **¿Por qué el gráfico del test mide por cada $100 y no en plata?** Porque en plata total gana la estrategia que más veces apuesta, aunque deje menos por apuesta. Por cada $100 se comparan en la misma escala.
- **¿Qué es la banda gris?** Lo que da apostar al azar la misma cantidad de veces que el modelo (90% central de 300 sorteos).
- **¿Por qué mostrar sólo el test?** Es lo único que el modelo final nunca vio. La validación está en los «Cómo lo sabemos».

**Sobre el proceso y la app**

- **¿Qué cambió desde la Entrega 3?** Separamos predecir de apostar, sumamos estrategias simples y el azar, y mostramos sólo el test.
- **¿Por qué el Explorá antes daba +4,6%?** El modelo final había entrenado con esos partidos. Ahora sólo se muestran predicciones fuera de muestra.
- **¿La app calcula el modelo en vivo?** Sí: es una fórmula, la misma del notebook, y corre en el navegador.
- **¿Se podría usar para partidos de hoy?** No con esta base, que termina en 2016. Haría falta una fuente actual de cuotas y alineaciones.

## 10. Glosario rápido

| Término | Qué es |
|---|---|
| Underdog | El equipo no favorito |
| Cuota | Lo que devuelve la casa por cada peso apostado |
| Probabilidad implícita | 1 / cuota |
| Margen de la casa | Lo que las probabilidades suman por encima de 100% |
| Tasa base | Proporción de partidos que gana el underdog (20%) |
| AUC | Qué tan bien ordena el modelo los partidos (0,5 = azar) |
| Log-loss / Brier | Cuánto se equivocan las probabilidades (menos es mejor) |
| Calibración | Que lo que dice el modelo coincida con lo que pasa |
| Valor esperado | Lo que se espera ganar por peso apostado en el largo plazo |
| Validación temporal | Entrenar con el pasado y probar con lo que viene |
| Test | La temporada 2015/16, que el modelo nunca vio |
| Dummy | Estrategia simple que no usa ningún modelo, para comparar |
| Bronze / Silver | Capas del pipeline: crudo / listo para modelar |
| DAG | El flujo de tareas de Airflow que arma los datos |
