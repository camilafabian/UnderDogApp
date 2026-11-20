# Guion de la exposición final

**¿Gana el underdog?** · Proyecto integrador · Ciencia de Datos · UTN FRM · 5K10

**15 minutos · 6 integrantes · 2 min 30 s cada uno**

| # | Parte de la página | Tiempo |
|---|---|---|
| 1 | Portada y 0 · El problema | 0:00 – 2:30 |
| 2 | 1 · Los datos y 2 · Lo que encontramos | 2:30 – 5:00 |
| 3 | 3 · ¿El modelo predice? (etapa 1) | 5:00 – 7:30 |
| 4 | 4 · ¿Sirve para apostar? La regla, con y sin | 7:30 – 10:00 |
| 5 | 4 · ¿Sirve para apostar? Las estrategias y el azar | 10:00 – 12:30 |
| 6 | El recorrido, Explorá, Probá el modelo y cierre | 12:30 – 15:00 |

**Reglas para todos**

- Leer lo que está en pantalla, en registro formal: «el modelo», «la cuota», «la casa de apuestas».
- Señalar el gráfico mientras se habla.
- No usar el simulador en vivo y no abrir los «Cómo lo sabemos»: si el profe pregunta, la respuesta está ahí.
- Cada uno termina con su frase de pase, para que se note el hilo.

<div class="salto"></div>

## Integrante 1 · Portada y El problema

<p class="meta">0:00 – 2:30 · Secciones: portada y 0 · El problema</p>

> **En pantalla:** la portada.

Buenas tardes. El 17 de abril de 2016, el Barcelona perdió 1 a 2 en su cancha contra el Valencia, al que las casas de apuestas le
daban un 6% de chances. Nuestro proyecto se pregunta si una sorpresa así se puede ver venir antes del partido.

Lo vamos a responder con dos preguntas: primero, si se puede predecir quién gana; después, si se le puede ganar a la casa de apuestas.

En cada partido hay un favorito, el equipo al que las casas le dan más chances, y un underdog, el no favorito. Usamos 17.959 partidos
de 9 ligas europeas, entre 2008 y 2016, y predecimos si gana el underdog, sí o no, con las cuotas y el rating FIFA de los once titulares.

> **En pantalla:** 0 · El problema, las tres cuotas del partido.

Un partido tiene tres resultados: gana el local, empatan o gana el visitante. Antes del partido, las casas de apuestas publican una
cuota para cada uno: cuánto devuelven por cada peso apostado.

Tomemos un partido real: Leverkusen contra Frankfurt, abril de 2016. La cuota de Frankfurt era 8,56: si apostábamos $100 y ganaba,
nos devolvían $856. Si dividimos 1 por la cuota, obtenemos una probabilidad: 11,7%. Leverkusen, 70,9%. El empate, 19,3%.

> **En pantalla:** la barra de probabilidades.

Las tres suman 101,9%, no 100%. Ese 1,9% que sobra es el margen de la casa: paga un poco menos de lo que vale cada resultado.
Por eso apostar a ciegas pierde en el largo plazo.

> **En pantalla:** las cuatro tarjetas de definiciones.

Quitando el margen, el favorito es el equipo con más probabilidad de ganar. Sólo usamos partidos donde favorito y underdog se separan
por más de 5 puntos, porque por debajo dos casas de apuestas pueden elegir distinto favorito. Una fila es un partido. El underdog ganó
3.654 de los 17.959: el 20%. Esa es la tasa base, el piso que cualquier modelo tiene que superar.

<p class="pase">Pase: «Antes de construir un modelo, miramos qué dicen los datos.»</p>

<div class="salto"></div>

## Integrante 2 · Los datos y Lo que encontramos

<p class="meta">2:30 – 5:00 · Secciones: 1 · Los datos y 2 · Lo que encontramos</p>

> **En pantalla:** 1 · Los datos, el gráfico de barras.

En este gráfico agrupamos los partidos según qué tan favorito era el favorito para la cuota. La altura de cada barra es cuántos de
cada 100 partidos ganó el underdog.

Cuando el partido es parejo, el underdog gana 3 de cada 10. Cuando el favorito es abrumador, 3 de cada 100. La línea punteada es el
promedio, 20 de cada 100, y ese promedio esconde partidos muy distintos. La cuota ya acierta bastante: cuanto más favorito dice que es
un equipo, menos gana el underdog.

> **En pantalla:** 2 · Lo que encontramos, las dos líneas.

Nuestra hipótesis era que los jugadores podían agregar información. Comparamos el rating FIFA medio de los once titulares del underdog
contra el del favorito, en cinco grupos del mismo tamaño.

La línea azul es lo que pasó: cuando el once del underdog es mejor, sus victorias pasan de 11% a 27%, más que se duplican. Pero la línea
naranja punteada es lo que esperaba la cuota en esos mismos partidos, y va pegada a la azul. Las casas de apuestas ya ponían en el
precio la calidad de los titulares.

En el análisis exploratorio lo confirmamos: al controlar por la cuota, el efecto del once casi desaparece. Esa hipótesis quedó refutada,
y nos anticipa lo que va a pasar con el modelo.

<p class="pase">Pase: «Con esto construimos el modelo, y lo medimos en dos etapas. La primera, sin hablar de plata.»</p>

<div class="salto"></div>

## Integrante 3 · ¿El modelo predice?

<p class="meta">5:00 – 7:30 · Sección: 3 · ¿El modelo predice? (etapa 1)</p>

> **En pantalla:** el título y la tabla de métricas.

El modelo es una regresión logística: una fórmula que toma la probabilidad de la cuota y diez diferencias de rating entre los titulares,
y devuelve la probabilidad de que gane el underdog. Lo entrenamos con 2008 a 2015 y lo probamos una sola vez en la temporada 2015/16:
2.399 partidos que nunca vio.

Lo comparamos con dos pisos: decir siempre 20%, que no mira nada, y la cuota sola. El AUC mide si el modelo le da más chance a los
partidos que el underdog terminó ganando: 0,5 es tirar una moneda. El modelo da 0,660 y la cuota, 0,656. En log-loss y Brier, que miden
cuánto se equivocan las probabilidades, también quedan empatados, y los dos muy por encima del piso.

> **En pantalla:** el gráfico de calibración.

Este gráfico responde cuánto se equivoca. Partimos los partidos en diez grupos según la probabilidad que dio cada uno, y la comparamos
con lo que pasó. Sobre la diagonal, lo que dice es lo que pasa. El modelo y la cuota siguen la diagonal, casi siempre juntos.

> **En pantalla:** las dos matrices de confusión.

Si el modelo tuviera que decir sí o no, cortamos en 20%, la tasa base. De los 530 partidos que ganó el underdog, el modelo marcó 393, a
cambio de 941 falsas alarmas. La cuota, con el mismo corte, hace casi lo mismo. No cortamos en 50% porque el modelo no diría «gana» en
ningún partido: acertaría el 78% sin servir para nada.

> **En pantalla:** el recuadro oscuro «Etapa 1, en una línea».

La conclusión de la etapa 1: el modelo predice, pero no mejor que la cuota.

<p class="pase">Pase: «Entonces, ¿esa pequeña diferencia sirve para algo? Esa es la etapa 2.»</p>

<div class="salto"></div>

## Integrante 4 · La regla de apuesta

<p class="meta">7:30 – 10:00 · Sección: 4 · ¿Sirve para apostar? (primera mitad)</p>

> **En pantalla:** el título y la fórmula de la regla.

Predecir igual que la cuota no alcanza para ganarle a la casa. Lo que cambia el resultado es una segunda etapa, aparte del modelo: una
regla para decidir cuándo apostar. Se apuestan $100 al underdog sólo cuando la probabilidad del modelo por la cuota, menos 1, es mayor
que cero. Es decir, cuando el modelo cree que el underdog tiene más chances de las que paga la cuota.

> **En pantalla:** el recuadro con el ejemplo de Frankfurt.

Volvamos a Frankfurt. La cuota 8,56 paga como si tuviera 11,7% de chances. El modelo le daba 14,8%. Entonces 0,148 por 8,56, menos 1,
da más 26,5%: por cada $100, el modelo espera ganar unos $27 en el largo plazo, y apuesta. Leverkusen ganó 3 a 0 y la apuesta se perdió.
Valor esperado positivo no es ganar cada vez: es cobrar, en el largo plazo, más de lo que vale el riesgo.

> **En pantalla:** el gráfico de puntos con la línea «lo que paga la cuota».

Acá está la regla para todos los partidos del test. En horizontal, lo que paga la cuota; en vertical, lo que cree el modelo. Arriba de la
línea punteada, el modelo apuesta: 404 de 2.399 partidos. Casi todos los puntos quedan abajo, y eso es el margen de la casa.

> **En pantalla:** el gráfico de barras «con y sin la regla».

La regla funciona como una calibración posterior al modelo. Para medir cuánto aporta, probamos el mismo modelo sin la regla: apostar cada
vez que su probabilidad supera un umbral fijo, 15%, el mejor en validación. Sin la regla, en validación pierde 5,8%, casi lo mismo que
apostar a ciegas, que pierde 7,7%. Con la regla, gana 0,8%. En el test, una temporada buena para los underdogs, a casi todo le fue mejor.
Por eso la diferencia se mide en la validación: cinco temporadas contra una.

<p class="pase">Pase: «Ahora, ¿le gana a estrategias simples, que no necesitan ningún modelo?»</p>

<div class="salto"></div>

## Integrante 5 · Las estrategias y el azar

<p class="meta">10:00 – 12:30 · Sección: 4 · ¿Sirve para apostar? (segunda mitad)</p>

> **En pantalla:** el subtítulo «¿Le gana a estrategias simples?».

Comparamos el modelo con reglas simples. Cada una eligió sus parámetros mirando sólo la validación, igual que el modelo: apostar a todos,
apostar cuando la cuota le da al underdog entre 10% y 20%, apostar cuando su once es mejor, no apostar nunca y apostar al azar.

> **En pantalla:** el gráfico de ganancia acumulada.

Este gráfico muestra sólo la temporada de test, empezando en cero. El modelo con la regla, la línea verde gruesa, termina en más $2.672.
Apostar a todos termina en menos $2.250, y el rango de cuota, en menos $3.627. La banda gris es lo que da apostar al azar la misma cantidad
de veces: una línea adentro de la banda no se distingue de la suerte.

> **En pantalla:** la tabla y los tres puntos de lectura.

Hay que leerlo con honestidad. Primero, el modelo le gana a apostar a todos y al rango de cuota. Segundo, «once mejor» y el modelo sin la
regla dan algo más en el test, pero los dos perdían en validación: sin mirar el test, no los habríamos elegido. Tercero, el 6,6% todavía
no se distingue del azar: el 5% de los sorteos supera 15,1%, y el intervalo de confianza incluye el cero. Con 404 apuestas, la suerte pesa mucho.

Lo que sí se sostiene es que, de las estrategias que apuestan, es la única que no pierde plata ni en validación ni en el test.

> **En pantalla:** el recuadro oscuro «Etapa 2, en una línea».

La conclusión de la etapa 2: no le ganamos a las casas de apuestas, pero dejamos de perder contra ellas.

<p class="pase">Pase: «Llegar a esta conclusión nos llevó cuatro entregas.»</p>

<div class="salto"></div>

## Integrante 6 · El recorrido, la app y el cierre

<p class="meta">12:30 – 15:00 · Secciones: El recorrido, 5 · Explorá y 6 · Probá el modelo</p>

> **En pantalla:** El recorrido, la línea de tiempo.

En cada entrega concluimos algo distinto. En la primera pensábamos que los jugadores podían anticipar al underdog, y armamos el dataset.
En la segunda descubrimos que la cuota ya sabía lo de los jugadores. En la tercera elegimos el modelo que mejor predecía, y nos marcaron
que era la cuota con otro nombre: nunca tomaba una decisión. Al corregirlo, sumamos la pregunta económica. En la cuarta corregimos un número
demasiado optimista, porque el modelo había visto esos partidos. Y para esta exposición separamos predecir de apostar, y comparamos contra
estrategias simples y el azar. Lo que no cambió desde la segunda entrega: la cuota ya sabe casi todo.

> **En pantalla:** 5 · Explorá y 6 · Probá el modelo, sin interactuar.

La página tiene dos secciones para el usuario. En Explorá están los 2.399 partidos del test: se puede filtrar por liga o seleccionar una
zona, y las barras recalculan cuánto habría ganado cada estrategia. En Probá el modelo se carga un partido, con sus cuotas y sus titulares,
y la página calcula la probabilidad y si conviene apostar. Como el modelo es una fórmula, corre en el navegador.

> **En pantalla:** volver a la portada.

En resumen: el modelo predice tan bien como la cuota, no mejor. Pero usado con una regla de apuesta, es la única estrategia que apuesta y
no pierde plata en ningún período, aunque con estos datos su ganancia todavía podría ser suerte. Para saberlo harían falta más temporadas,
o cuotas de cierre, que son más difíciles de superar. Muchas gracias.

<div class="salto"></div>

## Preguntas probables y quién las responde

| Pregunta | Respuesta corta | Quién |
|---|---|---|
| ¿Por qué ese rango de cuota, o ese umbral de 15%? | Se eligieron mirando sólo la validación, entre opciones fijadas de antemano, igual que el modelo. Recién después se miró el test. | 4 o 5 |
| ¿Por qué no accuracy? | Decir siempre «no gana» acierta el 80% sin mirar nada. | 3 |
| ¿Por qué la matriz en 20% y no en 50%? | Con 50% el modelo no marca ningún partido. | 3 |
| ¿Hay fuga de información? | Los goles y resultados sólo se usan para construir el objetivo y quedan fuera del dataset de modelado; el pipeline falla si aparecen. | 2 |
| ¿Por qué el test anda mejor que la validación? | Fue una temporada buena para los underdogs: ganaron el 22,1%, contra 20% de la tasa base. | 4 |
| ¿Por qué 5 puntos para definir al underdog? | Por debajo, Pinnacle y Bet365 eligen distinto favorito en el 54% de los partidos casi parejos. | 1 |
| ¿Entonces el modelo gana plata? | No está demostrado: es la única estrategia que no pierde en ningún período, pero el +6,6% no se distingue del azar. | 5 |

## Ensayo

- Un ensayo completo cronometrado, con la página en pantalla. Meta: 14 minutos, para tener un minuto de margen.
- Si alguien se pasa de 2:45, recortar frases del guion, no números.
