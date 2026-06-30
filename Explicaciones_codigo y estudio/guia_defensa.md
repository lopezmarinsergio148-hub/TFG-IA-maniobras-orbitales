# Guía de defensa del TFG — IA para maniobras orbitales

> Documento de **estudio para la defensa**. Para cada tema:
> **QUÉ es** · **POR QUÉ se hizo así** (lo defendible) · **posibles preguntas** del tribunal.
> Documento vivo: se va ampliando según avanzamos. Lo "defendible" es lo que el tribunal
> puede preguntar; la "fontanería" de librerías (numpy, gymnasium, PPO interno) NO hace
> falta defenderla.

---

## 1. Fundamentos teóricos

### Energía orbital específica — ε = −μ/(2a)
- **Qué:** energía total (cinética + potencial) por unidad de masa; solo depende del semieje a.
- **Por qué es negativa:** el "cero" de energía se toma en el infinito; una órbita ligada
  está atrapada en el pozo gravitatorio → energía negativa. Etiqueta el tipo de órbita:
  **ε<0 elipse, ε=0 parábola, ε>0 hipérbola**.
- **Si a disminuye, ε disminuye** (más negativa): órbita más pequeña = menos energía. El
  drag roba energía → ε baja → a baja → la órbita decae.
- **Posible pregunta:** "¿Por qué el satélite va más rápido al decaer pero tiene menos
  energía?" → la energía potencial cae más de lo que sube la cinética (paradoja del satélite).

### Ecuación de la vis-viva — v² = μ(2/r − 1/a)
- **Qué:** da la velocidad en cualquier punto de la órbita.
- **El nombre:** "fuerza viva" (Leibniz) = el término mv² (cinético); la ecuación no es más
  que la **conservación de la energía** de la órbita.

### C3 (energía característica)
- **Qué:** C3 = v∞² (velocidad de exceso hiperbólico al cuadrado); energía por unidad de
  masa que "sobra" tras escapar del planeta de salida.
- **Por qué se usa:** depende solo de la trayectoria, no del cohete → medida estándar para
  comparar lanzamientos. C3=0 (parábola, escape justo); C3>0 (hipérbola).

### Elementos orbitales / RAAN
- 6 elementos: **a, e** (tamaño/forma); **i, Ω** (orientación del plano); **ω** (orientación
  de la elipse); **ν** (posición sobre la órbita, el único que cambia en Kepler).
- **RAAN Ω:** ángulo, medido en el plano de referencia desde el punto Aries, que localiza
  el nodo ascendente (donde la órbita cruza el plano de sur a norte).

### Cónicas parcheadas / esfera de influencia (SOI)
- **Qué:** el viaje interplanetario se trocea en tramos dominados por un solo cuerpo
  (hipérbola de escape → arco heliocéntrico → captura), cosidos en la SOI.
- **Por qué importa:** justifica calcular el Δv interplanetario como (arco de Lambert) −
  (velocidad del planeta), y es donde vive el C3.

### Maniobras (Hohmann / Lambert / porkchop)
- **Hohmann:** mínimo Δv entre dos órbitas circulares coplanarias; 2 impulsos tangenciales.
- **Lambert:** problema de **contorno** (fija r1, r2 y tiempo de vuelo → busca la órbita);
  devuelve v1, v2 → Δv. Tipo I (<180°) y tipo II (>180°), separados por la singularidad de
  los 180° (la franja blanca del porkchop).
- **Porkchop:** mapa de Δv/C3 según fechas de salida/llegada (muchos Lambert encadenados).

---

## 2. Física — Atmósferas y validación

### Modelo atmosférico por capas
- **Qué:** ρ(h) = ρ_base · exp(−(h−h_min)/H) por tramos; 9 cuerpos, 48 capas, 41 fronteras.
- **Decisión clave (defendible):** las alturas de escala H **no se ponen a mano**; se
  **derivan de la continuidad** entre capas. Las ρ_base (observadas) son el input.
- **Auto-comprobación:** si la H derivada se aleja de la física RT/μg → señal de densidad
  errónea (así se cazó el error de Neptuno).

### Validación (factor de desviación)
- Júpiter **1,32×** (Galileo, in-situ), Saturno **1,06×** (Cassini, in-situ), Urano **1,15×**,
  Neptuno **1,2×** (Voyager reconstruido). Todo dentro de factor 2.
- **Saturno — dispersión RMS ~2×:** NO es error del modelo; es la **variabilidad latitudinal
  real** (Cassini midió de −85° a +80°). El modelo da la **mediana** sobre latitudes, que es
  lo apropiado para el drag orbital integrado (el satélite recorre muchas latitudes).
- **RMS:** raíz cuadrática media; pondera más las desviaciones grandes que la mediana.

### Reconstrucción hidrostática (Urano/Neptuno) — APORTACIÓN
- **Por qué hace falta:** no hay datos in-situ; Voyager 2 solo sobrevoló (no entró). Los
  datos vienen de **radio-ocultación**: la señal de radio atraviesa la atmósfera al ocultarse
  la sonda → da perfiles de **T y P**, solo donde el gas es denso (troposfera/estratosfera).
- **Método (defendible):** de ~9 puntos T-P de Lindal (las **anclas**) → perfil de densidad
  completo por **integración hidrostática** (= el mismo método de los GRAM de la NASA).
  Validado de forma cruzada contra los GRAM (<2%).
- **Fiabilidad (honestidad):** alta de 0 a ~150 km (datos + anclas); datos hasta 262 km
  (Urano) / 300 km (Neptuno); por encima el modelo extrapola → capas altas = las menos fiables.

### Error de Neptuno — APORTACIÓN
- La validación detectó ρ a 50 km ~100× baja → factor típico **8,0× → 1,2×** tras recalibrar.
  Demuestra que el método de validación **funciona** (caza errores que invalidarían todo).

### Lección de las IAs generalistas — APORTACIÓN (honestidad)
- Datos de densidad de Júpiter de una IA generalista erraban **×2500** vs Galileo → regla del
  proyecto: solo fuentes primarias trazables.

---

## 3. Física — Propagador orbital

### J2 (achatamiento planetario)
- Primer término zonal (y dominante) del potencial; el abultamiento ecuatorial rompe la
  simetría esférica → la órbita **precesa**. Efectos **periódicos** (oscilan, media nula) y
  **seculares** (regresión nodal Ω, precesión del perigeo ω).

### Método de Cowell
- Integrar numéricamente la ecuación completa (gravedad central + J2 + drag) en vez de la
  solución analítica de Kepler. El drag usa la atmósfera **rotante** (v_rel = v − ω×r).

### Validación vs Brouwer + escalado del error
- Regresión nodal simulada vs fórmula de Brouwer (1er orden): **<1%** en J2 moderado.
- **El error crece ∝ J2(R/a)²** (de ~0,5% en Tierra a ~5% en Júpiter/Saturno).
- **Por qué (defendible):** Brouwer es el primer término de una serie; el error relativo de
  truncar en 1er orden ≈ el parámetro de expansión J2(R/a)². La que se desvía es la **fórmula
  analítica**, no la simulación (que integra la fuerza completa).

### Acoplamiento drag–J2 — APORTACIÓN
- La simulación reproduce la desviación de **0,8°** (caso ISS) sobre Brouwer: el drag baja a,
  sube el movimiento medio n y acelera la precesión. Brouwer (sin drag) no lo captura →
  muestra de fidelidad del propagador.

### Detección de reentrada + suite de tests
- Flag automático de reentrada + criterio de **fiabilidad de la pendiente** (ratio
  decaimiento/oscilación J2).
- **Suite de verificación: 54 checks (6 por cuerpo × 9), todos superados.**
- **Limitación honesta:** en Júpiter/Saturno (J2 alto), las órbitas muy bajas son inviables
  (la oscilación osculadora hunde el perigeo) → no es fallo, es física.

---

## 4. IA — Aprendizaje por Refuerzo (Bloque 4)

### ¿Qué es el RL?
- Aprender por **prueba y error con premios y castigos** (como adiestrar con chuches). El
  agente es un "piloto" que aprende a hacer la maniobra gastando el mínimo Δv.

### Las 5 piezas (qué es cada una, y qué es en NUESTRO caso)
Imagina al agente como un piloto en un simulador:
- **Estado** — lo que el agente *ve* antes de decidir. En nuestro entorno: las dos
  órbitas (actual y objetivo) como radios normalizados `[r1/R_REF, r2/R_REF]`. Su "foto".
- **Acción** — lo que *decide hacer*. En nuestro caso: los dos impulsos tangenciales
  `[Δv1, Δv2]` (dos números continuos). Su "decisión".
- **Recompensa** — la *nota* que recibe tras actuar: `−error_de_llegada − 0,01·Δv_total`.
  Cuanto más cerca de acabar circular en GEO gastando poco, mayor nota. Es lo que el
  agente intenta maximizar.
- **Episodio** — un *intento completo*: sale de LEO, da los dos impulsos y termina (1
  paso). El agente repite cientos de miles de episodios para aprender.
- **Política** — la *estrategia aprendida*: la red neuronal que, dado el estado, elige la
  acción. Es lo que **PPO** va afinando.

### Decisiones de diseño (DEFENDIBLES)
- **Estado = elementos limpios (a / energía), NO la altitud instantánea** → la altitud oscila
  por J2 y confundiría al agente.
- **Recompensa = −Δv − castigo por no llegar (+ bonus)** → su máximo coincide, por física,
  con la solución de Hohmann (gastar poco Y acertar).
- **Acción = impulsos tangenciales** → hace el aprendizaje tratable; Hohmann son justamente
  2 impulsos tangenciales.
- **PPO (stable-baselines3)** → estándar, robusto, bueno para **control continuo** (los Δv
  son continuos).
- **Criterio de éxito: Δv ≤ 5% del óptimo** (≤ 4,05 km/s) → margen razonable para afirmar
  "aprendió el óptimo".
- **Empezar por Hohmann LEO→GEO (Kepler)** → tiene **óptimo analítico** = juez objetivo;
  prueba de concepto antes de la Fase 2 (donde el RL sí es imprescindible).

### Diseño del entorno: estructura y parámetros
Los cuatro scripts de la Fase 1 (carpeta `ia/`):
- **`baselines.py` (el JUEZ):** constantes (μ, R, altitudes) → funciones (velocidad
  circular, fórmula de Hohmann) → `main` que imprime el óptimo (LEO→GEO = 3,854 km/s).
- **`env_hohmann.py` (el MUNDO):** parámetros del entorno → la clase `HohmannEnv` con los
  **3 métodos de Gymnasium**: `__init__` (define el estado y la acción), `reset` (empieza
  un episodio, fija/sortea r1 y r2) y `step` (aplica la acción, simula la maniobra y
  calcula la recompensa) → `main` (la prueba a mano).
- **`train_hohmann.py` (ENTRENAR):** crea el entorno → crea el modelo PPO → configura el
  guardado del mejor modelo (`EvalCallback`) → entrena. *Defendible:* `ent_coef`
  (exploración) y `EvalCallback` (guardar el MEJOR modelo, no el último); el resto es uso
  estándar de la librería (fontanería).
- **`evaluar_hohmann.py` (EXAMINAR):** carga el mejor modelo → pone el entorno en LEO→GEO →
  pide la acción al agente (`predict`) → la ejecuta y la compara con el óptimo → dice si
  cumple el criterio (≤5%).

**Parámetros de diseño** (el valor exacto es un "dial" que se ajusta entrenando; lo
defendible es el PORQUÉ, no el número):
- **`DV_MAX = 3` km/s** — techo de cada impulso (rango de la acción `[0, 3]`). *Por qué:*
  los impulsos óptimos son ~2,4 y ~1,5 km/s → 3 deja margen sin ser absurdo (un techo
  enorme haría perder tiempo explorando valores ridículos).
- **Recompensa SUAVE: `−error − C_DV·Δv`, con `W_ERROR = 1` y `C_DV = 0,01`** — el peso
  del coste en Δv es **minúsculo a propósito**. *Por qué:* así **LLEGAR domina sobre
  ahorrar**; si el coste pesara mucho, el paisaje se aplana y el agente aprende a no hacer
  nada (la "esquina muerta" de las lecciones de abajo).
- **Sin premio escalón (sin bonus)** — *Por qué:* un salto de recompensa al bajar de cierto
  error desestabilizaba el aprendizaje (ver lecciones); la recompensa final es continua.
- **`TOL_EXITO = 2%`** — tolerancia para dar por buena la "llegada circular" al objetivo.

### Arquitectura híbrida (la idea fuerte del proyecto)
- **RL** solo donde aporta (perturbaciones, atmósfera incierta: aerofrenado, desorbitado).
- **Solvers clásicos** donde hay óptimo conocido (interplanetario: Lambert/porkchop).
- **Capa LLM** (Claude API) que **orquesta** (elige la herramienta según el objetivo en
  lenguaje natural) y **explica**.
- **Por qué:** forzar RL en lo interplanetario (ya resuelto) sería un error metodológico.

### Posibles preguntas del tribunal
- *"¿Por qué usar RL para Hohmann si ya hay fórmula?"* → es **prueba de concepto** sobre un
  caso con solución conocida; valida que el sistema de RL funciona antes del plato fuerte
  (aerofrenado, donde no hay fórmula).
- *"¿Por qué la altitud no es el estado?"* → oscila por J2 y confunde al agente.
- *"¿Esto es RL de verdad o una optimización?"* → la Fase 1 es un **bandit contextual** (RL
  de horizonte 1, honesto); la Fase 2 será RL secuencial completo.
- *"¿Por qué PPO y no otro algoritmo (DQN, SAC, DDPG…)?"* → **DQN** queda descartado: es para
  acciones **discretas** y aquí los Δv son **continuos**. **SAC/DDPG/TD3** sí son de control
  continuo, pero son *off-policy* y más **sensibles al ajuste** (pueden divergir). **PPO** es
  *on-policy*, **robusto y estable**, el estándar de facto y va bien "de fábrica". Como
  nuestro entorno es **barato** (física analítica, no integración cara por paso), la mayor
  eficiencia de muestras de SAC **no compensa** su fragilidad → PPO es la elección sensata.
- *"¿Cómo evitas que el LLM alucine?"* → anclado al código y a los datos del proyecto; toda
  cifra sale de cálculos sobre el modelo real (coherente con la regla de no fiarse de IAs
  generalistas).

### Resultados de la Fase 1 y lecciones del entrenamiento
**Resultado:** el agente PPO aprende la transferencia LEO→GEO con un **Δv a 0,03% del
óptimo** de Hohmann (3,8528 vs 3,854 km/s) y llega a GEO circular con 0,08% de error
(criterio: ≤5%). Es decir, **sin conocer la fórmula, descubre la maniobra óptima**.

**El camino (muy defendible: demuestra dominio del diseño de recompensa):**
- *Reward hacking:* con la primera recompensa, el agente se **quedaba corto** (no llegaba
  a GEO) para ahorrar combustible → la recompensa premiaba mal.
- *Esquina muerta:* al penalizar demasiado el Δv, la recompensa se **aplanó** y el agente
  aprendió a **no hacer nada** → el coste del Δv debe pesar MUCHO menos que llegar.
- *Inestabilidad:* entrenar de más **descuadró** la política (PPO fluctúa en un óptimo
  fino) → solución: **guardar el mejor modelo** (`EvalCallback`), no el último.
- *Recompensa suave:* quitar el "premio escalón" (un salto al bajar del 2% de error)
  estabilizó el aprendizaje.

**Posible pregunta:** *"¿Por qué tanto ajuste de la recompensa?"* → porque el RL aprende
lo que se le premia, no lo que uno quiere; diseñar bien la recompensa (que el óptimo sea
el máximo y el paisaje tenga pendiente clara) **es** el trabajo.

### Generalización (Agente 2): cualquier transferencia, en cualquier planeta — ADIMENSIONALIZAR
**Qué es:** la Fase 1 clava UN caso (LEO→GEO). El Agente 2 lo generaliza: **un único
agente** que resuelve **cualquier** transferencia coplanaria (subir Y bajar) en
**cualquier** cuerpo (los 9), sin reentrenar.

**La clave — adimensionalizar (lo más vendible; es análisis dimensional puro):**
- La transferencia de Hohmann **no tiene escala propia**: si divides los impulsos entre la
  velocidad circular inicial `v_c1 = √(μ/r1)`, los `μ` y `r1` **se cancelan** y todo
  depende SOLO del ratio `R = r2/r1`.
- Por eso el estado del agente es **un solo número** (`R`, en escala logarítmica para
  tratar igual subir `R>1` y bajar `R<1`) y la acción son los impulsos en unidades de
  `v_c1`.
- **Frase potente:** *"el agente no aprende velocidades, aprende la FORMA de la maniobra"*
  (qué fracción de tu velocidad orbital gastar) — como un piloto que aprende "acelera al
  140% del crucero", no "ve a 7,8 km/s".
- **Consecuencia:** entrenado una sola vez en unidades adimensionales, **clava Marte,
  Júpiter, la Luna o Mercurio SIN reentrenar** — solo se multiplica su respuesta por la
  `v_c1` real del planeta.

**Resultado:** exceso sobre el óptimo de Hohmann **< 1%** en el rango útil (ratios de 0,2
a 11), y la **invariancia de escala** demostrada: los 9 cuerpos caen en el MISMO punto
para cada `R`.

**Lecciones del entrenamiento (defendibles):**
- **Error logarítmico, no lineal:** el error relativo lineal se disparaba en las bajadas
  profundas (~9) y reventaba la red de valor (el crítico) → no aprendía. El error en
  **escala logarítmica** es simétrico (subir/bajar) y acotado → lo arregló.
- **Límite físico en la acción:** topar el primer impulso por debajo del umbral de escape
  elimina un "castigo-acantilado" que saturaba al agente en un extremo.
- **Currículum:** ampliar el rango de `R` poco a poco (primero maniobras suaves) para que
  converja en un espacio de maniobras tan amplio.
- **Trade-off irreducible (honestidad):** clavar a la vez los dos extremos (subir y bajar
  1:12) no se pudo; reforzar uno degradaba el otro → se eligió el modelo equilibrado.

**Posibles preguntas del tribunal:**
- *"¿Por qué un solo agente vale para todos los planetas?"* → porque la gravedad no tiene
  escala propia (invariancia de escala): el problema solo depende del ratio de radios.
- *"¿Por qué el aerofrenado NO es universal y este sí?"* → porque las atmósferas SÍ tienen
  escala propia (densidad y altura de escala distintas) → el aerofrenado va con
  especialistas por planeta (sección 5).
- *"Si el Agente 2 ya hace LEO→GEO, ¿para qué el Agente 1? ¿No sobra?"* → no sobran, van en
  **escalada didáctica**: el **Agente 1** demuestra **precisión máxima** clavando UN caso
  fijo (0,03% del óptimo) → prueba que la tubería de RL puede igualar el óptimo analítico.
  El **Agente 2** demuestra **generalización** (cualquier órbita, cualquier planeta) a costa
  de algo de precisión (<1–2%). Son dos cosas distintas que se quieren enseñar: que el método
  *puede ser exacto* y que el método *puede generalizar*. El 1 es la prueba de concepto; el 2,
  el salto conceptual (adimensionalización).

### Extensión 3D (Agente 4): transferencias con CAMBIO DE PLANO
**Qué es:** extiende el Agente 2 a TRES dimensiones — ahora la órbita destino, además
de otro radio, está **inclinada** un ángulo `Δi`. El agente hace la transferencia de
Hohmann Y gira el plano, repartiendo el giro entre los dos impulsos.

**Física clave (defendible):**
- Girar el plano cuesta **`Δv = 2·v·sin(Δi/2)`** → es **más barato donde se va más lento**
  (en el apogeo). Por eso el óptimo mete **casi todo el giro en el 2.º impulso**.
- Cada impulso es un **vector**: parte tangencial (cambia la órbita, la Hohmann de siempre)
  + parte fuera del plano (gira) → se suman por la **ley del coseno** (de ahí el coste
  combinado). El juez (`baselines.delta_v_hohmann_plano`) halla el reparto óptimo.
- *Número emblemático:* LEO→GEO con 28,5° → el óptimo reparte **~2° abajo + ~26° arriba**
  (justo lo que hacen las misiones GTO→GEO reales).

**Diseño:** acción **vectorial 4D** (el agente da las 2 componentes de cada impulso; NO se le
dice cómo repartir el giro, lo **descubre solo**). Adimensional como el Agente 2 (estado
`[log R, Δi]`) → **invariancia de escala 3D**. Alcance: solo **subir** con cambio de plano
(caso real GTO→GEO; bajar + girar es raro y carísimo).

**Resultado:** exceso **< 2%** sobre el óptimo en todo el rango (R hasta **11,94** —el mismo
límite que el Agente 2— y Δi hasta 40°), llegando circular y a la inclinación pedida (<0,3°);
**Marte y Júpiter clavados sin reentrenar** (<1,2%). [Una versión anterior, con menos rango y
currículum más corto, se quedaba en ~4%; ver lección 4.]

**Lecciones del tuning (ORO — tres encadenadas):**
1. **El papel del coste cambia de 2D a 3D.** En 2D `C_DV` podía ser casi cero: llegar con 2
   impulsos tangenciales **fija** la solución de Hohmann. En 3D hay **grados de libertad de
   sobra** (muchas maniobras llegan a la misma órbita inclinada con costes distintos) → el
   coste **debe pesar** para que el agente elija la barata (si no, despilfarra: salió +27%).
2. **El colapso por escape es recurrente — y la lección del Agente 2 se transfiere.** Al subir
   el coste, la política colapsó saturando un impulso hasta **escapar**. Solución: **topar los
   impulsos por debajo del umbral de escape** (lo mismo que el Agente 2). Detalle revelador:
   al topar el 1.er impulso, el colapso **se mudó al 2.º** → hubo que cerrar las dos puertas.
3. **Tuning gradual, no a saltos.** Subir los pesos de golpe (×6) y el coste ×25 **desestabilizó**
   un entrenamiento que era estable → la forma correcta es subir el coste **poco a poco** hasta
   el punto dulce (quedó en `C_DV=0,05`, 5× el del 2D).
4. **Más entrenamiento gana a restringir a la fuerza (la lección más bonita).** Para bajar el
   exceso de ~4% a ~2% se probó primero **estrechar** los rangos de impulso (quitar espacio al
   agente, la receta del Agente 2). **Fracasó**: el agente sacrificaba la llegada (no completaba
   el giro en los `Δi` grandes, error de inclinación 8-15°). Lo que SÍ funcionó fue lo contrario:
   **ampliar** el problema (rango de R de 8 a 11,94) y **alargar el currículum** (etapa extra,
   ~1,3 M pasos) → bajó el exceso a **<2% Y** amplió el alcance, todo a la vez. *Lección: cuando
   el agente despilfarra, a veces la cura no es recortarle opciones sino darle más rango y más
   entrenamiento bien escalonado.* Modelo previo (R≤8, ~4%) respaldado en
   `modelo_transfer3d/best_model_backup_R8.zip`.

**Posible pregunta:** *"¿Por qué el agente 3D solo sube?"* → porque combinar bajada **y** cambio
de plano es una maniobra rara y muy cara; centrarlo en subir (inyección a GEO/órbitas altas
inclinadas) es el caso de uso real y acota el problema.

---

## 5. IA — Fase 2: Aerofrenado (aerobraking) multi-planeta

### ¿Qué es y por qué RL aquí?
- **Aerofrenado:** bajar el apogeo rozando la atmósfera en cada paso por el perigeo,
  casi sin combustible (técnica real: MGS, MRO, MAVEN...).
- **RL irreemplazable:** no hay fórmula (atmósfera incierta, control en lazo cerrado) y
  **usa tu atmósfera de Marte validada**. A diferencia de la Fase 1 (1 paso), es **RL
  secuencial** (decenas de pasadas encadenadas).

### El dilema y el diseño
- Perigeo bajo → frena rápido pero **riesgo de destrucción**; alto → seguro pero lento.
- **Estado:** [apogeo, perigeo, apogeo objetivo]. **Acción:** perigeo de la próxima
  pasada. **Recompensa:** progreso (apogeo que baja) − tiempo (por pasada) − fuel;
  +éxito; −destrucción.
- **Física por pasada (King-Hele):** Δv ≈ ½·(Cd·A/m)·ρ(perigeo)·v_p·√(2π·a·H), con ρ y H
  de la atmósfera validada. **Sin J2** (no quita energía → no afecta al aerofrenado).
- **¿De dónde sale esa fórmula y qué asume? (por si preguntan):** es la aproximación clásica
  de **King-Hele** para el frenado en UNA pasada. La idea: el drag solo importa cerca del
  **perigeo** (donde ρ es máxima; sube exponencialmente al bajar). Al integrar el impulso de
  arrastre a lo largo de ese arco, suponiendo **atmósfera localmente exponencial** (escala H)
  y que **una sola pasada cambia poco la órbita**, la integral se resuelve y aparece el factor
  **√(2π·a·H)** = la "longitud efectiva" del trozo de atmósfera densa que la nave atraviesa.
  Es estándar en mecánica orbital (King-Hele, *Theory of Satellite Orbits in an Atmosphere*).
  *Defendible:* evita integrar la órbita paso a paso → entorno **rápido** (PPO necesita ~10⁵–10⁶
  pasos), a cambio de no resolver la órbita completa, que aquí no hace falta.

### El criterio de peligro: PRESIÓN DINÁMICA, no altitud (la clave del multi-planeta)
- **Qué es:** la nave se destruye si la **presión dinámica** `q = ½·ρ·v²` en el perigeo
  supera un límite `q_max` — NO a una altitud fija.
- **Por qué `q` y no la densidad (muy vendible):** lo que rompe la nave depende también de
  la **velocidad al cuadrado**. A igual densidad, en Júpiter (vas ~12× más rápido que en
  Marte) el calentamiento es **~600× mayor**. La densidad sola engañaría; `q` lo captura. Y
  `q` es justo lo que controlan las **misiones reales** de aerobraking.
- **`q_max = 0,6 N/m²`** = **valor real** (Mars Global Surveyor), no casero. Validación
  cruzada: el perigeo crítico que sale para Marte (~110 km) coincide con los ~108 km
  calibrados a mano antes.
- **Consecuencia:** el **corredor de perigeo seguro de cada planeta se deriva solo** de su
  atmósfera + `q_max` (nada a ojo) y **transfiere** entre cuerpos → permite un especialista
  por planeta con el mismo código.
- **Dónde se evalúa `q`:** en el **PERIGEO** (ahí ρ y v son máximas). En el apogeo ρ≈0 →
  `q`≈0, no pasa nada. Y la `q` del perigeo **baja durante el aerofrenado** aunque el
  perigeo esté fijo: la densidad no cambia, pero al encogerse la órbita la velocidad de
  paso baja (vis-viva) → el peligro es máximo **al principio** (órbita grande = paso rápido).

### Cómo es un aerofrenado REAL (por si lo preguntan — 4 fases)
1. **Captura:** encendido de motor → órbita muy elíptica con el **perigeo ALTO** (sobre la
   atmósfera densa; MRO: 426 × 44.500 km). Aún no roza la atmósfera.
2. **Walk-in:** con impulsos de motor en el apogeo, baja el perigeo hasta meterlo en el
   corredor. NO se entra de golpe; se coloca a propósito.
3. **Aerofrenado principal:** cientos de pasadas bajando el apogeo, ajustando el perigeo
   pasada a pasada. ← **ESTO es lo que modela nuestro agente.**
4. **Walk-out:** encendido final para subir el perigeo fuera de la atmósfera y dejar la
   órbita estable.

*Nuestro agente empieza en la fase 3* (perigeo ya en el corredor, apogeo alto); las fases
1, 2 y 4 son maniobras de motor estándar (no RL).

### Resultado
- **Multi-planeta:** un **especialista por planeta** para los **7 cuerpos con atmósfera**
  (Venus, Tierra, Marte, Júpiter, Saturno, Urano, Neptuno). Cada uno resuelve **cualquier**
  escenario de su planeta (apogeo inicial y objetivo aleatorios): **50/50 escenarios**
  nuevos cada uno → aprende la **estrategia**, no memoriza un caso. Algo que **ninguna
  fórmula da**.
- Frente a una **estrategia tonta** (perigeo alto fijo, que frena lentísimo o ni llega), el
  agente apura el perigeo y llega en una fracción de las pasadas (en Marte: 134 vs 862).
- **Comportamiento emergente (vendible):** el agente no busca el perigeo mínimo, sino el
  **óptimo seguro** (deja margen frente a `q_max`). El margen es **amplio donde la
  atmósfera ya frena con eficacia** (Urano, q~0,16) y **ajustado donde necesita apurar**
  (Júpiter, q~0,40). No se programó: emerge del aprendizaje.
- **Duración real** (sumando periodos): de ~2 semanas (Marte) a ~3 meses (Júpiter).

### ¿Cómo se valida si NO hay óptimo analítico? (PREGUNTA CLAVE muy probable)
En la Fase 1 hay un juez (la fórmula de Hohmann). En el aerofrenado **no existe fórmula
óptima** → ¿cómo se sabe que el agente lo hace bien? Por **cuatro anclajes convergentes**:
1. **Frente a una baseline tonta** (perigeo alto fijo): el agente llega en una fracción de
   las pasadas (Marte: 134 vs 862) o llega donde la tonta ni termina. Mejora medible.
2. **Cruce con una calibración independiente:** el perigeo crítico que el modelo de presión
   dinámica deriva para Marte (~110-112 km) **coincide** con los ~108 km que se habían
   calibrado a mano por separado → dos caminos distintos dan lo mismo.
3. **Coherencia física:** el agente se queda **dentro del corredor de `q`** (nunca supera
   `q_max`) y apura más donde puede y menos donde no → comportamiento físicamente sensato,
   no números arbitrarios.
4. **Realismo del orden de magnitud:** sale en **cientos de pasadas** y semanas-meses, igual
   que las misiones reales (MRO ~445 pasadas) → la escala es la correcta.

*Frase para el tribunal:* "no tengo un óptimo cerrado contra el que medir, así que valido por
**convergencia**: mejora sobre la baseline, coincidencia con la calibración manual, coherencia
física y realismo frente a misiones reales."

### Lecciones del entrenamiento (defendible)
- **Agente miedoso:** un "acantilado" de castigo pegado al óptimo envenenaba la zona
  buena → el agente se iba a perigeo alto y no llegaba (timeout).
- **Desequilibrio de escala:** el −200 de destrucción ahogaba el +10/pasada → PPO huía de
  lo profundo. SOLUCIÓN: dar **colchón** (peligro 112→108 km) + **castigo moderado**
  (−200→−50). Misma familia de lección que la Fase 1: la recompensa hay que escalarla bien.
- **El horizonte (gamma) — lección clave del multi-planeta:** con horizonte largo
  (cientos de pasadas) y el factor de descuento por defecto (0,99), el premio de éxito
  queda descontado a casi nada (200·0,99⁵⁰⁰ ≈ 1,3) y el agente se vuelve tímido → la
  **Tierra fallaba** (timeout). Subir gamma a **0,999** (200·0,999⁵⁰⁰ ≈ 122) hace visible
  ese premio lejano → 8/8. *Lección:* el descuento debe encajar con la longitud del episodio.

### Alcance / limitaciones (honestidad)
- **Solo BAJA órbitas:** el drag solo quita energía; subir una órbita necesita motor (otra
  maniobra). Y el perigeo debe meterse en la atmósfera.
- **Especialistas por planeta** (7 cuerpos), NO un único cerebro universal: las atmósferas
  sí tienen escala propia (densidad, altura de escala), a diferencia de las transferencias.
  Un único agente multi-planeta necesitaría meter la densidad/presión sentida en el estado
  (extensión).
- **Júpiter (honestidad):** sale **operable según el modelo de arrastre**, pero **inviable
  en la realidad** por factores no modelados (radiación letal, calentamiento a ~51 km/s).
  Mejor que un simple "no se puede".
- **Coeficiente balístico `Cd·A/m = 0,22 m²/kg`** = nave de ejemplo con mucha área; es
  **~5-10× el de MGS/MRO** (~0,02-0,05), por lo que nuestro aerofrenado es más ágil que las
  misiones reales (junto con un escenario de captura más modesto). Supuesto documentado.
- **J2 NO cambia la inclinación** (precesa Ω y ω; `i` se mantiene). Lo que cambia `i` un
  poco es el drag con atmósfera rotante. Mantener `i` exacta = entorno **3D** con control
  fuera del plano (extensión).
- **Atmósfera NO rotante en el entorno del agente** (honestidad — posible pregunta del
  tribunal): el drag y la presión dinámica se calculan con la velocidad **inercial**
  (vis-viva), no con la **relativa al aire** `v_rel = v − ω×r`. *(Ojo: el propagador del
  Bloque 2 SÍ usa `v_rel`; esta simplificación es solo del entorno de RL.)* Justificación:
  (1) es **conservadora** → `v_inercial > v_rel` en órbita prógrada, así que **sobreestima**
  el drag y la `q` → el agente se cree el perigeo más peligroso de lo que es → margen de
  seguridad extra; (2) el entorno es **2D sin inclinación**, así que incluir ω obligaría a
  fijar la hipótesis "ecuatorial prógrada" (caso particular; en órbita polar `v_rel ≈
  v_inercial`) → hacerlo bien es el entorno 3D (extensión); (3) **no cambia ninguna
  conclusión**: el efecto es ~5% en Marte/Tierra, despreciable en Venus (rota lentísimo) y
  solo ~25% en Júpiter, que ya está excluido por inviable. Misma familia que la omisión de
  J2: simplificación declarada, no descuido.
- Atmósfera **determinista** → el óptimo es el borde (perigeo más profundo seguro). Con
  **incertidumbre atmosférica** el agente tendría que mantener un margen (extensión realista).

### Agente 5: mantenimiento orbital (station-keeping) — IMPLEMENTADO
**Qué es:** el quinto agente. Mantiene una órbita operativa frente a las perturbaciones a
lo largo del tiempo. Es un problema **temporal multi-paso** (la órbita se degrada poco a
poco y hay que decidir cuándo/cuánto corregir), a diferencia de los otros agentes, que son
de un solo impulso o una sola maniobra.

**Qué corrige y qué NO (clave física, MUY preguntable):**
- El **arrastre** baja el semieje y circulariza la órbita → el agente lo corrige con
  **re-boost** para mantener la geometría (a, e, i) dentro de una banda. ESTO es lo que hace.
- El **J2 NO degrada la geometría**: en media secular deja a, e, i constantes; solo
  **precesa la orientación** (Ω y ω). → **NO se combate**; se modela y se contabiliza, pero
  no se corrige. *(OJO: esto corrige el plan inicial, que hablaba de "compensar Ω y ω".)*
  Corregir esa precesión sería fuera de plano y **carísimo/irrealista** — las órbitas
  heliosíncronas hasta la **aprovechan**.
- En la Luna y Mercurio (sin atmósfera) la geometría no se degrada → **nada que mantener**,
  excluidos (igual que en el aerofrenado).

**Cómo:** física **secular analítica** (rápida, sin Cowell por paso): `da/dt =
−(Cd·A/m)·ρ·v·a`, con ρ del **modelo atmosférico validado**; la precesión de Ω se sigue con
la **fórmula de J2 verificada vs Brouwer**. Episodio = **misión de duración fija**;
objetivo: mantener en banda con el mínimo Δv. **Especialista por planeta** (como el
aerofrenado: la atmósfera no es invariante de escala). La **franja de altitudes** de cada
cuerpo **se deriva de su atmósfera** (donde mantener cuesta 2–250 m/s/año), como el
corredor de perigeo del aerofrenado.

**Resultado:** los 7 cuerpos con atmósfera, **30/30** escenarios aleatorios cada uno.
Ahorro frente a la estrategia ingenua **según el régimen**: Marte **~45 %** (franja
estrecha + altura de escala pequeña → el control fino importa), gigantes **~0–2 %** (la
física ya fija el Δv y la heurística es casi óptima). El agente **nunca es peor**.

**Lecciones (ORO para la defensa):**
- **El Δv de mantenimiento está casi FIJADO por la física** (repone la energía que el
  arrastre quita) → no se puede "ahorrar por arte de magia"; el ahorro es marginal y
  **honesto**. El valor real del agente es la **generalización**, no el ahorro.
- **Reward hacking → bonus de supervivencia:** con una recompensa solo negativa, al agente
  le salía a cuenta **"suicidarse"** (salirse pronto = un castigo único < acumular
  negativos toda la misión). Se arregló premiando **cada día** mantenido en banda.
- **El tope del re-boost se ata a la BANDA, no a la velocidad:** escalarlo por la velocidad
  circular hacía que en los gigantes un re-boost moviera el semieje **4× la banda** → se
  "pasaba de largo" (la heurística parecía fallar por **artefacto**, no por física). Se ató
  al ancho de banda vía `Δa ≈ 2a·Δv/v`. *(Bug detectado al evaluar y corregido: honestidad.)*
- **Validación bonita:** la regresión nodal sale **~−5°/día para una órbita tipo ISS**, que
  coincide con la **real** de la estación → la fórmula secular de J2 clava la precesión.

**Preguntas del tribunal probables:**
- *"¿Por qué no corriges el J2?"* → porque J2 no degrada la geometría (a, e, i), solo gira
  la órbita; corregir esa rotación es fuera de plano y carísimo, no se hace en la práctica.
  Mantener la geometría = corregir el arrastre.
- *"¿El agente ahorra combustible?"* → poco, y es honesto: el Δv lo fija la física. El valor
  es la generalización (un mismo diseño mantiene los 7 cuerpos).
- *"¿Por qué un especialista por planeta y no uno universal?"* → como el aerofrenado: la
  atmósfera no es invariante de escala, a diferencia de las transferencias keplerianas.

---

## 6. Repaso de los scripts: qué defender y qué es "fontanería"

> Lectura transversal de los 9 scripts de `ia/` (3 entornos + 3 train + 3 evaluar +
> baselines). La idea: si el tribunal abre un archivo, saber **qué líneas son decisiones
> tuyas** (defendibles) y cuáles son **uso estándar de librería** (no hay que defender).

### Lo que es FONTANERÍA (no hace falta defender, basta reconocerlo)
- **`import gymnasium`, `spaces.Box`, heredar de `gym.Env`** → es solo el "molde" estándar
  para que stable-baselines3 entienda tu entorno. No lo programaste tú.
- **`PPO("MlpPolicy", ...)`, `model.learn()`, `model.predict()`, `PPO.load()`** → API de
  stable-baselines3. No reimplementas PPO; lo aplicas.
- **`Monitor(env)`** → solo registra la recompensa por episodio para ver el progreso.
- **`np.clip`, reescalados `[-1,1]→[lo,hi]`, `np.sqrt`** → utilidades.
- **El algoritmo PPO por dentro** (clipping del ratio, ventaja, GAE…) → teoría de la
  librería; puedes citar qué es PPO pero no tienes que derivarlo.

### Lo que SÍ es tuyo y defendible (transversal a los 3 agentes)
- **La formulación del problema como MDP**: qué metiste en el **estado**, qué es la
  **acción**, cómo es la **recompensa**. Esto es 100% diseño tuyo y es el corazón de la
  defensa (ya detallado en §4 y §5).
- **`reset()` y `step()`**: dentro de `step` vive **tu física** (vis-viva, elipse de
  transferencia, King-Hele, presión dinámica). Eso NO es de la librería; lo escribiste tú a
  partir de los Fundamentos.
- **`baselines.py` es el JUEZ, separado del agente**: el óptimo clásico (Hohmann) se calcula
  aparte y solo sirve para **puntuar** al agente. *Defendible:* el agente nunca "ve" la
  fórmula → cuando se acerca a ella, es que de verdad aprendió. `hohmann_adim(R)` es la
  versión adimensional (demuestra que el óptimo solo depende de `R`).

### Detalles concretos del código que te pueden preguntar
- **`seed=0` (reproducibilidad):** fija el azar para que el entrenamiento sea repetible. *Si
  preguntan "¿es suerte de la semilla?"* → no: el aerofrenado se reevaluó con **semillas
  nuevas y sale 50/50**; la transferencia generaliza a planetas nunca vistos. La estrategia
  es robusta, no un golpe de suerte.
- **`deterministic=True` en `predict` (al evaluar):** el agente toma su **acción media**, sin
  el ruido de exploración del entrenamiento → resultado fijo y reproducible. Al entrenar SÍ
  explora (con ruido); al examinar, no.
- **`EvalCallback` con un `eval_env` aparte:** se evalúa en un entorno **distinto** del de
  entrenamiento y se guarda el **mejor** modelo visto (no el último). *Defendible:* evita
  quedarte con una política que empeoró por la inestabilidad de PPO (lección de la Fase 1).
- **`MlpPolicy` = red neuronal pequeña** (2 capas de 64 por defecto en SB3). *Si preguntan
  por la arquitectura:* es la estándar; el problema no necesita más. Se anotó que una red
  mayor `[128,128]` podría ayudar al extremo difícil de la transferencia (no se vio
  necesario).
- **`ent_coef` (0,01–0,02):** premio extra por **explorar** (mantener variedad de acciones)
  → evita que el agente se "cierre" pronto en una solución mala. Es un dial, lo defendible es
  el porqué.
- **Reescalar la acción `[-1,1] → rango físico`:** las redes neuronales trabajan mejor con
  entradas/salidas normalizadas en torno a 0 → el agente decide en `[-1,1]` y el entorno lo
  traduce a Δv reales o a una altitud de perigeo.

### Específicos de cada entorno (lo más fino, por si tiran del hilo)
- **`env_hohmann` — `aleatorio=False`:** se entrena SOLO el caso fijo LEO→GEO (prueba de
  concepto que se clava al 0,03%). El modo `aleatorio=True` (cualquier r1,r2) aprende la
  tendencia pero no clava el extremo GEO (muy sensible al primer impulso) → se generalizó
  mejor por otra vía (el Agente 2 adimensional).
- **`env_transfer` — normalización FIJA con `LOG_RMAX`:** aunque el currículum entrene con un
  rango de `R` menor al principio, el estado se normaliza SIEMPRE con el rango pleno → la
  "escala" que ve el agente no cambia entre etapas ni al evaluar (coherencia). **Límites
  asimétricos `DV1/DV2` a medida:** topar el 1.er impulso por debajo del umbral de escape
  (0,40 < √2−1) mete el **límite físico dentro de la acción** y elimina el "castigo-acantilado".
- **`env_drag` — `_altura_para_q` (bisección):** el corredor de perigeo de cada planeta se
  **deriva** de su atmósfera + `q_max`, no se pone a ojo. **Normalización por `H_REF` y
  `V_REF`:** divide longitudes y velocidades por escalas propias del planeta → los **mismos
  pesos de recompensa** valen en Marte y en Júpiter pese a tamaños y velocidades muy
  distintos. **`gamma=0,999`:** horizonte largo (cientos de pasadas) → ver la lección del
  descuento en §5.

### Específicos del entrenamiento / evaluación
- **`train_transfer.py` — currículum por etapas:** se reutiliza el **mismo** modelo entre
  etapas (`set_env`, `reset_num_timesteps=False`) ampliando el rango de `R`; el `EvalCallback`
  evalúa **siempre en el rango pleno** → "el mejor modelo" se mide donde de verdad importa.
- **`evaluar_transfer.py` — la demo estrella (Parte B):** coge el modelo entrenado SOLO en
  adimensional y lo prueba en Marte/Júpiter **sin reentrenar**, multiplicando su impulso por
  la `v_c1` real del planeta → es la prueba visible de la **invariancia de escala**.
- **`evaluar_drag.py`:** lanza N escenarios aleatorios (`reset(seed=s)`) del mismo planeta →
  demuestra que **generaliza dentro del planeta** (no memoriza un caso).

---

> **Pendiente de añadir según avancemos:** Bloque 5 (capa LLM) y cualquier decisión nueva.
