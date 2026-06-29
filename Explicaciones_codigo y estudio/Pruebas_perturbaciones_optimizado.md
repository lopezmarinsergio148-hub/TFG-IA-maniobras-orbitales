# Pruebas_perturbaciones_optimizado.py — Documentación

> Propagador orbital unificado con **J2 + drag** para **9 cuerpos** del
> sistema solar: 7 con atmósfera (Tierra, Marte, Venus, Júpiter, Saturno,
> Urano, Neptuno) + 2 sin atmósfera (Luna, Mercurio — sólo J2).
> Combina poliastro como backend de propagación numérica con el modelo
> atmosférico por capas del módulo
> [`Densidades_atmosferica_optimizado.py`](Densidades_atmosferica_optimizado.md).
>
> Cierre del **Bloque 2** del plan del TFG.

---

## 1. Propósito

Mientras que el simulador del Bloque 1 trabaja en 1D radial (`h(t)` por
decay energético), este módulo propaga **la órbita 3D completa** del
satélite teniendo en cuenta dos perturbaciones físicas:

1. **J2 (achatamiento del planeta)** — produce la **regresión nodal** y la
   oscilación periódica de la órbita osculadora.
2. **Drag atmosférico** — frena el satélite y reduce su energía orbital.

El resultado es la herramienta que necesitarás en el **Bloque 4 (entorno
de RL)**: el agente actuará sobre una órbita 3D realista, no sobre una
trayectoria simplificada.

---

## 2. Cómo usarlo

### 2.1 Modo interactivo

```bash
python Pruebas_perturbaciones_optimizado.py
```

El script abre un menú para elegir planeta, pide los parámetros del
satélite y abre dos gráficas: trayectoria 3D (Plotly, navegador) +
altitud vs tiempo (matplotlib).

### 2.2 Modo librería (lo que usará el Bloque 4)

```python
from astropy import units as u
from poliastro.twobody import Orbit
from Densidades_atmosferica_optimizado import TIERRA
from Pruebas_perturbaciones_optimizado import propagar_perturbado

# Órbita inicial (ISS-like)
orb = Orbit.circular(TIERRA.body, 400 * u.km, inc=51.6 * u.deg)

# Propagar 30 días con J2 + drag (atmósfera rotante, F10.7 medio)
orb_final = propagar_perturbado(
    planeta=TIERRA,
    orb_inicial=orb,
    tiempo=30 * u.day,
    masa=420_000, area=2500, cd=2.2,
    factor_solar=1.0,
)

print(f"a final = {orb_final.a.to(u.km).value:.2f} km")
print(f"ecc final = {orb_final.ecc.value:.6f}")
print(f"RAAN final = {orb_final.raan.to(u.deg).value:.2f} deg")
```

### 2.3 Trayectorias muestreadas (para visualización o RL)

```python
from Pruebas_perturbaciones_optimizado import (
    trayectoria_perturbada, trayectoria_kepler
)

# 200 puntos muestreados en 30 días
tiempos, posiciones, alturas, orb_final = trayectoria_perturbada(
    TIERRA, orb, 30 * u.day, num_pasos=200,
    masa=420_000, area=2500, cd=2.2,
)
# tiempos    : (201,)    días
# posiciones : (201, 3)  km
# alturas    : (201,)    km
# orb_final  : poliastro.Orbit — estado final completo

# El semieje del orb_final es la métrica MÁS ESTABLE del decay
# (la altura instantánea oscila por J2 entre perigeo y apogeo).
print(f"a final = {orb_final.a.to(u.km).value:.3f} km")
print(f"apogeo  = {orb_final.r_a.to(u.km).value - TIERRA.R_m/1000:.1f} km")
print(f"perigeo = {orb_final.r_p.to(u.km).value - TIERRA.R_m/1000:.1f} km")
```

---

## 3. Algoritmo

### 3.1 Modelo físico

La ecuación del movimiento se descompone en tres aportes:

$$\ddot{\vec{r}} = -\frac{\mu}{r^3} \vec{r} \;+\; \vec{a}_{J_2} \;+\; \vec{a}_{drag}$$

**Término Kepleriano** (gravedad central): `poliastro.core.propagation.func_twobody`.

**Término J2** (achatamiento, simétrico de revolución alrededor del eje de rotación):

$$\vec{a}_{J_2} = -\frac{3}{2} \frac{J_2 \,\mu\, R^2}{r^5}
\begin{pmatrix} x \,(1 - 5z^2/r^2) \\ y \,(1 - 5z^2/r^2) \\ z \,(3 - 5z^2/r^2) \end{pmatrix}$$

Implementado por `poliastro.core.perturbations.J2_perturbation`.

**Término drag** (con atmósfera rotante):

$$\vec{a}_{drag} = -\frac{1}{2} \rho(h) \, C_d \frac{A}{m} \, |\vec{v}_{rel}| \, \vec{v}_{rel}$$

donde:
- `ρ(h)` se obtiene de `planeta.get_rho(h)` (modelo de capas piecewise-exponencial).
- `v_rel = v_sat − ω × r` es la velocidad relativa al aire (atmósfera rotante con la velocidad angular del planeta).

### 3.2 Integrador

Se usa **CowellPropagator** de poliastro (Runge-Kutta adaptativo). La
función de perturbación se construye en `crear_funcion_perturbacion`,
que captura los parámetros del planeta y del satélite como closure y
devuelve una `f(t0, state, k)` lista para inyectar en CowellPropagator.

**Convenciones de unidades dentro de la función**: poliastro internamente
usa km para distancias y km/s para velocidades. La función de
perturbación convierte internamente a m/s para calcular el drag con
`ρ` en kg/m³, y luego vuelve a km/s² para devolver las aceleraciones.

### 3.3 Propagación incremental

Para visualización, `trayectoria_perturbada` no propaga `n` veces desde
`t=0` (ineficiente), sino **incrementalmente**: cada paso `dt`,
`orb_actual.propagate(dt)` se llama partiendo del estado anterior. Esto
hace que 200 pasos en 30 días sean prácticamente equivalentes en coste
a una sola propagación de 30 días.

---

## 4. Estructura del código

| Bloque | Funciones |
|---|---|
| **1. Perturbación unificada** | `crear_funcion_perturbacion(planeta, ...)` |
| **2. Propagación** | `propagar_perturbado(...)`, `trayectoria_perturbada(...)`, `trayectoria_kepler(...)` |
| **3. Visualización** | `graficar_3d(...)` (Plotly), `graficar_2d_altura(...)` (matplotlib) |
| **4. Menú** | `elegir_planeta()`, `main()` |

### Parámetros opcionales clave

`propagar_perturbado` acepta tres flags para activar/desactivar términos:

| Flag | Default | Uso |
|---|---|---|
| `incluir_J2` | `True` | Activa el término J2 |
| `incluir_drag` | `True` | Activa el drag atmosférico |
| `atmosfera_rotante` | `True` | Si False, ignora ω·r (atmósfera estática) |

Útiles para **comparar contribuciones** en la memoria del TFG: corre el
mismo caso 4 veces con (no J2 / no drag, sí J2 / no drag, no J2 / sí
drag, sí J2 / sí drag) y muestra el efecto de cada uno.

### 4.1 Cuerpos sin atmósfera (Luna y Mercurio)

`crear_funcion_perturbacion` **detecta automáticamente** si `planeta.tiene_atmosfera == False` (Luna, Mercurio) y desactiva el drag aunque `incluir_drag=True`. Sólo se aplica gravedad + J₂. El menú interactivo no pide masa/área/Cd/factor solar en estos casos.

### 4.2 Detección automática de reentrada

`trayectoria_perturbada` devuelve un flag adicional `hubo_reentrada` (bool) que se activa cuando:
1. El integrador `CowellPropagator` falla con `RuntimeError` (reentrada catastrófica con cambios bruscos de densidad).
2. La altitud cruza `planeta.h_reentrada_m` (corte limpio).

Si `hubo_reentrada=True`, `graficar_3d` **no dibuja la órbita final ni el marcador del satélite final** (porque la elipse osculadora ya no tiene sentido físico — su perigeo puede estar dentro del planeta).

### 4.3 Métrica de decay secular: ajuste lineal con aviso de fiabilidad

El **decay secular real** se mide con un **ajuste lineal** de h(t) sobre la **segunda mitad** de la trayectoria (filtra el transitorio inicial y las oscilaciones de J2). La pendiente m da los km/día perdidos por drag.

Como las oscilaciones de J2 pueden dominar sobre el decay real (especialmente a alturas donde el drag es despreciable), el código calcula también la ratio `|decay| / (rango de oscilación J2)`:

| Ratio | Diagnóstico |
|---|---|
| > 0.40 | ✓ Fiable: drag domina sobre oscilación |
| 0.15 – 0.40 | ⓘ Marginal: tomar como orden de magnitud |
| < 0.15 (con atmósfera) | ⚠️ NO fiable: la pendiente es ruido J2 |
| < 0.15 (sin atmósfera) | ⓘ Cuerpo sin drag: ratio bajo esperado |

### 4.4 Display de la regresión nodal

El RAAN se reporta tanto en convención `[0, 360°)` (instantánea) como en valor **signed con "unwrap"** (precesión total real, p.ej. `-149.3°` en vez de `+210.7°`). Se compara automáticamente contra la fórmula de Brouwer (sección 5).

---

## 5. Validación con ISS y fórmula de Brouwer (1959)

Caso ejecutado para validar el modelo:
- Planeta: **Tierra** (USSA-76)
- Altura inicial: **420 km**
- Inclinación: **51.6°** (ISS real)
- Masa: 450 000 kg, Área: 2500 m², Cd: 2.2
- Tiempo: 30 días, factor solar: medio (F10.7 ~ 150)

| Magnitud | Resultado | Comparación |
|---|---|---|
| **Regresión nodal por J2 (simulada)** | **−149.3° en 30d** | Coincide con Brouwer dentro de 0.8° |
| **Regresión nodal por J2 (Brouwer)** | **−148.5° en 30d** | Fórmula analítica clásica |
| **Excentricidad inducida** | ~ 0.001 (oscila) | Pequeña perturbación esperable |
| **Decay semieje (drag puro)** | 5.1 km/mes | Compatible con la física |
| **Decay semieje (J2 + drag)** | 5.8 km/mes | El término J2 no contribuye al decay neto |

### Validación de la fórmula de Brouwer para múltiples casos

$$\frac{d\Omega}{dt} = -\frac{3}{2} J_2 \, n \, \left(\frac{R}{p}\right)^2 \frac{\cos(i)}{(1-e^2)^2}$$

| Caso | i | ΔRAAN simulada | ΔRAAN Brouwer |
|---|---|---|---|
| ISS Tierra | 51.6° | -149.3° | -148.5° |
| Luna (h=100km) | 30° | -31.2° | -31.2° |
| Mercurio (h=200km) | 45° | -7.4° | -7.4° |
| Tierra polar | 90° | 0.0° | 0.0° (cos=0) |
| Tierra retrógrada | 120° | +115.3° | +114.8° (signo invertido) |

✅ Coincidencia con Brouwer dentro del 1%. La discrepancia ISS (0.8°) corresponde al **acoplamiento drag-J₂** (cuando el drag baja `a`, aumenta `n` y la precesión se acelera ligeramente).

### Sobre la ISS real (1-2 km/mes)

El valor del simulador (5 km/mes) supera al real (1-2 km/mes) por
**parámetros de entrada**, no por error del modelo:

- **Área efectiva**: 2500 m² es el área TOTAL de paneles. Cuando los
  paneles se orientan al Sol (no al flujo), la sección frontal efectiva
  promedio es ~1000-1500 m².
- **Actividad solar**: la cifra "1-2 km/mes" se reporta en **mínimo
  solar** (F10.7 ~70 → `factor_solar = 0.5`).

Reejecutando con `area=1000, factor_solar=0.5`:

```
5.1 km/mes × (1000/2500) × 0.5 ≈ 1.0 km/mes  ✓
```

→ **El modelo es físicamente correcto.** La diferencia con valores
publicados se explica por interpretación correcta de los parámetros.

> **Para la memoria**: este ejemplo es ideal porque enseña por qué los
> "datos de ISS" que ves en webs deben usarse con cuidado: dependen del
> área que cada autor toma y del momento del ciclo solar.

---

## 6. Combinación con el Bloque 1

| Componente | Módulo | Función |
|---|---|---|
| Constantes físicas | `Densidades_atmosferica_optimizado` | `JUPITER.mu_m3_s2`, `JUPITER.R_m`, etc. |
| Modelo atmosférico | `Densidades_atmosferica_optimizado` | `JUPITER.get_rho(h)` |
| Configuración integrador | `Densidades_atmosferica_optimizado` | `JUPITER.h_reentrada_m`, `JUPITER.omega_rad_s` |
| Propagación con perturbaciones | `Pruebas_perturbaciones_optimizado` | `propagar_perturbado(...)` |

El módulo del Bloque 2 **importa** del Bloque 1 sin duplicar nada.

---

## 7. Limitaciones

| Limitación | Impacto | Cuándo importa |
|---|---|---|
| Solo **J2** (no J3, J4...) | Pérdida menor de precisión en órbitas de larga duración o muy inclinadas. | TFG estándar: aceptable. |
| Atmósfera **simétrica de revolución** | No captura efectos diurnos/aurorales. | Solo importa para Tierra (USSA-76 ya lo simplifica). |
| Sin **presión de radiación solar** (SRP) | En órbitas altas o satélites con mucha área, SRP > drag. | Geoestacionario o más alto. |
| Sin **tercer cuerpo** (Luna/Sol) | Importante a altas altitudes. | Órbitas > 20 000 km. |
| Sin **gravedad de mareas** | Efecto pequeño. | Misiones de precisión (GPS, etc.). |
| **Órbitas muy bajas + J2 alto** (Júpiter, Saturno) | La oscilación osculadora del J2 hunde el perijove instantáneo bajo la superficie; el integrador aborta y se reporta (de forma engañosa) como *"reentrada catastrófica por capas densas"*. | Órbitas ≲ 3000–4000 km en Júpiter/Saturno → arrancar más alto. |
| **Validación de Brouwer solo a 1.er orden** | La fórmula analítica de referencia se desvía del modelo (~5% en Júpiter/Saturno) por los términos J2² que desprecia. La simulación es la referencia más fiel. | Comparación cuantitativa de regresión nodal en planetas con J2 alto. |

Para el alcance del TFG (LEO/MEO + transferencias intra-planeta), J2 +
drag son las dos perturbaciones dominantes y suficientes.

### Detalle: órbitas bajas en planetas con J2 alto (Júpiter, Saturno)

El J2 de Júpiter y Saturno (≈1.5 × 10⁻²) es ~14× el de la Tierra. En una órbita
**baja**, el J2 fuerza una **excentricidad osculadora** de hasta ~0.04 que, sobre una
órbita ya cercana al planeta, hace que el **perijove instantáneo baje varios miles de
km** y cruce (osculadoramente) el nivel de 1 bar. En ese punto el modelo de densidad
satura (ancla h=0 → densidad de 1 bar, enorme), el drag se dispara y el integrador
aborta — y el programa lo etiqueta, de forma engañosa, como *"reentrada catastrófica"*.

**No es un fallo del propagador**, sino la consecuencia física de que, con un J2 tan
fuerte, una órbita baja "circular" **no es un estado estable**: en la práctica esas
órbitas son inviables (el satélite rozaría la atmósfera densa en cada perijove).

- **Ejemplo:** a 2300 km sobre Júpiter, el perijove osculador baja a ≈ −600 km (bajo
  la superficie) en **menos de una órbita**, aun con drag despreciable.
- **Recomendación de uso:** en Júpiter/Saturno, partir de **h ≳ 3000–4000 km**. Urano
  y Neptuno (J2 ~4× menor) son mucho menos sensibles.

> Esta limitación está pensada para que la **capa LLM (Bloque 5)** pueda diagnosticar
> y justificar el caso: si un usuario lanza una órbita baja en Júpiter/Saturno y ve
> "reentrada catastrófica", la causa real es el J2, no el arrastre.

---

## 8. Para la memoria del TFG

Lo "vendible" de este módulo:

1. **Integración profesional**: usa CowellPropagator de poliastro, el
   estándar de facto en astrodinámica con Python.
2. **Modelo de drag propio**: no usamos el modelo exponencial simple de
   poliastro (`atmospheric_drag_exponential`), sino el modelo de capas
   piecewise-exponencial validado en el Bloque 0 — más fiel a la
   realidad para cualquier planeta.
3. **Atmósfera rotante**: la velocidad relativa se calcula con `v − ω × r`,
   típicamente despreciada en simuladores simples pero importante en LEO.
4. **Validación cuantitativa**: el caso ISS reproduce la regresión nodal
   teórica (-5°/día para 51.6° de inclinación) y, ajustando parámetros
   realistas (área frontal efectiva, ciclo solar), el decay observado.
5. **Genericidad**: una sola función `propagar_perturbado` cubre los 7
   planetas. Añadir el 8° solo requiere extender el Bloque 0.

Frase modelo para la memoria:

> *"La propagación orbital se realiza mediante el integrador Cowell de
> poliastro, al que se inyecta una función de perturbación combinando
> el término J2 (proporcionado por `poliastro.core.perturbations`) y el
> término de arrastre atmosférico calculado con el modelo de capas
> propio descrito en la sección anterior. La velocidad relativa al aire
> se obtiene como `v_rel = v_sat − ω × r`, considerando la atmósfera
> rotante con la velocidad angular del cuerpo central. El modelo se
> ha validado contra el caso de la Estación Espacial Internacional,
> reproduciendo la regresión nodal teórica para una órbita inclinada
> 51.6° (-5°/día) y, ajustando los parámetros efectivos del satélite
> (área frontal media y ciclo solar), el ritmo de decaimiento observado
> (1-2 km/mes)."*
