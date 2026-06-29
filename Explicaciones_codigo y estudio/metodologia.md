# METODOLOGÍA DEL SIMULADOR DE DECAIMIENTO ORBITAL POR DRAG

> Documento técnico que recoge el modelo físico, las ecuaciones, el integrador
> numérico y las limitaciones del simulador desarrollado para el TFG.
> Pensado para servir de base al capítulo de **metodología** de la memoria.

---

## 1. Resumen del problema

Se quiere simular el **decaimiento orbital** de un satélite por **arrastre atmosférico** (drag) alrededor de un planeta. El problema:

- **Entradas**: planeta (μ, R), modelo atmosférico `ρ(h)`, parámetros del satélite (masa `m`, área frontal `A`, coeficiente de arrastre `C_d`), condiciones iniciales (altura `h₀`).
- **Salida**: evolución de la altitud `h(t)` hasta que el satélite alcanza la altitud crítica de reentrada `h_critica`, o hasta agotar el tiempo simulado.

El modelo cubre **9 cuerpos del sistema solar**:
- **Con atmósfera** (7): Tierra, Marte, Venus, Júpiter, Saturno, Urano, Neptuno.
- **Sin atmósfera** (2): Luna y Mercurio (solo gravedad central + J₂, sin drag).

Cada uno con sus constantes físicas propias. Los cuerpos sin atmósfera permiten estudiar perturbaciones orbitales puras (J₂) sin drag.

---

## 2. Modelo físico

### 2.1 Energía orbital

Asumiendo **órbita cuasi-circular** en cada instante (aproximación válida cuando el drag es perturbativo), la energía orbital total de un satélite de masa `m` a distancia `r` del centro del planeta es:

$$E_{orb} = -\frac{G M m}{2r}$$

donde:
- `G` = constante gravitacional universal
- `M` = masa del planeta
- `r = R + h` (radio = R del planeta + altitud)

### 2.2 Fuerza de arrastre

La fuerza de arrastre aerodinámico sobre el satélite es:

$$F_{drag} = \frac{1}{2} \rho \, v^2 \, C_d \, A$$

donde:
- `ρ` = densidad atmosférica a la altura actual (kg/m³)
- `v = √(GM/r)` = velocidad orbital circular
- `C_d` = coeficiente de arrastre (≈ 2.0–2.4 para satélites estándar)
- `A` = área frontal expuesta al flujo (m²)

### 2.3 Pérdida de energía por drag

La potencia disipada por el drag (energía perdida por unidad de tiempo):

$$\frac{dE}{dt} = -F_{drag} \cdot v = -\frac{1}{2} \rho \, v^3 \, C_d \, A$$

Discretizando con paso `dt`:

$$dE = F_{drag} \cdot v \cdot dt$$

### 2.4 Actualización de la órbita

A cada paso, restamos `dE` a `E_orbital`:

$$E_{nueva} = E_{actual} - dE$$

Y recalculamos `r` despejando de la ecuación de energía:

$$r_{nueva} = -\frac{G M m}{2 \, E_{nueva}}$$

De donde la nueva altitud:

$$h_{nueva} = r_{nueva} - R$$

---

## 3. Modelo atmosférico por capas

### 3.1 Origen físico: del equilibrio hidrostático a la fórmula exponencial

La forma exponencial de la densidad **no es arbitraria**: sale de combinar dos
leyes físicas. Esta es la derivación completa.

**(1) Equilibrio hidrostático.** Una capa fina de atmósfera de espesor `dh` está
en equilibrio cuando la diferencia de presión entre su base y su techo sostiene
exactamente su propio peso. Eso da:

$$\frac{dP}{dh} = -\rho \, g$$

(el signo `−`: la presión disminuye al subir).

**(2) Ley de los gases ideales.** Relaciona presión y densidad mediante la
temperatura y el peso molecular medio `μ`:

$$P = \frac{\rho}{\mu} R \, T \qquad\Longrightarrow\qquad \rho = \frac{P\,\mu}{R\,T}$$

con `R = 8.314 J/(mol·K)` y `μ` en kg/mol.

**(3) Se combinan.** Sustituyendo `ρ` de (2) en (1):

$$\frac{dP}{dh} = -\frac{P\,\mu\,g}{R\,T} \qquad\Longrightarrow\qquad \frac{dP}{P} = -\frac{dh}{H}, \quad\text{donde}\quad \boxed{H \equiv \frac{R\,T}{\mu\,g}}$$

**(4) Se integra** (suponiendo `T`, `μ`, `g` ≈ constantes dentro de la capa —
aproximación **isoterma** por tramos). Desde el límite inferior `h_min`:

$$\ln\frac{P}{P_{base}} = -\frac{h - h_{min}}{H} \qquad\Longrightarrow\qquad \rho(h) = \rho_{base}\cdot\exp\!\left(-\frac{h - h_{min}}{H}\right)$$

(como `T` es constante en la capa, `ρ ∝ P`, así que la misma exponencial vale
para la densidad). **Esa es exactamente la fórmula que usa el modelo.**

**Significado físico de `H` (altura de escala):** es la altura en la que la
densidad cae por un factor `e` (≈2.718). `H = RT/(μg)` → una atmósfera caliente,
de gas ligero (μ pequeño) y gravedad baja es **muy extendida** (H grande). Por
eso los gigantes gaseosos tienen H de cientos de km y la Tierra solo ~8 km.

> **Conexión con la validación de Urano/Neptuno:** en `datos_validacion/voyager_*`
> usamos esta MISMA física al revés. Allí no suponemos capas isotermas: integramos
> `dP/dh = −ρg` numéricamente con el perfil real `T(P)` medido por Voyager para
> reconstruir `ρ(h)`. Aquí, para el modelo de capas, la simplificamos a tramos
> isotermos y obtenemos la exponencial.

> **Auto-comprobación:** en el modelo, `H` se fija por *continuidad* (§3.3), no
> calculando `RT/μg`. Pero ambas deben coincidir si los datos son buenos. Cuando
> no coinciden, salta un error: en Neptuno la continuidad daba `H = 6.84 km`
> mientras la física daba `RT/μg ≈ 18 km` → eso destapó que el `ρ_base` a 50 km
> estaba ~100× mal (ver `neptuno.md §7`).

### 3.2 Forma piecewise-exponencial

Cada planeta se divide en `N` capas. Dentro de cada capa, la densidad sigue:

$$\rho(h) = \rho_{base,i} \cdot \exp\left(-\frac{h - h_{min,i}}{H_i}\right)$$

donde:
- `ρ_base,i` = densidad de referencia en el límite inferior de la capa `i` (datos observados de sondas)
- `H_i` = altura de escala de la capa `i` (km)
- `h_min,i` = altitud del límite inferior de la capa `i`

### 3.3 Derivación de H para continuidad

**Las densidades `ρ_base` son medidas observacionales** extraídas de:
- **Tierra**: librería Python `ussa1976` (implementación oficial del modelo USSA-76).
- **Marte**: consulta directa al **Mars Climate Database (MCD)** del LMD/ESA en el sitio de Opportunity con escenario climatology ave solar.
- **Venus**: perfil T(h) público de divulgación (basado en Pioneer Venus/Venera) + cálculo propio de ρ con ecuación de los gases ideales y g/μ tabulados.
- **Júpiter**: ✅ **recalibrado con el dataset oficial Galileo Probe ASI** (NASA-PDS, 693 puntos in-situ, DOI:10.17189/tfsa-pb91, Seiff 1998).
- **Saturno**: ✅ **recalibrado con el dataset oficial Cassini Grand Finale** (NASA-PDS, 763 puntos in-situ, DOI:10.17189/518e-p721, Koskinen et al. 2019).
- **Urano / Neptuno**: ✅ **validados contra Voyager 2 reconstruido** (Lindal 1987/1990) por integración hidrostática — no existe dataset PDS in-situ. Urano: factor 2.49× → 1.15× tras recalibrar. Neptuno: la validación detectó y corrigió un error de hasta 100× (factor 8.0× → 1.2×), con la reconstrucción validada cruzándola con el Neptune-GRAM. Ver `datos_validacion/voyager_{urano,neptuno}/`.

Para garantizar que la densidad sea **continua** entre capas adyacentes, las alturas de escala `H_i` se derivan de:

$$H_i = \frac{h_{min,i-1} - h_{min,i}}{\ln\left(\rho_{base,i} / \rho_{base,i-1}\right)}$$

Esta condición impone:

$$\rho_{base,i} \cdot \exp\left(-\frac{h_{min,i-1} - h_{min,i}}{H_i}\right) = \rho_{base,i-1}$$

Sin esta derivación, las capas tendrían saltos artificiales de densidad en sus fronteras, lo que introduce discontinuidades no físicas en la simulación.

### 3.4 Búsqueda eficiente de capa

El algoritmo de búsqueda explota que **el satélite siempre baja** durante el decaimiento. El índice de capa `idx_capa` se mantiene entre pasos consecutivos y solo se incrementa cuando se cruza un límite. Coste: `O(1)` amortizado por paso (en vez de `O(N)` de una búsqueda binaria por paso).

### 3.5 Niveles de referencia "h = 0"

| Planeta | Convención de h=0 |
|---|---|
| Tierra, Marte, Venus | Superficie sólida (nivel del mar o datum equivalente) |
| Júpiter, Saturno, Urano, Neptuno | **Nivel de presión P = 1 bar** (no hay superficie sólida) |

Esta diferencia es crítica para interpretar los resultados: en gigantes gaseosos, alturas negativas (`h < 0`) representan el interior fluido del planeta, no un "subterráneo".

---

## 4. Integrador numérico

### 4.1 Paso adaptativo `dt`

El paso de integración varía con la altitud para mantener precisión sin gastar recursos:

```
DT_SCHEDULE por planeta (zona alta: dt grande; zona baja: dt pequeño)
```

Esto refleja que el drag escala con `ρ·v²`, y ambos crecen al bajar.

### 4.2 Halvings (refinamiento del paso)

Antes de cada paso, se aplican tres salvaguardas en cascada que **reducen `dt` a la mitad** si:

1. **Energía**: el paso disiparía más del **0.1% de `|E_orbital|`** en un solo `dt`. Esto evita saltos energéticos demasiado grandes.

2. **Estabilidad numérica**: la `E_nueva` pasaría a positiva (órbita hiperbólica, físicamente imposible bajo drag puro).

3. **Cinemática**: la caída de radio `Δr` en el paso superaría **10 km**. Crítico en planetas grandes (Júpiter R = 71 500 km) donde una pequeña fracción de energía puede traducirse en cientos de km de Δr.

```python
while dt > MIN_DT:
    dE = F_drag · v · dt
    if dE > |E_actual| · 0.001:    # criterio energético
        dt /= 2;  continue
    E_prov = E_actual - dE
    if E_prov >= 0:                # estabilidad
        dt /= 2;  continue
    r_prov = -GMm / (2 · E_prov)
    if (r - r_prov) > 10_000:      # cinemática
        dt /= 2;  continue
    break  # paso aceptado
```

### 4.3 Detección de reentrada

Tras calcular la nueva altitud, se comprueba **antes** de muestrear:

```python
if h_m <= H_REENTRADA:
    registrar reentrada
    break
```

Comprobar antes del muestreo evita guardar puntos intermedios absurdos (por ejemplo, `h_m` negativa fuertemente).

### 4.4 Muestreo de datos

Se guarda un punto en la lista de resultados cada vez que:
- El satélite ha caído al menos **1 km** desde el último punto guardado, **o**
- Ha pasado al menos **1 hora** simulada desde el último punto.

Esto da gráficas suaves sin saturar memoria.

---

## 5. Limitaciones del modelo

El simulador captura bien la **fenomenología del decaimiento orbital por drag**, pero hay limitaciones que conviene declarar.

**Nota importante**: las limitaciones de esta sección aplican al **Bloque 0** (`Densidades_atmosferica_optimizado.py`, decay 1D por energía). El **Bloque 2** (`Pruebas_perturbaciones_optimizado.py`) extiende el modelo añadiendo **propagación orbital 3D** con `poliastro.CowellPropagator`, **perturbación J₂**, **atmósfera rotante** y comparación contra la fórmula clásica de **Brouwer (1959)** para la regresión nodal.

| Limitación (Bloque 0) | ¿Resuelto en Bloque 2? | Cuándo importa |
|---|---|---|
| **No incluye J₂** | ✅ SÍ en Bloque 2 (vía `J2_perturbation` de poliastro) | Solo Bloque 0 |
| **Órbita cuasi-circular asumida** | Parcial (Bloque 2 propaga con elementos osculadores reales) | Sondas con e>0.3 |
| **Modelo 1D radial** (`h(t)`) | ✅ SÍ en Bloque 2 (propagación 3D completa) | Visualizaciones 3D |
| **Atmósfera estática** | ✅ SÍ en Bloque 2 (incluye `v_rel = v − ω × r`) | Órbitas LEO bajas |
| **Sin actividad solar variable** | Solo Tierra (F10.7 baja/media/alta); otros planetas: ignorado | Atmósferas tenues |
| **No es entrada balística** | NO (ningún bloque lo cubre) | Replicar Galileo/Huygens |
| **Datos de Urano/Neptuno limitados** | ✅ Validados vs Voyager 2 reconstruido (no existe PDS in-situ); factor típico 1.1-1.2× | Termosfera (h>300 km): factor 2 |

---

## 6. Verificación del modelo

Se han realizado **cinco tests de consistencia y validación**:

### 6.1 Test interno: continuidad entre capas

Tras recalcular las H, **todos los 38 límites de capas** de los 7 planetas con atmósfera tienen continuidad de ρ con error **< 0.01%**. Ver tabla detallada en cada `.md` de planeta.

### 6.2 Test de Cassini (Saturno) — consistencia de drag balístico

Con los **parámetros reales** de Cassini (2150 kg, 12.6 m², C_d=2.2) frente a un satélite **genérico** (100 kg, 1 m², C_d=2.2) a 1600 km:

| Satélite | C_d·A/m | Reentrada simulada |
|---|---|---|
| Genérico | 0.022 | 4 días |
| Cassini  | 0.013 | 6.8 días |

**Ratio temporal**: 6.8 / 4 = **1.7×**. **Ratio teórico** (inverso del Cd·A/m): 0.022 / 0.013 = **1.7×**. ✅

### 6.3 Validación cuantitativa con NASA-PDS — Júpiter

Comparación contra los **693 puntos in-situ del Galileo Probe ASI** (dataset oficial PDS, DOI:10.17189/tfsa-pb91):

| Rango altitud (km) | Factor mediano (modelo/Galileo) |
|---|---|
| 0 – 100 | 1.15× |
| 100 – 300 | 1.45× |
| 300 – 600 | 1.35× |
| 600 – 1029 | 1.36× |

**Acuerdo global**: factor 1.32× sobre 431 puntos (datos h ≥ 0). RMS factor 1.45×. Todos los rangos dentro de factor 2×.

> Nota: la versión inicial del modelo (basada en lecturas aproximadas de figuras de Seiff 1998) sobreestimaba la densidad termosférica por **factor ~10×**. Tras la recalibración con datos PDS directos, el error bajó a factor < 2×.

### 6.4 Validación cuantitativa con NASA-PDS — Saturno

Comparación contra los **763 puntos in-situ del Cassini Grand Finale UVIS** (DOI:10.17189/518e-p721, Koskinen et al. 2019):

| Rango (km) | Factor mediano |
|---|---|
| 600 – 900 | 1.07× |
| 900 – 1200 | 1.23× |
| 1200 – 1500 | 1.13× |
| 1500 – 1800 | 0.97× |
| 1800 – 2500 | 0.98× |

**Acuerdo global**: factor 1.056× (5.6% de diferencia con Cassini). La dispersión RMS (~2×) refleja **variabilidad latitudinal real** (Cassini cruzó -85° a +80°; las regiones polares aurorales están sistemáticamente más densas).

### 6.5 Validación del Bloque 2: regresión nodal contra fórmula de Brouwer (1959)

Para la ISS (h=420 km, i=51.6°, 30 días), comparación entre la regresión nodal numérica del simulador y la solución analítica clásica:

$$\frac{d\Omega}{dt} = -\frac{3}{2} J_2 \, n \, \left(\frac{R}{p}\right)^2 \frac{\cos(i)}{(1-e^2)^2}$$

| Caso | ΔRAAN simulada | ΔRAAN Brouwer | Discrepancia |
|---|---|---|---|
| ISS (i=51.6°) | -149.3° | -148.5° | 0.8° (acoplamiento drag-J₂) |
| Luna baja (i=30°) | -31.2° | -31.2° | 0.02° |
| Mercurio (i=45°) | -7.4° | -7.4° | 0.003° |
| Tierra polar (i=90°) | 0.00° | 0.00° | 0° |
| Tierra retrógrada (i=120°) | +115.3° | +114.8° | 0.6° |

✅ Coincidencia con Brouwer dentro del 1% en todos los casos. La pequeña discrepancia en órbitas con drag corresponde al **acoplamiento drag-J₂** (al bajar `a` cambia `n` y la precesión se acelera ligeramente).

---

## 7. Ecuaciones para incluir en la memoria

Resumen compacto de las ecuaciones clave:

$$\text{Energía orbital:} \quad E_{orb} = -\frac{G M m}{2r}$$

$$\text{Velocidad circular:} \quad v = \sqrt{\frac{G M}{r}}$$

$$\text{Fuerza de drag:} \quad F_{drag} = \frac{1}{2} \rho(h) \, v^2 \, C_d \, A$$

$$\text{Pérdida de energía:} \quad \frac{dE}{dt} = -F_{drag} \cdot v$$

$$\text{Modelo atmosférico:} \quad \rho(h) = \rho_{base,i} \cdot e^{-(h - h_{min,i})/H_i}$$

$$\text{Continuidad entre capas:} \quad H_i = \frac{h_{min,i-1} - h_{min,i}}{\ln(\rho_{base,i} / \rho_{base,i-1})}$$

$$\text{Decay del radio:} \quad \frac{dr}{dt} = -\rho \, v \, C_d \, A \cdot \frac{r}{m}$$

$$\text{Vida orbital característica:} \quad \tau \approx \frac{H}{|dr/dt|}$$

(La última fórmula es útil para estimaciones de orden de magnitud sin integrar.)
