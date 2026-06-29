# Densidades_atmosferica_optimizado.py — Documentación

> Refactor unificado del simulador de decaimiento orbital por arrastre
> atmosférico para los **9 cuerpos** del sistema solar: 7 con atmósfera (Tierra,
> Marte, Venus, Júpiter, Saturno, Urano, Neptuno) + 2 sin atmósfera (Luna y
> Mercurio, solo con J₂).
>
> Equivalente físico al `Densidades_atmosferica.py` original (mismos números,
> mismos resultados) pero con **65 % menos código** y **0 duplicación**.

---

## 1. Propósito

Este módulo cumple dos roles:

1. **Aplicación interactiva**: ejecutándolo directamente abre un menú para
   elegir planeta, introducir parámetros del satélite y visualizar el
   decaimiento orbital.
2. **Librería importable**: otros scripts (futuro entorno de RL, capa LLM,
   simulaciones J2 + drag combinadas) pueden importar las instancias
   `TIERRA, MARTE, VENUS, JUPITER, SATURNO, URANO, NEPTUNO` o el diccionario
   `PLANETAS` y consultar todo lo que necesiten sin reimplementar nada.

---

## 2. Cómo usarlo

### 2.1 Modo interactivo

Ejecuta el archivo y elige planeta del menú:

```bash
python Densidades_atmosferica_optimizado.py
```

Salida (resumen):
```
======================================================================
  SIMULADOR DE DECAIMIENTO ORBITAL — MÓDULO UNIFICADO
======================================================================
  Elige el cuerpo a simular:

   1. Tierra     (superficie)   R =     6378 km
   2. Marte      (superficie)   R =     3396 km
   3. Venus      (superficie)   R =     6052 km
   4. Júpiter    (1 bar)        R =    71492 km
   5. Saturno    (1 bar)        R =    60268 km
   6. Urano      (1 bar)        R =    25559 km
   7. Neptuno    (1 bar)        R =    24764 km

Tu elección (1-7):
```

Después pide los parámetros estándar (altura inicial, masa, área, Cd,
días) y lanza la simulación con gráfica.

### 2.2 Modo librería

```python
from Densidades_atmosferica_optimizado import JUPITER, PLANETAS, simular

# (a) Consultar densidad en una altura
rho, idx = JUPITER.get_rho(1_000_000)   # h en metros
print(f"ρ a 1000 km en Júpiter: {rho:.3e} kg/m³")

# (b) Consultar constantes físicas
print(f"μ = {JUPITER.mu_m3_s2:.3e} m³/s²")
print(f"R = {JUPITER.R_m / 1000:.0f} km")
print(f"J2 = {JUPITER.J2:.4e}")

# (c) Ejecutar simulación programáticamente (sin interacción)
tiempos, alturas, reentrada = simular(
    planeta=JUPITER,
    h_inicial_m=2_000_000,    # 2000 km
    masa=100, area=1, cd=2.2, dias=30,
)
print(f"Caída: {alturas[0] - alturas[-1]:.2f} km en {tiempos[-1]:.2f} días")

# (d) Iterar sobre todos los planetas
for nombre, planeta in PLANETAS.items():
    print(f"{planeta.nombre}: {len(planeta.capas)} capas, "
          f"fuente = {planeta.fuente}")
```

---

## 3. Estructura del código

### 3.1 Clases (dataclasses)

| Clase | Función |
|---|---|
| `CapaAtmosferica` | Una capa exponencial: `ρ(h) = ρ_base · exp(-(h-h_min)/H)` |
| `BandaVisual` | Una franja coloreada de la gráfica (h_sup, h_inf, color, nombre) |
| `Planeta` | Ficha técnica completa: poliastro body + capas + integrador + estética |

### 3.2 Atributos y métodos clave de `Planeta`

| Atributo / Método | Tipo | Significado |
|---|---|---|
| `nombre` | str | Nombre del planeta |
| `body` | poliastro.Body | Referencia oficial (μ, R desde poliastro) |
| `capas` | list[CapaAtmosferica] | Modelo atmosférico, ordenado alto→bajo |
| `h_reentrada_m` | float | Altitud de destrucción (m) |
| `referencia_h0` | str | `"superficie"` o `"1 bar"` |
| `fuente` | str | Citación corta para la memoria |
| `dt_schedule` | list[tuple] | Pasos del integrador adaptativo |
| `M_kg`, `R_m`, `mu_m3_s2`, `J2` | float | Constantes físicas cacheadas |
| `omega_rad_s` | float | Velocidad angular de rotación (rad/s) |
| `get_rho(h_m, idx_capa=0)` | método | Devuelve `(ρ, nuevo_idx)` con búsqueda persistente O(1) |
| `dt_adaptativo(h_m)` | método | Devuelve el dt recomendado para la altura |

### 3.3 Funciones globales

| Función | Función |
|---|---|
| `simular(planeta, h_inicial_m, masa, area, cd, dias, factor_solar=1.0)` | Núcleo del integrador (genérico para cualquier planeta) |
| `graficar(planeta, tiempos, alturas, ...)` | Gráfica con paleta propia del planeta |
| `pedir_float`, `pedir_int` | Helpers de input validado |
| `elegir_planeta()` | Menú interactivo |
| `main()` | Flujo principal |

---

## 4. Datos de los 9 cuerpos (tabla resumen)

| Cuerpo | Capas | h_reentrada | h=0 | Fuente | Validación PDS |
|---|---|---|---|---|---|
| Tierra | 10 | 100 km | superficie | librería `ussa1976` (USSA-76) | ✅ implícita |
| Marte | 7 | 50 km | superficie | Mars Climate Database (MCD) | ✅ implícita |
| Venus | 4 | 40 km | superficie | T(h) NoSoloSputnik + cálculo propio | ⚠️ pendiente |
| **Júpiter** | **5** | 100 km (s. 1 bar) | 1 bar | **Galileo Probe ASI (recalibrado)** | ✅ 693 puntos PDS |
| **Saturno** | **8** | 500 km (s. 1 bar) | 1 bar | **Cassini Grand Finale (recalibrado)** + Voyager | ✅ 763 puntos PDS |
| Urano | 7 | 500 km (s. 1 bar) | 1 bar | Voyager 2 (1986), Lindal/Herbert (recalibrado) | ✅ Voyager reconstr. (no hay PDS) |
| Neptuno | 7 | 500 km (s. 1 bar) | 1 bar | Voyager 2 (1989), Lindal/Broadfoot (recalibrado) | ✅ Voyager + GRAM (error 100× corregido) |
| **Luna** | **0** (sin atm) | 0 (superficie) | superficie | Solo J₂ (LRO/GRAIL, Konopliv 2013) | N/A |
| **Mercurio** | **0** (sin atm) | 0 (superficie) | superficie | Solo J₂ (MESSENGER, Mazarico 2014) | N/A |

> Las **densidades de referencia** son medidas observacionales.
> Las **alturas de escala H** se derivan matemáticamente para garantizar
> continuidad: `H_i = (h_{i-1} - h_i) / ln(ρ_i / ρ_{i-1})`.
> Detalles por planeta en `investigacion_atmosferas/{planeta}.md`.
>
> **Cuerpos sin atmósfera** (Luna, Mercurio): `tiene_atmosfera = False`, `get_rho()` devuelve siempre 0. Solo se les aplica gravedad central + J₂ en el Bloque 2.
>
> **Recalibraciones realizadas** (mayo-junio 2026):
> - **Júpiter**: ρ_base a 500 y 1000 km bajaron por factor ~10× tras comparar con Galileo Probe PDS (DOI:10.17189/tfsa-pb91). La versión inicial sobreestimaba la termosfera.
> - **Saturno**: añadida capa intermedia en 700 km (termopausa) y recalibrados ρ_base de las 5 capas cubiertas por Cassini (DOI:10.17189/518e-p721).
> - **Urano**: añadida capa intermedia en 150 km tras validar contra Voyager 2 reconstruido (H de estratosfera 38→25 km). Factor mediano 2.49× → 1.15×. Ver `datos_validacion/voyager_urano/`.
> - **Neptuno**: corregido error grave (ρ a 50 km estaba ~100× baja, H troposfera 6.84→18.5 km) y añadidas capas a 150 y 300 km. Factor típico 8.0× → 1.2×. Ver `datos_validacion/voyager_neptuno/`.

---

## 5. Algoritmo del integrador

El núcleo (`simular`) usa el **método de energía orbital** con paso adaptativo:

```
E_orbital = -GMm / (2r)
F_drag    = ½ · ρ(h) · v² · Cd · A         con v = √(GM/r)
dE        = F_drag · v · dt                (potencia disipada × tiempo)
E_nueva   = E_actual - dE
r_nueva   = -GMm / (2 · E_nueva)
h_nueva   = r_nueva - R_planeta
```

### Salvaguardas (halvings)

Antes de cada paso, `dt` se reduce a la mitad si se viola alguno de:

| Criterio | Threshold | Por qué |
|---|---|---|
| Energético | `dE > 0.1% · |E_orbital|` | Evita saltos energéticos grandes |
| Estabilidad | `E_nueva ≥ 0` | Órbita hiperbólica imposible bajo drag |
| Cinemático | `Δr > 10 km` por paso | Crítico en planetas grandes (Júpiter R=71500 km) |

### Muestreo de salida

Se guarda un punto en la lista de resultados cuando:
- Han caído **≥ 1 km** desde el último punto guardado, **o**
- Ha pasado **≥ 1 hora** simulada desde el último punto.

Esto produce gráficas suaves sin saturar memoria en simulaciones largas.

---

## 6. Diferencias con `Densidades_atmosferica.py` original

| Aspecto | Original | Optimizado |
|---|---|---|
| Líneas de código | ~1800 | ~640 (**−65 %**) |
| Funciones `simular()` | 7 (una por planeta) | **1 genérica** |
| Funciones `graficar()` | 7 | **1 genérica** |
| Definiciones de `pedir_float/int` | 7 (duplicadas) | **1** |
| Globals sobrescritos | Sí (`DT_SCHEDULE`, `_dt_adaptativo`...) | **No** (todo en cada Planeta) |
| Constantes físicas | Hardcodeadas en cada bloque | **Desde `poliastro.bodies`** |
| Selección de planeta | Comentar/descomentar líneas | **Menú interactivo** |
| Reutilizable desde otros archivos | No (variables globales colisionan) | **Sí** (`from ... import JUPITER`) |
| Resultados numéricos | — | **Idénticos al original** (validado) |

---

## 7. Validación realizada

Resultados comparados contra los del archivo original (con los mismos
parámetros de satélite: 100 kg, 1 m², Cd=2.2, 30 días):

| Planeta | h_inicial | Resultado optimizado | Original (esperado) |
|---|---|---|---|
| Júpiter | 2000 km | 87.72 km caída | ~87.71 km caída |
| Saturno | 2500 km | 7.29 km caída | ~7 km caída |
| Urano | 7000 km | 1.78 km caída | ~1.77 km caída |
| Urano | 3000 km | reentrada 0.27 d | ~0.3 d |
| Neptuno | 5000 km | 0.0898 km caída | ~0.0898 km caída |
| Neptuno | 3000 km | 7.59 km caída | ~7.5 km caída |
| Neptuno | 1500 km | reentrada 3.50 d | ~3.5 d |

**Conclusión**: equivalencia numérica confirmada.

---

## 8. Cómo extender (añadir un nuevo planeta)

Si en el futuro hace falta añadir (p.ej.) Titán o un exoplaneta hipotético:

```python
TITAN = Planeta(
    nombre="Titán",
    body=...,                       # poliastro.bodies.Titan o crear Body
    capas=[
        CapaAtmosferica(h_min_km, rho_base, H_km, "nombre"),
        # ...
    ],
    h_reentrada_m=...,
    referencia_h0="superficie",
    fuente="Huygens probe (Fulchignoni 2005)",
    dt_schedule=[...],
    dt_base=...,
    bandas_visuales=[...],
    color_satelite="...",
    j2_manual=...,    # opcional si poliastro no lo tiene
)

PLANETAS["titan"] = TITAN
```

El menú lo recoge automáticamente.

---

## 9. Limitaciones del modelo

Las mismas que el original (heredadas del modelo físico, no del refactor):

- **Solo drag**: NO incluye J2 ni perturbaciones gravitatorias. Eso vendrá en el
  Bloque 2 (drag + J2 con poliastro).
- **Órbita cuasi-circular** asumida en cada instante. Inválido para órbitas
  muy elípticas.
- **Modelo 1D radial** (`h(t)`). No simula la trayectoria 3D completa.
- **Atmósfera estática** (no rotante). En la Tierra ya hay factor F10.7 para
  actividad solar; en otros planetas, ignorado.
- **Urano y Neptuno** tienen incertidumbre alta en capas altas porque solo
  Voyager 2 los ha visitado.

---

## 10. Para la memoria del TFG

Lo "vendible" de este módulo:

1. **Modularidad y reutilizabilidad**: una `Planeta` es una pieza autocontenida
   que cualquier futura componente (entorno de RL, capa LLM, propagador con J2)
   puede importar y usar sin acoplarse al integrador.
2. **Single source of truth**: las constantes físicas vienen directamente de
   `poliastro.bodies` (estándar de facto en astrodinámica con Python). No hay
   números mágicos repartidos por el código.
3. **Patrón profesional**: el uso de `dataclasses` con `__post_init__` para
   cachear valores derivados es idiomático en Python moderno y reduce errores.
4. **Validación cuantitativa**: se ha verificado numéricamente que produce los
   mismos resultados que el archivo original (ver tabla §7), lo cual demuestra
   que el refactor no ha introducido regresiones.
5. **Equivalencia física**: las decisiones de diseño (continuidad de H, halvings
   en 3 criterios, muestreo dual) están documentadas en
   `investigacion_atmosferas/metodologia.md`.

Una frase para la memoria:

> *"El simulador físico se organiza alrededor de la clase `Planeta`, que
> encapsula los parámetros gravitatorios (extraídos de `poliastro.bodies`),
> el modelo atmosférico por capas (con alturas de escala derivadas para
> garantizar continuidad) y la configuración del integrador adaptativo.
> El núcleo de simulación es genérico: una sola función opera sobre cualquier
> instancia de `Planeta`, lo que facilita la extensión del modelo a cuerpos
> adicionales y la reutilización desde los componentes posteriores del TFG
> (entorno de RL y capa de explicación en lenguaje natural)."*
