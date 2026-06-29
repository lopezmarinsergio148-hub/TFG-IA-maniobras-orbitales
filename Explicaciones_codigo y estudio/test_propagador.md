# Explicación: `tests/test_propagador.py`

> Suite de pruebas automáticas del propagador orbital J2 + drag.
> Comprueba que `Pruebas_perturbaciones_optimizado.py` funciona bien en los
> **9 cuerpos** del modelo, sin intervención manual.

---

## 1. ¿Qué es y para qué sirve?

Es un **test de regresión**: un script que ejecuta el propagador en condiciones
controladas y verifica, con criterios objetivos de PASS/FALLO, que todo se
comporta como debe. Sirve para responder de un vistazo a la pregunta *"¿he roto
algo?"* cada vez que se toca el propagador o el modelo de densidades.

No introduce física nueva ni valida contra datos reales (eso es trabajo de
`datos_validacion/`). Solo comprueba la **salud del código**: que no peta, que no
da valores absurdos (NaN, infinitos, divergencias), que detecta reentradas y que
los cuerpos sin atmósfera ignoran el drag.

---

## 2. Cómo ejecutarlo

```bash
python tests/test_propagador.py
```

- Tarda **~1-2 minutos** (hace ~27 propagaciones reales con `CowellPropagator`).
- Imprime el resultado de cada check por cuerpo y un resumen final.
- **Código de salida**: `0` si todo pasa, `1` si algún check falla
  (útil para integración continua / scripts automáticos).

Salida esperada al final:
```
RESULTADO: ✅ TODO OK — 54 checks superados en 9 cuerpos.
```

---

## 3. Cómo funciona por dentro

El script **no usa el menú interactivo** de `Pruebas_perturbaciones_optimizado.py`.
En su lugar **importa sus funciones internas** y las llama directamente con
parámetros fijos. Tres detalles técnicos lo hacen posible:

1. **`matplotlib.use("Agg")`** antes de importar nada: backend sin ventana, para
   que no intente abrir gráficas.
2. **`sys.path.insert(0, RAIZ)`**: añade la carpeta padre al path para poder
   importar los módulos del proyecto desde dentro de `tests/`.
3. Llama a `trayectoria_perturbada(...)` y `trayectoria_kepler(...)`
   directamente (las funciones que el menú usa por debajo), construyendo la
   órbita con `Orbit.circular(planeta.body, h*u.km, inc=...)`, igual que el menú.

Las advertencias de poliastro/numpy se silencian con `warnings.catch_warnings()`
para que la salida quede limpia.

---

## 4. Qué cuerpos recorre

En este orden (con atmósfera primero, sin atmósfera al final):

```
Tierra · Marte · Venus · Júpiter · Saturno · Urano · Neptuno · Luna · Mercurio
```

El **satélite de prueba** es genérico para todos:
`masa = 500 kg`, `área = 10 m²`, `Cd = 2.2` → coeficiente balístico A/m ≈ 0.02 m²/kg
(un satélite "normal", ni paracaídas ni bala).

---

## 5. Los checks (criterio de PASS/FALLO)

Cada cuerpo pasa por 6 checks (5 si no tiene atmósfera):

| Check | Qué verifica | Criterio de PASS |
|---|---|---|
| **radio R_m==poliastro** | que la altura que pides coincide con la que usa la densidad | \|R_modelo − R_poliastro\| < 1 km |
| **densidad finita+monótona** *(con atm)* | `get_rho` bien definida en todo el rango | 25 muestras de 0 a la capa superior, todas finitas, ≥0 y decrecientes |
| **sin atmósfera → ρ=0** *(sin atm)* | Luna/Mercurio no tienen aire | `get_rho(100 km) == 0` |
| **Kepler ideal plano** | la órbita SIN perturbaciones no se mueve | variación de altura < 1 km en 3 días |
| **propaga alta sin crash+finito** | el integrador Cowell+J2+drag corre limpio | sin excepción y todos los valores finitos |
| **órbita alta acotada (no diverge)** | no hay explosión numérica | ningún punto sube por encima de `h₀ + max(200 km, 5% de r)` |
| **drag decae/reentra** *(con atm)* | el arrastre realmente quita energía | la órbita reentra **o** pierde >5 km de altura |
| **airless: estable solo J2** *(sin atm)* | sin aire NO se reentra por drag | no reentra, finito y \|h_final − h₀\| < 50 km |

> La cota del check "no diverge" (un 5% del radio) es **generosa a propósito**:
> tolera la oscilación de la altitud osculadora por J2 (que en los gigantes
> llega a ~100 km), pero atrapa cualquier divergencia numérica real.

---

## 6. Cómo elige las altitudes de prueba

Para cada cuerpo se prueban dos órbitas, derivadas automáticamente de su ficha:

- **Órbita alta** (`h_alta`): la altura de la **capa superior** del modelo
  (`capas[0].h_min_km`), donde la densidad es mínima → órbita lo más estable
  posible. Para cuerpos sin atmósfera: 200 km fijos.
- **Órbita baja** (`h_baja`): cerca del límite de reentrada
  (`h_reentrada + max(50 km, 30%)`), donde el drag es fuerte → fuerza el
  decaimiento/reentrada. Para cuerpos sin atmósfera: 100 km fijos.

Esto hace que la suite se **adapte sola** a cada planeta sin números a mano: en
la Tierra prueba a 500/150 km, en Neptuno a 4000/650 km, etc.

---

## 7. Cómo interpretar un fallo

Si un check da `✗ FALLO`, el detalle a su derecha dice por qué. Casos típicos:

- **`CRASH ...`** → el integrador lanzó una excepción no esperada. Mirar el tipo
  de error; suele ser un problema de unidades o de un parámetro fuera de rango.
- **densidad no monótona / no finita** → algún `ρ_base` o `H` mal puesto en las
  capas (romper la continuidad o meter un valor negativo).
- **Kepler no plano** → algo perturba la órbita "ideal" cuando no debería
  (la trayectoria Kepler no debe llevar J2 ni drag).
- **drag no decae** → el arrastre no está actuando (densidad nula donde no toca,
  o `incluir_drag` desactivado por error).
- **airless inestable / reentra** → un cuerpo sin atmósfera está sintiendo drag
  (no debería) o el J2 lo desestabiliza.

---

## 8. Notas y limitaciones

- **No es validación física**: que un cuerpo reentre rápido (p. ej. Venus o los
  gigantes incluso desde su capa más alta) es correcto y esperado — sus
  atmósferas son densas. La suite solo comprueba que el **código** lo maneja bien.
- El caso de **reentrada catastrófica de 1 punto** (gráfica vacía) se considera
  un PASS: el código lo gestiona sin romperse, aunque la gráfica salga vacía.
- Es deliberadamente **rápida y tolerante**: usa pocos pasos y cotas generosas,
  porque su objetivo es detectar *roturas*, no medir precisión.

---

## 9. Cómo extenderla

- **Añadir un cuerpo**: incluir su clave en la lista `ORDEN` (debe existir en
  `PLANETAS`). Las altitudes y checks se aplican solos.
- **Añadir un check**: dentro de `test_cuerpo(clave)`, añadir una tupla
  `(nombre, ok_bool, detalle_str)` a la lista `checks`. El resumen y el código de
  salida la recogen automáticamente.
