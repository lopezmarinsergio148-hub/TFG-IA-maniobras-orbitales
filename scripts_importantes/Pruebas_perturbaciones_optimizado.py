"""
═══════════════════════════════════════════════════════════════════════════════
 PROPAGADOR ORBITAL CON PERTURBACIONES — J2 + DRAG UNIFICADO
 Propagación realista de órbitas alrededor de los 7 planetas del sistema solar
 que tienen atmósfera y 2 cuerpos sin atmósfera como Mercurio y la Luna, combinando:
   - Atractor central (gravedad newtoniana)            → poliastro.func_twobody
   - Achatamiento J2                                   → poliastro.J2_perturbation
   - Arrastre atmosférico con MODELO DE CAPAS PROPIO   → Densidades_atmosferica_optimizado

 El núcleo es UNA SOLA función `propagar_perturbado` que funciona para cualquier
 planeta sin código duplicado, gracias a la abstracción `Planeta` del módulo
 unificado de densidades.

 USO RÁPIDO:
     python Pruebas_perturbaciones_optimizado.py     # menú interactivo

 USO PROGRAMÁTICO:
     from poliastro.twobody import Orbit
     from astropy import units as u
     from Densidades_atmosferica_optimizado import TIERRA
     from Pruebas_perturbaciones_optimizado import propagar_perturbado

     orb = Orbit.circular(TIERRA.body, 400 * u.km, inc=51.6 * u.deg)
     orb_final = propagar_perturbado(TIERRA, orb, 30 * u.day,
                                     masa=420_000, area=2500, cd=2.2)
═══════════════════════════════════════════════════════════════════════════════
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from astropy import units as u
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator
from poliastro.core.perturbations import J2_perturbation
from poliastro.core.propagation import func_twobody

from Densidades_atmosferica_optimizado import (
    PLANETAS, Planeta, pedir_float, pedir_int, FACTORES_SOLAR
)

# Encoding UTF-8 (para que los emojis funcionen en cmd.exe / PowerShell)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Plotly abre las gráficas en el navegador
pio.renderers.default = "browser"


# ═══════════════════════════════════════════════════════════════════════════
# 1. FUNCIÓN DE PERTURBACIÓN UNIFICADA (compatible con CowellPropagator)
# ═══════════════════════════════════════════════════════════════════════════

def crear_funcion_perturbacion(
    planeta: Planeta,
    masa: float,
    area: float,
    cd: float,
    factor_solar: float = 1.0,
    incluir_J2: bool = True,
    incluir_drag: bool = True,
    atmosfera_rotante: bool = True,
):
    # Auto-desactivar drag si el cuerpo no tiene atmósfera (Luna, Mercurio)
    if incluir_drag and not planeta.tiene_atmosfera:
        incluir_drag = False
    """
    Construye y devuelve la función f(t0, state, k) que CowellPropagator
    necesita para integrar la trayectoria con perturbaciones.

    Convenciones de unidades (las de poliastro):
      - state[0:3] = posición en km
      - state[3:6] = velocidad en km/s
      - k          = GM en km³/s²
      - aceleraciones devueltas en km/s²

    Parámetros
    ----------
    planeta            : instancia de `Planeta` (con su modelo atmosférico)
    masa, area, cd     : parámetros del satélite (kg, m², adimensional)
    factor_solar       : multiplicador de densidad (solo Tierra, F10.7)
    incluir_J2         : si True, añade el término J2 (achatamiento)
    incluir_drag       : si True, añade el término de arrastre atmosférico
    atmosfera_rotante  : si True, la atmósfera gira con ω del planeta
    """
    # Precomputamos todo lo que no varía dentro del integrador
    R_km           = planeta.R_m / 1000.0
    J2_val         = planeta.J2
    h_reentrada_km = planeta.h_reentrada_m / 1000.0
    A_over_m       = area / masa                                # m²/kg
    omega          = planeta.omega_rad_s if atmosfera_rotante else 0.0

    # Captura del Planeta como closure (necesario para get_rho)
    planeta_ref = planeta

    def f(t0, state, k):
        # ── Parte Kepleriana (gravedad central) ────────────────────────
        du = func_twobody(t0, state, k)
        ax = ay = az = 0.0

        # ── Perturbación J2 ────────────────────────────────────────────
        if incluir_J2 and J2_val > 0:
            ax_j2, ay_j2, az_j2 = J2_perturbation(t0, state, k, J2_val, R_km)
            ax += ax_j2; ay += ay_j2; az += az_j2

        # ── Drag con modelo de capas propio ────────────────────────────
        if incluir_drag:
            x, y, z, vx, vy, vz = state
            r_km = np.sqrt(x*x + y*y + z*z)
            h_km = r_km - R_km

            if h_km > h_reentrada_km:
                # Densidad en kg/m³ (búsqueda completa desde capa 0; es O(N≤10))
                rho_kg_m3, _ = planeta_ref.get_rho(h_km * 1000.0, 0)
                rho_kg_m3 *= factor_solar

                # Velocidad relativa al aire: v_rel = v_sat - ω × r
                # Atmósfera rota con ω alrededor del eje z (aprox. ecuatorial)
                v_atm_x = -omega * (y * 1000.0)        # m/s
                v_atm_y =  omega * (x * 1000.0)
                v_atm_z = 0.0

                v_rel_x = vx * 1000.0 - v_atm_x         # m/s
                v_rel_y = vy * 1000.0 - v_atm_y
                v_rel_z = vz * 1000.0 - v_atm_z

                v_rel_norm = np.sqrt(v_rel_x*v_rel_x +
                                      v_rel_y*v_rel_y +
                                      v_rel_z*v_rel_z)

                # a_drag = -0.5 · ρ · Cd · (A/m) · |v_rel| · v_rel
                fac = -0.5 * rho_kg_m3 * cd * A_over_m * v_rel_norm  # 1/s

                # Convertir a km/s² (dividiendo por 1000 al pasar de m/s²)
                ax += fac * v_rel_x / 1000.0
                ay += fac * v_rel_y / 1000.0
                az += fac * v_rel_z / 1000.0

        return du + np.array([0.0, 0.0, 0.0, ax, ay, az])

    return f


# ═══════════════════════════════════════════════════════════════════════════
# 2. PROPAGACIÓN (interfaz pública)
# ═══════════════════════════════════════════════════════════════════════════

def propagar_perturbado(
    planeta: Planeta,
    orb_inicial: Orbit,
    tiempo,
    masa: float, area: float, cd: float,
    factor_solar: float = 1.0,
    incluir_J2: bool = True,
    incluir_drag: bool = True,
    atmosfera_rotante: bool = True,
) -> Orbit:
    """
    Propaga `orb_inicial` durante `tiempo` con J2 + drag y devuelve la órbita
    final. `tiempo` debe ser una Quantity con dimensiones de tiempo
    (ej. `30 * u.day`).
    """
    f_pert = crear_funcion_perturbacion(
        planeta, masa, area, cd, factor_solar,
        incluir_J2, incluir_drag, atmosfera_rotante,
    )
    propagador = CowellPropagator(f=f_pert)
    return orb_inicial.propagate(tiempo, method=propagador)


def trayectoria_perturbada(
    planeta: Planeta,
    orb_inicial: Orbit,
    tiempo_total,
    num_pasos: int,
    masa: float, area: float, cd: float,
    factor_solar: float = 1.0,
    incluir_J2: bool = True,
    incluir_drag: bool = True,
    atmosfera_rotante: bool = True,
):
    """
    Propaga incrementalmente desde t=0 hasta `tiempo_total` muestreando
    `num_pasos` puntos intermedios.

    Retorna
    -------
    tiempos_dias : np.ndarray  — (num_pasos+1,)
    posiciones   : np.ndarray  — (num_pasos+1, 3) en km
    alturas_km   : np.ndarray  — (num_pasos+1,)
    orb_final    : poliastro.Orbit — estado final tras la propagación completa
    """
    f_pert = crear_funcion_perturbacion(
        planeta, masa, area, cd, factor_solar,
        incluir_J2, incluir_drag, atmosfera_rotante,
    )
    propagador = CowellPropagator(f=f_pert)

    R_km = planeta.R_m / 1000.0
    dt   = tiempo_total / num_pasos

    pos0  = orb_inicial.r.to(u.km).value
    tiempos_dias = [0.0]
    posiciones   = [pos0]
    alturas_km   = [float(np.linalg.norm(pos0) - R_km)]

    hubo_reentrada = False
    orb_actual = orb_inicial
    for i in range(1, num_pasos + 1):
        # Intentar propagar; si el integrador falla (reentrada catastrófica
        # con cambios de densidad demasiado bruscos), salimos limpiamente.
        try:
            orb_nuevo = orb_actual.propagate(dt, method=propagador)
        except RuntimeError:
            t_fallo = (i * dt).to(u.day).value
            print()
            print(f"  🔥 REENTRADA CATASTRÓFICA en t ≈ {t_fallo:.2f} días "
                  f"(paso {i}/{num_pasos}).")
            print(f"     El satélite ha atravesado capas atmosféricas densas")
            print(f"     más rápido de lo que el integrador puede resolver.")
            print(f"     Última altura registrada: {alturas_km[-1]:.2f} km.")
            print(f"     Se devuelve la trayectoria parcial hasta el momento del fallo.")
            print()
            hubo_reentrada = True
            break

        orb_actual = orb_nuevo
        pos        = orb_actual.r.to(u.km).value
        h_actual   = float(np.linalg.norm(pos) - R_km)
        t_paso     = (i * dt).to(u.day).value

        # Detección suave de reentrada: si h cae por debajo del límite del
        # planeta, paramos ANTES de añadir el punto (puede tener un valor
        # absurdo si el integrador hizo un salto grande).
        if h_actual * 1000.0 <= planeta.h_reentrada_m:
            # En vez del valor absurdo, anclamos el punto final al límite de
            # reentrada para que las gráficas y el resumen sean realistas.
            h_clamp = planeta.h_reentrada_m / 1000.0
            tiempos_dias.append(t_paso)
            posiciones.append(pos)
            alturas_km.append(h_clamp)
            print()
            print(f"  🔥 REENTRADA: el satélite ha cruzado el límite de "
                  f"{planeta.h_reentrada_m/1000:.0f} km en t ≈ {t_paso:.2f} días.")
            print(f"     Simulación detenida en h = {h_clamp:.1f} km.")
            print()
            hubo_reentrada = True
            break

        # Caso normal: añadimos el punto y seguimos
        tiempos_dias.append(t_paso)
        posiciones.append(pos)
        alturas_km.append(h_actual)

    return (np.array(tiempos_dias), np.array(posiciones),
            np.array(alturas_km), orb_actual, hubo_reentrada)


def trayectoria_kepler(planeta: Planeta, orb_inicial: Orbit,
                       tiempo_total, num_pasos: int):
    """Idéntico a `trayectoria_perturbada` pero SIN perturbaciones (Kepler puro).
    Sirve de línea de referencia para comparar.

    Retorna `(tiempos_dias, posiciones, alturas_km, orb_final_ideal, hubo_reentrada)`.
    Kepler puro NUNCA tiene reentrada → `hubo_reentrada` es siempre False.
    """
    R_km = planeta.R_m / 1000.0
    dt   = tiempo_total / num_pasos

    pos0 = orb_inicial.r.to(u.km).value
    tiempos_dias = [0.0]
    posiciones   = [pos0]
    alturas_km   = [float(np.linalg.norm(pos0) - R_km)]

    orb_final = orb_inicial
    for i in range(1, num_pasos + 1):
        orb_final = orb_inicial.propagate(i * dt)
        pos       = orb_final.r.to(u.km).value
        tiempos_dias.append((i * dt).to(u.day).value)
        posiciones.append(pos)
        alturas_km.append(float(np.linalg.norm(pos) - R_km))

    return (np.array(tiempos_dias), np.array(posiciones),
            np.array(alturas_km), orb_final, False)


# ═══════════════════════════════════════════════════════════════════════════
# 3. VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def graficar_3d(planeta: Planeta, orb_inicial: Orbit, orb_final: Orbit,
                titulo: str = "", trayectoria_real=None,
                hubo_reentrada: bool = False):
    """
    Visualización 3D limpia: muestra la órbita inicial, la órbita final
    (cada una propagada UN período Kepleriano), el planeta como esfera y
    marcadores del satélite. Incluye desplegable para mostrar/ocultar
    elementos.

    Si HUBO REENTRADA en cualquier momento de la simulación (flag explícito
    de `trayectoria_perturbada` o detección automática por perigeo bajo R),
    NO se dibuja la órbita final — esta dejaría de tener sentido físico.

    Parámetros
    ----------
    hubo_reentrada : bool
        Flag externo: si la simulación rompió por reentrada (RuntimeError o
        h < h_reentrada), `True`. Tiene prioridad sobre cualquier otra
        detección automática.
    trayectoria_real : (ignorado) — mantenido por compatibilidad.
    """
    R_km  = planeta.R_m / 1000.0
    n_pts = 200

    # Propagar la órbita inicial UN período Kepleriano (línea limpia)
    T_ini_s = orb_inicial.period.to(u.s).value
    pos_ini = np.array([
        orb_inicial.propagate(T_ini_s * t * u.s).r.to(u.km).value
        for t in np.linspace(0, 1, n_pts)
    ])

    # ── Decidir si dibujar la "órbita final" ─────────────────────────
    # Dos criterios (cualquiera basta para suprimir el dibujo):
    #   (a) Flag externo: la simulación ya marcó reentrada explícita.
    #   (b) Detección geométrica: perigeo osculador < R_planeta.
    r_per_final_km    = orb_final.r_p.to(u.km).value
    reentrada_geom    = (r_per_final_km < R_km)
    suprimir_orbita_fin = hubo_reentrada or reentrada_geom

    if suprimir_orbita_fin:
        pos_fin    = None
        label_fin  = "Órbita final — REENTRADA (no se dibuja)"
    else:
        T_fin_s = orb_final.period.to(u.s).value
        pos_fin = np.array([
            orb_final.propagate(T_fin_s * t * u.s).r.to(u.km).value
            for t in np.linspace(0, 1, n_pts)
        ])
        label_fin = "Órbita final (tras J2+drag)"

    # Posiciones de los marcadores (satélite al inicio de cada órbita)
    r_sat_ini = orb_inicial.r.to(u.km).value
    r_sat_fin = orb_final.r.to(u.km).value

    fig = go.Figure()

    # ── Traza 0: esfera del planeta ─────────────────────────────────
    u_s = np.linspace(0, 2 * np.pi, 60)
    v_s = np.linspace(0, np.pi, 30)
    xs  = R_km * np.outer(np.cos(u_s), np.sin(v_s))
    ys  = R_km * np.outer(np.sin(u_s), np.sin(v_s))
    zs  = R_km * np.outer(np.ones_like(u_s), np.cos(v_s))
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        colorscale="Blues",
        opacity=0.5, showscale=False,
        name=f"PLANETA_{planeta.nombre}",
        hoverinfo="skip",
    ))

    # ── Traza 1: órbita inicial (1 período Kepler) ──────────────────
    fig.add_trace(go.Scatter3d(
        x=pos_ini[:, 0], y=pos_ini[:, 1], z=pos_ini[:, 2],
        mode="lines", line=dict(color="lime", width=4),
        name="Órbita inicial",
    ))

    # ── Traza 2: órbita final (solo si NO hay reentrada) ────────────
    # Si hay reentrada, añadimos una traza vacía para mantener el orden
    # de índices del desplegable, pero no se ve nada.
    if pos_fin is not None:
        fig.add_trace(go.Scatter3d(
            x=pos_fin[:, 0], y=pos_fin[:, 1], z=pos_fin[:, 2],
            mode="lines", line=dict(color="red", width=4),
            name=label_fin,
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode="lines", line=dict(color="red", width=4),
            name=label_fin,
        ))

    # ── Traza 3: satélite en posición inicial ───────────────────────
    fig.add_trace(go.Scatter3d(
        x=[r_sat_ini[0]], y=[r_sat_ini[1]], z=[r_sat_ini[2]],
        mode="markers", marker=dict(size=8, color="lime"),
        name="SAT_INICIAL",
    ))

    # ── Traza 4: satélite en posición final ─────────────────────────
    # Si hubo reentrada, NO mostramos el marcador final (el satélite ya
    # no está en órbita). Mantenemos la traza vacía para conservar el
    # orden de índices del desplegable.
    if suprimir_orbita_fin:
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode="markers", marker=dict(size=8, color="red"),
            name="SAT_FINAL — REENTRADA (no se dibuja)",
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=[r_sat_fin[0]], y=[r_sat_fin[1]], z=[r_sat_fin[2]],
            mode="markers", marker=dict(size=8, color="red"),
            name="SAT_FINAL",
        ))

    # ── Desplegable: visibilidad fija por índice de traza ───────────
    # Orden: [planeta, orb_ini, orb_fin, sat_ini, sat_fin]
    fig.update_layout(
        title=titulo or f"Órbitas inicial y final — {planeta.nombre}",
        scene=dict(
            xaxis_title="x (km)",
            yaxis_title="y (km)",
            zaxis_title="z (km)",
            aspectmode="data",
        ),
        paper_bgcolor="#0a0a1a",
        font=dict(color="white"),
        updatemenus=[{
            "type": "dropdown",
            "direction": "down",
            "x": 0.05, "y": 1.10,
            "showactive": True,
            "buttons": [
                {"label": "Ver todo",      "method": "update",
                 "args": [{"visible": [True,  True,  True,  True,  True]}]},
                {"label": "Solo órbitas",  "method": "update",
                 "args": [{"visible": [False, True,  True,  False, False]}]},
                {"label": "Sin planeta",   "method": "update",
                 "args": [{"visible": [False, True,  True,  True,  True]}]},
                {"label": "Solo final",    "method": "update",
                 "args": [{"visible": [True,  False, True,  False, True]}]},
                {"label": "Solo inicial",  "method": "update",
                 "args": [{"visible": [True,  True,  False, True,  False]}]},
            ],
        }],
    )
    fig.show()


def graficar_2d_altura(planeta: Planeta, tiempos, h_ideal, h_real, titulo: str = ""):
    """Gráfica 2D: altura vs tiempo (matplotlib)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    ax.plot(tiempos, h_ideal, color="#00ff66", linewidth=2,
            label="Ideal (Kepler)")
    ax.plot(tiempos, h_real,  color="#ff4444", linewidth=2,
            label="Realista (J2 + drag)")

    ax.set_title(titulo or f"Altura vs Tiempo en {planeta.nombre}",
                 color="white", fontsize=13, pad=14)
    ax.set_xlabel("Tiempo (días)", color="white", fontsize=11)
    label_y = ("Altitud sobre 1 bar (km)"
               if planeta.referencia_h0 == "1 bar"
               else "Altitud sobre la superficie (km)")
    ax.set_ylabel(label_y, color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, color="#333", linewidth=0.5, alpha=0.6)
    ax.legend(facecolor="#111", edgecolor="#555", labelcolor="white", fontsize=10)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 4. MENÚ INTERACTIVO Y FLUJO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def elegir_planeta() -> Planeta:
    """Menú: muestra los 9 cuerpos (con marca para los sin atmósfera)."""
    print("=" * 75)
    print("  PROPAGADOR ORBITAL CON PERTURBACIONES (J2 + DRAG)")
    print("=" * 75)
    print("  Elige el cuerpo a simular:\n")
    nombres = list(PLANETAS.keys())
    for i, n in enumerate(nombres, start=1):
        p = PLANETAS[n]
        ref = "(1 bar)" if p.referencia_h0 == "1 bar" else "(superficie)"
        tag = "" if p.tiene_atmosfera else "  ⚠️ sin atm. (sólo J2)"
        print(f"   {i}. {p.nombre:<10s} {ref:<12s}  "
              f"R = {p.R_m/1000:>8.0f} km, J2 = {p.J2:.4e}{tag}")
    print()
    idx = pedir_int(f"Tu elección (1-{len(nombres)}): ",
                    minimo=1, maximo=len(nombres))
    return PLANETAS[nombres[idx - 1]]


def main():
    planeta = elegir_planeta()

    print()
    print("=" * 70)
    print(f"  PLANETA SELECCIONADO: {planeta.nombre.upper()}")
    print(f"  Fuente atm.: {planeta.fuente}")
    print(f"  J2 = {planeta.J2:.4e} | R = {planeta.R_m/1000:.1f} km | "
          f"ω = {planeta.omega_rad_s:.3e} rad/s")
    print("=" * 70)

    # ── Decidir si el cuerpo tiene atmósfera ─────────────────────────
    aplicar_drag = planeta.tiene_atmosfera
    if not aplicar_drag:
        print("  ⚠️  Este cuerpo NO tiene atmósfera modelada.")
        print("      Sólo se aplicará la perturbación J2; el drag se ignora.")
        print("      (No se pedirán masa/área/Cd ni actividad solar.)")
        print()

    # Actividad solar (solo Tierra y sólo si hay drag)
    factor_solar = 1.0
    factor_str   = ""
    if aplicar_drag and planeta.factor_solar_disponible:
        while True:
            act = input("Actividad solar (baja / media / alta): ").strip().lower()
            if act in FACTORES_SOLAR:
                factor_solar = FACTORES_SOLAR[act]
                factor_str   = f" · F10.7: {act}"
                break
            print("  ⚠️  Elige entre: baja, media o alta.")

    # Parámetros orbitales (siempre se piden)
    h_min_input = max(planeta.h_reentrada_m / 1000.0, 50.0)
    h_ini_km = pedir_float(
        f"Altura inicial (km)  [Ej: 400 para ISS] : ",
        minimo=h_min_input, maximo=50_000)
    inc_deg  = pedir_float(
        "Inclinación (grados) [Ej: 51.6 para ISS] : ",
        minimo=0.0, maximo=180.0)

    # Parámetros del satélite (sólo si hay drag)
    if aplicar_drag:
        masa = pedir_float("Masa del satélite (kg)   [Ej: 420000 ISS]: ",
                           minimo=1e-3)
        area = pedir_float("Área frontal (m²)        [Ej: 2500 ISS]  : ",
                           minimo=1e-3)
        cd   = pedir_float("Coeficiente de arrastre  [Ej: 2.2]       : ",
                           minimo=0.1, maximo=10.0)
    else:
        # Valores cualquiera (no se usan en J2 puro)
        masa, area, cd = 100.0, 1.0, 2.2

    dias  = pedir_int("Días de simulación       [Ej: 30]        : ", minimo=1)
    pasos = pedir_int("Pasos de muestreo        [Ej: 200]       : ",
                      minimo=20, maximo=5000)

    # Crear órbita inicial
    orb_inicial = Orbit.circular(
        planeta.body, h_ini_km * u.km, inc=inc_deg * u.deg
    )
    tiempo = dias * u.day

    print(f"\nPropagando {dias} días con {pasos} pasos (esto puede tardar)...")

    t_real, pos_real, h_real, orb_real_final, hubo_reentrada = trayectoria_perturbada(
        planeta, orb_inicial, tiempo, pasos,
        masa=masa, area=area, cd=cd, factor_solar=factor_solar,
    )

    t_ide, pos_ide, h_ide, orb_ideal_final, _ = trayectoria_kepler(
        planeta, orb_inicial, tiempo, pasos
    )

    # ── Si la perturbada se cortó por reentrada, recortamos la ideal ──
    if len(t_real) < len(t_ide):
        print(f"  ⓘ  Trayectoria perturbada parcial: {len(t_real)}/{len(t_ide)} "
              f"puntos antes de la reentrada.")
        t_ide   = t_ide[:len(t_real)]
        pos_ide = pos_ide[:len(t_real)]
        h_ide   = h_ide[:len(t_real)]

    # ── Si la trayectoria es muy corta, saltamos el análisis estadístico ──
    if len(h_real) < 4:
        print()
        print(f"{'─' * 70}")
        print(f"  RESULTADOS PARCIALES  ({planeta.nombre})")
        print(f"{'─' * 70}")
        print(f"  Reentrada muy temprana: solo {len(h_real)} puntos antes del fallo.")
        print(f"  Tiempo simulado : {t_real[-1]:.4f} días")
        print(f"  Altura inicial  : {h_real[0]:.2f} km")
        print(f"  Altura final    : {h_real[-1]:.2f} km")
        print(f"  Sugerencia: aumenta la altura inicial o reduce los días.")
        print(f"{'─' * 70}\n")
        # Mostrar al menos las gráficas con lo que tenemos
        titulo = (
            f"{planeta.nombre} · REENTRADA TEMPRANA · "
            f"h₀={h_ini_km:.0f} km · i={inc_deg:.1f}°"
        )
        graficar_3d(planeta, orb_inicial, orb_real_final, titulo,
                    hubo_reentrada=hubo_reentrada)
        graficar_2d_altura(planeta, t_real, h_ide, h_real, titulo)
        return

    # ── Resumen numérico mejorado ────────────────────────────────────
    #
    # AVISO: los elementos OSCULADORES (a, ecc, h_apo, h_per) oscilan con
    # el período orbital por efecto del J2. Por ejemplo, el semieje a
    # osculador de la ISS oscila ±7 km cada órbita (~90 min).
    #
    # Para distinguir el "decay secular" (causado por drag, monotónico)
    # del "ruido osculador" (causado por J2, oscilatorio), se PROMEDIA
    # la altura en las últimas muestras que cubren ≥10 períodos orbitales.
    # Ese promedio filtra la oscilación J2 y deja solo la tendencia drag.
    #
    R_km   = planeta.R_m / 1000.0
    a_ini  = orb_inicial.a.to(u.km).value
    h_ini  = a_ini - R_km

    # Elementos osculadores finales (un solo instante → oscilan)
    a_fin_osc  = orb_real_final.a.to(u.km).value
    h_apo_osc  = orb_real_final.r_a.to(u.km).value - R_km
    h_per_osc  = orb_real_final.r_p.to(u.km).value - R_km
    ecc_osc    = orb_real_final.ecc.value
    raan_osc   = orb_real_final.raan.to(u.deg).value

    # ── Métrica secular: ajuste lineal a la segunda mitad ─────────────
    # Filtramos las oscilaciones osculadoras de J2 con una regresión
    # lineal h(t) = m·t + b sobre la segunda mitad de los datos.
    #
    # La PENDIENTE (m) es la métrica más limpia: km perdidos por día
    # por puro decaimiento secular (drag). Multiplicada por el tiempo
    # total da los km totales perdidos por drag, sin el offset que
    # introduciría la fase de muestreo de las oscilaciones J2.
    n_mitad = max(len(h_real) // 2, 2)
    slope_real, _intercept_real = np.polyfit(t_real[n_mitad:], h_real[n_mitad:], 1)
    slope_km_dia  = float(slope_real)                 # km/día (negativo = decay)
    decay_secular = -slope_km_dia * t_real[-1]        # km perdidos en t_total

    # ── Regresión nodal: comparar simulada vs fórmula J2 teórica ─────
    # La fórmula clásica de Brouwer:
    #     dΩ/dt = -3/2 · J2 · n · (R/p)² · cos(i) / (1-e²)²
    # con n = sqrt(μ/a³), p = a(1-e²)
    mu_si    = planeta.mu_m3_s2
    a_si     = orb_inicial.a.to(u.m).value
    e_ini    = orb_inicial.ecc.value
    i_rad    = orb_inicial.inc.to(u.rad).value
    R_si     = planeta.R_m
    n_rad_s  = np.sqrt(mu_si / a_si**3)
    p_si     = a_si * (1.0 - e_ini**2) if e_ini < 1.0 else a_si
    if planeta.J2 > 0 and p_si > 0:
        omega_dot_teor = (-1.5 * planeta.J2 * n_rad_s
                          * (R_si / p_si)**2 * np.cos(i_rad)
                          / (1.0 - e_ini**2)**2)
    else:
        omega_dot_teor = 0.0
    t_total_s         = float(t_real[-1] * 86400.0)
    delta_raan_teor   = float(np.degrees(omega_dot_teor * t_total_s))   # ° (signed)

    # Precesión real: unwrap usando la teoría para resolver ambigüedad mod 360
    raan_ini_deg     = orb_inicial.raan.to(u.deg).value
    raan_fin_deg     = orb_real_final.raan.to(u.deg).value
    delta_naive      = ((raan_fin_deg - raan_ini_deg + 180.0) % 360.0) - 180.0
    if t_total_s > 0:
        n_revs       = round((delta_raan_teor - delta_naive) / 360.0)
    else:
        n_revs       = 0
    delta_raan_real  = delta_naive + 360.0 * n_revs
    tasa_real_dia    = delta_raan_real / t_real[-1] if t_real[-1] > 0 else 0.0
    tasa_teor_dia    = delta_raan_teor / t_real[-1] if t_real[-1] > 0 else 0.0

    print(f"\n{'─' * 70}")
    print(f"  RESULTADOS  ({planeta.nombre}, {dias} días)")
    print(f"{'─' * 70}")
    print(f"  ── Estado inicial ───────────────────────────────────")
    print(f"  Semieje mayor a        : {a_ini:>10.3f} km")
    print(f"  Altura (circular)      : {h_ini:>10.3f} km")
    print(f"  Inclinación            : {orb_inicial.inc.to(u.deg).value:>10.3f}°")
    print()
    # Estimación de la amplitud de oscilaciones J2 en la 2ª mitad
    h_2m = h_real[n_mitad:]
    amplitud_J2 = (float(np.max(h_2m)) - float(np.min(h_2m))) / 2.0  # ±km
    if amplitud_J2 > 1e-9:
        ratio_sn = abs(decay_secular) / (2.0 * amplitud_J2)   # decay vs rango total osc
    else:
        ratio_sn = float("inf")

    print(f"  ── Decaimiento secular (efecto del drag, sin osc. J2) ──")
    print(f"  Pendiente del decaimiento : {slope_km_dia:>+10.5f} km/día"
          f"   (regresión lineal 2ª mitad)")
    print(f"  Pérdida de altura         : {abs(decay_secular):>10.4f} km en {dias} días")
    print(f"  Tasa equivalente          : {abs(decay_secular)/dias*30:>10.3f} km/mes")
    print(f"  Amplitud osc. J2 (2ª mit.): ±{amplitud_J2:>9.3f} km   "
          f"(rango total: {2*amplitud_J2:.3f} km)")
    print(f"  Ratio decay / rango J2    : {ratio_sn:>10.4f}")

    # ── Aviso si el decay no es medible con fiabilidad ───────────────
    if ratio_sn < 0.15:
        print()
        if not planeta.tiene_atmosfera:
            # Caso Luna/Mercurio: NO hay drag por diseño, no es un fallo
            print(f"  ⓘ  Cuerpo SIN atmósfera modelada: decay esperado = 0 km.")
            print(f"      La pendiente mostrada ({slope_km_dia:+.5f} km/día) es")
            print(f"      RUIDO del muestreo de las oscilaciones J2; no es física.")
            print(f"      {planeta.nombre} no decae por drag — solo precesa por J2.")
        else:
            # Caso planeta CON atmósfera pero a altitud donde el drag es nulo
            print(f"  ⚠️  AVISO: pendiente NO FIABLE — drag despreciable a esta altitud.")
            print(f"      El decay ({abs(decay_secular):.4f} km) es mucho menor que la")
            print(f"      oscilación J2 (rango {2*amplitud_J2:.2f} km).")
            print(f"      Lo que mide la pendiente es ruido del muestreo, no física.")
            print(f"      Para detectar decay real: sube los días de simulación o")
            print(f"      reduce la altura inicial (más drag).")
    elif ratio_sn < 0.4 and planeta.tiene_atmosfera:
        print()
        print(f"  ⓘ  Aviso: pendiente con incertidumbre (decay marginal vs osc. J2).")
        print(f"      Tómala como orden de magnitud, no como valor exacto.")
    print()
    print(f"  ── Estado osculador final (instantáneo, oscila ±) ──")
    print(f"  Semieje a (osculador)  : {a_fin_osc:>10.3f} km")
    print(f"  Apogeo  (osculador)    : {h_apo_osc:>10.3f} km")
    print(f"  Perigeo (osculador)    : {h_per_osc:>10.3f} km")
    print(f"  Excentricidad          : {ecc_osc:>10.6f}")
    print(f"  RAAN final             : {raan_osc:>10.3f}°   "
          f"(en convención [0°, 360°))")
    print()
    print(f"  ── Regresión nodal por J₂ ──────────────────────────")
    print(f"  Precesión RAAN total   : {delta_raan_real:>+10.3f}°   "
          f"({tasa_real_dia:+.4f}°/día)")
    print(f"  Teórica (fórmula J₂)   : {delta_raan_teor:>+10.3f}°   "
          f"({tasa_teor_dia:+.4f}°/día)")
    diff_raan = delta_raan_real - delta_raan_teor
    print(f"  Discrepancia           : {diff_raan:>+10.3f}°   "
          f"(idealmente cero; pequeña diferencia por drag)")
    print(f"{'─' * 70}\n")

    # ── Visualización ────────────────────────────────────────────────
    titulo = (
        f"{planeta.nombre} · {dias}d · h₀={h_ini_km:.0f} km · "
        f"i={inc_deg:.1f}° · {masa:.0f}kg · {area:.0f}m² · Cd={cd}{factor_str}"
    )
    graficar_3d(planeta, orb_inicial, orb_real_final, titulo,
                hubo_reentrada=hubo_reentrada)
    graficar_2d_altura(planeta, t_real, h_ide, h_real, titulo)


if __name__ == "__main__":
    main()
