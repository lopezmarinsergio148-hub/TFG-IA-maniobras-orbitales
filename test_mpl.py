import numpy as np
import plotly.graph_objects as go
from astropy import units as u
from poliastro.bodies import Sun
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.plotting import OrbitPlotter3D


def add_dv_line(fig, r_vec, dv_vec, name, color, scale=7e6):
    """Dibuja el vector delta-v en el espacio 3D."""
    r0 = r_vec.to(u.km).value
    dv_val = np.linalg.norm(dv_vec.to(u.km / u.s).value)
    rf = r0 + (dv_vec.to(u.km / u.s).value * scale)

    fig.add_trace(go.Scatter3d(
        x=[r0[0], rf[0]],
        y=[r0[1], rf[1]],
        z=[r0[2], rf[2]],
        mode="lines+markers",
        line=dict(color=color, width=8),
        marker=dict(size=[0, 5], color=color),
        name=f"{name}: {dv_val:.2f} km/s",
        hoverinfo="name"
    ))


def hohmann_tierra_marte_detallado():
    # 1. Parámetros orbitales
    r1 = 1.0 * u.AU
    r2 = 1.523679 * u.AU

    earth_orb = Orbit.circular(Sun, r1)
    mars_orb = Orbit.circular(Sun, r2)

    # 2. Calcular Maniobra de Hohmann
    man = Maneuver.hohmann(earth_orb, r2)

    # Extraer datos numéricos
    dv1_vec = man.impulses[0][1]
    dv2_vec = man.impulses[1][1]
    dv1_mag = np.linalg.norm(dv1_vec).to(u.km / u.s)
    dv2_mag = np.linalg.norm(dv2_vec).to(u.km / u.s)
    dv_total = dv1_mag + dv2_mag

    # Tiempo de vuelo
    tof = man.get_total_time().to(u.day)

    # 3. Órbita de transferencia
    transfer_orb, _ = earth_orb.apply_maneuver(man, intermediate=True)

    # 4. Visualización con OrbitPlotter3D
    plotter = OrbitPlotter3D()
    plotter.plot(earth_orb, label="Órbita Tierra")
    plotter.plot(mars_orb, label="Órbita Marte")
    plotter.plot(transfer_orb, label="Transferencia (Hohmann)")

    fig = plotter.show()

    # 5. Añadir los vectores Δv (Flechas)
    r_exit = earth_orb.r
    # El punto de llegada es el apoapsis de la transferencia (tras el TOF)
    r_arrival = transfer_orb.propagate(tof).r

    add_dv_line(fig, r_exit, dv1_vec, "Δv1 (Salida)", "red", scale=8e6)
    add_dv_line(fig, r_arrival, dv2_vec, "Δv2 (Llegada)", "green", scale=8e6)

    # 6. Añadir cuadro de información en el gráfico
    info_text = (
        f"<b>DATOS DE LA MISIÓN</b><br>"
        f"-----------------------<br>"
        f"Tiempo de vuelo: {tof.value:.2f} días<br>"
        f"Δv1 (Inyección): {dv1_mag.value:.3f} km/s<br>"
        f"Δv2 (Captura): {dv2_mag.value:.3f} km/s<br>"
        f"<b>Δv TOTAL: {dv_total.value:.3f} km/s</b>"
    )

    fig.update_layout(
        title="Transferencia de Hohmann: Tierra -> Marte",
        scene=dict(
            aspectmode='data'  # Mantiene las proporciones reales de los ejes
        ),
        annotations=[
            dict(
                showarrow=False,
                text=info_text,
                x=0.05, y=0.95,
                xref="paper", yref="paper",
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )

    # Imprimir también por consola para tenerlo a mano
    print(f"--- RESULTADOS ---")
    print(f"TOF:      {tof}")
    print(f"Delta-v1: {dv1_mag}")
    print(f"Delta-v2: {dv2_mag}")
    print(f"Total:    {dv_total}")

    fig.show()


import numpy as np
import plotly.graph_objects as go
from astropy import units as u
from poliastro.bodies import Sun
from poliastro.twobody import Orbit
from poliastro.iod import lambert
from poliastro.plotting import OrbitPlotter3D


def add_dv_arrow(fig, r_vec, dv_vec, name, color, scale=6e6):
    """Dibuja el vector delta-v como una línea con un marcador en la punta."""
    r0 = r_vec.to(u.km).value
    dv_km_s = dv_vec.to(u.km / u.s).value
    dv_val = np.linalg.norm(dv_km_s)
    rf = r0 + (dv_km_s * scale)

    fig.add_trace(go.Scatter3d(
        x=[r0[0], rf[0]], y=[r0[1], rf[1]], z=[r0[2], rf[2]],
        mode="lines+markers",
        line=dict(color=color, width=6),
        marker=dict(size=[0, 4], color=color),
        name=f"{name}: {dv_val:.2f} km/s"
    ))


def transferencia_rapida_pulida(days=150):
    # 1. Órbitas circulares de referencia
    earth_orb = Orbit.circular(Sun, 1.0 * u.AU)
    mars_orb = Orbit.circular(Sun, 1.524 * u.AU)

    # 2. Definir puntos de transferencia
    r_ini = earth_orb.r
    theta_dest = 140 * u.deg

    # Forma alternativa y robusta de obtener r y v en un ángulo:
    # Creamos una órbita para Marte con la anomalía verdadera deseada
    mars_at_arrival = Orbit.from_classical(
        Sun, mars_orb.a, mars_orb.ecc, mars_orb.inc,
        mars_orb.raan, mars_orb.argp, theta_dest
    )
    r_dest = mars_at_arrival.r
    v_mars_end = mars_at_arrival.v

    # 3. Solucionar Lambert
    it = lambert(Sun.k, r_ini, r_dest, days * u.day)
    v_ini, v_dest = it[0] if isinstance(it, list) else it

    # 4. Crear órbita de transferencia e impulsos
    trans_orb = Orbit.from_vectors(Sun, r_ini, v_ini)
    dv1 = v_ini - earth_orb.v
    dv2 = v_mars_end - v_dest

    # 5. Graficar
    plotter = OrbitPlotter3D()
    plotter.plot(earth_orb, label="Tierra (Origen)")
    plotter.plot(mars_orb, label="Órbita Marte (Referencia)")
    plotter.plot(trans_orb, label=f"Trayectoria Rápida ({days} días)")

    fig = plotter.show()

    # 6. Añadir Flechas de Impulso y punto de encuentro
    add_dv_arrow(fig, r_ini, dv1, "Inyección (Δv1)", "red")
    add_dv_arrow(fig, r_dest, dv2, "Captura (Δv2)", "green")

    # Marcador de Marte en el encuentro
    fig.add_trace(go.Scatter3d(
        x=[r_dest[0].to(u.km).value],
        y=[r_dest[1].to(u.km).value],
        z=[r_dest[2].to(u.km).value],
        mode="markers",
        marker=dict(size=8, color="orange", symbol="diamond"),
        name="Posición de Marte al llegar"
    ))

    # 7. Resumen de datos
    dv1_mag = np.linalg.norm(dv1.to(u.km / u.s))
    dv2_mag = np.linalg.norm(dv2.to(u.km / u.s))

    fig.update_layout(
        title=f"Lambert: Tierra -> Marte en {days} días",
        annotations=[dict(
            text=(f"<b>ANÁLISIS DE MISIÓN</b><br>Días de viaje: {days}<br>"
                  f"Δv1 (Salida): {dv1_mag.value:.2f} km/s<br>"
                  f"Δv2 (Llegada): {dv2_mag.value:.2f} km/s"),
            xref="paper", yref="paper", x=0.05, y=0.9, showarrow=False,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="black"
        )]
    )
    fig.show()


import numpy as np
import plotly.graph_objects as go
from astropy import units as u
from astropy.time import Time

from poliastro.bodies import Earth, Mars, Sun
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.iod import lambert
from poliastro.plotting import OrbitPlotter3D


def add_dv_arrow(fig, r_vec, dv_vec, name, color, scale=7e6):
    r0 = r_vec.to(u.km).value
    dv_km_s = dv_vec.to(u.km / u.s).value
    dv_val = np.linalg.norm(dv_km_s)
    rf = r0 + (dv_km_s * scale)

    fig.add_trace(go.Scatter3d(
        x=[r0[0], rf[0]], y=[r0[1], rf[1]], z=[r0[2], rf[2]],
        mode="lines+markers",
        line=dict(color=color, width=6),
        marker=dict(size=[0, 4], color=color),
        name=f"{name}: {dv_val:.2f} km/s"
    ))


def mision_marte_real_2026():
    # 1. Definir fechas en TDB (escala recomendada para poliastro)
    fecha_lanzamiento = Time("2026-12-01 12:00:00", scale="tdb")
    tiempo_vuelo = 175 * u.day
    fecha_llegada = fecha_lanzamiento + tiempo_vuelo

    print(f"Calculando trayectoria real...")
    print(f"Salida: {fecha_lanzamiento}")
    print(f"Llegada: {fecha_llegada}")

    # 2. Obtener Efemérides con MARGEN para evitar el error de interpolación
    # Añadimos 2 días de margen al inicio y al final
    t_span = Time(np.linspace(
        (fecha_lanzamiento - 2 * u.day).jd,
        (fecha_llegada + 2 * u.day).jd,
        150), format="jd", scale="tdb")

    earth_ephem = Ephem.from_body(Earth, t_span)
    mars_ephem = Ephem.from_body(Mars, t_span)

    # 3. Posiciones y velocidades exactas interpoladas
    r_ini, v_ini_earth = earth_ephem.rv(fecha_lanzamiento)
    r_dest, v_mars_arrival = mars_ephem.rv(fecha_llegada)

    # 4. Resolver Lambert
    it = lambert(Sun.k, r_ini, r_dest, tiempo_vuelo)
    v_trans_ini, v_trans_arrival = it[0] if isinstance(it, list) else it

    # 5. Crear órbitas para el plotter
    ss_earth = Orbit.from_vectors(Sun, r_ini, v_ini_earth, epoch=fecha_lanzamiento)
    ss_mars = Orbit.from_vectors(Sun, r_dest, v_mars_arrival, epoch=fecha_llegada)
    trans_orb = Orbit.from_vectors(Sun, r_ini, v_trans_ini, epoch=fecha_lanzamiento)

    # 6. Cálculos de Delta-v
    dv1 = v_trans_ini - v_ini_earth
    dv2 = v_mars_arrival - v_trans_arrival
    dv_total = np.linalg.norm(dv1.to(u.km / u.s)) + np.linalg.norm(dv2.to(u.km / u.s))

    # 7. Graficar (usando OrbitPlotter3D como antes)
    plotter = OrbitPlotter3D()
    plotter.plot(ss_earth, label="Tierra (Lanzamiento)")
    plotter.plot(ss_mars, label="Marte (Llegada)")
    plotter.plot(trans_orb, label="Transferencia")

    fig = plotter.show()

    # Añadir vectores manualmente a la figura del plotter
    add_dv_arrow(fig, r_ini, dv1, "Inyección TLI", "red")
    add_dv_arrow(fig, r_dest, dv2, "Inserción en Marte", "green")

    fig.update_layout(
        title=f"Misión Real Tierra-Marte (Ventana 2026) | Δv Total: {dv_total.value:.2f} km/s",
        scene=dict(aspectmode='data')
    )

    fig.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Mars, Sun
from poliastro.iod import lambert
from poliastro.ephem import Ephem


def porkchop_marte_2026():
    print("Calculando la ventana óptima para Marte (Oct 2026 - Ene 2027)...")

    # 1. Rango de lanzamiento: Centrado en Noviembre 2026
    lanzamientos = Time(np.linspace(Time("2026-09-01").jd, Time("2027-01-31").jd, 150), format="jd", scale="tdb")
    # TOF para Marte: el estándar es entre 180 y 300 días
    duraciones = np.linspace(150, 350, 150) * u.day

    dv_matrix = np.full((len(duraciones), len(lanzamientos)), np.nan)

    # 2. Efemérides con cobertura total (Lanzamiento + TOF)
    # Cubrimos hasta finales de 2027 para que no de error la llegada
    t_span = Time(np.linspace(Time("2026-08-15").jd, Time("2028-06-01").jd, 500), format="jd", scale="tdb")
    earth_ephem = Ephem.from_body(Earth, t_span)
    mars_ephem = Ephem.from_body(Mars, t_span)

    for i, launch_date in enumerate(lanzamientos):
        r_earth, v_earth = earth_ephem.rv(launch_date)
        for j, tof in enumerate(duraciones):
            arrival_date = launch_date + tof
            r_mars, v_mars = mars_ephem.rv(arrival_date)
            try:
                # Resolvemos la trayectoria directa
                it = lambert(Sun.k, r_earth, r_mars, tof)
                v_ini, v_arr = it[0] if isinstance(it, list) else it

                dv1 = np.linalg.norm((v_ini - v_earth).to(u.km / u.s).value)
                dv2 = np.linalg.norm((v_mars - v_arr).to(u.km / u.s).value)
                total = dv1 + dv2

                # Filtro de 15 km/s para Marte
                if total < 15:
                    dv_matrix[j, i] = total
            except:
                continue

    # 3. Localizar el mínimo
    idx = np.unravel_index(np.nanargmin(dv_matrix), dv_matrix.shape)
    dv_min = dv_matrix[idx]
    fecha_min = lanzamientos[idx[1]].datetime

    # 4. Gráfica de Porkchop
    fig, ax = plt.subplots(figsize=(12, 8))
    X, Y = np.meshgrid(lanzamientos.datetime, duraciones.value)

    levels = np.linspace(np.nanmin(dv_matrix), 12, 50)
    cp = ax.contourf(X, Y, dv_matrix, levels=levels, cmap='viridis_r', extend='max')

    plt.colorbar(cp, label='Delta-V Total (km/s)')
    ax.plot(fecha_min, duraciones[idx[0]].value, 'r*', markersize=15, label=f"Mínimo: {dv_min:.2f} km/s")

    ax.set_title(f"Porkchop Plot: Tierra-Marte 2026\nÓptimo: {fecha_min.strftime('%d-%b-%Y')}", fontsize=14)
    ax.set_xlabel("Fecha de Lanzamiento")
    ax.set_ylabel("Días de viaje (TOF)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

    plt.legend()
    plt.grid(True, alpha=0.1)
    plt.show()

    print(f"--- RESULTADO MARTE 2026 ---")
    print(f"Mejor fecha: {fecha_min}")
    print(f"Delta-V mínima: {dv_min:.4f} km/s")



import numpy as np
import plotly.graph_objects as go
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Mars, Sun, Venus
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.iod import lambert


def crear_animacion_marte():
    print("Iniciando cálculos de efemérides y trayectoria óptima...")

    # 1. Configuración de fechas (Punto óptimo del Porkchop)
    fecha_lanzamiento = Time("2026-12-01 12:00:00", scale="tdb")
    tiempo_vuelo = 175 * u.day
    fecha_llegada = fecha_lanzamiento + tiempo_vuelo

    num_pasos = 120
    tiempos = Time(np.linspace(fecha_lanzamiento.jd, fecha_llegada.jd, num_pasos), format="jd", scale="tdb")

    # 2. Obtención de datos reales (Efemérides)
    t_span = Time(np.linspace((fecha_lanzamiento - 5 * u.day).jd, (fecha_llegada + 5 * u.day).jd, 200), format="jd",
                  scale="tdb")
    earth_ephem = Ephem.from_body(Earth, t_span)
    mars_ephem = Ephem.from_body(Mars, t_span)

    r_ini, v_e = earth_ephem.rv(fecha_lanzamiento)
    r_dest, v_m = mars_ephem.rv(fecha_llegada)

    # 3. Cálculo de la maniobra (Lambert)
    it = lambert(Sun.k, r_ini, r_dest, tiempo_vuelo)
    v_trans_ini, v_trans_arrival = it[0] if isinstance(it, list) else it
    trans_orb = Orbit.from_vectors(Sun, r_ini, v_trans_ini, epoch=fecha_lanzamiento)

    # Cálculo de magnitudes de impulsos
    dv1 = np.linalg.norm((v_trans_ini - v_e).to(u.km / u.s)).value
    dv2 = np.linalg.norm((v_m - v_trans_arrival).to(u.km / u.s)).value
    dv_total = dv1 + dv2

    # 4. Generación de coordenadas para animación
    pos_tierra = np.array([earth_ephem.rv(t)[0].to(u.km).value for t in tiempos])
    pos_marte = np.array([mars_ephem.rv(t)[0].to(u.km).value for t in tiempos])
    pos_nave = np.array([trans_orb.propagate(t - fecha_lanzamiento).r.to(u.km).value for t in tiempos])

    # 5. Construcción de la Figura
    fig = go.Figure()

    # TRAZOS MÓVILES (0, 1, 2) y SOL (3)
    fig.add_trace(go.Scatter3d(x=[pos_tierra[0, 0]], y=[pos_tierra[0, 1]], z=[pos_tierra[0, 2]],
                               mode='markers', marker=dict(color='blue', size=8), name="Tierra"))
    fig.add_trace(go.Scatter3d(x=[pos_marte[0, 0]], y=[pos_marte[0, 1]], z=[pos_marte[0, 2]],
                               mode='markers', marker=dict(color='orange', size=7), name="Marte"))
    fig.add_trace(go.Scatter3d(x=[pos_nave[0, 0]], y=[pos_nave[0, 1]], z=[pos_nave[0, 2]],
                               mode='markers', marker=dict(color='green', size=6), name="Nave"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers',
                               marker=dict(size=12, color='yellow'), name="Sol"))

    # TRAZOS ESTÁTICOS (Órbitas con grosor aumentado)
    fig.add_trace(go.Scatter3d(x=pos_tierra[:, 0], y=pos_tierra[:, 1], z=pos_tierra[:, 2],
                               mode='lines', line=dict(color='blue', width=4, dash='dot'), name="Órbita Tierra"))
    fig.add_trace(go.Scatter3d(x=pos_marte[:, 0], y=pos_marte[:, 1], z=pos_marte[:, 2],
                               mode='lines', line=dict(color='orange', width=4, dash='dot'), name="Órbita Marte"))
    fig.add_trace(go.Scatter3d(x=pos_nave[:, 0], y=pos_nave[:, 1], z=pos_nave[:, 2],
                               mode='lines', line=dict(color='green', width=5), name="Trayectoria Nave"))

    # 6. Definición de Frames (Animación)
    frames = []
    for k in range(num_pasos):
        dias = int((tiempos[k] - fecha_lanzamiento).value)
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=[pos_tierra[k, 0]], y=[pos_tierra[k, 1]], z=[pos_tierra[k, 2]]),
                go.Scatter3d(x=[pos_marte[k, 0]], y=[pos_marte[k, 1]], z=[pos_marte[k, 2]]),
                go.Scatter3d(x=[pos_nave[k, 0]], y=[pos_nave[k, 1]], z=[pos_nave[k, 2]]),
                go.Scatter3d(x=[0], y=[0], z=[0])
            ],
            traces=[0, 1, 2, 3],
            name=str(k),
            layout=go.Layout(title_text=(
                f"Misión Marte 2026 | Día {dias} | "
                f"Δv1: {dv1:.2f} km/s | Δv2: {dv2:.2f} km/s | Total: {dv_total:.2f} km/s"
            ))
        ))

    fig.frames = frames

    # 7. Layout y Controles
    fig.update_layout(
        scene=dict(aspectmode='data', xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"),
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 35, "redraw": True}, "fromcurrent": True}],
                 "label": "Despegue (Play)", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                 "label": "Pausa", "method": "animate"}
            ],
            "type": "buttons", "showactive": False, "x": 0.1, "y": 0.9
        }]
    )

    print("Cálculo completado. Abriendo visualización interactiva...")
    fig.show()


import numpy as np
import plotly.graph_objects as go
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Venus, Sun
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.iod import lambert


def animacion_tierra_venus_2028():
    print("Calculando encuentro con Venus y Delta-V necesario...")

    # 1. Configuración de fechas
    fecha_lanzamiento = Time("2028-01-08 01:48:32", scale="tdb")
    tiempo_vuelo = 120 * u.day
    fecha_llegada = fecha_lanzamiento + tiempo_vuelo

    num_pasos = 100
    tiempos = Time(np.linspace(fecha_lanzamiento.jd, fecha_llegada.jd, num_pasos), format="jd", scale="tdb")

    # 2. Efemérides y vectores de estado
    t_span = Time(np.linspace((fecha_lanzamiento - 5 * u.day).jd, (fecha_llegada + 5 * u.day).jd, 200), format="jd",
                  scale="tdb")
    earth_ephem = Ephem.from_body(Earth, t_span)
    venus_ephem = Ephem.from_body(Venus, t_span)

    r_ini, v_earth_lanz = earth_ephem.rv(fecha_lanzamiento)
    r_dest, v_venus_lleg = venus_ephem.rv(fecha_llegada)

    # 3. Resolución de Lambert e Impulso
    it = lambert(Sun.k, r_ini, r_dest, tiempo_vuelo)
    v_trans_ini, v_trans_lleg = it[0] if isinstance(it, list) else it

    # Cálculo de los impulsos (Delta-V)
    dv_lanzamiento = np.linalg.norm((v_trans_ini - v_earth_lanz).to(u.km / u.s).value)
    dv_llegada = np.linalg.norm((v_venus_lleg - v_trans_lleg).to(u.km / u.s).value)
    dv_total = dv_lanzamiento + dv_llegada

    print(f"\n--- ANALISIS DE IMPULSO ---")
    print(f"DV Lanzamiento (Salida Tierra): {dv_lanzamiento:.3f} km/s")
    print(f"DV Captura (Llegada Venus): {dv_llegada:.3f} km/s")
    print(f"DV Total Requerido: {dv_total:.3f} km/s\n")

    # Crear órbita de transferencia para propagar
    trans_orb = Orbit.from_vectors(Sun, r_ini, v_trans_ini, epoch=fecha_lanzamiento)

    # 4. Generar trayectorias
    pos_e = np.array([earth_ephem.rv(t)[0].to(u.km).value for t in tiempos])
    pos_v = np.array([venus_ephem.rv(t)[0].to(u.km).value for t in tiempos])
    pos_n = np.array([trans_orb.propagate(t - fecha_lanzamiento).r.to(u.km).value for t in tiempos])

    # 5. Construcción de la figura Plotly
    fig = go.Figure()

    # Añadir trazas iniciales
    fig.add_trace(go.Scatter3d(x=[pos_e[0, 0]], y=[pos_e[0, 1]], z=[pos_e[0, 2]], mode='markers',
                               marker=dict(color='blue', size=7), name="Tierra"))
    fig.add_trace(go.Scatter3d(x=[pos_v[0, 0]], y=[pos_v[0, 1]], z=[pos_v[0, 2]], mode='markers',
                               marker=dict(color='orange', size=7), name="Venus"))
    fig.add_trace(go.Scatter3d(x=[pos_n[0, 0]], y=[pos_n[0, 1]], z=[pos_n[0, 2]], mode='markers',
                               marker=dict(color='green', size=5), name="Nave"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=15, color='yellow'), name="Sol"))

    # Órbitas completas
    fig.add_trace(go.Scatter3d(x=pos_e[:, 0], y=pos_e[:, 1], z=pos_e[:, 2], mode='lines',
                               line=dict(color='blue', width=2, dash='dot'), name="Órbita Tierra"))
    fig.add_trace(go.Scatter3d(x=pos_v[:, 0], y=pos_v[:, 1], z=pos_v[:, 2], mode='lines',
                               line=dict(color='orange', width=2, dash='dot'), name="Órbita Venus"))
    fig.add_trace(
        go.Scatter3d(x=pos_n[:, 0], y=pos_n[:, 1], z=pos_n[:, 2], mode='lines', line=dict(color='green', width=5),
                     name="Trayectoria Nave"))

    # 6. Definición de Frames con información de DV
    frames = []
    for k in range(num_pasos):
        dias = int((tiempos[k] - fecha_lanzamiento).value)
        # Mostrar el DV total en el título del frame
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=[pos_e[k, 0]], y=[pos_e[k, 1]], z=[pos_e[k, 2]]),
                go.Scatter3d(x=[pos_v[k, 0]], y=[pos_v[k, 1]], z=[pos_v[k, 2]]),
                go.Scatter3d(x=[pos_n[k, 0]], y=[pos_n[k, 1]], z=[pos_n[k, 2]]),
                go.Scatter3d(x=[0], y=[0], z=[0])
            ],
            traces=[0, 1, 2, 3],
            name=str(k),
            layout=go.Layout(title_text=f"Misión Venus 2028 | Día {dias} | DV Requerido: {dv_total:.2f} km/s")
        ))

    fig.frames = frames

    # Configuración de botones y layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)",
            aspectmode='data'
        ),
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}], "label": "Play",
                 "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], "label": "Pausa",
                 "method": "animate"}
            ],
            "type": "buttons", "showactive": False, "x": 0.1, "y": 0.9
        }]
    )

    fig.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Venus, Sun
from poliastro.iod import lambert
from poliastro.ephem import Ephem


def porkchop_venus():
    print("Calculando Porkchop Venus con rango de 35 km/s...")

    # 1. Rango de lanzamientos (Oct 2027 - Abr 2028)
    lanzamientos = Time(np.linspace(Time("2027-10-01").jd, Time("2028-04-30").jd, 200), format="jd", scale="tdb")
    # Duraciones de hasta 250 días para ver las zonas de alta energía
    duraciones = np.linspace(100, 250, 150) * u.day

    dv_matrix = np.full((len(duraciones), len(lanzamientos)), np.nan)

    # 2. Efemérides: EXTENDIDAS hasta 2029 para evitar el ValueError
    # Cubrimos desde antes del lanzamiento hasta después de la llegada más tardía
    t_min = Time("2027-09-01").jd
    t_max = Time("2029-06-01").jd
    t_span = Time(np.linspace(t_min, t_max, 600), format="jd", scale="tdb")

    earth_ephem = Ephem.from_body(Earth, t_span)
    venus_ephem = Ephem.from_body(Venus, t_span)

    # 3. Cálculo de la matriz
    for i, launch_date in enumerate(lanzamientos):
        r_earth, v_earth = earth_ephem.rv(launch_date)
        for j, tof in enumerate(duraciones):
            arrival_date = launch_date + tof
            r_venus, v_venus = venus_ephem.rv(arrival_date)
            try:
                # Resolvemos Lambert
                it = lambert(Sun.k, r_earth, r_venus, tof)
                v_ini, v_arr = it[0] if isinstance(it, list) else it

                dv1 = np.linalg.norm((v_ini - v_earth).to(u.km / u.s).value)
                dv2 = np.linalg.norm((v_venus - v_arr).to(u.km / u.s).value)
                total = dv1 + dv2

                # --- FILTRO SUBIDO A 35 KM/S ---
                if total < 35:
                    dv_matrix[j, i] = total
            except:
                continue

    # 4. Gestión de niveles automática
    v_min = np.nanmin(dv_matrix)
    v_max = 30  # Límite visual de la barra de colores
    levels = np.linspace(v_min, v_max, 60)

    # 5. Gráfica
    fig, ax = plt.subplots(figsize=(12, 8))
    X, Y = np.meshgrid(lanzamientos.datetime, duraciones.value)

    # Usamos un mapa de colores que resalte bien las diferencias
    cp = ax.contourf(X, Y, dv_matrix, levels=levels, cmap='magma_r', extend='both')
    plt.colorbar(cp, label='Delta-V Total (km/s)')

    # Marcamos el mínimo absoluto
    idx = np.unravel_index(np.nanargmin(dv_matrix), dv_matrix.shape)
    ax.plot(lanzamientos[idx[1]].datetime, duraciones[idx[0]].value, 'w*', markersize=12, label="Mínimo")

    ax.set_title(
        f"Porkchop Venus: Análisis de Alta Energía\nValle de mínima energía detectado en {dv_matrix[idx]:.2f} km/s",
        fontsize=14)
    ax.set_xlabel("Fecha de Lanzamiento")
    ax.set_ylabel("Días de viaje (TOF)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.grid(True, alpha=0.1)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Jupiter, Sun
from poliastro.iod import lambert
from poliastro.ephem import Ephem


def porkchop_jupiter():
    print("Generando Porkchop a Júpiter... Barriendo espacio de fases...")

    # 1. Rangos (Noviembre 2026 es el punto clave)
    lanzamientos = Time(np.linspace(Time("2026-09-01").jd, Time("2027-02-01").jd, 100), format="jd", scale="tdb")
    # Ampliamos TOF para asegurar que pillamos la transferencia de Hohmann (~1000 días)
    duraciones = np.linspace(600, 1300, 100) * u.day

    dv_matrix = np.full((len(duraciones), len(lanzamientos)), np.nan)

    # 2. Efemérides (Margen amplio hasta 2031)
    t_span = Time(np.linspace(Time("2026-08-01").jd, Time("2031-01-01").jd, 500), format="jd", scale="tdb")
    earth_ephem = Ephem.from_body(Earth, t_span)
    jupiter_ephem = Ephem.from_body(Jupiter, t_span)

    for i, launch_date in enumerate(lanzamientos):
        r_earth, v_earth = earth_ephem.rv(launch_date)
        for j, tof in enumerate(duraciones):
            arrival_date = launch_date + tof
            r_jup, v_jup = jupiter_ephem.rv(arrival_date)
            try:
                # Resolvemos Lambert (trayectoria directa prograde)
                it = lambert(Sun.k, r_earth, r_jup, tof)
                v_ini, v_arr = it[0] if isinstance(it, list) else it

                dv_total = np.linalg.norm((v_ini - v_earth).to(u.km / u.s).value) + \
                           np.linalg.norm((v_jup - v_arr).to(u.km / u.s).value)

                # Filtro generoso de 30 km/s para asegurar que Matplotlib tenga datos
                if dv_total < 30:
                    dv_matrix[j, i] = dv_total
            except:
                continue

    # 3. Gestión de niveles a prueba de errores
    if np.all(np.isnan(dv_matrix)):
        print("ERROR: No se han encontrado soluciones válidas. Revisa los rangos.")
        return

    v_min = np.nanmin(dv_matrix)
    v_max = v_min + 10  # Mostramos un rango de 10 km/s por encima del mínimo
    levels = np.linspace(v_min, v_max, 50)

    # 4. Gráfica
    fig, ax = plt.subplots(figsize=(12, 8))
    X, Y = np.meshgrid(lanzamientos.datetime, duraciones.value)

    # Dibujamos los contornos
    cp = ax.contourf(X, Y, dv_matrix, levels=levels, cmap='viridis_r', extend='max')
    cbar = plt.colorbar(cp)
    cbar.set_label('Delta-V Total (km/s)')

    # Encontrar y marcar el mínimo
    idx = np.unravel_index(np.nanargmin(dv_matrix), dv_matrix.shape)
    ax.plot(lanzamientos[idx[1]].datetime, duraciones[idx[0]].value, 'r*', markersize=12)

    ax.set_title(f"Porkchop Tierra-Júpiter 2026\nMínimo: {dv_matrix[idx]:.2f} km/s | TOF: {duraciones[idx[0]]:.1f}",
                 fontsize=14)
    ax.set_xlabel("Fecha de Lanzamiento")
    ax.set_ylabel("Tiempo de Vuelo (días)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.grid(True, alpha=0.2)
    plt.show()




import numpy as np
import plotly.graph_objects as go
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Jupiter, Sun
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.iod import lambert


def animacion_jupiter():
    print("Calculando trayectoria Tierra-Júpiter 2026...")

    # 1. Fechas óptimas (Ventana Noviembre 2026)
    fecha_lanzamiento = Time("2026-11-10 12:00:00", scale="tdb")
    tiempo_vuelo = 950 * u.day  # Aproximadamente 2.2 años
    fecha_llegada = fecha_lanzamiento + tiempo_vuelo

    num_pasos = 150
    tiempos = Time(np.linspace(fecha_lanzamiento.jd, fecha_llegada.jd, num_pasos), format="jd", scale="tdb")

    # 2. Efemérides con margen amplio
    t_span = Time(np.linspace(Time("2026-10-01").jd, Time("2030-06-01").jd, 400), format="jd", scale="tdb")
    earth_ephem = Ephem.from_body(Earth, t_span)
    jupiter_ephem = Ephem.from_body(Jupiter, t_span)

    # Vectores para Lambert
    r_ini, v_earth_lanz = earth_ephem.rv(fecha_lanzamiento)
    r_dest, v_jup_lleg = jupiter_ephem.rv(fecha_llegada)

    # 3. Resolver Lambert e Impulso
    it = lambert(Sun.k, r_ini, r_dest, tiempo_vuelo)
    v_trans_ini, v_trans_lleg = it[0] if isinstance(it, list) else it

    # Cálculo de Delta-V
    dv_salida = np.linalg.norm((v_trans_ini - v_earth_lanz).to(u.km / u.s).value)
    dv_llegada = np.linalg.norm((v_jup_lleg - v_trans_lleg).to(u.km / u.s).value)
    dv_total = dv_salida + dv_llegada

    # Órbita de la nave
    trans_orb = Orbit.from_vectors(Sun, r_ini, v_trans_ini, epoch=fecha_lanzamiento)

    # 4. Posiciones para la animación
    pos_e = np.array([earth_ephem.rv(t)[0].to(u.km).value for t in tiempos])
    pos_j = np.array([jupiter_ephem.rv(t)[0].to(u.km).value for t in tiempos])
    pos_n = np.array([trans_orb.propagate(t - fecha_lanzamiento).r.to(u.km).value for t in tiempos])

    # 5. Crear Figura
    fig = go.Figure()

    # Trazas de objetos (0: Tierra, 1: Júpiter, 2: Nave, 3: Sol)
    fig.add_trace(go.Scatter3d(x=[pos_e[0, 0]], y=[pos_e[0, 1]], z=[pos_e[0, 2]], mode='markers',
                               marker=dict(color='blue', size=6), name="Tierra"))
    fig.add_trace(go.Scatter3d(x=[pos_j[0, 0]], y=[pos_j[0, 1]], z=[pos_j[0, 2]], mode='markers',
                               marker=dict(color='brown', size=10), name="Júpiter"))
    fig.add_trace(go.Scatter3d(x=[pos_n[0, 0]], y=[pos_n[0, 1]], z=[pos_n[0, 2]], mode='markers',
                               marker=dict(color='green', size=4), name="Sonda"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(color='yellow', size=15), name="Sol"))

    # Trazas de órbitas (líneas fijas)
    fig.add_trace(go.Scatter3d(x=pos_e[:, 0], y=pos_e[:, 1], z=pos_e[:, 2], mode='lines',
                               line=dict(color='blue', width=2, dash='dot'), name="Órbita Tierra"))
    fig.add_trace(go.Scatter3d(x=pos_j[:, 0], y=pos_j[:, 1], z=pos_j[:, 2], mode='lines',
                               line=dict(color='brown', width=2, dash='dot'), name="Órbita Júpiter"))
    fig.add_trace(
        go.Scatter3d(x=pos_n[:, 0], y=pos_n[:, 1], z=pos_n[:, 2], mode='lines', line=dict(color='green', width=4),
                     name="Trayectoria"))

    # 6. Frames de la animación
    frames = []
    for k in range(num_pasos):
        dias_transcurridos = int((tiempos[k] - fecha_lanzamiento).value)
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=[pos_e[k, 0]], y=[pos_e[k, 1]], z=[pos_e[k, 2]]),
                go.Scatter3d(x=[pos_j[k, 0]], y=[pos_j[k, 1]], z=[pos_j[k, 2]]),
                go.Scatter3d(x=[pos_n[k, 0]], y=[pos_n[k, 1]], z=[pos_n[k, 2]]),
                go.Scatter3d(x=[0], y=[0], z=[0])
            ],
            name=str(k),
            layout=go.Layout(
                title_text=f"Misión Júpiter | Día {dias_transcurridos} / {int(tiempo_vuelo.value)} | Impulso Total: {dv_total:.2f} km/s")
        ))

    fig.frames = frames

    # 7. Configuración de Menús (Botones Play/Pausa)
    fig.update_layout(
        scene=dict(aspectmode='data', xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"),
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 20, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "Pausa",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                }
            ],
            "direction": "left", "pad": {"r": 10, "t": 87}, "showactive": False, "x": 0.1, "xanchor": "right", "y": 0,
            "yanchor": "top"
        }]
    )

    fig.show()


if __name__ == "__main__":
    #hohmann_tierra_marte_detallado()
    #transferencia_rapida_pulida()
    #mision_marte_real_2026()
    #porkchop_marte_2026()
    #crear_animacion_marte()
    #animacion_tierra_venus_2028()
    #porkchop_venus()
    porkchop_jupiter()
    #animacion_jupiter()