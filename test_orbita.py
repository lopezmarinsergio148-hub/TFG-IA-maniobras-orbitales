
#Lo he sacado de la galería de poliastro de trazado 3D

def ejercicio_churi():
    #Ejercicio 1
    import numpy as np
    from poliastro.bodies import Earth, Sun
    from poliastro.constants import J2000
    from poliastro.examples import churi, iss, molniya
    from poliastro.plotting import OrbitPlotter3D
    from poliastro.twobody import Orbit

    # Cambiar renderizador para que funcione en PyCharm
    import plotly.io as pio
    pio.renderers.default = "browser"

    # Mostrar la órbita en 3D
    fig = churi.plot(interactive=True, use_3d=True)
    fig.show() # Con esta linea consigo graficar en 3D en otra ventana y funciona bien en PyCharm.

#Este me lo ha dado chatgpt para usar la función Orbitplotter3D
def ejercicio_prueba_OrbitPlotter3D():
    #Ejercicio 2
    from poliastro.plotting import OrbitPlotter3D
    from poliastro.bodies import Earth, Mars
    from poliastro.twobody import Orbit
    from astropy import units as u

    r = [-6000, -1000, 0] << u.km
    v = [1.4, 3, 0] << u.km / u.s

    orb = Orbit.from_vectors(Mars, r, v)

    # Crear el plotter
    plotter = OrbitPlotter3D()
    plotter.set_attractor(Mars)  # Opcional, solo si quieres usar otro cuerpo central

    # Dibujar la órbita
    fig = plotter.plot(orb, label="Órbita de prueba")
    fig.show()  # Muy importante en PyCharm para que se muestre el gráfico


def marte_clasico():
    #Ejercicio 2
    from poliastro.plotting import OrbitPlotter3D
    from poliastro.bodies import Earth, Mars, Sun
    from poliastro.twobody import Orbit
    from astropy import units as u

    # Data for Mars at J2000 from JPL HORIZONS
    a = 1.523679 << u.AU
    ecc = 0.093315 << u.one
    inc = 1.85 << u.deg
    raan = 49.562 << u.deg
    argp = 286.537 << u.deg
    nu = 23.33 << u.deg

    orb = Orbit.from_classical(Sun, a, ecc, inc, raan, argp, nu)

    # Crear el plotter
    plotter = OrbitPlotter3D()
    plotter.set_attractor(Sun)  # Opcional, solo si quieres usar otro cuerpo central

    # Dibujar la órbita
    fig = plotter.plot(orb, label="Órbita de prueba")
    fig.show()  # Muy importante en PyCharm para que se muestre el gráfico


def prueba():
    from astropy import units as u
    from astropy.time import Time
    from poliastro.bodies import Earth, Sun
    from poliastro.ephem import Ephem
    from poliastro.twobody import Orbit
    from poliastro.plotting import OrbitPlotter3D
    import plotly.io as pio

    # Renderizador para que se abra en el navegador (funciona guay en PyCharm)
    pio.renderers.default = "browser"


    # Definir fecha
    epoch = Time("2025-01-01", scale="tdb")

    # Efemérides de la Tierra
    earth_ephem = Ephem.from_body(Earth, epoch)

    # Crear órbita desde efemérides
    orb = Orbit.from_ephem(Sun, earth_ephem, epoch)

    # Crear el plotter
    plotter = OrbitPlotter3D()
    plotter.set_attractor(Sun)

    # Graficar
    fig = plotter.plot(orb, label="Tierra alrededor del Sol")
    fig.show()


def prueba_2d():
    import matplotlib.pyplot as plt
    from astropy.time import Time
    from poliastro.bodies import Earth, Sun
    from poliastro.ephem import Ephem
    from poliastro.twobody import Orbit

    # Fecha de referencia
    epoch = Time("2025-01-01", scale="tdb")

    # Efemérides de la Tierra
    earth_ephem = Ephem.from_body(Earth, epoch)

    # Crear órbita desde efemérides
    orb = Orbit.from_ephem(Sun, earth_ephem, epoch)

    # Obtener puntos de la órbita
    points = orb.sample(200)

    # Graficar con matplotlib (ventana de plots de PyCharm)
    plt.figure()
    plt.plot(points.x.to_value(), points.y.to_value(), label="Tierra")
    plt.scatter(0, 0, color="yellow", label="Sol", marker="o")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("Órbita de la Tierra alrededor del Sol (2D)")
    plt.legend()
    plt.axis("equal")
    plt.show()



def prueba_matplotlib():
    import matplotlib.pyplot as plt
    from astropy.time import Time
    from poliastro.bodies import Earth, Sun, Mars
    from poliastro.ephem import Ephem
    from poliastro.twobody import Orbit

    # Definir fecha
    epoch = Time("2025-01-01", scale="tdb")

    # Crear órbitas desde efemérides
    earth_orbit = Orbit.from_ephem(Sun, Ephem.from_body(Earth, epoch), epoch)
    mars_orbit = Orbit.from_ephem(Sun, Ephem.from_body(Mars, epoch), epoch)

    # Obtener puntos de las órbitas
    earth_points = earth_orbit.sample(200)  # 200 puntos
    mars_points = mars_orbit.sample(200)

    # Graficar con matplotlib
    plt.figure()
    plt.plot(earth_points.x.to_value(), earth_points.y.to_value(), label="Tierra")
    plt.plot(mars_points.x.to_value(), mars_points.y.to_value(), label="Marte")
    plt.scatter(0, 0, color="yellow", label="Sol", marker="o")  # el Sol en el centro
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.legend()
    plt.title("Órbitas Tierra y Marte - 2D")
    plt.axis("equal")
    plt.show()


from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Mars, Sun
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver

def hohmann_earth_to_mars(epoch_str="2025-01-01"):
    epoch = Time(epoch_str, scale="tdb")

    # Órbitas
    earth_orbit = Orbit.from_ephem(Sun, Ephem.from_body(Earth, epoch), epoch)
    mars_orbit = Orbit.from_ephem(Sun, Ephem.from_body(Mars, epoch), epoch)

    print(f"Semieje mayor Tierra (a1): {earth_orbit.a.to(u.AU):.6f}")
    print(f"Semieje mayor Marte  (a2): {mars_orbit.a.to(u.AU):.6f}")

    # Maniobra Hohmann
    man = Maneuver.hohmann(earth_orbit, mars_orbit.a)

    print(f"\nΔv total (Hohmann): {man.get_total_cost().to(u.km/u.s):.6f}")

    transfer_orbit = earth_orbit.apply_maneuver(man)
    print(f"Semieje mayor transferencia: {transfer_orbit.a.to(u.AU):.6f}")
    print(f"Periodo transferencia: {transfer_orbit.period.to(u.day):.3f}")
    print(f"Tiempo de vuelo (semiperíodo, Hohmann): {(transfer_orbit.period/2).to(u.day):.3f}")

    # Impulsos (vectores Δv)
    print("\nImpulsos (vector Δv) por maniobra:")
    for i, impulse in enumerate(man.impulses):
        dt, dv = impulse  # tiempo relativo y vector Δv
        print(f"  Impulso {i+1}: en t+{dt}, Δv = {dv.to(u.km/u.s)}")


def hohmann_earth_to_mars_3D(epoch_str="2025-01-01"):
    from astropy import units as u
    from astropy.time import Time
    from poliastro.bodies import Earth, Mars, Sun
    from poliastro.ephem import Ephem
    from poliastro.twobody import Orbit
    from poliastro.maneuver import Maneuver
    from poliastro.plotting import OrbitPlotter3D

    epoch = Time(epoch_str, scale="tdb")

    # Órbitas reales de Tierra y Marte
    earth_orbit = Orbit.from_ephem(Sun, Ephem.from_body(Earth, epoch), epoch)
    mars_orbit = Orbit.from_ephem(Sun, Ephem.from_body(Mars, epoch), epoch)

    # Maniobra Hohmann
    man = Maneuver.hohmann(earth_orbit, mars_orbit.a)

    # Aplicar maniobra
    transfer_orbit = earth_orbit.apply_maneuver(man)

    # Crear plotter 3D
    plotter = OrbitPlotter3D()
    plotter.set_attractor(Sun)

    # Graficar órbitas
    plotter.plot(earth_orbit, label="Tierra")
    plotter.plot(mars_orbit, label="Marte")
    fig = plotter.plot(transfer_orbit, label="Transferencia Hohmann")

    # Mostrar en ventana del navegador
    fig.show()


def ejercicio_jupiter(): #Creo que no está del todod bien
    # --- DEPENDENCIAS ---
    # pip install poliastro==0.17.0 astropy plotly jplephem numpy

    import numpy as np
    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import solar_system_ephemeris
    solar_system_ephemeris.set("jpl")

    from poliastro.bodies import Sun, Earth, Jupiter
    from poliastro.ephem import Ephem
    from poliastro.twobody import Orbit
    from poliastro.frames import Planes
    from poliastro.plotting import OrbitPlotter3D
    from poliastro.iod import izzo

    import plotly.io as pio
    pio.renderers.default = "browser"  # en PyCharm abre en el navegador

    # === FECHAS (misión Juno) ===
    date_launch = Time("2011-08-05 16:25", scale="utc").tdb
    date_arrival = Time("2016-07-05 03:18", scale="utc").tdb

    # === EFEMÉRIDES Y ÓRBITAS HELIOCÉNTRICAS ===
    earth_ephem_launch = Ephem.from_body(Earth, date_launch)
    jupiter_ephem_arr = Ephem.from_body(Jupiter, date_arrival)

    earth_launch = Orbit.from_ephem(Sun, earth_ephem_launch, epoch=date_launch)
    jupiter_arr = Orbit.from_ephem(Sun, jupiter_ephem_arr, epoch=date_arrival)

    # === LAMBERT Tierra->Júpiter (transferencia ideal) ===
    tof = (date_arrival - date_launch).to(u.s)
    r1, r2 = earth_launch.r, jupiter_arr.r

    # En poliastro 0.17.0 devuelve directamente (v0, v1)
    v0, v1 = izzo.lambert(Sun.k, r1, r2, tof)

    lambert_trf = Orbit.from_vectors(Sun, r1, v0, epoch=date_launch)

    # C3 requerido por la Lambert
    v_inf_vec_req = (v0 - earth_launch.v).to(u.km / u.s)
    C3_req = np.linalg.norm(v_inf_vec_req.value) ** 2 * (u.km ** 2 / u.s ** 2)
    print(f"C3 requerido por Lambert: {C3_req:.2f}")

    # === BÚSQUEDA DE DIRECCIÓN DE v∞ CON C3 FIJO ===
    C3_fixed = 31.1 * (u.km ** 2 / u.s ** 2)
    v_inf_mag = np.sqrt(C3_fixed.value) * (u.km / u.s)

    # Base RTN (Radial-Tangencial-Normal) en el lanzamiento
    r_vec = earth_launch.r.to(u.km).value
    v_vec = earth_launch.v.to(u.km / u.s).value

    r_hat = r_vec / np.linalg.norm(r_vec)
    t_hat = v_vec / np.linalg.norm(v_vec)
    n_hat = np.cross(r_hat, t_hat)
    n_hat = n_hat / np.linalg.norm(n_hat)

    def dir_from_angles(theta, phi):
        """Devuelve un vector unitario en el marco RTN."""
        return (np.cos(theta) * t_hat +
                np.sin(theta) * (np.cos(phi) * r_hat + np.sin(phi) * n_hat))

    def miss_distance_km(theta, phi):
        """Devuelve la distancia nave-Júpiter y la órbita para una dirección dada."""
        d = dir_from_angles(theta, phi)
        d = d / np.linalg.norm(d)
        r0 = earth_launch.r
        # v_inf_mag ya tiene unidades km/s → no se multiplica otra vez por (u.km/u.s)
        v0_sc = earth_launch.v + v_inf_mag * d
        sc_orbit = Orbit.from_vectors(Sun, r0, v0_sc, epoch=date_launch)
        sc_arr = sc_orbit.propagate(tof)
        miss = (sc_arr.r - jupiter_arr.r).to(u.km).value
        return np.linalg.norm(miss), sc_orbit, sc_arr

    # --- BÚSQUEDA EN REJILLA ---
    thetas = np.linspace(-np.pi, np.pi, 61)
    phis = np.linspace(-np.pi, np.pi, 61)

    best = {"miss_km": np.inf, "theta": None, "phi": None, "orbit": None, "arr": None}

    for th in thetas:
        for ph in phis:
            miss_km, sc_orb, sc_arr = miss_distance_km(th, ph)
            if miss_km < best["miss_km"]:
                best.update({"miss_km": miss_km, "theta": th, "phi": ph,
                             "orbit": sc_orb, "arr": sc_arr})

    print("\n=== RESULTADOS ===")
    print(f"C3 fijo           : {C3_fixed:.2f}")
    print(f"C3 Lambert (req.) : {C3_req:.2f}")
    print(f"Mejor dirección   : theta={best['theta']:.3f} rad, phi={best['phi']:.3f} rad")
    print(f"Distancia mínima nave-Júpiter @ llegada: {best['miss_km']:.0f} km")

    # === GRÁFICO 3D ===
    plotter = OrbitPlotter3D(plane=Planes.EARTH_ECLIPTIC)
    fig = plotter.plot_body_orbit(Earth, date_launch, label="Órbita de la Tierra")
    fig = plotter.plot_body_orbit(Jupiter, date_arrival, label="Órbita de Júpiter")

    # Trayectoria Lambert (óptima)
    fig = plotter.plot(lambert_trf, label="Transferencia Lambert E→J (óptima)")

    # Mejor trayectoria con C3 fijo
    fig = plotter.plot(best["orbit"], label=f"Nave (C3={C3_fixed.value:.1f} km²/s²)")
    fig = plotter.plot(best["arr"], label="Nave @ llegada (mejor caso)")

    fig.update_layout(
        title=f"Lambert vs mejor C3 fijo — C3_req≈{C3_req.value:.1f}, miss≈{best['miss_km']:.0f} km"
    )
    fig.show()


#if __name__ == '__main__':
    #Para ejecutar sólo uno,comenta o descomenta:
    #ejercicio_prueba_OrbitPlotter3D()
    #ejercicio_churi()
    #marte_clasico()
    #prueba()
    #prueba_2d
    #prueba_matplotlib()
    #hohmann_earth_to_mars("2025-01-01")
    #hohmann_earth_to_mars_3D(epoch_str="2025-01-01")
    #ejercicio_jupiter()

## Nuevos ejercicios para hacerme con poliastro.

from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

# Órbita circular de 7000 km de radio (≈ LEO)
orb = Orbit.circular(Earth, 7000 * u.km)
period = orb.period.to(u.hour)  # Esto imprime el periodo orbital en horas

print(f"Periodo de la órbita circular: {period}")
print(orb)  # imprime parámetros orbitales

