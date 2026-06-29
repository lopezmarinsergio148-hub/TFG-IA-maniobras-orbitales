import plotly.io as pio  #Sirve para representar las figuras
pio.renderers.default = "browser" #Esto nos ayuda a representar las figuras ya que con esta línea le
                                  #decimos que abra un HTML interactivo en el navegador


from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import OrbitPlotter3D
import numpy as np
import plotly.graph_objects as go

"""
Estos primeros ejercicios son solo con órbitas, para ver como representarlas.
"""


def orbita_simple():
    orb = Orbit.circular(Earth, Earth.R + 700 * u.km) #Crea una orbita circular de radio 7000 km (Unos 700 km de altura).

    plotter = OrbitPlotter3D() #Nos crea una figura 3D basado en Plotly.
    plotter.set_attractor(Earth) #Cuerpo central la Tierra.

    fig = plotter.plot(orb, label="LEO 7000 km") #Dibuja la orbita en el plotter y devuelve una figura Plotly.
    fig.update_layout(title="Ejercicio 1: órbita circular") #Modificamos la gráfica para meterle un título.
    fig.show() #Muestra la figura usando el renderer configurado (browser).

    #input("Pulsa Enter para cerrar...")  # evita que el script termine instantáneo


def parametros_orbitales():
#En este ejercicio simplemente vemos como se sacan los parámetros de la órbita
    orb = Orbit.circular(Earth, (700 * u.km)) #Esos 700 km es la altura de la órbita.

    print("a (semieje mayor):", orb.a) #Al ser circular me dará el valor del radio también.
    print("ecc (excentricidad):", orb.ecc)
    print("inc (inclinación):", orb.inc)
    print("periodo:", orb.period.to(u.minute))
    print("r (posición):", orb.r) #Nos dará el radio de la órbita que es la suma de la Tierra y la altura que le pongamos.
    print("v (velocidad):", orb.v)


def orbita_eliptica():

    # Órbita elíptica: a y e
    orb_e = Orbit.from_classical(       #Con esta línea creamos una orbita con los parámetros clásicos.
        Earth,   #Cuerpo atractor y ahora todos los parámetros los fijamos de acuerdo a la Tierra.
        10000 * u.km,   # semieje mayor (a)
        0.4 * u.one,    # excentricidad (e)
        0 * u.deg,      # inclinación  (i)
        0 * u.deg,      # RAAN  (Lóngitud del nodo ascendente)
        0 * u.deg,      # arg periapsis  (omega)
        0 * u.deg       # anomaly (Anomalía verdadera inicial)
    )

    plotter = OrbitPlotter3D()  #Nos crea una figura 3D basado en Plotly.
    plotter.set_attractor(Earth)  #Cuerpo central la Tierra.

    fig = plotter.plot(orb_e, label="Elíptica a=10000km e=0.4")  #Dibuja la orbita en el plotter y devuelve una figura Plotly.
    fig.update_layout(title="Ejercicio 3: Órbita elíptica")  #Modificamos la gráfica para meterle un título.
    fig.show() #Muestra la figura usando el renderer configurado (browser).


"""
Con estos ejercicios ya vemos algo de propagación en la órbita y también algo de impulsos (Tangencial,
radial, normal...) para conseguir movernos de una órbita a otra. 
"""


def propagacion():

    orb = Orbit.from_classical(
        Earth, 15000 * u.km, 0.8 * u.one,
        0 * u.deg, 0 * u.deg, 0 * u.deg, 0 * u.deg
    )

    orb_30m = orb.propagate(30 * u.minute)
    orb_60m = orb.propagate(60 * u.minute)

    plotter = OrbitPlotter3D()
    plotter.set_attractor(Earth)

    # 1) Dibujar órbitas
    fig = plotter.plot(orb, label="t = 0")
    fig = plotter.plot(orb_30m, label="t = 30 min")
    fig = plotter.plot(orb_60m, label="t = 60 min")

    # 2) Quitar esferas (Tierra y marcadores grandes internos)
    fig.data = tuple(tr for tr in fig.data if tr.type not in ("surface", "mesh3d"))

    # 3) Añadir puntos satélite con tamaño fijo
    def add_point(fig, orb, name, color, size=20):
        # Extraemos el vector de posición r (x, y, z) en kilómetros
        # orb.r devuelve un vector con unidades (astropy.units)
        r = orb.r.to(u.km).value
        fig.add_scatter3d(
            x=[r[0]], y=[r[1]], z=[r[2]],
            mode="markers",
            marker=dict(size=size, color=color),
            name=name
        )

    add_point(fig, orb, "Sat t=0", "blue", 20)
    add_point(fig, orb_30m, "Sat t=30", "orange", 20)
    add_point(fig, orb_60m, "Sat t=60", "green", 20)

    # 4) Añadir flechas de velocidad (conos)
    def add_velocity_arrow(fig, orb, name, scale_factor=800):
        # Posición en km
        r = orb.r.to(u.km).value

        # Velocidad en km/s
        v = orb.v.to(u.km / u.s).value

        # Módulo real de la velocidad
        speed = np.linalg.norm(v)

        # Escalado proporcional al módulo
        uu, vv, ww = v * scale_factor #Se descompone la v en sus tres ejes y se multiplica por un factor
        #ya que las velocidades orbitales son pequeñas en comparación a las distancias.

        fig.add_trace(go.Cone(  #Con el go.cone usamos un cono y le da aspecto de vector de fuerza.
            x=[r[0]], y=[r[1]], z=[r[2]],
            u=[uu], v=[vv], w=[ww],
            anchor="tail", #Con esto conseguimos que la base del cono está pegada al satélite y
            #la punta señala hacia el futuro de la trayectoria.
            sizemode="absolute", #El tamaño de la flecha debe basarse en unidades reales del gráfico
            #no en un porcentaje de la pantalla.
            sizeref=1500, #Factor de escala visual.
            showscale=False, #Desactiva la barra de colores que se suele poner al lado
            name=name
        ))

    add_velocity_arrow(fig, orb, "v(t=0)")
    add_velocity_arrow(fig, orb_30m, "v(t=30)")
    add_velocity_arrow(fig, orb_60m, "v(t=60)")

    # 5) Mostrar al final (una sola vez)
    fig.update_layout(title="Propagación: órbita + puntos + vectores velocidad")
    fig.show()

    # 6) Imprimir velocidades para comprobar
    def speed(orb):
        v = orb.v.to(u.km/u.s).value
        return np.linalg.norm(v) * (u.km/u.s)

    print("v(t=0):", orb.v.to(u.km/u.s), "||v||=", speed(orb))
    print("v(t=30):", orb_30m.v.to(u.km/u.s), "||v||=", speed(orb_30m))
    print("v(t=60):", orb_60m.v.to(u.km/u.s), "||v||=", speed(orb_60m))


def impulso_simple_tangencial():

    import numpy as np
    from poliastro.maneuver import Maneuver

    # 1) Órbita circular inicial
    orb0 = Orbit.circular(Earth, 7000 * u.km)

    # 2) Magnitud del impulso
    dv_mag = 1500 * u.m / u.s #Metemos el impulso

    # 3) Dirección tangencial (dirección de la velocidad), en forma NUMÉRICA (sin unidades)
    v_vec = orb0.v.to(u.m / u.s).value  # numpy array (3,) sin unidades, el comando orb0.v nos da el vector velocidad con unidades
    # y el comando .to(u.m / u.s).value lo convierte a un número puro.
    v_hat = v_vec / np.linalg.norm(v_vec)  # unitario sin unidades, el denominador es el módulo y así se obtine el unitario.
    #Así aplicamos el impulso en la misma dirección que en la que se mueve el satélite.

    # 4) Vector Δv con unidades (m/s) y forma (3,)
    dv_vec = (dv_mag.value * v_hat) * (u.m / u.s) #Construimos el vector del impulso, cogemos la dirección tangencial (v_hat)
    # y la multiplicamos por la magnitud del impulso y le ponemos unidades a todo

    # 5) Crear maniobra y aplicar
    man = Maneuver.impulse(dv_vec)   #Creamos el impulso
    orb1 = orb0.apply_maneuver(man)  #Aplicamos el impulso en la orbita inicial

    print("Δv aplicado:", man.get_total_cost().to(u.m / u.s)) #Imprimimos el valor del impulso aplicado.

    # 6) Graficar antes y después
    plotter = OrbitPlotter3D()
    plotter.set_attractor(Earth)

    dv_value = man.get_total_cost().to(u.m / u.s)  # Esto solo lo hago para poder ponerlo bien en la leyenda luego.
    fig = plotter.plot(orb0, label="Antes (circular)")
    fig = plotter.plot(orb1, label=f"Después (Δv = {dv_value})")
    fig.update_layout(title="Ejercicio 5: Impulso tangencial")
    fig.show()


def impulso_simple_radial():

    import numpy as np
    from poliastro.maneuver import Maneuver

    # 1) Órbita circular inicial
    orb0 = Orbit.circular(Earth, 7000 * u.km)

    # 2) Magnitud del impulso
    dv_mag = 1500 * u.m / u.s  # Metemos el impulso

    # 3) Dirección tangencial (dirección de la velocidad), en forma NUMÉRICA (sin unidades)
    r_vec = orb0.r.to(u.km).value  # numpy array (3,) sin unidades, el comando orb0.v nos da el vector velocidad con unidades
    # y el comando .to(u.m / u.s).value lo convierte a un número puro.
    r_hat = r_vec / np.linalg.norm(r_vec)  # unitario sin unidades, el denominador es el módulo y así se obtine el unitario.
    # Así aplicamos el impulso en la misma dirección que en la que se mueve el satélite.

    # 4) Vector Δv con unidades (m/s) y forma (3,)
    dv_vec = (dv_mag.value * r_hat) * (u.m / u.s)  # Construimos el vector del impulso, cogemos la dirección tangencial (v_hat)
    # y la multiplicamos por la magnitud del impulso y le ponemos unidades a todo

    # 5) Crear maniobra y aplicar
    man = Maneuver.impulse(dv_vec)  # Creamos el impulso
    orb1 = orb0.apply_maneuver(man)  # Aplicamos el impulso en la orbita inicial

    print("Δv aplicado:", man.get_total_cost().to(u.m / u.s))  # Imprimimos el valor del impulso aplicado.

    # 6) Graficar antes y después
    plotter = OrbitPlotter3D()
    plotter.set_attractor(Earth)

    dv_value = man.get_total_cost().to(u.m / u.s)  # Esto solo lo hago para poder ponerlo bien en la leyenda luego.
    fig = plotter.plot(orb0, label="Antes (circular)")
    fig = plotter.plot(orb1, label=f"Después (Δv = {dv_value})")
    fig.update_layout(title="Ejercicio 5: Impulso radial")
    fig.show()


def impulso_simple_normal():

    import numpy as np
    from poliastro.maneuver import Maneuver

    # 1) Órbita circular inicial
    orb0 = Orbit.circular(Earth, 7000 * u.km)

    # 2) Magnitud del impulso
    dv_mag = 1500 * u.m / u.s  # Metemos el impulso

    # 3) Dirección tangencial (dirección de la velocidad), en forma NUMÉRICA (sin unidades)
    r = orb0.r.to(u.km).value
    v = orb0.v.to(u.km / u.s).value

    h_vec = np.cross(r, v) #Nos da un vetor que es perpendicular tanto a la posición como a la velocidad
    #ya que hace el producto vectorial.
    h_hat = h_vec / np.linalg.norm(h_vec)

    dv_vec = (dv_mag.value * h_hat) * (u.m / u.s)  # normal (cambia inclinación)

    # 5) Crear maniobra y aplicar
    man = Maneuver.impulse(dv_vec)  # Creamos el impulso
    orb1 = orb0.apply_maneuver(man)  # Aplicamos el impulso en la orbita inicial

    print("Δv aplicado:", man.get_total_cost().to(u.m / u.s))  # Imprimimos el valor del impulso aplicado.

    # 6) Graficar antes y después
    plotter = OrbitPlotter3D()
    plotter.set_attractor(Earth)

    dv_value = man.get_total_cost().to(u.m / u.s)  # Esto solo lo hago para poder ponerlo bien en la leyenda luego.
    fig = plotter.plot(orb0, label="Antes (circular)")
    fig = plotter.plot(orb1, label=f"Después (Δv = {dv_value})")  # Después del value:.0f y redondea
    fig.update_layout(title="Ejercicio 5: Impulso normal")
    fig.show()


def impulso_simple_cualquierdireccion():

    import numpy as np
    from poliastro.maneuver import Maneuver

    # 1) Órbita circular inicial
    orb0 = Orbit.circular(Earth, 7000 * u.km)

    # 2) Magnitud del impulso
    dv_mag = 1500 * u.m / u.s  #Metemos el impulso

    # 3) Dirección tangencial (dirección de la velocidad), en forma NUMÉRICA (sin unidades)
    d = np.array([1.0, 1.0, 0.2])
    d_hat = d / np.linalg.norm(d)

    dv_vec = (dv_mag.value * d_hat) * (u.m / u.s)

    # 5) Crear maniobra y aplicar
    man = Maneuver.impulse(dv_vec)  # Creamos el impulso
    orb1 = orb0.apply_maneuver(man)  # Aplicamos el impulso en la orbita inicial

    print("Δv aplicado:", man.get_total_cost().to(u.m / u.s))  # Imprimimos el valor del impulso aplicado.

    # 6) Graficar antes y después
    plotter = OrbitPlotter3D()
    plotter.set_attractor(Earth)

    dv_value = man.get_total_cost().to(u.m / u.s)  # Esto solo lo hago para poder ponerlo bien en la leyenda luego.
    fig = plotter.plot(orb0, label="Antes (circular)")
    fig = plotter.plot(orb1, label=f"Después (Δv = {dv_value})")  # Después del value:.0f y redondea
    fig.update_layout(title="Ejercicio 5: Impulso cualquier dirección")
    fig.show()


def impulso_simple_cualquierdireccionpro():

    import numpy as np
    from poliastro.maneuver import Maneuver

    # 1) Órbita circular inicial
    orb0 = Orbit.circular(Earth, 7000 * u.km)

    # 2) Magnitud del impulso
    dv_mag = 1500 * u.m / u.s  # Metemos el impulso

    # 3) Dirección tangencial (dirección de la velocidad), en forma NUMÉRICA (sin unidades)
    r = orb0.r.to(u.km).value
    v = orb0.v.to(u.km / u.s).value

    R_hat = r / np.linalg.norm(r)
    T_hat = v / np.linalg.norm(v)
    N_hat = np.cross(R_hat, T_hat)
    N_hat = N_hat / np.linalg.norm(N_hat)

    # Componentes en RTN (elige tú)
    dv_R = 200  # m/s radial
    dv_T = 100  # m/s tangencial
    dv_N = 1500  # m/s normal

    dv_vec = (dv_R * R_hat + dv_T * T_hat + dv_N * N_hat) * (u.m / u.s)

    # 5) Crear maniobra y aplicar
    man = Maneuver.impulse(dv_vec)  # Creamos el impulso
    orb1 = orb0.apply_maneuver(man)  # Aplicamos el impulso en la orbita inicial

    print("Δv aplicado:", man.get_total_cost().to(u.m / u.s))  # Imprimimos el valor del impulso aplicado.

    # 6) Graficar antes y después
    plotter = OrbitPlotter3D()
    plotter.set_attractor(Earth)

    dv_value = man.get_total_cost().to(u.m / u.s) #Esto solo lo hago para poder ponerlo bien en la leyenda luego.
    fig = plotter.plot(orb0, label="Antes (circular)")
    fig = plotter.plot(orb1, label=f"Después (Δv = {dv_value})") #Después del value:.0f y redondea
    fig.update_layout(title="Ejercicio 5: Impulso cualquier dirección pro")


    # 1) Quitar esferas (mesh/surface)
    #fig.data = tuple(tr for tr in fig.data if tr.type not in ("surface", "mesh3d"))


    def add_marker(fig, orb, name, size=15):
        r = orb.r.to(u.km).value
        fig.add_scatter3d(x=[r[0]], y=[r[1]], z=[r[2]],
                      mode="markers",
                      marker=dict(size=size),
                      name=name)

    add_marker(fig, orb0, "Sat inicial", 15)

    fig.show()


"""
Este ejercicio era una comprobación
"""



def impulso_pro_dos_orbitas_limpio():
    import numpy as np
    from poliastro.maneuver import Maneuver
    from poliastro.plotting import OrbitPlotter3D
    import plotly.graph_objects as go
    from astropy import units as u
    from poliastro.bodies import Earth
    from poliastro.twobody import Orbit

    # 1) Órbita inicial y Maniobra
    orb0 = Orbit.circular(Earth, 7000 * u.km)

    # Supongamos tu maniobra (ejemplo rápido para que veas la segunda órbita)
    dv_vec = [0.5, 0.5, 1.5] * u.km / u.s
    man = Maneuver.impulse(dv_vec)
    orb1 = orb0.apply_maneuver(man)

    # 2) Graficar AMBAS con Poliastro
    plotter = OrbitPlotter3D()
    fig = plotter.plot(orb0, label="Trayectoria Inicial")
    fig = plotter.plot(orb1, label="Trayectoria Post-Impulso")

    # 3) LIMPIEZA TOTAL: Solo nos quedamos con las líneas (las dos órbitas)
    # Al hacer esto, la esferas de Poliastro desaparece para siempre
    fig.data = tuple(tr for tr in fig.data if getattr(tr, 'mode', None) == 'lines')

    # 4) AÑADIMOS TUS MARCADORES PERSONALIZADOS
    # Marcador Tierra
    fig.add_scatter3d(x=[0], y=[0], z=[0],
                      mode="markers", marker=dict(size=15, color="blue"),
                      name="TIERRA_MARKER")

    # Marcador Satélite (en la posición de ignición)
    r_sat = orb0.r.to(u.km).value
    fig.add_scatter3d(x=[r_sat[0]], y=[r_sat[1]], z=[r_sat[2]],
                      mode="markers", marker=dict(size=10, color="red"),
                      name="SATELITE_MARKER")

    # 5) MENÚ INTERACTIVO (Asegurando que vea todas las líneas)
    traces = fig.data

    # Función para generar la máscara de visibilidad
    def get_vis(logic_func):
        return [logic_func(tr) for tr in traces]

    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {"label": "Ver Todo", "method": "update", "args": [{"visible": [True] * len(traces)}]},
                {"label": "Solo Órbitas", "method": "update",
                 "args": [{"visible": get_vis(lambda t: t.mode == 'lines')}]},
                {"label": "Solo satélite", "method": "update",
                 "args": [{"visible": get_vis(lambda t: t.name != "TIERRA_MARKER")}]},
                {"label": "Solo Tierra", "method": "update",
                 "args": [{"visible": get_vis(lambda t: t.name != "SATELITE_MARKER")}]}
            ],
            "direction": "down", "x": 0.1, "y": 1.15
        }],
        title="IA de Trayectorias: Simulación de Maniobra Limpia"
    )

    fig.show()


if __name__ == "__main__":
    #orbita_simple()
    #parametros_orbitales()
    #orbita_eliptica()
    #propagacion()
    #impulso_simple_tangencial()
    #impulso_simple_radial()
    #impulso_simple_normal()
    #impulso_simple_cualquierdireccion()
    #impulso_simple_cualquierdireccionpro()
    #crear_animacion_interactiva_tfg()
    impulso_pro_dos_orbitas_limpio()



