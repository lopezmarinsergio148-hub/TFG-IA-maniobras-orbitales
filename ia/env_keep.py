# ═══════════════════════════════════════════════════════════════════════════
#  ia/env_keep.py — Entorno Gymnasium de MANTENIMIENTO ORBITAL (station-keeping)
#
#  Al reves que el aerofrenado: aqui el satelite NO quiere caerse. Parte de una
#  orbita operativa (caso base: ISS, circular en LEO) y debe MANTENERLA dentro de
#  una banda de tolerancia durante toda la mision, luchando contra el arrastre,
#  gastando el minimo combustible.
#
#  Que erosiona y que se controla (lo defendible):
#   - El DRAG baja el semieje a (la orbita cae) y circulariza. Es lo que hay que
#     corregir con re-boost (impulso tangencial que devuelve energia).
#   - El J2 NO cambia a, e ni i: solo PRECESA la orientacion (Omega regresion nodal,
#     omega precesion del perigeo). Por eso NO se combate (seria carisimo e
#     irrealista; las heliosincronas hasta lo aprovechan). Aqui se MODELA y se
#     TRACKEA (cuanto ha girado el plano) para ensenarlo, pero no entra en la
#     recompensa. Decision de alcance acordada: mantener la GEOMETRIA (a, e, i).
#   - La i apenas la toca el drag (atmosfera rotante) -> mantenerla sale casi gratis;
#     se monitoriza. El plato fuerte real es el re-boost del semieje.
#
#  Fisica del decay (orbita circular), promediada por revolucion:
#     dE/dt = -1/2 (Cd*A/m) rho v^3   ->   da/dt = -(Cd*A/m) rho(h) v a
#  Se integra con sub-pasos dentro de cada paso porque rho cambia con la altura.
#
#  Episodio = MISION DE DURACION FIJA T: mantener la orbita en banda T tiempo con
#  el minimo Dv. 1 paso = un intervalo DT (varias orbitas). El agente decide cuanto
#  re-boostear cada intervalo (puede ser 0 -> el "timing" es la gracia: dejar caer
#  y subir a tiempo gasta menos que mantener rigido).
#
#  GENERALIZACION (objetivo final, como el aerofrenado): empezar por la Tierra (ISS)
#  y luego cualquier orbita / cualquier cuerpo. El estado va adimensionalizado
#  (normalizado por R y v_circular del cuerpo) + la densidad sentida, para que un
#  mismo agente transfiera entre planetas. Unidades internas: SI (m, m/s, s).
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 ENV_KEEP — entorno Gymnasium de mantenimiento orbital (station-keeping)
 Lo contrario del aerofrenado: el satélite defiende su órbita operativa contra el
 drag durante una misión de duración fija, re-boosteando lo justo para no salirse de
 la banda de tolerancia y gastando el mínimo Δv. El J2 se modela y se trackea (giro
 del plano) pero no se corrige. Estado adimensional para transferir entre cuerpos.

 ÍNDICE DE CLASES/FUNCIONES:
   - _dv_anual_mantenimiento(planeta, h): Δv/año para mantener una órbita circular.
   - _altura_para_dv(planeta, dv)       : altura cuyo Δv anual iguala un objetivo.
   - KeepEnv                            : entorno Gym; 1 paso = DT días de misión.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Atmosfera VALIDADA (clase Planeta) del modulo del Bloque 0/1.
_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_RAIZ, "scripts_importantes"))
from Densidades_atmosferica_optimizado import PLANETAS

# ── Constantes de la NAVE (satelite operativo normal, NO config de aerofrenado) ──
CD_A_M = 0.01            # Cd*A/m [m^2/kg] tipico de un satelite (el aerofrenado usa 0.22)

# ── Mision ───────────────────────────────────────────────────────────────────
DT_DIAS = 1.0           # duracion de un paso (dias): se re-evalua/corrige cada dia
T_MISION_DIAS = 365.0   # duracion de la mision (un ano); n_pasos = T/DT
BANDA_KM = 20.0         # tolerancia: la altitud debe quedarse a +-BANDA del objetivo
# Tope de re-boost por paso, DERIVADO de la banda: un re-boost maximo mueve el semieje
# K_DV * banda (ni mas). CLAVE (jun 2026): escalar dv_max por V_REF (1er intento) era un
# ERROR -> en los gigantes un re-boost a tope movia el semieje ~80 km, 4x la banda de
# 20 km, y la maniobra se PASABA por arriba (la heuristica fallaba por artefacto, no por
# fisica). Atando dv_max a la banda y a la dinamica local (da = 2a dv/v), la accion
# significa lo mismo (una fraccion de banda) en TODOS los cuerpos. Se calcula en reset()
# porque depende de la orbita objetivo del episodio.
K_DV = 0.5
# SIN tope de combustible (decision de diseno, jun 2026): el agente mantiene la banda
# gastando lo MINIMO y el Dv/ano resultante ES la metrica de salida; la viabilidad ("es
# mucho para una mision?") se juzga APARTE (memoria / capa LLM). Asi no hay un FUEL_MAX
# arbitrario que calibrar por planeta, y un "fallo" solo ocurre por salirse de banda.

# ── Pesos de la recompensa (normalizados para valer en cualquier cuerpo) ─────
# LECCION (reward hacking): con recompensa SOLO negativa por paso + premio al final,
# al agente le compensaba SALIRSE pronto (un castigo unico) antes que sumar negativos
# 365 dias -> se "suicidaba". Solucion: un BONUS por sobrevivir en banda CADA dia, para
# que mantenerse sea siempre mejor que rendirse. (Mismo espiritu que el gamma alto del
# aerofrenado: el premio de aguantar tiene que ser VISIBLE paso a paso.)
R_VIVO = 1.0           # premio por cada dia mantenido dentro de banda
W_DESV = 0.5           # castigo suave por desviacion del objetivo (unidades de banda)
C_DV = 2000.0          # castigo por combustible. Se aplica al Dv NORMALIZADO (dv/V_REF),
                       # adimensional -> los mismos pesos valen en cualquier cuerpo. LECCION
                       # fisica: el Dv de mantenimiento esta casi FIJADO por el drag; este
                       # coste solo afina el reparto, no permite "ahorrar" Dv de verdad.
PREMIO_EXITO = 100.0    # bonus final por completar la mision en banda
CASTIGO_FUERA = 200.0   # se salio de la banda (o reentro): siempre peor que aguantar

# ── Franja de altitudes OPERATIVAS, derivada de la atmosfera de cada cuerpo ───
# El rango de altitud NO puede ser fijo: a la misma altura, Marte (atmosfera tenue) casi
# no frena y la Tierra si. Se DERIVA por cuerpo: las altitudes donde mantener un ano
# cuesta entre DV_OP_BAJO (apenas hay que tocar) y DV_OP_ALTO (carisimo). Asi cada cuerpo
# ve el mismo "espectro de dificultad" y los resultados son comparables. Mismo espiritu
# que el corredor de perigeo del aerofrenado (derivado de la atmosfera + Q_MAX, no a ojo).
DV_OP_ALTO = 250.0      # Dv/ano en el borde BAJO de la franja (mantenimiento caro)
DV_OP_BAJO = 2.0        # Dv/ano en el borde ALTO de la franja (mantenimiento barato).
                        # Bajado de 10 a 2 (jun 2026) para estirar el borde alto a ~600 km
                        # en la Tierra y cubrir el LEO real (500-700 km). Es global: cada
                        # cuerpo llega a su altura de "2 m/s/ano" (mismo espectro de coste).


def _dv_anual_mantenimiento(planeta, h_m):
    """Dv [m/s] para mantener un ano una orbita circular a altura h (reponer el drag).
    Tasa de re-boost dv/dt = 1/2 (Cd*A/m) rho v^2  ->  *T_mision."""
    a = planeta.R_m + h_m
    v2 = planeta.mu_m3_s2 / a
    rho, _ = planeta.get_rho(h_m, 0)
    dvdt = 0.5 * CD_A_M * rho * v2
    return dvdt * (T_MISION_DIAS * 86400.0)


def _altura_para_dv(planeta, dv_objetivo):
    """Altura (m) donde el Dv anual de mantenimiento iguala dv_objetivo. dv decrece
    monotona con la altura (rho cae) -> biseccion (permite extrapolar sobre el techo)."""
    lo = planeta.capas[-1].h_min_km * 1000.0                          # baja altura -> dv alto
    hi = (planeta.capas[0].h_min_km + 40.0 * planeta.capas[0].H_km) * 1000.0  # alta -> dv bajo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _dv_anual_mantenimiento(planeta, mid) > dv_objetivo:
            lo = mid          # cuesta demasiado: subir la altura
        else:
            hi = mid
    return 0.5 * (lo + hi)


class KeepEnv(gym.Env):
    """Mantenimiento de una orbita operativa contra el drag. 1 paso = DT dias."""
    metadata = {"render_modes": []}

    def __init__(self, planeta="tierra", aleatorio=False):
        """Configura el entorno para un planeta: escalas de normalización, franja de
        altitudes operativas (derivada de su atmósfera), duración de la misión y los
        espacios de observación (5 números) y acción (magnitud de re-boost)."""
        super().__init__()
        if planeta not in PLANETAS or not PLANETAS[planeta].tiene_atmosfera:
            raise ValueError(f"'{planeta}' no es un planeta con atmosfera valido")
        self.planeta = PLANETAS[planeta]
        self.nombre = planeta
        self.MU = self.planeta.mu_m3_s2
        self.R = self.planeta.R_m
        self.J2 = self.planeta.J2
        # aleatorio=False -> caso fijo (ISS). aleatorio=True -> generalizar (futuro).
        self.aleatorio = aleatorio

        # Escalas de normalizacion propias del planeta (como en env_drag)
        self.V_REF = np.sqrt(self.MU / self.R)         # velocidad circular en superficie
        self.banda_m = BANDA_KM * 1000.0
        self.n_pasos = int(round(T_MISION_DIAS / DT_DIAS))
        self.dt_s = DT_DIAS * 86400.0

        # Franja de altitudes operativas DERIVADA de la atmosfera de este cuerpo
        # (donde mantener un ano cuesta entre DV_OP_BAJO y DV_OP_ALTO m/s).
        self.h_op_min = _altura_para_dv(self.planeta, DV_OP_ALTO)   # mas baja (caro)
        self.h_op_max = _altura_para_dv(self.planeta, DV_OP_BAJO)   # mas alta (barato)

        # Estado: [desv_a/banda, e, di_norm, t/T, rho_norm]  (5 numeros; sin fuel, ya no
        # hay tope de combustible). La densidad sentida (rho_norm) es la pista que permite
        # transferir entre cuerpos.
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(5,), dtype=np.float32)
        # Accion: 1 valor [-1,1] -> magnitud de re-boost este paso (0..dv_max)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.a = None            # semieje actual (m)
        self.a_obj = None        # semieje objetivo (m)
        self.e = None            # excentricidad (circular base ~0)
        self.i = None            # inclinacion (rad)
        self.i_obj = None
        self.dv_max = None       # tope de re-boost por paso (se fija en reset segun la orbita)
        self.dv_acum = None      # Dv total gastado en la mision (metrica de salida) [m/s]
        self.raan_acum = None    # Omega acumulado por J2 (solo tracking, rad)
        self.paso = 0

    # ── helpers de fisica ────────────────────────────────────────────────────
    def _h(self, a):
        """Altura sobre la superficie para un semieje (orbita circular)."""
        return a - self.R

    def _v_circ(self, a):
        """Velocidad circular para un semieje a [m/s]."""
        return np.sqrt(self.MU / a)

    def _da_dt(self, a):
        """Tasa de decaimiento del semieje por drag (orbita circular) [m/s]."""
        h = self._h(a)
        rho, _ = self.planeta.get_rho(h, 0)
        v = self._v_circ(a)
        return -CD_A_M * rho * v * a, rho

    def _raan_dot(self, a, e, i):
        """Regresion nodal secular por J2 [rad/s] (solo se trackea, no se corrige)."""
        if self.J2 <= 0 or a <= 0:        # a<=0: orbita ya no fisica (reentro) -> sin sqrt(neg)
            return 0.0
        n = np.sqrt(self.MU / a**3)
        p = a * (1.0 - e * e)
        return -1.5 * self.J2 * n * (self.R / p) ** 2 * np.cos(i) / (1.0 - e * e) ** 2

    # ── API gym ──────────────────────────────────────────────────────────────
    def _obs(self):
        """Observación (5 números): desviación de semieje en unidades de banda,
        excentricidad, desviación de inclinación, avance temporal y densidad sentida
        (log-normalizada, la pista que permite transferir entre cuerpos)."""
        rho, _ = self.planeta.get_rho(self._h(self.a), 0)
        # rho normalizada en log para que cubra ordenes de magnitud entre cuerpos
        rho_norm = np.log10(max(rho, 1e-30)) / 12.0
        return np.array([
            (self.a - self.a_obj) / self.banda_m,
            self.e,
            (self.i - self.i_obj) / np.radians(1.0),
            self.paso / self.n_pasos,
            rho_norm,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Reinicia la misión: fija la órbita objetivo (altitud e inclinación, por
        'options', aleatoria o caso ISS), calcula el tope de re-boost por paso y
        arranca clavado en el objetivo con contadores a cero."""
        super().reset(seed=seed)
        if options and "h_obj_km" in options:
            self.a_obj = self.R + float(options["h_obj_km"]) * 1000.0
            self.i_obj = np.radians(float(options.get("inc_deg", 51.6)))
        elif self.aleatorio:
            # GENERALIZACION: sortear altitud (en la franja operativa de ESTE cuerpo) e
            # inclinacion del objetivo. Asi el agente ve de orbitas baratas a carisimas.
            self.a_obj = self.R + float(self.np_random.uniform(self.h_op_min, self.h_op_max))
            self.i_obj = np.radians(float(self.np_random.uniform(0.0, 90.0)))
        else:
            # Caso base ISS: ~420 km, inclinacion 51.6 grados.
            self.a_obj = self.R + 420e3
            self.i_obj = np.radians(51.6)
        # Tope de re-boost: un impulso maximo mueve el semieje K_DV*banda (da = 2a dv/v).
        self.dv_max = K_DV * self.banda_m * self._v_circ(self.a_obj) / (2.0 * self.a_obj)
        self.a = self.a_obj            # arranca clavado en el objetivo
        self.e = 0.0
        self.i = self.i_obj
        self.dv_acum = 0.0
        self.raan_acum = 0.0
        self.paso = 0
        return self._obs(), {}

    def step(self, action):
        """Un intervalo DT de misión: aplica el re-boost del agente, propaga el decay
        por drag (con sub-pasos), trackea el J2, mide la desviación y da la recompensa
        (bonus por seguir en banda, castigo por Δv y por salirse/reentrar).
        Devuelve (obs, recompensa, terminado, truncado, info)."""
        self.paso += 1

        # 1) Re-boost que decide el agente (impulso tangencial, sube el semieje).
        dv = float(np.clip(action[0], -1.0, 1.0) + 1.0) * 0.5 * self.dv_max   # 0..dv_max
        # Correccion pequena de semieje circular: dv ~ 1/2 v (da/a) -> da = 2 a dv / v
        v = self._v_circ(self.a)
        self.a += 2.0 * self.a * dv / v
        self.dv_acum += dv

        # 2) Propagacion secular del decay durante DT (sub-pasos: rho cambia con h).
        n_sub = 24                       # 24 sub-pasos por dia (~horario)
        dt_sub = self.dt_s / n_sub
        for _ in range(n_sub):
            da_dt, _ = self._da_dt(self.a)
            self.a += da_dt * dt_sub
            if self._h(self.a) <= 0:     # se ha estrellado
                break

        # 3) J2: solo se trackea cuanto ha girado el plano (no se corrige).
        self.raan_acum += self._raan_dot(self.a, self.e, self.i) * self.dt_s

        # 4) Desviacion respecto al objetivo y recompensa.
        #    +R_VIVO por seguir en banda este dia (anti-suicidio) - desviacion - Dv (norm).
        desv = (self.a - self.a_obj) / self.banda_m          # en unidades de banda
        recompensa = R_VIVO - W_DESV * abs(desv) - C_DV * (dv / self.V_REF)
        info = {
            "planeta": self.nombre, "paso": self.paso,
            "h_km": self._h(self.a) / 1000.0, "h_obj_km": self._h(self.a_obj) / 1000.0,
            "desv_km": (self.a - self.a_obj) / 1000.0, "dv": dv,
            "dv_acum": self.dv_acum, "raan_deg": np.degrees(self.raan_acum),
        }

        # 5) Fin por fracaso: fuera de banda o reentrada.
        if abs(desv) > 1.0 or self._h(self.a) <= 0:
            return self._obs(), -CASTIGO_FUERA, True, False, {**info, "fallo": "fuera_banda"}

        # 6) Fin por exito: mision completada en banda.
        if self.paso >= self.n_pasos:
            return self._obs(), recompensa + PREMIO_EXITO, True, False, {**info, "exito": True}

        return self._obs(), recompensa, False, False, info


if __name__ == "__main__":
    # PRUEBA A MANO (no entrena): estrategia INGENUA de re-boost. Deja caer hasta
    # el borde inferior de la banda y entonces re-boostea de golpe (diente de sierra).
    # Sirve para ver que el decay y el re-boost salen fisicamente razonables ANTES
    # de entrenar, y como BASELINE ("juez") contra el que comparar al agente PPO.
    #   python env_keep.py [planeta]
    planeta = sys.argv[1] if len(sys.argv) > 1 else "tierra"
    env = KeepEnv(planeta=planeta, aleatorio=False)
    # Caso representativo de CUALQUIER cuerpo: el centro de su franja operativa.
    h_centro = 0.5 * (env.h_op_min + env.h_op_max) / 1000.0
    env.reset(options={"h_obj_km": h_centro, "inc_deg": 0.0})

    print("=" * 70)
    print(f"  STATION-KEEPING en {planeta.upper()}   (prueba a mano, estrategia ingenua)")
    print(f"  franja operativa: {env.h_op_min/1000:.0f} - {env.h_op_max/1000:.0f} km")
    print(f"  objetivo: h = {env._h(env.a_obj)/1000:.0f} km, i = {np.degrees(env.i_obj):.1f} deg"
          f"   banda +-{BANDA_KM:.0f} km")
    print(f"  mision: {T_MISION_DIAS:.0f} dias  |  Cd*A/m = {CD_A_M} m2/kg"
          f"  |  dv_max/paso = {env.dv_max:.2f} m/s")
    print("=" * 70)

    dv_total = 0.0
    term = trunc = False
    info = {}
    while not (term or trunc):
        # Ingenua: si esta en la mitad inferior de la banda, re-boost al maximo; si no, 0.
        desv_km = (env.a - env.a_obj) / 1000.0
        accion = np.array([1.0 if desv_km < -BANDA_KM * 0.5 else -1.0], dtype=np.float32)
        obs, rec, term, trunc, info = env.step(accion)
        dv_total += info["dv"]
        if info["paso"] % 30 == 0 or term or trunc:
            print(f"  dia {info['paso']:4d}:  h = {info['h_km']:7.2f} km"
                  f"   desv = {info['desv_km']:+6.2f} km   Dv acum = {info['dv_acum']:6.1f} m/s"
                  f"   Omega girado = {info['raan_deg']:+7.1f} deg")
    estado = "EXITO" if info.get("exito") else "FALLO (" + info.get("fallo", "?") + ")"
    print("-" * 70)
    print(f"  Resultado: {estado} en {info['paso']} dias")
    print(f"  Dv total gastado en mantener la orbita: {dv_total:.1f} m/s")
    print(f"  El plano (Omega) ha girado {info['raan_deg']:.1f} deg por J2 (NO se corrige).")
    print("=" * 70)
