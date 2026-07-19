# ═══════════════════════════════════════════════════════════════════════════
#  ia/env_drag.py — Entorno Gymnasium de AEROFRENADO multi-planeta (Fase 2)
#
#  El satélite llega en una órbita muy elíptica (apogeo alto) y debe BAJAR el
#  apogeo hasta una órbita objetivo aprovechando el rozamiento de la atmósfera
#  en cada paso por el perigeo (aerofrenado), gastando casi nada de combustible.
#
#  El agente controla, en cada pasada, la ALTITUD DE PERIGEO de la siguiente:
#    - perigeo bajo  -> mucho drag -> frena rápido, pero RIESGO de destrucción.
#    - perigeo alto  -> seguro, pero lentísimo.
#  Debe aprender a ir lo más profundo posible SIN pasarse (control en lazo cerrado).
#
#  GENERALIZACIÓN multi-planeta (jun 2026): un ESPECIALISTA por planeta (mismo
#  código, se elige el cuerpo con AeroBrakingEnv(planeta="...")). Usa la ATMÓSFERA
#  VALIDADA de cada cuerpo (clase Planeta del Bloque 0/1).
#
#  CRITERIO DE PELIGRO = PRESIÓN DINÁMICA, no altitud fija. La nave se destruye si
#  q = ½·ρ·v_perigeo² supera Q_MAX (límite estructural/térmico de la nave). Esto:
#    - es físicamente correcto (lo que rompe la nave es q, que crece con v²);
#    - es lo que usan las misiones reales de aerobraking (corredor de q);
#    - transfiere entre planetas (en Júpiter, a igual densidad, q es ENORME por la v).
#  Q_MAX = 0,6 N/m² = valor de referencia real (Mars Global Surveyor).
#
#  El corredor de perigeo (rango elegible) de cada planeta se DERIVA de su propia
#  atmósfera + Q_MAX, así que no se ponen altitudes "a ojo" por planeta.
#  Unidades internas: SI (m, m/s). La recompensa va normalizada para que los mismos
#  pesos valgan en todos los planetas (escalas de longitud/velocidad muy distintas).
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 ENV_DRAG — entorno Gymnasium de aerofrenado multi-planeta (Fase 2)
 Un especialista de RL por planeta que baja el apogeo aprovechando el drag en cada
 paso por el perigeo. El agente controla la altitud de perigeo (corredor derivado
 de la atmósfera + presión dinámica máxima Q_MAX) y debe frenar deprisa sin que la
 presión dinámica destruya la nave. Recompensa normalizada para valer en todos.

 ÍNDICE DE CLASES/FUNCIONES:
   - _q_dinamica(planeta, h, h_apo)   : presión dinámica q en un perigeo dado.
   - _altura_para_q(planeta, q, h_apo): altitud de perigeo que da una q objetivo.
   - AeroBrakingEnv                   : entorno Gym; 1 paso = 1 pasada por perigeo.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Importar la atmósfera VALIDADA (clase Planeta) del módulo del Bloque 0/1.
_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_RAIZ, "scripts_importantes"))
from Densidades_atmosferica_optimizado import PLANETAS

# ── Constantes de la NAVE (iguales en todos los planetas) ───────────────────
CD_A_M = 0.22             # Cd·A/m  [m²/kg] (configuración de aerofrenado)
Q_MAX = 0.6              # presión dinámica máxima [N/m²]: límite de la nave.
                         # Referencia real: Mars Global Surveyor (~0,6 N/m²).

# ── Escenario orbital: apogeos como FRACCIÓN del radio del planeta ───────────
# (la escala orbital depende del tamaño del cuerpo; el perigeo, en cambio, lo fija
#  la atmósfera vía la presión dinámica, no el radio).
F_APO_INI_MIN, F_APO_INI_MAX = 0.75, 2.35   # órbita de captura (muy elíptica)
F_APO_OBJ_MIN, F_APO_OBJ_MAX = 0.09, 0.35   # órbita objetivo (baja, casi circular)

# ── Umbrales de presión dinámica (múltiplos de Q_MAX) para derivar el corredor ─
Q_MORTAL = 2.0           # perigeo MÍNIMO elegible: q = 2·Q_MAX (zona mortal alcanzable,
                         #   para que el agente PUEDA pasarse y aprenda a no hacerlo)
Q_FLOJO = 30.0           # perigeo MÁXIMO elegible: q = Q_MAX/30 (frenado despreciable)
Q_INICIAL = 5.0          # perigeo inicial de captura: q = Q_MAX/5 (frenado leve, seguro)
MAX_PASADAS = 1200       # el aerobraking real son cientos-miles de orbitas (MRO ~445).
                         # Planetas con atmosfera tenue para su tamano (Tierra) necesitan
                         # muchas mas pasadas que Marte; 500 se quedaba corto.

# ── Pesos de la recompensa (NORMALIZADOS; calibrados con el Marte validado) ──
K_PROG = 800.0           # premio por progreso (bajada de apogeo / H_REF)
C_PASO = 1.0             # castigo por pasada (empuja a terminar rápido)
C_FUEL = 200.0           # castigo por Δv de ajuste de perigeo (normalizado por V_REF)
PREMIO_EXITO = 200.0
CASTIGO_PELIGRO = 50.0   # destrucción (moderado: no tan gigante que ahogue el gradiente)


def _q_dinamica(planeta, h_m, h_apo_m):
    """Presión dinámica q = ½·ρ·v_perigeo² [N/m²] en un perigeo h_m con apogeo dado."""
    R, MU = planeta.R_m, planeta.mu_m3_s2
    r_p, r_a = R + h_m, R + h_apo_m
    a = 0.5 * (r_p + r_a)
    v_p = np.sqrt(MU * (2.0 / r_p - 1.0 / a))
    rho, _ = planeta.get_rho(h_m, 0)
    return 0.5 * rho * v_p * v_p


def _altura_para_q(planeta, q_objetivo, h_apo_ref):
    """
    Altitud de perigeo (m) donde la presión dinámica iguala q_objetivo, con un
    apogeo de referencia. q decrece monótona con la altura -> bisección.
    Permite EXTRAPOLAR por encima del techo del modelo (la capa más alta se
    prolonga con su escala H; menos fiable, igual que en Urano/Neptuno).
    """
    lo = planeta.capas[-1].h_min_km * 1000.0                        # baja altura -> q alta
    hi = (planeta.capas[0].h_min_km + 40.0 * planeta.capas[0].H_km) * 1000.0  # alta -> q baja
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _q_dinamica(planeta, mid, h_apo_ref) > q_objetivo:
            lo = mid          # aún demasiada presión: subir el perigeo
        else:
            hi = mid
    return 0.5 * (lo + hi)


class AeroBrakingEnv(gym.Env):
    """Aerofrenado en un planeta. 1 paso = 1 pasada por el perigeo."""
    metadata = {"render_modes": []}

    def __init__(self, planeta="marte", aleatorio=True):
        """Configura el entorno para un planeta: escalas de normalización, corredor
        de perigeo (derivado de su atmósfera y Q_MAX) y espacios de obs./acción."""
        super().__init__()
        if planeta not in PLANETAS or not PLANETAS[planeta].tiene_atmosfera:
            raise ValueError(f"'{planeta}' no es un planeta con atmósfera valido")
        self.planeta = PLANETAS[planeta]
        self.nombre = planeta
        self.MU = self.planeta.mu_m3_s2
        self.R = self.planeta.R_m
        # aleatorio=True  -> escenario GENERAL (sortea apogeo inicial y objetivo).
        # aleatorio=False -> escenario fijo reproducible.
        self.aleatorio = aleatorio

        # Escalas de normalización propias del planeta
        self.H_REF = F_APO_INI_MAX * self.R          # longitud (apogeo de captura máximo)
        self.V_REF = np.sqrt(self.MU / self.R)       # velocidad (circular en superficie)

        # Corredor de perigeo DERIVADO de la atmósfera + Q_MAX (con el peor caso de v_p,
        # el apogeo inicial máximo). El peligro real se evalúa luego pasada a pasada.
        apo_ref = F_APO_INI_MAX * self.R
        self.H_PER_MIN = _altura_para_q(self.planeta, Q_MORTAL * Q_MAX, apo_ref)
        self.H_PER_MAX = _altura_para_q(self.planeta, Q_MAX / Q_FLOJO, apo_ref)
        self.H_PER_0 = _altura_para_q(self.planeta, Q_MAX / Q_INICIAL, apo_ref)

        # Estado: [apogeo, perigeo, apogeo_objetivo] normalizados por H_REF.
        self.observation_space = spaces.Box(low=0.0, high=3.0, shape=(3,), dtype=np.float32)
        # Acción: 1 valor en [-1,1] -> altitud de perigeo de la próxima pasada.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.h_apo = None
        self.h_per = None
        self.h_apo_objetivo = None
        self.pasos = 0

    def _obs(self):
        """Observación: [apogeo, perigeo, apogeo objetivo] normalizados por H_REF."""
        return np.array([self.h_apo / self.H_REF, self.h_per / self.H_REF,
                         self.h_apo_objetivo / self.H_REF], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Reinicia el episodio: fija apogeo inicial y objetivo (por 'options',
        aleatorio o caso histórico) y arranca el perigeo en su valor seguro."""
        super().reset(seed=seed)
        if options and "apo_ini_km" in options and "apo_obj_km" in options:
            # escenario FIJO dado en km (lo usa la herramienta del LLM)
            self.h_apo = float(options["apo_ini_km"]) * 1000.0
            self.h_apo_objetivo = float(options["apo_obj_km"]) * 1000.0
        elif self.aleatorio:
            self.h_apo = float(self.np_random.uniform(F_APO_INI_MIN * self.R,
                                                      F_APO_INI_MAX * self.R))
            obj_max = min(F_APO_OBJ_MAX * self.R, self.h_apo - 0.30 * self.R)
            self.h_apo_objetivo = float(self.np_random.uniform(F_APO_OBJ_MIN * self.R, obj_max))
        else:
            self.h_apo = 1.766 * self.R          # ~6000 km en Marte (caso histórico)
            self.h_apo_objetivo = 0.118 * self.R  # ~400 km en Marte
        self.h_per = self.H_PER_0
        self.pasos = 0
        return self._obs(), {}

    def _velocidad(self, r, a):
        """Velocidad orbital (vis-viva) a distancia r para semieje a [m/s]."""
        return np.sqrt(self.MU * (2.0 / r - 1.0 / a))

    def step(self, action):
        """Una pasada por el perigeo: fija el nuevo perigeo, cobra su Δv de ajuste,
        aplica el frenado (o destruye la nave si q > Q_MAX) y calcula la recompensa.
        Devuelve (obs, recompensa, terminado, truncado, info)."""
        self.pasos += 1
        # 1) El agente elige el perigeo de la próxima pasada (reescala [-1,1] -> corredor)
        h_per_nuevo = float(self.H_PER_MIN + (action[0] + 1.0) * 0.5 * (self.H_PER_MAX - self.H_PER_MIN))
        h_per_nuevo = float(np.clip(h_per_nuevo, self.H_PER_MIN, self.H_PER_MAX))

        # Coste en Δv de mover el perigeo (impulso en el apogeo actual)
        r_a = self.R + self.h_apo
        a_viejo = (r_a + (self.R + self.h_per)) / 2.0
        a_nuevo = (r_a + (self.R + h_per_nuevo)) / 2.0
        dv_ajuste = abs(self._velocidad(r_a, a_nuevo) - self._velocidad(r_a, a_viejo))
        self.h_per = h_per_nuevo

        info = {"planeta": self.nombre, "h_apo_km": self.h_apo / 1000,
                "h_per_km": self.h_per / 1000, "pasos": self.pasos, "dv_ajuste": dv_ajuste}

        # 2) Física de la pasada por el perigeo (King-Hele)
        r_p = self.R + h_per_nuevo
        a = (r_p + r_a) / 2.0
        v_p = self._velocidad(r_p, a)
        rho, idx = self.planeta.get_rho(h_per_nuevo, 0)

        # 3) ¿Presión dinámica por encima del límite de la nave? -> destrucción
        q_din = 0.5 * rho * v_p * v_p
        info["q_din"] = q_din
        if q_din > Q_MAX:
            return self._obs(), -CASTIGO_PELIGRO, True, False, {**info, "fallo": "destruido"}

        # 4) Frenazo por drag: baja el apogeo (el perigeo no cambia)
        H_escala = self.planeta.capas[idx].H_km * 1000.0
        L_eff = np.sqrt(2.0 * np.pi * a * H_escala)
        dv_drag = 0.5 * CD_A_M * rho * v_p * L_eff
        v_p_nueva = v_p - dv_drag
        energia = 0.5 * v_p_nueva**2 - self.MU / r_p
        a_post = -self.MU / (2.0 * energia)
        r_a_post = 2.0 * a_post - r_p
        h_apo_anterior = self.h_apo
        self.h_apo = r_a_post - self.R

        # 5) Recompensa (normalizada para que los pesos valgan en cualquier planeta)
        bajada = h_apo_anterior - self.h_apo                          # cuánto bajó el apogeo (m)
        recompensa = (K_PROG * (bajada / self.H_REF) - C_PASO
                      - C_FUEL * (dv_ajuste / self.V_REF))
        info.update({"dv_drag": dv_drag, "bajada_km": bajada / 1000,
                     "h_apo_km": self.h_apo / 1000})

        if self.h_apo <= self.h_apo_objetivo:                          # objetivo alcanzado
            recompensa += PREMIO_EXITO
            return self._obs(), recompensa, True, False, {**info, "exito": True}
        if self.pasos >= MAX_PASADAS:                                  # demasiadas pasadas
            return self._obs(), recompensa, False, True, {**info, "timeout": True}
        return self._obs(), recompensa, False, False, info


if __name__ == "__main__":
    # PRUEBA A MANO (no entrena): aerofrenado con perigeo FIJO en el planeta dado,
    # para ver que el apogeo desciende de forma razonable y se ve el corredor.
    #   python env_drag.py [planeta]
    planeta = sys.argv[1] if len(sys.argv) > 1 else "marte"
    env = AeroBrakingEnv(planeta=planeta, aleatorio=False)
    env.reset()

    print("=" * 64)
    print(f"  AEROFRENADO en {planeta.upper()}  (q_max = {Q_MAX} N/m2)")
    print(f"  corredor de perigeo: {env.H_PER_MIN/1000:.1f} .. {env.H_PER_MAX/1000:.1f} km"
          f"   (inicio {env.H_PER_0/1000:.1f} km)")
    print(f"  apogeo {env.h_apo/1000:.0f} km  ->  objetivo {env.h_apo_objetivo/1000:.0f} km")
    print("=" * 64)
    # Perigeo fijo a 1/3 del corredor desde abajo (frena bien, seguro)
    h_fijo = env.H_PER_MIN + 0.33 * (env.H_PER_MAX - env.H_PER_MIN)
    a_norm = (h_fijo - env.H_PER_MIN) / (env.H_PER_MAX - env.H_PER_MIN) * 2.0 - 1.0
    accion = np.array([a_norm], dtype=np.float32)
    term = trunc = False
    info = {}
    while not (term or trunc):
        obs, rec, term, trunc, info = env.step(accion)
        if info["pasos"] % 25 == 0 or term or trunc:
            print(f"  pasada {info['pasos']:3d}:  apogeo = {info['h_apo_km']:8.1f} km"
                  f"   perigeo {info['h_per_km']:.1f} km   q = {info.get('q_din', 0):.3f} N/m2")
    estado = "EXITO" if info.get("exito") else ("DESTRUIDO" if info.get("fallo") else "TIMEOUT")
    print("-" * 64)
    print(f"  Resultado: {estado} en {info['pasos']} pasadas")
    print("=" * 64)
