# ═══════════════════════════════════════════════════════════════════════════
#  ia/env_hohmann.py — Entorno Gymnasium para la Fase 1 (Hohmann LEO -> GEO)
#
#  Formulación "2 impulsos guiados": el agente elige los dos Δv tangenciales
#  (Δv1 en la órbita inicial y Δv2 al llegar al apogeo); el entorno se encarga
#  del "coasting" (la espera hasta el apogeo). Es un episodio de UN paso (un
#  "bandit contextual"): cada episodio se sortean el radio inicial r1 y el
#  objetivo r2, y el agente debe aprender la función (r1, r2) -> (Δv1, Δv2). La
#  solución óptima conocida es la de Hohmann (ia/baselines.py), que hace de juez.
#
#  Estado:     [r1/R_REF, r2/R_REF]      (radios normalizados)
#  Acción:     [a1, a2] in [-1, 1]^2  ->  Δv1, Δv2 in [0, DV_MAX]  (km/s)
#  Recompensa: -W*error_relativo - C_DV*(Δv1+Δv2)   (suave, sin escalón)
# ═══════════════════════════════════════════════════════════════════════════
#
#  NOTAS DE DISEÑO (lecciones del entrenamiento, jun 2026 — ver guía de defensa):
#  La forma final de la recompensa y del entrenamiento corrige 4 problemas reales
#  que fueron apareciendo:
#    1. REWARD HACKING: el agente se quedaba CORTO (no llegaba a GEO) para ahorrar
#       Δv → la recompensa debe premiar LLEGAR muy por encima de ahorrar.
#    2. ESQUINA MUERTA: con el coste de Δv alto el paisaje se aplanaba y el agente
#       aprendía a NO HACER NADA → C_DV muy pequeño (LLEGAR domina).
#    3. INESTABILIDAD: entrenar de más descuadraba la política → guardar el MEJOR
#       modelo (EvalCallback en train_ppo.py), no el último.
#    4. PREMIO ESCALÓN: un salto de recompensa al bajar del 2% de error
#       desestabilizaba → recompensa SUAVE y continua (sin escalón).
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from baselines import MU_TIERRA, R_TIERRA, velocidad_circular, H_LEO, H_GEO

# ── Rangos del problema (km) ────────────────────────────────────────────────
R1_MIN, R1_MAX = R_TIERRA + 300.0,   R_TIERRA + 1000.0    # órbita inicial (LEO)
R2_MIN, R2_MAX = R_TIERRA + 20000.0, R_TIERRA + 36000.0   # objetivo (MEO alto..GEO)
R_REF = R_TIERRA + 36000.0     # radio de referencia para normalizar (~GEO)
DV_MAX = 3.0       # Δv máximo por impulso (km/s) — cubre el óptimo (~2,4) con margen
W_ERROR = 1.0      # peso del error de llegada (recompensa escalada a O(1))
C_DV = 0.01        # peso MUY suave del coste en Δv: LLEGAR domina (si no, el paisaje
                   # se aplana y el agente no sabe hacia dónde ir). Reaching ⟹ Hohmann.
ESCAPE_PEN = -5.0  # castigo si un impulso hace escapar la nave
TOL_EXITO = 0.02   # tolerancia de "llegar" circular a r2 (2%)


class HohmannEnv(gym.Env):
    """Entorno de un paso: el agente da [Δv1, Δv2] y se evalúa la maniobra."""
    metadata = {"render_modes": []}

    def __init__(self, mu=MU_TIERRA, aleatorio=False):
        super().__init__()
        self.mu = mu #El self. guarda las variables en el entorno.
        # aleatorio=False -> entrena SOLO el caso LEO->GEO (objetivo fijo): el agente
        #   lo clava con precisión (prueba de concepto limpia de la Fase 1).
        # aleatorio=True  -> sortea r1, r2 (política general); más vistoso pero más
        #   difícil de clavar en el extremo GEO (sensibilidad). Queda como extensión.
        self.aleatorio = aleatorio
        # observación: 2 radios normalizados (valores ~0.15 a ~1.0)
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(2,), dtype=np.float32)
        # acción: 2 valores en [-1, 1] (se reescalan a [0, DV_MAX] km/s)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.r1 = None
        self.r2 = None

    def _obs(self): #Prepara la observación que recibe le agente
        return np.array([self.r1 / R_REF, self.r2 / R_REF], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Permite fijar r1, r2 (para evaluar en un caso concreto, p.ej. LEO->GEO)
        if options and "r1" in options and "r2" in options:
            self.r1 = float(options["r1"])
            self.r2 = float(options["r2"])
        elif self.aleatorio:
            self.r1 = float(self.np_random.uniform(R1_MIN, R1_MAX))
            self.r2 = float(self.np_random.uniform(R2_MIN, R2_MAX))
        else:
            self.r1 = R_TIERRA + H_LEO       # objetivo fijo: LEO -> GEO
            self.r2 = R_TIERRA + H_GEO
        return self._obs(), {}

    def step(self, action):
        # Reescala la acción [-1, 1] -> [0, DV_MAX]
        #El agente proporciona el impulso dando los valores entre [-1,1] y luego los transformamos con las líneas siguientes.
        #Es mejor de esta forma por las redes neuronales trabajan mejor con valores normalizados cercanos a 0.
        dv1 = float(np.clip((action[0] + 1.0) * 0.5 * DV_MAX, 0.0, DV_MAX)) #El np.clip es una protección extra que limita por si acaso el agente propociona un valor mayor a 1 o menor a -1.
        dv2 = float(np.clip((action[1] + 1.0) * 0.5 * DV_MAX, 0.0, DV_MAX))
        mu, r1, r2 = self.mu, self.r1, self.r2
        info = {"dv1": dv1, "dv2": dv2, "dv_total": dv1 + dv2, "r1": r1, "r2": r2}

        # --- Impulso 1 en r1 (prógrado): crea la elipse de transferencia ---
        v1 = velocidad_circular(r1, mu) + dv1
        energia1 = 0.5 * v1**2 - mu / r1
        if energia1 >= 0.0:                       # se escaparía: fracaso
            return self._obs(), ESCAPE_PEN, True, False, {**info, "fallo": "escape_1"}
        a1 = -mu / (2.0 * energia1)
        r_apo = 2.0 * a1 - r1                     # apogeo (r1 actúa de perigeo)

        # --- Coasting hasta el apogeo + impulso 2 (prógrado) ---
        v_apo = np.sqrt(mu * (2.0 / r_apo - 1.0 / a1))
        v2 = v_apo + dv2
        energia2 = 0.5 * v2**2 - mu / r_apo
        if energia2 >= 0.0:
            return self._obs(), ESCAPE_PEN, True, False, {**info, "fallo": "escape_2"}
        a2 = -mu / (2.0 * energia2)
        r_otro = 2.0 * a2 - r_apo                 # la otra ápside de la órbita final
        r_peri_f = min(r_apo, r_otro)
        r_apo_f = max(r_apo, r_otro)

        # --- Recompensa: gastar poco Y acabar circular en r2 ---
        error_rel = (abs(r_apo_f - r2) + abs(r_peri_f - r2)) / (2.0 * r2)
        # Recompensa SUAVE: domina LLEGAR (−error), con un empujón leve a ahorrar Δv.
        # Sin "premio escalón" (creaba un salto que desestabilizaba el aprendizaje).
        recompensa = -W_ERROR * error_rel - C_DV * (dv1 + dv2)
        exito = error_rel < TOL_EXITO
        info.update({"error_rel": error_rel, "r_peri_f": r_peri_f,
                     "r_apo_f": r_apo_f, "exito": exito})
        return self._obs(), float(recompensa), True, False, info


if __name__ == "__main__":
    # PRUEBA A MANO (no entrena): se alimenta el entorno con el ÓPTIMO de Hohmann
    # y se comprueba que el error es ~0 y la recompensa la esperada.
    from baselines import hohmann_leo_geo, H_LEO, H_GEO
    opt = hohmann_leo_geo()
    r1 = R_TIERRA + H_LEO
    r2 = R_TIERRA + H_GEO

    env = HohmannEnv()
    env.reset(options={"r1": r1, "r2": r2})
    # Convierte los Δv óptimos a acción en [-1, 1]
    a1 = opt.dv1 / DV_MAX * 2.0 - 1.0
    a2 = opt.dv2 / DV_MAX * 2.0 - 1.0
    obs, rec, term, trunc, info = env.step(np.array([a1, a2], dtype=np.float32))

    print("=" * 62)
    print("  PRUEBA A MANO del entorno con el optimo de Hohmann")
    print("=" * 62)
    print(f"  dv1={info['dv1']:.4f}  dv2={info['dv2']:.4f}  dv_total={info['dv_total']:.4f} km/s")
    print(f"  r2 objetivo   = {r2:.1f} km")
    print(f"  perigeo final = {info['r_peri_f']:.1f} km")
    print(f"  apogeo  final = {info['r_apo_f']:.1f} km")
    print(f"  error relativo= {info['error_rel'] * 100:.4f} %    exito = {info['exito']}")
    print(f"  recompensa    = {rec:.4f}")
    print("=" * 62)
