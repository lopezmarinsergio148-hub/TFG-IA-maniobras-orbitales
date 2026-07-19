# ═══════════════════════════════════════════════════════════════════════════
#  ia/env_transfer.py — Entorno Gymnasium para la Fase 1 GENERALIZADA
#
#  Transferencia coplanar entre dos orbitas circulares CUALESQUIERA, en
#  CUALQUIER planeta, con un unico agente. La clave es trabajar en unidades
#  ADIMENSIONALES (analisis dimensional):
#
#     mu = 1,  r1 = 1,  v_c1 = sqrt(mu/r1) = 1
#
#  La fisica orbital no tiene escala propia: el optimo de Hohmann solo depende
#  del RATIO R = r2/r1 (ver baselines.hohmann_adim). Por eso el agente aprende
#  "la forma" de la maniobra (que fraccion de su velocidad orbital gastar), no
#  velocidades concretas. Una vez entrenado en estas unidades, la MISMA politica
#  vale para Marte, Jupiter, etc.: basta multiplicar su impulso adimensional por
#  la v_c1 real del planeta. Y cubre subir (R>1) y bajar (R<1) por simetria.
#
#  Estado:     [log(R) / log(R_SPAN)]   in [-1, 1]  (R=1 -> 0; subir>0; bajar<0)
#  Accion:     [a1, a2] in [-1, 1]^2  ->  dv1, dv2 in [-DV_ADIM_MAX, +DV_ADIM_MAX]
#  Recompensa: -error_relativo - C_DV*(|dv1|+|dv2|)   (misma filosofia Fase 1)
#
#  Es un episodio de UN paso ("bandit contextual"), igual que env_hohmann: el
#  agente da los dos impulsos tangenciales y el entorno hace el "coasting".
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════════
 ENV_TRANSFER — entorno Gymnasium de la Fase 1 GENERALIZADA (adimensional)
 Transferencia coplanar entre dos órbitas circulares cualesquiera, en cualquier
 planeta, con un único agente. Trabaja en unidades adimensionales (mu=1, r1=1), de
 modo que el óptimo depende solo del ratio R=r2/r1: la política aprendida es
 invariante de escala y cubre subir (R>1) y bajar (R<1). Episodio de un paso.

 ÍNDICE DE CLASES/FUNCIONES:
   - _map(a, lo, hi)   : reescala una acción [-1,1] al rango físico [lo, hi].
   - _unmap(dv, lo, hi): inversa de _map (impulso físico -> acción).
   - TransferEnv       : entorno Gym adimensional de un paso; acción = [dv1, dv2].
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Rangos del problema (adimensionales) ────────────────────────────────────
R_SPAN = 11.94      # ratio maximo de subida (y su inverso, de bajada). 11.94 es
                    # el limite clasico: por encima, la transferencia BIELIPTICA
                    # gana a Hohmann -> ahi Hohmann ya NO es el optimo, asi que no
                    # entrenamos fuera de ese rango (honestidad: el "juez" deja de
                    # ser valido). LEO->GEO (R=6.22) queda comodamente dentro.
LOG_RMAX = np.log(R_SPAN)   # normalizacion FIJA del estado (siempre el rango pleno)

# Limites FISICOS de cada impulso (en unidades de v_c1), ASIMETRICOS a proposito:
#  - dv1 (impulso en r1): tope +0.40 < (sqrt(2)-1)=0.414 -> el PRIMER impulso NUNCA
#    puede provocar escape. Asi se elimina el "acantilado" de castigo que hacia que
#    el agente huyera al extremo de dv1 y se quedara saturado. El suelo -0.80 cubre
#    la bajada profunda (R~1/12 pide dv1 ~ -0.61).
#  - dv2 (impulso en la apside de transferencia): rango [-1.40, +0.30]; bajar a
#    fondo pide frenar hasta ~1.24 v_c1, mientras que subir solo pide ~0.18.
DV1_LO, DV1_HI = -0.80, 0.40
DV2_LO, DV2_HI = -1.40, 0.30
C_DV = 0.01         # coste suave del combustible: LLEGAR domina (lección Fase 1:
                    # si el coste pesa mucho, el paisaje se aplana y no aprende).
ESCAPE_PEN = -2.0   # castigo si el 2.º impulso hace escapar (suave, no acantilado)
TOL_EXITO = 0.02    # tolerancia de "llegar" circular a R (2%)


def _map(a, lo, hi):
    """Reescala una accion en [-1, 1] al rango fisico [lo, hi]."""
    return lo + (np.clip(a, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)


def _unmap(dv, lo, hi):
    """Inversa de _map: del impulso fisico a la accion en [-1, 1]."""
    return (dv - lo) / (hi - lo) * 2.0 - 1.0


class TransferEnv(gym.Env):
    """Entorno de un paso, adimensional: el agente da [dv1, dv2] y se evalua."""
    metadata = {"render_modes": []}

    def __init__(self, aleatorio=True, r_span=R_SPAN):
        """Configura el entorno adimensional: modo (aleatorio o R fijo), el span de R
        del currículum y los espacios de observación (log R) y acción (2 impulsos)."""
        super().__init__()
        # aleatorio=True  -> sortea R en todo el rango (politica GENERAL): es el
        #   modo de entrenamiento (subir y bajar, cualquier ratio).
        # aleatorio=False -> R fijo = 6.22 (LEO->GEO), util para depurar.
        self.aleatorio = aleatorio
        # r_span limita el rango de R sorteado SOLO al entrenar (currículum): se
        # empieza con r_span pequeño (maniobras suaves) y se va ampliando. La
        # NORMALIZACION del estado usa siempre LOG_RMAX (rango pleno) para que la
        # escala que ve el agente no cambie entre etapas ni en evaluacion.
        self.log_span = np.log(r_span)
        # estado: log(R) normalizado a [-1, 1] (margen extra por seguridad)
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(1,), dtype=np.float32)
        #Con el observation le damos al PPO la información que va a recibir y el box indica valores continuos en el rango.
        # accion: 2 impulsos en [-1, 1] (se reescalan a sus rangos fisicos en step)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        #Con el action las decisiones que puede tomar el PPO y el shape=(2,) indica dos decisiones de salida.
        self.R = None

    def _obs(self):
        """Observación: log(R) normalizado por LOG_RMAX (R=1 -> 0; subir>0; bajar<0)."""
        return np.array([np.log(self.R) / LOG_RMAX], dtype=np.float32)

    def reset(self, *, seed=None, options=None): #El * obliga a que seed y options se indiquen por nombre.
        """Reinicia el episodio fijando el ratio R (por 'options', log-uniforme dentro
        del span del currículum, o el caso fijo LEO->GEO). Devuelve la observación."""
        #Además el seed sirve para fijar una semilla y que los valores aleatorios puedan repetirse y options sirve para pasar datos concretos al reiniciar, por ejemplo fijar un ratio R.
        super().reset(seed=seed) #Generador aleatorio del entorno.
        if options and "R" in options:
            self.R = float(options["R"]) #Si le das un valor de R, el entorno lo usará directamente.
        elif self.aleatorio:
            # log-uniforme dentro del span actual del currículum: subir (R>1) y
            # bajar (R<1) igual de probables (simetria)
            logR = self.np_random.uniform(-self.log_span, self.log_span)
            self.R = float(np.exp(logR))
        else:
            self.R = 6.22          # caso emblematico LEO -> GEO (r_geo / r_leo)
        return self._obs(), {}

    def step(self, action):
        """Aplica los dos impulsos con signo: crea la elipse de transferencia, hace el
        coasting a la otra ápside, ajusta la órbita final y mide el error logarítmico
        respecto a R. Penaliza escape/órbitas degeneradas.
        Devuelve (obs, recompensa, terminado, truncado, info)."""
        # Reescala cada accion [-1, 1] a su rango fisico (asimetrico, con signo)
        dv1 = float(_map(action[0], DV1_LO, DV1_HI))
        dv2 = float(_map(action[1], DV2_LO, DV2_HI))
        R = self.R
        info = {"dv1": dv1, "dv2": dv2, "dv_total": abs(dv1) + abs(dv2), "R": R}

        # --- Impulso 1 en r1 = 1 (tangencial, con signo: + progrado, - retrogrado) ---
        v1 = 1.0 + dv1
        energia1 = 0.5 * v1 * v1 - 1.0            # mu = 1, r1 = 1
        if energia1 >= 0.0:                        # se escaparia: fracaso
            return self._obs(), ESCAPE_PEN, True, False, {**info, "fallo": "escape_1"}
        a1 = -1.0 / (2.0 * energia1)
        r_t = 2.0 * a1 - 1.0                       # la OTRA apside de la elipse de transferencia
        if r_t <= 0.0:                             # orbita degenerada (cae al centro)
            return self._obs(), ESCAPE_PEN, True, False, {**info, "fallo": "apside_invalida"}

        # --- Coasting hasta r_t (apogeo si subimos, perigeo si bajamos) + impulso 2 ---
        v_t = np.sqrt(2.0 / r_t - 1.0 / a1)       # vis-viva en r_t (mu = 1)
        v2 = v_t + dv2
        energia2 = 0.5 * v2 * v2 - 1.0 / r_t
        if energia2 >= 0.0:
            return self._obs(), ESCAPE_PEN, True, False, {**info, "fallo": "escape_2"}
        a2 = -1.0 / (2.0 * energia2)
        r_otro = 2.0 * a2 - r_t                    # la otra apside de la orbita final
        if r_otro <= 0.0:                          # orbita final degenerada
            return self._obs(), ESCAPE_PEN, True, False, {**info, "fallo": "apside2_invalida"}
        r_peri_f = min(r_t, r_otro)
        r_apo_f = max(r_t, r_otro)

        # --- Recompensa: acabar circular en R, medido en escala LOGARITMICA ---
        # El error logaritmico es SIMETRICO (pasarse a 2R y quedarse en R/2 penalizan
        # IGUAL) y acotado O(1) en todo el rango. El error relativo lineal, en cambio,
        # se disparaba en las bajadas profundas (R<<1, error ~ (1-R)/R ~ 9) y reventaba
        # la funcion de valor del critico (value_loss enorme) -> el agente no aprendia.
        error = 0.5 * (abs(np.log(r_apo_f / R)) + abs(np.log(r_peri_f / R)))
        recompensa = -error - C_DV * (abs(dv1) + abs(dv2))
        exito = error < TOL_EXITO
        info.update({"error": error, "r_peri_f": r_peri_f,
                     "r_apo_f": r_apo_f, "exito": exito})
        return self._obs(), float(recompensa), True, False, info


if __name__ == "__main__":
    # PRUEBA A MANO (no entrena): alimenta el entorno con el OPTIMO adimensional
    # de Hohmann en varios ratios (subir y bajar) y comprueba que el error ~ 0.
    from baselines import hohmann_adim

    print("=" * 72)
    print("  PRUEBA A MANO del entorno ADIMENSIONAL con el optimo de Hohmann")
    print("=" * 72)
    print(f"  {'R':>7} {'tipo':>6} | {'dv1':>9} {'dv2':>9} {'dv_total':>9} | "
          f"{'error':>8} {'exito':>6}")
    print("-" * 72)
    for R in [11.0, 6.22, 2.0, 1.5, 0.7, 0.5, 0.2, 0.1]:
        env = TransferEnv()
        env.reset(options={"R": R})
        dv1_opt, dv2_opt, _ = hohmann_adim(R)
        accion = np.array([_unmap(dv1_opt, DV1_LO, DV1_HI),
                           _unmap(dv2_opt, DV2_LO, DV2_HI)], dtype=np.float32)
        _, rec, _, _, info = env.step(accion)
        tipo = "subir" if R > 1.0 else "bajar"
        print(f"  {R:7.3f} {tipo:>6} | {info['dv1']:+9.4f} {info['dv2']:+9.4f} "
              f"{info['dv_total']:9.4f} | {info['error'] * 100:7.3f}% "
              f"{str(info['exito']):>6}")
    print("=" * 72)
