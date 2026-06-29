# ═══════════════════════════════════════════════════════════════════════════
#  SUITE DE PRUEBAS del propagador J2+drag (Pruebas_perturbaciones_optimizado)
#  Recorre los 9 cuerpos y verifica que el código funciona bien.
#
#  Uso:   python tests/test_propagador.py
#  Sale con código 0 si todo pasa, 1 si algún check falla.
#
#  Llama a las funciones internas del script SIN disparar el menú interactivo
#  ni abrir gráficas (backend Agg). No es física nueva: solo comprueba que el
#  propagador no se rompe, da valores finitos, detecta reentradas, etc.
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys
import warnings
import numpy as np

# Permitir importar los módulos del proyecto. Los scripts optimizados pueden
# estar en la raíz o dentro de scripts_importantes/ (según reorganización),
# así que añadimos ambas ubicaciones al path para ser robustos.
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts_importantes"))

import matplotlib
matplotlib.use("Agg")  # sin ventanas

from astropy import units as u
from poliastro.twobody import Orbit

import Pruebas_perturbaciones_optimizado as P
from Densidades_atmosferica_optimizado import PLANETAS

# Parámetros de un satélite genérico (A/m ≈ 0.02 m²/kg, tipo normal)
SAT = dict(masa=500.0, area=10.0, cd=2.2)

# Orden de recorrido (con atmósfera primero, sin atmósfera al final)
ORDEN = ["tierra", "marte", "venus", "jupiter", "saturno",
         "urano", "neptuno", "luna", "mercurio"]


def _propaga(pl, h0_km, inc_deg, dias, pasos):
    """Propaga y devuelve (t, pos, h, reentro) o lanza la excepción."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orb = Orbit.circular(pl.body, h0_km * u.km, inc=inc_deg * u.deg)
        t, pos, h, orbf, reentro = P.trayectoria_perturbada(
            pl, orb, dias * u.day, pasos, **SAT)
    return t, pos, h, reentro


def test_cuerpo(clave):
    """Devuelve lista de (nombre_check, ok:bool, detalle:str)."""
    pl = PLANETAS[clave]
    checks = []
    tiene_atm = pl.tiene_atmosfera

    # ── C0: consistencia de radios (modelo vs poliastro) ────────────────
    Rmod = pl.R_m / 1000.0
    Rpol = pl.body.R.to(u.km).value
    ok = abs(Rmod - Rpol) < 1.0
    checks.append(("radio R_m==poliastro", ok, f"Δ={Rmod-Rpol:+.2f} km"))

    # ── C1: densidad monótona y finita (solo con atmósfera) ─────────────
    if tiene_atm:
        h_top = pl.capas[0].h_min_km
        muestras = np.linspace(0, h_top, 25)
        rhos = [pl.get_rho(h * 1000.0, 0)[0] for h in muestras]
        finitas = all(np.isfinite(r) and r >= 0 for r in rhos)
        monot = all(rhos[i] <= rhos[i-1] * 1.0001 for i in range(1, len(rhos)))
        checks.append(("densidad finita+monótona", finitas and monot,
                       f"ρ(0)={rhos[0]:.2e} ρ(top)={rhos[-1]:.2e}"))
    else:
        rho0 = pl.get_rho(100_000.0, 0)[0]
        checks.append(("sin atmósfera → ρ=0", rho0 == 0.0, f"ρ={rho0}"))

    # ── Altitudes de prueba ─────────────────────────────────────────────
    h_re = pl.h_reentrada_m / 1000.0
    if tiene_atm:
        h_alta = pl.capas[0].h_min_km                 # exosfera: densidad mínima
        h_baja = h_re + max(50.0, 0.3 * h_re)         # cerca de reentrada
    else:
        h_alta = 200.0                                # órbita cómoda sin aire
        h_baja = 100.0

    # ── C2: Kepler ideal debe ser plano ─────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orb = Orbit.circular(pl.body, h_alta * u.km, inc=45 * u.deg)
            t, pos, h, orbf, _ = P.trayectoria_kepler(pl, orb, 3 * u.day, 40)
        var = float(h.max() - h.min())
        ok = np.all(np.isfinite(h)) and var < 1.0
        checks.append(("Kepler ideal plano", ok, f"variación={var:.3f} km"))
    except Exception as e:
        checks.append(("Kepler ideal plano", False, f"CRASH {type(e).__name__}: {e}"))

    # ── C3+C4: propagación perturbada en órbita alta (estable, finita) ──
    try:
        t, pos, h, reentro = _propaga(pl, h_alta, 45, 5, 50)
        finito = bool(np.all(np.isfinite(h)) and np.all(np.isfinite(pos)))
        # "No diverge": ningún punto sube por encima de una cota generosa.
        # La cota tolera la oscilación osculadora por J2 (que en los gigantes
        # llega a ~100 km) pero atrapa cualquier explosión numérica real.
        R_km = pl.R_m / 1000.0
        cota = h_alta + max(200.0, 0.05 * (R_km + h_alta))
        acotada = (len(h) < 2) or (float(np.max(h)) <= cota)
        checks.append(("propaga alta sin crash+finito", finito, f"npts={len(h)}"))
        checks.append(("órbita alta acotada (no diverge)", acotada,
                       f"h0={h[0]:.0f}→hf={h[-1]:.0f} km, h_max={float(np.max(h)):.0f}"))
    except Exception as e:
        checks.append(("propaga alta sin crash+finito", False,
                       f"CRASH {type(e).__name__}: {e}"))
        checks.append(("órbita alta acotada (no diverge)", False, "n/a"))

    # ── C5/C6: comportamiento en órbita baja ────────────────────────────
    try:
        t, pos, h, reentro = _propaga(pl, h_baja, 45, 20, 50)
        finito = bool(np.all(np.isfinite(h)))
        if tiene_atm:
            # con atmósfera: el drag debe hacerla decaer o reentrar
            decae = reentro or (len(h) >= 2 and h[-1] < h[0] - 5.0)
            checks.append(("drag hace decaer/reentrar", decae and finito,
                           f"reentró={reentro} h0={h[0]:.0f}→hf={h[-1]:.0f}"))
        else:
            # sin atmósfera: NO debe reentrar por drag; órbita estable y acotada
            estable = (not reentro) and finito and (len(h) >= 2) and \
                      (abs(h[-1] - h[0]) < 50.0)
            checks.append(("airless: estable solo J2 (sin drag)", estable,
                           f"reentró={reentro} h0={h[0]:.0f}→hf={h[-1]:.0f}"))
    except Exception as e:
        checks.append(("órbita baja sin crash", False,
                       f"CRASH {type(e).__name__}: {e}"))

    return checks


def main():
    print("=" * 80)
    print("  SUITE DE PRUEBAS — propagador J2+drag · 9 cuerpos")
    print("=" * 80)
    total = 0
    fallos = 0
    for clave in ORDEN:
        pl = PLANETAS[clave]
        etiq = "(con atm)" if pl.tiene_atmosfera else "(sin atm)"
        print(f"\n── {pl.nombre.upper()} {etiq} " + "─" * (60 - len(pl.nombre)))
        for nombre, ok, detalle in test_cuerpo(clave):
            total += 1
            if not ok:
                fallos += 1
            marca = "✓ PASS" if ok else "✗ FALLO"
            print(f"   [{marca}] {nombre:34s} {detalle}")

    print("\n" + "=" * 80)
    if fallos == 0:
        print(f"  RESULTADO: ✅ TODO OK — {total} checks superados en 9 cuerpos.")
    else:
        print(f"  RESULTADO: ❌ {fallos}/{total} checks FALLARON.")
    print("=" * 80)
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
