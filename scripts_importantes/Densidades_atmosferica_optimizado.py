"""
═══════════════════════════════════════════════════════════════════════════════
 DENSIDADES ATMOSFÉRICAS — VERSIÓN OPTIMIZADA
 Simulador unificado de decaimiento orbital por drag para 9 cuerpos del
 sistema solar (Tierra, Marte, Venus, Júpiter, Saturno, Urano, Neptuno, Mercurio, Luna).

 Refactor de `Densidades_atmosferica.py` que mantiene la misma física,
 los mismos números y los mismos resultados, pero:
   - Una sola función `simular()` y una sola `graficar()` para todos.
   - Constantes físicas (M, R, J2) tomadas de `poliastro.bodies`.
   - Modelo atmosférico de cada planeta como una "ficha técnica" limpia.
   - Selección de planeta vía menú interactivo (sin comentar/descomentar).

 USO BÁSICO desde otros archivos:
     from Densidades_atmosferica_optimizado import JUPITER, PLANETAS
     rho, _ = JUPITER.get_rho(1000_000)        # densidad a 1000 km
     mu     = JUPITER.mu_m3_s2                 # GM en m³/s²

 USO INTERACTIVO:
     python Densidades_atmosferica_optimizado.py
     → menú para elegir planeta y ejecutar simulación.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from poliastro.bodies import (
    Earth, Mars, Venus, Jupiter, Saturn, Uranus, Neptune, Moon, Mercury
)

# Reconfigura stdout a UTF-8 para que los emojis (🔥, ⚠️) funcionen también en
# consolas Windows (cmd.exe / PowerShell) que por defecto usan cp1252.
# En PyCharm normalmente ya está en UTF-8, pero esto lo blinda en cualquier shell.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES GLOBALES
# ═══════════════════════════════════════════════════════════════════════════

G = 6.67430e-11        # Constante gravitacional universal (m³ kg⁻¹ s⁻²)
MIN_DT = 1e-6          # Paso temporal mínimo del integrador (s)


# ═══════════════════════════════════════════════════════════════════════════
# 2. ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CapaAtmosferica:
    """Una capa exponencial: ρ(h) = ρ_base · exp(-(h - h_min)/H)."""
    h_min_km: float           # límite inferior de la capa (km)
    rho_base_kg_m3: float     # densidad de referencia en h_min (kg/m³)
    H_km: float               # altura de escala (km)
    nombre: str


@dataclass(frozen=True)
class BandaVisual:
    """Una banda coloreada de la gráfica (puramente estética)."""
    h_sup_km: float
    h_inf_km: float
    color_hex: str
    nombre: str


@dataclass
class Planeta:
    """
    Ficha técnica de un cuerpo del sistema solar:
      - Constantes físicas extraídas de poliastro.bodies.
      - Modelo atmosférico por capas (piecewise-exponencial).
      - Configuración del integrador adaptativo.
      - Paleta de colores para la visualización.
    """
    nombre: str
    body: object                                # poliastro Body
    capas: List[CapaAtmosferica]                # ordenadas: alta → baja altitud
    h_reentrada_m: float                        # altitud crítica destructiva (m)
    referencia_h0: str                          # "superficie" o "1 bar"
    fuente: str                                 # citación corta para la memoria

    # Configuración del integrador (paso adaptativo según altitud)
    dt_schedule: List[Tuple[float, float]]      # [(umbral_m, dt_s), ...]
    dt_base: float                              # dt por defecto en zona alta (s)

    # Visualización
    bandas_visuales: List[BandaVisual]
    color_satelite: str
    titulo_extra: str = ""                      # subtítulo (ej. "Cassini Grand Finale")

    # Solo Tierra usa factor de actividad solar (F10.7)
    factor_solar_disponible: bool = False

    # J2 manual: poliastro NO declara J2 para Júpiter/Saturno/Urano/Neptuno,
    # así que lo pasamos a mano cuando hace falta. Si es None, se intenta
    # leer de body.J2 (Tierra, Marte, Venus).
    j2_manual: Optional[float] = None

    # Valores numéricos cacheados (se inicializan en __post_init__)
    M_kg: float = field(init=False)
    R_m: float = field(init=False)
    mu_m3_s2: float = field(init=False)
    J2: float = field(init=False)
    omega_rad_s: float = field(init=False)

    def __post_init__(self):
        # μ = GM directamente de poliastro (siempre disponible)
        self.mu_m3_s2 = float(self.body.k.to_value(u.m**3 / u.s**2))
        self.M_kg     = self.mu_m3_s2 / G
        self.R_m      = float(self.body.R.to_value(u.m))

        # J2: prioridad al valor manual; si no, intenta leer de poliastro
        if self.j2_manual is not None:
            self.J2 = self.j2_manual
        else:
            try:
                j2 = self.body.J2
                self.J2 = float(j2.value) if hasattr(j2, "value") else float(j2)
            except (AttributeError, ValueError):
                self.J2 = 0.0

        # Velocidad angular de rotación: no siempre está en poliastro,
        # la guardamos en el campo `omega_rad_s` (lo añade el constructor abajo
        # vía object.__setattr__ post-init si hace falta).
        # Aquí dejamos 0 si no se puede; los valores se sobreescriben fuera.
        try:
            ang = self.body.angular_velocity
            self.omega_rad_s = float(ang.to_value(u.rad / u.s))
        except (AttributeError, ValueError):
            self.omega_rad_s = 0.0

    # ─────────────────────── modelo atmosférico ───────────────────────
    @property
    def tiene_atmosfera(self) -> bool:
        """True si el cuerpo tiene modelo atmosférico (capas no vacías).
        Luna y Mercurio devuelven False — sólo aplica gravedad y J2."""
        return bool(self.capas)

    def get_rho(self, h_m: float, idx_capa: int = 0) -> Tuple[float, int]:
        """
        Devuelve (densidad kg/m³, índice de capa actual).
        Patrón persistente: se pasa idx_capa del paso anterior para coste O(1)
        amortizado (el satélite solo baja, nunca sube).

        Si el cuerpo NO tiene atmósfera (Luna, Mercurio), devuelve (0, 0).
        """
        if not self.capas:
            return 0.0, 0
        h_m = max(h_m, 0.0)
        n = len(self.capas)
        while idx_capa < n - 1:
            if h_m < self.capas[idx_capa].h_min_km * 1000.0:
                idx_capa += 1
            else:
                break
        capa = self.capas[idx_capa]
        h_km = h_m / 1000.0
        rho  = capa.rho_base_kg_m3 * np.exp(-(h_km - capa.h_min_km) / capa.H_km)
        return rho, idx_capa

    # ─────────────────────── integrador adaptativo ─────────────────────
    def dt_adaptativo(self, h_m: float) -> float:
        """Devuelve el paso temporal recomendado según la altitud (s)."""
        if not self.dt_schedule:
            return self.dt_base
        for umbral_m, dt_s in self.dt_schedule:
            if h_m < umbral_m:
                return dt_s
        return self.dt_base


# ═══════════════════════════════════════════════════════════════════════════
# 3. INSTANCIAS DE LOS 7 PLANETAS
# ═══════════════════════════════════════════════════════════════════════════
#
# Las densidades de referencia ρ_base provienen de:
#   - Tierra:    librería `ussa1976` (USSA-76)
#   - Marte:     Mars Climate Database (MCD, LMD/ESA) — sitio Opportunity
#   - Venus:     perfil T(h) de NoSoloSputnik + cálculo ρ con ecuación de gases
#   - Júpiter:   Galileo Probe (Seiff 1998) + Juno (Iess 2018)
#   - Saturno:   Cassini Grand Finale (Brown 2020, Yelle 2018)
#   - Urano:     Voyager 2 (Lindal 1987, Herbert 1987)
#   - Neptuno:   Voyager 2 (Lindal 1990, Broadfoot 1989)
#
# Las alturas de escala H están derivadas matemáticamente para garantizar
# CONTINUIDAD de ρ entre capas:
#   H_i = (h_min_{i-1} - h_min_i) / ln(ρ_base_i / ρ_base_{i-1})
#
# Detalles completos en: investigacion_atmosferas/{planeta}.md
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────── TIERRA ──────────────────────────────────
TIERRA = Planeta(
    nombre="Tierra",
    body=Earth,
    capas=[
        # H recalculadas para garantizar continuidad de ρ entre capas.
        CapaAtmosferica(500,   5.563e-13, 99.0450, "Exosfera alta"),       # H estimada (capa más alta)
        CapaAtmosferica(400,   2.984e-12, 59.5340, "Exosfera baja"),
        CapaAtmosferica(300,   2.019e-11, 52.3034, "Termosfera alta"),
        CapaAtmosferica(200,   2.617e-10, 39.0318, "Termosfera media"),
        CapaAtmosferica(150,   2.109e-9,  23.9605, "Termosfera baja"),
        CapaAtmosferica(100,   5.612e-7,   8.9544, "Mesosfera alta"),
        CapaAtmosferica( 80,   1.846e-5,   5.7253, "Mesosfera baja"),
        CapaAtmosferica( 50,   1.027e-3,   7.4649, "Estratosfera"),
        CapaAtmosferica( 12,   3.119e-1,   6.6480, "Tropopausa"),
        CapaAtmosferica(  0,   1.225,      8.7718, "Troposfera"),
    ],
    h_reentrada_m=100_000,
    referencia_h0="superficie",
    fuente="USSA-76 (librería ussa1976)",
    dt_schedule=[
        (400_000, 30),
        (250_000, 10),
        (150_000,  5),
        (100_000,  1),
    ],
    dt_base=100,
    bandas_visuales=[
        BandaVisual(500, 300, "#0a0a1a", "Exosfera"),
        BandaVisual(300, 200, "#0d1b3e", "Termosfera alta"),
        BandaVisual(200, 150, "#102050", "Termosfera media"),
        BandaVisual(150,  80, "#1a3a6e", "Termosfera baja"),
        BandaVisual( 80,  50, "#1e5080", "Mesosfera"),
        BandaVisual( 50,  12, "#1a6644", "Estratosfera"),
        BandaVisual( 12,   0, "#2d6e2d", "Troposfera"),
    ],
    color_satelite="steelblue",
    titulo_extra="Modelo USSA-76 con actividad solar variable",
    factor_solar_disponible=True,
)

# ────────────────────────────── MARTE ──────────────────────────────────
MARTE = Planeta(
    nombre="Marte",
    body=Mars,
    capas=[
        # Valores corregidos según el .md de Marte (mayo 2026):
        # - ρ_base de Mesosfera (h=40): 3.9e-5 → 3.9e-4 (factor 10×, valor correcto
        #   del MCD; el inicial era un error tipográfico).
        # - H de Mesosfera y Troposfera ajustadas a los valores del MCD original
        #   (8.37 y 10.96 km), que con la nueva ρ_base SÍ mantienen continuidad.
        # Continuidad final entre capas: error < 0.4% en todos los límites.
        CapaAtmosferica(300,   1.5e-14,   63.4300, "Exosfera"),                      # H estimada (capa más alta)
        CapaAtmosferica(250,   1.05e-13,  25.6949, "Exosfera / Termosfera alta"),
        CapaAtmosferica(200,   1.50e-12,  18.8022, "Termosfera alta"),
        CapaAtmosferica(150,   2.10e-10,  10.1181, "Termosfera media"),
        CapaAtmosferica(100,   3.00e-07,   6.8829, "Mesopausa / Tránsito orbital"),
        CapaAtmosferica( 40,   3.90e-04,   8.3700, "Mesosfera"),
        CapaAtmosferica(  0,   1.50e-02,  10.9600, "Troposfera / Superficie"),
    ],
    h_reentrada_m=50_000,
    referencia_h0="superficie",
    fuente="Mars Climate Database (MCD, LMD/ESA) — sitio Opportunity, Ls=339°",
    dt_schedule=[
        (150_000, 30),
        (100_000, 10),
        ( 60_000,  5),
        ( 50_000,  1),
    ],
    dt_base=60,
    bandas_visuales=[
        BandaVisual(300, 250, "#0a0a1a", "Exosfera"),
        BandaVisual(250, 200, "#0d1b3e", "Termosfera alta"),
        BandaVisual(200, 150, "#102050", "Termosfera media"),
        BandaVisual(150, 100, "#1a3a6e", "Mesopausa / Tránsito"),
        BandaVisual(100,  40, "#1e5080", "Mesosfera"),
        BandaVisual( 40,   0, "#c1440e", "Troposfera"),
    ],
    color_satelite="#00e5ff",
    titulo_extra="Modelo MCD por capas",
)

# ────────────────────────────── VENUS ──────────────────────────────────
VENUS = Planeta(
    nombre="Venus",
    body=Venus,
    capas=[
        # H recalculadas para continuidad <0.01%.
        CapaAtmosferica(200,   1.76e-10, 36.0000, "Exosfera / Termosfera"),  # H estimada
        CapaAtmosferica(100,   4.71e-5,   8.0017, "Termosfera media"),
        CapaAtmosferica( 50,   1.039,     4.9993, "Mesosfera / Nubes"),
        CapaAtmosferica(  0,  67.0,      12.0007, "Troposfera densa"),
    ],
    h_reentrada_m=40_000,
    referencia_h0="superficie",
    fuente="Perfil T(h) de NoSoloSputnik + cálculo propio (μ, g por tramo)",
    dt_schedule=[
        (200_000, 1.0),
        (120_000, 0.1),
    ],
    dt_base=30.0,
    bandas_visuales=[
        BandaVisual(200, 100, "#0a0a1a", "Exosfera / Termosfera"),
        BandaVisual(100,  50, "#0d1b3e", "Termosfera media"),
        BandaVisual( 50,  40, "#1a3a6e", "Mesosfera / Nubes"),
        BandaVisual( 40,   0, "#c17a0e", "Troposfera densa"),
    ],
    color_satelite="darkorange",
    titulo_extra="Modelo VIRA por capas",
)

# ───────────────────────────── JÚPITER ─────────────────────────────────
JUPITER = Planeta(
    nombre="Júpiter",
    body=Jupiter,
    capas=[
        # Recalibrado contra el dataset oficial del Galileo Probe ASI
        # (NASA-PDS, 693 puntos in-situ, Seiff 1998 DOI:10.17189/tfsa-pb91).
        # ρ_base a 500 y 1000 km bajaron por factor ~10 respecto a la versión
        # anterior, que sobreestimaba la termosfera. H recalculadas para
        # continuidad <0.01%.
        CapaAtmosferica(1000,  3.0e-11, 150.0000, "Exosfera"),                # H estimada
        CapaAtmosferica( 500,  2.0e-9,  119.0560, "Termosfera alta"),
        CapaAtmosferica( 320,  1.4e-7,   42.3667, "Termosfera baja / mesopausa"),
        CapaAtmosferica(  50,  2.5e-2,   22.3274, "Estratosfera"),
        CapaAtmosferica(   0,  0.16,     26.9353, "Troposfera / nubes"),
    ],
    h_reentrada_m=100_000,
    referencia_h0="1 bar",
    fuente="Galileo Probe (Seiff 1998) + Juno (Iess 2018) + Yelle (2004)",
    j2_manual=1.4736e-2,   # Iess 2018 (no está en poliastro)
    dt_schedule=[
        (500_000, 30),
        (200_000, 10),
        (100_000,  1),
    ],
    dt_base=60,
    bandas_visuales=[
        BandaVisual(1000, 500, "#1a0d2e", "Exosfera"),
        BandaVisual( 500, 320, "#3d1a47", "Termosfera alta"),
        BandaVisual( 320,  50, "#6b2f1a", "Termosfera baja"),
        BandaVisual(  50,   0, "#a8541f", "Estratosfera / nubes"),
    ],
    color_satelite="#ffcc66",
    titulo_extra="Datos: Galileo Probe + Voyager + Juno",
)

# ───────────────────────────── SATURNO ─────────────────────────────────
SATURNO = Planeta(
    nombre="Saturno",
    body=Saturn,
    capas=[
        # Recalibrado contra el dataset oficial Cassini Grand Finale 2017
        # (NASA-PDS, 763 puntos in-situ UVIS, DOI:10.17189/518e-p721).
        # ρ_base ajustados a la mediana de Cassini sobre todas las latitudes
        # en cada h_min cubierto (700, 950, 1450, 1800, 2200 km).
        # La capa intermedia de 1100 km se eliminó por inconsistencia.
        # Se añadió capa en 700 km (zona termopausa/estratopausa) para
        # mejor ajuste en el rango 600-900 km.
        # Las capas bajas (0/80/300 km) se mantienen de Voyager (Tyler 1982).
        # H recalculadas para continuidad <0.01%.
        CapaAtmosferica(2200, 1.185e-12, 250.0000, "Exosfera / Termosfera superior"),  # H estimada
        CapaAtmosferica(1800, 4.617e-12, 294.0000, "Termosfera media"),
        CapaAtmosferica(1450, 4.326e-11, 156.5026, "Termosfera baja"),
        CapaAtmosferica( 950, 1.556e-9,  139.5460, "Mesosfera / Termosfera inferior"),
        CapaAtmosferica( 700, 1.064e-8,  130.0080, "Termopausa"),
        CapaAtmosferica( 300, 1.4e-4,     42.1781, "Estratosfera"),
        CapaAtmosferica(  80, 1.9e-2,     44.8015, "Tropopausa"),
        CapaAtmosferica(   0, 0.19,       34.7436, "Troposfera / nubes"),
    ],
    h_reentrada_m=500_000,
    referencia_h0="1 bar",
    fuente="Cassini Grand Finale (Brown 2020, Yelle 2018) + Voyager (Tyler 1982)",
    j2_manual=1.6298e-2,   # Iess 2019 — el J2 más alto del sistema solar
    dt_schedule=[
        (2_500_000, 60),
        (1_500_000, 20),
        (1_000_000,  5),
        (  500_000,  1),
    ],
    dt_base=120,
    bandas_visuales=[
        BandaVisual(2200, 1800, "#0a0a1a", "Exosfera / Termos. superior"),
        BandaVisual(1800, 1450, "#1a1a2e", "Termosfera media"),
        BandaVisual(1450,  950, "#2d2a3e", "Termosfera baja"),
        BandaVisual( 950,  700, "#4a3c4a", "Mesos. / Termos. inferior"),
        BandaVisual( 700,  300, "#7a5a3a", "Termopausa"),
        BandaVisual( 300,   80, "#a87c4a", "Estratosfera"),
        BandaVisual(  80,    0, "#c9a570", "Troposfera / nubes"),
    ],
    color_satelite="#ffd97d",
    titulo_extra="Datos: Cassini Grand Finale + Voyager",
)

# ───────────────────────────── URANO ───────────────────────────────────
URANO = Planeta(
    nombre="Urano",
    body=Uranus,
    capas=[
        # Validado contra Voyager 2 reconstruido (Lindal 1987) por integración
        # hidrostática — ver datos_validacion/voyager_urano/. Se añadió la capa
        # intermedia de 150 km (ρ≈1.1e-3) porque el tramo 50→320 km original
        # usaba H=38 km (demasiado grande para una estratosfera fría de 50-120 K),
        # sobreestimando ρ hasta ~4×. Tras recalibrar: factor mediano 1.15×
        # (antes 2.49×) en el rango 0-262 km cubierto por Voyager.
        # H recalculadas para continuidad <0.01%.
        CapaAtmosferica(6500, 1.0e-13, 1500.0000, "Exosfera"),                 # H estimada
        CapaAtmosferica(4000, 1.0e-10,  361.9121, "Termosfera superior (energy crisis ~800 K)"),
        CapaAtmosferica(1000, 5.0e-7,   352.2287, "Termosfera inferior / mesopausa"),
        CapaAtmosferica( 320, 5.0e-5,   147.6601, "Estratosfera superior"),
        CapaAtmosferica( 150, 1.1e-3,    54.9976, "Estratosfera media (recalibrada Voyager)"),
        CapaAtmosferica(  50, 6.0e-2,    25.0060, "Estratosfera inferior (recalibrada Voyager)"),
        CapaAtmosferica(   0, 0.42,      25.6949, "Troposfera"),
    ],
    h_reentrada_m=500_000,
    referencia_h0="1 bar",
    fuente="Voyager 2 (Lindal 1987, Herbert 1987) — validado vs reconstrucción hidrostática",
    j2_manual=3.3434e-3,   # Jacobson 2014
    dt_schedule=[
        (4_000_000, 60),
        (2_000_000, 20),
        (1_000_000,  5),
        (  500_000,  1),
    ],
    dt_base=180,
    bandas_visuales=[
        BandaVisual(6500, 4000, "#0a1428", "Exosfera"),
        BandaVisual(4000, 1000, "#0d2b3e", "Termos. superior (energy crisis)"),
        BandaVisual(1000,  320, "#0e3d52", "Termos. inferior / mesopausa"),
        BandaVisual( 320,   50, "#155b6e", "Estratosfera superior"),
        BandaVisual(  50,    0, "#2d8095", "Estratosfera inf. / Troposfera"),
    ],
    color_satelite="#7fdbe0",
    titulo_extra="Datos: Voyager 2 (1986)",
)

# ───────────────────────────── NEPTUNO ─────────────────────────────────
NEPTUNO = Planeta(
    nombre="Neptuno",
    body=Neptune,
    capas=[
        # Recalibrado (2026) contra Voyager 2 reconstruido (Lindal 1990/1992) y
        # validado con los puntos del Neptune-GRAM — ver datos_validacion/voyager_neptuno/.
        # El modelo anterior subestimaba ρ por factores de 5-50× entre 20-300 km:
        # la H de troposfera era 6.84 km (física ~18) y ρ_base a 50 km era 3e-4,
        # ~100× demasiado baja (lo correcto ≈ 0.03). Se corrigió ρ(50 km) y se
        # añadieron capas a 150 y 300 km. Factor típico vs Voyager: 8.0× → 1.2×.
        # H recalculadas para continuidad <0.01%.
        CapaAtmosferica(4000, 1.0e-14, 950.0000, "Exosfera"),                  # H estimada
        CapaAtmosferica(1500, 5.0e-11, 293.5239, "Termosfera alta (~750 K)"),
        CapaAtmosferica( 600, 3.0e-8,  140.6925, "Termosfera / Mesosfera"),
        CapaAtmosferica( 300, 6.0e-6,   56.6217, "Mesosfera (recalibrada Voyager)"),
        CapaAtmosferica( 150, 2.5e-4,   40.2177, "Estratosfera alta (recalibrada Voyager)"),
        CapaAtmosferica(  50, 3.0e-2,   20.8878, "Estratosfera (recalibrada Voyager)"),
        CapaAtmosferica(   0, 0.45,     18.4635, "Troposfera (recalibrada Voyager)"),
    ],
    h_reentrada_m=500_000,
    referencia_h0="1 bar",
    fuente="Voyager 2 (Lindal 1990, Broadfoot 1989) — validado vs reconstrucción hidrostática + Neptune-GRAM",
    j2_manual=3.411e-3,    # NSSDCA
    dt_schedule=[
        (4_000_000, 60),
        (2_000_000, 20),
        (1_000_000,  5),
        (  500_000,  1),
    ],
    dt_base=180,
    bandas_visuales=[
        BandaVisual(4000, 1500, "#050a20", "Exosfera"),
        BandaVisual(1500,  600, "#0a1545", "Termosfera alta (~750 K)"),
        BandaVisual( 600,   50, "#0f2370", "Termosfera / Mesosfera"),
        BandaVisual(  50,    0, "#1a3a9e", "Estratosfera / Troposfera"),
    ],
    color_satelite="#7fb3ff",
    titulo_extra="Datos: Voyager 2 (1989)",
)


# ────────────────────────── LUNA (sin atmósfera) ──────────────────────
# La Luna tiene una exosfera ultrarrenue (10⁻¹⁵ kg/m³) que se desprecia
# para órbitas. Sólo aplican gravedad central y J2.
LUNA = Planeta(
    nombre="Luna",
    body=Moon,
    capas=[],                              # sin atmósfera
    h_reentrada_m=0,                       # impacto con superficie
    referencia_h0="superficie",
    fuente="Sin atmósfera (exosfera < 10⁻¹⁵ kg/m³); J2 de LRO (Konopliv 2013)",
    dt_schedule=[],
    dt_base=60.0,                          # dt cómodo, no hay capas densas
    bandas_visuales=[],
    color_satelite="white",
    titulo_extra="Cuerpo sin atmósfera — sólo gravedad + J2",
    j2_manual=2.0327e-4,                   # LRO 2013/2014 (no está en poliastro)
)

# ─────────────────────── MERCURIO (sin atmósfera) ─────────────────────
# Mercurio tiene una exosfera de Na/H/He completamente despreciable.
# Sólo aplican gravedad central y J2.
MERCURIO = Planeta(
    nombre="Mercurio",
    body=Mercury,
    capas=[],                              # sin atmósfera
    h_reentrada_m=0,                       # impacto con superficie
    referencia_h0="superficie",
    fuente="Sin atmósfera (exosfera Na/H/He); J2 de MESSENGER (Mazarico 2014)",
    dt_schedule=[],
    dt_base=60.0,
    bandas_visuales=[],
    color_satelite="white",
    titulo_extra="Cuerpo sin atmósfera — sólo gravedad + J2",
    j2_manual=5.03e-5,                     # MESSENGER 2014 (no está en poliastro)
)


# ═══════════════════════════════════════════════════════════════════════════
# 4. DICCIONARIO DE ACCESO POR NOMBRE
# ═══════════════════════════════════════════════════════════════════════════

PLANETAS = {
    "tierra":   TIERRA,
    "marte":    MARTE,
    "venus":    VENUS,
    "jupiter":  JUPITER,
    "saturno":  SATURNO,
    "urano":    URANO,
    "neptuno":  NEPTUNO,
    "luna":     LUNA,
    "mercurio": MERCURIO,
}


# ═══════════════════════════════════════════════════════════════════════════
# 5. VALIDACIÓN DE INPUTS
# ═══════════════════════════════════════════════════════════════════════════

def pedir_float(mensaje: str, minimo: Optional[float] = None,
                maximo: Optional[float] = None) -> float:
    """Input numérico con validación de tipo y rango opcional."""
    while True:
        try:
            valor = float(input(mensaje))
            if minimo is not None and valor < minimo:
                print(f"  ⚠️  El valor debe ser ≥ {minimo}.")
                continue
            if maximo is not None and valor > maximo:
                print(f"  ⚠️  El valor debe ser ≤ {maximo}.")
                continue
            return valor
        except ValueError:
            print("  ⚠️  Introduce un número válido.")


def pedir_int(mensaje: str, minimo: int = 1, maximo: Optional[int] = None) -> int:
    """Input entero con validación."""
    while True:
        try:
            valor = int(input(mensaje))
            if valor < minimo:
                print(f"  ⚠️  El valor debe ser ≥ {minimo}.")
                continue
            if maximo is not None and valor > maximo:
                print(f"  ⚠️  El valor debe ser ≤ {maximo}.")
                continue
            return valor
        except ValueError:
            print("  ⚠️  Introduce un número entero válido.")


# ═══════════════════════════════════════════════════════════════════════════
# 6. NÚCLEO DE SIMULACIÓN (GENÉRICO PARA CUALQUIER PLANETA)
# ═══════════════════════════════════════════════════════════════════════════

def simular(planeta: Planeta, h_inicial_m: float, masa: float, area: float,
            cd: float, dias: int, factor_solar: float = 1.0):
    """
    Integra el decaimiento orbital por arrastre atmosférico mediante el
    método de energía orbital, con paso adaptativo y tres salvaguardas:

      (a) energía: dE ≤ 0.1 % de |E_orbital|
      (b) estabilidad numérica: E_nueva debe seguir siendo negativa
      (c) cinemática: la caída de radio en un paso ≤ 10 km

    Parámetros
    ----------
    planeta       : instancia de Planeta
    h_inicial_m   : altitud inicial en METROS sobre el nivel h=0 del planeta
    masa          : masa del satélite (kg)
    area          : área frontal (m²)
    cd            : coeficiente de arrastre
    dias          : duración máxima de la simulación (días terrestres)
    factor_solar  : modificador de densidad por actividad solar (Tierra: 0.5/1.0/3.0)

    Retorna
    -------
    tiempos   : list[float]  — tiempo en días por muestra
    alturas   : list[float]  — altitud en km por muestra
    reentrada : bool         — True si el satélite alcanzó h_reentrada
    """
    G_M_m   = planeta.mu_m3_s2 * masa
    R_m     = planeta.R_m
    h_rein  = planeta.h_reentrada_m

    h_m       = float(h_inicial_m)
    t_s       = 0.0
    t_max_s   = dias * 86_400.0
    idx_capa  = 0

    alturas                 = [h_m / 1000.0]
    tiempos                 = [0.0]
    ultimo_guardado_km      = h_m / 1000.0
    ultimo_tiempo_guardado  = 0.0

    reentrada = False

    while t_s < t_max_s:
        dt = planeta.dt_adaptativo(h_m)

        r        = R_m + h_m
        v        = np.sqrt(G_M_m / (masa * r))            # = sqrt(μ/r)
        rho_b, idx_capa = planeta.get_rho(h_m, idx_capa)
        rho      = rho_b * factor_solar

        fd       = 0.5 * rho * v**2 * cd * area
        E_actual = -G_M_m / (2.0 * r)

        # ── Halvings: tres criterios en cascada ──────────────────────
        while dt > MIN_DT:
            dE = fd * v * dt
            # (a) criterio energético
            if dE > abs(E_actual) * 0.001:
                dt /= 2.0
                continue
            # (b) estabilidad numérica
            E_prov = E_actual - dE
            if E_prov >= 0:
                dt /= 2.0
                continue
            # (c) criterio cinemático
            r_prov = -G_M_m / (2.0 * E_prov)
            if (r - r_prov) > 10_000:
                dt /= 2.0
                continue
            break  # paso aceptado

        dE      = fd * v * dt
        E_nueva = E_actual - dE
        if E_nueva >= 0:
            print("  ⚠️  Inestabilidad numérica (E ≥ 0). Revisa los parámetros.")
            break

        r_nueva = -G_M_m / (2.0 * E_nueva)
        h_m     = r_nueva - R_m
        t_s    += dt

        # ── Check de reentrada ANTES del muestreo ────────────────────
        if h_m <= h_rein:
            alturas.append(h_rein / 1000.0)
            tiempos.append(t_s / 86_400.0)
            print(f"\n  🔥 ¡REENTRADA! Satélite destruido a {h_rein//1000:.0f} km "
                  f"({'sobre 1 bar' if planeta.referencia_h0 == '1 bar' else 'de altitud'}) "
                  f"al día {t_s/86400:.2f}.")
            reentrada = True
            break

        # ── Muestreo dual: cada 1 km de caída o cada hora ────────────
        caida_km              = ultimo_guardado_km - (h_m / 1000.0)
        tiempo_desde_guardado = t_s - ultimo_tiempo_guardado

        if caida_km >= 1.0 or tiempo_desde_guardado >= 3600:
            alturas.append(h_m / 1000.0)
            tiempos.append(t_s / 86_400.0)
            ultimo_guardado_km     = h_m / 1000.0
            ultimo_tiempo_guardado = t_s

    return tiempos, alturas, reentrada


# ═══════════════════════════════════════════════════════════════════════════
# 7. GRÁFICA (GENÉRICA)
# ═══════════════════════════════════════════════════════════════════════════

def graficar(planeta: Planeta, tiempos, alturas, h_ini_km, masa, area, cd,
             reentrada: bool, factor_solar_str: str = ""):
    """Genera la figura de decaimiento con paleta propia del planeta."""

    h_rein_km  = planeta.h_reentrada_m / 1000.0
    y_min_plot = max(h_rein_km - max(50, h_rein_km * 0.05), 0)
    y_max_plot = h_ini_km + max(50, h_ini_km * 0.05)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    # ── Bandas atmosféricas coloreadas ───────────────────────────────
    for banda in planeta.bandas_visuales:
        if banda.h_sup_km < y_min_plot or banda.h_inf_km > y_max_plot:
            continue
        h_top = min(banda.h_sup_km, y_max_plot)
        h_bot = max(banda.h_inf_km, y_min_plot)
        ax.axhspan(h_bot, h_top, color=banda.color_hex, alpha=0.45, zorder=0)
        mid = (h_top + h_bot) / 2.0
        if y_min_plot < mid < y_max_plot:
            ax.text(tiempos[-1] * 0.98, mid, banda.nombre,
                    color="white", fontsize=7.5, alpha=0.6,
                    ha="right", va="center", style="italic")

    # ── Líneas divisorias entre capas físicas (no visuales) ──────────
    for capa in planeta.capas[1:]:
        if y_min_plot < capa.h_min_km < y_max_plot:
            ax.axhline(capa.h_min_km, color="white",
                       linewidth=0.4, linestyle=":", alpha=0.3)

    # ── Línea de reentrada ───────────────────────────────────────────
    ax.axhline(h_rein_km, color="#ff4444", linewidth=1.5, linestyle="--",
               label=f"Límite de reentrada ({h_rein_km:.0f} km)", zorder=3)

    # ── Trayectoria del satélite ─────────────────────────────────────
    ax.plot(tiempos, alturas, color=planeta.color_satelite, linewidth=2.2,
            label="Altitud del satélite", zorder=4)

    if reentrada:
        ax.scatter(tiempos[-1], alturas[-1], color="#ff4444", s=60, zorder=5)
        ax.annotate(f"  Reentrada\n  día {tiempos[-1]:.2f}",
                    xy=(tiempos[-1], alturas[-1]),
                    color="#ff8888", fontsize=9)

    # ── Layout y etiquetas ───────────────────────────────────────────
    sub = f" · {factor_solar_str}" if factor_solar_str else ""
    titulo = (
        f"Decaimiento Orbital en {planeta.nombre} — {planeta.titulo_extra}\n"
        f"Satélite: {masa} kg · {area} m² · Cd = {cd}{sub}"
    )
    ax.set_title(titulo, color="white", fontsize=13, pad=14)
    ax.set_xlabel("Tiempo (días)", color="white", fontsize=11)
    ylabel = ("Altitud sobre nivel 1 bar (km)"
              if planeta.referencia_h0 == "1 bar"
              else "Altitud sobre la superficie (km)")
    ax.set_ylabel(ylabel, color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_xlim(0, tiempos[-1] * 1.02)
    ax.set_ylim(y_min_plot, y_max_plot)
    ax.grid(True, color="#333", linewidth=0.5, alpha=0.6)
    ax.legend(facecolor="#111", edgecolor="#555", labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 8. MENÚ INTERACTIVO Y FLUJO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

# Factores de actividad solar (solo para Tierra, USSA-76 con índice F10.7)
FACTORES_SOLAR = {"baja": 0.5, "media": 1.0, "alta": 3.0}


def elegir_planeta(solo_con_atmosfera: bool = False) -> Planeta:
    """
    Muestra el menú y devuelve el Planeta seleccionado por el usuario.

    Si `solo_con_atmosfera=True`, filtra Luna y Mercurio (no aplica para
    simulación de drag).
    """
    print("=" * 70)
    print("  SIMULADOR DE DECAIMIENTO ORBITAL — MÓDULO UNIFICADO")
    print("=" * 70)
    print("  Elige el cuerpo a simular:")
    print()
    nombres = [n for n, p in PLANETAS.items()
               if (not solo_con_atmosfera) or p.tiene_atmosfera]
    for i, n in enumerate(nombres, start=1):
        p = PLANETAS[n]
        ref = "(1 bar)" if p.referencia_h0 == "1 bar" else "(superficie)"
        tag = "" if p.tiene_atmosfera else "  ⚠️ sin atmósfera"
        print(f"   {i}. {p.nombre:<10s} {ref:<12s}  R = {p.R_m/1000:>8.0f} km{tag}")
    print()
    idx = pedir_int(f"Tu elección (1-{len(nombres)}): ",
                    minimo=1, maximo=len(nombres))
    return PLANETAS[nombres[idx - 1]]


def main():
    # Este simulador es DE DRAG; filtramos cuerpos sin atmósfera (Luna, Mercurio)
    planeta = elegir_planeta(solo_con_atmosfera=True)

    print()
    print("=" * 70)
    print(f"  PLANETA SELECCIONADO: {planeta.nombre.upper()}")
    print(f"  Modelo: {planeta.titulo_extra}")
    print(f"  Fuente: {planeta.fuente}")
    print(f"  Nivel h=0: {planeta.referencia_h0}")
    print("=" * 70)

    # Aviso especial sobre el nivel de referencia para gaseosos
    if planeta.referencia_h0 == "1 bar":
        print("  Nota: la altitud se mide sobre el nivel donde P = 1 bar.")
        print("        Este planeta no tiene superficie sólida.")
        print("=" * 70)

    # Factor de actividad solar (solo Tierra)
    factor_solar     = 1.0
    factor_solar_str = ""
    if planeta.factor_solar_disponible:
        while True:
            actividad = input("Actividad solar (baja / media / alta): ").strip().lower()
            if actividad in FACTORES_SOLAR:
                factor_solar     = FACTORES_SOLAR[actividad]
                factor_solar_str = f"Actividad solar: {actividad}"
                break
            print("  ⚠️  Elige entre: baja, media o alta.")

    # Parámetros comunes
    h_ini_min = planeta.h_reentrada_m / 1000.0
    h_ini_km  = pedir_float(
        f"Altura inicial (km sobre nivel h=0)  [Ej: {int(h_ini_min*5)}] : ",
        minimo=h_ini_min, maximo=50_000)
    masa = pedir_float("Masa del satélite (kg)              [Ej: 100] : ",
                       minimo=1e-3)
    area = pedir_float("Área frontal (m²)                   [Ej: 5]   : ",
                       minimo=1e-3)
    cd   = pedir_float("Coeficiente de arrastre Cd          [Ej: 2.2] : ",
                       minimo=0.1, maximo=10.0)
    dias = pedir_int  ("Días de simulación                  [Ej: 365] : ",
                       minimo=1)

    # Aviso para órbitas muy bajas
    if h_ini_km < h_ini_min * 2:
        print(f"\n  ⚠️  Aviso: a {h_ini_km:.0f} km estás cerca del límite de reentrada "
              f"({h_ini_min:.0f} km). La caída puede ser muy rápida.")

    # Cabecera del integrador
    print(f"\nSimulando con dt adaptativo:")
    print(f"  > {planeta.dt_schedule[0][0]/1000:.0f} km  → {planeta.dt_base} s/paso")
    for umbral, dt_v in planeta.dt_schedule:
        print(f"  < {umbral/1000:>5.0f} km  → {dt_v} s/paso")
    print()

    # Simulación + gráfica
    tiempos, alturas, reentrada = simular(
        planeta      = planeta,
        h_inicial_m  = h_ini_km * 1000.0,
        masa         = masa,
        area         = area,
        cd           = cd,
        dias         = dias,
        factor_solar = factor_solar,
    )

    # Resumen numérico
    print(f"\n{'─' * 70}")
    print(f"  Altura inicial         : {alturas[0]:.2f} km")
    print(f"  Días de supervivencia  : {tiempos[-1]:.4f} días")
    if reentrada:
        print(f"  Estado                 : REENTRADA "
              f"({planeta.h_reentrada_m/1000:.0f} km {planeta.referencia_h0})")
        if len(alturas) >= 2:
            print(f"  Último punto registrado: {alturas[-2]:.2f} km")
    else:
        perdida = alturas[0] - alturas[-1]
        print(f"  Estado                 : En órbita al finalizar la simulación")
        print(f"  Altitud final          : {alturas[-1]:.4f} km")
        print(f"  Pérdida de altitud     : {perdida:.4f} km")
    print(f"{'─' * 70}\n")

    graficar(planeta, tiempos, alturas, h_ini_km, masa, area, cd,
             reentrada, factor_solar_str)


if __name__ == "__main__":
    main()
