from ussa1976 import compute
import numpy as np

# Definir altura en METROS (acepta arrays o valores sueltos)
# Probemos 150 km y 500 km para verificar
alturas_m = np.array([400000, 1000000])

# Calcular (devuelve un DataFrame de xarray/pandas)
df = compute(z=alturas_m)

# Extraer densidades (rho) en kg/m³ y Temperaturas (t) en Kelvin
densidades = df['rho'].values
temperaturas = df['t'].values

print("--- US Standard Atmosphere 1976 ---")
for i, h in enumerate(alturas_m):
    print(f"Altura: {h/1000} km")
    print(f"  Densidad: {densidades[i]:.4e} kg/m³")
    print(f"  Temperatura: {temperaturas[i]:.2f} K")
