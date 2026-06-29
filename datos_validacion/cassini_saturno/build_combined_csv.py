"""
Construye cassini_saturno_combined.csv a partir de los .csv PDS (Koskinen et al.)
- Carpeta: ../cassini_saturno/*.csv (formato Koskinen, sin cabecera de columnas en csv)
- Columnas del .csv (orden, sin header): Alt_km, Rad_km, Lat_deg, Lon_deg, Nden_m-3,
  dNden_m-3, Temp_K, dTemp_K, FMNden_m-3, FMTemp_K
- Conversion: rho_kg_m3 = N_H2 * m_H2  donde m_H2 = 2.0159 * 1.66054e-27 kg
  (Saturno = 96% H2 + 3.25% He -> mu_mol = 2.135 g/mol; aqui solo medimos n(H2),
   por eso usamos m_H2 puro. Para densidad de masa total se podria reescalar +5%.)
- Para cada altitud:
    * Si Nden > 0 -> Direct retrieval (preferido, source=INV)
    * Si Nden == 0 pero FMNden > 0 -> Forward model (source=FM)
"""
import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
M_H2_KG = 2.01588 * 1.66053906660e-27   # masa H2 (kg)
# Si quisieramos densidad de mezcla total H2+He:
# mu = 0.96*2.01588 + 0.0325*4.002602 = 2.0653 g/mol -> factor sobre m_H2:
# factor_mix = (0.96*2.01588 + 0.0325*4.002602) / (0.96*2.01588) approx 1.0335
# (Asumiendo n_He/n_H2 ~ 0.0325/0.96)

FACTOR_MIX_TOTAL = (0.96 * 2.01588 + 0.0325 * 4.002602) / 2.01588  # ~1.066

rows_out = []

for path in sorted(glob.glob(os.path.join(HERE, "EUV*_results.csv"))):
    fname = os.path.basename(path)
    # parse ST id
    st = fname.split("_ST")[-1].split("_")[0]
    source_file = fname.replace("_results.csv", "")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 10:
                continue
            try:
                alt = float(r[0]); rad = float(r[1]); lat = float(r[2]); lon = float(r[3])
                nden = float(r[4]); dnd = float(r[5])
                temp = float(r[6]); dtmp = float(r[7])
                fmn  = float(r[8]); fmt  = float(r[9])
            except ValueError:
                continue
            if nden > 0:
                src = "INV"; n_use = nden; t_use = temp; dn = dnd; dt = dtmp
            elif fmn > 0:
                src = "FM";  n_use = fmn;  t_use = fmt;  dn = 0.0;  dt = 0.0
            else:
                continue
            rho_h2_only = n_use * M_H2_KG
            rho_mix     = rho_h2_only * FACTOR_MIX_TOTAL
            rows_out.append({
                "altitude_km": alt,
                "density_kg_m3": rho_mix,
                "density_H2only_kg_m3": rho_h2_only,
                "nH2_m-3": n_use,
                "temperature_K": t_use,
                "lat_deg": lat,
                "lon_deg": lon,
                "radius_km": rad,
                "data_type": src,
                "occultation": "ST" + st,
                "source_file": source_file,
            })

# Ordenar por altitud
rows_out.sort(key=lambda d: (d["occultation"], d["altitude_km"]))

out_path = os.path.join(HERE, "cassini_saturno_combined.csv")
fieldnames = ["altitude_km","density_kg_m3","density_H2only_kg_m3","nH2_m-3",
              "temperature_K","lat_deg","lon_deg","radius_km","data_type",
              "occultation","source_file"]
with open(out_path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in rows_out:
        w.writerow(row)

# Estadisticas rapidas
n_total = len(rows_out)
n_inv   = sum(1 for r in rows_out if r["data_type"] == "INV")
n_fm    = n_total - n_inv
alts    = [r["altitude_km"] for r in rows_out]
lats    = [r["lat_deg"] for r in rows_out]
print(f"Filas totales: {n_total}")
print(f"  Direct inversion (INV): {n_inv}")
print(f"  Forward model    (FM):  {n_fm}")
print(f"Altitud min/max: {min(alts):.1f} / {max(alts):.1f} km sobre 1 bar")
print(f"Latitud min/max: {min(lats):.2f} / {max(lats):.2f} deg")
print(f"Archivos procesados: {len(set(r['source_file'] for r in rows_out))}")
print(f"Salida: {out_path}")
