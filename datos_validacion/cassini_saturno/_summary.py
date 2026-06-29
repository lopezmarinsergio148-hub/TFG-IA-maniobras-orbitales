import csv, os
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE,'cassini_saturno_combined.csv'),encoding='utf-8')))
g = defaultdict(list)
for r in rows: g[r['occultation']].append(r)
print('{:<6} {:>7} {:>5} {:>7} {:>7}'.format('ST','lat','N','h_min','h_max'))
for st, rs in sorted(g.items(), key=lambda kv: int(kv[0][2:])):
    alts=[float(r['altitude_km']) for r in rs]
    lat=float(rs[0]['lat_deg'])
    print('{:<6} {:>7.2f} {:>5d} {:>7.1f} {:>7.1f}'.format(st,lat,len(rs),min(alts),max(alts)))
print('\nMuestra ST87 (lat ~ 0, ecuador) cada ~200 km:')
st87=[r for r in rows if r['occultation']=='ST87']
print('{:>8} {:>13} {:>13} {:>7} {:>7} {:>5}'.format('h_km','rho_kg_m3','nH2_m-3','T_K','lat','type'))
seen=set()
for r in sorted(st87, key=lambda d:float(d['altitude_km'])):
    b=int(float(r['altitude_km'])//200)
    if b in seen: continue
    seen.add(b)
    print('{:>8.1f} {:>13.4e} {:>13.4e} {:>7.1f} {:>7.2f} {:>5}'.format(
        float(r['altitude_km']),float(r['density_kg_m3']),float(r['nH2_m-3']),
        float(r['temperature_K']),float(r['lat_deg']),r['data_type']))
