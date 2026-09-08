# =============================================================================
# PHARMACY NETWORK MODEL  ·  PART 1  ·  DATA AND CONFIG
# Pranshi Saxena  ·  Optum Rx Quantum Initiative
#
# Run:   python pnm_data.py            builds the four tables, derives model inputs,
#                                      prints the data report, writes exports/
# Quick: PNM_SMOKE=1 python pnm_data.py
#
# This file has NO model in it. It builds exactly the data the signed off formula needs:
#   members, sites, drug classes, claims, all in Optum column names,
#   then derives from claims: days by class and bucket, the member's own price per day,
#   visits by kind of place, today's site and distance by kind of place, mail status and
#   30/90 supply held at today's, and the footprint of ZIPs for territory coverage.
# Part 2 imports load_data() from here.
# =============================================================================
import os, time, sys
import numpy as np, pandas as pd
from math import radians, sin, cos, sqrt, atan2
pd.set_option('display.width', 180); pd.set_option('display.max_columns', 40)
SMOKE = os.environ.get('PNM_SMOKE') == '1'
T0 = time.time()
def stamp(msg): print(f"[{time.time()-T0:5.0f}s] {msg}", flush=True)

# =============================================================================
# CONFIG. Defaults chosen to be defensible. Change any line, nothing else needs to.
# =============================================================================
CONFIG = {
    # ---- scale ----
    'MEMBERS_PER_TIER': dict(URBAN=100, SUBURBAN=100, RURAL=100) if SMOKE else dict(URBAN=3000, SUBURBAN=3000, RURAL=3000),
    # sites generated per tier from what the tier is like: dense in the core, sparse and far apart rural
    'RETAIL_PER_1000' : dict(URBAN=60, SUBURBAN=35, RURAL=12),
    'SPEC_PER_1000'   : dict(URBAN=5,  SUBURBAN=3,  RURAL=1),
    'INF_PER_1000'    : dict(URBAN=3.5,SUBURBAN=2.5,RURAL=1),
    'N_MAIL'          : 4 if SMOKE else 8,
    'SEEDS'      : dict(members=7, sites=11, claims=13),

    # ---- geography, miles from the metro centre. Members by setting, sites denser in the core ----
    'CENTER'     : (44.9778, -93.2650),
    'TIER_MILES' : dict(URBAN=4, SUBURBAN=10, RURAL=25) if SMOKE else dict(URBAN=8, SUBURBAN=20, RURAL=50),
    'ZIP_GRID_MILES' : 2.0 if SMOKE else 3.0,        # synthetic ZIP = grid cell of this size
    'SETTLED_SHARE'  : dict(URBAN=1.0, SUBURBAN=0.75, RURAL=0.30),   # share of grid cells that are populated

    # ---- contract rates, discount off AWP, by bucket. Realistic ranges, not the old 75 to 96 ----
    'RATE_RANGE' : dict(GENERIC=(0.80, 0.90), BRAND=(0.14, 0.22), SPECIALTY=(0.16, 0.26)),
    'PHR_RATE_DEEPER': (0.02, 0.06),                 # pharmacy reimbursed this much deeper than client rate
    'FEE_RANGE'  : (0.75, 2.75),
    'SETTING'    : dict(MAIL=0.90, HOSPITAL_OPD=1.60, INFUSION_CTR=1.00, PHYSICIAN_OFC=1.15, HOME_INFUSION=0.90),
    'ADMIN_RANGE': dict(RETAIL_CHAIN=(18_000,34_000), RETAIL_INDEP=(28_000,46_000), MAIL=(55_000,70_000),
                        SPECIALTY=(40_000,60_000), INFUSION=(35_000,55_000)),
    # infusion capacity in infusions per year by setting. Home infusion is a nursing service, not a room with chairs
    'INFUSION_CAPACITY': dict(HOSPITAL_OPD=(3000,6000), INFUSION_CTR=(1500,3000), PHYSICIAN_OFC=(500,1500), HOME_INFUSION=(2_000_000,2_000_001)),
    'CURRENT_NETWORK_SHARE': 0.55,                    # candidate pool is wider than today's network
    'CLOSED_TO_NEW_SHARE'  : 0.05,

    # ---- channels, held at today's values in the model ----
    'SPEC_MEMBER_SHARE': 0.035,                      # members with specialty or infused therapy, real world 2 to 4%
    'MAIL_SHARE_TODAY' : 0.05,                       # members whose maintenance drugs go to mail today
    'NINETY_SHARE_TODAY': 0.25,                      # members on 90 day supply for maintenance today
    'SPECIALTY_DELIVERED': True,                     # specialty ships to the door: no travel, no wall

    # ---- baseline realism ----
    'BASELINE_REALISM': 'UNMANAGED',                 # members parked at dear stores, hospitals for infusion
    'HOSPITAL_SHARE_UNMANAGED': 0.55,

    # ---- future ready ----
    'HEADROOM'   : 0.80,                             # sites may be filled to this share of capacity
    'TREND'      : {'27':1.03,'58':1.05,'27B':1.04,'01':1.00,'66':1.02,'44':1.02,'66B':1.08,'21':1.08,'66C':1.10},
    'ADEQUACY_MILES' : dict(URBAN=2.0, SUBURBAN=5.0, RURAL=10.0),   # X by setting, territory rule
    'ADEQUACY_FALLBACK': dict(RURAL=20.0),                          # if a rural ZIP cannot meet 10, the rule is 20
    'ADEQUACY_COUNT' : dict(URBAN=2,   SUBURBAN=1,   RURAL=1),      # R by setting

    # ---- access wall, disruption dials, solver. Used by Part 2, listed here so the config is one place ----
    'WALL_MILES' : 10.0,
    'A_SWEEP'    : [0, 1, 2, 5, 10, 25, 50],          # movement penalty weight per visit moved
    'LAMBDA_SWEEP': [0, 0.5, 1, 2, 5, 10],            # distance penalty weight per extra mile per visit
    'A_HOLD'     : 5,  'LAMBDA_HOLD': 2,
    'QUALITY_FLOOR': 4.0,
    'AWP_STATES' : {'WI'},
}
os.makedirs('exports', exist_ok=True)

MI_LAT, MI_LON = 69.0, 49.0
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8; dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R*2*atan2(sqrt(a), sqrt(1-a))
def place(rng, tier):
    lo = {'URBAN':0.0, 'SUBURBAN':CONFIG['TIER_MILES']['URBAN'], 'RURAL':CONFIG['TIER_MILES']['SUBURBAN']}[tier]
    hi = CONFIG['TIER_MILES'][tier]
    r = sqrt(rng.uniform(lo**2, hi**2)); th = rng.uniform(0, 2*np.pi)
    return CONFIG['CENTER'][0] + r*cos(th)/MI_LAT, CONFIG['CENTER'][1] + r*sin(th)/MI_LON
def tier_of(lat, lon):
    d = haversine_miles(lat, lon, *CONFIG['CENTER'])
    return 'URBAN' if d < CONFIG['TIER_MILES']['URBAN'] else ('SUBURBAN' if d < CONFIG['TIER_MILES']['SUBURBAN'] else 'RURAL')
def state_of(lon):
    c = CONFIG['CENTER'][1]
    return 'MN' if lon < c-0.05 else ('WI' if lon < c+0.05 else 'IA')

# =============================================================================
# DRUG CLASSES. Nine, real GPI codes. Each has a generic and a brand price per day.
# =============================================================================
GPI = [ # code  name                             share form       inj ins spec benefit maint  gen$/day brand$/day gen share
 ('27' ,'Cardiovascular Agents',          0.20,'ORAL',      0,0,0,'RX',      1,   3.20,   9.5, 0.88),
 ('58' ,'Antidiabetics Oral',             0.11,'ORAL',      0,0,0,'RX',      1,   3.60,  16.0, 0.75),
 ('27B','Insulins',                       0.07,'INJ_COLD',  1,1,0,'RX',      1,   9.00,  15.0, 0.30),
 ('01' ,'Anti-infective Agents',          0.15,'ORAL',      0,0,0,'RX',      0,   4.50,  12.0, 0.90),
 ('66' ,'Anti-inflammatory Oral',         0.18,'ORAL',      0,0,0,'RX',      1,   2.80,   9.0, 0.92),
 ('44' ,'Respiratory Agents',             0.11,'INHALED',   0,0,0,'RX',      1,   6.50,  14.0, 0.45),
 ('66B','Immunologic Agents Injectable',  0.07,'PEN_COLD',  1,0,1,'RX',      0,  40.00, 120.0, 0.15),
 ('21' ,'Antineoplastic Oral',            0.05,'ORAL_SPEC', 0,0,1,'RX',      0,  55.00, 200.0, 0.20),
 ('66C','Immunologic Agents Infused',     0.06,'INFUSION',  1,0,1,'MEDICAL', 0,  90.00, 170.0, 0.25),
]
INF_TYPES = ['HOSPITAL_OPD','INFUSION_CTR','PHYSICIAN_OFC','HOME_INFUSION']

def gen_drug_dim():
    return pd.DataFrame([dict(GPI_SK=f'G{i:03d}', GPI_CLASS_CD=c, GPI_CLASS_NM=n, DRUG_FORM_CD=f, INJECTABLE_IND=inj,
                              INSULIN_PLAN_IND=ins, SPECIALTY_DRUG_PRG_IND=sp, BENEFIT_TYPE_CD=b, MAINT_IND=mt,
                              AWP_DAY_GENERIC=g, AWP_DAY_BRAND=br, GENERIC_SHARE=gs, _SEED_SHARE=sh)
                         for i,(c,n,sh,f,inj,ins,sp,b,mt,g,br,gs) in enumerate(GPI)])

# =============================================================================
# MEMBERS. Location and setting only. Everything else about a member comes from claims.
# =============================================================================
def gen_member_dim():
    rng = np.random.default_rng(CONFIG['SEEDS']['members'])
    tiers = np.array([t for t,k in CONFIG['MEMBERS_PER_TIER'].items() for _ in range(k)]); n = len(tiers)
    pts = np.array([place(rng, t) for t in tiers])
    return pd.DataFrame({'MBR_SK': [f'M{i:05d}' for i in range(n)], 'MBR_LAT': pts[:,0], 'MBR_LON': pts[:,1],
                         'MBR_STATE_CD': [state_of(x) for x in pts[:,1]], 'MBR_URBAN_RURAL_CD': tiers,
                         'PATIENT_AGE': rng.integers(18, 85, n)})

# =============================================================================
# SITES. Four types. Three contract rates each. Candidate pool wider than today's network.
# =============================================================================
def gen_pharmacy_dim():
    rng = np.random.default_rng(CONFIG['SEEDS']['sites']); rows = []; i = [0]
    def rate(bucket): lo, hi = CONFIG['RATE_RANGE'][bucket]; return round(float(rng.uniform(lo, hi)), 3)
    def add(t, cap, ql, qh, soc='NA', tier=None):
        lat, lon = place(rng, tier or rng.choice(['URBAN','SUBURBAN','RURAL']))
        chain = (rng.choice(['ChainA','ChainB','ChainC','Independent'], p=[.3,.25,.2,.25]) if t=='RETAIL'
                 else {'MAIL':'OptumHomeDelivery','SPECIALTY':'SpecialtyNet','INFUSION':'InfusionNet'}[t])
        adm_key = ('RETAIL_INDEP' if chain=='Independent' else 'RETAIL_CHAIN') if t=='RETAIL' else t
        rg, rb, rs = rate('GENERIC'), rate('BRAND'), rate('SPECIALTY')
        deeper = lambda r: round(float(min(0.985, r + rng.uniform(*CONFIG['PHR_RATE_DEEPER']))), 3)
        rows.append(dict(
            PHR_PMT_CNTR_SK=f'P{i[0]:05d}', PHR_NPI=f'{1500000000+i[0]}', PHR_LAT=lat, PHR_LON=lon,
            PHR_STATE_CD=state_of(lon), PHR_TYPE_CD=t, PHR_CHAIN_NM=chain, PHR_SITE_OF_CARE_CD=soc,
            PHR_IN_CURRENT_NTWK=int(rng.random() < CONFIG['CURRENT_NETWORK_SHARE']) if t!='INFUSION' else 1,
            PHR_ACCEPT_NEW_PATIENT=int(rng.random() > CONFIG['CLOSED_TO_NEW_SHARE']),
            PHR_CAPACITY_SCRIPTS=cap, PHR_QUALITY_SCORE=round(float(rng.uniform(ql, qh)), 1),
            # what the client is charged, by bucket
            CLT_RATE_GENERIC=rg, CLT_RATE_BRAND=rb, CLT_RATE_SPECIALTY=rs, CLIENT_CONTRACT_DIS_FEE=round(float(rng.uniform(*CONFIG['FEE_RANGE'])),2),
            # what the pharmacy is reimbursed, always deeper, reporting only
            PHR_RATE_GENERIC=deeper(rg), PHR_RATE_BRAND=deeper(rb), PHR_RATE_SPECIALTY=deeper(rs), PHR_DISP_FEE_CTRCTD=round(float(rng.uniform(0.55,1.60)),2),
            PHR_ADMIN_COST=int(rng.uniform(*CONFIG['ADMIN_RANGE'][adm_key])),
            SETTING_FACTOR=CONFIG['SETTING'].get(soc if t=='INFUSION' else t, 1.0)))
        i[0] += 1
    k_inf = 0
    for tier, n_m in CONFIG['MEMBERS_PER_TIER'].items():
        per = n_m/1000
        for _ in range(max(1, round(CONFIG['RETAIL_PER_1000'][tier]*per))): add('RETAIL', int(rng.uniform(600, 2600)), 3.5, 5.0, tier=tier)
        for _ in range(max(1, round(CONFIG['SPEC_PER_1000'][tier]*per))):   add('SPECIALTY', 900, 4.2, 5.0, tier=tier)
        for _ in range(max(1, round(CONFIG['INF_PER_1000'][tier]*per))):
            soc = INF_TYPES[k_inf % 4]
            add('INFUSION', int(rng.uniform(*CONFIG['INFUSION_CAPACITY'][soc])), 4.0, 5.0, soc, tier=tier); k_inf += 1
    for _ in range(CONFIG['N_MAIL']): add('MAIL', 2_000_000, 4.3, 4.9)
    return pd.DataFrame(rows)

# =============================================================================
# CLAIMS. One row per fill, Optum column names. Fills cluster into visits on shared dates.
# =============================================================================
def gen_claims(members, pharmacies, drugs):
    rng = np.random.default_rng(CONFIG['SEEDS']['claims']); rows = []; clm = 0
    pl = pharmacies.set_index('PHR_PMT_CNTR_SK'); lat_p, lon_p = pl.PHR_LAT.to_dict(), pl.PHR_LON.to_dict()
    by_type = {t: pharmacies[pharmacies.PHR_TYPE_CD==t].PHR_PMT_CNTR_SK.tolist() for t in pharmacies.PHR_TYPE_CD.unique()}
    cur_net = set(pharmacies[pharmacies.PHR_IN_CURRENT_NTWK==1].PHR_PMT_CNTR_SK)
    base = drugs._SEED_SHARE.to_numpy(); spec_idx = drugs.index[drugs.SPECIALTY_DRUG_PRG_IND==1].to_numpy()
    unmanaged = CONFIG['BASELINE_REALISM']=='UNMANAGED'
    rate_col = {'GENERIC':'CLT_RATE_GENERIC','BRAND':'CLT_RATE_BRAND','SPECIALTY':'CLT_RATE_SPECIALTY'}
    for _, m in members.iterrows():
        w = base.copy(); w[spec_idx] *= 9.0 if rng.random() < CONFIG['SPEC_MEMBER_SHARE'] else 0.012
        tilt = rng.dirichlet(w*14 + 0.05)
        n_scripts = max(1, int(rng.gamma(2.2, 1.6)))                       # distinct ongoing prescriptions
        is_mail = rng.random() < CONFIG['MAIL_SHARE_TODAY']
        ninety = rng.random() < CONFIG['NINETY_SHARE_TODAY']
        # the member's usual site per kind of place, chosen once
        usual = {}
        for t in by_type:
            cand = [p for p in by_type[t] if p in cur_net] or by_type[t]
            near = sorted(cand, key=lambda q: haversine_miles(m.MBR_LAT, m.MBR_LON, lat_p[q], lon_p[q]))[:6]
            if unmanaged and t=='INFUSION':
                hosp = [q for q in near if pl.loc[q,'PHR_SITE_OF_CARE_CD']=='HOSPITAL_OPD']
                usual[t] = hosp[0] if hosp and rng.random() < CONFIG['HOSPITAL_SHARE_UNMANAGED'] else near[0]
            elif unmanaged and t=='RETAIL':
                ww = np.array([(1.0 - pl.loc[q,'CLT_RATE_GENERIC'])**2 for q in near]); ww /= ww.sum()   # dearer store likelier
                usual[t] = near[rng.choice(len(near), p=ww)]
            else:
                usual[t] = near[rng.integers(min(3, len(near)))]
        scripts = [drugs.iloc[rng.choice(len(drugs), p=tilt/tilt.sum())] for _ in range(n_scripts)]
        for d in scripts:
            if d.BENEFIT_TYPE_CD=='MEDICAL': t='INFUSION'
            elif d.SPECIALTY_DRUG_PRG_IND==1: t='SPECIALTY'
            elif is_mail and d.MAINT_IND==1: t='MAIL'
            else: t='RETAIL'
            p_id = usual[t]
            bucket = 'SPECIALTY' if d.SPECIALTY_DRUG_PRG_IND==1 else ('GENERIC' if rng.random() < d.GENERIC_SHARE else 'BRAND')
            awp_day = d.AWP_DAY_GENERIC if bucket=='GENERIC' else d.AWP_DAY_BRAND
            days = 90 if (t=='MAIL' or (t=='RETAIL' and d.MAINT_IND==1 and ninety)) else 30
            n_fills = 365 // days if d.MAINT_IND==1 or t in ('SPECIALTY','INFUSION') else int(rng.integers(1, 4))
            # fills for this script land on the member's visit dates for that kind of place, so fills share dates
            for f in range(n_fills):
                fill_day = int((f * days + (hash(m.MBR_SK+t) % days)) % 365)
                paid = awp_day * days * (1 - pl.loc[p_id, rate_col[bucket]]) * pl.loc[p_id,'SETTING_FACTOR']
                rows.append(dict(CLM_SK=f'C{clm:07d}', MBR_SK=m.MBR_SK, PHR_PMT_CNTR_SK=p_id, GPI_SK=d.GPI_SK,
                    FILLED_DT=pd.Timestamp('2025-01-01') + pd.Timedelta(days=fill_day),
                    DLVRY_CHANL={'RETAIL':'Retail','MAIL':'MAIL','SPECIALTY':'SPECIALTY','INFUSION':'MEDICAL'}[t],
                    CLAIM_STAT_ID='PAID', REJ_CNT=0, REVERSAL_IND='N',
                    PRORATED_DAYS_SUPPLY=days, BRND_TRADE_NM_FLAG={'GENERIC':'G','BRAND':'B','SPECIALTY':'B'}[bucket],
                    PRICE_BUCKET=bucket, MAIL_ORDR_CORP_IND=int(t=='MAIL'), RETL_90_CORP_IND=int(t=='RETAIL' and days==90),
                    SPECIALTY_DRUG_PRG_MBR_IND=int(d.SPECIALTY_DRUG_PRG_IND),
                    CAL_INGRED_COST_PAID=round(paid,2), CAL_DISPENSING_FEE=pl.loc[p_id,'CLIENT_CONTRACT_DIS_FEE'],
                    CAL_PATIENT_PAY_AMT=round(paid*float(rng.uniform(.02,.25)),2), BENEFIT_TYPE_CD=d.BENEFIT_TYPE_CD))
                clm += 1
    return pd.DataFrame(rows)

# =============================================================================
# DERIVATIONS. Everything the formula needs, computed from claims, nothing assumed.
# =============================================================================
KIND = {'Retail':'RETAIL', 'MAIL':'RETAIL', 'SPECIALTY':'SPECIALTY', 'MEDICAL':'INFUSION'}   # mail is the retail kind, held

def derive(members, pharmacies, drugs, claims):
    c = claims[(claims.CLAIM_STAT_ID=='PAID') & (claims.REVERSAL_IND=='N') & (claims.REJ_CNT==0)].copy()
    c['KIND'] = c.DLVRY_CHANL.map(KIND)
    pl = pharmacies.set_index('PHR_PMT_CNTR_SK'); dd = drugs.set_index('GPI_SK')
    rate_col = {'GENERIC':'CLT_RATE_GENERIC','BRAND':'CLT_RATE_BRAND','SPECIALTY':'CLT_RATE_SPECIALTY'}
    # days by member, class, bucket, with utilization trend applied
    c['TREND'] = c.GPI_SK.map(dd.GPI_CLASS_CD).map(CONFIG['TREND']).fillna(1.0)
    days = (c.assign(D=c.PRORATED_DAYS_SUPPLY*c.TREND).groupby(['MBR_SK','GPI_SK','PRICE_BUCKET']).D.sum()
             .rename('DAYS').reset_index())
    # the member's own price per day: ingredient paid per day grossed back up by today's rate and setting
    c['RATE_TODAY'] = [pl.loc[p, rate_col[b]] for p,b in zip(c.PHR_PMT_CNTR_SK, c.PRICE_BUCKET)]
    c['SETTING_TODAY'] = c.PHR_PMT_CNTR_SK.map(pl.SETTING_FACTOR)
    c['AWP_PAID'] = c.CAL_INGRED_COST_PAID / ((1 - c.RATE_TODAY) * c.SETTING_TODAY)
    awp = (c.groupby(['MBR_SK','GPI_SK','PRICE_BUCKET']).agg(AWP=('AWP_PAID','sum'), DS=('PRORATED_DAYS_SUPPLY','sum'))
             .assign(AWP_DAY=lambda x: x.AWP/x.DS).reset_index()[['MBR_SK','GPI_SK','PRICE_BUCKET','AWP_DAY']])
    demand = days.merge(awp, on=['MBR_SK','GPI_SK','PRICE_BUCKET'])
    demand['KIND'] = demand.GPI_SK.map(lambda k: 'INFUSION' if dd.loc[k,'BENEFIT_TYPE_CD']=='MEDICAL'
                                       else ('SPECIALTY' if dd.loc[k,'SPECIALTY_DRUG_PRG_IND']==1 else 'RETAIL'))
    demand['SUPPLY'] = [90 if (dd.loc[k,'MAINT_IND']==1 and mk in set(c[c.PRORATED_DAYS_SUPPLY==90].MBR_SK)) else 30
                        for k,mk in zip(demand.GPI_SK, demand.MBR_SK)]
    # visits: distinct fill dates by member and kind of place, trended
    visits = (c.groupby(['MBR_SK','KIND']).agg(VISITS=('FILLED_DT','nunique'), FILLS=('CLM_SK','size'), TREND=('TREND','mean'))
               .reset_index())
    visits['VISITS'] = (visits.VISITS*visits.TREND).round(1); visits['FILLS'] = (visits.FILLS*visits.TREND).round(1)
    # today's site by member and kind of place: the one used most. Mail members' retail kind = their mail site
    today = (c.groupby(['MBR_SK','KIND','PHR_PMT_CNTR_SK']).size().rename('n').reset_index()
               .sort_values('n', ascending=False).drop_duplicates(['MBR_SK','KIND'])
               .rename(columns={'PHR_PMT_CNTR_SK':'TODAY_SITE'})[['MBR_SK','KIND','TODAY_SITE']])
    me = members.set_index('MBR_SK')
    today['D0'] = [0.0 if pl.loc[p,'PHR_TYPE_CD']=='MAIL' else
                   haversine_miles(me.loc[m,'MBR_LAT'], me.loc[m,'MBR_LON'], pl.loc[p,'PHR_LAT'], pl.loc[p,'PHR_LON'])
                   for m,p in zip(today.MBR_SK, today.TODAY_SITE)]
    flags = c.groupby('MBR_SK').agg(MAIL_TODAY=('MAIL_ORDR_CORP_IND','max'), NINETY_TODAY=('RETL_90_CORP_IND','max')).reset_index()
    flags['NINETY_TODAY'] = ((flags.NINETY_TODAY==1) | (flags.MAIL_TODAY==1)).astype(int)
    # how many members use more than one store of the same kind today, the consolidation caveat
    multi = (c.groupby(['MBR_SK','KIND']).PHR_PMT_CNTR_SK.nunique().rename('n_sites').reset_index())
    return dict(demand=demand, visits=visits, today=today, flags=flags, multi=multi)

# =============================================================================
# FOOTPRINT. Synthetic ZIPs as grid cells over the metro. Populated whether or not members live there.
# =============================================================================
def gen_footprint(members):
    g = CONFIG['ZIP_GRID_MILES']; R = CONFIG['TIER_MILES']['RURAL']; rows = []; rng = np.random.default_rng(17)
    lat0, lon0 = CONFIG['CENTER']; n = int(np.ceil(R/g))
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            lat = lat0 + i*g/MI_LAT; lon = lon0 + j*g/MI_LON
            d = haversine_miles(lat, lon, lat0, lon0)
            if d <= R:
                t = tier_of(lat, lon)
                if rng.random() < CONFIG['SETTLED_SHARE'][t]:
                    rows.append(dict(ZIP=f'Z{len(rows):04d}', ZIP_LAT=lat, ZIP_LON=lon, SETTING=t))
    fp = pd.DataFrame(rows)
    # members to nearest grid ZIP
    zl, zn = fp[['ZIP_LAT','ZIP_LON']].to_numpy(), fp.ZIP.to_numpy()
    idx = [int(np.argmin((zl[:,0]-a)**2*MI_LAT**2 + (zl[:,1]-b)**2*MI_LON**2)) for a,b in zip(members.MBR_LAT, members.MBR_LON)]
    members['MBR_ZIP_CD'] = zn[idx]
    fp['MEMBERS_TODAY'] = fp.ZIP.map(members.MBR_ZIP_CD.value_counts()).fillna(0).astype(int)
    return fp

# =============================================================================
# BUILD, REPORT, EXPORT
# =============================================================================
def build():
    stamp('generating')
    drugs = gen_drug_dim(); members = gen_member_dim(); pharmacies = gen_pharmacy_dim()
    claims = gen_claims(members, pharmacies, drugs)
    footprint = gen_footprint(members)
    stamp('deriving from claims')
    dv = derive(members, pharmacies, drugs, claims)
    return dict(drugs=drugs, members=members, pharmacies=pharmacies, claims=claims, footprint=footprint, **dv)

def report(D):
    members, ph, claims, drugs, fp = D['members'], D['pharmacies'], D['claims'], D['drugs'], D['footprint'].copy()
    dem, vis, tod, fl, multi = D['demand'], D['visits'], D['today'], D['flags'], D['multi']
    pl = ph.set_index('PHR_PMT_CNTR_SK')
    print("\n================ DATA REPORT ================")
    print(f"members {len(members):,} | sites {len(ph)} {ph.PHR_TYPE_CD.value_counts().to_dict()} | claims {len(claims):,} | footprint ZIPs {len(fp)}")
    print(f"members by setting {members.MBR_URBAN_RURAL_CD.value_counts().to_dict()}")
    print(f"channel mix {claims.DLVRY_CHANL.value_counts().to_dict()} | buckets {claims.PRICE_BUCKET.value_counts().to_dict()}")
    print(f"mail members today {int(fl.MAIL_TODAY.sum())} | on 90 day today {int(fl.NINETY_TODAY.sum())} | held at these values")
    print(f"current network {int(ph.PHR_IN_CURRENT_NTWK.sum())} of {len(ph)} candidates | closed to new {int((ph.PHR_ACCEPT_NEW_PATIENT==0).sum())}")

    print("\n--- contract rates, price per day of a $100 AWP drug, cheapest to dearest store ---")
    for b, col in [('GENERIC','CLT_RATE_GENERIC'),('BRAND','CLT_RATE_BRAND'),('SPECIALTY','CLT_RATE_SPECIALTY')]:
        r = ph[ph.PHR_TYPE_CD.isin(['RETAIL','SPECIALTY'])][col]
        print(f"  {b:9s} rate {r.min():.3f} to {r.max():.3f} -> ${100*(1-r.max()):.2f} to ${100*(1-r.min()):.2f} per day, "
              f"spread {100*((1-r.min())/(1-r.max())-1):.0f}%")
    print("\n--- spend by bucket today ---")
    sp = claims.groupby('PRICE_BUCKET').CAL_INGRED_COST_PAID.sum(); print('  ' + ' | '.join(f"{k} ${v:,.0f} ({100*v/sp.sum():.0f}%)" for k,v in sp.items()))
    spec = claims[claims.SPECIALTY_DRUG_PRG_MBR_IND==1]
    print(f"  specialty and infused: {100*len(spec)/len(claims):.1f}% of fills, {100*spec.CAL_INGRED_COST_PAID.sum()/claims.CAL_INGRED_COST_PAID.sum():.0f}% of spend")

    print("\n--- visits, the disruption weight ---")
    v = vis.groupby('KIND').agg(members=('MBR_SK','nunique'), visits_avg=('VISITS','mean'), fills_avg=('FILLS','mean')).round(1)
    print(v.to_string())
    print(f"  fills per visit overall {claims.groupby(['MBR_SK','FILLED_DT']).size().mean():.2f}")

    print("\n--- today's sites ---")
    print(f"  every member has a today site for each kind they use: {tod.groupby('MBR_SK').KIND.count().sum() == len(tod)}")
    print(f"  distance to today's site, avg by kind: {tod.groupby('KIND').D0.mean().round(2).to_dict()}")
    mm = multi[(multi.KIND=='RETAIL') & (multi.n_sites>1)]
    print(f"  members using more than one retail store today: {len(mm)} of {multi[multi.KIND=='RETAIL'].MBR_SK.nunique()} "
          f"(consolidated to their most used store, the stated caveat)")

    print("\n--- geography and headroom, by setting ---")
    ret = ph[ph.PHR_TYPE_CD=='RETAIL']
    for t in ('URBAN','SUBURBAN','RURAL'):
        mm_ = members[members.MBR_URBAN_RURAL_CD==t]
        near = [min(haversine_miles(a,b,c_,d) for c_,d in zip(ret.PHR_LAT,ret.PHR_LON)) for a,b in zip(mm_.MBR_LAT,mm_.MBR_LON)]
        print(f"  {t:9s} {len(mm_):>5} members | nearest retail avg {np.mean(near):4.1f} mi | retail sites in tier {sum(1 for a,b in zip(ret.PHR_LAT,ret.PHR_LON) if tier_of(a,b)==t):>3}")
    r_ = claims[claims.DLVRY_CHANL=='Retail']; terc = ph.CLT_RATE_GENERIC.quantile(0.33)
    h_ = claims[claims.DLVRY_CHANL=='MEDICAL'].PHR_PMT_CNTR_SK.map(pl.PHR_SITE_OF_CARE_CD)
    print(f"  headroom check: {100*(r_.PHR_PMT_CNTR_SK.map(pl.CLT_RATE_GENERIC)<terc).mean():.0f}% of retail fills at bottom third rate stores | "
          f"hospital share of infusions {100*(h_=='HOSPITAL_OPD').mean():.0f}%  ({CONFIG['BASELINE_REALISM']} baseline)")

    print("\n--- territory pre check: can each footprint ZIP meet adequacy with ANY candidate accepting new patients ---")
    ok_new = ret[ret.PHR_ACCEPT_NEW_PATIENT==1]
    fp['ADEQ_MILES'] = np.nan; fp['ADEQ_STATUS'] = ''
    for t in ('URBAN','SUBURBAN','RURAL'):
        X = CONFIG['ADEQUACY_MILES'][t]; XF = CONFIG['ADEQUACY_FALLBACK'].get(t); R_ = CONFIG['ADEQUACY_COUNT'][t]
        z = fp[fp.SETTING==t]
        def count_within(zl, zn, X_): return sum(1 for a,b in zip(ok_new.PHR_LAT,ok_new.PHR_LON) if haversine_miles(zl,zn,a,b)<=X_)
        met, fb, gap = 0, 0, 0
        for idx, row in z.iterrows():
            if count_within(row.ZIP_LAT, row.ZIP_LON, X) >= R_: fp.loc[idx,'ADEQ_MILES']=X; fp.loc[idx,'ADEQ_STATUS']='primary'; met+=1
            elif XF and count_within(row.ZIP_LAT, row.ZIP_LON, XF) >= R_: fp.loc[idx,'ADEQ_MILES']=XF; fp.loc[idx,'ADEQ_STATUS']='fallback'; fb+=1
            else: fp.loc[idx,'ADEQ_STATUS']='gap'; gap+=1
        print(f"  {t:9s} {len(z):>4} ZIPs, need {R_} within {X:.0f} mi: {met} meet it" +
              (f" | {fb} need the {XF:.0f} mi fallback" if XF else "") + f" | {gap} cannot be met by any candidate, reported as gaps"
              + f" | {int((z.MEMBERS_TODAY==0).sum())} have no member today")
    D['footprint'] = fp
    print("\n--- projected load vs capacity, sites over headroom today ---")
    load = vis.merge(tod, on=['MBR_SK','KIND']).groupby('TODAY_SITE').FILLS.sum()
    cap = pl.PHR_CAPACITY_SCRIPTS
    over = [(p, load[p], cap[p]) for p in load.index if load[p] > CONFIG['HEADROOM']*cap[p] and pl.loc[p,'PHR_TYPE_CD']!='MAIL']
    print(f"  {len(over)} sites already above {100*CONFIG['HEADROOM']:.0f}% of capacity on projected load; they stay feasible at their own load")
    print("=============================================\n")

def export(D):
    D['members'].to_csv('exports/members.csv', index=False); D['pharmacies'].to_csv('exports/pharmacies.csv', index=False)
    D['drugs'].to_csv('exports/drugs.csv', index=False); D['claims'].to_csv('exports/claims.csv', index=False)
    D['footprint'].to_csv('exports/footprint_zips.csv', index=False)
    for k in ('demand','visits','today','flags'): D[k].to_csv(f'exports/derived_{k}.csv', index=False)
    stamp('exports written')

def load_data(verbose=True):
    D = build()
    if verbose: report(D)
    return D

if __name__ == '__main__':
    D = load_data(); export(D)
