# =============================================================================
# PHARMACY NETWORK MODEL  ·  PART 2  ·  MODEL, SWEEPS, CHARTS, EXCEL
# Pranshi Saxena  ·  Optum Rx Quantum Initiative
#
# Run:   python pnm_model.py             (put pnm_data.py in the same folder)
# Quick: PNM_SMOKE=1 python pnm_model.py
# Needs: pip install pyomo highspy pandas numpy matplotlib openpyxl
#
# The signed off formula.
#   minimize   drug bill + network cost
#            + A · Σ visits[m,c] · moved[m,c]                 moved or not, all kinds of place
#            + λ · Σ visits[m,c] · extra_miles[m,c]           travel kinds only
#   rules      everyone served in network at the right kind of site, the 10 mile wall,
#              capacity with headroom, closed stores keep incumbents, chains by state,
#              mandates, quality by today's fills, territory adequacy per footprint ZIP.
#   sweeps     A with λ held: savings against members moved.
#              λ with A held: savings against extra distance.
#   reported   savings pure of A and λ. Bands on extra miles with copay economics.
# =============================================================================
import os, sys, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pyomo.environ import (ConcreteModel, Var, Objective, Constraint, ConstraintList, Binary, minimize, value, SolverFactory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pnm_data import load_data, export, CONFIG, haversine_miles, tier_of
SMOKE = os.environ.get('PNM_SMOKE') == '1'
T0 = time.time()
def stamp(msg): print(f"[{time.time()-T0:5.0f}s] {msg}", flush=True)
os.makedirs('charts', exist_ok=True); os.makedirs('exports', exist_ok=True)
CONFIG.update({'CANDIDATE_MODES': ['CURRENT','ALL'],   # CURRENT: keep or drop today's sites only. ALL: may also add candidates
               'MAX_OPTIONS': 8, 'SOLVER_TIME_LIMIT': 300, 'SOLVER_GAP': 0.0005,
               'BANDS': [(-1e9,0.0,'0 or closer'),(0.0,2.0,'0 to 2'),(2.0,5.0,'2 to 5'),(5.0,8.0,'5 to 8'),(8.0,10.0,'8 to 10')],
               'COPAY_INCENTIVE_BY_BAND': {'0 or closer':0,'0 to 2':0,'2 to 5':100,'5 to 8':250,'8 to 10':400}})
if SMOKE: CONFIG.update(A_SWEEP=[0, 5, 50], LAMBDA_SWEEP=[0, 2, 10])

# =============================================================================
# 1. DATA IN
# =============================================================================
D = load_data(verbose=True); export(D)          # data tables always sit beside the results
members, ph_df, drugs, claims, fp = D['members'], D['pharmacies'], D['drugs'], D['claims'], D['footprint']
demand, visits, today, flags = D['demand'], D['visits'], D['today'], D['flags']
ph = ph_df.set_index('PHR_PMT_CNTR_SK'); me = members.set_index('MBR_SK'); dd = drugs.set_index('GPI_SK')
SITES = ph.index.tolist()
BY_TYPE = {t: [p for p in SITES if ph.loc[p,'PHR_TYPE_CD']==t] for t in ('RETAIL','MAIL','SPECIALTY','INFUSION')}
KIND_TYPE = {'RETAIL':'RETAIL', 'SPECIALTY':'SPECIALTY', 'INFUSION':'INFUSION'}
TRAVEL = ['RETAIL','INFUSION'] if CONFIG['SPECIALTY_DELIVERED'] else ['RETAIL','SPECIALTY','INFUSION']
RATE_COL = {'GENERIC':'CLT_RATE_GENERIC','BRAND':'CLT_RATE_BRAND','SPECIALTY':'CLT_RATE_SPECIALTY'}
PHR_COL  = {'GENERIC':'PHR_RATE_GENERIC','BRAND':'PHR_RATE_BRAND','SPECIALTY':'PHR_RATE_SPECIALTY'}
CURRENT_SITES = sorted(set(today.TODAY_SITE) | set(p for p in SITES if ph.loc[p,'PHR_IN_CURRENT_NTWK']==1))
stamp('data loaded')

# =============================================================================
# 2. COST OF SERVING A MEMBER AT A SITE, PER KIND OF PLACE
#    Mail members' maintenance drugs stay at their mail site, held. Cost is a constant.
# =============================================================================
mail_today = set(flags[flags.MAIL_TODAY==1].MBR_SK)
mail_site = (claims[claims.DLVRY_CHANL=='MAIL'].groupby(['MBR_SK','PHR_PMT_CNTR_SK']).size().rename('n').reset_index()
               .sort_values('n', ascending=False).drop_duplicates('MBR_SK').set_index('MBR_SK').PHR_PMT_CNTR_SK.to_dict())
dem = demand.merge(dd[['MAINT_IND']], left_on='GPI_SK', right_index=True)
dem['HELD_AT_MAIL'] = dem.MBR_SK.isin(mail_today) & (dem.MAINT_IND==1) & (dem.KIND=='RETAIL')
dem['AWP_TOTAL'] = dem.DAYS*dem.AWP_DAY; dem['FILLS_N'] = dem.DAYS/dem.SUPPLY
free = dem[~dem.HELD_AT_MAIL]
AGG = free.groupby(['MBR_SK','KIND','PRICE_BUCKET']).AWP_TOTAL.sum().unstack(fill_value=0.0)
for b in ('GENERIC','BRAND','SPECIALTY'):
    if b not in AGG: AGG[b] = 0.0
FILLS_N = free.groupby(['MBR_SK','KIND']).FILLS_N.sum()
def cost_of(m, c, p):
    a = AGG.loc[(m,c)]
    return (float(sum(a[b]*(1-ph.loc[p,RATE_COL[b]]) for b in ('GENERIC','BRAND','SPECIALTY')))*ph.loc[p,'SETTING_FACTOR']
            + float(FILLS_N.loc[(m,c)])*ph.loc[p,'CLIENT_CONTRACT_DIS_FEE'])
def reimb_of(m, c, p):
    a = AGG.loc[(m,c)]
    return (float(sum(a[b]*(1-ph.loc[p,PHR_COL[b]]) for b in ('GENERIC','BRAND','SPECIALTY')))*ph.loc[p,'SETTING_FACTOR']
            + float(FILLS_N.loc[(m,c)])*ph.loc[p,'PHR_DISP_FEE_CTRCTD'])
held = dem[dem.HELD_AT_MAIL].copy()
HELD_COST = sum(r.AWP_TOTAL*(1-ph.loc[mail_site[r.MBR_SK],RATE_COL[r.PRICE_BUCKET]])*0.90
                + r.FILLS_N*ph.loc[mail_site[r.MBR_SK],'CLIENT_CONTRACT_DIS_FEE'] for r in held.itertuples()) if len(held) else 0.0

# =============================================================================
# 3. STRUCTURES. Members by kind, today's site and distance, candidates inside the wall
# =============================================================================
def build_and_run(MODE):
  global SITES, BY_TYPE, IDX, PAIRS, CAND, DIST, C, R, DELTA, TOD, D0, VIS, FIL, mdl, HOME, ALLOT_CAND, ALLOT_GAP, ACCESS_TODAY, NORET, B0, B0_PHARM, ADMIN_TODAY, proj_load, TOTF, QF, q_today, n_terr, HOLD, RUNS, BANDS, DONE, runs, band_rows, member_rows, site_rows, allot_rows, worst_rows, RATE_OF
  SITES = [p for p in ph.index if MODE=='ALL' or p in CURRENT_SITES]
  BY_TYPE = {t: [p for p in SITES if ph.loc[p,'PHR_TYPE_CD']==t] for t in ('RETAIL','MAIL','SPECIALTY','INFUSION')}
  stamp(f"===== candidate mode {MODE}: {len(SITES)} sites the model may use =====")
  PAIRS = sorted(set(map(tuple, free[['MBR_SK','KIND']].drop_duplicates().to_numpy())))
  TOD = {(r.MBR_SK, r.KIND): r.TODAY_SITE for r in today.itertuples()}
  D0  = {(r.MBR_SK, r.KIND): r.D0 for r in today.itertuples()}
  VIS = {(r.MBR_SK, r.KIND): r.VISITS for r in visits.itertuples()}
  FIL = {(r.MBR_SK, r.KIND): r.FILLS  for r in visits.itertuples()}
  # a mail member's retail kind today site may be their mail site; their acute retail today site is the retail store they use
  ret_today = (claims[claims.DLVRY_CHANL=='Retail'].groupby(['MBR_SK','PHR_PMT_CNTR_SK']).size().rename('n').reset_index()
                 .sort_values('n', ascending=False).drop_duplicates('MBR_SK').set_index('MBR_SK').PHR_PMT_CNTR_SK.to_dict())
  for (m,c) in PAIRS:
      if c=='RETAIL' and ph.loc[TOD.get((m,c),''),'PHR_TYPE_CD'] if (m,c) in TOD else True:
          pass
  for (m,c) in PAIRS:
      if c=='RETAIL' and ((m,c) not in TOD or ph.loc[TOD[(m,c)],'PHR_TYPE_CD']=='MAIL'):
          p0 = ret_today.get(m) or min(BY_TYPE['RETAIL'], key=lambda q: haversine_miles(me.loc[m,'MBR_LAT'],me.loc[m,'MBR_LON'],ph.loc[q,'PHR_LAT'],ph.loc[q,'PHR_LON']))
          TOD[(m,c)] = p0; D0[(m,c)] = haversine_miles(me.loc[m,'MBR_LAT'],me.loc[m,'MBR_LON'],ph.loc[p0,'PHR_LAT'],ph.loc[p0,'PHR_LON'])
      VIS.setdefault((m,c), 1.0); FIL.setdefault((m,c), 1.0)
  stamp('building candidates and costs')
  DIST = {}
  CAND = {}
  HOME = set(p for p in SITES if ph.loc[p,'PHR_SITE_OF_CARE_CD']=='HOME_INFUSION')   # care comes to the member: no travel
  def is_home(p): return p in HOME
  for (m,c) in PAIRS:
      la, lo = me.loc[m,'MBR_LAT'], me.loc[m,'MBR_LON']; t = KIND_TYPE[c]
      ds = {p: (0.0 if is_home(p) else haversine_miles(la, lo, ph.loc[p,'PHR_LAT'], ph.loc[p,'PHR_LON'])) for p in BY_TYPE[t]}
      if c in TRAVEL:
          travel = sorted([p for p in ds if not is_home(p) and ds[p] <= CONFIG['WALL_MILES']], key=ds.get)[:CONFIG['MAX_OPTIONS']]
          home = sorted([p for p in ds if is_home(p)], key=lambda p: cost_of(m,c,p))[:3]      # home infusion, any distance
          opts = travel + home
          if c=='INFUSION':   # no downgrade: an infusion patient is never moved to a more expensive setting than today's
              f0 = ph.loc[TOD[(m,c)],'SETTING_FACTOR']
              opts = [p for p in opts if ph.loc[p,'SETTING_FACTOR'] <= f0 + 1e-9]
      else:
          opts = sorted(ds, key=lambda p: cost_of(m,c,p))[:CONFIG['MAX_OPTIONS']]       # delivered: nearest by price
      if TOD[(m,c)] not in opts: opts.append(TOD[(m,c)])
      CAND[(m,c)] = opts
      for p in opts: DIST[(m,c,p)] = ds[p]
  for k in list(D0):                                                                     # today at home infusion: no travel today
      if is_home(TOD[k]): D0[k] = 0.0
  IDX = [(m,c,p) for (m,c),o in CAND.items() for p in o]
  C   = {t: cost_of(t[0],t[1],t[2]) for t in IDX}
  R   = {t: reimb_of(t[0],t[1],t[2]) for t in IDX}
  DELTA = {t: (max(0.0, DIST[t]-D0[(t[0],t[1])]) if t[1] in TRAVEL else 0.0) for t in IDX}
  ADMIN = ph.PHR_ADMIN_COST.to_dict(); NEWOK = ph.PHR_ACCEPT_NEW_PATIENT.to_dict()
  B0 = sum(cost_of(m,c,TOD[(m,c)]) for (m,c) in PAIRS) + HELD_COST
  B0_PHARM = sum(reimb_of(m,c,TOD[(m,c)]) for (m,c) in PAIRS)
  ADMIN_TODAY = sum(ADMIN[p] for p in CURRENT_SITES)
  proj_load = {p: 0.0 for p in SITES}
  for (m,c) in PAIRS: proj_load[TOD[(m,c)]] += FIL[(m,c)]
  stamp(f"assignment variables {len(IDX):,} | member kind pairs {len(PAIRS):,} | baseline drug bill ${B0:,.0f} | admin today ${ADMIN_TODAY:,.0f}")

  # =============================================================================
  # 4. THE MODEL
  # =============================================================================
  stamp('building model')
  mdl = ConcreteModel()
  CHAIN_STATE = sorted(set((ph.loc[p,'PHR_CHAIN_NM'], ph.loc[p,'PHR_STATE_CD']) for p in SITES))
  mdl.x = Var(SITES, within=Binary); mdl.a = Var(IDX, within=Binary); mdl.z = Var(CHAIN_STATE, within=Binary)
  mdl.c = ConstraintList()
  for (m,c),o in CAND.items(): mdl.c.add(sum(mdl.a[m,c,p] for p in o) == 1)              # everyone served
  for t in IDX: mdl.c.add(mdl.a[t] <= mdl.x[t[2]])                                          # in network only
  use = {p: [] for p in SITES}
  for t in IDX: use[t[2]].append(t)
  for p in SITES:                                                                           # capacity with headroom
      if use[p] and ph.loc[p,'PHR_TYPE_CD']!='MAIL':
          cap = max(CONFIG['HEADROOM']*ph.loc[p,'PHR_CAPACITY_SCRIPTS'], proj_load[p])
          mdl.c.add(sum(FIL[(t[0],t[1])]*mdl.a[t] for t in use[p]) <= cap*mdl.x[p])
  for t in IDX:                                                                             # closed stores keep incumbents
      if NEWOK[t[2]]==0 and TOD[(t[0],t[1])] != t[2]: mdl.c.add(mdl.a[t]==0)
  for p in SITES:                                                                           # chains by state, mandates
      h, s_ = ph.loc[p,'PHR_CHAIN_NM'], ph.loc[p,'PHR_STATE_CD']
      if h in ('ChainA','ChainB'): mdl.c.add(mdl.x[p]==mdl.z[h,s_])
      if ph.loc[p,'PHR_TYPE_CD']=='MAIL' or (ph.loc[p,'PHR_TYPE_CD']=='RETAIL' and s_ in CONFIG['AWP_STATES']): mdl.c.add(mdl.x[p]==1)
  Q_W = {t: FIL[(t[0],t[1])]*ph.loc[t[2],'PHR_QUALITY_SCORE'] for t in IDX}               # quality, today's fills as weights
  TOTF = sum(FIL[k] for k in PAIRS)
  q_today = sum(FIL[k]*ph.loc[TOD[k],'PHR_QUALITY_SCORE'] for k in PAIRS)/TOTF
  q_best  = sum(FIL[k]*max(ph.loc[p,'PHR_QUALITY_SCORE'] for p in CAND[k]) for k in PAIRS)/TOTF
  QF = min(CONFIG['QUALITY_FLOOR'], q_today-1e-3, q_best-1e-3)
  mdl.c.add(sum(Q_W[t]*mdl.a[t] for t in IDX) >= QF*TOTF)
  # territory adequacy per footprint ZIP, retail, pharmacies accepting new patients, at the ZIP's own distance
  n_terr = 0; expansion = []
  for r in fp.itertuples():
      if r.ADEQ_STATUS not in ('primary','fallback'): continue
      near = [p for p in BY_TYPE['RETAIL'] if NEWOK[p]==1 and haversine_miles(r.ZIP_LAT, r.ZIP_LON, ph.loc[p,'PHR_LAT'], ph.loc[p,'PHR_LON']) <= r.ADEQ_MILES]
      if len(near) >= CONFIG['ADEQUACY_COUNT'][r.SETTING]: mdl.c.add(sum(mdl.x[p] for p in near) >= CONFIG['ADEQUACY_COUNT'][r.SETTING]); n_terr += 1
      else: expansion.append(r.ZIP)
  EXPANSION[MODE] = expansion
  print(f"territory: {n_terr} ZIP rules enforced with {MODE} candidates | {len(expansion)} ZIPs cannot be met by {MODE} candidates: "
        + ("future network expansion required" if MODE=='CURRENT' else "reported as gaps"))
  # every member is allotted a retail store, whether or not they used retail this year.
  # zero volume means zero cost, so this is a guarantee, not an optimization: an open store accepting
  # new patients within their setting's adequacy distance must exist. Assigned to the nearest one after the solve.
  NORET = [m for m in me.index if (m,'RETAIL') not in CAND]
  ALLOT_CAND, ALLOT_GAP = {}, []
  cur_ret = [p for p in BY_TYPE['RETAIL'] if p in CURRENT_SITES]
  for m in NORET:
      la, lo = me.loc[m,'MBR_LAT'], me.loc[m,'MBR_LON']; t = me.loc[m,'MBR_URBAN_RURAL_CD']
      ds = {p: haversine_miles(la, lo, ph.loc[p,'PHR_LAT'], ph.loc[p,'PHR_LON']) for p in BY_TYPE['RETAIL'] if NEWOK[p]==1}
      X = CONFIG['ADEQUACY_MILES'][t]; near = [p for p,d in ds.items() if d <= X]
      if not near and t in CONFIG['ADEQUACY_FALLBACK']: X = CONFIG['ADEQUACY_FALLBACK'][t]; near = [p for p,d in ds.items() if d <= X]
      if near: ALLOT_CAND[m] = near; mdl.c.add(sum(mdl.x[p] for p in near) >= 1)
      else: ALLOT_GAP.append(m)
  ACCESS_TODAY = {m: min(haversine_miles(me.loc[m,'MBR_LAT'], me.loc[m,'MBR_LON'], ph.loc[p,'PHR_LAT'], ph.loc[p,'PHR_LON']) for p in cur_ret) for m in NORET}
  print(f"retail allotment: {len(NORET)} members have no retail claims this year; {len(ALLOT_CAND)} get a guaranteed store within reach, "
        f"{len(ALLOT_GAP)} have no candidate within reach and are reported as access gaps")

  def set_objective(m_, A, LAM):
      if hasattr(m_,'obj'): m_.del_component(m_.obj)
      m_.obj = Objective(expr=sum(C[t]*m_.a[t] for t in IDX) + sum(ADMIN[p]*m_.x[p] for p in SITES)
                            + A*sum(VIS[k]*(1 - m_.a[k[0],k[1],TOD[k]]) for k in PAIRS)
                            + LAM*sum(VIS[(t[0],t[1])]*DELTA[t]*m_.a[t] for t in IDX), sense=minimize)
  def solve(m_):
      t0=time.time(); opt=SolverFactory('appsi_highs')
      opt.options['time_limit']=CONFIG['SOLVER_TIME_LIMIT']; opt.options['mip_rel_gap']=CONFIG['SOLVER_GAP']
      r=opt.solve(m_, load_solutions=False); st=str(r.solver.termination_condition)
      ok = st in ('optimal','maxTimeLimit','feasible')
      if ok:
          try: m_.solutions.load_from(r)
          except Exception as e: ok=False; st=f'{st} load failed'
      return ok, st, time.time()-t0
  stamp(f"model built | quality floor applied {QF:.2f} (today {q_today:.2f}) | territory rules {n_terr} ZIPs")

  # =============================================================================
  # 5. READING A SOLUTION. Savings pure of A and λ.
  # =============================================================================
  def band_of(d):
      for lo,hi,nm in CONFIG['BANDS']:
          if lo < d <= hi or (nm=='0 or closer' and d<=0): return nm
      return '8 to 10'
  def read(tag, A, LAM):
      av = {t: round(value(mdl.a[t])) for t in IDX}
      sel = set(p for p in SITES if round(value(mdl.x[p]))==1)
      asg = {(t[0],t[1]): t[2] for t in IDX if av[t]==1}
      drug = sum(C[t] for t in IDX if av[t]==1) + HELD_COST; adm = sum(ADMIN[p] for p in sel)
      pharm = sum(R[t] for t in IDX if av[t]==1)
      rows = []
      for k,p in asg.items():
          m,c = k; mv = int(p != TOD[k]); dlt = max(0.0, DIST[(m,c,p)]-D0[k]) if c in TRAVEL else 0.0
          rows.append(dict(design=tag, MBR_SK=m, kind=c, setting=me.loc[m,'MBR_URBAN_RURAL_CD'], today_site=TOD[k], new_site=p,
                           moved=mv, miles_today=round(D0[k],2), miles_new=round(DIST[(m,c,p)],2) if c in TRAVEL else np.nan,
                           extra_miles=round(dlt,2), visits=VIS[k], saving=round(cost_of(m,c,TOD[k]) - C[(m,c,p)],2),
                           band=band_of(dlt) if (mv and c in TRAVEL) else ('not moved' if not mv else 'delivered')))
      mv_df = pd.DataFrame(rows)
      moved_any = mv_df[mv_df.moved==1].MBR_SK.nunique()
      # the indefensible move: costs money AND sends the member farther. Costing a few dollars to bring someone closer is the distance term working
      neg = mv_df[(mv_df.moved==1) & (mv_df.saving < -1) & (mv_df.extra_miles > 0)]
      neg_moves, neg_dollars = len(neg), round(neg.saving.sum())
      closer_paid = mv_df[(mv_df.moved==1) & (mv_df.saving < -1) & (mv_df.extra_miles <= 0)]
      mv_df['home'] = mv_df.new_site.isin(HOME)
      tr = mv_df[(mv_df.moved==1) & mv_df.kind.isin(TRAVEL) & ~mv_df.home]
      trips = mv_df[mv_df.kind.isin(TRAVEL)]
      avg_before = float((trips.miles_today*trips.visits).sum()/trips.visits.sum()) if len(trips) else 0.0
      avg_after  = float((trips.miles_new*trips.visits).sum()/trips.visits.sum()) if len(trips) else 0.0
      # bands are built from member channel ROWS, so every moved row lands in exactly one band and savings reconcile
      bands = []
      def band_row(nm, kind_, g, inc):
          n = len(g)
          bands.append(dict(design=tag, band=nm, kind=kind_, member_channel_moves=n, saving=round(g.saving.sum()), _raw=float(g.saving.sum()),
                            saving_per_move=round(g.saving.sum()/n) if n else 0, copay_incentive_each=inc,
                            incentive_total=int(inc*n), net_after_incentive=round(g.saving.sum()-inc*n)))
      for lo,hi,nm in CONFIG['BANDS']:
          for kind_ in TRAVEL: band_row(nm, kind_, tr[(tr.band==nm) & (tr.kind==kind_)], CONFIG['COPAY_INCENTIVE_BY_BAND'][nm])
      hm = mv_df[(mv_df.moved==1) & mv_df.home]
      band_row('home infusion, no travel', 'INFUSION', hm, 0)
      dl = mv_df[(mv_df.moved==1) & (mv_df.kind=='SPECIALTY')]
      band_row('delivered, no travel', 'SPECIALTY', dl, 0)
      band_total = sum(b['_raw'] for b in bands); row_total = mv_df[mv_df.moved==1].saving.sum()
      assert abs(row_total - (B0-drug)) < 1.0, f'reconciliation failed: rows {row_total:,.0f} vs gross {B0-drug:,.0f}'
      assert abs(band_total - (B0-drug)) < 2.0, f'band reconciliation failed: bands {band_total:,.0f} vs gross {B0-drug:,.0f}'
      # allotted stores for members with no retail volume
      allot = []
      for m, cands in ALLOT_CAND.items():
          opn = [p for p in cands if p in sel]
          if opn:
              p_ = min(opn, key=lambda q: haversine_miles(me.loc[m,'MBR_LAT'], me.loc[m,'MBR_LON'], ph.loc[q,'PHR_LAT'], ph.loc[q,'PHR_LON']))
              allot.append(dict(design=tag, MBR_SK=m, setting=me.loc[m,'MBR_URBAN_RURAL_CD'], allotted_site=p_,
                                access_miles_today=round(ACCESS_TODAY[m],2),
                                access_miles_after=round(haversine_miles(me.loc[m,'MBR_LAT'], me.loc[m,'MBR_LON'], ph.loc[p_,'PHR_LAT'], ph.loc[p_,'PHR_LON']),2)))
      allot_df = pd.DataFrame(allot)
      coverage = dict(served_pairs=len(asg), of_pairs=len(PAIRS), allotted=len(allot_df), allot_gaps=len(ALLOT_GAP),
                      allot_access_before=round(float(allot_df.access_miles_today.mean()),2) if len(allot_df) else np.nan,
                      allot_access_after=round(float(allot_df.access_miles_after.mean()),2) if len(allot_df) else np.nan)
      summary = dict(design=tag, mode=MODE, A=A, lam=LAM, gross_saving=round(B0-drug), gross_saving_pct=round(100*(B0-drug)/B0,1),
                     admin_today=ADMIN_TODAY, admin_new=adm, net_saving=round(B0-drug+ADMIN_TODAY-adm),
                     members_moved=moved_any, members_moved_pct=round(100*moved_any/len(members),1),
                     moves_costing_money_and_farther=neg_moves, cost_of_those_moves=neg_dollars,
                     moves_paid_to_come_closer=len(closer_paid), paid_to_come_closer=round(-closer_paid.saving.sum()),
                     moved_retail=int(mv_df[(mv_df.moved==1)&(mv_df.kind=='RETAIL')].shape[0]),
                     moved_specialty=int(mv_df[(mv_df.moved==1)&(mv_df.kind=='SPECIALTY')].shape[0]),
                     moved_infusion=int(mv_df[(mv_df.moved==1)&(mv_df.kind=='INFUSION')].shape[0]),
                     sent_farther=int(tr[tr.extra_miles>0].MBR_SK.nunique()),
                     extra_miles_total=int((tr.extra_miles*tr.visits).sum()), avg_extra_among_farther=round(float(tr[tr.extra_miles>0].extra_miles.mean()),2) if (tr.extra_miles>0).any() else 0.0,
                     avg_trip_before=round(avg_before,2), avg_trip_after=round(avg_after,2),
                     sites_open=len(sel), kept=len([p for p in CURRENT_SITES if p in sel]), dropped=len([p for p in CURRENT_SITES if p not in sel]),
                     added=len([p for p in sel if p not in CURRENT_SITES]),
                     client_pays=round(drug), pharmacy_receives=round(pharm+HELD_COST), optum_spread=round(drug-pharm-HELD_COST),
                     incentive_total=int(sum(b['incentive_total'] for b in bands)), net_after_incentives=round(B0-drug-sum(b['incentive_total'] for b in bands)))
      summary.update(coverage)
      return summary, pd.DataFrame(bands).drop(columns='_raw'), mv_df, sel, allot_df

  # =============================================================================
  # 6. THE TWO SWEEPS
  # =============================================================================
  runs, band_rows, member_rows, site_rows, allot_rows, worst_rows = [], [], {}, [], {}, []
  DONE = set()
  def run(A, LAM, tag):
      if tag in DONE: return None                    # the held design appears in both sweeps, solve it once
      DONE.add(tag)
      set_objective(mdl, A, LAM); ok, st, dt = solve(mdl)
      if not ok: print(f"  {tag:<28} NOT SOLVED: {st}"); return None
      s, b, mv, sel, al = read(tag, A, LAM); s.update(status=st, solve_s=round(dt,1)); runs.append(s); band_rows.append(b); member_rows[tag]=mv; allot_rows[tag]=al
      w_ = mv[(mv.moved==1) & mv.kind.isin(TRAVEL) & ~mv.home].copy(); w_['extra_miles_year'] = (w_.extra_miles*w_.visits).round(0); w_['mode'] = MODE
      worst_rows.append(w_.sort_values('extra_miles_year', ascending=False).head(50))
      print(f"      coverage: {s['served_pairs']} of {s['of_pairs']} member kind pairs served in network | retail allotted to {s['allotted']} members "
            f"with no retail claims (access {s['allot_access_before']} -> {s['allot_access_after']} mi) | {s['allot_gaps']} access gaps")
      site_rows.append(dict(design=tag, mode=MODE, **{p: int(p in sel) for p in ph.index}))
      print(f"  {tag:<28} | gross ${s['gross_saving']:>11,} ({s['gross_saving_pct']:>4}%) net ${s['net_saving']:>11,} | moved {s['members_moved']:>5} "
            f"({s['members_moved_pct']:>4}%) farther {s['sent_farther']:>4} | extra mi {s['extra_miles_total']:>7,} | sites {s['sites_open']:>3} | "
          f"bad moves {s['moves_costing_money_and_farther']} | {st} {dt:.0f}s")
      return s
  stamp('benchmark: no penalties at all, the pure money optimum under the rules')
  run(0, 0, 'A=0 lam=0')
  stamp('sweep A, λ held')
  if not SMOKE: CONFIG['A_SWEEP'] = CONFIG['A_SWEEP'] + [100, 250, 500]
  for A in CONFIG['A_SWEEP']: run(A, CONFIG['LAMBDA_HOLD'], f"A={A} lam={CONFIG['LAMBDA_HOLD']}")
  stamp('sweep λ, A held')
  for LAM in CONFIG['LAMBDA_SWEEP']: run(CONFIG['A_HOLD'], LAM, f"A={CONFIG['A_HOLD']} lam={LAM}")
  RUNS = pd.DataFrame(runs); BANDS = pd.concat(band_rows, ignore_index=True); RUNS['mode'] = MODE; BANDS['mode'] = MODE
  HOLD = f"A={CONFIG['A_HOLD']} lam={CONFIG['LAMBDA_HOLD']}"
  print("\n=== RUNS ==="); print(RUNS[['design','gross_saving','gross_saving_pct','net_saving','members_moved','members_moved_pct','sent_farther',
                                      'extra_miles_total','avg_trip_before','avg_trip_after','sites_open','dropped','added','moves_costing_money_and_farther','moves_paid_to_come_closer','status']].to_string(index=False))
  print(f"\n=== BANDS, design {HOLD}, mode {MODE}. Rows are member channel moves, one member may appear in two kinds ==="); print(BANDS[BANDS.design==HOLD].to_string(index=False))
  print(f"bands reconcile to gross saving: {abs(BANDS[BANDS.design==HOLD].saving.sum() - RUNS[RUNS.design==HOLD].gross_saving.iloc[0]) < 2}")

  # =============================================================================
  # 7. CHARTS. Money on one axis, people on the other.
  # =============================================================================
  stamp('charts')
  NAVY, RED, GREEN, GREY, ORANGE, LBLUE = '#1F3B57', '#B3202C', '#2E7D5B', '#7A8590', '#E08A1E', '#5C8CB8'
  plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
  ra = RUNS[RUNS.lam==CONFIG['LAMBDA_HOLD']].sort_values('A'); rl = RUNS[RUNS.A==CONFIG['A_HOLD']].sort_values('lam')

  fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  ax = axes[0]; ax.plot(ra.members_moved, ra.gross_saving/1e6, '-o', color=NAVY, lw=2, ms=7)
  for _, r in ra.iterrows(): ax.annotate(f"A={r.A:g}", (r.members_moved, r.gross_saving/1e6), textcoords='offset points', xytext=(5,-12), fontsize=8, color=GREY)
  ax.set_xlabel('members who change pharmacy'); ax.set_ylabel('gross saving, $ million per year')
  ax.set_title(f"How much stability costs. Movement weight A swept, λ = {CONFIG['LAMBDA_HOLD']}", fontsize=10); ax.grid(alpha=.25)
  ax = axes[1]; ax.plot(rl.extra_miles_total, rl.gross_saving/1e6, '-s', color=ORANGE, lw=2, ms=7)
  for _, r in rl.iterrows(): ax.annotate(f"λ={r.lam:g}", (r.extra_miles_total, r.gross_saving/1e6), textcoords='offset points', xytext=(5,-12), fontsize=8, color=GREY)
  ax.set_xlabel('extra member miles per year, all trips'); ax.set_ylabel('gross saving, $ million per year')
  ax.set_title(f"How much distance costs. Distance weight λ swept, A = {CONFIG['A_HOLD']}", fontsize=10); ax.grid(alpha=.25)
  plt.suptitle(f'The trade off, {MODE} candidates. Every point is one complete lawful network', fontsize=12, fontweight='bold')
  plt.tight_layout(); plt.savefig(f'charts/{MODE}_01_tradeoff_curves.png', bbox_inches='tight'); plt.close('all')

  bh = BANDS[(BANDS.design==HOLD) & ~BANDS.band.isin(['delivered, no travel','home infusion, no travel'])]
  fig, ax = plt.subplots(figsize=(11, 5)); labels = [b for _,_,b in CONFIG['BANDS']]; xp = np.arange(len(labels)); w = 0.38
  for j, (kind_, col_) in enumerate([('RETAIL', NAVY), ('INFUSION', RED)]):
      g = bh[bh.kind==kind_].set_index('band').reindex(labels).fillna(0)
      bars = ax.bar(xp + (j-0.5)*w, g.member_channel_moves, w, color=col_, label=f'{kind_.lower()} moves')
      for i, r in enumerate(g.itertuples()):
          if r.member_channel_moves > 0: ax.text(xp[i] + (j-0.5)*w, r.member_channel_moves, f"${r.saving/1000:,.0f}k\n${r.saving_per_move:,.0f}/move", ha='center', va='bottom', fontsize=7.5)
  dl = BANDS[(BANDS.design==HOLD) & (BANDS.band=='delivered, no travel')]
  ax.set_xticks(xp); ax.set_xticklabels([f"{b} mi" for b in labels]); ax.set_ylabel('members moved'); ax.set_ylim(0, max(1, bh.member_channel_moves.max())*1.4)
  hm_ = BANDS[(BANDS.design==HOLD) & (BANDS.band=='home infusion, no travel')]
  ax.legend(frameon=False); ax.set_title(f"Who moves and how much farther, {MODE} candidates. Design {HOLD}. Plus {int(dl.member_channel_moves.sum())} delivered specialty moves (${dl.saving.sum()/1e6:.2f}M) and {int(hm_.member_channel_moves.sum())} moves to home infusion (${hm_.saving.sum()/1e6:.2f}M), no travel", fontsize=9, fontweight='bold')
  plt.tight_layout(); plt.savefig(f'charts/{MODE}_02_bands.png', bbox_inches='tight'); plt.close('all')

  g2 = RUNS.sort_values('gross_saving').reset_index(drop=True)
  fig, ax = plt.subplots(figsize=(11, 0.42*len(g2)+1.5))
  ax.barh(g2.design, g2.gross_saving/1e6, color=GREEN, label='gross drug saving, $M')
  ax.barh(g2.design, g2.net_saving/1e6, color=NAVY, alpha=.55, label='net after network cost, $M')
  for i, r in g2.iterrows(): ax.text(max(r.gross_saving,r.net_saving)/1e6, i, f"  {r.members_moved:,} moved, {r.sent_farther:,} farther, {r.extra_miles_total:,} extra mi", va='center', fontsize=8, color=RED)
  ax.set_xlabel('$ million per year'); ax.legend(loc='lower right', frameon=False); ax.set_title(f'The ledger, {MODE} candidates. Money on the bar, people in the label', fontsize=11, fontweight='bold')
  plt.tight_layout(); plt.savefig(f'charts/{MODE}_03_ledger.png', bbox_inches='tight'); plt.close('all')

  mvh = member_rows[HOLD]; sel = set(p for p in SITES if site_rows[[s_['design'] for s_ in site_rows].index(HOLD)][p]==1)
  fig, ax = plt.subplots(figsize=(9, 8))
  for t_, col_, mk, sz in [('URBAN',LBLUE,'.',4),('SUBURBAN',GREY,'.',5),('RURAL',ORANGE,'.',6)]:
      g = members[members.MBR_URBAN_RURAL_CD==t_]; ax.scatter(g.MBR_LON, g.MBR_LAT, s=sz, c=col_, alpha=.35, linewidths=0, label=f'{t_.lower()} members')
  mvr = mvh[(mvh.moved==1) & mvh.kind.isin(TRAVEL)]
  for r in mvr.itertuples():
      ax.plot([me.loc[r.MBR_SK,'MBR_LON'], ph.loc[r.new_site,'PHR_LON']], [me.loc[r.MBR_SK,'MBR_LAT'], ph.loc[r.new_site,'PHR_LAT']], color=RED if r.extra_miles>2 else GREY, lw=.4, alpha=.5)
  ret = ph[(ph.PHR_TYPE_CD=='RETAIL') & ph.index.isin(SITES)]
  ax.scatter(ret[~ret.index.isin(sel)].PHR_LON, ret[~ret.index.isin(sel)].PHR_LAT, s=18, facecolors='white', edgecolors=GREY, marker='s', label='retail not in network')
  ax.scatter(ret[ret.index.isin(sel) & ret.index.isin(CURRENT_SITES)].PHR_LON, ret[ret.index.isin(sel) & ret.index.isin(CURRENT_SITES)].PHR_LAT, s=22, c=NAVY, marker='s', label='retail kept')
  ax.scatter(ret[ret.index.isin(sel) & ~ret.index.isin(CURRENT_SITES)].PHR_LON, ret[ret.index.isin(sel) & ~ret.index.isin(CURRENT_SITES)].PHR_LAT, s=26, c=GREEN, marker='s', label='retail added')
  inf_ = ph[ph.PHR_TYPE_CD=='INFUSION']; ax.scatter(inf_[inf_.index.isin(sel)].PHR_LON, inf_[inf_.index.isin(sel)].PHR_LAT, s=60, c=RED, marker='X', label='infusion site open')
  ax.set_aspect(1/np.cos(np.radians(CONFIG['CENTER'][0]))); ax.set_xticks([]); ax.set_yticks([])
  ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
  ax.set_title(f"The network, {MODE} candidates. Design {HOLD}. Lines are moved members, red if more than 2 miles farther", fontsize=10, fontweight='bold')
  plt.tight_layout(); plt.savefig(f'charts/{MODE}_04_network_map.png', bbox_inches='tight'); plt.close('all')

  fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
  tr_ = mvh[mvh.kind.isin(TRAVEL)]
  axes[0].hist(tr_.miles_today, bins=30, alpha=.6, color=GREY, label='today'); axes[0].hist(tr_.miles_new, bins=30, alpha=.6, color=NAVY, label='after')
  axes[0].set_xlabel('miles to pharmacy'); axes[0].set_ylabel('member kind pairs'); axes[0].legend(frameon=False); axes[0].set_title('Trip length, everyone, before and after', fontsize=10)
  if len(mvr): axes[1].hist(mvr.extra_miles, bins=20, color=RED, alpha=.85)
  axes[1].set_xlabel('extra miles versus today, moved members only'); axes[1].set_ylabel('members'); axes[1].set_title(f'How much farther the moved go ({len(mvr)} moved)', fontsize=10)
  plt.tight_layout(); plt.savefig(f'charts/{MODE}_05_distances.png', bbox_inches='tight'); plt.close('all')

  fig, ax = plt.subplots(figsize=(10, 4.5)); w=.27; xp=np.arange(len(RUNS))
  ax.bar(xp-w, RUNS.client_pays/1e6, w, color=NAVY, label='client pays'); ax.bar(xp, RUNS.pharmacy_receives/1e6, w, color=LBLUE, label='pharmacies receive')
  ax.bar(xp+w, RUNS.optum_spread/1e6, w, color=ORANGE, label='spread, modeled')
  ax.axhline(B0/1e6, ls='--', color=NAVY, lw=1); ax.text(len(RUNS)-.5, B0/1e6, ' client today', va='bottom', fontsize=8, color=NAVY)
  ax.set_xticks(xp); ax.set_xticklabels(RUNS.design, rotation=30, ha='right', fontsize=8); ax.set_ylabel('$ million per year'); ax.legend(frameon=False)
  ax.set_title(f'Who pays whom, per design, {MODE} candidates. Synthetic rates, illustrative', fontsize=11, fontweight='bold')
  plt.tight_layout(); plt.savefig(f'charts/{MODE}_06_perspectives.png', bbox_inches='tight'); plt.close('all')


  return dict(RUNS=RUNS, BANDS=BANDS, member_rows=member_rows, site_rows=site_rows, allot_rows=allot_rows, worst=pd.concat(worst_rows, ignore_index=True) if worst_rows else pd.DataFrame(), HOLD=HOLD, B0=B0, B0_PHARM=B0_PHARM,
              ADMIN_TODAY=ADMIN_TODAY, QF=QF, n_terr=n_terr, IDX=len(IDX), CURRENT=len(CURRENT_SITES), SITES=len(SITES))

EXPANSION = {}
RESULTS = {MODE: build_and_run(MODE) for MODE in CONFIG['CANDIDATE_MODES']}
RUNS = pd.concat([RESULTS[m]['RUNS'] for m in RESULTS], ignore_index=True)
BANDS = pd.concat([RESULTS[m]['BANDS'] for m in RESULTS], ignore_index=True)
site_rows = sum([RESULTS[m]['site_rows'] for m in RESULTS], [])
first = RESULTS[CONFIG['CANDIDATE_MODES'][0]]; HOLD = first['HOLD']; member_rows = first['member_rows']; allot_rows = first['allot_rows']
B0, B0_PHARM, ADMIN_TODAY, QF, n_terr = first['B0'], first['B0_PHARM'], first['ADMIN_TODAY'], first['QF'], first['n_terr']
print("\n=== VALUE OF ADDING PHARMACIES: same design, CURRENT against ALL candidates ===")
cmp = RUNS[RUNS.design==HOLD][['mode','gross_saving','net_saving','members_moved','sites_open','dropped','added']].copy()
cmp['territory_zips_met'] = [RESULTS[m]['n_terr'] for m in cmp['mode']]; cmp['zips_needing_expansion'] = [len(EXPANSION.get(m,[])) for m in cmp['mode']]
print(cmp.to_string(index=False))
if len(cmp)==2:
    print(f"adding candidates changes net saving by ${int(cmp.net_saving.iloc[1]-cmp.net_saving.iloc[0]):,} on this design, "
          f"while covering {int(cmp.territory_zips_met.iloc[1]-cmp.territory_zips_met.iloc[0])} more footprint ZIPs. "
          f"Read the two together: CURRENT is cheaper partly because it leaves ZIPs unmet.")
exp_df = pd.DataFrame([dict(mode=m, ZIP=z) for m in EXPANSION for z in EXPANSION[m]]).merge(fp[['ZIP','SETTING','MEMBERS_TODAY','ADEQ_MILES']], on='ZIP', how='left') if any(EXPANSION.values()) else pd.DataFrame(columns=['mode','ZIP','SETTING','MEMBERS_TODAY','ADEQ_MILES'])
# =============================================================================
# 8. EXCEL. Everything in one workbook.
# =============================================================================
stamp('writing Excel')
cfg = pd.DataFrame([(k, str(v)) for k,v in CONFIG.items()], columns=['setting','value'])
sites_df = pd.DataFrame(site_rows).assign(design=lambda d: d['mode']+' '+d['design']).drop(columns='mode').set_index('design').T.reset_index().rename(columns={'index':'PHR_PMT_CNTR_SK'})
sites_df = sites_df.merge(ph_df[['PHR_PMT_CNTR_SK','PHR_TYPE_CD','PHR_CHAIN_NM','PHR_STATE_CD','PHR_IN_CURRENT_NTWK','PHR_ADMIN_COST','CLT_RATE_GENERIC','CLT_RATE_BRAND']], on='PHR_PMT_CNTR_SK')
readme = pd.DataFrame({'sheet':['Config','Runs','Bands','Moved_members','Sites_by_design','Footprint','Data_summary','Retail_allotted','Expansion_required','Worst_moves'],
    'what':['every setting used for this run',
            'one row per design: A, λ, gross and net saving, members moved, sent farther, extra miles, trips before and after, sites open, kept, dropped, added, who pays whom, solver status',
            'per design, moved members by band of extra miles: count, saving, saving per member, copay incentive, net after incentive',
            f'every member and kind of place for the held design {HOLD}: today site, new site, moved, miles before and after, extra miles, visits, saving, band',
            'every candidate site: open or not in each design, with type, chain, state, in current network, admin cost, rates',
            'every footprint ZIP: setting, members today, adequacy distance applied and status: primary, fallback, gap',
            'baseline figures and data counts',
            f'members with no retail claims this year and the retail store allotted to them in design {HOLD}, with access distance before and after',
            'footprint ZIPs whose adequacy rule cannot be met with the candidate set of that mode. Under CURRENT, these ZIPs need network expansion',
            'the fifty members sent farthest in each design and mode: extra miles per trip and per year, visits, saving from their move']})
datasum = pd.DataFrame([dict(item='members', value=len(members)), dict(item='candidate sites, ALL mode', value=len(ph)), dict(item='sites in current network', value=len(CURRENT_SITES)),
                        dict(item='members with retail or mail fills', value=int(visits[visits.KIND=="RETAIL"].MBR_SK.nunique())), dict(item='members with retail store demand after mail held', value=len(set(k[0] for k in first['member_rows'][HOLD][['MBR_SK','kind']].itertuples(index=False) if k[1]=='RETAIL'))),
                        dict(item='claims', value=len(claims)), dict(item='baseline drug bill', value=round(B0)), dict(item='baseline admin', value=ADMIN_TODAY),
                        dict(item='pharmacies receive today', value=round(B0_PHARM+HELD_COST)), dict(item='quality floor applied', value=round(QF,2)),
                        dict(item='territory rules', value=n_terr), dict(item='assignment variables, first mode', value=first['IDX'])])
with pd.ExcelWriter('exports/results.xlsx', engine='openpyxl') as xw:
    readme.to_excel(xw, sheet_name='README', index=False); cfg.to_excel(xw, sheet_name='Config', index=False); RUNS.to_excel(xw, sheet_name='Runs', index=False)
    BANDS.to_excel(xw, sheet_name='Bands', index=False); member_rows[HOLD].to_excel(xw, sheet_name='Moved_members', index=False)
    sites_df.to_excel(xw, sheet_name='Sites_by_design', index=False); fp.to_excel(xw, sheet_name='Footprint', index=False); datasum.to_excel(xw, sheet_name='Data_summary', index=False)
    exp_df.to_excel(xw, sheet_name='Expansion_required', index=False)
    pd.concat([RESULTS[m_]['worst'] for m_ in RESULTS], ignore_index=True).to_excel(xw, sheet_name='Worst_moves', index=False)
    if len(allot_rows.get(HOLD, [])): allot_rows[HOLD].to_excel(xw, sheet_name='Retail_allotted', index=False)
RUNS.to_csv('exports/runs.csv', index=False); BANDS.to_csv('exports/bands.csv', index=False)
stamp(f"done. {len(RUNS)} designs across {len(RESULTS)} candidate modes. charts/ has 12 charts. exports/results.xlsx has everything.")
