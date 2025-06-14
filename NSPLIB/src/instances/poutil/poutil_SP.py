# --------------- THIS SCRIPT WAS AUTO-GENERATED FROM GAMS2PYOMO ---------------
# ------------------------- FILE SOURCE: 'poutil.gms' --------------------------

from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Binary, Constraint, Objective, NonNegativeIntegers, Set, Param
from ...main import StoModelBuilder
import numpy as np

np.random.seed(0)


def const_model():
	pm = ConcreteModel(name='Load following Contract (LFC)')

	"""We discuss a portfolio optimization problem occurring in the energy
	market. Energy distributing public services have to decide how much
	of the requested energy demand has to be produced in their own power
	plant, and which complementary amount has to be bought from the spot
	market and from load following contracts.

	This problem is formulated as a mixed-integer linear programming
	problem and implemented in GAMS. The formulation is applied to real data
	of a German electricity distributor.

	Most equations contain the reference number of the formula in the
	publication.


	Rebennack, S, Kallrath, J, and Pardalos, P M, Energy Portfolio
	Optimization for Electric Utilities: Case Study for Germany. In
	Bj�rndal, E, Bj�rndal, M, Pardalos, P.M. and R�nnqvist, M Eds,.
	Springer, pp. 221-246, 2010.

	Keywords: mixed integer linear programming, energy economics, portfolio optimization,
			unit commitment, economic dispatch, power plant control, day-ahead market
	"""

	pm.S = Set(initialize=[0, 1, 2])
	pm.T = Set(initialize=['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40', 't41', 't42', 't43', 't44', 't45', 't46', 't47', 't48', 't49', 't50', 't51', 't52', 't53', 't54', 't55', 't56', 't57', 't58', 't59', 't60', 't61', 't62', 't63', 't64', 't65', 't66', 't67', 't68', 't69', 't70', 't71', 't72', 't73', 't74', 't75', 't76', 't77', 't78', 't79', 't80', 't81', 't82', 't83', 't84', 't85', 't86', 't87', 't88', 't89', 't90', 't91', 't92', 't93', 't94', 't95', 't96'], ordered=True, doc='time slices (quarter-hour)')
	pm.M = Set(initialize=['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8'], ordered=True, doc='stage of the power plant')
	pm.IS = Set(initialize=['iS0', 'iS1', 'iS2', 'iS3', 'iS4', 'iS5', 'iS6', 'iS7', 'iS8'], ordered=True, doc='interval for constant PP operation')
	pm.II = Set(initialize=['iI0', 'iI1', 'iI2', 'iI3', 'iI4', 'iI5', 'iI6', 'iI7', 'iI8', 'iI9', 'iI10', 'iI11', 'iI12', 'iI13', 'iI14', 'iI15', 'iI16'], ordered=True, doc='length of idle time period')
	pm.B = Set(initialize=['b1', 'b2', 'b3'], ordered=True, doc='support points of the zone prices')

	p1 = 20 * np.random.uniform(0, 1)
	p2 = 20 * np.random.uniform(-0.5, 0.5)
	p3 = 20 * np.random.uniform(-1, 0)
	pm.pSM_perturb = Param(pm.S, initialize={0: p1, 1: p2, 2: p3})
	pm.prob = Param(pm.S, initialize={0: 1 / 3, 1: 1 / 3, 2: 1 / 3})

	pm.cPPVar = Param(mutable=False, initialize=25, doc='variable cost of power plant [euro / MWh]')
	pm.pPPMax = Param(mutable=False, initialize=300, doc='maximal capacity of power plant      [MW]')
	pm.cBL = Param(mutable=False, initialize=32, doc='cost for one base load contract [euro / MWh]')
	pm.cPL = Param(mutable=False, initialize=41, doc='cost for one peak load contract [euro / MWh]')
	pm.IPL0 = Param(pm.T, mutable=True, doc='indicator function for peak load contracts')
	for t in pm.T:
		if (list(pm.T).index(t) + 1 >= 33) and (list(pm.T).index(t) + 1 <= 80):
			pm.IPL0[t] = 1
		else:
			pm.IPL0[t] = 0
	pm.IPL = Param(pm.T, mutable=False, doc='indicator function for peak load contracts', initialize=pm.IPL0)
	pm.pLFCref = Param(mutable=False, initialize=400, doc='power reference level for the LFC')

	pm.PowerForecast = Param(pm.T, mutable=False, initialize={'t1': 287, 't2': 275, 't3': 262, 't4': 250, 't5': 255, 't6': 260, 't7': 265, 't8': 270, 't9': 267, 't10': 265, 't11': 262, 't12': 260, 't13': 262, 't14': 265, 't15': 267, 't16': 270, 't17': 277, 't18': 285, 't19': 292, 't20': 300, 't21': 310, 't22': 320, 't23': 330, 't24': 340, 't25': 357, 't26': 375, 't27': 392, 't28': 410, 't29': 405, 't30': 400, 't31': 395, 't32': 390, 't33': 400, 't34': 410, 't35': 420, 't36': 430, 't37': 428, 't38': 427, 't39': 426, 't40': 425, 't41': 432, 't42': 440, 't43': 447, 't44': 455, 't45': 458, 't46': 462, 't47': 466, 't48': 470, 't49': 466, 't50': 462, 't51': 458, 't52': 455, 't53': 446, 't54': 437, 't55': 428, 't56': 420, 't57': 416, 't58': 412, 't59': 408, 't60': 405, 't61': 396, 't62': 387, 't63': 378, 't64': 370, 't65': 375, 't66': 380, 't67': 385, 't68': 390, 't69': 383, 't70': 377, 't71': 371, 't72': 365, 't73': 368, 't74': 372, 't75': 376, 't76': 380, 't77': 386, 't78': 392, 't79': 398, 't80': 405, 't81': 408, 't82': 412, 't83': 416, 't84': 420, 't85': 413, 't86': 407, 't87': 401, 't88': 395, 't89': 386, 't90': 377, 't91': 368, 't92': 360, 't93': 345, 't94': 330, 't95': 315, 't96': 300}, doc='electric power forecast')
	pm.eLFCbY = Param(pm.B, mutable=False, initialize={'b1': 54750, 'b2': 182500, 'b3': 9000000}, doc='amount of energy at support point b')
	pm.cLFCvar = Param(pm.B, mutable=False, initialize={'b1': 80, 'b2': 65, 'b3': 52}, doc='specific energy price in segment b')
	pm.eLFCb0 = Param(pm.B, mutable=True, doc='daily border of energy volumes for LFC')
	pm.cLFCs0 = Param(pm.B, mutable=True, doc='accumulated cost for LFC up to segment b')

	#  calculate the daily borders of the energy volumes for the zones
	for b in pm.B:
		pm.eLFCb0[b] = pm.eLFCbY[b] / 365

	#  calculate the accumulated cost
	pm.cLFCs0['b1'] = 0
	pm.cLFCs0['b2'] = pm.cLFCvar['b1'] * pm.eLFCb0['b1']
	for b in pm.B:
		if list(pm.B).index(b) + 1 > 2:
			pm.cLFCs0[b] = pm.cLFCs0[pm.B.prev(b, 1)] + pm.cLFCvar[pm.B.prev(b, 1)] * (pm.eLFCb0[pm.B.prev(b, 1)] - pm.eLFCb0[pm.B.prev(b, 2)])
	pm.eLFCb = Param(pm.B, mutable=False, doc='daily border of energy volumes for LFC', initialize=pm.eLFCb0)
	pm.cLFCs = Param(pm.B, mutable=False, doc='accumulated cost for LFC up to segment b', initialize=pm.cLFCs0)

	# first-stage variables
	pm.alpha = Var(doc='quantity of base load contracts')
	pm.beta = Var(doc='quantity of peak load contracts')

	# second-stage variables
	pm.c = Var(pm.S, doc='total cost')
	pm.cSM = Var(pm.S, doc='cost of energy from SM')
	pm.cLFC = Var(pm.S, doc='cost of LFC which is the enery rate')
	pm.eLFCtot = Var(pm.S, doc='total energy amount of LFC')
	pm.cPP = Var(pm.S, doc='cost of PP usage')
	pm.pPP = Var(pm.T, pm.S, doc='power withdrawn from power plant')
	pm.delta = Var(pm.M, pm.T, pm.S, doc='indicate if the PP is in stage m at time t')
	pm.chiS = Var(pm.T, pm.S, doc='indicate if there is a PP stage change')
	pm.chiI = Var(pm.T, pm.S, doc='indicate if the PP left the idle stage')
	pm.pSM = Var(pm.T, pm.S, doc='power from the spot market')
	pm.eLFCs = Var(pm.B, pm.S, doc='energy from LFC in segment b')
	pm.pLFC = Var(pm.T, pm.S, doc='power from the LFC')
	pm.mu = Var(pm.B, pm.S, doc='indicator for segment b (for zone prices)')
	pm.cPP.domain = NonNegativeReals
	pm.pPP.domain = NonNegativeReals
	pm.chiS.domain = NonNegativeReals
	pm.chiI.domain = NonNegativeReals
	pm.cSM.domain = NonNegativeReals
	pm.pSM.domain = NonNegativeReals
	pm.cLFC.domain = NonNegativeReals
	pm.eLFCtot.domain = NonNegativeReals
	pm.eLFCs.domain = NonNegativeReals
	pm.pLFC.domain = NonNegativeReals
	pm.delta.domain = Binary
	pm.mu.domain = Binary
	# m.alpha.domain = NonNegativeIntegers
	# m.beta.domain = NonNegativeIntegers
	pm.alpha.domain = NonNegativeReals
	pm.beta.domain = NonNegativeReals

	pm.alpha.setub(max(pm.PowerForecast[t] for t in pm.T))
	pm.beta.setub(pm.alpha.ub)
	for t in pm.T:
		for s in pm.S:
			pm.pLFC[t, s].setub(pm.pLFCref)

	# TODO: write new obj

	#  the objective function: total cost; eq. (6)
	def total_cost(m, s):
		return m.c[s] == m.cPP[s] + m.cSM[s] + m.cLFC[s]
	pm.obj = Constraint(pm.S, rule=total_cost)

	#  (fix cost +) variable cost * energy amount produced; eq. (7) & (8)
	def PPcost(m, s):
		return m.cPP[s] == m.cPPVar * sum(0.25 * m.pPP[t, s] for t in m.T)
	pm.PPcost = Constraint(pm.S, rule=PPcost)

	#  meet the power demand for each time period exactly; eq. (23)
	def demand(m, t, s):
		return m.pPP[t, s] + m.pSM[t, s] + m.pLFC[t, s] == m.PowerForecast[t]
	pm.demand = Constraint(pm.T, pm.S, rule=demand)
	#  power produced by the power plant; eq. (26)
	def PPpower(m, t, s):
		return m.pPP[t, s] == m.pPPMax * sum(0.1 * (list(m.M).index(_m) + 1 + 2) * m.delta[_m, t, s] for _m in m.M if (list(m.M).index(_m) + 1 > 1))
	pm.PPpower = Constraint(pm.T, pm.S, rule=PPpower)
	#  the power plant is in exactly one stage at any time; eq. (25)
	def PPstage(m, t, s):
		return sum(m.delta[_m, t, s] for _m in m.M) == 1
	pm.PPstage = Constraint(pm.T, pm.S, rule=PPstage)
	#  next constraints model the minimum time period a power plant is in the
	#  same state and the constraint of the minimum idle time
	#  we need variable 'chiS' to find out when a status change takes place
	#  eq. (27)
	def PPchiS1(m, t, _m, s):
		if list(m.T).index(t) + 1 > 1:
			return m.chiS[t, s] >= m.delta[_m, t, s] - m.delta[_m, m.T.prev(t, 1), s]
		else:
			return Constraint.Skip
	pm.PPchiS1 = Constraint(list(pm.T)[1:], pm.M, pm.S, rule=PPchiS1)
	#  second constraint for 'chiS' variable; eq. (28)
	def PPchiS2(m, t, _m, s):
		if (list(m.T).index(t) + 1 > 1):
			return m.chiS[t, s] >= (m.delta[_m, m.T.prev(t, 1), s] - m.delta[_m, t, s])
		else:
			return Constraint.Skip
	pm.PPchiS2 = Constraint(list(pm.T)[1:], pm.M, pm.S, rule=PPchiS2)
	#  control the minimum change time period; eq. (29)
	# PPstageChange(t)$(ord(t) < card(t) - card(iS) + 2).. sum(iS, chiS(t + ord(iS))) =l= 1;
	def PPstageChange(m, t, s):
		if t == 't88':
			return sum(m.chiS[m.T.next(t, list(m.IS).index(_is) + 1), s] for _is in list(m.IS)[:-1]) <= 1
		elif list(m.T).index(t) + 1 < len(list(m.T)) - len(list(m.IS)) + 2:
			return sum(m.chiS[m.T.next(t, list(m.IS).index(_is) + 1), s] for _is in m.IS) <= 1
		else:
			return Constraint.Skip
	pm.PPstageChange = Constraint(pm.T, pm.S, rule=PPstageChange)
	#  indicate if the plant left the idle state; eq. (30)
	def PPstarted(m, t, s):
		return m.chiI[t, s] >= (m.delta['m1', m.T.prev(t, 1), s] - m.delta['m1', t, s])
	pm.PPstarted = Constraint(list(pm.T)[1:], pm.S, rule=PPstarted)
	def PPstarted_init(m, s):
		return m.chiI['t1', s] >= - m.delta['m1', 't1', s]
	pm.PPstarted_init = Constraint(pm.S, rule=PPstarted_init)
	#  control the minimum idle time period:
	#  it has to be at least Nk2 time periods long; eq. (31)
	def PPidleTime(m, t, s):
		if t == 't80':
			return sum(m.chiI[m.T.next(t, list(m.II).index(ii) + 1), s] for ii in list(m.II)[:-1]) <= 1
		elif list(m.T).index(t) + 1 < len(list(m.T)) - len(list(m.II)) + 2:
			return sum(m.chiI[m.T.next(t, list(m.II).index(ii) + 1), s] for ii in m.II) <= 1
		else:
			return Constraint.Skip
	pm.PPidleTime = Constraint(pm.T, pm.S, rule=PPidleTime)
	#  cost for the spot market; eq. (12)
	#  consistent of the base load (alpha) and peak load (beta) contracts
	def SMcost(m, s):
		return m.cSM[s] == (24 * m.cBL) * m.alpha + (12 * m.cPL) * m.beta
	pm.SMcost = Constraint(pm.S, rule=SMcost)
	#  Spot Market power contribution; eq. (9)
	def SMpower(m, t, s):
		return m.pSM[t, s] == m.alpha + m.IPL[t] * m.beta + m.pSM_perturb[s]
	pm.SMpower = Constraint(pm.T, pm.S, rule=SMpower)
	#  cost of the LFC is given by the energy rate; eq. (14) & (21)
	def LFCcost(m, s):
		return m.cLFC[s] == sum((m.cLFCs[b] * m.mu[b, s] + m.cLFCvar[b] * m.eLFCs[b, s]) for b in m.B)
	pm.LFCcost = Constraint(pm.S, rule=LFCcost)
	#  total energy from the LFC; eq. (16)
	#  connect the eLFC(t) variables with eLFCtot
	def LFCenergy(m, s):
		return m.eLFCtot[s] == sum((0.25 * m.pLFC[t, s]) for t in m.T)
	pm.LFCenergy = Constraint(pm.S, rule=LFCenergy)
	#  indicator variable 'mu':
	#  we are in exactly one price segment b; eq. (18)
	def LFCmu(m, s):
		return sum(m.mu[b, s] for b in m.B) == 1
	pm.LFCmu = Constraint(pm.S, rule=LFCmu)
	#  connect the 'mu' variables with the total energy amount; eq. (19)
	def LFCenergyS(m, s):
		return m.eLFCtot[s] == sum(m.eLFCb[m.B.prev(b, 1)] * m.mu[b, s] for b in m.B if list(m.B).index(b) + 1 > 1) + sum(m.eLFCs[b, s] for b in m.B)
	pm.LFCenergyS = Constraint(pm.S, rule=LFCenergyS)
	#  accumulated energy amount for segment "b1"; eq. (20)
	def LFCemuo(m, s):
		return m.eLFCs['b1', s] <= (m.eLFCb['b1'] * m.mu['b1', s])
	pm.LFCemuo = Constraint(pm.S, rule=LFCemuo)
	#  accumulated energy amount for all other segments (then "b1"); eq. (20)
	def LFCemug(m, b, s):
		if list(m.B).index(b) + 1 > 1:
			return m.eLFCs[b, s] <= (m.eLFCb[b] - m.eLFCb[m.B.prev(b, 1)]) * m.mu[b, s]
		else:
			return Constraint.Skip
	pm.LFCemug = Constraint(list(pm.B)[1:], pm.S, rule=LFCemug)

	#  relative termination criterion for MIP (relative gap)
	pm._obj_ = Objective(rule=sum(pm.prob[s] * pm.c[s] for s in pm.S), sense=1)


	# model transformation -----------------------------------------------------

	builder = StoModelBuilder('pyomo', name='poutil', m_type='MILP', hint=False)

	scenarios = list(pm.S)
	var1_names = ['alpha', 'beta']
	con1_names = []

	def _obj(m, s):
		return m.prob[s] * m.c[s]

	objs = {s: _obj for s in scenarios}

	_content = {
		'pm': pm,
		'y_set': var1_names,
		'scenarios': scenarios,
		'con_1': con1_names,
		'objs': objs,
		'obj_sense': 1
	}

	sto_m = builder.build(**_content)

	return sto_m
