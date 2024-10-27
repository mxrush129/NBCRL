from functools import reduce
from itertools import product
import sympy as sp
from SumOfSquares import SOSProblem
from utils.Config import CegisConfig
from RL.Env import Zones, Example


class SOS:
    def __init__(self, config: CegisConfig, poly_list):
        self.config = config
        self.ex: Example = config.example
        self.n = config.example.n_obs
        self.var_count = 0
        self.x = sp.symbols(['x{}'.format(i + 1) for i in range(self.n)])
        self.poly_list = poly_list

    def verify_positive(self, expr, con, deg=2):
        x = self.x
        prob = SOSProblem()
        for c in con:
            P, par, terms = self.polynomial(deg)
            prob.add_sos_constraint(P, x)
            expr = expr - c * P
        expr = sp.expand(expr)
        prob.add_sos_constraint(expr, x)
        try:
            prob.solve(solver='mosek')
            return True
        except:
            return False

    def verify_positive_multiplier(self, A, B, con, deg=2, R_deg=2):
        x = self.x
        prob = SOSProblem()
        expr = A
        for c in con:
            P, par, terms = self.polynomial(deg)
            prob.add_sos_constraint(P, x)
            expr = expr - c * P

        R, par, terms = self.polynomial(R_deg)
        expr = expr - R * B
        expr = sp.expand(expr)
        prob.add_sos_constraint(expr, x)
        try:
            prob.solve(solver='mosek')
            return True
        except:
            return False

    def verify_continuous(self):
        b1, bm1 = self.poly_list
        deg = self.config.DEG_continuous
        x = self.x
        state = [True] * 3
        ################################
        # init
        state[0] = self.verify_positive(b1, self.get_con(self.ex.I_zones), deg=deg[0])
        if not state[0]:
            print('The init condition is not satisfied.')
        ################################
        # Lie
        expr = sum([sp.diff(b1, x[i]) * self.ex.f[i](*x) for i in range(self.n)])
        # expr = expr - bm1 * b1
        # state[1] = self.verify_positive(expr, self.get_con(self.ex.l1), deg=deg[1])
        state[1] = self.verify_positive_multiplier(expr, b1, self.get_con(self.ex.D_zones), deg=deg[1], R_deg=deg[2])
        if not state[1]:
            print('The lie condition is not satisfied.')
        ################################
        # unsafe
        state[2] = self.verify_positive(-b1, self.get_con(self.ex.U_zones), deg=deg[3])
        if not state[2]:
            print('The unsafe condition is not satisfied.')

        result = True
        for e in state:
            result = result and e
        return result, state

    def get_con(self, zone: Zones):
        x = self.x
        if zone.shape == 'ball':
            poly = zone.r
            for i in range(self.n):
                poly = poly - (x[i] - zone.center[i]) ** 2
            return [poly]
        elif zone.shape == 'box':
            poly = []
            for i in range(self.n):
                poly.append((x[i] - zone.low[i]) * (zone.up[i] - x[i]))
            return poly

    def polynomial(self, deg=2):  # Generating polynomials of degree n-ary deg.
        if deg == 2:
            parameters = []
            terms = []
            poly = 0
            parameters.append(sp.symbols('parameter' + str(self.var_count)))
            self.var_count += 1
            poly += parameters[-1]
            terms.append(1)
            for i in range(self.n):
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(self.x[i])
                poly += parameters[-1] * terms[-1]
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(self.x[i] ** 2)
                poly += parameters[-1] * terms[-1]
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    parameters.append(sp.symbols('parameter' + str(self.var_count)))
                    self.var_count += 1
                    terms.append(self.x[i] * self.x[j])
                    poly += parameters[-1] * terms[-1]
            return poly, parameters, terms
        else:
            parameters = []
            terms = []
            exponents = list(product(range(deg + 1), repeat=self.n))  # Generate all possible combinations of indices.
            exponents = [e for e in exponents if sum(e) <= deg]  # Remove items with a count greater than deg.
            poly = 0
            for e in exponents:  # Generate all items.
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(reduce(lambda a, b: a * b, [self.x[i] ** exp for i, exp in enumerate(e)]))
                poly += parameters[-1] * terms[-1]
            return poly, parameters, terms


if __name__ == '__main__':
    from benchmarks.Examples import get_example_by_name

    ex = get_example_by_name('H3')
    sos = SOS(CegisConfig(**{'example': ex}), [])
    print(sos.get_con(ex.l1))
