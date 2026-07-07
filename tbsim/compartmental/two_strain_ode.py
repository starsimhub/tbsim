"""
Two-strain drug-resistance compartmental TB model.

This is the deterministic ODE validation reference for the resistance overlay:
the single-strain LSHTM spectrum-of-disease model extended into strain A
(treatment-susceptible), strain B (treatment-resistant), and mixed AB
infection. It includes superinfection, a transmission bottleneck, a
progression bottleneck, de-novo resistance, and a strain-resolved treatment
operator.
"""

import numpy as np
import sciris as sc
from scipy.integrate import odeint

__all__ = ['two_strain_defaults', 'TwoStrainODE']


STATES = [
    'SUS', 'CLE', 'REC', 'TRD',
    'L_A', 'L_B', 'L_AB',
    'N_A', 'N_B', 'N_AB',
    'AS_A', 'AS_B', 'AS_AB',
    'SY_A', 'SY_B', 'SY_AB',
    'TA_A', 'TA_B', 'TA_AB',
    'TY_A', 'TY_B', 'TY_AB',
    'DTH',
]
_IDX = {s: i for i, s in enumerate(STATES)}


def two_strain_defaults():
    """Return the default two-strain ODE parameters."""
    return sc.objdict(
        N=1e5, mu=1/70,
        beta=16.45, trans_asymp=0.82,
        inf_cle=1.90, inf_non=0.16, inf_asy=0.06,
        non_rec=0.18, non_asy=0.25,
        asy_non=1.66, asy_sym=0.88, sym_asy=0.54, sym_dead=0.34,
        rr_reinfection_cleared=1.00, rr_reinfection_rec=0.21, rr_reinfection_treat=3.15,
        fit_a=1.0, fit_b=0.575,
        rr_reinfection_inf=1.0, rr_reinfection_non=1.0, rr_reinfection_asy=0.0, rr_reinfection_sym=0.0,
        p_multi=1.0, rr_prog_super=1.0, rr_clear_super=1.0,
        r_treat_asym=0.01, r_treat_sym=1.204, delta=2.0, eff_a=0.75, eff_b=0.25,
        q_treat=0.05, q_prog=0.0001,
        transmission_independent=0, prog_resist_mix=1, prog_select_fitness=0,
    )


class TwoStrainODE(sc.prettyobj):
    """
    Two-strain compartmental TB model integrated with ``scipy.odeint``.

    Args:
        kwargs: Parameter overrides applied on top of
            :func:`two_strain_defaults`.

    Example::

        ode = TwoStrainODE(beta=16.45)
        df = ode.run(SY_A=1e3, SY_B=1e1)
    """

    def __init__(self, **kwargs):
        self.pars = two_strain_defaults()
        self.pars.update(kwargs)
        self.df = None
        return

    def init_conditions(self, **seeds):
        """Return initial compartment counts from state-name seed counts."""
        y = np.zeros(len(STATES))
        for key, val in seeds.items():
            y[_IDX[key]] = val
        dyn = sum(y) - y[_IDX['DTH']]
        y[_IDX['SUS']] = self.pars.N - dyn
        if y[_IDX['SUS']] < 0:
            raise ValueError('Seeds exceed N; SUS would be negative.')
        return y

    def derivs(self, y, t):
        """Return the right-hand side of the two-strain ODE system."""
        p = self.pars
        s = {name: y[i] for i, name in enumerate(STATES)}
        SUS, CLE, REC, TRD = s['SUS'], s['CLE'], s['REC'], s['TRD']
        L_A, L_B, L_AB = s['L_A'], s['L_B'], s['L_AB']
        N_A, N_B, N_AB = s['N_A'], s['N_B'], s['N_AB']
        AS_A, AS_B, AS_AB = s['AS_A'], s['AS_B'], s['AS_AB']
        SY_A, SY_B, SY_AB = s['SY_A'], s['SY_B'], s['SY_AB']
        TA_A, TA_B, TA_AB = s['TA_A'], s['TA_B'], s['TA_AB']
        TY_A, TY_B, TY_AB = s['TY_A'], s['TY_B'], s['TY_AB']

        g_A = p.fit_a / (p.fit_a + p.fit_b)
        g_B = p.fit_b / (p.fit_a + p.fit_b)
        r_max = max(p.fit_a, p.fit_b)

        P_A = p.trans_asymp * AS_A + SY_A
        P_B = p.trans_asymp * AS_B + SY_B
        P_AB = p.trans_asymp * AS_AB + SY_AB

        if p.transmission_independent == 1:
            lambda_A = (p.beta / p.N) * p.fit_a * (P_A + P_AB)
            lambda_B = (p.beta / p.N) * p.fit_b * (P_B + P_AB)
        else:
            lambda_A = (p.beta / p.N) * (p.fit_a * P_A + g_A * r_max * P_AB)
            lambda_B = (p.beta / p.N) * (p.fit_b * P_B + g_B * r_max * P_AB)
        Lambda = lambda_A + lambda_B

        U = SUS + p.rr_reinfection_cleared * CLE + p.rr_reinfection_rec * REC + p.rr_reinfection_treat * TRD
        sumY = SY_A + SY_B + SY_AB
        B_birth = p.mu * p.N + p.sym_dead * sumY

        Pi_AB = p.rr_prog_super * p.inf_asy * L_AB + p.rr_prog_super * p.non_asy * N_AB
        h_A, h_B = (g_A, g_B) if p.prog_select_fitness == 1 else (0.5, 0.5)
        G_AB = p.p_multi * Pi_AB
        G_A = (1 - p.p_multi) * h_A * Pi_AB
        G_B = (1 - p.p_multi) * h_B * Pi_AB

        mix = p.prog_resist_mix
        rep = 1 - mix

        ea, eb, q = p.eff_a, p.eff_b, p.q_treat
        pi_A_to_A = (1 - ea) * (1 - q)
        pi_A_to_B = (1 - ea) * q
        pi_B_to_B = (1 - eb)
        pi_AB_to_A = (1 - ea) * eb * (1 - q)
        pi_AB_to_B = ea * (1 - eb) + (1 - ea) * eb * q + (1 - ea) * (1 - eb) * q
        pi_AB_to_AB = (1 - ea) * (1 - eb) * (1 - q)
        d = p.delta
        Phi_A_a = d * (pi_A_to_A * TA_A + pi_AB_to_A * TA_AB)
        Phi_B_a = d * (pi_A_to_B * TA_A + pi_B_to_B * TA_B + pi_AB_to_B * TA_AB)
        Phi_AB_a = d * (pi_AB_to_AB * TA_AB)
        Phi_A_y = d * (pi_A_to_A * TY_A + pi_AB_to_A * TY_AB)
        Phi_B_y = d * (pi_A_to_B * TY_A + pi_B_to_B * TY_B + pi_AB_to_B * TY_AB)
        Phi_AB_y = d * (pi_AB_to_AB * TY_AB)
        cure_flow = d * (ea * (TA_A + TY_A) + eb * (TA_B + TY_B) + ea * eb * (TA_AB + TY_AB))

        rri, rrn, rra, rry = p.rr_reinfection_inf, p.rr_reinfection_non, p.rr_reinfection_asy, p.rr_reinfection_sym
        om, ps = p.rr_clear_super, p.rr_prog_super
        dd = {}
        dd['SUS'] = B_birth - Lambda * SUS - p.mu * SUS
        dd['CLE'] = p.inf_cle * (L_A + L_B) + om * p.inf_cle * L_AB - p.rr_reinfection_cleared * Lambda * CLE - p.mu * CLE
        dd['REC'] = p.non_rec * (N_A + N_B) + om * p.non_rec * N_AB - p.rr_reinfection_rec * Lambda * REC - p.mu * REC
        dd['TRD'] = cure_flow - p.rr_reinfection_treat * Lambda * TRD - p.mu * TRD

        dd['L_A'] = lambda_A * U - (p.inf_cle + p.inf_non + p.inf_asy + rri * lambda_B + p.mu) * L_A
        dd['L_B'] = lambda_B * U - (p.inf_cle + p.inf_non + p.inf_asy + rri * lambda_A + p.mu) * L_B
        dd['L_AB'] = rri * (lambda_B * L_A + lambda_A * L_B) - (om * p.inf_cle + p.inf_non + ps * p.inf_asy + p.mu) * L_AB

        dd['N_A'] = p.inf_non * (1 - p.q_prog) * L_A + p.asy_non * AS_A - (p.non_rec + p.non_asy + rrn * lambda_B + p.mu) * N_A
        dd['N_B'] = p.inf_non * L_B + rep * p.inf_non * p.q_prog * L_A + p.asy_non * AS_B - (p.non_rec + p.non_asy + rrn * lambda_A + p.mu) * N_B
        dd['N_AB'] = p.inf_non * L_AB + mix * p.inf_non * p.q_prog * L_A + p.asy_non * AS_AB + rrn * (lambda_B * N_A + lambda_A * N_B) - (om * p.non_rec + ps * p.non_asy + p.mu) * N_AB

        dd['AS_A'] = p.inf_asy * (1 - p.q_prog) * L_A + p.non_asy * N_A + p.sym_asy * SY_A + G_A + Phi_A_a - (p.asy_non + p.asy_sym + p.r_treat_asym + rra * lambda_B + p.mu) * AS_A
        dd['AS_B'] = p.inf_asy * L_B + rep * p.inf_asy * p.q_prog * L_A + p.non_asy * N_B + p.sym_asy * SY_B + G_B + Phi_B_a - (p.asy_non + p.asy_sym + p.r_treat_asym + rra * lambda_A + p.mu) * AS_B
        dd['AS_AB'] = p.sym_asy * SY_AB + G_AB + mix * p.inf_asy * p.q_prog * L_A + rra * (lambda_B * AS_A + lambda_A * AS_B) + Phi_AB_a - (p.asy_non + ps * p.asy_sym + p.r_treat_asym + p.mu) * AS_AB

        dd['SY_A'] = p.asy_sym * AS_A + Phi_A_y - (p.sym_asy + p.sym_dead + p.r_treat_sym + rry * lambda_B + p.mu) * SY_A
        dd['SY_B'] = p.asy_sym * AS_B + Phi_B_y - (p.sym_asy + p.sym_dead + p.r_treat_sym + rry * lambda_A + p.mu) * SY_B
        dd['SY_AB'] = ps * p.asy_sym * AS_AB + rry * (lambda_B * SY_A + lambda_A * SY_B) + Phi_AB_y - (p.sym_asy + p.sym_dead + p.r_treat_sym + p.mu) * SY_AB

        dd['TA_A'] = p.r_treat_asym * AS_A - (d + p.mu) * TA_A
        dd['TA_B'] = p.r_treat_asym * AS_B - (d + p.mu) * TA_B
        dd['TA_AB'] = p.r_treat_asym * AS_AB - (d + p.mu) * TA_AB
        dd['TY_A'] = p.r_treat_sym * SY_A - (d + p.mu) * TY_A
        dd['TY_B'] = p.r_treat_sym * SY_B - (d + p.mu) * TY_B
        dd['TY_AB'] = p.r_treat_sym * SY_AB - (d + p.mu) * TY_AB

        dynamic_sum = sum(y) - s['DTH']
        dd['DTH'] = p.mu * dynamic_sum + p.sym_dead * sumY
        return np.array([dd[name] for name in STATES])

    def run(self, start_time=1500, end_time=2020, by=1, **seeds):
        """Integrate and return compartments plus key observables."""
        if not seeds:
            seeds = dict(SY_A=1e3, SY_B=1e1)
        y0 = self.init_conditions(**seeds)
        t = np.arange(start_time, end_time + by, by)
        out = odeint(self.derivs, y0, t)
        df = sc.dataframe(data=out, columns=STATES)
        df['time'] = t
        active = df.AS_A + df.AS_B + df.AS_AB + df.SY_A + df.SY_B + df.SY_AB
        active_B = df.AS_B + df.SY_B + df.AS_AB + df.SY_AB
        active_AB = df.AS_AB + df.SY_AB
        df['prev_active'] = active / self.pars.N
        df['frac_resist'] = np.where(active > 0, active_B / active, 0.0)
        df['frac_super'] = np.where(active > 0, active_AB / active, 0.0)
        self.df = df
        return df

    def collapse(self):
        """Return strain-summed compartments for comparison with agent-level TB states."""
        d = self.df
        return sc.dataframe(
            time=d.time, SUSCEPTIBLE=d.SUS,
            INFECTION=d.L_A + d.L_B + d.L_AB, CLEARED=d.CLE, RECOVERED=d.REC,
            NON_INFECTIOUS=d.N_A + d.N_B + d.N_AB,
            ASYMPTOMATIC=d.AS_A + d.AS_B + d.AS_AB,
            SYMPTOMATIC=d.SY_A + d.SY_B + d.SY_AB,
            TREATMENT=d.TA_A + d.TA_B + d.TA_AB + d.TY_A + d.TY_B + d.TY_AB,
            TREATED=d.TRD,
        )
