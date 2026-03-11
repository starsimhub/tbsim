"""Tests for tbsim.Sim and tbsim.demo."""

import sciris as sc
import tbsim
sc.options(interactive=False)


def test_sim():
    sim = tbsim.Sim(n_agents=200, verbose=0)
    sim.run()
    tb = sim.get_tb()
    assert isinstance(tb, tbsim.TB_LSHTM)
    assert len(tb.results['timevec']) > 0


def test_demo():
    sim = tbsim.demo(n_agents=200, verbose=0, plot=False)
    assert isinstance(sim, tbsim.Sim)


if __name__ == '__main__':
    test_sim()
    test_demo()
    print('All tests passed!')
