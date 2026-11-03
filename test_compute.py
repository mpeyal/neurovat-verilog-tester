"""Analytic compute checks independent of the GUI and training engine."""
import threading
import unittest

import numpy as np

from simulation_core.model_v2 import EcfetV2
from vatester.compute import Settings, bounds, initialize, logic, matrix, parse_matrix, terminal_energy


class ComputeTests(unittest.TestCase):
    def test_signed_matrix_and_physical_energy(self):
        cfg = Settings(levels=5, read_voltage=.2, read_ns=10, peripheral_pj=2)
        a = np.array([[1, -.5], [.25, 0]])
        x = np.array([[1, -1, 0], [.5, 1, 0]])
        r = matrix(EcfetV2, a, x, cfg)
        np.testing.assert_allclose(r['actual'], a @ x, atol=1e-12)
        gp, gm = np.array(r['g_plus_S']), np.array(r['g_minus_S'])
        expected = sum((.2*x[i, k])**2*(gp[j, i]+gm[j, i])*10e-9
                       for j in range(2) for i in range(2) for k in range(3))
        self.assertAlmostEqual(r['array_energy_J']/expected, 1)
        self.assertAlmostEqual(r['accounted_energy_J']/(expected+6e-12), 1)

    def test_banks_schedule_without_changing_answers_or_energy(self):
        a, x = [[1, -1]], [[1, 0, -1, .5, 1], [0, 1, 0, .5, 1]]
        serial = matrix(EcfetV2, a, x)
        parallel = matrix(EcfetV2, a, x, Settings(banks=3))
        self.assertEqual(serial['actual'], parallel['actual'])
        self.assertEqual(serial['array_energy_J'], parallel['array_energy_J'])
        self.assertEqual(parallel['read_cycles'], 2)
        self.assertEqual(parallel['physical_devices'], 12)
        self.assertAlmostEqual(serial['estimated_latency_s']/parallel['estimated_latency_s'], 2.5)

    def test_logic_exhaustive_and_failure_detection(self):
        for op in ('NAND', 'AND', 'OR', 'XOR', 'NOT'):
            r = logic(EcfetV2, op)
            self.assertEqual(r['passed'], r['cases'], op)
        for bits in range(1, 5):
            r = logic(EcfetV2, 'ADD', bits)
            self.assertEqual(r['cases'], 2**(2*bits+1))
            self.assertEqual(r['passed'], r['cases'])
            self.assertEqual(r['gate_count'], 9*bits)
        noisy = logic(EcfetV2, 'ADD', 4, Settings(mismatch=1, seed=17))
        self.assertLess(noisy['passed'], noisy['cases'])

    def test_quantization_and_reproducible_mismatch(self):
        r = matrix(EcfetV2, [[1, .3]], [[0], [1]], Settings(levels=3))
        self.assertAlmostEqual(r['actual'][0][0], .5)
        cfg = Settings(mismatch=.1)
        one = matrix(EcfetV2, [[1, -1]], [[1], [1]], cfg)
        self.assertEqual(one, matrix(EcfetV2, [[1, -1]], [[1], [1]], cfg))
        self.assertGreater(one['max_abs_error'], 0)

    def test_all_supported_twins_reachable(self):
        from simulation_core.model_v1 import EcfetV1
        from simulation_core.model_v3 import EcfetV3
        from simulation_core import FeFET          # None when the private model is absent
        for cls in (c for c in (EcfetV1, EcfetV2, EcfetV3, FeFET) if c is not None):
            low, high = bounds(cls)
            for g in (low, (low+high)/2, high):
                self.assertAlmostEqual(initialize(cls, g)/g, 1, places=6)
            r = matrix(cls, [[1, -1]], [[1], [0]])
            self.assertAlmostEqual(r['actual'][0][0], 1)

    def test_zero_and_invalid_inputs_and_cancel(self):
        r = matrix(EcfetV2, [[0]], [[0]])
        self.assertEqual(r['actual'], [[0.]])
        self.assertEqual(r['array_energy_J'], 0)
        self.assertIsNone(r['relative_l2_error'])
        for text in ('', '1,2\n3', 'nan', '1e12'):
            with self.assertRaises(ValueError):
                parse_matrix(text)
        with self.assertRaises(ValueError):
            matrix(EcfetV2, [[1]], [[1], [2]])
        with self.assertRaises(ValueError):
            matrix(EcfetV2, [[1]], [[1]], Settings(read_ns=-1))
        stop = threading.Event(); stop.set()
        with self.assertRaises(InterruptedError):
            matrix(EcfetV2, [[1]], [[1]], stop=stop)

    def test_multiterminal_energy_and_returned_energy(self):
        r = terminal_energy('t_s,v_a,i_a,v_b,i_b\n0,2,3,1,-1\n1,2,3,1,-1\n3,2,3,1,-1')
        self.assertEqual(r['net_energy_J'], 15)
        self.assertEqual(r['terminal_energy_J'], {'a': 18, 'b': -3})
        self.assertEqual(r['cumulative_energy_J'], [0, 5, 15])
        r = terminal_energy('t_s,v_1,i_1\n0,1,-2\n2,1,-2')
        self.assertEqual(r['net_energy_J'], -4)
        # Nonuniform trapezoids of a linearly increasing power.
        r = terminal_energy('t_s,v_1,i_1\n0,1,0\n1,1,1\n3,1,3')
        self.assertEqual(r['net_energy_J'], 4.5)

    def test_trace_rejects_bad_schema_and_time(self):
        for text in ('t_s,v_1\n0,1\n1,2', 't_s,v_1,i_1\n0,1,2\n0,1,2',
                     't_s,v_1,i_1\n0,1,2,3\n1,1,2', 't_s,v_1,i_1\n0,1\n1,1,2',
                     't_s,v_1,i_1\n0,1,2\n1,1,nan'):
            with self.assertRaises(ValueError):
                terminal_energy(text)


if __name__ == '__main__':
    unittest.main()
