from unittest import TestCase
from neuronal.io import get_example_data_file_path, NeuronalData
from neuronal.model import *
from neuronal.plot import *
import matplotlib
import pymc3 as pm


class TestPlot(TestCase):
    def test_plot(self):
        example_file_path = get_example_data_file_path('single_PSP_data.txt')
        data = NeuronalData(example_file_path)
        initial_guess = {
            'b_start': -30.39,
            'b_end': -30.39,
            'b': [-30.39],
            'a': [0.3],
            't_psp': [451.43],
            'tau_d': [0.01],
            'tau_r': [0.001]
        }
        sample = psp_fit(data, 5, initial_guess, seed=42, tune=10, suppress_warnings=True)
        summary = pm.summary(sample)
        model = psp_model_for_plot(data, summary)
        ax = plot_fit(data, summary, show_plot=False)
        self.assertTrue(isinstance(ax, matplotlib.axes._base._AxesBase))
        self.assertTrue(len(model) == 942)
        self.assertAlmostEqual(model[0], -30.411777684553478)
        self.assertAlmostEqual(model[-1], -30.407399100212388)