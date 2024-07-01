import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from plots import plot_epi, plot_hh, plot_nut, plot_active_infections

class TestPlots(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe for testing
        self.df_epi = pd.DataFrame({
            'year': [1942, 1943, 1944],
            'rand_seed': [0, 0, 0],
            'arm': ['A', 'A', 'A'],
            'channel1': [10, 15, 20],
            'channel2': [5, 10, 15]
        })

        self.df_hh = pd.DataFrame({
            'HH Size': [1, 2, 3],
            'rand_seed': [0, 0, 0],
            '1942': [5, 10, 15],
            '1943': [3, 6, 9]
        }).set_index('HH Size')
        
        self.df_active_infections = pd.DataFrame({
            'arm': ['VITAMIN', 'VITAMIN', 'CONTROL', 'CONTROL'],
            'rand_seed': [0, 0, 0, 0],
            'year': [1942, 1943, 1942, 1943],
            'cum_active_infections': [10, 15, 20, 25]
        })

    @patch('plots.sc.savefig')
    def test_plot_epi(self, mock_savefig):
        plot_epi(self.df_epi)
        mock_savefig.assert_called_once()
        plt.show()
        plt.close('all')  

    @patch('plots.sc.savefig')
    def test_plot_hh(self, mock_savefig):
        plot_hh(self.df_hh)
        mock_savefig.assert_called_once()
        plt.show()
        plt.close('all')  

    @patch('plots.sc.savefig')
    def test_plot_nut(self, mock_savefig):
        data={
                'Arm': ['CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN',
                        'VITAMIN', 'VITAMIN', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'VITAMIN', 'VITAMIN'],
                'Macro': ['MARGINAL', 'MARGINAL', 'SLIGHTLY_BELOW_STANDARD', 'SLIGHTLY_BELOW_STANDARD', 'STANDARD_OR_ABOVE', 
                        'UNSATISFACTORY', 'MARGINAL', 'MARGINAL', 'SLIGHTLY_BELOW_STANDARD', 'SLIGHTLY_BELOW_STANDARD', 
                        'STANDARD_OR_ABOVE', 'UNSATISFACTORY', 'STANDARD_OR_ABOVE', 'MARGINAL', 'MARGINAL', 'SLIGHTLY_BELOW_STANDARD', 
                        'SLIGHTLY_BELOW_STANDARD', 'STANDARD_OR_ABOVE', 'MARGINAL', 'MARGINAL'],
                'Micro': ['DEFICIENT', 'NORMAL', 'DEFICIENT', 'NORMAL', 'NORMAL', 'DEFICIENT', 'DEFICIENT', 'NORMAL', 'DEFICIENT', 'NORMAL',
                        'NORMAL', 'DEFICIENT', 'DEFICIENT', 'DEFICIENT', 'NORMAL', 'DEFICIENT', 'NORMAL', 'NORMAL', 'DEFICIENT', 'NORMAL'],
                '1942': [76.0, 35.0, 33.0, 127.0, 98.0, 58.0, 48.0, 23.0, 29.0, 99.0, 83.0, 76.0, np.nan, 96.0, 35.0, 26.0, 62.0, 70.0, 29.0, 84.0],
                '1944': [36.0, 4.0, 46.0, 33.0, 210.0, 12.0, np.nan, 35.0, np.nan, 70.0, 213.0, 21.0, 69.0, 25.0, 4.0, 43.0, 25.0, 131.0, 6.0, 1.0],
                'rand_seed': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
            }   
        df = pd.DataFrame(data)
        plot_nut(df)
        mock_savefig.assert_called_once()
        plt.show()
        plt.close('all')  

    @patch('plots.sc.savefig')
    def test_plot_active_infections(self, mock_savefig):
        plot_active_infections(self.df_active_infections)
        mock_savefig.assert_called_once()
        plt.show()
        plt.close('all')  
if __name__ == '__main__':
    unittest.main()
