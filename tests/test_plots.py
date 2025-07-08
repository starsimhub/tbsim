
import pytest
import numpy as np
from unittest import mock
import matplotlib
import sys
from tbsim.utils import plots
import tempfile
import os
import shutil
import sciris as sc
 
matplotlib.use('Agg')  # Use non-interactive backend for testing
# Patch sys.modules to mock sciris and starsim if not installed
modules_to_mock = {}
for mod in ['sciris', 'starsim']:
    if mod not in sys.modules:
        modules_to_mock[mod] = mock.MagicMock()
sys.modules.update(modules_to_mock)
 
class DummyResult:
    def __init__(self, timevec, values):
        self.timevec = np.array(timevec)
        self.values = np.array(values)
 
@pytest.fixture
def flat_results():
    return {
        'Scenario1': {
            'incidence': DummyResult([0, 1, 2], [0.1, 0.2, 0.3]),
            'mortality': DummyResult([0, 1, 2], [0.05, 0.07, 0.1]),
            'metric15': DummyResult([0, 1, 2], [1, 2, 3]),
        },
        'Scenario2': {
            'incidence': DummyResult([0, 1, 2], [0.08, 0.15, 0.25]),
            'mortality': DummyResult([0, 1, 2], [0.03, 0.05, 0.08]),
        }
    }
    
@mock.patch('tbsim.utils.plots.sc')
@mock.patch('tbsim.utils.plots.plt.show')
def test_plot_results_creates_valid_png(mock_show, mock_sc, tmp_path, flat_results):
    # This test checks that the saved file is a valid PNG image
    import PIL.Image
    import io

    # Patch sciris.now to return a fixed timestamp
    mock_sc.now.return_value = '20240101_120000'
    # Use a custom output directory
    outdir = tmp_path / "png_check"
    plots.plot_results(flat_results, outdir=str(outdir), savefig=True)
    # Check that the PNG file exists
    png_file = outdir / "scenarios_20240101_120000.png"
    assert png_file.exists()
    # Try to open the file as an image
    with open(png_file, "rb") as f:
        img_bytes = f.read()
        img = PIL.Image.open(io.BytesIO(img_bytes))
        assert img.format == "PNG"
    assert mock_show.called
            
@mock.patch('tbsim.utils.plots.sc')
@mock.patch('tbsim.utils.plots.plt.show')
def test_plot_results_custom_outdir(mock_show, mock_sc, flat_results, tmp_path):
    mock_sc.now.return_value = '20240101_120000'
    mock_sc.thisdir.return_value = str(tmp_path)
    outdir = tmp_path / "custom_results"
    # Ensure directory does not exist before
    if outdir.exists():
        shutil.rmtree(outdir)
    plots.plot_results(flat_results, outdir=str(outdir))
    # Check that the file was created in the custom outdir
    files = list(outdir.glob("scenarios_20240101_120000.png"))
    assert len(files) == 1
    assert files[0].is_file()
    assert mock_show.called

@mock.patch('tbsim.utils.plots.sc')
@mock.patch('tbsim.utils.plots.plt.show')
def test_plot_results_savefig_false(mock_show, mock_sc, flat_results, tmp_path):
    mock_sc.now.return_value = '20240101_120000'
    mock_sc.thisdir.return_value = str(tmp_path)
    outdir = tmp_path / "nosave_results"
    plots.plot_results(flat_results, outdir=str(outdir), savefig=False)
    # Directory should not be created since savefig is False
    assert not outdir.exists()
    assert mock_show.called

@mock.patch('tbsim.utils.plots.sc')
@mock.patch('tbsim.utils.plots.plt.show')
def test_plot_results_dark_theme(mock_show, mock_sc, flat_results, tmp_path):
    mock_sc.now.return_value = '20240101_120000'
    mock_sc.thisdir.return_value = str(tmp_path)
    plots.plot_results(flat_results, dark=True, savefig=False)
    assert mock_show.called

@mock.patch('tbsim.utils.plots.sc')
@mock.patch('tbsim.utils.plots.plt.show')
def test_plot_results_light_theme(mock_show, mock_sc, flat_results, tmp_path):
    mock_sc.now.return_value = '20240101_120000'
    mock_sc.thisdir.return_value = str(tmp_path)
    plots.plot_results(flat_results, dark=False, savefig=False)
    assert mock_show.called

@mock.patch('tbsim.utils.plots.sc')
@mock.patch('tbsim.utils.plots.plt.show')
def test_plot_results_multiple_metrics_and_scenarios(mock_show, mock_sc, flat_results, tmp_path):
    mock_sc.now.return_value = '20240101_120000'
    mock_sc.thisdir.return_value = str(tmp_path)
    # Should plot both 'incidence' and 'mortality' for both scenarios
    plots.plot_results(flat_results, n_cols=2, savefig=False)
    assert mock_show.called
    
@mock.patch('tbsim.utils.plots.sc')
@mock.patch('tbsim.utils.plots.plt.show')
def test_plot_results_keywords_and_exclude(mock_show, mock_sc, flat_results):
    mock_sc.now.return_value = '20240101_120000'
    mock_sc.thisdir.return_value = '.'
    # Only 'incidence' should be plotted, 'metric15' excluded by default
    plots.plot_results(flat_results, keywords=['incidence'], n_cols=1)
    assert mock_show.called
 
@mock.patch('tbsim.utils.plots.sc')
@mock.patch('tbsim.utils.plots.plt.show')
 
def test_plot_results_no_metrics(mock_show, mock_sc, flat_results, capsys):
    mock_sc.now.return_value = '20240101_120000'
    mock_sc.thisdir.return_value = '.'
    # Use a keyword that doesn't match any metric
    plots.plot_results(flat_results, keywords=['notfound'])
    captured = capsys.readouterr()
    assert "No metrics to plot." in captured.out
 
@mock.patch('tbsim.utils.plots.sc')
@mock.patch('tbsim.utils.plots.plt.show')
def test_plot_results_style_fallback(mock_show, mock_sc, flat_results, capsys):
    mock_sc.now.return_value = '20240101_120000'
    mock_sc.thisdir.return_value = '.'
    # Use a non-existent style to trigger fallback
    plots.plot_results(flat_results, style='nonexistent_style')
    captured = capsys.readouterr()
    assert "Warning: nonexistent_style style not found" in captured.out
 
# @mock.patch('tbsim.utils.plots.sc')
# @mock.patch('tbsim.utils.plots.plt.show')
# def test_plot_results_handles_empty(flat_results, mock_show, mock_sc, capsys):
@mock.patch('tbsim.utils.plots.plt.show')
@mock.patch('tbsim.utils.plots.sc')
def test_plot_results_handles_empty(mock_sc, mock_show, flat_results, capsys):
    mock_sc.now.return_value = '20240101_120000'
    mock_sc.thisdir.return_value = '.'
    # Empty input
    plots.plot_results({})
    captured = capsys.readouterr()
    assert "No metrics to plot." in captured.out
 
 