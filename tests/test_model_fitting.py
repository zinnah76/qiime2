import unittest
import os
import tempfile
import numpy as np
import pandas as pd

# Use a non-interactive backend for plotting in CI/environments without a display
import matplotlib
matplotlib.use("Agg")

import model_fitting

class TestModelFitting(unittest.TestCase):
    def test_cone_model_zero_time(self):
        # Verify that cone_model handles t=0 safely without division by zero

        # scalar check (accept scalar or 0-D numpy return)
        val = model_fitting.cone_model(0, 100.0, 0.1, 2.0)
        val_arr = np.asarray(val)
        # allow either scalar 0.0 or array-like with first element 0.0
        self.assertEqual(float(val_arr.ravel()[0]), 0.0)

        # array check including 0
        t_arr = np.array([0.0, 1.0, 2.0])
        out = model_fitting.cone_model(t_arr, 100.0, 0.1, 2.0)
        out_arr = np.asarray(out)
        # first element should be 0.0, and all values non-negative
        self.assertEqual(float(out_arr.ravel()[0]), 0.0)
        self.assertTrue(np.all(out_arr >= 0.0))

    def test_gompertz_zero_p(self):
        # Verify that gompertz handles P=0 safely
        t_arr = np.array([0.0, 1.0, 2.0])
        out = model_fitting.gompertz(t_arr, 0.0, 5.0, 2.0)
        out_arr = np.asarray(out)
        # Expect all zeros; use exact equality since zeros are exact
        np.testing.assert_array_equal(out_arr, np.zeros_like(t_arr))

    def test_logistic_zero_p(self):
        # Verify that logistic handles P=0 safely
        t_arr = np.array([0.0, 1.0, 2.0])
        out = model_fitting.logistic(t_arr, 0.0, 5.0, 2.0)
        out_arr = np.asarray(out)
        np.testing.assert_array_equal(out_arr, np.zeros_like(t_arr))

    def test_plot_model_fitting_runs_and_saves_figure(self):
        # Generate some mock data with 3 treatments to test dynamic axes handling
        treatments = ["Treatment_A", "Treatment_B", "Treatment_C"]
        time_pts = np.linspace(0, 47, 10)
        df_data = {"Time": time_pts}
        for tr in treatments:
            df_data[tr] = time_pts * 5.0

        df = pd.DataFrame(df_data)

        # Parameter dataframes
        gompertz_rows = []
        first_rows = []
        logistic_rows = []
        cone_rows = []
        for tr in treatments:
            gompertz_rows.append({"Treatment": tr, "P (mL)": 200.0, "Rm (mL/day)": 10.0, "Lag (day)": 2.0})
            first_rows.append({"Treatment": tr, "P (mL)": 200.0, "k (day⁻¹)": 0.1})
            logistic_rows.append({"Treatment": tr, "P (mL)": 200.0, "Rm (mL/day)": 10.0, "Lag (day)": 2.0})
            cone_rows.append({"Treatment": tr, "P (mL)": 200.0, "k (day⁻¹)": 0.1, "n": 1.5})

        gompertz_df = pd.DataFrame(gompertz_rows)
        first_df = pd.DataFrame(first_rows)
        logistic_df = pd.DataFrame(logistic_rows)
        cone_df = pd.DataFrame(cone_rows)

        # Use a temp file to avoid collisions and ensure proper cleanup
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            model_fitting.plot_model_fitting(
                df, gompertz_df, first_df, logistic_df, cone_df, save_path=tmp_path
            )

            # Check that the figure exists and has non-zero size
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
