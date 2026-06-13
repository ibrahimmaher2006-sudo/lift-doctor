"""Simulates ESP sensor streams with injectable failure modes."""
import numpy as np
import pandas as pd


class ESPWell:
    def __init__(self, name="WELL-01", seed=None):
        self.name = name
        self.rng = np.random.default_rng(seed)

    def generate(self, hours=24, failure=None, failure_start_hr=18,
                 messy=True):
        """Returns a DataFrame of 1-min sensor readings.
        failure: None | 'gas_lock' | 'pump_wear' | 'tubing_leak'
        messy: if True, adds real-world noise, dropouts, variable severity.
        """
        n = hours * 60
        t = np.arange(n)
        start = failure_start_hr * 60
        dur = n - start

        # Healthy baselines with daily drift + noise
        intake_psi = 900 + 15 * np.sin(t / 120) + self.rng.normal(0, 5, n)
        motor_temp = 240 + 5 * np.sin(t / 200) + self.rng.normal(0, 1.5, n)
        current_amps = 58 + self.rng.normal(0, 0.8, n)
        vibration = 0.15 + self.rng.normal(0, 0.02, n)

        # Variable severity: each well fails by a different amount
        sev = self.rng.uniform(0.5, 1.5) if messy else 1.0

        if failure == "gas_lock":
            intake_psi[start:] -= np.linspace(0, 200 * sev, dur)
            current_amps[start:] += self.rng.normal(0, 4 * sev, dur)
        elif failure == "pump_wear":
            vibration[start:] += np.linspace(0, 0.25 * sev, dur)
            motor_temp[start:] += np.linspace(0, 25 * sev, dur)
        elif failure == "tubing_leak":
            intake_psi[start:] += np.linspace(0, 80 * sev, dur)
            current_amps[start:] -= np.linspace(0, 8 * sev, dur)

        df = pd.DataFrame({
            "minute": t,
            "intake_psi": intake_psi,
            "motor_temp_f": motor_temp,
            "current_amps": current_amps,
            "vibration_g": vibration,
        })

        if messy:
            df = self._add_realism(df)
        return df

    def _add_realism(self, df):
        """Inject sensor spikes and brief dropouts into all channels."""
        sensor_cols = ["intake_psi", "motor_temp_f", "current_amps", "vibration_g"]
        n = len(df)
        for col in sensor_cols:
            # Occasional spikes: ~0.5% of readings jump wildly
            spike_mask = self.rng.random(n) < 0.005
            df.loc[spike_mask, col] *= self.rng.uniform(1.2, 1.8)

            # Brief dropouts: a few short windows where the sensor flatlines/NaNs
            if self.rng.random() < 0.5:
                d_start = self.rng.integers(0, n - 10)
                df.loc[d_start:d_start + self.rng.integers(2, 8), col] = np.nan

        # Forward-fill dropouts (how real systems handle missing reads)
        df[sensor_cols] = df[sensor_cols].ffill().bfill()
        return df