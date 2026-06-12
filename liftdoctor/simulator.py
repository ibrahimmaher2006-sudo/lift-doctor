"""Simulates ESP sensor streams with injectable failure modes."""
import numpy as np
import pandas as pd


class ESPWell:
    def __init__(self, name="WELL-01", seed=None):
        self.name = name
        self.rng = np.random.default_rng(seed)

    def generate(self, hours=24, failure=None, failure_start_hr=18):
        """Returns a DataFrame of 1-min sensor readings.
        failure: None | 'gas_lock' | 'pump_wear' | 'tubing_leak'
        """
        n = hours * 60
        t = np.arange(n)
        start = failure_start_hr * 60

        # Healthy baselines with daily drift + noise
        intake_psi = 900 + 15 * np.sin(t / 120) + self.rng.normal(0, 5, n)
        motor_temp = 240 + 5 * np.sin(t / 200) + self.rng.normal(0, 1.5, n)
        current_amps = 58 + self.rng.normal(0, 0.8, n)
        vibration = 0.15 + self.rng.normal(0, 0.02, n)

        if failure == "gas_lock":
            dur = n - start
            intake_psi[start:] -= np.linspace(0, 200, dur)
            current_amps[start:] += self.rng.normal(0, 4, dur)  # erratic load
        elif failure == "pump_wear":
            dur = n - start
            vibration[start:] += np.linspace(0, 0.25, dur)
            motor_temp[start:] += np.linspace(0, 25, dur)
        elif failure == "tubing_leak":
            dur = n - start
            intake_psi[start:] += np.linspace(0, 80, dur)   # backpressure drops
            current_amps[start:] -= np.linspace(0, 8, dur)  # pump unloads

        return pd.DataFrame({
            "minute": t,
            "intake_psi": intake_psi,
            "motor_temp_f": motor_temp,
            "current_amps": current_amps,
            "vibration_g": vibration,
        })