import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulate 24 hours of ESP intake pressure readings (one per minute)
minutes = np.arange(0, 1440)
normal_pressure = 900 + 15 * np.sin(minutes / 120) + np.random.normal(0, 5, 1440)

# Inject a "gas lock" event: pressure drops sharply at hour 18
pressure = normal_pressure.copy()
pressure[1080:] -= np.linspace(0, 200, 360)

# Simple anomaly flag: anything below 800 psi
anomaly = pressure < 800
# Rolling rate of change: psi per hour over a 30-min window
series = pd.Series(pressure)
rate = series.diff().rolling(30).mean() * 60   # psi/hour

# New rule: alert if sustained decline faster than 40 psi/hr
anomaly = rate < -40
plt.figure(figsize=(12, 5))
plt.plot(minutes / 60, pressure, label="Intake pressure (psi)")
plt.scatter(minutes[anomaly] / 60, pressure[anomaly], color="red", s=10, label="ANOMALY")
plt.axhline(800, color="orange", linestyle="--", label="Alert threshold")
plt.xlabel("Hours")
plt.ylabel("Pressure (psi)")
plt.title("Lift Doctor v0.001 — ESP Intake Pressure Monitor")
plt.legend()
plt.show()

