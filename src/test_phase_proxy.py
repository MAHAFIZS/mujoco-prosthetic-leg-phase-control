import numpy as np
import matplotlib.pyplot as plt

from src.uci_inertial import load_uci_inertial
from src.uci_phase_labels import stance_swing_proxy

WALK_LABELS = {1,2,3}

tr, _ = load_uci_inertial()

# pick one walking window
idx = np.where(np.isin(tr.y, list(WALK_LABELS)))[0][0]
X = tr.X[idx:idx+1]  # [1,T,6]

phase, thr = stance_swing_proxy(X)

acc_mag = np.linalg.norm(X[0,:,0:3], axis=-1)
gyr_mag = np.linalg.norm(X[0,:,3:6], axis=-1)

plt.figure()
plt.plot(acc_mag, label="|acc|")
plt.plot(gyr_mag, label="|gyro|")
plt.plot(phase[0]*np.max(acc_mag), "--", label="stance (proxy)")
plt.title(f"Proxy phase example (thr={thr:.3f})")
plt.legend()
plt.show()
