# Doppler-Based Multi-Target Velocity Estimation

This project simulates and analyses Doppler-shifted signals from multiple moving targets (e.g. drones, planes, missiles) to estimate their velocities using short-time Fourier transforms and spectral peak interpolation. It demonstrates signal processing techniques relevant to defence-oriented tracking systems.

---

## Project Overview

- Simulates a sensor receiving a mixture of sine wave signals, each Doppler-shifted by target velocity
- Applies short-time Fourier transform (STFT) using windowed FFTs with 50% overlap
- Detects multiple frequency peaks per window using prominence-based peak finding
- Refines peak frequencies using logarithmic parabolic interpolation
- Classifies each peak by source and estimates velocity via Doppler equations
- Smooths results with moving average filters and outputs threat classification logic

---

## How It Works

1. **Signal Generation**
   - Three targets emit constant-frequency tones.
   - Each tone is Doppler-shifted based on velocity and summed into a single signal.
   - Gaussian noise is added to simulate realistic sensor noise.

2. **Signal Processing**
   - The signal is split into overlapping chunks and windowed with a Hanning filter.
   - FFT is applied to each chunk to generate a time-frequency representation.
   - Frequency peaks are found and refined with logarithmic parabolic interpolation.

3. **Velocity Estimation**
   - Doppler equation is used to estimate target velocities from peak frequencies.
   - Each frequency is classified to its likely source target.
   - Velocities are smoothed using a moving average filter.

4. **Classification & Reporting**
   - Final average velocities are classified into categories:
     - **Drone** (Low threat)
     - **Plane** (Medium threat)
     - **Missile** (High threat)
   - Outputs a textual threat assessment and recommended response.

---

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy

## 🚀 Running the Code

Simply run:

```bash
python doppler_tracking.py
```

It will:
- Simulate signal reception
- Estimate velocities
- Print a threat classification report to the console

---

## 📈 Example Output

```
Target 1:
Estimated Velocity: 44.95 m/s
Classification: Drone
Threat Level: Low
Action Required: Continue Monitoring Proximity. No Immediate Action Required.
...
```

---

## 🔄 Possible Extensions

- Simulate **nonlinear motion** (acceleration, turning)
- Add **target loss** and recovery handling
- Integrate **visualization** (e.g., velocity vs. time plots)
- Include **true vs. estimated** velocity error metrics
- Extend to **real audio data** or SDR input

---

## 🛡️ Context

This project was created as part of a personal initiative to build skills in signal processing, target tracking, and defence-relevant sensing systems.

---

## 📚 Acknowledgements

- Doppler shift equations and signal processing principles based on standard physics and DSP
- Guidance and debugging support from ChatGPT-4
