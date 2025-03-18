import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.optimize import minimize

# Utility functions for loading a subject from the PWDB

# Function to load data for a specific subject ID
def load_subject_data(subject_id=1, pwdb_data=None, loc='AorticRoot'):
    """
    Loads data for a specific subject ID from the preloaded PWDB dataset.

    Args:
        subject_id (int): The ID of the subject to load data for.
        pwdb_data (dict): Preloaded PWDB data from a .mat file.
        loc (str): String indicating the arterial site (AorticRoot, ThorAorta, Radial, Digital, ..., cf. https://github.com/peterhcharlton/pwdb/wiki/pwdb_data.mat#datawaves).

    Returns:
        tuple: (time, pressure, velocity, area, flow, HR)
    """
    if pwdb_data is None:
        raise ValueError("The 'pwdb_data' parameter must be provided with preloaded data.")
    
    # Construct the field names dynamically based on the 'loc' input
    pressure_key = f'P_{loc}'
    velocity_key = f'U_{loc}'
    area_key = f'A_{loc}'

    # Determine total subjects from the data structure
    try:
        total_subjects = pwdb_data['waves'][0, 0][pressure_key][0, 0].shape[1]
    except KeyError:
        raise ValueError(f"Invalid 'loc' value: '{loc}'. Check the available locations in the dataset.")

    if subject_id < 1 or subject_id > total_subjects:
        raise ValueError(f"Subject ID must be between 1 and {total_subjects}")

    # Extract data for the chosen subject ID
    pressure = pwdb_data['waves'][0, 0][pressure_key][0, 0][0, subject_id - 1].flatten() # in mmHg
    velocity = pwdb_data['waves'][0, 0][velocity_key][0, 0][0, subject_id - 1].flatten() # in m/s
    area = pwdb_data['waves'][0, 0][area_key][0, 0][0, subject_id - 1].flatten() # in m^2
    HR = pwdb_data['haemods'][0, 0][0][subject_id - 1]['HR'][0, 0]
    flow = velocity*area*1e6 # in ml/s
  
    # Sampling frequency (500 Hz for PWDB)
    fs = pwdb_data['waves'][0, 0]['fs'][0, 0]
    # Generate the time vector
    time_vector = np.arange(len(pressure)) / fs 
    time = np.linspace(0, 60 / HR, len(pressure))
    print(f"Loaded data for Subject ID: {subject_id}")
    #print('NB. Pressure is in mmHg, flow in ml/s, area in m^2')
    print(f"Heart Rate: {HR}")

    return pressure, flow, area, HR, pvr, time

def plot_pressure_and_flow(time, pressure, flow):
    """
    Plots pressure and flow side by side.

    Parameters:
        time (numpy array): Time vector (seconds)
        pressure (numpy array): Pressure data (mmHg)
        flow (numpy array): Flow data (ml/s)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Pressure
    axes[0].plot(time, pressure)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Pressure (mmHg)")
    axes[0].set_title("Aortic Pressure")
    axes[0].grid(True)

    # Plot Flow
    axes[1].plot(time, flow)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Flow (ml/s)")
    axes[1].set_title("Aortic Flow")
    axes[1].grid(True)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


# Utility functions for calculating input and characteristic impedance
def calculate_input_impedance(time, pressure, flow, show=True):
    """
    Computes the input impedance of the cardiovascular system and displays results.
    
    Parameters:
        time (numpy array): Time vector (seconds)
        pressure (numpy array): Pressure data (mmHg)
        flow (numpy array): Flow data (ml/s)
        show (bool): Whether to display impedance plots (default=True)
    
    Returns:
        tuple: (Zin, Zin_magnitude, Zin_phase, frequency_vector)
    """
    num_points = len(pressure)
    T = np.max(time)

    # Compute input impedance (FFT of pressure and flow)
    Zin = fft(pressure) / fft(flow)
    # Compute magnitude and phase of impedance
    Zin_magnitude = np.abs(Zin)
    Zin_phase = np.unwrap(np.angle(Zin)) * 180 / np.pi  # Convert phase to degrees

    # Compute frequency vector
    frequency_vector = np.arange(num_points) / T

    if show:
        # Plot impedance modulus
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes[0].semilogy(frequency_vector[:11], Zin_magnitude[:11], 'm+', label='Impedance Magnitude')
        axes[0].plot(frequency_vector[:11], Zin_magnitude[:11], 'b')
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel("Impedance (mmHg/(ml/s))")
        axes[0].set_title("Modulus of Input Impedance")
        axes[0].grid(True)
        axes[0].legend()

        # Plot impedance phase
        axes[1].plot(frequency_vector[:11], Zin_phase[:11], 'm+', label='Impedance Phase')
        axes[1].plot(frequency_vector[:11], Zin_phase[:11], 'b')
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Phase Angle (°)")
        axes[1].set_title("Phase of Input Impedance")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        
        # Print harmonic information
        print("\nHarmonic    Frequency (Hz)    Modulus    Phase (°)")
        print("---------------------------------------------------")
        for i in range(11):
            print(f"{i+1:<10}{frequency_vector[i]:<18.5f}{Zin_magnitude[i]:<12.5f}{Zin_phase[i]:<10.5f}")

    return Zin, Zin_magnitude, Zin_phase, frequency_vector

def calculate_characteristic_impedance(time, pressure, flow, show=False):
    """
    Computes the characteristic impedance (Zc) from the input impedance.

    Parameters:
        time (numpy array): Time vector (seconds)
        pressure (numpy array): Pressure data (mmHg)
        flow (numpy array): Flow data (ml/s)
        show (bool): Whether to display impedance plots (default=False)

    Returns:
        float: Characteristic impedance (Zc)
    """
    T = np.max(time)  # Heart cycle duration

    # Get input impedance magnitude using `calculate_input_impedance`
    _, Zin_magnitude, _ , _ = calculate_input_impedance(time, pressure, flow, show)

    # Compute harmonic indices
    first_harmonic = int(1 + np.ceil(3 * T))
    last_harmonic = int(1 + np.floor(15 * T))

    # Calculate characteristic impedance
    Zc = np.mean(Zin_magnitude[first_harmonic:last_harmonic])

    # Print result
    print(f'Zc = {Zc:.4f}')

    return Zc

def calculate_compliance_using_PPM(time, pressure, flow, C_WK2_init=1.0):
    """
    Computes pulse pressure and evaluates compliance using the 2-element Windkessel model.

    Parameters:
        time (numpy array): Time vector (seconds)
        pressure (numpy array): Pressure data (mmHg)
        flow (numpy array): Flow data (ml/s)
        C_WK2_init (float): Initial guess for arterial compliance (mL/mmHg)

    Returns:
        tuple: (TPR, truePP, PP_WK2, C_WK2)
    """
    num_points = len(pressure)
    meanP = np.mean(pressure)
    meanQ = np.mean(flow)

    # Compute total peripheral resistance (TPR)
    TPR = meanP / meanQ  # mmHg.s/mL
    T = np.max(time)

    # Compute input impedance
    Zin, Zin_magnitude, Zin_phase, frequency_vector = calculate_input_impedance(time, pressure, flow, show=False)

    # Pulse pressure method: get user input for compliance
    C_WK2 = C_WK2_init  # Default value for compliance

    # True Pulse Pressure (PP) = max - min pressure in early systole
    sys_idx = np.argmax(pressure[:num_points])  # Find systolic peak
    truePP = np.max(pressure[:num_points]) - np.min(pressure[:sys_idx])

    # Remove DC level and compute FFT on flow
    Qtofft = flow - meanQ
    fftQ = fft(Qtofft)

    # Initialize FFT pressure response
    fftP_WK2 = np.zeros(num_points, dtype=complex)

    # Compute P-response of 2-element WK model
    Z_WK2 = np.zeros(num_points, dtype=complex)
    Z_WK2[0] = TPR  # DC component

    for j in range(1, num_points):
        freq_j = (j) / T
        Z_WK2[j] = TPR / (1 + 1j * 2 * np.pi * freq_j * TPR * C_WK2)
        fftP_WK2[j] = Z_WK2[j] * fftQ[j]

    # Compute inverse FFT to get the estimated pressure waveform
    P_WK2 = 2 * np.real(ifft(fftP_WK2[:num_points//2], num_points)) + meanP

    # Compute estimated pulse pressure from Windkessel model
    PP_WK2 = np.max(P_WK2) - np.min(P_WK2)

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # Plot Impedance
    axes[0].semilogy(frequency_vector[:11], np.abs(Zin[:11]), 'm+', label='Measured')
    axes[0].semilogy(frequency_vector[:11], np.abs(Z_WK2[:11]), 'y', label='2-WK Model')
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Impedance (mmHg.s/m)")
    axes[0].set_title("Input Impedance")
    axes[0].legend()
    axes[0].grid()

    # Plot Pressure
    axes[1].plot(time, pressure, label='Measured Pressure', color='blue')
    axes[1].plot(time, P_WK2, label=f'2-WK Model (C={C_WK2:.2f} mL/mmHg)', linestyle='dashed', color='red')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Pressure (mmHg)")
    axes[1].set_title("Pressure Comparison")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()

    print(f"WK2 Compliance: {C_WK2:.2f} mL/mmHg")
    print(f"True Pulse Pressure: {truePP:.2f} mmHg, 2-WK Pulse Pressure: {PP_WK2:.2f} mmHg")
    
    return TPR, truePP, PP_WK2, C_WK2

# Functions to fit the 3-element Windkessel model
def function_cost_wk3(p, time, pressure, flow, T):
    """
    Computes the error (e) between measured pressure and modeled pressure 
    response using a 3-element Windkessel model.

    Parameters:
        p (list or numpy array): Windkessel model parameters [R_WK3, Zc_WK3, C_WK3]
        time (numpy array): Time vector (seconds)
        pressure (numpy array): Measured pressure data (mmHg)
        flow (numpy array): Measured flow data (ml/s)
        T (float): Heart cycle duration

    Returns:
        float: Error value (sum of squared differences)
    """
    num_points = len(pressure)  # Number of data points

    # Compute FFT of flow signal
    fftQ = fft(flow)

    # Extract Windkessel parameters
    R_WK3, Zc_WK3, C_WK3 = p

    # Step 1: Transform FFT of flow into harmonics
    fftQ[0] = fftQ[0] / num_points  # Normalize DC component

    if num_points % 2 == 0:  # Even case
        for k in range(1, num_points // 2 - 1):
            fftQ[k] = 2 * fftQ[k] / num_points
    else:  # Odd case
        for k in range(1, int(np.ceil(num_points / 2)) - 1):
            fftQ[k] = 2 * fftQ[k] / num_points

    # Step 2: Compute input impedance (Z_WK3) and pressure harmonics (fftP_WK3)
    freq = np.zeros(num_points // 2 - 1)  # Frequency vector
    Z_WK3 = np.zeros(num_points // 2 - 1, dtype=complex)  # Impedance vector
    fftP_WK3 = np.zeros(num_points, dtype=complex)  # Pressure spectrum

    if num_points % 2 == 0:  # Even case
        for j in range(num_points // 2 - 1):
            freq[j] = (j) / T
            Z_WK3[j] = Zc_WK3 + R_WK3 / (1 + 1j * 2 * np.pi * freq[j] * R_WK3 * C_WK3)
            fftP_WK3[j] = Z_WK3[j] * fftQ[j]
    else:  # Odd case
        for j in range(int(np.ceil(num_points / 2)) - 1):
            freq[j] = (j) / T
            Z_WK3[j] = Zc_WK3 + R_WK3 / (1 + 1j * 2 * np.pi * freq[j] * R_WK3 * C_WK3)
            fftP_WK3[j] = Z_WK3[j] * fftQ[j]

    # Step 3: Apply scaling for inverse FFT
    fftP_WK3[0] = fftP_WK3[0] * num_points

    if num_points % 2 == 0:  # Even case
        for j in range(num_points // 2 - 1):
            fftP_WK3[j + 1] = fftP_WK3[j + 1] * num_points / 2
            fftP_WK3[num_points - j - 1] = np.conj(fftP_WK3[j + 1])
        fftP_WK3[num_points // 2] = fftP_WK3[j + 1] * num_points / 2
    else:  # Odd case
        for j in range(int(np.ceil(num_points / 2)) - 1):
            fftP_WK3[j + 1] = fftP_WK3[j + 1] * num_points / 2
            fftP_WK3[num_points - j - 1] = np.conj(fftP_WK3[j + 1])

    # Step 4: Compute inverse FFT to get time-domain pressure response
    P_WK3 = np.real(ifft(fftP_WK3))

    # Step 5: Compute error between measured and modeled pressure
    if pressure.shape == P_WK3.shape:
        verschil = (pressure - P_WK3) ** 2
    else:
        verschil = (pressure - P_WK3.T) ** 2  # Handle shape mismatch

    e = np.sum(verschil)  # Sum of squared differences

    return e  # Return error value

def wk3(time, pressure, flow):
    """
    Computes the fitted parameters of a 3-element Windkessel (WK3) model.

    Parameters:
        time (numpy array): Time vector (seconds)
        pressure (numpy array): Measured pressure data (mmHg)
        flow (numpy array): Measured flow data (ml/s)

    Returns:
        tuple: (R_WK3, Zc_WK3, C_WK3)
    """

    # Number of data points
    num_points = len(pressure)

    # Compute mean pressure and flow
    meanP = np.mean(pressure)
    meanQ = np.mean(flow)

    # Compute Total Peripheral Resistance (TPR)
    TPR = meanP / meanQ
    T = np.max(time)  # Heart cycle duration

    # Compute characteristic impedance
    Zin, _ , _ , _ = calculate_input_impedance(time, pressure, flow, show=False)
    Zc = calculate_characteristic_impedance(time, pressure, flow, show=False)

    # Initial guesses for Windkessel parameters
    R_initial = TPR
    Zc_initial = Zc
    C_initial = 1.5

    # Optimization using scipy's minimize function
    initial_guess = [R_initial, Zc_initial, C_initial]
    result = minimize(function_cost_wk3, initial_guess, args=(time, pressure, flow, T), method='Nelder-Mead')

    # Extract optimized parameters
    R_WK3, Zc_WK3, C_WK3 = result.x

    # Compute estimated pressure from 3-element WK model
    fftQ = fft(flow)
    fftQ[0] = fftQ[0] / num_points  # Normalize DC component

    # Scale harmonics
    for j in range(1, num_points // 2):
        fftQ[j] = 2 * fftQ[j] / num_points

    # Compute input impedance for WK3 model
    freq = np.zeros(num_points // 2)
    Z_WK3 = np.zeros(num_points // 2, dtype=complex)
    fftP_3WK = np.zeros(num_points, dtype=complex)

    for j in range(num_points // 2):
        freq[j] = j / T
        Z_WK3[j] = Zc_WK3 + R_WK3 / (1 + 1j * 2 * np.pi * freq[j] * R_WK3 * C_WK3)
        fftP_3WK[j] = Z_WK3[j] * fftQ[j]

    # Apply scaling for inverse FFT
    fftP_3WK[0] = fftP_3WK[0] * num_points

    if num_points % 2 == 0:  # Even case
        for j in range(num_points // 2 - 1):
            fftP_3WK[j + 1] = fftP_3WK[j + 1] * num_points / 2
            fftP_3WK[num_points - j - 1] = np.conj(fftP_3WK[j + 1])
        fftP_3WK[num_points // 2] = fftP_3WK[j + 1] * num_points / 2

    else:  # Odd case
        for j in range(int(np.ceil(num_points / 2)) - 1):
            fftP_3WK[j + 1] = fftP_3WK[j + 1] * num_points / 2
            fftP_3WK[num_points - j - 1] = np.conj(fftP_3WK[j + 1])

    # Compute inverse FFT to get estimated pressure waveform
    P_3WK = np.real(ifft(fftP_3WK))

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # Plot Impedance
    axes[0].semilogy(freq[:11], np.abs(Zin[:11]), '+', label='True')
    axes[0].plot(freq[:11], np.abs(Z_WK3[:11]), linewidth=2, label='3-WK model')
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Impedance (mmHg.s/mL)")
    axes[0].set_title("Input Impedance")
    axes[0].legend()
    axes[0].grid()

    # Plot Pressure
    axes[1].plot(time, pressure, label="True Pressure")
    axes[1].plot(time, P_3WK, linewidth=2, label="3-WK Model Pressure")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Pressure (mmHg)")
    axes[1].set_title("Pressure Comparison")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()

    # Print estimated parameters
    print(f"\nEstimated R: {R_WK3:.4f} mmHg·s/mL, Zc: {Zc_WK3:.4f} mmHg·s/mL, 3-WK C: {C_WK3:.4f} mL/mmHg")

    return R_WK3, Zc_WK3, C_WK3

def calculate_pwv(rho, A, Zc):
    Z_SI = Zc * 133.32 * 1e6  # Convert to SI units
    PWV = Z_SI * A / rho  # Pulse wave velocity in m/s
    print(f'PWV = {PWV:.2f}')
    return PWV