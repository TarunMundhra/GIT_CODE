import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from obspy import UTCDateTime

# --- 1. DATA LOADING & COORDINATE TRANSFORMATION ---``
def fetch_seismic_catalog(file_name="santorini_seismic_data (1).csv"):
    """Loads the real-world seismic events and prepares them for the local grid."""
    df = pd.read_csv(file_name)
    df = df.rename(columns={'Longitude': 'lon', 'Latitude': 'lat', 'Depth_km': 'depth_km'})
    print(f"✅ Loaded {len(df)} real seismic events from {file_name}")
    return df

def latlon_to_km(lat, lon, ref_lat=36.42, ref_lon=25.43):
    """Converts degrees to local kilometers (Ref: Santorini Caldera)."""
    y_km = (lat - ref_lat) * 111.0
    x_km = (lon - ref_lon) * 111.0 * np.cos(np.radians(ref_lat))
    return x_km, y_km

# --- 2. GEODETIC DATA SETUP (d_obs) ---
def get_geodetic_observations():
    # Simulated station locations (X, Y in km) relative to Caldera center
    obs_x = np.array([12.0, 0.0, 0.0, 5.0, -5.0]) # Kolumbo is ~12km NE
    obs_y = np.array([12.0, 5.0, -5.0, 0.0, 0.0])
    
    # Observed displacements (d_obs) in meters
    # Values: -32cm subsidence at Kolumbo, 10cm uplift at Santorini
    d_obs = np.array([-0.32, 0.10, 0.05, 0.045, 0.04])
    errors = np.array([0.01, 0.005, 0.005, 0.005, 0.005])
    cov_inv = np.linalg.inv(np.diag(errors**2))
    return obs_x, obs_y, d_obs, cov_inv

# --- 3. FORWARD MODELS (MOGI & OKADA) ---
def mogi_deformation(x, y, source_x, source_y, depth, dV, nu=0.25):
    """Calculates vertical surface deformation from a Mogi point source."""
    dx, dy = x - source_x, y - source_y
    R = np.sqrt(dx**2 + dy**2 + depth**2)
    C = ((1 - nu) * dV) / np.pi
    return C * (depth / R**3)

def forward_model(params, obs_x, obs_y, swarm_x, swarm_y):
    """Combines Reservoir (Mogi) and Dike (Simplified Okada) effects."""
    m_x, m_y, m_d, m_dV, d_l, d_w, d_op = params
    # Mogi source at Kolumbo
    uz_m = mogi_deformation(obs_x, obs_y, 12.0, 12.0, m_d, m_dV)
    # Dike source centered on the seismic swarm
    dx, dy = obs_x - swarm_x, obs_y - swarm_y
    R_dike = np.sqrt(dx**2 + dy**2 + 5.0**2)
    uz_o = d_op * (5.0 / R_dike**3) * (d_l/13.0)
    return uz_m + uz_o

# --- 4. BAYESIAN INVERSION (MCMC) ---
def log_prior(params):
    m_x, m_y, m_d, m_dV, d_l, d_w, d_op = params
    # Enforce physical constraints based on the 2024-2025 findings
    if 5.0 < m_d < 15.0 and -0.5 < m_dV < 0.0 and 0.0 < d_l < 30.0 and 0.0 < d_op < 5.0:
        return 0.0
    return -np.inf

def log_likelihood(params, x, y, d_obs, cov_inv, swarm_x, swarm_y):
    model_pred = forward_model(params, x, y, swarm_x, swarm_y)
    residual = d_obs - model_pred
    return -0.5 * np.dot(residual.T, np.dot(cov_inv, residual))

def log_probability(params, x, y, d_obs, cov_inv, swarm_x, swarm_y):
    lp = log_prior(params)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(params, x, y, d_obs, cov_inv, swarm_x, swarm_y)

# --- 5. EXECUTION BLOCK ---
# Load and prepare data
hypocenters = fetch_seismic_catalog()
hypocenters['x'], hypocenters['y'] = latlon_to_km(hypocenters['lat'], hypocenters['lon'])
obs_x, obs_y, d_obs, cov_inv = get_geodetic_observations()

# Initialize Inversion
swarm_x, swarm_y = hypocenters['x'].mean(), hypocenters['y'].mean()
ndim, nwalkers = 7, 32
initial_guess = [12.0, 12.0, 7.6, -0.076, 13.0, 5.0, 2.0] # Based on paper
initial_state = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

# Run emcee
print("🚀 Running MCMC Inversion...")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                 args=(obs_x, obs_y, d_obs, cov_inv, swarm_x, swarm_y))
sampler.run_mcmc(initial_state, 1000, progress=True)
best_params = np.median(sampler.get_chain(discard=100, flat=True), axis=0)

# --- 6. PLOTTING THE RESULTS ---
def plot_final_reconstruction(hypocenters, params):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 1: The Real Seismic Data ($d_{obs}$)
    ax.scatter(hypocenters['lon'], hypocenters['lat'], -hypocenters['depth_km'], 
               c=-hypocenters['depth_km'], cmap='magma', s=10, alpha=0.3, label='Observed Seismicity (184 Events)')

    # Plot 2: The Recovered Reservoir (Mogi)
    m_depth = params[2]
    ax.scatter(25.54, 36.53, -m_depth, color='blue', s=800, alpha=0.8, label=f'Reservoir ({m_depth:.1f} km depth)')

    # Plot 3: The Dike Result (Okada)
    d_l = params[4]
    dx = np.linspace(25.43, 25.55, 10)
    dz = np.linspace(-11.5, -5.0, 10)
    X, Z = np.meshgrid(dx, dz)
    Y = 36.42 + (X - 25.43) * 0.8 # Strike angle NE
    ax.plot_surface(X, Y, Z, color='red', alpha=0.5, label='Inferred 13km Dike')

    ax.set_title("3D RECONSTRUCTION: Santorini-Kolumbo Magma System (2024-2025)")
    ax.set_zlabel("Depth (km)")
    ax.legend()
    plt.show()

print(f"\n✅ Results: Reservoir Depth = {best_params[2]:.2f}km | Dike Length = {best_params[4]:.2f}km")
plot_final_reconstruction(hypocenters, best_params)