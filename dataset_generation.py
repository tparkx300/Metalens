import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
import time
from pathlib import Path
import json
import gc

# RCWA simulation package
from clabRCWA import Environment, Incident, RCWA, geometry, utils


def main():
    print("="*70)
    print("RCWA DATASET GENERATION")
    print("="*70)
    
    # ========================================
    # Load TiO2 Material Data
    # ========================================
    print("\n[1/5] Loading material data...")
    tio2_data = pd.read_csv('TiO2_refractive_index_Bodurov.csv')
    wavelength_data = tio2_data['wl'].values
    n_data = tio2_data['n'].values
    
    n_interp = interp1d(
        wavelength_data, 
        n_data, 
        kind='cubic',
        fill_value='extrapolate',
        bounds_error=False
    )
    print("✓ TiO₂ material data loaded")
    
    # ========================================
    # Wavelength and Material Settings
    # ========================================
    frequencies_thz = np.arange(4.3, 7.6, 0.04) # [THz]
    wavelengths_um = 3 / frequencies_thz # Convert to [μm]
    
    n_tio2_array = n_interp(wavelengths_um)
    epsilon_tio2_array = n_tio2_array ** 2
    
    # Glass substrate (constant)
    n_glass = 1.5
    epsilon_glass = n_glass ** 2
    
    # ========================================
    # Computational Settings
    # ========================================
    resolution = 512
    n_harmonics = 15
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.complex64
    
    # ========================================
    # Fixed Structure Parameters
    # ========================================
    period = 0.28           # Unit cell period [μm]
    post_height = 0.5    # TiO₂ post height [μm]
    
    # ========================================
    # Incident Light Parameters
    # ========================================
    theta_deg = 0.0    # Normal incidence
    phi_deg = 0.0
    
    print(f"\n[2/5] Simulation setup")
    print(f"  Device: {device}")
    print(f"  Grid resolution: {resolution} × {resolution}")
    print(f"  Fourier harmonics: {n_harmonics}")
    print(f"  Wavelengths: {len(wavelengths_um)} points ({wavelengths_um.min()*1000:.0f}-{wavelengths_um.max()*1000:.0f} nm)")
    print(f"  Period: {period*1000:.0f} nm, Height: {post_height*1000:.0f} nm")
    
    # ========================================
    # Dataset Parameters: Rectangle Geometries
    # ========================================
    print(f"\n[3/5] Defining geometry parameters...")
    
    # For testing: small number of samples
    lengths = np.linspace(0.03, 0.27, 18)   # 18 different lengths [μm]
    widths = np.linspace(0.03, 0.27, 18)    # 18 different widths [μm]
    
    total_geometries = len(lengths) * len(widths)
    print(f"✓ {len(lengths)} lengths × {len(widths)} widths = {total_geometries} geometries")
    
    # Create output directory structure
    base_dir = Path('dataset')
    base_dir.mkdir(exist_ok=True)
    
    # ========================================
    # Prepare Metadata Storage
    # ========================================
    metadata = []
    sample_idx = 0
    
    # ========================================
    # Main Dataset Generation Loop
    # ========================================
    print(f"\n[4/5] Generating dataset...")
    print("="*70)
    
    total_start = time.time()
    
    # ========================================
    # CREATE ONCE FOR ENTIRE DATASET 
    # ========================================
    print("\nInitializing simulation objects (one-time setup)...")
    
    # Create initial pattern (any geometry, will be updated)
    initial_pattern = geometry.rectangle(
        resolution=resolution,
        p=period,
        wx=0.1,  # Arbitrary initial value
        wy=0.1,
        device=device
    )
    
    # Create Environment
    env = Environment(period, period, n_harmonics, resolution, device, dtype)
    env.set_ref_layer(epsilon_r=epsilon_glass)
    env.add_layer(
        thickness=post_height,
        epsilon_material=epsilon_tio2_array[0],  # Use first wavelength's epsilon
        epsilon_background=1.0,
        geometry_pattern=initial_pattern
    )
    env.set_trn_layer(epsilon_r=1.0)
    env.build()
    
    # Create Incident
    inc = Incident(
        lamb0=wavelengths_um[0],  # Use first wavelength
        Lambda_x=period,
        Lambda_y=period,
        n_harmonics=n_harmonics,
        theta=theta_deg,
        phi=phi_deg,
        amp_TE=1.0,
        epsilon_ref=env._ref_layer['epsilon_r'][0],
        mu_ref=env._ref_layer['mu_r'][0],
        epsilon_trn=env._trn_layer['epsilon_r'][0],
        mu_trn=env._trn_layer['mu_r'][0],
        device=device,
        dtype=dtype
    )
    
    print("✓ Setup complete - now processing samples...")
    print()
    
    # ========================================
    # Loop Over Geometries and Wavelengths
    # ========================================
    
    for rect_length in lengths:
        for rect_width in widths:
            sample_id = f"sample_{sample_idx:05d}"
            
            print(f"[{sample_idx+1}/{total_geometries}] {sample_id}")
            print(f"  Geometry: L={rect_length*1000:.1f} nm, W={rect_width*1000:.1f} nm")
            
            # UPDATE geometry pattern (in-place)
            new_pattern = geometry.rectangle(
                resolution=resolution,
                p=period,
                wx=rect_length,
                wy=rect_width,
                device=device
            )
            env.layer_geometries[1] = new_pattern  # Update layer 1's geometry in-place
            
            # Create NEW RCWA solver for this sample
            rcwa_solver = RCWA(env, inc)
            
            # Storage for this geometry - CPU storage
            t_xx = torch.zeros(len(wavelengths_um), dtype=dtype, device='cpu')
            
            # X-polarized input only
            pol_start = time.time()
            
            with torch.no_grad():
                for i, (wavelength, epsilon_tio2) in enumerate(zip(wavelengths_um, epsilon_tio2_array)):
                    
                    # UPDATE epsilon (in-place)
                    env.epsilon_rs[1, 0] = epsilon_tio2
                    
                    # UPDATE wavelength (in-place via method)
                    inc.update_wavelength(wavelength, epsilon_glass)
                    
                    # SOLVE (solver is fresh per sample - consistent speed!)
                    rcwa_solver.clear_cache()
                    torch.cuda.empty_cache()
                    rcwa_solver.solve_S_matrix()
                    rcwa_solver.solve_RT()
                    
                    # Extract transmitted field at (0,0) order
                    _, t_field = rcwa_solver.field_by_order(nx=0, ny=0)
                    
                    t_xx[i] = t_field[0].cpu()  # Move to CPU immediately
                    
                    # Only delete the small t_field
                    del t_field
            
            pol_end = time.time()
            print(f"    X-pol: {utils.format_time(pol_end - pol_start)}")
            
            # Save transmission data for this sample
            np.save(base_dir / f"{sample_id}_t_xx.npy", t_xx.numpy())
            
            # Store metadata
            metadata.append({
                'sample_id': sample_id,
                'sample_index': sample_idx,
                'rect_length_um': rect_length,
                'rect_width_um': rect_width,
                'rect_length_nm': rect_length * 1000,
                'rect_width_nm': rect_width * 1000,
                'period_um': period,
                'post_height_um': post_height,
            })
            
            print(f"  ✓ Saved {sample_id}_t_xx.npy")
            
            # Clean up sample-specific objects
            del rcwa_solver, new_pattern, t_xx
            gc.collect()
            torch.cuda.empty_cache()
            
            sample_idx += 1
    
    # ========================================
    # FINAL CLEANUP - Delete the big simulation objects
    # ========================================
    del env, inc, initial_pattern
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    
    total_end = time.time()
    
    # ========================================
    # Save Metadata and Configuration
    # ========================================
    print(f"\n[5/5] Saving metadata and creating train/valid split...")
    
    # Save metadata as CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(base_dir / 'metadata.csv', index=False)
    print(f"✓ Saved metadata.csv ({len(metadata_df)} samples)")
    
    # Save global configuration
    config = {
        'wavelengths_um': wavelengths_um.tolist(),
        'wavelengths_nm': (wavelengths_um * 1000).tolist(),
        'n_wavelengths': len(wavelengths_um),
        'period_um': period,
        'post_height_um': post_height,
        'n_glass': n_glass,
        'epsilon_glass': epsilon_glass,
        'resolution': resolution,
        'n_harmonics': n_harmonics,
        'theta_deg': theta_deg,
        'phi_deg': phi_deg,
    }
    
    with open(base_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config.json")
    
    # ========================================
    # Create Train/Validation Split (90/10)
    # ========================================
    n_samples = len(metadata)
    n_train = int(0.9 * n_samples)
    n_valid = n_samples - n_train
    
    # Shuffle indices
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:]
    
    # Save split indices
    np.save(base_dir / 'train_indices.npy', train_indices)
    np.save(base_dir / 'valid_indices.npy', valid_indices)
    
    # Also save as text for easy inspection
    with open(base_dir / 'train_samples.txt', 'w') as f:
        for idx in train_indices:
            f.write(f"sample_{idx:05d}\n")
    
    with open(base_dir / 'valid_samples.txt', 'w') as f:
        for idx in valid_indices:
            f.write(f"sample_{idx:05d}\n")
    
    print(f"✓ Train/Valid split: {n_train}/{n_valid} samples")
    print(f"  train_indices.npy, valid_indices.npy")
    print(f"  train_samples.txt, valid_samples.txt")
    
    # ========================================
    # Save README for easy loading
    # ========================================
    readme = f"""# RCWA Dataset

## Dataset Information
- Total samples: {n_samples}
- Training samples: {n_train}
- Validation samples: {n_valid}
- Wavelengths: {len(wavelengths_um)} points ({wavelengths_um.min()*1000:.0f}-{wavelengths_um.max()*1000:.0f} nm)

## Files Structure

### Transmission Data
For each sample (e.g., sample_00000):
- `sample_XXXXX_t_xx.npy`: Complex transmission coefficient (x→x polarization)

Each file contains a 1D array of {len(wavelengths_um)} complex numbers (one per wavelength).

### Metadata Files
- `metadata.csv`: Sample geometries and parameters
  - Columns: sample_id, sample_index, rect_length_um, rect_width_um, rect_length_nm, rect_width_nm, period_um, post_height_um
- `config.json`: Global simulation parameters and wavelengths
- `train_indices.npy`: Indices of training samples
- `valid_indices.npy`: Indices of validation samples
- `train_samples.txt`: Sample IDs for training (human-readable)
- `valid_samples.txt`: Sample IDs for validation (human-readable)

## Loading Data Example (Python)

```python
import numpy as np
import pandas as pd
import json

# Load metadata
metadata = pd.read_csv('dataset/metadata.csv')
config = json.load(open('dataset/config.json'))
train_idx = np.load('dataset/train_indices.npy')
valid_idx = np.load('dataset/valid_indices.npy')

# Get wavelengths
wavelengths = np.array(config['wavelengths_um'])

# Load a specific sample
sample_id = 'sample_00000'
t_xx = np.load(f'dataset/{{sample_id}}_t_xx.npy')  # Complex array

# Get geometry for this sample
sample_info = metadata[metadata['sample_id'] == sample_id].iloc[0]
L = sample_info['rect_length_um']
W = sample_info['rect_width_um']

print(f"Sample {{sample_id}}: L={{L}}, W={{W}}")
print(f"Transmission at λ=500nm: t_xx={{t_xx[10]}}")
```

## Generation Info
- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Computation time: {utils.format_time(total_end - total_start)}
- Average per sample: {utils.format_time((total_end - total_start) / n_samples)}
"""
    
    with open(base_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"✓ Saved README.md")
    
    # Show sample metadata
    print("\nSample metadata (first 3 rows):")
    print(metadata_df.head(3).to_string(index=False))


if __name__ == '__main__':
    main()
