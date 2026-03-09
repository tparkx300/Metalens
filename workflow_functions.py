"""
workflow_functions.py

Composable workflow functions for the metalens design pipeline.
Each function is a self-contained pipeline step that can be discovered
and chained by an AI agent (e.g., Aethergraph).

Constraints (per agent workflow spec):
  1. All inputs and outputs are JSON-serializable (plain Python dicts,
     lists, floats, strings). Torch tensors and model objects are kept
     internal; only file paths are returned.
  2. Parameters are grouped into config dicts so agents can modify
     individual keys without disrupting the function signature.
  3. No major workflow function calls another — each is a standalone node.

Pipeline steps:
  1. generate_dataset       — RCWA simulation over geometry grid
  2. train_surrogate_model  — train PhysicsSurrogateModel on the dataset
  3. initialize_canvas      — build the metalens with a fresh SIREN param field
  4. optimize_metalens      — run differentiable ray-tracing optimization
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# 1. generate_dataset
# ==============================================================================

def generate_dataset(
    geometry_config: Dict[str, Any],
    sim_config: Dict[str, Any],
    material_data_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Run RCWA simulations over a 2D grid of rectangular nanopost geometries
    and save the complex transmission spectra to disk.

    Parameters
    ----------
    geometry_config : dict
        Geometry sweep parameters:
          - "lengths_um"    : list[float]  — pillar lengths to sweep [μm]
          - "widths_um"     : list[float]  — pillar widths to sweep [μm]
          - "period_um"     : float        — unit cell period [μm]
          - "post_height_um": float        — TiO2 post height [μm]
          - "n_glass"       : float        — substrate refractive index
                                             (default 1.5)

    sim_config : dict
        Simulation engine parameters:
          - "resolution"       : int   — spatial grid resolution (default 512)
          - "n_harmonics"      : int   — number of Fourier harmonics (default 15)
          - "freq_start_thz"   : float — start frequency [THz] (default 4.3)
          - "freq_stop_thz"    : float — stop frequency [THz] (default 7.6)
          - "freq_step_thz"    : float — frequency step [THz] (default 0.04)
          - "theta_deg"        : float — polar incidence angle (default 0.0)
          - "phi_deg"          : float — azimuthal incidence angle (default 0.0)

    material_data_path : str
        Path to the TiO2 refractive index CSV file.
        Required columns: "wl" (wavelength in μm), "n" (refractive index).

    output_dir : str
        Directory where dataset files will be written.

    Returns
    -------
    dict
        - "dataset_path"  : str   — path to the output directory
        - "n_samples"     : int   — total number of geometry samples
        - "config_path"   : str   — path to the saved config.json
        - "metadata_path" : str   — path to the saved metadata.csv
        - "elapsed_sec"   : float — wall-clock time in seconds
    """
    from clabRCWA import Environment, Incident, RCWA, geometry, utils

    # ---- Unpack configs ----
    lengths = np.array(geometry_config["lengths_um"], dtype=np.float64)
    widths  = np.array(geometry_config["widths_um"],  dtype=np.float64)
    period      = float(geometry_config["period_um"])
    post_height = float(geometry_config["post_height_um"])
    n_glass     = float(geometry_config.get("n_glass", 1.5))

    resolution   = int(sim_config.get("resolution",     512))
    n_harmonics  = int(sim_config.get("n_harmonics",    15))
    freq_start   = float(sim_config.get("freq_start_thz", 4.3))
    freq_stop    = float(sim_config.get("freq_stop_thz",  7.6))
    freq_step    = float(sim_config.get("freq_step_thz",  0.04))
    theta_deg    = float(sim_config.get("theta_deg",    0.0))
    phi_deg      = float(sim_config.get("phi_deg",      0.0))

    # ---- Material data ----
    tio2_data = pd.read_csv(material_data_path)
    n_interp = interp1d(
        tio2_data["wl"].values,
        tio2_data["n"].values,
        kind="cubic",
        fill_value="extrapolate",
        bounds_error=False,
    )

    frequencies_thz  = np.arange(freq_start, freq_stop, freq_step)
    wavelengths_um   = 3.0 / frequencies_thz
    n_tio2_array     = n_interp(wavelengths_um)
    epsilon_tio2_arr = n_tio2_array ** 2
    epsilon_glass    = n_glass ** 2

    # ---- Output directory ----
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    dtype  = torch.complex64

    # ---- One-time simulation setup ----
    initial_pattern = geometry.rectangle(
        resolution=resolution, p=period, wx=0.1, wy=0.1, device=device
    )

    env = Environment(period, period, n_harmonics, resolution, device, dtype)
    env.set_ref_layer(epsilon_r=epsilon_glass)
    env.add_layer(
        thickness=post_height,
        epsilon_material=epsilon_tio2_arr[0],
        epsilon_background=1.0,
        geometry_pattern=initial_pattern,
    )
    env.set_trn_layer(epsilon_r=1.0)
    env.build()

    inc = Incident(
        lamb0=wavelengths_um[0],
        Lambda_x=period,
        Lambda_y=period,
        n_harmonics=n_harmonics,
        theta=theta_deg,
        phi=phi_deg,
        amp_TE=1.0,
        epsilon_ref=env._ref_layer["epsilon_r"][0],
        mu_ref=env._ref_layer["mu_r"][0],
        epsilon_trn=env._trn_layer["epsilon_r"][0],
        mu_trn=env._trn_layer["mu_r"][0],
        device=device,
        dtype=dtype,
    )

    # ---- Main loop ----
    metadata    = []
    sample_idx  = 0
    total_start = time.time()

    for rect_length in lengths:
        for rect_width in widths:
            sample_id = f"sample_{sample_idx:05d}"

            new_pattern = geometry.rectangle(
                resolution=resolution, p=period,
                wx=rect_length, wy=rect_width, device=device
            )
            env.layer_geometries[1] = new_pattern

            rcwa_solver = RCWA(env, inc)

            t_xx = torch.zeros(len(wavelengths_um), dtype=dtype, device="cpu")

            with torch.no_grad():
                for i, (wl, eps_tio2) in enumerate(
                    zip(wavelengths_um, epsilon_tio2_arr)
                ):
                    env.epsilon_rs[1, 0] = eps_tio2
                    inc.update_wavelength(wl, epsilon_glass)
                    rcwa_solver.clear_cache()
                    rcwa_solver.solve_S_matrix()
                    rcwa_solver.solve_RT()
                    _, t_field = rcwa_solver.field_by_order(nx=0, ny=0)
                    t_xx[i] = t_field[0].cpu()
                    del t_field

            np.save(base_dir / f"{sample_id}_t_xx.npy", t_xx.numpy())

            metadata.append({
                "sample_id":       sample_id,
                "sample_index":    sample_idx,
                "rect_length_um":  float(rect_length),
                "rect_width_um":   float(rect_width),
                "rect_length_nm":  float(rect_length * 1000),
                "rect_width_nm":   float(rect_width  * 1000),
                "period_um":       period,
                "post_height_um":  post_height,
            })

            del rcwa_solver, new_pattern, t_xx
            gc.collect()

            sample_idx += 1

    del env, inc, initial_pattern
    gc.collect()

    elapsed = time.time() - total_start

    # ---- Save metadata ----
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(base_dir / "metadata.csv", index=False)

    config = {
        "wavelengths_um":  wavelengths_um.tolist(),
        "wavelengths_nm":  (wavelengths_um * 1000).tolist(),
        "n_wavelengths":   len(wavelengths_um),
        "period_um":       period,
        "post_height_um":  post_height,
        "n_glass":         n_glass,
        "epsilon_glass":   epsilon_glass,
        "resolution":      resolution,
        "n_harmonics":     n_harmonics,
        "theta_deg":       theta_deg,
        "phi_deg":         phi_deg,
    }
    config_path = str(base_dir / "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ---- Train / validation split (90/10) ----
    n_samples = len(metadata)
    np.random.seed(42)
    indices       = np.random.permutation(n_samples)
    n_train       = int(0.9 * n_samples)
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:]

    np.save(base_dir / "train_indices.npy", train_indices)
    np.save(base_dir / "valid_indices.npy", valid_indices)

    return {
        "dataset_path":  str(base_dir),
        "n_samples":     n_samples,
        "n_train":       int(n_train),
        "n_valid":       int(n_samples - n_train),
        "config_path":   config_path,
        "metadata_path": str(base_dir / "metadata.csv"),
        "elapsed_sec":   round(elapsed, 2),
    }


# ==============================================================================
# 2. train_surrogate_model
# ==============================================================================

def train_surrogate_model(
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    dataset_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Train PhysicsSurrogateModel on the RCWA dataset and save checkpoints.

    Parameters
    ----------
    model_config : dict
        Model architecture parameters:
          - "hidden_dims" : list[int] — MLP hidden layer sizes
                           (default [256, 512, 512, 512, 256])
          - "w_dim"       : int       — geometry input dimension (default 2)
          - "cond_dim"    : int       — condition (wavelength) dimension (default 1)

    train_config : dict
        Training hyperparameters:
          - "batch_size"         : int   (default 512)
          - "n_epochs"           : int   (default 500)
          - "lr"                 : float (default 1e-3)
          - "grad_loss_alpha"    : float — spectral gradient loss weight (default 0.2)
          - "scheduler_patience" : int   — ReduceLROnPlateau patience (default 10)
          - "scheduler_factor"   : float — LR reduction factor (default 0.5)

    dataset_path : str
        Path to the dataset directory produced by generate_dataset().
        Must contain: metadata.csv, config.json,
                      train_indices.npy, valid_indices.npy,
                      sample_XXXXX_t_xx.npy files.

    output_dir : str
        Directory where checkpoints will be saved.

    Returns
    -------
    dict
        - "best_checkpoint_path" : str   — path to best_surrogate.pth
        - "last_checkpoint_path" : str   — path to last_surrogate.pth
        - "history_path"         : str   — path to training_history.npy
        - "best_valid_loss"      : float
        - "best_epoch"           : int
        - "elapsed_sec"          : float
    """
    from metalens_tracing.networks import PhysicsSurrogateModel

    # ---- Unpack configs ----
    hidden_dims        = list(model_config.get("hidden_dims", [256, 512, 512, 512, 256]))
    w_dim              = int(model_config.get("w_dim",    2))
    cond_dim           = int(model_config.get("cond_dim", 1))

    batch_size         = int(train_config.get("batch_size",         512))
    n_epochs           = int(train_config.get("n_epochs",           500))
    lr                 = float(train_config.get("lr",                1e-3))
    grad_loss_alpha    = float(train_config.get("grad_loss_alpha",   0.2))
    scheduler_patience = int(train_config.get("scheduler_patience", 10))
    scheduler_factor   = float(train_config.get("scheduler_factor",  0.5))

    # ---- Paths ----
    dataset_dir    = Path(dataset_path)
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load dataset ----
    metadata      = pd.read_csv(dataset_dir / "metadata.csv")
    config        = json.load(open(dataset_dir / "config.json"))
    train_indices = np.load(dataset_dir / "train_indices.npy")
    valid_indices = np.load(dataset_dir / "valid_indices.npy")
    wavelengths   = np.array(config["wavelengths_um"], dtype=np.float32)
    n_samples     = len(metadata)
    n_wl          = len(wavelengths)

    all_t_xx = np.zeros((n_samples, n_wl), dtype=np.complex64)
    for idx in range(n_samples):
        all_t_xx[idx] = np.load(dataset_dir / f"sample_{idx:05d}_t_xx.npy")

    # ---- Normalization bounds (training set only) ----
    train_meta  = metadata.iloc[train_indices]
    norm_bounds = {
        "length_min":     float(train_meta["rect_length_um"].min()),
        "length_max":     float(train_meta["rect_length_um"].max()),
        "width_min":      float(train_meta["rect_width_um"].min()),
        "width_max":      float(train_meta["rect_width_um"].max()),
        "wavelength_min": float(wavelengths.min()),
        "wavelength_max": float(wavelengths.max()),
    }

    # ---- Dataset and loaders ----
    class _RCWADataset(Dataset):
        def __init__(self, sample_indices):
            self.meta = metadata.iloc[sample_indices].reset_index(drop=True)
            self.n_wl = n_wl

        def __len__(self):
            return len(self.meta) * self.n_wl

        def __getitem__(self, idx):
            s_idx  = idx // self.n_wl
            wl_idx = idx  % self.n_wl
            row    = self.meta.iloc[s_idx]
            length = float(row["rect_length_um"])
            width  = float(row["rect_width_um"])
            wl     = float(wavelengths[wl_idx])
            t      = all_t_xx[int(row.name), wl_idx]
            w    = torch.tensor([length, width], dtype=torch.float32)
            cond = torch.tensor([wl],            dtype=torch.float32)
            y    = torch.tensor([t.real, t.imag], dtype=torch.float32)
            return w, cond, y

    train_dataset = _RCWADataset(train_indices)
    valid_dataset = _RCWADataset(valid_indices)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True,  num_workers=0, pin_memory=True)
    valid_loader  = DataLoader(valid_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)

    # ---- Model ----
    model = PhysicsSurrogateModel(
        w_dim=w_dim,
        cond_dim=cond_dim,
        hidden_dims=hidden_dims,
        norm_bounds=norm_bounds,
    ).to(device)

    mse       = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    # ---- Training loop ----
    best_valid_loss = float("inf")
    best_epoch      = 0
    history         = {"train": [], "valid": [], "lr": []}
    total_start     = time.time()

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for w_batch, cond_batch, y_batch in train_loader:
            w_batch    = w_batch.to(device)
            cond_batch = cond_batch.to(device)
            y_batch    = y_batch.to(device)

            w_in    = w_batch.unsqueeze(0)
            cond_in = cond_batch.unsqueeze(0)
            pred    = model(w_in, cond_in).squeeze(0)

            value_loss  = mse(pred, y_batch)
            grad_pred   = pred[1:] - pred[:-1]
            grad_target = y_batch[1:] - y_batch[:-1]
            grad_loss   = mse(grad_pred, grad_target)
            loss        = value_loss + grad_loss_alpha * grad_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * w_batch.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for w_batch, cond_batch, y_batch in valid_loader:
                w_batch    = w_batch.to(device)
                cond_batch = cond_batch.to(device)
                y_batch    = y_batch.to(device)
                w_in    = w_batch.unsqueeze(0)
                cond_in = cond_batch.unsqueeze(0)
                pred    = model(w_in, cond_in).squeeze(0)
                value_loss  = mse(pred, y_batch)
                grad_pred   = pred[1:] - pred[:-1]
                grad_target = y_batch[1:] - y_batch[:-1]
                grad_loss   = mse(grad_pred, grad_target)
                loss        = value_loss + grad_loss_alpha * grad_loss
                valid_loss += loss.item() * w_batch.size(0)

        valid_loss /= len(valid_dataset)
        scheduler.step(valid_loss)

        history["train"].append(train_loss)
        history["valid"].append(valid_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch      = epoch
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "train_loss":          train_loss,
                    "valid_loss":          valid_loss,
                    "norm_bounds":         norm_bounds,
                    "hidden_dims":         hidden_dims,
                    "w_dim":               w_dim,
                    "cond_dim":            cond_dim,
                },
                checkpoint_dir / "best_surrogate.pth",
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:4d}/{n_epochs}]  "
                f"Train: {train_loss:.6f}  Valid: {valid_loss:.6f}  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    # Save last checkpoint
    torch.save(
        {
            "epoch":               n_epochs - 1,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "train_loss":          history["train"][-1],
            "valid_loss":          history["valid"][-1],
            "norm_bounds":         norm_bounds,
            "hidden_dims":         hidden_dims,
            "w_dim":               w_dim,
            "cond_dim":            cond_dim,
        },
        checkpoint_dir / "last_surrogate.pth",
    )

    history_path = str(checkpoint_dir / "training_history.npy")
    np.save(history_path, history)

    elapsed = time.time() - total_start

    return {
        "best_checkpoint_path": str(checkpoint_dir / "best_surrogate.pth"),
        "last_checkpoint_path": str(checkpoint_dir / "last_surrogate.pth"),
        "history_path":         history_path,
        "best_valid_loss":      float(best_valid_loss),
        "best_epoch":           int(best_epoch),
        "elapsed_sec":          round(elapsed, 2),
    }


# ==============================================================================
# 3. initialize_canvas
# ==============================================================================

def initialize_canvas(
    canvas_config: Dict[str, Any],
    surrogate_checkpoint_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Build the metalens system — load and freeze the trained surrogate,
    construct the SIREN parameter field, and configure the lens and sensor
    surfaces. Save the initial canvas state to disk.

    Parameters
    ----------
    canvas_config : dict
        Lens and parameter field configuration:
          - "aperture_radius_m"    : float — lens aperture radius [m] (default 5e-4)
          - "focal_length_m"       : float — target focal length / sensor z [m] (default 2e-3)
          - "n_in"                 : float — refractive index of incident medium (default 1.5)
          - "n_out"                : float — refractive index of exit medium (default 1.0)
          - "cond_dim"             : int   — condition vector size fed to surrogate (default 1)
          - "ray_grid_nx"          : int   — ray bundle grid size in x (default 15)
          - "ray_grid_ny"          : int   — ray bundle grid size in y (default 15)
          - "ray_wavelength_m"     : float — wavelength of ray bundle [m] (default 532e-9)
          - "siren_hidden_dims"    : list[int] — SIREN hidden layer sizes (default [64,64,64])
          - "w_dim"                : int   — nanostructure parameter dim (default 2)

    surrogate_checkpoint_path : str
        Path to the trained surrogate checkpoint (best_surrogate.pth)
        produced by train_surrogate_model().

    output_dir : str
        Directory where the canvas state files will be saved.

    Returns
    -------
    dict
        - "canvas_path"        : str — path to canvas_state.json (describes the canvas)
        - "param_field_path"   : str — path to param_field_init.pth (SIREN weights)
        - "canvas_config"      : dict — the resolved canvas configuration
        - "n_param_field_params": int — number of trainable parameters in the SIREN field
    """
    from metalens_tracing.networks import PhysicsSurrogateModel, SurrogateWrapper, ParamNet, MetaLensConfig as NetConfig
    from metalens_tracing.param_field import NeuralParameterField
    from metalens_tracing.metalens import Metalens, MetalensConfig
    from metalens_tracing.surfaces import PlaneSurface, CircularAperture

    # ---- Unpack config ----
    aperture_radius  = float(canvas_config.get("aperture_radius_m",  5e-4))
    focal_length     = float(canvas_config.get("focal_length_m",     2e-3))
    n_in             = float(canvas_config.get("n_in",               1.5))
    n_out            = float(canvas_config.get("n_out",              1.0))
    cond_dim         = int(canvas_config.get("cond_dim",             1))
    ray_grid_nx      = int(canvas_config.get("ray_grid_nx",          15))
    ray_grid_ny      = int(canvas_config.get("ray_grid_ny",          15))
    ray_wavelength   = float(canvas_config.get("ray_wavelength_m",   532e-9))
    siren_hidden     = list(canvas_config.get("siren_hidden_dims",   [64, 64, 64]))
    w_dim            = int(canvas_config.get("w_dim",                2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load and freeze surrogate ----
    surrogate = PhysicsSurrogateModel.from_checkpoint(
        surrogate_checkpoint_path, device=device
    )
    surrogate = SurrogateWrapper(surrogate)
    for p in surrogate.parameters():
        p.requires_grad_(False)

    # ---- SIREN parameter field ----
    net_cfg = NetConfig(
        coord_dim=2,
        w_dim=w_dim,
        cond_dim=cond_dim,
        hidden_dims_param=siren_hidden,
    )

    class _NormalizedParamNet(nn.Module):
        """Normalizes lens coords from meters to [-1, 1] before the SIREN."""
        def __init__(self, rep_net: nn.Module, radius: float):
            super().__init__()
            self.rep_net = rep_net
            self.radius  = radius

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            return self.rep_net(coords / self.radius)

    param_net   = ParamNet(net_cfg).to(device)
    param_field = NeuralParameterField(
        _NormalizedParamNet(param_net, aperture_radius)
    ).to(device)

    n_param_field_params = sum(
        p.numel() for p in param_field.parameters() if p.requires_grad
    )

    # ---- Lens and sensor surfaces ----
    aperture     = CircularAperture(radius=aperture_radius)
    lens_surface = PlaneSurface(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        aperture=aperture,
    )

    ml_cfg   = MetalensConfig(cond_dim=cond_dim, n_in=n_in, n_out=n_out)
    metalens = Metalens(
        surface=lens_surface,
        param_field=param_field,
        surrogate_net=surrogate,
        cfg=ml_cfg,
    ).to(device)

    # ---- Save param field initial weights ----
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    param_field_path = str(output_dir_path / "param_field_init.pth")
    torch.save(param_field.state_dict(), param_field_path)

    # ---- Save canvas state descriptor ----
    resolved_config = {
        "aperture_radius_m":    aperture_radius,
        "focal_length_m":       focal_length,
        "n_in":                 n_in,
        "n_out":                n_out,
        "cond_dim":             cond_dim,
        "ray_grid_nx":          ray_grid_nx,
        "ray_grid_ny":          ray_grid_ny,
        "ray_wavelength_m":     ray_wavelength,
        "siren_hidden_dims":    siren_hidden,
        "w_dim":                w_dim,
    }

    canvas_state = {
        "canvas_config":             resolved_config,
        "surrogate_checkpoint_path": surrogate_checkpoint_path,
        "param_field_path":          param_field_path,
        "n_param_field_params":      n_param_field_params,
    }

    canvas_path = str(output_dir_path / "canvas_state.json")
    with open(canvas_path, "w") as f:
        json.dump(canvas_state, f, indent=2)

    return {
        "canvas_path":          canvas_path,
        "param_field_path":     param_field_path,
        "canvas_config":        resolved_config,
        "n_param_field_params": n_param_field_params,
    }


# ==============================================================================
# 4. optimize_metalens
# ==============================================================================

def optimize_metalens(
    optim_config: Dict[str, Any],
    canvas_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Run the differentiable ray-tracing optimization loop to minimize
    the focal spot-size loss and save the optimized canvas state.

    Parameters
    ----------
    optim_config : dict
        Optimization hyperparameters:
          - "lr"                 : float — Adam learning rate (default 4e-3)
          - "n_steps"            : int   — number of optimization steps (default 2000)
          - "scheduler_patience" : int   — ReduceLROnPlateau patience (default 100)
          - "scheduler_factor"   : float — LR reduction factor (default 0.5)
          - "grad_clip_norm"     : float — max gradient norm for clipping (default 0.1)
          - "log_every"          : int   — print frequency in steps (default 10)

    canvas_path : str
        Path to canvas_state.json produced by initialize_canvas()
        (or a previous optimize_metalens() call — supports warm-starting).

    output_dir : str
        Directory where the optimized canvas state will be saved.

    Returns
    -------
    dict
        - "canvas_path"         : str        — path to the updated canvas_state.json
        - "param_field_path"    : str        — path to param_field_optimized.pth
        - "initial_loss"        : float
        - "final_loss"          : float
        - "loss_history"        : list[float]
        - "elapsed_sec"         : float
    """
    from metalens_tracing.networks import PhysicsSurrogateModel, SurrogateWrapper, ParamNet, MetaLensConfig as NetConfig
    from metalens_tracing.param_field import NeuralParameterField
    from metalens_tracing.metalens import Metalens, MetalensConfig
    from metalens_tracing.surfaces import PlaneSurface, CircularAperture
    from metalens_tracing.rays import RayBundle

    # ---- Load canvas state ----
    with open(canvas_path) as f:
        canvas_state = json.load(f)

    cfg                       = canvas_state["canvas_config"]
    surrogate_checkpoint_path = canvas_state["surrogate_checkpoint_path"]
    saved_param_field_path    = canvas_state["param_field_path"]

    aperture_radius = float(cfg["aperture_radius_m"])
    focal_length    = float(cfg["focal_length_m"])
    n_in            = float(cfg["n_in"])
    n_out           = float(cfg["n_out"])
    cond_dim        = int(cfg["cond_dim"])
    ray_grid_nx     = int(cfg["ray_grid_nx"])
    ray_grid_ny     = int(cfg["ray_grid_ny"])
    ray_wavelength  = float(cfg["ray_wavelength_m"])
    siren_hidden    = list(cfg["siren_hidden_dims"])
    w_dim           = int(cfg["w_dim"])

    # ---- Unpack optim config ----
    lr                 = float(optim_config.get("lr",                 4e-3))
    n_steps            = int(optim_config.get("n_steps",              2000))
    scheduler_patience = int(optim_config.get("scheduler_patience",   100))
    scheduler_factor   = float(optim_config.get("scheduler_factor",   0.5))
    grad_clip_norm     = float(optim_config.get("grad_clip_norm",     0.1))
    log_every          = int(optim_config.get("log_every",            10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Rebuild surrogate (frozen) ----
    surrogate = PhysicsSurrogateModel.from_checkpoint(
        surrogate_checkpoint_path, device=device
    )
    surrogate = SurrogateWrapper(surrogate)
    for p in surrogate.parameters():
        p.requires_grad_(False)

    # ---- Rebuild param field and load saved weights ----
    net_cfg = NetConfig(
        coord_dim=2,
        w_dim=w_dim,
        cond_dim=cond_dim,
        hidden_dims_param=siren_hidden,
    )

    class _NormalizedParamNet(nn.Module):
        def __init__(self, rep_net: nn.Module, radius: float):
            super().__init__()
            self.rep_net = rep_net
            self.radius  = radius

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            return self.rep_net(coords / self.radius)

    param_net   = ParamNet(net_cfg).to(device)
    param_field = NeuralParameterField(
        _NormalizedParamNet(param_net, aperture_radius)
    ).to(device)
    param_field.load_state_dict(
        torch.load(saved_param_field_path, map_location=device, weights_only=True)
    )

    # ---- Rebuild metalens ----
    aperture     = CircularAperture(radius=aperture_radius)
    lens_surface = PlaneSurface(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        aperture=aperture,
    )
    ml_cfg   = MetalensConfig(cond_dim=cond_dim, n_in=n_in, n_out=n_out)
    metalens = Metalens(
        surface=lens_surface,
        param_field=param_field,
        surrogate_net=surrogate,
        cfg=ml_cfg,
    ).to(device)

    # ---- Build ray bundle ----
    N = ray_grid_nx * ray_grid_ny
    z0 = -focal_length

    x = torch.linspace(-aperture_radius * 0.7, aperture_radius * 0.7,
                       ray_grid_nx, device=device)
    y = torch.linspace(-aperture_radius * 0.7, aperture_radius * 0.7,
                       ray_grid_ny, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.full_like(xx, z0)

    origins     = torch.stack([xx, yy, zz], dim=-1).reshape(1, N, 3)
    directions  = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3).expand_as(origins)
    wavelengths = torch.full((1, N), ray_wavelength, device=device,
                             dtype=torch.get_default_dtype())
    fields      = torch.zeros(1, N, 3, device=device,
                              dtype=torch.get_default_dtype())
    fields[..., 0] = 1.0  # x-polarized

    rays_in = RayBundle(
        origins=origins, directions=directions,
        fields=fields, wavelengths=wavelengths,
    )

    # ---- Sensor plane ----
    sensor_aperture = CircularAperture(radius=aperture_radius * 10)
    sensor_plane    = PlaneSurface(
        origin=(0.0, 0.0, focal_length),
        normal=(0.0, 0.0, 1.0),
        aperture=sensor_aperture,
    )

    # ---- Loss function ----
    def forward_loss(require_higher_order: bool = False) -> torch.Tensor:
        phase_data = metalens.evaluate_phase_on_rays(
            rays_in,
            require_higher_order=require_higher_order,
            grad_method="autograd",
        )
        rays_hit         = phase_data["rays_hit"]
        phase_grad_world = phase_data["phase_grad_world"]

        rays_out = metalens.diffract_rays(
            rays_hit,
            phase_grad_world=phase_grad_world,
            drop_invalid=True,
        )

        if rays_out.origins.shape[0] == 0:
            return torch.tensor(1.0, device=device,
                                dtype=rays_in.origins.dtype)

        sensor_z  = float(sensor_plane.origin[2].item())
        o = rays_out.origins
        d = rays_out.directions
        t = (sensor_z - o[..., 2]) / d[..., 2].clamp(min=1e-9)
        pts_sensor = o + t.unsqueeze(-1) * d

        sensor_xy = pts_sensor[..., :2] * 1e3   # meters → mm
        return (sensor_xy ** 2).mean()

    # ---- Optimizer ----
    optimizer = optim.Adam(metalens.param_field.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    # ---- Initial loss ----
    initial_loss = forward_loss(require_higher_order=False).item()
    print(f"Initial loss: {initial_loss:.6e}")

    # ---- Optimization loop ----
    loss_history = []
    total_start  = time.time()

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss(require_higher_order=True)
        loss.backward()

        nn.utils.clip_grad_norm_(metalens.param_field.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step(loss.item())

        loss_history.append(loss.item())

        if (step + 1) % log_every == 0 or step == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Step {step+1:4d}/{n_steps} | loss = {loss.item():.4e} "
                  f"| lr = {current_lr:.4e}")

    final_loss = forward_loss(require_higher_order=False).item()
    print(f"Final loss: {final_loss:.6e}")

    elapsed = time.time() - total_start

    # ---- Save optimized param field ----
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    param_field_path = str(output_dir_path / "param_field_optimized.pth")
    torch.save(param_field.state_dict(), param_field_path)

    # ---- Save updated canvas state ----
    updated_canvas_state = {
        "canvas_config":             cfg,
        "surrogate_checkpoint_path": surrogate_checkpoint_path,
        "param_field_path":          param_field_path,
        "n_param_field_params":      canvas_state.get("n_param_field_params"),
    }
    canvas_out_path = str(output_dir_path / "canvas_state.json")
    with open(canvas_out_path, "w") as f:
        json.dump(updated_canvas_state, f, indent=2)

    return {
        "canvas_path":      canvas_out_path,
        "param_field_path": param_field_path,
        "initial_loss":     float(initial_loss),
        "final_loss":       float(final_loss),
        "loss_history":     [float(v) for v in loss_history],
        "elapsed_sec":      round(elapsed, 2),
    }
