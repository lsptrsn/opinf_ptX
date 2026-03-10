import os
import pickle
import numpy as np
from pathlib import Path
import torch


import os
import pickle
import numpy as np
from pathlib import Path
import torch


def load_results(data_dir: str, decoder_class=None):
    """Loads model components including CNN decoder if available."""
    results = {}
    npz_path = os.path.join(data_dir, "results.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Model file not found: {npz_path}")

    # Load NPZ
    data = np.load(npz_path, allow_pickle=True)

    # Standard Operators
    for key in ["A_OpInf", "B_OpInf", "C_OpInf", "H_OpInf", "basis", "initial_values", "time"]:
        if key in data:
            res_key = "y0" if key == "initial_values" else key
            res_key = "time_arr" if key == "time" else res_key
            results[res_key] = data[key]
            if results[res_key].ndim == 2 and (results[res_key].shape[0] == 1 or results[res_key].shape[1] == 1):
                results[res_key] = results[res_key].flatten()

    # Clean keys
    results["A"] = results.pop("A_OpInf")
    results["B"] = results.pop("B_OpInf")
    results["C"] = results.pop("C_OpInf")
    results["H"] = results.pop("H_OpInf")

    # Dimensions & Scaling
    results["r_F"] = int(data["r_F"])
    results["r_T"] = int(data["r_T"])
    results["scaling_fac_F"] = data["scaling_fac_F"]
    results["scaling_fac_T"] = data["scaling_fac_T"]
    results["reference_states_shifting"] = data["reference_states_shifting"].flatten()

    results["max_F"] = 1.0
    results["max_T"] = 1.0
    if "state_scaling_params" in data:
        ssp = data["state_scaling_params"].item()
        results["max_F"] = float(ssp.get("max_F", 1.0))
        results["max_T"] = float(ssp.get("max_T", 1.0))

    if "input_scaling_factors" in data:
        results["input_scaling_factors"] = data["input_scaling_factors"].flatten()
    else:
        results["input_scaling_factors"] = np.ones(3)

    # NL-POD
    def get_opt(key):
        if key in data:
            val = data[key]
            if val.shape == () and val.item() is None: return None
            return val
        return None
    results["V_reduced_nonlin"] = get_opt("V_reduced_nonlin")
    results["Xi"] = get_opt("Xi")

    # Pickle Data (CNN & Scalers)
    pkl_path = os.path.join(data_dir, "results.pkl")
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        results["basis_type"] = pkl_data.get("basis_type", "POD")

        # Load Scalers
        results["input_scaler"] = pkl_data.get("input_scaler")
        if results["input_scaler"] is None and "input_scaler_F" in pkl_data:
            # Fallback if specific key exists (though typically one scaler for all latent)
            results["input_scaler"] = pkl_data["input_scaler_F"]

        results["target_scaler_F"] = pkl_data.get("target_scaler_F")
        results["target_scaler_T"] = pkl_data.get("target_scaler_T")

        # Load Decoder
        if decoder_class and "decoder_state_dict" in pkl_data:
            try:
                decoder = decoder_class(**pkl_data["decoder_init_params"])
                decoder.load_state_dict(pkl_data["decoder_state_dict"])
                decoder.eval()
                results["decoder"] = decoder
            except Exception as e:
                print(f"Error loading decoder: {e}")
                results["decoder"] = None
    else:
        results["basis_type"] = "POD"
        results["decoder"] = None

    return results


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Metal (M1/M2)
    else:
        return torch.device("cpu")  # Fallback auf CPU


def setup_results_dir(folder_name: str) -> Path:
    """
    Creates (if not existing) a directory for results.
    Can be called directly from the main script.

    Parameters
    ----------
    folder_name : str
        Name of the results directory (e.g. "results").

    Returns
    -------
    str
        Absolute path of the results directory.
    """
    # Get absolute path
    results_path = Path(folder_name).resolve()
    results_path.mkdir(exist_ok=True)
    print(f"Results will be stored in '{results_path}'.")
    return results_path



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Metal (M1/M2)
    else:
        return torch.device("cpu")  # Fallback auf CPU


def setup_results_dir(folder_name: str) -> Path:
    """
    Creates (if not existing) a directory for results.
    Can be called directly from the main script.

    Parameters
    ----------
    folder_name : str
        Name of the results directory (e.g. "results").

    Returns
    -------
    str
        Absolute path of the results directory.
    """
    # Get absolute path
    results_path = Path(folder_name).resolve()
    results_path.mkdir(exist_ok=True)
    print(f"Results will be stored in '{results_path}'.")
    return results_path
