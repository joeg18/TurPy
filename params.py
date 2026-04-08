import torch

# this function returns the user-defined dictionary
def make_params():
    params = dict()  # construct the dictionary

    # OPTICAL PARAMS

    # System definition
    params["wavelength"] = 1064e-9  # wavelength in m
    params["n"] = 1.0  # index of refraction at wavelength
    params["k"] = 2 * torch.pi * params["n"] / params["wavelength"]

    # SIMULATION PARAMETERS

    # Sim definition
    params["sim_type"] = "2D"  # "1D" or "2D"
    params["prop_type"] = "coherent"  # "coherent", "incoherent"
    params["field_size"] = (2**10, 2**10)  # Keep as base 2, if 1D only FIRST value is used (organized W, H)
    params["dx"] = 0.00002  # grid size in m (assume symmetric discretization)
    params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    params["forward_delay"] = 0.001 # assumed period of time between instances of forward method #NOTE: this parameter only matters if using correlated phase screens
    params["decorr_time"] = 0.001

    # Phase screen modification
    params["strong_mode"] = False  # Use log amplitude screens (True) or just phase screens (False)
    params["chi"] = 0.25  # 0.25 for plane waves
    params["screen_evolution"] = "static"  # temporal - autoregression used to evolve phase screens. static - new phase screens created each forward method
    params["subharmonics"] = True  # Add subharmonics to phase screen (True) or not (False)
    params["p"] = 3  # level of subharmonics to inject (assume 3x3 sub-discretization per level and ignores DCs)
    
    # Absorption boundary definition
    params["absorb_boundary"] = True  # will define a linearlly decreasing absorbing boundary to reduce edge artifacts (linear minimizes artifacts)
    params["boundary_frac"] = 0.3  # size of total array length that will contribute to absorbing boundary (on each side)
    # params["device"] = "cpu" 

    # turbulence parameters
    params["wind_vec"] = [1, 1]  # Wind speed in m/s #NOTE: 1D uses first value, 2D uses both values for x,y speed respectively
    params["psd"] = "von_karman"  # "von_karman" - uses modified von karman profile with L0, lo, "custom" - uses user-set custom psd mapping in helpers
    params["l0"] = 0  # Inner scale in m
    params["L0"] = torch.inf  # Outer scale in m

    return params