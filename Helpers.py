import torch
import math
import numpy as np
from typing import List
from torchinterp1d import interp1d

############# FFT Shorthands ############# 
# Build fft shorthand with proper shifts & scaling
#@torch.jit.script
def fft(x:torch.Tensor, n=None, dim=-1)->torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) / x.numel()

#@torch.jit.script
def ifft(x:torch.Tensor, n=None, dim=-1)->torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) * x.numel() # approrpriate normalization

#@torch.jit.script
def fft2(x:torch.Tensor, dim:List[int]=(-2, -1))->torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) / x.numel()

#@torch.jit.script
def ifft2(x:torch.Tensor, dim:List[int]=(-2, -1))->torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) * x.numel() # approrpriate normalization

#@torch.jit.script
@torch.no_grad
def autocorrelation_1D(H: torch.Tensor)->torch.Tensor: 
    return torch.fft.fftshift(torch.fft.ifft(
            torch.square(torch.abs(torch.fft.fft(torch.fft.ifftshift(H))))))  # remove unnecessary shifts

#@torch.jit.script
@torch.no_grad
def autocorrelation_2D(H: torch.Tensor)->torch.Tensor: 
    return torch.fft.fftshift(torch.fft.ifft2(
            torch.square(torch.abs(torch.fft.fft2(torch.fft.ifftshift(H))))))


############# Propagators ############# 
#@torch.jit.script
@torch.no_grad
def H_angular(fR: torch.Tensor, wavelength: torch.Tensor, dz: torch.Tensor, n: torch.Tensor)->torch.Tensor:
    # return torch.exp(1j * 2*torch.pi*n/wavelength * dz * torch.sqrt(1 - (wavelength * fR / n) ** 2))
    return torch.exp(-1j * torch.pi * wavelength/n * dz * (fR**2))

@torch.no_grad
def H_fraun(fR: torch.Tensor, wavelength: torch.Tensor, dz: torch.Tensor, n: torch.Tensor)->torch.Tensor:
    return torch.exp(-1j * torch.pi * wavelength/n * dz * (fR**2))

# NOTE COHERENT STEP: Assume input field is COMPLEX AMPLITUDE
#@torch.jit.script
def coherent_step_1D(field: torch.Tensor, H: torch.Tensor)->torch.Tensor:
    return ifft(torch.mul(fft(field), H))
  
# NOTE INCOHERENT STEP: Assume input field is INTENSITY (TODO: Incoherent equivalent of split-step below)
#@torch.jit.script
def incoherent_step_1D(field: torch.Tensor, H: torch.Tensor)->torch.Tensor:
    return ifft(torch.mul(fft(field), autocorrelation_1D(H)))

# NOTE: Split-step inherently assumes coherence to deduce the output field
#@torch.jit.script 
def split_step_1D(field: torch.Tensor, H: torch.Tensor, phase: torch.Tensor)->torch.Tensor:
    return torch.mul(coherent_step_1D(field, H), phase)

# NOTE COHERENT STEP: Assume input field is COMPLEX AMPLITUDE
#@torch.jit.script
def coherent_step_2D(field: torch.Tensor, H: torch.Tensor)->torch.Tensor:
    return ifft2(torch.mul(fft2(field), H))
  
# NOTE INCOHERENT STEP: Assume input field is INTENSITY (TODO: Incoherent equivalent of split-step below)
#@torch.jit.script
def incoherent_step_2D(field: torch.Tensor, H: torch.Tensor)->torch.Tensor:
    return ifft2(torch.mul(fft2(field), autocorrelation_2D(H)))

# NOTE: Split-step inherently assumes coherence to deduce the output field
#@torch.jit.script  
def split_step_2D(field: torch.Tensor, H: torch.Tensor, phase: torch.Tensor)->torch.Tensor:
    return torch.mul(coherent_step_2D(field, H), phase)


############# Turbulence Profile Functions ############# 
"""
Generates the round_trip cn2 path variable for a given height - assumes R represents round trip through atmosphere

inputs:
    A: atmospheric constant at the ground
    W: wind speed in RMS
    R: Round Trip Distance array 

"""
def generate_round_trip_hv_model(A, W, R):
    Cn2 = torch.zeros(R.shape)  # setup structured constant variable
    mp = round(len(R)/2+0.01)
    Cn2[0:mp] = 0.00594 * (W / 27) ** 2 * (10 ** -5 * R[0:mp]) ** 10 * torch.exp(-R[0:mp] / 1000) + (2.7 * 10 ** -16) * torch.exp(
        -R[0:mp] / 1500) + A * torch.exp(-R[0:mp] / 100)  # one way forward path
    Cn2[mp:] = 0.00594 * (W / 27) ** 2 * (10 ** -5 * (R[-1] - R[mp:])) ** 10 * torch.exp(-(R[-1] - R[mp:])/ 1000) + (2.7 * 10 ** -16) * torch.exp(
        -(R[-1] - R[mp:]) / 1500) + A * torch.exp(-(R[-1] - R[mp:]) / 100)  # one way return path
    return Cn2

# same but one-way
def generate_oneway_trip_hv_model(A, W, R):
    Cn2 = 0.00594 * (W / 27) ** 2 * (10 ** -5 * R) ** 10 * torch.exp(-R / 1000) + (2.7 * 10 ** -16) * torch.exp(-R / 1500) + A * torch.exp(-R / 100)  # one way forward path
    return Cn2

############# Power Spectral Density Calculations ############# 
""" 
Returns returns a psd function parameterized by:

inputs:
    r0: fried parameter
    fR: radial spatial frequency grid
    L0: outer scale
    l0: inner scale

The von Karman PSD is equivalent to l0 = 0.
The Kolmogorov PSD is equivalent to L0 = 'inf' and l0 = 0. 

For additional details, consult: https://dial.uclouvain.be/downloader/downloader.php?pid=thesis%3A40614&datastream=PDF_01&cover=cover-mem
"""

def modified_von_karman(r0, fR, L0=torch.inf, l0=torch.tensor(0)):
    # First check for special cases:
    # CASE 1: KOLMOGOROV
    if (L0 == torch.inf) & (l0 == torch.tensor(0)):
        return 0.023 * r0**(-5/3) * fR**(-11/3)

    # CASE 2: VON KARMAN
    elif (L0 == torch.inf) & (l0 > torch.tensor(0)):
        f0 = 1 / L0
        return 0.023 * r0**(-5/3) * ((fR**2+f0**2)**(-11/6))
    
    # CASE 3: MODIFIED VON KARMAN
    else:
        f0 = 1 / L0
        fm = 5.92 / (2*torch.pi*l0)
        return 0.023 * r0**(-5/3) * torch.exp(-(fR/fm)**2) * ((fR**2+f0**2)**(-11/6))

# TODO: Add your custom mapping here - NOTE: its MUST include the same arguments as von_karman above even if unused
def custom_psd(r0, fR, L0=torch.inf, l0=torch.tensor(0)):
    return 0.023 * r0**(-5/3) * fR**(-8/3)


############# Params Functions ############# 
""" 
Method to convert a generated Cn2 profile into properly formulated stepsizes and r0 values for use in WaveTorch

inputs:
    params: dictionary of wavetorch parameters for use in calculation
    R: vector of stepsizes of range between minimum and maximum value.
    Cn2: calculated Cn2 profile based on R
    f0: cutoff frequency of the input field
    method: method of calculations, which include:
        "none": returns r0 values based on original input - no internal checks performed
        "anti-alias": anti-alias option enforces each step does not breach aliasing of transfer function nor weak step assumption of turbulence model

"""
def calculate_inst_rytov(params, R, Cn2, R_max=None):
    # Bring to CPU for slicing
    R.to('cpu')
    Cn2.to('cpu')

    # give option to feed in R_max if calculating partial array
    if R_max == None:
         R_max = torch.max(R)
    diff_R = torch.diff(R)
    Cn2_terms = Cn2[1:] * (1 - R[1:] / R_max)**(5/3) + Cn2[:-1] * (1 - R[:-1] / R_max)**(5/3)  # replacement of integral with summation
    return  3.27 * params["k"]**(7/6) * R_max**(5/6) * (diff_R / 2) * Cn2_terms  # Get the inst Rytov   

""" 
Simple linear interpolator for use in Rytov resampling

inputs:
    R1: x-value at step 1
    R2: x-value at step 2
    C1: y-value at step 1
    C2: y-value at step 2
    r: resampling interval

"""

def linear_interpolate(R1, R2, C1, C2, r):
    t = (r - R1) / (R2 - R1)
    return C1 * (1 - t) + C2 * t

""" 
Method to convert a generated Cn2 profile into properly formulated stepsizes and r0 values for use in WaveTorch

inputs:
    params: dictionary of wavetorch parameters for use in calculation
    R: vector of stepsizes of range between minimum and maximum value.
    Cn2: calculated Cn2 profile based on R
    f0: cutoff frequency of the input field
    method: method of calculations, which include:
        "none": returns r0 values based on original input - no internal checks performed
        "anti-alias": anti-alias option enforces each step does not breach aliasing of transfer function nor weak step assumption of turbulence model

"""

def adaptive_rytov_sampling(params, R, Cn2, rytov_limit=0.9, max_range=torch.inf):

    # STEP 0: Generate new arrays to hold resampled values. The first value will always be the same as input (R=0, Cn2[R=0]=A)
    R_new = R[0].unsqueeze(0)
    Cn2_new = Cn2[0].unsqueeze(0)

    # Get initial quantities used for adaptive resampling
    R_new_max = R_new.max()
    R_max = R.max()

    # STEP 1: Calculate Instintaneous Rytov Curve for Sim
    rytov = calculate_inst_rytov(params, R, Cn2)  # In this form, Rytov is cumulative (i.e., sum(rytov) = true integral over entire range)
    const = 3.28 * params["k"]**(7/6) * R_max**(5/6)  # constant used in rytov formula

    # STEP 2: Keep cycling through array until new max is achieved
    r = 0 # true index of original arrays
    j = 1 # psuedo-index used for interpolated arrays

    # Sweep through until my new range array reaches the desired distance
    while R_new_max < R_max:

        # Get rytov at point
        inst_rytov = rytov[r]

        # 2a: IF WE ARE BIGGER THAN OUR DESIRED RYTOV LIMIT - INTERPOLATE BETWEEN POINTS
        if inst_rytov > rytov_limit:
            # print('a')
            # Setup necessary quantities
            rytov_test = inst_rytov
            interval = 1  # how many intervals to break into

            # Loop until my interpolation gets the interval small enough
            while rytov_test > rytov_limit:
                interval += 1
                # NOTE: for ease we perform a linear approximation in the local neighborhood between points of Cn2
                test_R = torch.linspace(R[r], R[r+1], interval)
                test_Cn2 = torch.linspace(Cn2[r], Cn2[r+1], interval)

                # Try Rytov
                rytov_test = calculate_inst_rytov(params, test_R, test_Cn2, R_max=R_max).max()

            # If we escaped, we found the correct interval that no rytov variance exceeds our desired value
            R_new = torch.cat((R_new, test_R[1:]))
            Cn2_new = torch.cat((Cn2_new, test_Cn2[1:]))
            j += interval
            r += 1
            R_new_max = R_new.max()
        
        # 2b: IF WE ARE SMALLER THAN OUR DESIRED RYTOV LIMIT - WE MAY POTENTIALLY COMPRESS POINTS
        elif inst_rytov < rytov_limit:
            rytov_test = inst_rytov
            tmp = 0  # tmp index to see how many steps we need to take

            # Loop until I summed enough points that I get to the desired point
            while rytov_test < rytov_limit:
                    tmp += 1  # progress to next step to see if still within interval

                    # BREAK CONDITION 1: If attempt to access array beyond max index
                    if r + tmp == len(R):
                         tmp = len(R) - r
                         rytov_test = torch.inf
                    # BREAK CONDITION 2: Exceed desired max range step
                    elif  R[r+tmp] - R[r] > max_range:
                         rytov_test = torch.inf
                    else:
                        # try rytov
                        rytov_test = const * (R[r+tmp] - R[r])/2 * ((Cn2[r] * (1 - R[r] / R_max)**(5/3) + Cn2[r+tmp] * (1 - R[r+tmp] / R_max)**(5/3)))
                        delta_R = R[r+tmp] - R[r]

            # If we escaped, means we found biggest step we can take before breaking limit
            R_new = torch.cat((R_new, R[r+tmp-1].unsqueeze(0)))
            Cn2_new = torch.cat((Cn2_new, Cn2[r+tmp-1].unsqueeze(0)))
            j += 1
            r += tmp - 1

            R_new_max = R_new.max()
            

        # 2c: If neither, means interval is perfect as is
        else:
             R_new = torch.cat((R_new, R[r].unsqueeze(0)))
             Cn2_new = torch.cat((Cn2_new, Cn2[r].unsqueeze(0)))
             j+=1
             r+=1

             R_new_max = R_new.max()

    return R_new, Cn2_new

""" 
Method to convert a generated Cn2 profile into properly formulated stepsizes and r0 values for use in WaveTorch

inputs:
    params: dictionary of wavetorch parameters for use in calculation
    R: vector of stepsizes of range between minimum and maximum value.
    Cn2: calculated Cn2 profile based on R
    f0: cutoff frequency of the input field
    method: method of calculations, which include:
        "none": returns r0 values based on original input - no internal checks performed
        "anti-alias": anti-alias option enforces each step does not breach aliasing of transfer function nor weak step assumption of turbulence model

"""
def calculate_path(params, R, Cn2=None, f0=None, rytov_limit=0.9, max_step = torch.inf, method="anti-alias"):

    # TODO: Initial sanity checks

    # First, calculate the stepsize by the difference between points
    if method == "none":
        dr = torch.diff(R)
        # Second, calculate the corresponding fried parameter
        if Cn2 == None:
            r0 = None
        else:
            diff_R = torch.diff(R)  # trapezoidal step size
            Cn2_terms = Cn2[1:] + Cn2[:-1]  # proxy integral
            r0 = (0.423*params["k"]**2*diff_R/2*Cn2_terms)**(-3/5) # element-wise trapz approximation for rolling fried parameter
        return dr, r0
    
    else:
        # TODO: Add more than just angular spectrum kernel
        # NOTE: For right now, just assume angular spectrum kernel
        
        # Step 1: check for constrained cutoff
        # If not f0 given, need FULL transfer function. So f0 is max supported by simulation
        if f0 == None:
            f0 = 1/(2*params["dx"])

        # Used to determine Field of View - biggest constraint on largest FoV
        if params["sim_type"].lower() == "1d":
            N = params["field_size"][0]
        else:
            N = max(params["field_size"][0], params["field_size"][1])

        
        # TODO: CALCULATE RYTOV VARIANCE AT EACH STEP
        # TODO: DETERMINE IF ANY STEP IS > 1 (BAD) and MATHEMATICALLY DETERMINE WHAT RESAMPLED STEPSIZE NEEDS TO BE

        # From the above statement, we can calualte the maximum stepsize possible by comparing it to the nyquist condition
        # Stepsize argument modified from: https://opg.optica.org/optica/fulltext.cfm?uri=optica-10-11-1407&id=541154
        step_max = 0.75 * N * params["dx"] / 2 / (params["wavelength"] / params["n"] * f0 / np.sqrt(1-(params["wavelength"] / params["n"] *f0)**2))  # NOTE: still A BIT of aliasing at this full value, add scalar in front to mitigate (can tune as needed)
        print(f'maximum step size support by this simulation is = {step_max} m')

        if step_max < max_step:
            max_step = step_max

        # Step 3: Perform adaptive resampling such that any step does not violate rytov weak approximation nor fourier optics
        R_new, Cn2_new = adaptive_rytov_sampling(params, R, Cn2, rytov_limit=rytov_limit, max_range=max_step)

        # print(R_new, Cn2_new.size())
       
        # Step 4: Get differentials
        dr = torch.diff(R_new)
        Cn2_terms = Cn2_new[1:] + Cn2_new[:-1]  # proxy integral
        r0 = (0.423*params["k"]**2*dr/2*Cn2_terms)**(-3/5) # element-wise trapz approximation for rolling fried parameter

        return dr, r0


""" 
Determines the optical cutoff of an input field for use in calculate_path

inputs:
    params: dictionary of wavetorch parameters for use in calculation
    field: input optical field to propagate

"""
def determine_cutoff(params, field):
    return


############## ENCODER FUNCTIONS ##################

def upsample_1d(mask, ds):
    return mask.repeat_interleave(ds, dim=-1)

def upsample_2d(mask, ds):
    return mask.repeat_interleave(ds, dim=-1).repeat_interleave(ds, dim=-2)

def circular_1d(input_array, field_size):
    """
    Given a 1D input array, generate a mirrored array that is twice the length.

    Parameters:
    input_array (torch.Tensor): 1D tensor of size N
    field_size (tuple): WxH size of grid (1d uses just W)

    Returns:
    torch.Tensor: 1D tensor of size 2N, mirroring the original array
    """
    assert len(input_array) * 2 != field_size[0]  # double check that weights are of right size

    # Create the mirrored array
    return torch.cat([input_array, torch.flip(input_array, dims=[0])])

def circular_2d(input_vector, field_size):
    """
    Given an input vector of size N//2, generate an NxN array where each vector element is assigned to 
    concentric rings in circular geometry around a 2D grid. The first vector entry maps to the innermost ring.
    
    Parameters:
    input_vector (torch.Tensor): 1D tensor of size N//2
    field_size (tuple): WxH size of grid (1d uses just W)

    Returns:
    torch.Tensor: 2D tensor of size NxN, with values mapped according to concentric rings
    """
    assert len(input_array) * 2 == field_size[0], "input_array does not match field size along first dim"  # double check that weights are of right size

    N_half = input_vector.size(0)
    N = 2 * N_half

    # Create an index grid
    idx = torch.arange(N, device=input_vector.device) - N_half + 0.5
    idy = torch.arange(field_size[1], device=input_vector.device) - field_size[1]//2 + 0.5
    grid_x, grid_y = torch.meshgrid(idy, idx, indexing='ij')

    # Calculate the radial distances from the center
    dist_from_center = torch.sqrt(grid_x**2 + grid_y**2)

    # Quantize the distances and clamp them to the range of the input vector indices
    quantized_radii = dist_from_center.floor().long()
    quantized_radii = torch.clamp(quantized_radii, 0, N_half - 1)

    # Map the input vector values to the corresponding quantized radii
    ring_matrix = input_vector[quantized_radii]

    return ring_matrix

def launcher_1d(x, masks):
    """
    Given input masks, perform launching sequence for the optical encoder.
    
    Parameters:
    x (torch.Tensor): 1D input field tensor of size W
    masks (torch.Tensor): 2D phase mask tensor of size num_screens x W

    Returns:
    torch.Tensor: 2D tensor of size NxN, with values mapped according to concentric rings
    """
    return fft(fft(fft(x*masks[0])*masks[1]))

def launcher_2d(x, masks):
    """
    Given input masks, perform launching sequence for the optical encoder.
    
    Parameters:
    x (torch.Tensor): 2D input field tensor of size N
    masks (torch.Tensor): 3D phase mask tensor of size num_screens x W x H

    Returns:
    torch.Tensor: 2D tensor of size NxN, with values mapped according to concentric rings
    """
    return fft2(fft2(fft2(x*masks[0])*masks[1]))

if __name__ == "__main__":
    input_array = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
    field_size = [16, 16]
    output_matrix = circular_2d(input_array, field_size)
    print(output_matrix)