import random
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt

from Helpers import (
    fft, fft2, ifft, ifft2,
    coherent_step_1D, coherent_step_2D,
    modified_von_karman, custom_psd
)

################################################################################
# STATIC PHASE SCREEN
################################################################################

class StaticPhaseScreen:
    """
    Stateless phase screen generator.
    Single source of truth for spatial statistics.
    """

    def __init__(
        self, *,
        device,
        sim_type,
        fR, fX, fY,
        xx, yy,
        dfx, dfy,
        psd_fn,
        L0, l0,
        subharmonics=False,
        p=3,
    ):
        self.device = device
        self.sim_type = sim_type
        self.fR, self.fX, self.fY = fR, fX, fY
        self.xx, self.yy = xx, yy
        self.dfx, self.dfy = dfx, dfy
        self.psd = psd_fn
        self.L0, self.l0 = L0, l0
        self.subharmonics = subharmonics
        self.p = p

    def _complex_white_noise(self, shape, generator):
        return (
            torch.randn(shape, generator=generator, device=self.device, dtype=torch.cfloat)
            + 1j * torch.randn(shape, generator=generator, device=self.device, dtype=torch.cfloat)
        )

    def _draw_fourier_screen(self, r0, generator):
        psd = self.psd(r0, self.fR, self.L0, self.l0)
        psd[self.fR == 0] = 0

        if self.sim_type == "1d":
            return torch.sqrt(psd * self.dfx) * \
                   self._complex_white_noise((self.fX.numel(),), generator)
        else:
            return torch.sqrt(psd * self.dfx * self.dfy) * \
                   self._complex_white_noise(self.fX.shape, generator)
        
    def _fft(self, screen_f):
        return fft(screen_f) if self.sim_type == "1d" else fft2(screen_f)

    def _ifft(self, screen_f):
        return ifft(screen_f) if self.sim_type == "1d" else ifft2(screen_f)

    def _subharmonics_term(self, r0, generator):
        if not self.subharmonics:
            return 0.0

        SH = 0.0
        for p in range(1, self.p + 1):
            scale = 3 ** p

            if self.sim_type == "1d":
                fx = torch.tensor(
                    [-self.dfx / scale, self.dfx / scale],
                    device=self.device
                )[:, None]

                fr = torch.abs(fx)
                psd = torch.sqrt(self.psd(r0, fr, self.L0, self.l0) * self.dfx / scale)
                coeffs = self._complex_white_noise((2, 1), generator)

                SH += (
                    psd * coeffs *
                    torch.exp(1j * 2 * torch.pi * fx * self.xx)
                ).sum(0) / fx.numel()

            else:
                vals = torch.tensor(
                    [-self.dfx / scale, 0, self.dfx / scale],
                    device=self.device
                )
                fx, fy = torch.meshgrid(vals, vals, indexing="ij")
                mask = ~(fx == 0) | ~(fy == 0)
                fx, fy = fx[mask][:, None, None], fy[mask][:, None, None]
                fr = torch.sqrt(fx**2 + fy**2)

                psd = torch.sqrt(
                    self.psd(r0, fr, self.L0, self.l0)
                    * self.dfx / scale * self.dfy / scale
                )
                coeffs = self._complex_white_noise((fx.shape[0], 1, 1), generator)

                SH += (
                    psd * coeffs *
                    torch.exp(1j * 2 * torch.pi * (fx * self.xx + fy * self.yy))
                ).sum(0) / fx.numel()**2

        return SH.real

    @torch.no_grad()
    def sample(self, r0, seed=None):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed if seed is not None else random.randint(1, 100000))
        

        screen_f = self._draw_fourier_screen(r0, gen)
        screen = self._ifft(screen_f).real
        screen += self._subharmonics_term(r0, gen)

        return screen - screen.mean()


################################################################################
# TEMPORAL PHASE SCREEN (BUILT ON STATIC)
################################################################################

class TemporalPhaseScreen(StaticPhaseScreen):
    """
    Stateful phase screen with frozen-flow advection and AR(1) temporal update.
    """

    def __init__(
        self, *,
        alpha=0.9,
        wind=(0.0, 0.0),
        dt=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.alpha = torch.tensor(alpha)
        self.wind = wind
        self.dt = dt
        self.var_mod = torch.sqrt(torch.tensor((1 / 1-self.alpha) / (1+self.alpha)))

        self._rng = torch.Generator(device=self.device)
        self._screen_f = None

    def reset(self, seed=None):
        self._rng.manual_seed(seed if seed is not None else random.randint(1, 100000))
        self._screen_f = None

    @torch.no_grad()
    def step(self, r0):
        """
        Advance the screen by one temporal step.
        """
        if self._screen_f is None:
            self._screen_f = self._draw_fourier_screen(r0, self._rng)
            return self.synthesize(r0)
        
        if self.sim_type == "1d":
            adv = torch.exp(-2j * torch.pi * self.dt * self.wind[0] * self.fX)
        else:
            adv = torch.exp(-2j * torch.pi * self.dt *(self.wind[0] * self.fX + self.wind[1] * self.fY))

        # Check if new phase screen needed to merge
        if self.alpha < 1.0:
            noise = self._draw_fourier_screen(r0, self._rng)
            self._screen_f.mul_(self.alpha).add_((1 - self.alpha) * noise)
            #self._screen_f.mul_(self.var_mod) # variance stabilization term
        
        # Translate screene
        self._screen_f.mul_(adv)
        
        # synthsize & return
        return self.synthesize(r0)

    def synthesize(self, r0):
        screen = self._ifft(self._screen_f).real
        if (self.alpha < 1.0) & self.subharmonics:
            screen += self._subharmonics_term(r0, self._rng).mul_(1-self.alpha) # If alpha, need to restore subharmonics appropriately
        return screen - screen.mean()
    

################################################################################
#Log-Amplitude PHASE SCREEN (Strong Turbulence Amplitude modulations)
################################################################################
class LogAmplitudePhaseScreen(StaticPhaseScreen):
    """
    Log-Amplitude (chi) phase screen generator for strong turbulence regime.
    Builds directly on top of static phase screen statistics
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    # Convert phase screen into log amp variation

    def sample_logamp(self, r0, seed=None):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed if seed is not None else random.randint(1, 100000))

        phi_phi = modified_von_karman(r0, self.fR, self.L0, self.l0)  # Get modified von karmon spectrum
        phi_chi = phi_phi * self.fR**2  # get cross spectrum
        if self.sim_type == "1d":
            noise = self._complex_white_noise((self.fX.numel(),), generator=gen)
        else:
            noise = self._complex_white_noise(self.fX.shape, generator=gen)
        chi = self._ifft(torch.sqrt(phi_chi) * noise).real
        return chi - chi.mean()


################################################################################
# WAVETORCH MODULE
################################################################################

class TurPy(nn.Module):

    def __init__(self, params):
        super().__init__()

        # Assign attributes
        self.params = params
        self.device = params["device"]
        self.sim_type = params["sim_type"].lower()
        self.screen_mode = params["screen_evolution"].lower()
        self.boundary_frac = params["boundary_frac"]
        self.dx = params["dx"]
        self.W, self.H = params["field_size"]
        self.dfx = 1 / (self.dx * self.W)
        self.dfy = 1 / (self.dx * self.H)
        # Check for strong phase screens
        self.strong_mode = self.params.get("strong_mode", False)

        # Setup PSD
        self.psd = modified_von_karman if params["psd"] != "custom" else custom_psd

        # Build grids
        self._build_grids()
        self._build_propagator()
        self._build_phase_screen()
        self._build_boundary()


    def _build_grids(self):
        x = torch.linspace(
            -(self.W // 2),
            self.W // 2 - 1,
            self.W,
            device=self.device
        ) * self.dx

        fx = torch.linspace(
            -(self.W // 2),
            self.W // 2 - 1,
            self.W,
            device=self.device
        ) * self.dfx

        if self.sim_type == "1d":
            self.xx = x
            self.yy = None
            self.fX = fx
            self.fY = None
            self.fR = torch.abs(fx)
        else:
            self.xx, self.yy = torch.meshgrid(x, x, indexing="ij")
            self.fX, self.fY = torch.meshgrid(fx, fx, indexing="ij")
            self.fR = torch.sqrt(self.fX**2 + self.fY**2)

        # Preallocate fresnel TF phase term for speed
        self.sqrt_term = (
            torch.pi * self.params["wavelength"] / self.params["n"]
        ) * self.fR**2

    def _build_propagator(self):
        self.prop_step = coherent_step_1D if self.sim_type == "1d" else coherent_step_2D

    def _build_phase_screen(self):
        base_kwargs = dict(
            device=self.device,
            sim_type=self.sim_type,
            fR=self.fR,
            fX=self.fX,
            fY=self.fY,
            xx=self.xx,
            yy=self.yy,
            dfx=self.dfx,
            dfy=self.dfy,
            psd_fn=self.psd,
            L0=self.params["L0"],
            l0=self.params["l0"],
            subharmonics=self.params["subharmonics"],
            p=self.params["p"],
        )
        
        if self.screen_mode == "temporal":
            self.phase_screen = TemporalPhaseScreen(
                alpha=self.params.get("alpha", 0.9),
                wind=self.params["wind_vec"],
                dt=self.params["forward_delay"],
                **base_kwargs
            )
        else:
            self.phase_screen = StaticPhaseScreen(**base_kwargs)

        if self.strong_mode:
            self.logamp_screen = LogAmplitudePhaseScreen(**base_kwargs)

    def _build_boundary(self):
        if not self.params["absorb_boundary"]:
            self.boundary = None
            return
        
        if self.params["sim_type"].lower() == "1d":
            
            # Calculate the boundary length
            boundary_len = int(self.boundary_frac * self.W)
            boundary_len = max(1, boundary_len)
            
            # Create base boundary full of ones
            boundary = torch.ones(self.W)
            
            # Create decay factors
            decay = torch.linspace(1, 0, steps=boundary_len)
            
            # Apply decay factors
            boundary[:boundary_len] *= decay.flip(0)
            boundary[-boundary_len:] *= decay

            # assign the boundary
            self.boundary = boundary.to(self.device).detach()  # since static, can detach

        else:
            # Calculate boundary lengths
            boundary_len_W = int(self.boundary_frac * self.W)
            boundary_len_H = int(self.boundary_frac * self.H)
            
            # Ensure there is at least one point to decay
            boundary_len_W = max(1, boundary_len_W)
            boundary_len_H = max(1, boundary_len_H)
            
            # Create base boundary full of ones
            boundary = torch.ones((self.W, self.H))
            
            # Create decay factors for the boundaries
            decay_W = torch.linspace(1, 0, steps=boundary_len_W)
            decay_H = torch.linspace(1, 0, steps=boundary_len_H)
            
            # Apply decay factors to the boundary
            boundary[:boundary_len_W, :] *= decay_W.flip(0).unsqueeze(1)
            boundary[-boundary_len_W:, :] *= decay_W.unsqueeze(1)
            boundary[:, :boundary_len_H] *= decay_H.flip(0)
            boundary[:, -boundary_len_H:] *= decay_H

            self.boundary = boundary.to(self.device).detach()

    def forward(self, field, dr, r0=None):
        for i, dz in enumerate(dr):
            field = self.prop_step(field, torch.exp(1j * dz * self.sqrt_term))

            if r0 is not None:
                if self.screen_mode == "static":
                    phase = self.phase_screen.sample(r0[i])
                else:
                    phase = self.phase_screen.step(r0[i])

                if self.strong_mode:
                    chi = self.logamp_screen.sample_logamp(r0[i])
                    field = field * torch.exp(chi + 1j * phase)
                else:
                    field = field * torch.exp(1j * phase)
            
            if self.boundary is not None:
                field = field * self.boundary

        return field
