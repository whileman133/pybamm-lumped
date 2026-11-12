"""
libcell.py

PyBaMM partial-differential-equation (PDE) model for LIB cell.

2025.01.27 | Created | Wesley Hileman
2026.09.23 | Optional MSMR OCP/kinetics per electrode
"""
import json
from importlib.resources import files

import numpy as np
import pybamm

from .base import BaseLumpedModel

ELECTRODES = ("neg", "pos")


class LumpedLIBModel(BaseLumpedModel):
    """
    Model for a lithium-ion battery cell with neg, sep, and pos layers.
    Lumped parameters. Dimensionless geometry. Butler-Volmer kinetics.

    Each electrode uses either
      - standard OCP (Uocp lookup table), single-reaction kinetics, and constant Ds
        (the default, neg_J/pos_J=None), or
      - MSMR OCP and kinetics with J galleries and the Baker-Verbrugge
        stoichiometry-dependent Ds (neg_J/pos_J=J). The solid-diffusion equation
        is solved for the particle potential Us instead of thetas.

    MSMR exchange current follows the BMS3 toolbox (genFOM) convention:
        i0_j = k0_j * xss_j^omega_j * exp((1-alpha_j)*f*(Uss-U0_j)) * thetae^(1-alpha_j)
    (LumpedLMBModel uses the same unscaled k0_j.)

    NOTE 1: For simplicity, we set the potential reference point to the electrolyte at the surface
            of the Li-metal electrode, i.e., we define phi_e(0) = 0V. This makes phi_s_neg(0) nonzero,
            and it must be solved for to compute the cell voltage, thus phi_s_neg(0) appears as
            a variable of the model.

    NOTE 2: The double-layer models break when Rdl(pos)=0 or Rdl(neg)=0. Use small values instead.
    """

    def __init__(
        self,
        name="Unnamed lumped-parameter lithium-ion cell model",
        *,
        neg_J: int | None = None,
        pos_J: int | None = None,
    ):
        super().__init__(name)

        # Number of MSMR galleries per electrode (None => standard OCP).
        for e, J in (("neg", neg_J), ("pos", pos_J)):
            if J is not None and J < 1:
                raise ValueError(f"Number of MSMR galleries ({e}_J) must be >= 1 or None. Got {J} instead.")
        self.J = {"neg": neg_J, "pos": pos_J}

        # Initialize model.
        self._init_model_parameters()
        self._init_model_variables()
        self._init_model_equations()
        # self._init_model_events()

        # Collect model output variables.
        self.variables = {
            'time [s]': pybamm.t,
            'time [min]': pybamm.t / 60,
            'time [h]': pybamm.t / 3600,
            'iapp [A]': self.param.iapp,
            'iapp [C-rate]': self.param.iapp / self.param.Q,
            'iapp variable [A]': self.param.iapp,   # for compatability with EISSimulation class
            'pos_soc': self.var.pos_soc,
            'pos_soc [%]': self.var.pos_soc * 100,
            'neg_soc': self.var.neg_soc,
            'neg_soc [%]': self.var.neg_soc * 100,
            'vcell [V]': self.var.vcell,
            'pos_thetas_avg': self.var.pos_thetas_avg,
            'neg_thetas_avg': self.var.neg_thetas_avg,
            'thetae': self.var.thetae,
            'FNe [A]': self.var.FNe,
            'phie [V]': self.var.phie,
            'ie [A]': self.var.ie,
            'pos_thetas': self.var.pos_thetas,
            'neg_thetas': self.var.neg_thetas,
            'thetass': self.var.thetass,
            'pos_thetass': self.var.pos_thetass,
            'neg_thetass': self.var.neg_thetass,
            'neg_Uss [V]': self.var.neg_Uss,
            'pos_Uss [V]': self.var.pos_Uss,
            'phis [V]': self.var.phis,
            'is [A]': self.var.is_,
            'ifdl [A]': self.var.ifdl,
            'pos_if [A]': self.var.pos_if,
            'neg_if [A]': self.var.neg_if,
            'phise [V]': self.var.phise,
            'etas [V]': self.var.etas,
            'neg_phis [V]': self.var.neg_phis,
        }
        for e in ELECTRODES:
            if self.is_msmr(e):
                self.variables[f'{e}_Us [V]'] = self._v(e, "Us")

    def is_msmr(self, e):
        """True if electrode e ('neg' or 'pos') uses MSMR OCP/kinetics."""
        return self.J[e] is not None

    # Per-electrode attribute access: self._p("neg", "k0") is self.param.neg_k0.
    def _p(self, e, name):
        return getattr(self.param, f"{e}_{name}")

    def _v(self, e, name):
        return getattr(self.var, f"{e}_{name}")

    def _set_p(self, e, name, value):
        setattr(self.param, f"{e}_{name}", value)

    def _set_v(self, e, name, value):
        setattr(self.var, f"{e}_{name}", value)

    @property
    def default_geometry(self):
        return {
            "neg": {self.var.x_neg: {"min": 0, "max": 1}},
            "sep": {self.var.x_sep: {"min": 1, "max": 2}},
            "pos": {self.var.x_pos: {"min": 2, "max": 3}},
            "neg_particle": {self.var.r_neg: {"min": 0, "max": 1}},
            "pos_particle": {self.var.r_pos: {"min": 0, "max": 1}},
        }

    @property
    def default_submesh_types(self):
        return {
            "neg": pybamm.Uniform1DSubMesh,
            "sep": pybamm.Uniform1DSubMesh,
            "pos": pybamm.Uniform1DSubMesh,
            "neg_particle": pybamm.Uniform1DSubMesh,
            "pos_particle": pybamm.Uniform1DSubMesh,
        }

    @property
    def default_spatial_methods(self):
        return {
            "neg": pybamm.FiniteVolume(),
            "sep": pybamm.FiniteVolume(),
            "pos": pybamm.FiniteVolume(),
            "neg_particle": pybamm.FiniteVolume(),
            "pos_particle": pybamm.FiniteVolume(),
        }

    @property
    def default_var_pts(self):
        return {
            self.var.x_neg: 20,
            self.var.x_sep: 10,
            self.var.x_pos: 20,
            self.var.r_neg: 50,
            self.var.r_pos: 50,
        }

    @property
    def default_solver(self):
        return pybamm.CasadiSolver(root_tol=1e-3, atol=1e-6, rtol=1e-6)

    @property
    def default_quick_plot_variables(self):
        return [
            "iapp [C-rate]", ["neg_soc [%]", "pos_soc [%]"], "vcell [V]",
            "thetae", "FNe [A]", ["ie [A]", "is [A]"],
            "ifdl [A]", "phie [V]", "thetass",
        ]

    @property
    def default_parameter_values(self):
        if self.is_msmr("neg") or self.is_msmr("pos"):
            raise ValueError(
                "No default parameter values for MSMR electrodes. Build parameter values from an MSMR "
                "cell definition with get_lib_param_values()/get_initial(), or use the lumped.sim_* helpers."
            )
        # Defaults: cellNMC30 at 25 degC and 50 % SOC (scripts/make-lib-defaults.py).
        d = json.loads((files("lumped") / "lib-defaults.json").read_text())
        param = pybamm.ParameterValues(d["scalars"])
        param.update({
            f"{e}_Uss [V]": pybamm.Interpolant(
                np.array(d["Uocp"][e]["x"]), np.array(d["Uocp"][e]["y"]),
                getattr(self.var, f"{e}_thetass"), interpolator="linear")
            for e in ("neg", "pos")
        }, check_already_exists=False)
        param.update({"iapp [A]": 1.0*self.param.Q}, check_already_exists=False)
        return param

    def _init_model_parameters(self):
        # Applied current input.
        self.param.iapp = pybamm.FunctionParameter("iapp [A]", inputs={"t": pybamm.t})

        # Constants.
        self.param.F = pybamm.Parameter("Faraday constant [C.mol-1]")
        self.param.R = pybamm.Parameter("Ideal gas constant [J.K-1.mol-1]")

        # Cell-wide parameters.
        self.param.kD = pybamm.Parameter("kD")
        self.param.psi = pybamm.Parameter("psi [V]")
        self.param.T = pybamm.Parameter("T [K]")
        self.param.Q = pybamm.Parameter("Q [Ah]")

        # Separator.
        self.param.sep_kappa = pybamm.Parameter("sep_kappa [Ohm-1]")
        self.param.sep_qe = pybamm.Parameter("sep_qe [Ah]")

        # Electrodes.
        for e in ELECTRODES:
            self._set_p(e, "sigma", pybamm.Parameter(f"{e}_sigma [Ohm-1]"))
            self._set_p(e, "kappa", pybamm.Parameter(f"{e}_kappa [Ohm-1]"))
            self._set_p(e, "qe", pybamm.Parameter(f"{e}_qe [Ah]"))
            self._set_p(e, "theta0", pybamm.Parameter(f"{e}_theta0"))
            self._set_p(e, "theta100", pybamm.Parameter(f"{e}_theta100"))
            self._set_p(e, "Rdl", pybamm.Parameter(f"{e}_Rdl [Ohm]"))
            self._set_p(e, "Cdl", pybamm.Parameter(f"{e}_Cdl [F]"))
            self._set_p(e, "Rf", pybamm.Parameter(f"{e}_Rf [Ohm]"))
            self._set_p(e, "Uocp0", pybamm.Parameter(f"{e}_Uocp0 [V]"))  # initial OCP of electrode
            if self.is_msmr(e):
                J = self.J[e]
                self._set_p(e, "Dsref", pybamm.Parameter(f"{e}_Dsref [s-1]"))
                self._set_p(e, "U0", [pybamm.Parameter(f"{e}_U0_{j} [V]") for j in range(J)])
                self._set_p(e, "X", [pybamm.Parameter(f"{e}_X_{j}") for j in range(J)])
                self._set_p(e, "omega", [pybamm.Parameter(f"{e}_omega_{j}") for j in range(J)])
                self._set_p(e, "k0", [pybamm.Parameter(f"{e}_k0_{j} [A]") for j in range(J)])
                self._set_p(e, "alpha", [pybamm.Parameter(f"{e}_alpha_{j}") for j in range(J)])
            else:
                self._set_p(e, "Ds", pybamm.Parameter(f"{e}_Ds [s-1]"))
                self._set_p(e, "k0", pybamm.Parameter(f"{e}_k0 [A]"))
                self._set_p(e, "alpha", pybamm.Parameter(f"{e}_alpha"))
                self._set_p(e, "thetas0", pybamm.Parameter(f"{e}_thetas0"))

        # NOTE: Standard OCP function parameters appear in _init_solid_variables().

    def _init_model_variables(self):
        #
        # Spatial variables.
        #

        self.var.r_neg = pybamm.SpatialVariable(
            "r_neg",
            domain=["neg_particle"],
            auxiliary_domains={"secondary": "neg"},
            coord_sys="spherical polar",
        )
        self.var.r_pos = pybamm.SpatialVariable(
            "r_pos",
            domain=["pos_particle"],
            auxiliary_domains={"secondary": "pos"},
            coord_sys="spherical polar",
        )
        self.var.x_neg = pybamm.SpatialVariable("x_neg", domain=["neg"], coord_sys="cartesian")
        self.var.x_sep = pybamm.SpatialVariable("x_sep", domain=["sep"], coord_sys="cartesian")
        self.var.x_pos = pybamm.SpatialVariable("x_pos", domain=["pos"], coord_sys="cartesian")

        #
        # Electrolyte variables.
        #

        # Salt concentration.
        self.var.neg_thetae = pybamm.Variable("neg_thetae", domain="neg")
        self.var.sep_thetae = pybamm.Variable("sep_thetae", domain="sep")
        self.var.pos_thetae = pybamm.Variable("pos_thetae", domain="pos")
        self.var.thetae = pybamm.concatenation(self.var.neg_thetae, self.var.sep_thetae, self.var.pos_thetae)

        # Electrolyte potential.
        self.var.neg_phie = pybamm.Variable("neg_phie [V]", domain="neg")
        self.var.sep_phie = pybamm.Variable("sep_phie [V]", domain="sep")
        self.var.pos_phie = pybamm.Variable("pos_phie [V]", domain="pos")
        self.var.phie = pybamm.concatenation(self.var.neg_phie, self.var.sep_phie, self.var.pos_phie)

        #
        # Solid variables.
        #

        # Solid potential.
        self.var.neg_phis = pybamm.Variable("neg_phis [V]", domain="neg")
        self.var.sep_phis = pybamm.PrimaryBroadcast(pybamm.Scalar(np.nan), "sep")  # phis d.n.e. in sep
        self.var.pos_phis = pybamm.Variable("pos_phis [V]", domain="pos")
        self.var.phis = pybamm.concatenation(self.var.neg_phis, self.var.sep_phis, self.var.pos_phis)

        self.var.f = self.param.F / self.param.R / self.param.T

        # Particle stoichiometry and surface potential.
        for e in ELECTRODES:
            self._init_solid_variables(e)
        self.var.sep_thetas = pybamm.PrimaryBroadcast(np.nan, "sep")  # thetas d.n.e. in sep
        self.var.sep_thetass = pybamm.PrimaryBroadcast(np.nan, "sep")  # thetass d.n.e. in sep
        self.var.thetass = pybamm.concatenation(self.var.neg_thetass, self.var.sep_thetass, self.var.pos_thetass)

        #
        # Interface variables.
        #

        # Faradaic plus double-layer current.
        self.var.neg_ifdl = pybamm.Variable("neg_ifdl [A]", domain="neg")
        self.var.sep_ifdl = pybamm.PrimaryBroadcast(0, "sep")  # ifdl=0 in sep
        self.var.pos_ifdl = pybamm.Variable("pos_ifdl [A]", domain="pos")
        self.var.ifdl = pybamm.concatenation(self.var.neg_ifdl, self.var.sep_ifdl, self.var.pos_ifdl)

        # Double-layer capacitor voltage.
        self.var.neg_vdl = pybamm.Variable("neg_vdl [V]", domain="neg")
        self.var.sep_vdl = pybamm.PrimaryBroadcast(np.nan, "sep")  # vdl d.n.e. in sep
        self.var.pos_vdl = pybamm.Variable("pos_vdl [V]", domain="pos")
        self.var.vdl = pybamm.concatenation(self.var.neg_vdl, self.var.sep_vdl, self.var.pos_vdl)

    def _init_solid_variables(self, e):
        if not self.is_msmr(e):
            # Particle stoichiometry.
            thetas = pybamm.Variable(
                f"{e}_thetas", domain=f"{e}_particle", auxiliary_domains={"secondary": e})
            self._set_v(e, "thetas", thetas)
            self._set_v(e, "thetass", pybamm.surf(thetas))

            # OCP function.
            Uss = pybamm.FunctionParameter(f"{e}_Uss [V]", {f"{e}_thetass": self._v(e, "thetass")})
            self._set_p(e, "Uss", Uss)
            self._set_v(e, "Uss", Uss)
            return

        # MSMR: the particle potential Us is the state. We recast the solid-diffusion
        # equation from thetas to Us using the Baker-Verbrugge diffusivity model.
        Us = pybamm.Variable(f"{e}_Us [V]", domain=f"{e}_particle", auxiliary_domains={"secondary": e})
        self._set_v(e, "Us", Us)
        self._set_v(e, "Uss", pybamm.surf(Us))

        xs, xss, dxdU = [], [], []
        for j in range(self.J[e]):
            # Collect MSMR parameters for gallery j.
            U0_j = self._p(e, "U0")[j]
            X_j = self._p(e, "X")[j]
            omega_j = self._p(e, "omega")[j]

            # Partial stoichiometry of gallery j: x_j = X_j / (1 + g_j(U)).
            g_j = pybamm.exp((Us - U0_j) * self.var.f / omega_j)
            xs.append(X_j / (1 + g_j))
            xss.append(pybamm.surf(xs[j]))  # at surface of particles

            # Differential capacity (dxj/dU) of gallery j.
            dxdU.append(-self.var.f * X_j * g_j / omega_j / (1 + g_j) ** 2)

        # Accumulate solid stoichiometry and differential capacity.
        self._set_v(e, "xs", xs)
        self._set_v(e, "xss", xss)
        self._set_v(e, "thetas", sum(xs))
        self._set_v(e, "thetass", pybamm.surf(self._v(e, "thetas")))
        self._set_v(e, "dthetasdU", sum(dxdU))

    def _init_model_equations(self):

        #
        # Current-overpotential equations.
        #

        # Overpotential.
        self.var.neg_phise = self.var.neg_phis - self.var.neg_phie
        self.var.sep_phise = pybamm.PrimaryBroadcast(np.nan, "sep")
        self.var.pos_phise = self.var.pos_phis - self.var.pos_phie
        self.var.phise = pybamm.concatenation(self.var.neg_phise, self.var.sep_phise, self.var.pos_phise)
        for e in ELECTRODES:
            self._set_v(e, "etas", self._v(e, "phise") - self._v(e, "Uss") - self._p(e, "Rf") * self._v(e, "ifdl"))
        self.var.sep_etas = pybamm.PrimaryBroadcast(np.nan, "sep")
        self.var.etas = pybamm.concatenation(self.var.neg_etas, self.var.sep_etas, self.var.pos_etas)

        for e in ELECTRODES:
            # Faradaic current.
            self._init_faradaic_current(e)

            # Double-layer current.
            etas, Uss, vdl = self._v(e, "etas"), self._v(e, "Uss"), self._v(e, "vdl")
            self._set_v(e, "idl", (etas + Uss - vdl) / self._p(e, "Rdl"))

            # Current-overpotential equations.
            ifdl = self._v(e, "ifdl")
            self.algebraic[ifdl] = self._v(e, "if") + self._v(e, "idl") - ifdl
            self.rhs[vdl] = (etas + Uss - vdl) / self._p(e, "Rdl") / self._p(e, "Cdl")
            self.initial_conditions[ifdl] = 0
            self.initial_conditions[vdl] = self._p(e, "Uocp0")

        #
        # Charge conservation in solid.
        #

        self.var.neg_is = -self.param.neg_sigma * pybamm.grad(self.var.neg_phis)
        self.var.sep_is = pybamm.PrimaryBroadcast(np.nan, "sep")  # is d.n.e. in sep
        self.var.pos_is = -self.param.pos_sigma * pybamm.grad(self.var.pos_phis)
        self.var.is_ = pybamm.concatenation(self.var.neg_is, self.var.sep_is, self.var.pos_is)
        # neg
        self.algebraic[self.var.neg_phis] = pybamm.div(self.var.neg_is) + self.var.neg_ifdl
        self.boundary_conditions[self.var.neg_phis] = {
            "left": (-self.param.iapp / self.param.neg_sigma, "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[self.var.neg_phis] = self.param.neg_Uocp0
        # pos
        self.algebraic[self.var.pos_phis] = pybamm.div(self.var.pos_is) + self.var.pos_ifdl
        self.boundary_conditions[self.var.pos_phis] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (-self.param.iapp / self.param.pos_sigma, "Neumann"),
        }
        self.initial_conditions[self.var.pos_phis] = self.param.pos_Uocp0

        #
        # Mass conservation in solid.
        #

        for e in ELECTRODES:
            self._init_solid_diffusion(e)

        #
        # Charge conservation in electrolyte.
        #

        self.var.neg_ie = (
            -self.param.neg_kappa * pybamm.grad(self.var.neg_phie)
            - self.param.kD * self.param.neg_kappa * self.param.T * pybamm.grad(self.var.neg_thetae)
        )
        self.var.sep_ie = (
            -self.param.sep_kappa * pybamm.grad(self.var.sep_phie)
            - self.param.kD * self.param.sep_kappa * self.param.T * pybamm.grad(self.var.sep_thetae)
        )
        self.var.pos_ie = (
            -self.param.pos_kappa * pybamm.grad(self.var.pos_phie)
            - self.param.kD * self.param.pos_kappa * self.param.T * pybamm.grad(self.var.pos_thetae)
        )
        self.var.ie = pybamm.concatenation(self.var.neg_ie, self.var.sep_ie, self.var.pos_ie)
        self.algebraic[self.var.neg_phie] = pybamm.div(self.var.neg_ie) - self.var.neg_ifdl
        self.algebraic[self.var.sep_phie] = pybamm.div(self.var.sep_ie) - self.var.sep_ifdl
        self.algebraic[self.var.pos_phie] = pybamm.div(self.var.pos_ie) - self.var.pos_ifdl
        self.boundary_conditions[self.var.neg_phie] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),  # set phi_e=0 at x=0
            "right": (
                -self.param.iapp / self.param.neg_kappa
                - self.param.kD * self.param.T * pybamm.boundary_gradient(self.var.neg_thetae,"right"),
                "Neumann"
            ),
        }
        self.boundary_conditions[self.var.sep_phie] = {
            "left": (pybamm.boundary_value(self.var.neg_phie, "right"), "Dirichlet"),
            "right": (
                -self.param.iapp / self.param.sep_kappa
                - self.param.kD * self.param.T * pybamm.boundary_gradient(self.var.sep_thetae, "right"),
                "Neumann"
            ),
        }
        self.boundary_conditions[self.var.pos_phie] = {
            "left":  (pybamm.boundary_value(self.var.sep_phie, "right"), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann")
        }
        self.initial_conditions[self.var.neg_phie] = 0
        self.initial_conditions[self.var.sep_phie] = 0
        self.initial_conditions[self.var.pos_phie] = 0

        #
        # Mass conservation in electrolyte.
        #

        FNLi_neg = -self.param.neg_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.neg_thetae)
        FNLi_sep = -self.param.sep_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.sep_thetae)
        FNLi_pos = -self.param.pos_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.pos_thetae)
        self.rhs[self.var.neg_thetae] = (
            (self.var.neg_ifdl - pybamm.div(FNLi_neg))
            / 3600 / self.param.neg_qe
        )
        self.rhs[self.var.sep_thetae] = (
            (self.var.sep_ifdl - pybamm.div(FNLi_sep))
            / 3600 / self.param.sep_qe
        )
        self.rhs[self.var.pos_thetae] = (
            (self.var.pos_ifdl - pybamm.div(FNLi_pos))
            / 3600 / self.param.pos_qe
        )
        self.boundary_conditions[self.var.neg_thetae] = {
            # Zero salt flux:
            "left": (pybamm.Scalar(0), "Neumann"),
            # Flux continuity at neg-sep boundary:
            "right": (
                pybamm.boundary_gradient(self.var.sep_thetae, "left")
                * self.param.sep_kappa / self.param.neg_kappa,
                "Neumann"
            ),
        }
        self.boundary_conditions[self.var.sep_thetae] = {
            # Concentration continuity at neg-sep boundary:
            "left": (pybamm.boundary_value(self.var.neg_thetae, "right"), "Dirichlet"),
            # Concentration continuity at sep-pos boundary:
            "right": (pybamm.boundary_value(self.var.pos_thetae, "left"), "Dirichlet"),
        }
        self.boundary_conditions[self.var.pos_thetae] = {
            # Flux continuity at sep-pos boundary:
            "left": (
                pybamm.boundary_gradient(self.var.sep_thetae, "right")
                * self.param.sep_kappa / self.param.pos_kappa,
                "Neumann"
            ),
            # Zero salt flux:
            "right": (pybamm.Scalar(0), "Neumann")
        }
        self.initial_conditions[self.var.neg_thetae] = 1.0
        self.initial_conditions[self.var.sep_thetae] = 1.0
        self.initial_conditions[self.var.pos_thetae] = 1.0

        # Compute salt flux.
        self.var.neg_FNe = -(
            self.param.neg_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.neg_thetae)
            + self.var.neg_ie
        )
        self.var.sep_FNe = -(
            self.param.sep_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.sep_thetae)
            + self.var.sep_ie
        )
        self.var.pos_FNe = -(
            self.param.pos_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.pos_thetae)
            + self.var.pos_ie
        )
        self.var.FNe = pybamm.concatenation(self.var.neg_FNe, self.var.sep_FNe, self.var.pos_FNe)

        #
        # Cell-level quantities.
        #

        # Cell voltage.
        self.var.vcell = (
            pybamm.boundary_value(self.var.pos_phis, "right")
            - pybamm.boundary_value(self.var.neg_phis,"left")
        )

        for e in ELECTRODES:
            # Average stoichiometry.
            thetas_avg = pybamm.Integral(
                pybamm.Integral(
                    self._v(e, "thetas"),
                    getattr(self.var, f"r_{e}")
                )/(4*np.pi/3),  # average by volume of unit sphere
                getattr(self.var, f"x_{e}")
            )
            self._set_v(e, "thetas_avg", thetas_avg)

            # Electrode state of charge.
            self._set_v(e, "soc", (
                (thetas_avg - self._p(e, "theta0"))
                / (self._p(e, "theta100") - self._p(e, "theta0"))
            ))

    def _init_faradaic_current(self, e):
        thetass, thetae, etas = self._v(e, "thetass"), self._v(e, "thetae"), self._v(e, "etas")
        f = self.var.f

        if not self.is_msmr(e):
            k0, alpha = self._p(e, "k0"), self._p(e, "alpha")
            i0 = k0 * thetass ** alpha * (1 - thetass) ** (1 - alpha) * thetae ** (1 - alpha)
            self._set_v(e, "i0", i0)
            self._set_v(e, "if", i0 * (pybamm.exp((1 - alpha) * f * etas) - pybamm.exp(-alpha * f * etas)))
            return

        # MSMR: sum of per-gallery Butler-Volmer currents. The standard-form exchange current
        #     k0_j * xss_j^(omega_j*alpha_j) * (X_j - xss_j)^(omega_j*(1-alpha_j)) * thetae^(1-alpha_j)
        # is singular as xss_j -> X_j; substituting xss_j = X_j/(1+g_j) gives the potential form below.
        Uss = self._v(e, "Uss")
        i0j, ifj = [], []
        for j in range(self.J[e]):
            U0_j = self._p(e, "U0")[j]
            omega_j = self._p(e, "omega")[j]
            k0_j = self._p(e, "k0")[j]
            alpha_j = self._p(e, "alpha")[j]
            i0j.append(
                k0_j *
                self._v(e, "xss")[j] ** omega_j *
                pybamm.exp((1 - alpha_j) * f * (Uss - U0_j)) *
                thetae ** (1 - alpha_j)
            )
            ifj.append(i0j[j] * (pybamm.exp((1 - alpha_j) * f * etas) - pybamm.exp(-alpha_j * f * etas)))
        self._set_v(e, "i0j", i0j)
        self._set_v(e, "ifj", ifj)
        self._set_v(e, "if", sum(ifj))

    def _init_solid_diffusion(self, e):
        # Surface flux scale: converts faradaic current [A] to d(thetas)/dr at the particle surface.
        flux = self._v(e, "if") * abs(self._p(e, "theta100") - self._p(e, "theta0")) / 10_800 / self.param.Q

        if not self.is_msmr(e):
            thetas, Ds = self._v(e, "thetas"), self._p(e, "Ds")
            self.rhs[thetas] = pybamm.div(Ds * pybamm.grad(thetas))
            self.boundary_conditions[thetas] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (-flux / Ds, "Neumann"),
            }
            self.initial_conditions[thetas] = self._p(e, "thetas0")
            return

        # MSMR: Baker-Verbrugge Ds = Dsref*f*thetas*(1-thetas)*(-dU/dthetas), so
        # Ds*grad(thetas) = -D_coeff*grad(Us) with D_coeff = f*Dsref*thetas*(1-thetas).
        Us, thetas = self._v(e, "Us"), self._v(e, "thetas")
        D_coeff = self.var.f * self._p(e, "Dsref") * thetas * (1 - thetas)
        self.rhs[Us] = -pybamm.div(D_coeff * pybamm.grad(Us)) / self._v(e, "dthetasdU")
        self.boundary_conditions[Us] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (flux / pybamm.surf(D_coeff), "Neumann"),
        }
        self.initial_conditions[Us] = self._p(e, "Uocp0")

    def _init_model_events(self):
        self.events += [
            pybamm.Event("Minimum voltage", self.var.vcell - self.param.vmin),
            pybamm.Event("Maximum voltage", self.param.vmax - self.var.vcell),
        ]
