"""
lumped.fom.model

The BMS3-toolbox lumped full-order model (genFOM) as a PyBaMM model, built
from a CellSpec.

Coordinates follow the toolbox: x~ in [0, 1] (neg), [1, 2] (sep) and [2, 3]
(pos); a half cell (foil negative) has sep [0, 1] and pos [1, 2] with the foil
at x~ = 0. Particles use r~ = r/R in [0, 1]. Currents are lumped [A];
discharge is positive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybamm

from .spec import CellSpec, Param, SpecError

# Mesh presets: elements per region and per particle radius (coarsening 1-5).
# - Particles get many elements: a diffusivity that varies steeply with
#   stoichiometry makes the surface stoichiometry step between elements, and
#   only a fine radial mesh keeps the current from rippling along x.
MESH = {"region": [20, 15, 10, 8, 5], "radial": [400, 200, 100, 50, 30]}
BIG = 1e6        # moves an inactive event or mode term out of the way


@dataclass
class Limits:
    """Charge limits (any may be None): potentials [V], theta_e [-], SOC stop [0-1]."""
    Vmax: float | None = None
    phiseMin: float | None = None
    phiseMax: float | None = None
    thetaeMin: float | None = None
    thetaeMax: float | None = None
    SOCLimit: float | None = None
    thetaePinTau: float = 1.0
    thetaePinKp: float = 10.0

    @classmethod
    def from_dict(cls, d: dict | None) -> Limits:
        d = {k: v for k, v in (d or {}).items() if v is not None and not (isinstance(v, list) and not v)}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown charge limits: {', '.join(sorted(unknown))}.")
        return cls(**{k: float(v) for k, v in d.items()})

    @property
    def bounds(self) -> list[tuple[str, int]]:
        """Active bounds as (name, direction): +1 an upper bound, -1 a lower bound."""
        out = [("Vmax", 1), ("phiseMin", -1), ("phiseMax", 1), ("thetaeMin", -1), ("thetaeMax", 1)]
        return [(n, s) for n, s in out if getattr(self, n) is not None]


ELEMENT_KEYS = ("neg", "sep", "pos", "particle")


@dataclass
class Options:
    """
    Model options: thermal model, charge limits, mesh.

    mesh is a coarsening preset (1-5); elements overrides its counts per
    region ('neg', 'sep', 'pos') and per particle ('particle').
    """
    thermal: bool = False
    limits: Limits = field(default_factory=Limits)
    mesh: int = 1
    elements: dict = field(default_factory=dict)

    def __post_init__(self):
        bad = set(self.elements) - set(ELEMENT_KEYS)
        if bad:
            raise ValueError(f"Unknown mesh regions: {', '.join(sorted(bad))}; use {', '.join(ELEMENT_KEYS)}.")
        for k, n in self.elements.items():
            if int(n) != n or not 2 <= n <= 1000:
                raise ValueError(f"The {k} mesh needs a whole number of elements from 2 to 1000; got {n}.")
        self.elements = {k: int(n) for k, n in self.elements.items()}


def _nice_pow(x, y):
    """x^y with a non-negative base (genFOM nicePow)."""
    return pybamm.maximum(x, 1e-12) ** y


def _on(expr, domain):
    """expr on a domain (broadcast when it has none)."""
    if isinstance(expr, (int, float)):
        expr = pybamm.Scalar(expr)
    return expr if expr.domain else pybamm.PrimaryBroadcast(expr, domain)


class FOMModel(pybamm.BaseModel):
    """
    The toolbox's lumped full-order model of a cell.

    Parameters
    ----------
    spec : CellSpec
        The compiled cell model (lumped.fom.load_spec).
    z0 : float
        Initial cell SOC [0-1].
    T0 : float
        Initial (ambient) temperature [K].
    options : Options
        Thermal model, charge limits and mesh.

    Notes
    -----
    Inputs: 'Ides [A]' (the demanded current), 'Tamb [K]' (ambient), and, with
    charge limits, one mode flag per limit ('mode <name>', 0 or 1).
    """

    def __init__(self, spec: CellSpec, z0: float, T0: float, options: Options | None = None):
        super().__init__(f"BMS3 FOM ({spec.name or 'cell'})")
        self.spec, self.z0, self.T0 = spec, float(z0), float(T0)
        self.options = options or Options()
        if self.options.thermal and not spec.thermal:
            raise SpecError("The thermal model needs a cell with thermal parameters (#general thermal = 1).")
        self.half = spec.half_cell
        self.regions = ("sep", "pos") if self.half else ("neg", "sep", "pos")
        self.electrodes = ("pos",) if self.half else ("neg", "pos")
        self.F, self.R = spec.F, spec.R
        self.Ides = pybamm.InputParameter("Ides [A]")
        self.Tamb = pybamm.InputParameter("Tamb [K]")
        self.out = {}
        self._spatial()
        self._temperature()
        self._current()
        self._ocp_at_start()
        self._electrodes()
        self._electrolyte()
        self._solid_potential()
        self._foil()
        self._cell()
        if self.options.thermal:
            self._thermal()
        self._limits()

    # -- geometry ---------------------------------------------------------------

    def _spatial(self):
        lo = {"neg": 0, "sep": 1, "pos": 2} if not self.half else {"sep": 0, "pos": 1}
        self.xlim = {r: (lo[r], lo[r] + 1) for r in self.regions}
        self.x = {r: pybamm.SpatialVariable(f"x_{r}", domain=[r], coord_sys="cartesian") for r in self.regions}
        self.r = {}
        for e in self.electrodes:
            for i in range(len(self.spec[e].materials)):
                d = self._pdom(e, i)
                self.r[d] = pybamm.SpatialVariable(
                    f"r_{e}{i + 1}", domain=[d], auxiliary_domains={"secondary": e}, coord_sys="spherical polar")

    @staticmethod
    def _pdom(e, i):
        return f"{e}_particle{i + 1}"

    @property
    def default_geometry(self):
        g = {r: {self.x[r]: {"min": a, "max": b}} for r, (a, b) in self.xlim.items()}
        g.update({d: {r: {"min": 0, "max": 1}} for d, r in self.r.items()})
        return g

    @property
    def default_submesh_types(self):
        # Uniform particle meshes: the surface-flux condition divides by surf(Ds), which
        # equals the finite-volume boundary-edge coefficient only on a uniform mesh, so
        # the particle then gains exactly the faradaic charge.
        return {d: pybamm.Uniform1DSubMesh for d in list(self.regions) + list(self.r)}

    @property
    def default_var_pts(self):
        k = min(max(int(self.options.mesh), 1), 5) - 1
        n = self.options.elements
        pts = {self.x[r]: n.get(r, MESH["region"][k]) for r in self.regions}
        pts.update({r: n.get("particle", MESH["radial"][k]) for r in self.r.values()})
        return pts

    @property
    def default_spatial_methods(self):
        return {d: pybamm.FiniteVolume() for d in list(self.regions) + list(self.r)}

    # -- parameters -------------------------------------------------------------

    def _p(self, region: str, name: str, mat: int | None = None) -> Param:
        r = self.spec[region]
        return (r.params if mat is None else r.materials[mat])[name]

    def _has(self, region: str, name: str, mat: int | None = None) -> bool:
        r = self.spec.regions.get(region)
        return r is not None and name in (r.params if mat is None else r.materials[mat])

    def _const(self, name: str, x, T):
        return self._p("const", name).expr(x, T)

    # -- temperature and current ------------------------------------------------

    def _temperature(self):
        if self.options.thermal:
            self.T = {r: pybamm.Variable(f"T_{r} [K]", domain=r) for r in self.regions}
            self.Tc = pybamm.concatenation(*self.T.values())
        else:
            self.T = {r: self.Tamb for r in self.regions}
            self.Tc = self.Tamb

    def _Tp(self, e, i):
        """Temperature in a particle domain."""
        if not self.options.thermal:
            return self.Tamb
        return pybamm.PrimaryBroadcast(self.T[e], self._pdom(e, i))

    def _current(self):
        if self.options.limits.bounds:
            self.Iapp = pybamm.Variable("Iapp [A]")
        else:
            self.Iapp = self.Ides

    # -- initial state ----------------------------------------------------------

    def _ocp_at_start(self):
        """Electrode equilibrium potentials and particle states at z0 (genFOM UocpBlend, soc maps)."""
        self.U0 = {}
        self.theta_init = {}
        for e in self.electrodes:
            mats = self.spec[e].materials
            th = [float(np.squeeze(m["soc"](self.z0, self.T0))) for m in mats]
            self.theta_init[e] = th
            if len(mats) == 1:
                U = float(np.squeeze(mats[0]["Uocp"](th[0], self.spec.Tref)))
            else:
                U = float(np.squeeze(self._p(e, "Ueq")(self.z0, self.T0)))
            self.U0[e] = U
        if self.half:
            self.U0["neg"] = 0.0

    # -- electrodes -------------------------------------------------------------

    def _is_msmr(self, e, i):
        return self.spec.MSMR and self._has(e, "U0", i) and self._p(e, "U0", i).type == "vector"

    def _electrodes(self):
        F, R = self.F, self.R
        self.phis = {e: pybamm.Variable(f"phis_{e} [V]", domain=e) for e in self.electrodes}
        self.phie_r = {r: pybamm.Variable(f"phie_{r} [V]", domain=r) for r in self.regions}
        self.thetae_r = {r: pybamm.Variable(f"thetae_{r}", domain=r) for r in self.regions}
        self.mat = {}
        self.ifdl_e = {}
        for e in self.electrodes:
            T = self.T[e]
            f = F / (R * T)
            phise = self.phis[e] - self.phie_r[e]
            thetae = self.thetae_r[e]
            mats, total = [], 0
            for i in range(len(self.spec[e].materials)):
                m = self._material(e, i, phise, thetae, T, f)
                mats.append(m)
                total = total + m["ifdl"]
            self.mat[e] = mats
            self.ifdl_e[e] = total
            self.out[f"{e}_phise"] = phise

    def _material(self, e, i, phise, thetae, T, f):
        """One material's particle, kinetics and double layer."""
        F, R = self.F, self.R
        P = lambda name, x=0.5, TT=T: self._p(e, name, i).expr(x, TT)  # noqa: E731
        Tp = self._Tp(e, i)
        pdom = self._pdom(e, i)
        Qabs = P("QabsCell")
        m = {}
        ifdl = pybamm.Variable(f"ifdl_{e}{i + 1} [A]", domain=e)
        if self._is_msmr(e, i):
            Us = pybamm.Variable(f"U_{e}{i + 1} [V]", domain=pdom, auxiliary_domains={"secondary": e})
            U0p, Xp, wp = (self._p(e, n, i) for n in ("U0", "X", "omega"))
            J = U0p.size
            fp = F / (R * Tp)

            def gallery(U, j, ff, TT):
                g = pybamm.exp((U - U0p.expr(0.5, TT, j)) * ff / wp.expr(0.5, TT, j))
                return g, Xp.expr(0.5, TT, j) / (1 + g)

            theta, dthdU = 0, 0
            for j in range(J):
                g, x = gallery(Us, j, fp, Tp)
                theta = theta + x
                dthdU = dthdU - fp * Xp.expr(0.5, Tp, j) * g / wp.expr(0.5, Tp, j) / (1 + g) ** 2
            Uss = pybamm.surf(Us)
            thetass = 0
            for j in range(J):
                thetass = thetass + gallery(Uss, j, f, T)[1]
            Rf = P("Rf", thetass)
            eta = phise - Uss - Rf * ifdl
            i_f = 0
            for j in range(J):
                g, xss = gallery(Uss, j, f, T)
                a = self._p(e, "alpha", i).expr(0.5, T, j)
                k0 = self._p(e, "k0", i).expr(0.5, T, j)
                i0 = k0 * _nice_pow(xss, wp.expr(0.5, T, j)) * pybamm.exp((1 - a) * (Uss - U0p.expr(0.5, T, j)) * f) \
                    * _nice_pow(thetae, 1 - a)
                i_f = i_f + i0 * (pybamm.exp((1 - a) * f * eta) - pybamm.exp(-a * f * eta))
            Dc = P("Dsref", 0.5, Tp) * fp * theta * (theta - 1)
            self.rhs[Us] = pybamm.div(Dc * pybamm.grad(Us)) / dthdU
            self.boundary_conditions[Us] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (-i_f / (10800 * Qabs) / pybamm.surf(Dc), "Neumann"),
            }
            self.initial_conditions[Us] = pybamm.Scalar(self.U0[e])
            m.update(state=Us, theta=theta, Uss=Uss)
            Uss0 = self.U0[e]
        else:
            th = pybamm.Variable(f"thetas_{e}{i + 1}", domain=pdom, auxiliary_domains={"secondary": e})
            thetass = pybamm.surf(th)
            Uss = P("Uocp", thetass)
            Rf = P("Rf", thetass)
            eta = phise - Uss - Rf * ifdl
            a = P("alpha")
            i0 = P("k0", thetass) * _nice_pow(thetae, 1 - a) * _nice_pow(1 - thetass, 1 - a) * _nice_pow(thetass, a)
            i_f = i0 * (pybamm.exp((1 - a) * f * eta) - pybamm.exp(-a * f * eta))
            if self._has(e, "Dsref", i):                     # Baker-Verbrugge
                Ds = -P("Dsref", 0.5, Tp) * F / (R * Tp) * pybamm.AbsoluteValue(th * (th - 1)) \
                    * P("dUocp", th, Tp)
            else:
                Ds = P("Ds", th, Tp)
            self.rhs[th] = pybamm.div(Ds * pybamm.grad(th))
            self.boundary_conditions[th] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (-i_f / (10800 * Qabs) / pybamm.surf(Ds), "Neumann"),
            }
            th0 = self.theta_init[e][i]
            self.initial_conditions[th] = pybamm.Scalar(th0)
            m.update(state=th, theta=th, Uss=Uss)
            Uss0 = float(np.squeeze(self._p(e, "Uocp", i)(th0, self.T0)))
        idl = self._double_layer(f"{e}{i + 1}", phise - Rf * ifdl, ifdl, i_f, P("Rdl", thetass), P("Cdl", thetass),
                                 Uss0, self._p(e, "Rdl", i), self._p(e, "Cdl", i))
        self.initial_conditions[ifdl] = pybamm.Scalar(0)
        m.update(ifdl=ifdl, i_f=i_f, idl=idl, eta=eta, thetass=thetass, T=T)
        return m

    def _double_layer(self, tag, Vint, ifdl, i_f, Rdl, Cdl, V0, Rdl_p, Cdl_p):
        """
        Series Rdl-Cdl branch in parallel with the faradaic branch, behind Rf.

        idl + Rdl*Cdl*d(idl)/dt = Cdl*d(Vint)/dt with Vint = phise - Rf*ifdl; the
        capacitor voltage vdl is the state. Returns idl.
        """
        zero = lambda p: p.type == "scalar" and float(p.value) == 0  # noqa: E731
        if zero(Cdl_p):
            self.algebraic[ifdl] = ifdl - i_f
            return 0 * ifdl
        vdl = pybamm.Variable(f"vdl_{tag} [V]", domain=ifdl.domain)
        self.initial_conditions[vdl] = pybamm.Scalar(V0)
        if zero(Rdl_p):
            self.rhs[vdl] = (ifdl - i_f) / Cdl
            self.algebraic[ifdl] = Vint - vdl
            return ifdl - i_f
        idl = (Vint - vdl) / Rdl
        self.rhs[vdl] = idl / Cdl
        self.algebraic[ifdl] = i_f + idl - ifdl
        return idl

    # -- electrolyte ------------------------------------------------------------

    def _electrolyte(self):
        flux_form = self._has("const", "t0mNorm")
        th, phie, T = self.thetae_r, self.phie_r, self.T
        cat = lambda d: pybamm.concatenation(*[_on(d[r], r) for r in self.regions])  # noqa: E731
        kappa = cat({r: self._p(r, "kappa").expr(th[r], T[r]) for r in self.regions})
        qe = cat({r: self._p(r, "qe").expr(th[r], T[r]) for r in self.regions})
        kD = cat({r: self._const("kD", th[r], T[r]) for r in self.regions})
        psi = cat({r: self._const("psi", th[r], T[r]) for r in self.regions})
        Tc = cat(T)
        self.thetae = cat(th)
        self.phie = cat(phie)
        ifdl = cat({r: self.ifdl_e.get(r, pybamm.Scalar(0)) for r in self.regions})
        ie = -kappa * (pybamm.grad(self.phie) + kD * Tc / self.thetae * pybamm.grad(self.thetae))
        c = psi * Tc * kappa
        if flux_form:
            t0m = cat({r: self._const("t0mNorm", th[r], T[r]) for r in self.regions})
            self.rhs[self.thetae] = pybamm.div(c * pybamm.grad(self.thetae) + t0m * ie) / (3600 * qe)
        else:
            self.rhs[self.thetae] = (pybamm.div(c * pybamm.grad(self.thetae)) + ifdl) / (3600 * qe)
        self.algebraic[self.phie] = pybamm.div(ie) - ifdl
        self.initial_conditions[self.thetae] = pybamm.Scalar(1)
        self.initial_conditions[self.phie] = pybamm.Scalar(-self.U0["neg"])
        self._kappa, self._kD, self._c, self._Tcat = kappa, kD, c, Tc
        self.ie = ie
        zero = (pybamm.Scalar(0), "Neumann")
        self.boundary_conditions[self.thetae] = {"left": zero, "right": zero}
        self.boundary_conditions[self.phie] = {"left": zero, "right": zero}
        self._flux_form = flux_form

    # -- solid potential --------------------------------------------------------

    def _solid_potential(self):
        self._sigma = {}
        for e in self.electrodes:
            sigma = self._p(e, "sigma").expr(0.5, self.T[e])
            phis = self.phis[e]
            self.algebraic[phis] = pybamm.div(-sigma * pybamm.grad(phis)) + self.ifdl_e[e]
            self.initial_conditions[phis] = pybamm.Scalar(self.U0[e] - self.U0["neg"])
            if e == "neg":
                bc = {"left": (pybamm.Scalar(0), "Dirichlet"), "right": (pybamm.Scalar(0), "Neumann")}
            else:
                bc = {"left": (pybamm.Scalar(0), "Neumann"),
                      "right": (-self.Iapp / pybamm.boundary_value(_on(sigma, e), "right"), "Neumann")}
            self.boundary_conditions[phis] = bc
            self._sigma[e] = sigma

    # -- foil negative electrode ------------------------------------------------

    def _foil(self):
        if not self.half:
            return
        F, R = self.F, self.R
        T0 = pybamm.boundary_value(_on(self.T["sep"], "sep"), "left")
        phie0 = pybamm.boundary_value(self.phie, "left")
        thetae0 = pybamm.boundary_value(self.thetae, "left")
        P = lambda name: self._p("neg", name).expr(0.5, T0)  # noqa: E731
        f = F / (R * T0)
        ifdln = pybamm.Variable("ifdl_foil [A]")
        Rf = P("Rf")
        eta = -phie0 - Rf * self.Iapp                         # genFOM uses Iapp here
        a = P("alpha")
        i0 = P("k0") * _nice_pow(thetae0, 1 - a)
        i_f = i0 * (pybamm.exp((1 - a) * f * eta) - pybamm.exp(-a * f * eta))
        idl = self._double_layer("foil", -phie0 - Rf * ifdln, ifdln, i_f, P("Rdl"), P("Cdl"), 0.0,
                                 self._p("neg", "Rdl"), self._p("neg", "Cdl"))
        self.initial_conditions[ifdln] = pybamm.Scalar(0)
        kappa0 = pybamm.boundary_value(self._kappa, "left")
        kD0 = pybamm.boundary_value(self._kD, "left")
        c0 = pybamm.boundary_value(self._c, "left")
        if self._flux_form:
            t0m0 = self._const("t0mNorm", thetae0, T0)
            dth = -t0m0 * ifdln / c0
        else:
            dth = -self.Iapp / c0                              # genFOM: Iapp, not ifn+idln
        self.boundary_conditions[self.thetae]["left"] = (dth, "Neumann")
        self.boundary_conditions[self.phie]["left"] = (-ifdln / kappa0 - kD0 * T0 / thetae0 * dth, "Neumann")
        self.foil = {"ifdl": ifdln, "i_f": i_f, "idl": idl, "eta": eta}
        self.out["foil_phise"] = -phie0

    # -- cell-level quantities --------------------------------------------------

    def _cell(self):
        Rc = self._const("Rc", 1, self._T_at_cc()) if self._has("const", "Rc") else 0
        self.Vcell = pybamm.boundary_value(self.phis["pos"], "right") - self.Iapp * Rc
        zavg, thavg = {}, {}
        for e in self.electrodes:
            mats = self.spec[e].materials
            Q = np.array([float(np.squeeze(m["Q"](0, self.spec.Tref))) for m in mats])
            w = Q / Q.sum()
            z, t = 0, 0
            for i, (m, wi) in enumerate(zip(self.mat[e], w)):
                r = self.r[self._pdom(e, i)]
                avg = pybamm.Integral(pybamm.Integral(m["theta"], r) / (4 * np.pi / 3), self.x[e])
                th0 = float(np.squeeze(mats[i]["theta0"](0.5, self.spec.Tref)))
                th100 = float(np.squeeze(mats[i]["theta100"](0.5, self.spec.Tref)))
                z = z + wi * (avg - th0) / (th100 - th0)
                t = t + wi * avg
            zavg[e], thavg[e] = z, t
        self.SOC = zavg["pos"] if self.half else zavg["neg"]
        self.zavg, self.thavg = zavg, thavg

    def _T_at_cc(self):
        if not self.options.thermal:
            return self.Tamb
        return pybamm.boundary_value(self.Tc, "right")

    # -- thermal model ----------------------------------------------------------

    def _thermal(self):
        T, regions = self.T, self.regions
        cat = lambda d: pybamm.concatenation(*[_on(d[r], r) for r in regions])  # noqa: E731
        Rth = cat({r: self._p(r, "Rth").expr(0.5, T[r]) for r in regions})
        Cth = cat({r: self._p(r, "Cth").expr(0.5, T[r]) for r in regions})
        q = {}
        for r in regions:
            gphie = pybamm.grad(self.phie_r[r])
            kD05 = self._const("kD", 0.5, T[r])
            kappa = self._p(r, "kappa").expr(self.thetae_r[r], T[r])
            Qe = pybamm.inner(kappa * (gphie + kD05 * T[r] / self.thetae_r[r] * pybamm.grad(self.thetae_r[r])), gphie)
            if r in self.electrodes:
                Qi = sum(m["i_f"] * m["eta"] for m in self.mat[r])
                Qr = 0
                for i, m in enumerate(self.mat[r]):
                    if self._has(r, "dUdT", i):
                        Qr = Qr + m["i_f"] * T[r] * self._p(r, "dUdT", i).expr(m["thetass"], T[r])
                Qs = self._sigma[r] * pybamm.inner(pybamm.grad(self.phis[r]), pybamm.grad(self.phis[r]))
                q[r] = Qi + Qr + Qs + Qe
            else:
                q[r] = Qe
        Qdot = cat(q)
        self.rhs[self.Tc] = (pybamm.div(pybamm.grad(self.Tc) / Rth) + Qdot) / Cth
        if not self._has("const", "Rthse"):
            raise SpecError("The thermal model needs const Rthse.")
        TL, TR = pybamm.boundary_value(self.Tc, "left"), pybamm.boundary_value(self.Tc, "right")
        RcHalf = 0.5 * self.Iapp ** 2 * self._const("Rc", 0.5, TR) if self._has("const", "Rc") else 0
        gL = -(TL - self.Tamb) / self._const("Rthse", 0.5, TL) + RcHalf
        gR = -(TR - self.Tamb) / self._const("Rthse", 0.5, TR) + RcHalf
        if self.half:
            gL = gL + self.foil["ifdl"] * self.foil["eta"]
        self.boundary_conditions[self.Tc] = {
            "left": (-gL * pybamm.boundary_value(Rth, "left"), "Neumann"),
            "right": (gR * pybamm.boundary_value(Rth, "right"), "Neumann"),
        }
        self.initial_conditions[self.Tc] = pybamm.Scalar(self.T0)

    # -- charge limits ----------------------------------------------------------

    def _monitors(self):
        """Monitored quantity of each limit (genFOM addChargeLimits locations)."""
        mon = {"Vmax": self.Vcell,
               "thetaeMin": pybamm.boundary_value(self.thetae, "left"),
               "thetaeMax": pybamm.boundary_value(self.thetae, "right")}
        if self.half:
            mon["phiseMin"] = self.out["foil_phise"]
        else:
            mon["phiseMin"] = pybamm.boundary_value(self.out["neg_phise"], "right")
        mon["phiseMax"] = pybamm.boundary_value(self.out["pos_phise"], "left")
        return mon

    def _limits(self):
        lim = self.options.limits
        self.mode_flags = {}
        self.indicators = {}
        if lim.SOCLimit is not None:
            s = np.sign(lim.SOCLimit - self.z0) or 1.0
            self.events.append(pybamm.Event("SOC limit", s * (lim.SOCLimit - self.SOC)))
        if not lim.bounds:
            return
        mon = self._monitors()
        cc = 1
        residual = 0
        piXi = None
        for name, s in lim.bounds:
            flag = pybamm.InputParameter(f"mode {name}")
            self.mode_flags[name] = flag
            cc = cc - flag
            thr = getattr(lim, name)
            g = s * (thr - mon[name])                         # > 0 satisfied, < 0 violated
            self.indicators[name] = g
            if name.startswith("thetae"):
                if piXi is None:
                    piXi = pybamm.Variable("piXi")
                    self.piXi = piXi
                absI = pybamm.AbsoluteValue(self.Ides) + 1e-12
                residual = residual + flag * ((self.Iapp - self.Ides) / absI + lim.thetaePinKp * g - piXi)
            else:
                residual = residual + flag * (mon[name] - thr)
            self.events.append(pybamm.Event(f"limit {name}", g + BIG * flag))
        residual = residual + cc * (self.Iapp - self.Ides)
        self.algebraic[self.Iapp] = residual
        self.initial_conditions[self.Iapp] = self.Ides
        if piXi is not None:
            th = [n for n, _ in lim.bounds if n.startswith("thetae")]
            self.rhs[piXi] = -sum(self.mode_flags[n] * self.indicators[n] for n in th) / lim.thetaePinTau
            self.initial_conditions[piXi] = pybamm.Scalar(0)
        # Release: a potential limit holds while it restricts the current (Iapp >= Ides);
        # a theta_e limit while its integrator is non-negative.
        pot = sum((self.mode_flags[n] for n, _ in lim.bounds if not n.startswith("thetae")), pybamm.Scalar(0))
        self.events.append(pybamm.Event("limit release", (self.Iapp - self.Ides) + BIG * (1 - pot)))
        if piXi is not None:
            th_on = sum(self.mode_flags[n] for n, _ in lim.bounds if n.startswith("thetae"))
            self.events.append(pybamm.Event("limit release thetae", piXi + 1e-9 + BIG * (1 - th_on)))
