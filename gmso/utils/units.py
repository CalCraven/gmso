"""Source of available units registered within GMSO."""

import re

import numpy as np
import unyt as u
from sympy import Symbol

from gmso.exceptions import NotYetImplementedWarning


class GMSO_UnitRegistry(object):
    """A default unit registry class.

    The basic units that need to be added for various unit conversions done
    throughout GMSO.

    Attributes
    ----------
    reg : u.UnitRegistry
        The unit registry useful for conversions commonly used in molecular topologies
    """

    def __init__(self):
        self.reg_ = u.UnitRegistry()
        register_general_units(self.reg)

    def register_unit(
        self,
        name: str,
        conversion: float,
        dimensionsList: list,
        tex_repr=None,
    ):
        """Add units to the self.reg UnitRegistry.

        Parameters
        ----------
        registry : u.unyt_registy, required
            Unit registry to add the unit to. See unyt.unyt_registry for more information
        dimensionsList : list, required
            A list of the dimensions that the unit will be registered under. If using the inverse of a dimension
            be sure to supply 1/u.dimension as the element of the list.
        conversion : float, required
            The numerical value for the conversion in SI units with the same dimensions. See unyt.unyt_registry.add
            module for more information
        name : str, required
            Then name of the unyt to be referenced as string when calling u.Unit("unit_name")
        tex_repr : str, optional, default None
            The latex representation that is used to visualze the unit when pretty print is used.


        """
        dim = np.prod(dimensionsList)
        if not tex_repr:
            tex_repr = r"\rm{name}"
        self.reg_.add(
            symbol=name,
            base_value=conversion,
            dimensions=dim,
            tex_repr=tex_repr,
        )

    @property
    def reg(self):
        """Return the UnitRegistry attribute for the class."""
        return self.__dict__.get("reg_")

    @staticmethod
    def default_reg():
        """Return a default registry with extra units defined outside of unyt.

        Returns
        -------
        reg : u.unyt_registy
            A unyt registry with commonly used conversions defined.
        """
        reg = u.UnitRegistry()
        register_general_units(reg)
        return reg


@staticmethod
def register_general_units(reg: u.UnitRegistry):
    """Register units that are generally useful to a basic unyt UnitSystem."""
    conversion = 1 * getattr(u.physical_constants, "elementary_charge").value
    dim = u.dimensions.current_mks * u.dimensions.time
    reg.add(
        "elementary_charge",
        conversion,
        dim,
        r"\rm{e}",
    )  # proton charge
    conversion = (
        1 * getattr(u.physical_constants, "boltzmann_constant_mks").value
    )
    dim = u.dimensions.energy / u.dimensions.temperature
    reg.add(
        "kb", base_value=conversion, dimensions=dim, tex_repr=r"\rm{kb}"
    )  # boltzmann temperature
    conversion = (
        4
        * np.pi
        * getattr(u.physical_constants, "reduced_planck_constant").value ** 2
        * getattr(u.physical_constants, "eps_0").value
        / (
            getattr(u.physical_constants, "electron_charge").value ** 2
            * getattr(u.physical_constants, "electron_mass").value
        )
    )
    dim = u.dimensions.length
    reg.add(
        "a0", base_value=conversion, dimensions=dim, tex_repr=r"\rm{a0}"
    )  # bohr radius
    conversion = (
        getattr(u.physical_constants, "reduced_planck_constant").value ** 2
        / u.Unit("a0", registry=reg).base_value ** 2
        / getattr(u.physical_constants, "electron_mass").value
    )
    dim = u.dimensions.energy
    reg.add(
        "Ehartree",
        base_value=conversion,
        dimensions=dim,
        tex_repr=r"\rm{Ehartree}",
    )  # Hartree energy
    conversion = np.sqrt(
        10**9 / (4 * np.pi * getattr(u.physical_constants, "eps_0").value)
    )
    dim = u.dimensions.charge
    reg.add(
        "Statcoulomb_charge",
        base_value=conversion,
        dimensions=dim,
        tex_repr=r"\rm{Statcoulomb_charge}",
    )  # Static charge


class LAMMPS_UnitSystems:
    """Set of a unit systems distributed in LAMMPS (https://docs.lammps.org/units.html)."""

    def __init__(self, style: str, registry=None):
        if registry:
            self.reg_ = registry
        else:
            self.reg_ = GMSO_UnitRegistry().reg
        self.system_ = self.usystem_from_str(styleStr=style, reg=self.reg_)

    @property
    def system(self):
        """Return the UnitSystem attribute for the class."""
        return self.__dict__.get("system_")

    @property
    def reg(self):
        """Return the UnytRegistry attribute for the class."""
        return self.__dict__.get("reg_")

    def usystem_from_str(self, styleStr: str, reg: u.UnitRegistry):
        """Get systems for unit style."""
        #  NOTE: the when an angle is measured in lammps is not straightforwards. It depends not on the unit_style, but on the
        # angle_style, dihedral_style, or improper_style. For examples, harmonic angles, k is specificed in energy/radian, but the
        # theta_eq is written in degrees. For fourier dihedrals, d_eq is specified in degrees. When adding new styles, make sure that
        # this behavior is accounted for when converting the specific potential_type in the function
        # _parameter_converted_to_float
        if styleStr == "real":
            base_units = u.UnitSystem(
                "lammps_real", "Å", "amu", "fs", "K", "rad", registry=reg
            )
            base_units["energy"] = "kcal/mol"
            base_units["charge"] = "elementary_charge"
        elif styleStr == "metal":
            base_units = u.UnitSystem(
                "lammps_metal",
                "Å",
                "amu",
                "picosecond",
                "K",
                "rad",
                registry=reg,
            )
            base_units["energy"] = "eV"
            base_units["charge"] = "elementary_charge"
        elif styleStr == "si":
            base_units = u.UnitSystem(
                "lammps_si", "m", "kg", "s", "K", "rad", registry=reg
            )
            base_units["energy"] = "joule"
            base_units["charge"] = "coulomb"
        elif styleStr == "cgs":
            base_units = u.UnitSystem(
                "lammps_cgs", "cm", "g", "s", "K", "rad", registry=reg
            )
            base_units["energy"] = "erg"
            # Statcoulomb is strange. It is not a 1:1 correspondance to charge, with base units of
            # mass**1/2*length**3/2*time**-1.
            # However, assuming it is referring to a static charge and not a flux, it can be
            # converted to coulomb units. See the registry for the unit conversion to Coulombs
            base_units["charge"] = "Statcoulomb_charge"
        elif styleStr == "electron":
            base_units = u.UnitSystem(
                "lammps_electron", "a0", "amu", "s", "K", "rad", registry=reg
            )
            base_units["energy"] = "Ehartree"
            base_units["charge"] = "elementary_charge"
        elif styleStr == "micro":
            base_units = u.UnitSystem(
                "lammps_micro", "um", "picogram", "us", "K", "rad", registry=reg
            )
            base_units["energy"] = "ug*um**2/us**2"
            base_units["charge"] = "picocoulomb"
        elif styleStr == "nano":
            base_units = u.UnitSystem(
                "lammps_nano", "nm", "attogram", "ns", "K", "rad", registry=reg
            )
            base_units["energy"] = "attogram*nm**2/ns**2"
            base_units["charge"] = "elementary_charge"
        elif styleStr == "lj":
            base_units = ljUnitSystem(reg)
        else:
            raise NotYetImplementedWarning

        return base_units

    def convert_parameter(
        self,
        parameter,
        conversion_factorDict=None,
        n_decimals=3,
        name="",
    ):
        """Take a given parameter, and return a string of the parameter in the given style.

        This function will check the base_unyts, which is a unyt.UnitSystem object,
        and convert the parameter to those units based on its dimensions. It can
        also generate dimensionless units via normalization from conversion_factorsDict.
        """
        # TODO: need to handle thermal conversion
        if name in [
            "theta_eq",
            "chieq",
            "phi_eq",
        ]:  # eq angle are always in degrees
            return f"{round(float(parameter.to('degree').value), n_decimals):.{n_decimals}f}"
        new_dims = self._get_output_dimensions(parameter.units.dimensions)
        if isinstance(self.system, ljUnitSystem):
            if not conversion_factorDict:
                raise ValueError(
                    "Missing conversion_factorDict for a dimensionless unit system."
                )
            elif not np.all(
                [
                    key in conversion_factorDict
                    for key in ["energy", "length", "mass", "charge"]
                ]
            ):
                raise ValueError(
                    f"Missing dimensionless constant in conversion_factorDict {conversion_factorDict}"
                )
            # multiply object -> split into length, mass, energy, charge -> grab conversion factor from dict
            # first replace energy for (length)**2*(mass)/(time)**2 u.dimensions.energy. Then iterate through the free symbols
            # and figure out a way how to add those to the overall conversion factor
            dim_info = new_dims.as_terms()
            conversion_factor = 1
            for exponent, ind_dim in zip(dim_info[0][0][1][1], dim_info[1]):
                factor = conversion_factorDict.get(
                    ind_dim.name[1:-1],
                    1 * self.system[ind_dim.name[1:-1]],  # default value of 1
                )  # replace () in name
                current_unit = get_parameter_dimension(parameter, ind_dim.name)
                factor = factor.to(
                    current_unit
                )  # convert factor to units of parameter
                conversion_factor *= float(factor) ** (exponent)
            return f"""{round(
                float(parameter / conversion_factor),
                n_decimals
            ):.{n_decimals}f}"""  # Assuming that conversion factor is in right units
        new_dimStr = str(new_dims)
        ind_units = re.sub("[^a-zA-Z]+", " ", new_dimStr).split()
        for unit in ind_units:
            new_dimStr = new_dimStr.replace(unit, str(self.system[unit]))
        outFloat = float(
            parameter.to(u.Unit(new_dimStr, registry=self.system.registry))
        )

        return f"{outFloat:.{n_decimals}f}"

    @staticmethod
    def _dimensions_to_energy(dims):
        """Take a set of dimensions and substitute in Symbol("energy") where possible."""
        symsStr = str(dims.free_symbols)
        energy_inBool = np.all(
            [dimStr in symsStr for dimStr in ["time", "mass"]]
        )
        if not energy_inBool:
            return dims
        energySym = Symbol(
            "(energy)"
        )  # create dummy symbol to replace in equation
        dim_info = dims.as_terms()
        time_idx = np.where(
            list(map(lambda x: x.name == "(time)", dim_info[1]))
        )[0][0]
        energy_exp = (
            dim_info[0][0][1][1][time_idx] // 2
        )  # energy has 1/time**2 in it, so this is the hint of how many
        return (
            dims
            * u.dimensions.energy**energy_exp
            * energySym ** (-1 * energy_exp)
        )

    @staticmethod
    def _dimensions_to_charge(dims):
        """Take a set of dimensions and substitute in Symbol("charge") where possible."""
        symsStr = str(dims.free_symbols)
        charge_inBool = np.all(
            [dimStr in symsStr for dimStr in ["current_mks"]]
        )
        if not charge_inBool:
            return dims
        chargeSym = Symbol(
            "(charge)"
        )  # create dummy symbol to replace in equation
        dim_info = dims.as_terms()
        current_idx = np.where(
            list(map(lambda x: x.name == "(current_mks)", dim_info[1]))
        )[0][0]
        charge_exp = dim_info[0][0][1][1][
            current_idx
        ]  # charge has (current_mks) in it, so this is the hint of how many
        return (
            dims
            * u.dimensions.charge ** (-1 * charge_exp)
            * chargeSym**charge_exp
        )

    @staticmethod
    def _dimensions_from_thermal_to_energy(dims):
        """Take a set of dimensions and substitute in Symbol("energy") to replace temperature."""
        symsStr = str(dims.free_symbols)
        temp_inBool = np.all([dimStr in symsStr for dimStr in ["temperature"]])
        if not temp_inBool:
            return dims
        energySym = Symbol(
            "(energy)"
        )  # create dummy symbol to replace in equation
        dim_info = dims.as_terms()
        temp_idx = np.where(
            list(map(lambda x: x.name == "(temperature)", dim_info[1]))
        )[0][0]
        temp_exp = dim_info[0][0][1][1][
            temp_idx
        ]  # energy has 1/time**2 in it, so this is the hint of how many
        return (
            dims
            / u.dimensions.temperature**temp_exp
            * energySym ** (temp_exp)
        )

    @classmethod
    def _get_output_dimensions(cls, dims, thermal_equivalence=False):
        if str(dims) == "1":  # use string as all dims can be converted
            return u.dimensionless
        dims = cls._dimensions_to_energy(dims)
        dims = cls._dimensions_to_charge(dims)
        if thermal_equivalence:
            dims = cls._dimensions_from_thermal_to_energy(dims)
        return dims


class ljUnitSystem:
    """Use this so the empty unitsystem has getitem magic method."""

    def __init__(self, reg: u.UnitRegistry):
        self.registry = reg
        self.name = "lj"

    def __getitem__(self, item):
        """Return dimensionless units unless angle."""
        if item == "angle":
            return u.Unit("degree")
        return u.Unit("dimensionless")


def get_parameter_dimension(parameter, dimension):
    """Return a unit from the parameter in a given dimension."""
    param_terms = parameter.units.expr.as_terms()
    uStr = ""
    for symbol, exp in zip(param_terms[-1], param_terms[0][0][1][1]):
        outputDim = LAMMPS_UnitSystems._get_output_dimensions(
            u.Unit(symbol).dimensions
        )
        if str(outputDim) == dimension:
            uStr += f"{symbol}*"
        elif (
            str(outputDim) == "dimensionless" and dimension == "(energy)"
        ):  # add mol to units of energy
            uStr += f"{symbol}**{exp}*"
        elif (
            str(outputDim) == "dimensionless" and dimension == "(mass)"
        ):  # add mol to mass amu
            uStr += f"{symbol}**{exp}*"
    return u.Unit(uStr[:-1])
