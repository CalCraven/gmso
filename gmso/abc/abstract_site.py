"""Basic interaction site in GMSO that all other sites will derive from."""

import warnings
from typing import Any, ClassVar, Optional, Sequence, TypeVar, Union

import numpy as np
import unyt as u
from pydantic import (
    ConfigDict,
    Field,
    StrictStr,
    field_serializer,
    field_validator,
)
from unyt.exceptions import InvalidUnitOperation

from gmso.abc.gmso_base import GMSOBase
from gmso.abc.serialization_utils import unyt_to_dict
from gmso.exceptions import GMSOError

PositionType = Union[Sequence[float], np.ndarray, u.unyt_array]


class Molecule(GMSOBase):
    def __repr__(self):
        return (
            f"Molecule(name={self.name}, number={self.number}, isrigid={self.isrigid})"
        )

    __iterable_attributes__: ClassVar[set] = {
        "name",
        "number",
        "isrigid",
    }

    __base_doc__: ClassVar[str] = "Molecule label for interaction sites."

    name_: str = Field(
        "",
        validate_default=True,
        description="Name of the molecule",
        alias="name",
    )
    number_: int = Field(
        0,
        description="The index/number of the molecule",
        alias="number",
    )
    isrigid_: bool = Field(
        False,
        description="Indicate whether the molecule is rigid",
    )
    model_config = ConfigDict(
        alias_to_fields={
            "name": "name_",
            "number": "number_",
            "isrigid": "isrigid_",
        }
    )

    @property
    def name(self) -> str:
        """Return the name of the molecule."""
        return self.__dict__.get("name_")

    @property
    def number(self) -> int:
        """Return the index/number of the moleucle."""
        return self.__dict__.get("number_")

    @property
    def isrigid(self) -> bool:
        """Return the rigid label of the molecule."""
        return self.__dict__.get("isrigid_")

    def __hash__(self):
        return hash(tuple([(name, val) for name, val in self.__dict__.items()]))

    def __eq__(self, other):
        """Test if two objects are equivalent."""
        if isinstance(other, (list, tuple)):
            return all(
                [val1 == val2 for val1, val2 in zip(self.__dict__.values(), other)]
            )
        else:
            return self.__dict__ == other.__dict__


class Residue(GMSOBase):
    def __repr__(self):
        return f"Residue(name={self.name}, number={self.number}"

    __iterable_attributes__: ClassVar[set] = {
        "name",
        "number",
    }

    __base_doc__: ClassVar[str] = "Residue label for interaction sites."

    name_: str = Field(
        "",
        validate_default=True,
        description="Name of the residue",
        alias="name",
    )
    number_: int = Field(
        0,
        description="The index/number of the residue",
        alias="number",
    )
    model_config = ConfigDict(
        alias_to_fields={
            "name": "name_",
            "number": "number_",
        }
    )

    @property
    def name(self) -> str:
        """Return the name of the residue."""
        return self.__dict__.get("name_")

    @property
    def number(self) -> int:
        """Return the index/number of the residue."""
        return self.__dict__.get("number_")

    def __hash__(self):
        return hash(tuple([(name, val) for name, val in self.__dict__.items()]))

    def __eq__(self, other):
        """Test if two objects are equivalent."""
        if isinstance(other, (list, tuple)):
            return all(
                [val1 == val2 for val1, val2 in zip(self.__dict__.values(), other)]
            )
        else:
            return self.__dict__ == other.__dict__


SiteT = TypeVar("SiteT", bound="Site")

BASE_DOC_ATTR = "__base_doc__"
FIELDS_IN_DOCSTRING = "alias_to_fields"


def default_position():
    return u.unyt_array([np.nan] * 3, u.nm)


class Site(GMSOBase):
    def __repr__(self):
        """Return the formatted representation of the site."""
        return (
            f"<{self.__class__.__name__} {self.name},\n "
            f"position: {self.position},\n "
            f"label: {self.label if self.label else None},\n "
            f"Molecule: {self.molecule},\n"
            f"Residue: {self.residue},\n"
            f"id: {id(self)}>"
        )

    __iterable_attributes__: ClassVar[set] = {
        "name",
        "label",
        "group",
        "molecule",
        "residue",
        "position",
        "model_config",
    }

    __base_doc__: ClassVar[
        str
    ] = """An interaction site object in the topology hierarchy.

    Site is the object that represents any general interaction site in a molecular simulation.
    Sites have been designed to be as general as possible, making no assumptions about representing atoms or beads, or
    having mass or charge. That is, a Site can represent an atom in an atomistic system,
    a bead in a coarse-grained system, and much more.

    Notes
    -----
    The label attribute for a site takes its meaning when used with some sort of container (like topology)
    such that a label for a site can then be used to group sites together. The rules for defining a site label
    and their meaning the responsibility of the container where the sites will reside.
    """

    name_: str = Field(
        "",
        validate_default=True,
        description="Name of the site, defaults to class name",
        alias="name",
    )
    label_: str = Field(
        "", description="Label to be assigned to the site", alias="label"
    )

    group_: Optional[StrictStr] = Field(
        None,
        description="Flexible alternative label relative to site",
        alias="group",
    )

    molecule_: Optional[Union[Molecule, list, tuple]] = Field(
        None,
        description="Molecule label for the site, format of (molecule_name, molecule_number)",
        alias="molecule",
    )

    residue_: Optional[Union[Residue, list, tuple]] = Field(
        None,
        description="Residue label for the site, format of (residue_name, residue_number)",
        alias="residue",
    )

    position_: PositionType = Field(
        default_factory=default_position,
        description="The 3D Cartesian coordinates of the position of the site",
        alias="position",
    )

    model_config = ConfigDict(
        alias_to_fields={
            "name": "name_",
            "label": "label_",
            "group": "group_",
            "molecule": "molecule_",
            "residue": "residue_",
            "position": "position_",
        },
    )

    @property
    def name(self) -> str:
        """Return the name of the site."""
        return self.__dict__.get("name_")

    @property
    def position(self) -> u.unyt_array:
        """Return the 3D Cartesian coordinates of the site."""
        return self.__dict__.get("position_")

    @property
    def label(self) -> str:
        """Return the label assigned to the site."""
        return self.__dict__.get("label_")

    @property
    def group(self) -> str:
        """Return the group of the site."""
        return self.__dict__.get("group_")

    @property
    def molecule(self):
        """Return the molecule of the site."""
        return self.__dict__.get("molecule_")

    @property
    def residue(self):
        """Return the residue assigned to the site."""
        return self.__dict__.get("residue_")

    @field_serializer("position_")
    def serialize_position(self, position_: PositionType):
        return unyt_to_dict(position_)

    def __str__(self):
        """Return the string representation of the site."""
        return (
            f"<{self.__class__.__name__} {self.name}, "
            f"label: {self.label if self.label else None} id: {id(self)}>"
        )

    @field_validator("position_")
    @classmethod
    def is_valid_position(cls, position):
        """Validate attribute position."""
        if position is None:
            return u.unyt_array([np.nan] * 3, u.nm)

        if not isinstance(position, u.unyt_array):
            try:
                position *= u.nm
            except InvalidUnitOperation as e:
                raise GMSOError(
                    f"Converting object of type {type(position)} failed with following error: {e}"
                )
            warnings.warn("Positions are assumed to be in nm")

        try:
            position = np.reshape(position, newshape=(3,), order="C")
            if position.units != u.dimensionless:
                position.convert_to_units(u.nm)
        except ValueError:
            raise ValueError(
                f"Position of shape {position.shape} is not valid. "
                "Accepted values: (a.) list-like of length 3"
                "(b.) np.array or unyt.unyt_array of shape (3,)"
            )

        return position

    @field_validator("name_")
    def parse_name(cls, value):
        if value == "" or value is None:
            return cls.__name__
        else:
            return value

    @field_validator("residue_")
    def parse_residue(cls, value):
        if isinstance(value, (tuple, list)):
            assert len(value) == 2
            value = Residue(name=value[0], number=value[1])
        return value

    @field_validator("molecule_")
    def parse_molecule(cls, value):
        if isinstance(value, (tuple, list)):
            if len(value) == 2:
                value = Molecule(name=value[0], number=value[1])
            elif len(value) == 3:
                value = Molecule(name=value[0], number=value[1], isrigid=value[2])
        return value

    @classmethod
    def __new__(cls, *args: Any, **kwargs: Any) -> SiteT:
        if cls is Site:
            raise TypeError("Cannot instantiate abstract class of type Site")
        else:
            return object.__new__(cls)
