import lxml
import pytest
import unyt as u
from lxml.etree import DocumentInvalid
from sympy import sympify
import os
import glob
import filecmp

from gmso.core.forcefield import ForceField
from gmso.core.improper_type import ImproperType
from gmso.exceptions import (
    ForceFieldError,
    ForceFieldParseError,
    MissingAtomTypesError,
    MissingPotentialError,
)
from gmso.utils.io import get_fn
from gmso.tests.base_test import BaseTest
from gmso.tests.utils import allclose_units_mixed, get_path

from forcefield_utilities import GMSOFFs

# Make source directory for all xmls to grab from
XML_DIR = get_fn("gmso_xmls")
TEST_XMLS = glob.glob(os.path.join(XML_DIR, "*.xml"))

def compare_xml_files(fn1, fn2):
    """Hash files to check for lossless conversion."""
    # TODO: this requires the files look the same, might be a smarter way
    return filecmp.cmp(fn1, fn2)

def are_equivalent_ffs(ff1, ff2):
    """Compare forcefields for equivalency"""
    # TODO: write __eq__ method for a forcefield
    return ff1 == ff2

class TestXMLHandling(BaseTest):
    @pytest.fixture
    def ff(self):
        return ForceField(get_path("ff-example0.xml"))

    @pytest.fixture
    def named_groups_ff(self):
        return ForceField(get_path("ff-example1.xml"))

    @pytest.fixture
    def opls_ethane_foyer(self):
        return ForceField(
            get_path(filename=get_path("oplsaa-ethane_foyer.xml"))
        )

    def test_write_xml(self, opls_ethane_foyer):
        opls_ethane_foyer.to_xml("test_xml_writer.xml")
        reloaded_xml = ForceField("test_xml_writer.xml")
        get_names = lambda ff, param: [
            typed for typed in getattr(ff, param).keys()
        ]
        for param in [
            "atom_types",
            "bond_types",
            "angle_types",
            "dihedral_types",
        ]:
            assert get_names(opls_ethane_foyer, param) == get_names(
                reloaded_xml, param
            )

    def test_foyer_xml_conversion(self):
        """Validate xml converted from Foyer can be written out correctly."""
        pass

    def test_write_xml_from_topology(self):
        """Validate xml from a typed topology matches loaded xmls."""
        pass

    @pytest.mark.parametrize("xml", TEST_XMLS)
    def test_load__direct_from_forcefield_utilities(self, xml):
        """Validate loaded xmls from ff-utils match original file."""
        ff = GMSOFFs().load_xml(xml).to_gmso_ff()
        assert isinstance(ff, ForceField)

    @pytest.mark.parametrize("xml", TEST_XMLS)
    def test_ffutils_backend(self, xml):
        ff = ForceField()
        ff.load_backend_forcefield_utilities(xml)
        assert isinstance(ff, ForceField)
        ff = ForceField().load_backend_forcefield_utilities(xml)
        assert isinstance(ff, ForceField)
        ff = ForceField(xml, backend="forcefield-utilities")
        assert isinstance(ff, ForceField)

    @pytest.mark.parametrize("xml", TEST_XMLS)
    def test_gmso_backend(self, xml):
        ff = ForceField(xml)
        assert isinstance(ff, ForceField)

    @pytest.mark.parametrize("xml", TEST_XMLS)
    def test_load_write_xmls_gmso_backend(self, xml):
        """Validate loaded xmls written out match original file."""
        ff1 = ForceField(xml)
        ff1.to_xml("tmp.xml", overwrite=True)
        ff2 = ForceField("tmp.xml")
        assert validate_xmls("tmp.xml", xml)
        assert ff1 == ff2

    @pytest.mark.parametrize("xml", TEST_XMLS)
    def test_load_write_xmls_ffutils_backend(self, xml):
        """Validate loaded xmls written out match original file."""
        ff1 = ForceField()
        ff1.load_backend_forcefield_utilities(xml)
        ff1.to_xml("tmp.xml", overwrite=True)
        ff2 = GMSOFFs().load_xml("tmp.xml").to_gmso_ff()
        assert compare_xml_files("tmp.xml", xml)
        assert are_equivalent_ffs(ff1, ff2)

    def test_xml_error_handling(self):
        """Validate bad xml formatting in xmls."""
        pass

