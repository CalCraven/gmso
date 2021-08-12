import numpy as np
import pytest
import unyt as u
from unyt.testing import assert_allclose_units

from gmso.external.convert_mol2 import from_mol2
from gmso.tests.base_test import BaseTest
from gmso.utils.io import get_fn


class TestMol2(BaseTest):
    def test_read_mol2(self):
        top = from_mol2(get_fn("parmed.mol2"))

        assert top.name == "parmed"
        assert top.n_sites == 8
        assert_allclose_units(
            top.box.lengths,
            ([8.2693, 7.9100, 6.6460] * u.Å).to("nm"),
            rtol=1e-5,
            atol=1e-8,
        )

        top = from_mol2(get_fn("tip3p.mol2"))

        assert top.name == "tip3p"
        assert top.n_sites == 3
        assert_allclose_units(
            top.box.lengths, 3.0130 * np.ones(3) * u.Å, rtol=1e-5, atol=1e-8
        )

        top = from_mol2(get_fn("vmd.mol2"))

        assert top.name == "vmd"
        assert top.n_sites == 6
        assert len(top.bonds) == 5
        assert top.bonds[0].connection_members[0] == top.sites[0]
        assert top.box == None

    def test_lj_system(self):
        top = from_mol2(get_fn("parmed.mol2"), site_type="lj")
        assert np.all([site.element == None for site in top.sites])

    def test_wrong_path(self):
        with pytest.raises(OSError) as e:
            from_mol2("not_a_file.mol2")
        assert e.value.args[0] == "Provided path to file that does not exist"
        top = from_mol2(get_fn("ethane.gro"))
        assert len(top.sites) == 0
        assert len(top.bonds) == 0

    def test_broken_files(self):
        match = "The record type indicator @<TRIPOS>MOLECULE_extra_text\n is not supported"
        with pytest.warns(UserWarning) as record:
            from_mol2(get_fn("broken.mol2"))
        assert record[0].message.args[0] == match
        match = "This mol2 file has two boxes to be read in, only reading in one with dimensions Box(a=0.72 nm, b=0.9337999999999999 nm, c=0.6646 nm, alpha=90.0 degree, beta=90.0 degree, gamma=90.0 degree)"
        assert record[-1].message.args[0] == match
