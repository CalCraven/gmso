import pytest
import numpy as np
import mbuild as mb
import mbuild.recipes
import unyt as u
import foyer

from gmso.core.box import Box
from gmso.core.topology import Topology
from gmso.core.element import Hydrogen, Oxygen
from gmso.core.site import Site
from gmso.core.angle import Angle
from gmso.core.atom_type import AtomType
from gmso.core.forcefield import ForceField
from gmso.external.convert_mbuild import from_mbuild
from gmso.tests.utils import get_path
from gmso.utils.io import get_fn
from gmso.external import from_parmed

class BaseTest:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()

    @pytest.fixture
    def lengths(self):
        return u.nm * np.ones(3)

    @pytest.fixture
    def angles(self):
        return u.degree * [90, 90, 90]

    @pytest.fixture
    def charge(self):
        return u.elementary_charge * 1

    @pytest.fixture
    def mass(self):
        return 1 * u.gram/u.mol

    @pytest.fixture
    def box(self):
        return Box(lengths=u.nm*np.ones(3))

    @pytest.fixture
    def top(self):
        return Topology(name='mytop')

    @pytest.fixture
    def topology_site(self):
        def _topology(sites=1):
            top = Topology()
            top.box = Box(lengths=[1, 1, 1])
            H = Hydrogen
            site1 = Site(name='site1',
                         element=H,
                         atom_type=AtomType(name="at1",
                                            mass=H.mass),
                         )
            for i in range(sites):
                top.add_site(site1)

            return top

        return _topology


    @pytest.fixture
    def ar_system(self):
        ar = mb.Compound(name='Ar')

        packed_system = mb.fill_box(
            compound=ar,
            n_compounds=100,
            box=mb.Box([3, 3, 3]),
        )

        return from_mbuild(packed_system)

    @pytest.fixture
    def typed_single_ar(self):
        top = from_mbuild(mb.Compound(name="Ar"))

        ff = ForceField(get_fn("ar.xml"))

        for site in top.sites:
            site.atom_type = ff.atom_types["Ar"]

        top.update_topology()
        return top

    @pytest.fixture
    def typed_single_xe_mie(self):
        top = from_mbuild(mb.Compound(name="Xe"))

        ff = ForceField(get_path("noble_mie.xml"))

        for site in top.sites:
            site.atom_type = ff.atom_types["Xe"]

        top.update_topology()
        return top

    @pytest.fixture
    def typed_ar_system(self, ar_system):
        top = ar_system

        ff = ForceField(get_fn('ar.xml'))

        for site in top.sites:
            site.atom_type = ff.atom_types['Ar']

        top.update_topology()

        return top

    @pytest.fixture
    def water_system(self):
        water = mb.load(get_path('tip3p.mol2'))
        water.name = 'water'
        water[0].name = 'opls_111'
        water[1].name = water[2].name = 'opls_112'

        packed_system = mb.fill_box(
                compound=water,
                n_compounds=2,
                box=mb.Box([2, 2, 2])
                )

        return  from_mbuild(packed_system)

    @pytest.fixture
    def parmed_methylnitroaniline(self):
        compound = mb.load('CC1=C(C=CC(=C1)[N+](=O)[O-])N', smiles=True)
        oplsaa = foyer.Forcefield(name='oplsaa')
        pmd_structure = oplsaa.apply(compound)
        return pmd_structure

    @pytest.fixture
    def typed_methylnitroaniline(self):
        compound = mb.load('CC1=C(C=CC(=C1)[N+](=O)[O-])N', smiles=True)
        oplsaa = foyer.Forcefield(name='oplsaa')
        pmd_structure = oplsaa.apply(compound)
        top = from_parmed(pmd_structure)
        return top

    @pytest.fixture
    def parmed_chloroethanol(self):
        compound = mb.load('C(CCl)O', smiles=True)
        oplsaa = foyer.Forcefield(name='oplsaa')
        pmd_structure = oplsaa.apply(compound)
        return pmd_structure

    @pytest.fixture
    def typed_chloroethanol(self):
        compound = mb.load('C(CCl)O', smiles=True)
        oplsaa = foyer.Forcefield(name='oplsaa')
        pmd_structure = oplsaa.apply(compound)
        top = from_parmed(pmd_structure)
        return top

    @pytest.fixture
    def parmed_hexane_box(self):
        compound = mb.recipes.Alkane(6)
        compound.name = "HEX"
        compound_box = mb.fill_box(compound, n_compounds=6, box=[6,6,6])
        oplsaa = foyer.Forcefield(name='oplsaa')
        pmd_structure = oplsaa.apply(compound_box, residues="HEX")
        return pmd_structure

    @pytest.fixture
    def typed_water_system(self, water_system):
        top = water_system

        ff = ForceField(get_path('tip3p.xml'))

        element_map = {"O": "opls_111", "H": "opls_112"}

        for atom in top.sites:
            atom.atom_type = ff.atom_types[atom.name]

        for bond in top.bonds:
            bond.connection_type = ff.bond_types["opls_111~opls_112"]

        for subtop in top.subtops:
            angle = Angle(
                connection_members=[site for site in subtop.sites],
                name="opls_112~opls_111~opls_112",
                connection_type=ff.angle_types["opls_112~opls_111~opls_112"]
            )
            top.add_connection(angle)

        top.update_topology()
        return top
