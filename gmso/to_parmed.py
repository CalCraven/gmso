import numpy as np
import unyt as u
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
import warnings
import parmed as pmd

from gmso.utils.io import import_, has_parmed
from gmso.core.element import element_by_name, element_by_symbol, element_by_atom_type

def to_parmed(top, refer_type=True):
    """Convert a topology.Topology to a parmed.Structure

    At this point we only assume a three level structure for topology
    Topology - Subtopology - Sites, which transform to three level of
    Parmed Structure - Residue - Atoms.
    If we decide to support multiple level Subtopology in the future,
    this method will need some re-work. Tentative plan is to have the
    Parmed Residue to be equivalent to the Subtopology right above Site.

    Parameters
    ----------
    top : topology.Topology
        topology.Topology instance that need to be converted
    refer_type : bool, optional, default=True
        Whether or not to transfer AtomType, BondType, AngleTye,
        and DihedralType information

    Returns
    -------
    structure : parmed.Structure
    """

    # Set up Parmed structure and define general properties
    structure = pmd.Structure()
    structure.title = top.name
    structure.box = np.concatenate((top.box.lengths.to('angstrom').value,
                                    top.box.angles.to('degree').value))

    # Maps
    subtop_map = dict() # Map site to subtop
    atom_map = dict() # Map site to atom
    bond_map = dict() # Map top's bond to structure's bond
    angle_map = dict() # Map top's angle to strucutre's angle
    dihedral_map = dict() # Map top's dihedral to structure's dihedral

    # Set up unparametrized system
    # Build subtop_map (site -> top)
    default_residue = pmd.Residue('RES')
    for subtop in top.subtops:
        for site in subtop.sites:
            subtop_map[site] = subtop

    # Build up atom
    for site in top.sites:
        if site in subtop_map:
            residue = subtop_map[site].name
            residue_name = residue[:residue.find('[')]
            residue_idx = int(residue[residue.find('[')+1:residue.find(']')])
            #since subtop contains information needed to build residue
        else:
            residue = default_residue
        # Check element
        if site.element:
            atomic_number = site.element.atomic_number
            charge = site.element.charge
        else:
            atomic_number = 0
            charge = 0

        pmd_atom = pmd.Atom(atomic_number=atomic_number, name=site.name,
                            mass=site.mass, charge=site.charge)
        pmd_atom.xx, pmd_atom.xy, pmd_atom.xz = site.position.to('angstrom').value

        # Add atom to structure
        structure.add_atom(pmd_atom, resname=residue_name, resnum=residue_idx)
        atom_map[site] = pmd_atom

    # "Claim" all of the item it contains and subsequently index all of its item
    structure.residues.claim()

    # Create and add bonds to Parmed structure
    for bond in top.bonds:
        site1, site2 = bond.connection_members
        pmd_bond = pmd.Bond(atom_map[site1], atom_map[site2])
        structure.bonds.append(pmd_bond)
        bond_map[bond] = pmd_bond

    # Create and add angles to Parmed structure
    for angle in top.angles:
        site1, site2, site3 = angle.connection_members
        pmd_angle = pmd.Angle(atom_map[site1],
                              atom_map[site2],
                              atom_map[site3])
        structure.angles.append(pmd_angle)
        angle_map[angle] = pmd_angle

    # Create and add dihedrals to Parmed structure

    for dihedral in top.dihedrals:
        site1, site2, site3, site4 = dihedral.connection_members
        pmd_dihedral = pmd.Dihedral(atom_map[site1],
                                      atom_map[site2],
                                      atom_map[site3],
                                      atom_map[site4])
        if (dihedral.connection_type and
         dihedral.connection_type.expression == parse_expr(
                                        'c0 * cos(phi)**0 + ' +
                                        'c1 * cos(phi)**1 + ' +
                                        'c2 * cos(phi)**2 + ' +
                                        'c3 * cos(phi)**3 + ' +
                                        'c4 * cos(phi)**4 + ' +
                                        'c5 * cos(phi)**5')):
            structure.rb_torsions.append(pmd_dihedral)
        else:
            structure.dihedrals.append(pmd_dihedral)
        dihedral_map[dihedral] = pmd_dihedral

    # Set up structure for Connection Type conversion
    if refer_type:
    # Need to add a warning if Topology does not have types information
        if top.atom_types:
            _atom_types_from_gmso(top, structure, atom_map)
        if top.bond_types:
            _bond_types_from_gmso(top, structure, bond_map)
        if top.angle_types:
            _angle_types_from_gmso(top, structure, angle_map)
        if top.dihedral_types:
            _dihedral_types_from_gmso(top, structure, dihedral_map)

    return structure

def _atom_types_from_gmso(top, structure, atom_map):
    """Helper function to convert Topology AtomType to Structure AtomType

    This function will first check the AtomType expression of Topology and make sure it match with the one default in Parmed.
    After that, it would start atomtyping and parametrizing this part of the structure.

    Parameters
    ----------
    top : topology.Topology
        The topology that need to be converted
    structure: parmed.Structure
        The destination parmed Structure
    """
    # Maps
    atype_map = dict()
    for atom_type in top.atom_types:
        msg = "Atom type {} expression does not match Parmed AtomType default expression".format(atom_type.name)
        assert atom_type.expression == parse_expr("4*epsilon*(-sigma**6/r**6 + sigma**12/r**12)"), msg
        # Extract Topology atom type information
        atype_name = atom_type.name
        # Convert charge to elementary_charge
        atype_charge = float(atom_type.charge.to('Coulomb').value) / (1.6 * 10**(-19))
        atype_sigma = float(atom_type.parameters['sigma'].to('angstrom').value)
        atype_epsilon = float(atom_type.parameters['epsilon'].to('kcal/mol').value)
        atype_element = element_by_atom_type(atom_type)
        atype_rmin = atype_sigma * 2**(1/6) / 2 # to rmin/2
        # Create unique Parmed AtomType object
        atype = pmd.AtomType(atype_name, None, atype_element.mass,
                             atype_element.atomic_number, atype_charge)
        atype.set_lj_params(atype_epsilon, atype_rmin)
        # Type map to match AtomType to its name
        atype_map[atype_name] = atype

    for site in top.sites:
        #Assign atom_type to atom
        pmd_atom = atom_map[site]
        pmd_atom.type = site.name
        #comment out so atom_type is saved as atom_name for py3dmol viewing
        #pmd_atom.atom_type = atype_map[site.atom_type.name]
        pmd_atom.atom_type = site.name

def _bond_types_from_gmso(top, structure, bond_map):
    """Helper function to convert Topology BondType to Structure BondType

    This function will first check the BondType expression of Topology and make sure it match with the one default in Parmed.
    After that, it would start atomtyping and parametrizing this part of the structure.

    Parameters
    ----------
    top : topology.Topology
        The topology that need to be converted
    structure: parmed.Structure
        The destination parmed Structure
    """
    btype_map = dict()
    for bond_type in top.bond_types:
        msg = "Bond type {} expression does not match Parmed BondType default expression".format(bond_type.name)
        assert bond_type.expression == parse_expr("0.5 * k * (r-r_eq)**2"), msg
        # Extract Topology bond_type information
        btype_k =  0.5 * float(bond_type.parameters['k'].to('kcal / (angstrom**2 * mol)').value)
        btype_r_eq = float(bond_type.parameters['r_eq'].to('angstrom').value)
        # Create unique Parmed BondType object
        btype = pmd.BondType(btype_k, btype_r_eq)
        # Type map to match Topology BondType with Parmed BondType
        btype_map[bond_type] = btype
        # Add BondType to structure.bond_types
        structure.bond_types.append(btype)

    for bond in top.bonds:
        #Assign bond_type to bond
        pmd_bond = bond_map[bond]
        pmd_bond.type = btype_map[bond.connection_type]
    structure.bond_types.claim()

def _angle_types_from_gmso(top, structure, angle_map):
    """Helper function to convert Topology AngleType to Structure AngleType

    This function will first check the AngleType expression of Topology and make sure it match with the one default in Parmed.
    After that, it would start atomtyping and parametrizing the structure.

    Parameters
    ----------
    top : topology.Topology
        The topology that need to be converted
    structure: parmed.Structure
        The destination parmed Structure
    """
    agltype_map = dict()
    for angle_type in top.angle_types:
        msg = "Angle type {} expression does not match Parmed AngleType default expression".format(angle_type.name)
        assert angle_type.expression == parse_expr("0.5 * k * (theta-theta_eq)**2"), msg
        # Extract Topology angle_type information
        agltype_k = 0.5 * float(angle_type.parameters['k'].to('kcal / (rad**2 * mol)').value)
        agltype_theta_eq = float(angle_type.parameters['theta_eq'].to('degree').value)
        # Create unique Parmed AngleType object
        agltype = pmd.AngleType(agltype_k, agltype_theta_eq)
        # Type map to match Topology AngleType with Parmed AngleType
        agltype_map[angle_type] = agltype
        # Add AngleType to structure.angle_types
        structure.angle_types.append(agltype)

    for angle in top.angles:
        #Assign angle_type to angle
        pmd_angle = angle_map[angle]
        pmd_angle.type = agltype_map[angle.connection_type]
    structure.angle_types.claim()

def _dihedral_types_from_gmso(top, structure, dihedral_map):
    """Helper function to convert Topology DihedralType to Structure DihedralType

    This function will first check the DihedralType expression of Topology and
    make sure it match with the one default in Parmed.
    After that, it would start atomtyping and parametrizing the structure.

    Parameters
    ----------
    top : topology.Topology
        The topology that need to be converted
    structure: parmed.Structure
        The destination parmed Structure
    """
    dtype_map = dict()
    for dihedral_type in top.dihedral_types:
        msg = "Dihedral type {} expression does not match Parmed DihedralType default expressions (Periodics, RBTorsions)".format(dihedral_type.name)
        if dihedral_type.expression == parse_expr('k * (1 + cos(n * phi - phi_eq))**2'):
            dtype_k = float(dihedral_type.parameters['k'].to('kcal/mol').value)
            dtype_phi_eq = float(dihedral_type.parameters['phi_eq'].to('degrees').value)
            dtype_n = float(dihedral_type.parameters['n'].value)
            # Create unique Parmed DihedralType object
            dtype = pmd.DihedralType(dtype_k, dtype_n, dtype_phi_eq)
            # Add DihedralType to structure.dihedral_types
            structure.dihedral_types.append(dtype)
        elif (dihedral_type.expression == parse_expr(
                                                  'c0 * cos(phi)**0 + ' +
                                                  'c1 * cos(phi)**1 + ' +
                                                  'c2 * cos(phi)**2 + ' +
                                                  'c3 * cos(phi)**3 + ' +
                                                  'c4 * cos(phi)**4 + ' +
                                                  'c5 * cos(phi)**5')):
            dtype_c0 = float(dihedral_type.parameters['c0'].to('kcal/mol').value)
            dtype_c1 = float(dihedral_type.parameters['c1'].to('kcal/mol').value)
            dtype_c2 = float(dihedral_type.parameters['c2'].to('kcal/mol').value)
            dtype_c3 = float(dihedral_type.parameters['c3'].to('kcal/mol').value)
            dtype_c4 = float(dihedral_type.parameters['c4'].to('kcal/mol').value)
            dtype_c5 = float(dihedral_type.parameters['c5'].to('kcal/mol').value)
            # Create unique DihedralType object
            dtype = pmd.RBTorsionType(dtype_c0, dtype_c1, dtype_c2,
                                      dtype_c3, dtype_c4, dtype_c5)
            # Add RBTorsionType to structure.rb_torsion_types
            structure.rb_torsion_types.append(dtype)
        else:
            raise GMSOException('msg')
        dtype_map[dihedral_type] = dtype

    for dihedral in top.dihedrals:
        pmd_dihedral = dihedral_map[dihedral]
        pmd_dihedral.type = dtype_map[dihedral.connection_type]
    structure.dihedral_types.claim()
    structure.rb_torsions.claim()
