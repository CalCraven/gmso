"""Convert to and from a TRIPOS mol2 file."""
# TODO add sources of mol2 files
import os
import warnings

import unyt as u

from gmso import Atom, Bond, Box, Topology
from gmso.core.element import element_by_name, element_by_symbol


def from_mol2(
    filename, site_type="Atom"
):  # TODO add flags for information to return
    # TODO: descriptions and examples
    # TODO: Be sure to be clear about how to read in to mbuild compound using gmso.external.to_mbuild function

    msg = "Provided path to file that does not exist"
    if not os.path.isfile(filename):
        raise OSError(msg)

    # Initialize topology
    topology = Topology(name=os.path.splitext(os.path.basename(filename))[0])
    # save the name from the filename
    f = open(filename, "r")
    line = f.readline()
    while f:
        # check for header character in line
        if line.startswith("@<TRIPOS>"):
            # if header character in line, send to a function that will direct it properly
            line, topology = parse_record_type_indicator(
                f, line, topology, site_type
            )
        elif line == "":
            break
        else:
            # else, skip to next line
            line = f.readline()
    f.close()

    # return warnings if any sections are not covered
    # save sections to a list for each
    # Iterate through list of <ATOM> to save to Topology.sites
    # Iterate through list of <BOND> to save to Topology.bonds
    # Save box dimensions to Topology.box
    # Make sure other attributes of the topology are updated accordingly
    # TODO: read in parameters to correct attribute as well
    return topology


def load_top_sites(f, topology, site_type="Atom"):
    """Take a mol2 file section with the heading @<TRIPOS>ATOM and save to the topology.sites attribute"""
    while True:
        line = f.readline()
        if "@" not in line and not line == "\n":
            line = line.split()
            position = [float(x) for x in line[2:5]] * u.Å
            # TODO: make sure charges are also saved as a unyt value
            # TODO: add validation for element names
            if site_type == "lj":
                element = None
            elif element_by_symbol(line[5]):
                element = element_by_symbol(line[5])
            elif element_by_name(line[5]):
                element = element_by_name(line[5])
            else:
                raise UserWarning(
                    "No element detected for site {} with index{}, consider manually adding the element to the topology".format(
                        line[1], len(topology.sites) + 1
                    )
                )
                element = None
            try:
                charge = float(line[8])
            except IndexError:
                warnings.warn(
                    "No charges were detected for site {} with index {}".format(
                        line[1], line[0]
                    )
                )
                charge = None
            atom = Atom(
                name=line[1],
                position=position.to("nm"),
                charge=charge,
                element=element,
            )
            topology.add_site(atom)
        else:
            break
    return line, topology


def load_top_bonds(f, topology, **kwargs):
    """Take a mol2 file section with the heading @<TRIPOS>BOND and save to the topology.bonds attribute"""
    while True:
        line = f.readline()
        if "@" not in line and not line == "\n":
            line = line.split()
            bond = Bond(
                connection_members=(
                    topology.sites[int(line[1]) - 1],
                    topology.sites[int(line[2]) - 1],
                )
            )
            topology.add_connection(bond)
        else:
            break
    return line, topology


def load_top_box(f, topology, **kwargs):
    """Take a mol2 file section with the heading @<TRIPOS>FF_PBC and save to a topology"""
    if topology.box:
        raise warnings.UserWarning(
            "This mol2 file has two boxes to be read in, only reading in one with dimensions {}".format(
                topology.box
            )
        )
        f.readline()
        return line, topology
    while True:
        line = f.readline()
        if "@" not in line and not line == "\n":
            line = line.split()
            # TODO: write to box information
            topology.box = Box(
                lengths=[float(x) for x in line[0:3]] * u.Å,
                angles=[float(x) for x in line[3:6]] * u.degree,
            )
        else:
            break
    return line, topology


def parse_record_type_indicator(f, line, topology, site_type):
    """Take a specific record type indicator from a mol2 file format and save to the proper attribute of a gmso topology.
    Supported record type indicators include Atom, Bond, FF_PBC."""
    supported_rti = {
        "@<TRIPOS>ATOM\n": load_top_sites,
        "@<TRIPOS>BOND\n": load_top_bonds,
        "@<TRIPOS>CRYSIN\n": load_top_box,
        "@<TRIPOS>FF_PBC\n": load_top_box,
    }
    # read in to atom attribute
    try:
        return supported_rti[line](f, topology, site_type=site_type)
    except KeyError:
        warnings.warn(
            "The record type indicator {} is not supported".format(line)
        )
        line = f.readline()
        return line, topology
