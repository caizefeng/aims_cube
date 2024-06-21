import itertools
from typing import Tuple, Optional, Dict
from deprecated import deprecated

import numpy as np
from monty.io import zopen
from numpy.typing import NDArray
from pymatgen.core import Site, Structure, Lattice, Molecule
from pymatgen.io.vasp import Poscar
from pymatgen.io.vasp.outputs import VolumetricData
from scipy.interpolate import RegularGridInterpolator

from aims_cube.utils.geometry_utils import find_translation_vector


class VolumetricDataFromAims(VolumetricData):
    bohr_to_angstrom = 0.52917721  # FHI-aims uses the NIST CODATA 2002 unit conventions, P42

    def __init__(self, *args,
                 cube_dict: Dict,
                 pre_process_shift: Optional[NDArray] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cube_dict = cube_dict
        self.pbc = all(self.structure.lattice.pbc)

        # Override incorrect implementation for interpolation in superclass (both PBC and non-PBC case)
        self._reconstruct_interpolator()

        # Compensate for all shifts introduced during pre-processing
        if self.pbc:
            self.translate_pbc(-pre_process_shift, frac_coords=False)

    @classmethod
    def from_cube_and_geo_in(cls, cube_filename, geometry_in_filename, use_cube_box_as_lattice=False):
        """
        Initialize the VolumetricDataFromAims object and store the volumetric data as data.

        Args:
            cube_filename (str): of the FHI-aims-generated .cube file to read.
            geometry_in_filename (str): of the geometry.in to read.
            use_cube_box_as_lattice (bool): Whether to use the non-periodic box defined in cube file as the new lattice or use the original lattice of geometry.in.
        """

        # Get Structure from geometry.in file, for afterward data trimming, interpolation, and validation
        structure_geo_in = cls.read_structure_from_geo_in(geometry_in_filename)

        # Extract all information from .cube file
        cube_dict = cls.read_cube_aims(cube_filename, length_conversion_ratio=cls.bohr_to_angstrom)
        origin: NDArray[float] = cube_dict["origin"]
        dim: Tuple = cube_dict["dim"]
        cube_box: Lattice = cube_dict["box"]
        sites: Molecule = cube_dict["sites"]
        voxel: NDArray[float, float] = cube_dict["voxel"]
        data: NDArray[float, float, float] = cube_dict["data"]  # volumetric data

        # Keep track of all shifts (Cartesian) introduced during pre-processing
        pre_process_shift = np.zeros(3, dtype=float)

        # Check if coordinates are close
        if structure_geo_in.cart_coords.shape == sites.cart_coords.shape and np.allclose(
                structure_geo_in.cart_coords, sites.cart_coords,
                rtol=1e-5, atol=1e-6):
            print(
                f"Input files compatibility check: Atom positions match between {cube_filename} and {geometry_in_filename}.")
        else:
            raise RuntimeError(
                f"Input files compatibility check: Atom positions DO NOT match between {cube_filename} and {geometry_in_filename}. They are not from the same FHI-aims run, or some error occurred during the calculation. Exiting.")

        # Shift all sites of `structure_geo_in` to align its origin with the origin of the volumetric data
        structure_geo_in.translate_sites(list(range(structure_geo_in.num_sites)), -origin, frac_coords=False,
                                         to_unit_cell=True)
        pre_process_shift += -origin

        if not use_cube_box_as_lattice:
            print(f"Box for volumetric data: Using periodic lattice defined in {geometry_in_filename}.")

            # Check if normalized lattice (i.e. shape and orientation) are close
            if np.allclose(cls._normalize_lattice(structure_geo_in.lattice), cls._normalize_lattice(cube_box)):
                print(
                    f"Lattices in {cube_filename} and {geometry_in_filename} match in shape and orientation. Trying trimming method first.")

                # Rule out the situation where cube box is smaller than geometry.in lattice
                lattice_lengths_geo_in = np.array(structure_geo_in.lattice.lengths)
                lattice_lengths_cube = np.array(cube_box.lengths)
                is_cube_smaller = np.any(np.where(np.isclose(lattice_lengths_geo_in, lattice_lengths_cube), False,
                                                  lattice_lengths_cube < lattice_lengths_geo_in))
                if is_cube_smaller:
                    raise RuntimeError(
                        f"The non-periodic box defined by {cube_filename} is SMALLER than the unit cell in {geometry_in_filename}, which is unsupported for geometry.in-based trimming. Set `use_cube_box_as_lattice` to True instead (use `-b True` for CLI). Exiting.")

                # Trim grid points based on the difference between structure and structure_geo_in
                voxel_lengths = np.linalg.norm(voxel, axis=1)
                dim_trimmed_array = (np.where(np.isclose(lattice_lengths_geo_in, lattice_lengths_cube), np.array(dim),
                                              np.floor(lattice_lengths_geo_in / voxel_lengths + 1))).astype(int)
                dim_trimmed = tuple(dim_trimmed_array)
                print(
                    f"Grid dimension from {cube_filename} is {dim}, trimmed to {dim_trimmed} based on {geometry_in_filename}.")

                # Needed for RegularGridInterpolator input
                grid_max_frac = (dim_trimmed_array - 1) * voxel_lengths / lattice_lengths_geo_in

                # Trim volumetric data
                data_trimmed = (data[0:dim_trimmed[0], 0:dim_trimmed[1], 0:dim_trimmed[2]]).copy()

                # Interpolate to an evenly-spaced mesh
                final_data = cls._interpolate_to_even(data=data_trimmed, dim=dim_trimmed, grid_max_frac=grid_max_frac)
                print(
                    f"The volumetric data has been further interpolated to an even rectilinear grid with dimension {dim_trimmed}.")

                return cls(structure=structure_geo_in, data={"total": final_data}, cube_dict=cube_dict,
                           pre_process_shift=pre_process_shift)

            else:
                print(
                    f"Lattices in {cube_filename} and {geometry_in_filename} differ in shape and orientation. Using interpolation instead of trimming to obtain the volumetric data within the unit cell.")

                # Rule out the situation where cube box is "smaller" by solving polyhedral containment problem (linear programming)
                # Determine the translation vector if the containment is possible
                success, translation_to_contain = find_translation_vector(
                    contained_matrix=structure_geo_in.lattice.matrix,
                    containing_matrix=cube_box.matrix)
                if success:
                    print(
                        f"The unit cell in {geometry_in_filename} can be fully contained within the non-periodic box defined by {cube_filename} "
                        f"with translation vector {translation_to_contain}.")
                else:
                    raise RuntimeError(
                        f"The unit cell in {geometry_in_filename} CANNOT be fully contained within the non-periodic box defined by {cube_filename}, "
                        f"which is unsupported for geometry.in-based re-interpolation. Set `use_cube_box_as_lattice` to True instead (use `-b True` for CLI). Exiting.")

                # Translate sites and lattice, and generate new grid
                structure_geo_in.translate_sites(list(range(structure_geo_in.num_sites)), -translation_to_contain,
                                                 frac_coords=False, to_unit_cell=True)
                pre_process_shift += -translation_to_contain
                (new_X_in_cube, new_Y_in_cube, new_Z_in_cube), new_dim = cls.generate_grid_same_density(
                    old_lattice=cube_box, old_dim=dim,
                    new_lattice=structure_geo_in.lattice,
                    translation_vector=translation_to_contain)

                print(f"Grid dimension from {cube_filename} is {dim}.")
                print(
                    f"Constructing and interpolating to a new grid within the unit cell with dimension {new_dim} to maintain similar density.")

                final_data = cls._interpolate(data=data, dim=dim,
                                              coords_to_evaluate=(new_X_in_cube, new_Y_in_cube, new_Z_in_cube))

                return cls(structure=structure_geo_in, data={"total": final_data}, cube_dict=cube_dict,
                           pre_process_shift=pre_process_shift)

        else:
            print(
                f"Box for volumetric data: Using non-periodic lattice defined by {cube_filename}.")

            # Map geometry.in sites and their periodic images into the box defined by cube file
            num_sites = 0
            range_to_consider = 1
            while True:
                structure_cube = cls.map_sites_to_new_lattice(cube_box, structure_geo_in, image_range=range_to_consider)
                if structure_cube.num_sites > num_sites:
                    num_sites = structure_cube.num_sites
                    range_to_consider += 1
                else:
                    print(
                        f"Total atoms in the non-periodic box is {structure_cube.num_sites}.")
                    print(
                        f"Required periodic images (i.e. the unit cell) range: ({-range_to_consider}, +{range_to_consider + 1}) to complete mapping.")
                    break

            structure_cube.sort()

            return cls(structure=structure_cube, data={"total": data}, cube_dict=cube_dict)

    @staticmethod
    def read_cube_aims(cube_filename, length_conversion_ratio):
        file = zopen(cube_filename, "rt")

        # skip header lines
        file.readline()
        file.readline()

        # number of atoms followed by the position of the origin of the volumetric data
        line = file.readline().split()
        origin = np.array([length_conversion_ratio * float(val) for val in line[1:]])
        natoms = int(line[0])

        # The number of voxels along each axis (x, y, z) followed by the axis vector.
        line = file.readline().split()
        num_x_voxels = int(line[0])
        voxel_x = np.array([length_conversion_ratio * float(val) for val in line[1:]])

        line = file.readline().split()
        num_y_voxels = int(line[0])
        voxel_y = np.array([length_conversion_ratio * float(val) for val in line[1:]])

        line = file.readline().split()
        num_z_voxels = int(line[0])
        voxel_z = np.array([length_conversion_ratio * float(val) for val in line[1:]])

        # The last section in the header is one line for each atom consisting of 5 numbers,
        # the first is the atom number, second is charge,
        # the last three are the x,y,z coordinates of the atom center.
        sites = []
        for _ in range(natoms):
            line = file.readline().split()
            sites.append(
                Site(line[0], np.multiply(length_conversion_ratio, list(map(float, line[2:])))))

        box = Lattice([voxel_x * num_x_voxels, voxel_y * num_y_voxels, voxel_z * num_z_voxels],
                      pbc=(False, False, False))

        all_sites = Molecule(
            species=[s.specie for s in sites],
            coords=[s.coords for s in sites],
        )

        # Volumetric data
        data = np.reshape(np.array(file.read().split()).astype(float), (num_x_voxels, num_y_voxels, num_z_voxels))

        return {
            "origin": origin,  # type: NDArray[float]
            "dim": tuple([num_x_voxels, num_y_voxels, num_z_voxels]),  # type: Tuple[int, int, int]
            "voxel": np.array([voxel_x, voxel_y, voxel_z]),  # type: NDArray[float, float]
            "box": box,  # type: Lattice
            "sites": all_sites,  # type: Molecule
            "data": data  # type: NDArray[float, float, float]
        }

    @staticmethod
    def read_structure_from_geo_in(geo_in_filename):
        file = zopen(geo_in_filename, "rt")
        lines = file.readlines()

        lattice = []
        sites = []
        for line in lines:
            if line.startswith("lattice_vector"):
                lattice.append(np.array([float(val) for val in line.split()[1:]]))

            if line.startswith("atom"):
                sites.append(Site(line.split()[4], list(map(float, line.split()[1:4]))))

        structure_from_geo_in = Structure(
            lattice=lattice,
            species=[s.specie for s in sites],
            coords=[s.coords for s in sites],
            coords_are_cartesian=True,
        )

        return structure_from_geo_in

    @staticmethod
    def map_sites_to_new_lattice(new_lattice, old_structure, image_range):
        # Create the new empty structure with the new lattice
        new_structure = Structure(new_lattice, [], [])

        # Map atoms from the old structure to the new structure considering periodic images
        for site in old_structure:  # type: Site
            for i in range(-image_range, image_range + 1):
                for j in range(-image_range, image_range + 1):
                    for k in range(-image_range, image_range + 1):
                        shifted_coords = site.frac_coords + np.array([i, j, k])
                        new_cart_coords = old_structure.lattice.get_cartesian_coords(shifted_coords)
                        new_frac_coords = new_lattice.get_fractional_coords(new_cart_coords)
                        if all(0 <= coord < 1 for coord in new_frac_coords):
                            new_structure.append(site.species, new_frac_coords, coords_are_cartesian=False)

        # Remove duplicate sites if necessary
        new_structure.merge_sites(tol=1e-3, mode="average")

        return new_structure

    @staticmethod
    def generate_grid_same_density(old_lattice: Lattice, old_dim: Tuple, new_lattice: Lattice,
                                   translation_vector: NDArray):

        volume_per_voxel = old_lattice.volume / (old_dim[0] * old_dim[1] * old_dim[2])  # i.e. 1/density

        new_normalized_voxel = Lattice(VolumetricDataFromAims._normalize_lattice(new_lattice))
        edge_length = np.cbrt(volume_per_voxel / new_normalized_voxel.volume)
        new_dim_array = np.ceil(np.array(new_lattice.lengths) / edge_length).astype(int)  # type: NDArray
        new_dim = tuple(new_dim_array.tolist())

        xx = np.linspace(0, 1, new_dim[0], endpoint=False)
        yy = np.linspace(0, 1, new_dim[1], endpoint=False)
        zz = np.linspace(0, 1, new_dim[2], endpoint=False)
        new_X, new_Y, new_Z = np.meshgrid(xx, yy, zz, indexing='ij')  # type: NDArray  # noqa

        new_grid = np.vstack([new_X.ravel(), new_Y.ravel(), new_Z.ravel()]).T
        cart = new_lattice.get_cartesian_coords(new_grid)
        new_grid_in_old = old_lattice.get_fractional_coords(cart + translation_vector)

        (new_X_in_old, new_Y_in_old, new_Z_in_old) = (new_grid_in_old[:, 0].reshape(new_X.shape),
                                                      new_grid_in_old[:, 1].reshape(new_Y.shape),
                                                      new_grid_in_old[:, 2].reshape(new_Z.shape))
        return (new_X_in_old, new_Y_in_old, new_Z_in_old), new_dim

    @staticmethod
    def _interpolate(data, dim, coords_to_evaluate: Tuple[NDArray, NDArray, NDArray]) -> NDArray:

        xpoints, ypoints, zpoints = VolumetricDataFromAims._even_grid_xyz(dim)
        interpolator = RegularGridInterpolator(
            (xpoints, ypoints, zpoints),
            data,
            bounds_error=True,
        )
        return interpolator(coords_to_evaluate)

    @staticmethod
    def _interpolate_to_even(data, dim, grid_max_frac: NDArray) -> NDArray:

        # Recreate the uneven grid caused by trimming
        # (For perfectly matched geometry.in and cube, grid_max_frac = [(dim[0] - 1)/dim[0]] x3)
        xpoints = np.ones(dim[0] + 1, dtype=float)
        ypoints = np.ones(dim[1] + 1, dtype=float)
        zpoints = np.ones(dim[2] + 1, dtype=float)
        xpoints[:dim[0]] = np.linspace(0.0, grid_max_frac[0], num=dim[0])
        ypoints[:dim[1]] = np.linspace(0.0, grid_max_frac[1], num=dim[1])
        zpoints[:dim[2]] = np.linspace(0.0, grid_max_frac[2], num=dim[2])
        interpolator = RegularGridInterpolator(
            (xpoints, ypoints, zpoints),
            np.pad(data, (0, 1), 'wrap'),  # Padding following PBC
            bounds_error=True,
        )

        # Interpolate trimmed data to an evenly-spaced grid
        xpoints, ypoints, zpoints = VolumetricDataFromAims._even_grid_xyz(dim)
        X, Y, Z = np.meshgrid(xpoints, ypoints, zpoints, indexing='ij')  # noqa

        return interpolator((X, Y, Z))

    def _reconstruct_interpolator(self):
        if self.pbc:
            self.xpoints = np.linspace(0.0, 1.0, num=self.dim[0] + 1)
            self.ypoints = np.linspace(0.0, 1.0, num=self.dim[1] + 1)
            self.zpoints = np.linspace(0.0, 1.0, num=self.dim[2] + 1)
            self.interpolator = RegularGridInterpolator(
                (self.xpoints, self.ypoints, self.zpoints),
                np.pad(self.data["total"], (0, 1), 'wrap'),  # Padding following PBC
                bounds_error=True,
            )
        else:
            self.xpoints, self.ypoints, self.zpoints = self._even_grid_xyz(self.dim)
            self.interpolator = RegularGridInterpolator(
                (self.xpoints, self.ypoints, self.zpoints),
                self.data["total"],
                bounds_error=True,
            )

    @staticmethod
    def _normalize_lattice(lattice):
        # Extract the matrix of the lattice
        matrix = lattice.matrix
        # Normalize each row vector to have unit length
        normalized_matrix = np.array([row / np.linalg.norm(row) for row in matrix])
        return normalized_matrix

    @staticmethod
    def _even_grid_xyz(dim):
        xpoints = np.linspace(0.0, 1.0, num=dim[0], endpoint=False)
        ypoints = np.linspace(0.0, 1.0, num=dim[1], endpoint=False)
        zpoints = np.linspace(0.0, 1.0, num=dim[2], endpoint=False)
        return xpoints, ypoints, zpoints

    def scale2vasp(self, stm_voltage=1.0):
        # FHI-aims: n(r)*ngridpts (unit 1/A^3), P436
        # VASP: n(r)*ngridpts*volume (unit 1)
        # if "output cube stm V": n(r)*ngridpts*V(Volt) (unit energy), P438
        self.scale(self.structure.volume / stm_voltage)

    def translate_pbc(self, translation_vector, frac_coords=True):
        """
        Translate all sites and the volumetric data by some vector, keeping them within the unit cell. (Note: n/a w/o PBC)

        Args:
            translation_vector (List[float, float, float]): Translation vector in fractional coordinates.
            frac_coords (bool): Whether the vector corresponds to fractional or Cartesian coordinates.
        """

        if not self.pbc:
            raise RuntimeError("Translation cannot be performed on a box without PBC. Exiting.")

        if frac_coords:
            translation_vector = np.array(translation_vector)
        else:
            translation_vector = self.structure.lattice.get_fractional_coords(translation_vector)
        # Translate sites
        self.structure.translate_sites(list(range(self.structure.num_sites)), translation_vector, frac_coords=True,
                                       to_unit_cell=True)

        # Create interpolator of a (2,2,2) supercell
        xpoints = np.linspace(0.0, 2.0, num=2 * self.dim[0] + 1)
        ypoints = np.linspace(0.0, 2.0, num=2 * self.dim[1] + 1)
        zpoints = np.linspace(0.0, 2.0, num=2 * self.dim[2] + 1)
        doublesize_interpolator = RegularGridInterpolator(
            (xpoints, ypoints, zpoints),
            np.pad(np.tile(self.data["total"], (2, 2, 2)), (0, 1), 'wrap'),  # Padding following PBC
            bounds_error=True,
        )

        # Translate volumetric data
        start_fcoords = np.mod(-translation_vector, 1)  # Atoms go t means grid goes -t
        end_fcoords = start_fcoords + 1
        xx = np.linspace(start_fcoords[0], end_fcoords[0], num=self.dim[0], endpoint=False)
        yy = np.linspace(start_fcoords[1], end_fcoords[1], num=self.dim[1], endpoint=False)
        zz = np.linspace(start_fcoords[2], end_fcoords[2], num=self.dim[2], endpoint=False)
        X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij')  # noqa
        self.data["total"] = doublesize_interpolator((X, Y, Z))

        # Reconstruct interpolator
        self._reconstruct_interpolator()

    @deprecated(reason="This method will be removed in future versions. Use `translate_pbc` instead.")
    def translate_pbc_cart_approx(self, approximate_translation):
        """
        Translate all sites and the volumetric data by some vector, keeping them within the unit cell. (Note: n/a w/o PBC)

        Args:
            approximate_translation (List[float, float, float]): Approximate translation vector in Cartesian coordinates.
        """
        # Only allow translations that are integer multiples of voxel_lengths to avoid interpolation every time.

        if not self.pbc:
            raise RuntimeError("Translation cannot be performed on a box without PBC. Exiting.")

        voxel_lengths = self.structure.lattice.lengths / np.array(self.dim)
        approximate_translation = np.array(approximate_translation)

        steps = np.rint(approximate_translation / voxel_lengths).astype(int)
        actual_translation = steps * voxel_lengths
        print(f"The actual translation for each axis is {tuple(actual_translation)}.")
        # Translate volumetric data
        self.data["total"] = np.roll(self.data["total"], steps, axis=(0, 1, 2)).copy()

        # Translate sites
        self.structure.translate_sites(list(range(self.structure.num_sites)), actual_translation, frac_coords=False,
                                       to_unit_cell=True)

        # Reconstruct interpolator
        self._reconstruct_interpolator()

    def write_poscar(self, file_name, vasp4_compatible=False):
        with zopen(file_name, "wt") as f:
            p = Poscar(self.structure)

            # use original name if it's been set (e.g. from Chgcar)
            comment = getattr(self, "name", p.comment)

            lines = comment + "\n"
            lines += "   1.00000000000000\n"
            for vec in self.structure.lattice.matrix:
                lines += f" {vec[0]:12.6f}{vec[1]:12.6f}{vec[2]:12.6f}\n"
            if not vasp4_compatible:
                lines += "".join(f"{s:5}" for s in p.site_symbols) + "\n"
            lines += "".join(f"{x:6}" for x in p.natoms) + "\n"
            lines += "Direct\n"
            for site in self.structure:
                a, b, c = site.frac_coords
                lines += f"{a:10.6f}{b:10.6f}{c:10.6f}\n"
            lines += " \n"
            f.write(lines)

    # Override only to fix a bug in local function `_print_fortran_float`
    def write_file(self, file_name, vasp4_compatible=False):
        """
        Write the VolumetricData object to a vasp compatible file.

        Args:
            file_name (str): Path to a file
            vasp4_compatible (bool): True if the format is vasp4 compatible
        """

        def _print_fortran_float(f):
            """
            Fortran codes print floats with a leading zero in scientific
            notation. When writing CHGCAR files, we adopt this convention
            to ensure written CHGCAR files are byte-to-byte identical to
            their input files as far as possible.
            :param f: float
            :return: str.
            """
            s = f"{f:.10E}"

            if not s.startswith("-"):
                return f"0.{s[0]}{s[2:12]}E{int(s[13:]) + 1:+03}"
            return f"-.{s[1]}{s[3:13]}E{int(s[14:]) + 1:+03}"

        with zopen(file_name, "wt") as f:
            p = Poscar(self.structure)

            # use original name if it's been set (e.g. from Chgcar)
            comment = getattr(self, "name", p.comment)

            lines = comment + "\n"
            lines += "   1.00000000000000\n"
            for vec in self.structure.lattice.matrix:
                lines += f" {vec[0]:12.6f}{vec[1]:12.6f}{vec[2]:12.6f}\n"
            if not vasp4_compatible:
                lines += "".join(f"{s:5}" for s in p.site_symbols) + "\n"
            lines += "".join(f"{x:6}" for x in p.natoms) + "\n"
            lines += "Direct\n"
            for site in self.structure:
                a, b, c = site.frac_coords
                lines += f"{a:10.6f}{b:10.6f}{c:10.6f}\n"
            lines += " \n"
            f.write(lines)
            a = self.dim

            def write_spin(data_type):
                lines = []
                count = 0
                f.write(f"   {a[0]}   {a[1]}   {a[2]}\n")
                for k, j, i in itertools.product(list(range(a[2])), list(range(a[1])), list(range(a[0]))):
                    lines.append(_print_fortran_float(self.data[data_type][i, j, k]))
                    count += 1
                    if count % 5 == 0:
                        f.write(" " + "".join(lines) + "\n")
                        lines = []
                    else:
                        lines.append(" ")
                if count % 5 != 0:
                    f.write(" " + "".join(lines) + " \n")
                f.write("".join(self.data_aug.get(data_type, [])))

            write_spin("total")
            if self.is_spin_polarized and self.is_soc:
                write_spin("diff_x")
                write_spin("diff_y")
                write_spin("diff_z")
            elif self.is_spin_polarized:
                write_spin("diff")
