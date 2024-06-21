# aims_cube

FHI-aims often uses arbitrary origins and edges to generate volumetric data in its cube file format. This can sometimes be less than ideal due to its decoupling from lattice or atom information. This minimal module is designed to read `.cube` files and `geometry.in` files from FHI-aims and convert them into an instance of `pymatgen`'s `VolumetricData` with additional features. Specifically, it can pad atoms into the arbitrary volumetric data box or interpolate the volumetric data into a smaller, more symmetry-aligned box, typically the unit cell from `geometry.in`.

## Installation

```shell
pip install git+https://github.com/caizefeng/aims_cube.git
```

## Usage

Read data from FHI-aims `.cube` file and `geometry.in` file:

```python
from aims_cube.VolumetricDataFromAims import VolumetricDataFromAims

volume_data = VolumetricDataFromAims.from_cube_and_geo_in(cube_filename="/path/to/xxx.cube",
                                                          geometry_in_filename="/path/to/geometry.in",
                                                          use_cube_box_as_lattice=False)
```

You can then use all methods defined in the superclass `VolumetricData`, such as `+`, `-`, `value_at` (to get a data value, potentially interpolated, at a given point (x, y, z) in terms of fractional lattice parameters), `get_average_along_axis` (to get the averaged total of the volumetric data along a certain axis direction), and more.

Some additional methods include:

1. `translate_pbc`: Translate all sites and the volumetric data by a translation vector, keeping them within the unit cell (Note: not applicable without PBC).
2. `write_file`: Write the VolumetricData object to a VASP-compatible CHGCAR/PARCHG file.
3. `write_poscar`: Write the structure to a VASP-compatible POSCAR file.

## CLI Tool

`cube2parchg`: Convert the volumetric data and the structure from FHI-aims to VASP-compatible PARCHG and POSCAR files.

```
usage: cube2parchg [-h] [-o OUTPUT] [-b CUBE_BOX] [-t TRANSLATE_A TRANSLATE_B TRANSLATE_C] [-f FRAC_COORDS] [-c VASP_CONVERSION] [-v STM_VOLTAGE] cube geometry_in

FHI-aims .cube file to VASP-compatible PARCHG converter.

positional arguments:
  cube                  Path to the input FHI-aims .cube file.
  geometry_in           Path to the input FHI-aims geometry.in file.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to the output directory. (default: .)
  -b CUBE_BOX, --cube_box CUBE_BOX
                        Whether to use the non-periodic box defined in the .cube file as the output box. (default: False)
  -t TRANSLATE_A TRANSLATE_B TRANSLATE_C, --translate TRANSLATE_A TRANSLATE_B TRANSLATE_C
                        Translation vector in fractional coordinates by which all sites and the volumetric data will be translated (three float values, one for each lattice axis, required). (default: None)
  -f FRAC_COORDS, --frac_coords FRAC_COORDS
                        Whether the translation vector corresponds to fractional or Cartesian coordinates. (default: True)
  -c VASP_CONVERSION, --vasp_conversion VASP_CONVERSION
                        Whether to perform the VASP density unit conversion (only applicable when the volumetric data is in density units). (default: False)
  -v STM_VOLTAGE, --stm_voltage STM_VOLTAGE
                        STM voltage when the volumetric data is generated using the `cube stm` command. Only works when --vasp_conversion is True. (default: 1.0)
```

Example:
```shell
cube2parchg ./cube_002_stm_01.cube ./geometry.in -o ./
```
This will create `./PARCHG` and `./POSCAR`.
