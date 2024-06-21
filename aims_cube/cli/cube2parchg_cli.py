import argparse
import os
import warnings

from aims_cube.VolumetricDataFromAims import VolumetricDataFromAims
from aims_cube.utils.str_utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(description='FHI-aims .cube file to VASP-compatible PARCHG converter.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('cube', type=str, help='Path to the input FHI-aims .cube file.')
    parser.add_argument('geometry_in', type=str, help='Path to the input FHI-aims geometry.in file.')
    parser.add_argument('-o', '--output', type=str, default='.', help='Path to the output directory.')
    parser.add_argument('-b', '--cube_box', type=str2bool, default=False,
                        help='Whether to use the non-periodic box defined in the .cube file as the output box.')
    parser.add_argument('-t', '--translate', type=float, nargs=3, default=None,
                        metavar=('TRANSLATE_A', 'TRANSLATE_B', 'TRANSLATE_C'),
                        help='Translation vector in fractional coordinates by which all sites and the volumetric data will be translated (three float values, one for each lattice axis, required).')
    parser.add_argument('-f', '--frac_coords', type=str2bool, default=True,
                        help='Whether the translation vector corresponds to fractional or Cartesian coordinates.')
    parser.add_argument('-c', '--vasp_conversion', type=str2bool, default=False,
                        help='Whether to perform the VASP density unit conversion (only applicable when the volumetric data is in density units).')
    parser.add_argument('-v', '--stm_voltage', type=float, default=1.0,
                        help='STM voltage when the volumetric data is generated using the `cube stm` command. Only works when --vasp_conversion is True.')
    return parser.parse_args()


def main():
    warnings.simplefilter('always', DeprecationWarning)
    args = parse_args()
    volume_data = VolumetricDataFromAims.from_cube_and_geo_in(cube_filename=args.cube,
                                                              geometry_in_filename=args.geometry_in,
                                                              use_cube_box_as_lattice=args.cube_box)
    if args.vasp_conversion:
        volume_data.scale2vasp(stm_voltage=args.stm_voltage)

    if args.translate is not None:
        volume_data.translate_pbc(args.translate, frac_coords=args.frac_coords)

    output_directory = args.output if args.output.endswith('/') else args.output + '/'
    parchg_path = os.path.join(output_directory, "PARCHG")
    poscar_path = os.path.join(output_directory, "POSCAR")
    volume_data.write_file(parchg_path)
    volume_data.write_poscar(poscar_path)
    print(f"Conversion completed successfully. The converted PARCHG and POSCAR are located under {output_directory}")


if __name__ == "__main__":
    main()
