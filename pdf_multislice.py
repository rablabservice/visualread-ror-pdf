import os
import argparse
import os.path as op
import subprocess
import nifti_plotting_2rows as niiplot

# Add helper functions
def add_presuf(filename, suffix):
    base, ext = os.path.splitext(filename)
    return f"{base}{suffix}{ext}"

def run_cmd(cmd):
    process = subprocess.run(cmd, shell=True, check=True, text=True)
    return process.returncode

# Merge Multislice Function
def merge_multislice(infile, template_dir, tracer, remove_infile=False, overwrite=False, verbose=True):
    assert infile.endswith(".pdf")
    outfile = add_presuf(infile, suffix="_merged")
    if op.isfile(outfile) and not overwrite:
        if verbose:
            print(f"Merged PDF already exists: {outfile}")
        return outfile
    templatef = op.join(template_dir, f"{tracer}_template.pdf")
    cmd = f"qpdf --linearize --qdf --optimize-images --empty --pages {templatef} 1 {infile} 1 -- {outfile}"
    run_cmd(cmd)
    if verbose:
        print(f"Merged PDF: {outfile}")
    if remove_infile:
        os.remove(infile)
        if verbose:
            print(f"Removed {infile}")
    return outfile

# Argument Parsing
def _parse_args():
    parser = argparse.ArgumentParser(description="Create PDF multislice for RoR.")
    parser.add_argument("-pet", "--petf", type=str, required=True,
                        help="Path to the input affine transformed PET scan")
    parser.add_argument("-mri", "--mrif", type=str, required=False,
                        help="Path to the input affine transformed MRI scan")
    parser.add_argument("-s", "--subject", type=str, required=True,
                        help="Subject ID (e.g., 001)")
    parser.add_argument("-d", "--scan_date", type=str, required=True,
                        help="Date of the scan (e.g., 2023-01-01)")
    parser.add_argument("-m", "--modality", type=str, choices=["FBB", "FBP", "NAV", "PIB", "FTP", "MK6240", "PI2620","MRI-T1"], required=True,
                        help="Modality of the input scan (choices: %(choices)s)")
    parser.add_argument("-v", "--SUVR", type=str, required=False,
                        help="Quantification in SUVR")
    parser.add_argument("-cl", "--centiloid", type=float, required=False,
                        help="Centiloid value for the scan")
    parser.add_argument("-vr", "--visual_read", type=str, choices=["Elevated", "Non-elevated"], required=False,
                        help="Visual read of the scan (choices: %(choices)s)")
    parser.add_argument("-t", "--template_dir", type=str, required=True, default="/mnt/coredata/projects/scripts/visualread-ror-pdf/templates/",
                        help="Path to the template directory for merging multislice images (default: %(default)s)")
    parser.add_argument("-z", "--slices", type=int, nargs="+", default=[-50, -44, -38, -32, -26, -20, -14, -8, -2, 4, 10, 16, 22, 28, 34, 40],
                        help="List of image slices to show along the z-axis (default: %(default)s)")
    parser.add_argument("--crop", default=True, action=argparse.BooleanOptionalAction,
                        help="Crop the multislice images to the brain")
    parser.add_argument("--mask_thresh", type=float, default=0.05,
                        help="Threshold for cropping empty voxels outside the brain (default: %(default)s)")
    parser.add_argument("--crop_prop", type=float, default=0.05,
                        help="Proportion of empty voxels allowed when cropping (default: %(default)s)")
    parser.add_argument("--cmap", type=str, help="Colormap to use for the multislice images")
    parser.add_argument("--vmin", type=float, default=0, help="Minimum intensity threshold (default: %(default)s)")
    parser.add_argument("--vmax", type=float, help="Maximum intensity threshold")
    parser.add_argument("--autoscale", action="store_true",
                        help="Autoscale vmax to a percentile of image values > 0")
    parser.add_argument("--autoscale_max_pct", type=float, default=99.5,
                        help="Percentile for autoscaling vmax (default: %(default)s)")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("-q", "--quiet", action="store_true", help="Run without printing output")
    return parser.parse_args()

# Main Script
if __name__ == "__main__":
    args = _parse_args()
    verbose = not args.quiet
    
    # Process the multislice images
    _, multislicef = niiplot.create_multislice(
        petf=args.petf,
        mrif=args.mrif,
        subj=args.subject,
        tracer=args.modality,
        suvr=args.SUVR,
        centiloid=args.centiloid,
        visual_read=args.visual_read,
        image_date=args.scan_date,
        cut_coords=args.slices,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        hide_cbar_values=False,
        autoscale=args.autoscale,
        autoscale_max_pct=args.autoscale_max_pct,
        crop=args.crop,
        mask_thresh=args.mask_thresh,
        crop_prop=args.crop_prop,
        overwrite=args.overwrite,
        verbose=verbose,
    )

    merged_multislicef = merge_multislice(
        infile=multislicef,
        template_dir=args.template_dir,
        tracer=args.modality,
        remove_infile=False,
        overwrite=args.overwrite,
        verbose=verbose,
    )