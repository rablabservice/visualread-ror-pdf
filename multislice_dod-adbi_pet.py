import os
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Create PDF multislice for DoD-ADBI.")
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
    parser.add_argument("-d", "--dry_run", action="store_true", help="Show scans but don't process")
    return parser.parse_args()

# Main Script
if __name__ == "__main__":
    args = _parse_args()
    verbose = not args.quiet
    project_dir = '/mnt/coredata/Projects/DoD-ADBI/data/processed/'
    pet_proc = pd.DataFrame(columns=["ADBI_ID", "proc_petf", "tracer", "pet_date", "multislicef", "merged_multislicef"])

    # Iterate through the subject directories
    for root, dirs, files in os.walk(project_dir):
        for subject_dir in dirs:
            subject_path = os.path.join(root, subject_dir)
            print(f"Checking subject: {subject_path}")  # Debug

            # Check for tracer_date directories inside each subject directory
            for tracer_date_dir in os.listdir(subject_path):
                tracer_date_path = os.path.join(subject_path, tracer_date_dir)
                print(f"Checking tracer_date: {tracer_date_path}")  # Debug

                if os.path.isdir(tracer_date_path) and "_" in tracer_date_dir:
                    try:
                        # Split tracer_date into tracer and date
                        tracer, pet_date = tracer_date_dir.split("_", 1)

                        # Find the .nii file named "proc_petf"
                        
                        nii_files = [f for f in os.listdir(tracer_date_path) if f.endswith(".nii") and "ars6" in f]
                        print(f"Found .nii files: {nii_files}")  # Debug

                        if nii_files:
                            proc_petf_path = os.path.join(tracer_date_path, nii_files[0])

                            # Append information to DataFrame
                            pet_proc = pd.concat([
                                pet_proc,
                                pd.DataFrame({
                                    "ADBI_ID": os.path.basename(subject_path),
                                    "proc_petf": [proc_petf_path],
                                    "tracer": [tracer],
                                    "pet_date": [pet_date],
                                })
                            ], ignore_index=True)
                    except ValueError:
                        print(f"Skipping directory {tracer_date_dir}: Unable to split into tracer and date")

    print(f"DataFrame after parsing directories:\n{pet_proc}")  # Debug

    # Merge the excel data with the existing pet_proc DataFrame on the 'subj' column
    excel_df = pd.read_excel("/mnt/coredata/Projects/DoD-ADBI/metadata/DoD-ADBI_quantification_visual_reads.xlsx")

    # Now merge the DataFrames
    pet_proc = pet_proc.merge(excel_df[['ADBI_ID', 'SUMMARY_SUVR', 'CENTILOID', 'VISUAL_READ']], on='ADBI_ID', how='left')
    pet_proc['SUMMARY_SUVR'] = pet_proc['SUMMARY_SUVR'].round(2)
    pet_proc['CENTILOID'] = pet_proc['CENTILOID'].round(0).astype(int)
    
    # Optionally, debug the merged DataFrame
    print(f"Merged DataFrame:\n{pet_proc.head()}")

    # Process the multislice images
    for idx, row in pet_proc.iterrows():
        if args.dry_run:
            print(f"Dry run: Would process {row['proc_petf']}")
            continue

        _, multislicef = niiplot.create_multislice(
            imagef=row["proc_petf"],
            subj=row["ADBI_ID"],
            tracer=row["tracer"],
            suvr=row["SUMMARY_SUVR"],
            centiloid=row["CENTILOID"],
            visual_read=row["VISUAL_READ"],
            image_date=row["pet_date"],
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
        pet_proc.at[idx, "multislicef"] = multislicef

        merged_multislicef = merge_multislice(
            infile=multislicef,
            template_dir='/mnt/coredata/Projects/DoD-ADBI/metadata/templates/',
            tracer=row["tracer"],
            remove_infile=False,
            overwrite=args.overwrite,
            verbose=verbose,
        )
        pet_proc.at[idx, "merged_multislicef"] = merged_multislicef

    # Output results to CSV
    output_csv_path = os.path.join(project_dir, "processed_files.csv")
    pet_proc.to_csv(output_csv_path, index=False)
    if verbose:
        print(f"Processing completed. Results saved to {output_csv_path}")

