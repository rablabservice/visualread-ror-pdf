# Compose a two-page PDF document that includes axial multislice images of the participants’ scans 

This code generates an image of the selected axial slices for a single PET scan and includes both elevated and non-elevated example scans for clarity. It offers the capability to modify the colormap, color range, incorporate quantification results and visual read information, and superimpose PET data onto the participant’s MRI scan. Deafults for each modality can be found in get_tracer_defaults function defined in nifti_plotting_2rows.py.

The PDF output is developed by the UCSF Visual Read Core and is a standard output for the ADNI, US POINTER, CLARiTI, and DoD-ADBI studies.

## Input Files
- Tracer-specific color ranges and selected slices are assumed to be for an SUVR image with re-centered coordinates (to the center of the field of view). For most studies, affine-transformed (to MNI) SUVr PET or Step 4 image from LONI (Average, Standard Image, and Vox Size, Uniform Resolution) is an input. Only supports nifti image as an input.
- The code does not assume any specific filename structure, so the input filename can be any.
- To correctly record subject information, including ID, date of scan, and modality, must be entered. Centiloid, SUVr (cortical summary mask for amyloid or meta-temporal for tau), and visual read information can be provided by parsing arguments (see below).

## Output files
- Multi-slice visualization: `[inputfilename]_multislice.pdf`
- Merged template visualization: `[inputfilename]_multislice_merged.pdf`

## Options:
- `-h`                    Help
- `-pet`                  Path to the input affine transformed PET scan
- `-mri`                  Path to the input affine transformed MRI scan
- `-s`                    Subject ID (e.g., 001_S_001)
- `-d`                    Date of the scan (e.g., 2023-01-01)
- `-m`                    Modality of the input scan (choices: FBB, FBP, NAV, PIB, FTP, FDG, MK6240, PI2620, MRI-T1)
- `-suvr`                 Quantification in SUVR
- `-cl`                   Centiloid value for the scan
- `-vr`                   Visual read of the scan (choices: Elevated, Non-elevated)
- `-t`                    Path to the template directory for merging multislice images (default: templates/)
- `-z`                    List of image slices to show along the z-axis (default: [-50, -44, -38, -32, -26, -20, -14, -8, -2, 4, 10, 16, 22, 28, 34, 40])
- `--crop, --no-crop`     Crop the multislice images to the brain (default: True)
- `--mask_thresh`         Threshold for cropping empty voxels outside the brain (default: 0.05)
- `--crop_prop`           Proportion of empty voxels allowed when cropping (default: 0.05)
- `--cmap`                Colormap to use for the multislice images
- `--vmin`                Minimum intensity threshold (default: 0)
- `--vmax`                Maximum intensity threshold
- `--autoscale`           Autoscale vmax to a percentile of image values > 0
- `--autoscale_max_pct`   Percentile for autoscaling vmax (default: 99.5)
- `-o`                    Overwrite existing files
- `-q`                    Run without printing output
- `-savename`             File save name

### Example run from the command line [assuming python is the call to local python version]
#### Amyloid PET standard output with CL and visual read
`python pdf_multislice.py -pet myfilename.nii -s 001_S_001 -d 2025-01-01 -m FBB -cl 24 -vr Elevated`
#### Tau PET standard output with meta-temporal SUVr and visual read
`python pdf_multislice.py -pet myftpfilename.nii -mri mymrifilename.nii -s 001_S_001 -d 2025-01-01 -m FTP -suvr 1.07 -vr Non-elevated`
#### MRI T1 standard output (no template)
`python pdf_multislice.py -mri mymrifilename.nii -s 001_S_001 -d 2025-01-01 -m MRI-T1`

## Code requires python basic installation and these modules
### Python
- numpy
- matplotlib
- nibabel
- nilearn
- seaborn

### Other
- qpdf [can be installed as `brew install qpdf`]
