# Create a two-page PDF containing axial multislice image of the participants scan with the examples of elevated and non-elevated scans

This code generates an image of the selected axial slices for each of the available amyloid and tau PET tracers with the possibility to change colormap, color range, add quantification results and visual read information, as well as display PET overlaid on the participant's MRI scan.
The PDF output is developed by the UCSF visual read core and is a standard output for ADNI, US POINTER, CLARiTI, and DoD-ADBI studies.

## Input files
 - Tracer-specific color-ranges and selected slices are selected assuming the scan is an SUVR image with re-center coordinates (to the center of the FOV). For most of studies, affine-transformed (to MNI) SUVr PET or Step 4 image from LONI (Avg, Std Img and Vox Size, Uniform Res)
    is an input.
 - Code does not assume any filename structure, therefore, input filename can be any. 
 - To correctly record subject information, ID, date of scan, and modality need to be entered. Centiloid, SUVr, and visual read information can be provided by parsing arguments (see below).

## Output files
- Multi-slice visualization: `[subject]_[tracer]_[date]_multislice.pdf`
- Merged template visualization: `[subject]_[tracer]_[date]_multislice_merged.pdf`

## Options:
  -h, --help            show this help message and exit
  -pet PET              Path to the input affine transformed PET scan
  -mri MRI              Path to the input affine transformed MRI scan
  -s S                  Subject ID (e.g., 001_S_001)
  -d D                  Date of the scan (e.g., 2023-01-01)
  -m {FBB,FBP,NAV,PIB,FTP,MK6240,PI2620,MRI-T1}
                        Modality of the input scan (choices: FBB, FBP, NAV, PIB, FTP, MK6240, PI2620, MRI-T1)
  -suvr SUVR            Quantification in SUVR
  -cl CL                Centiloid value for the scan
  -vr {Elevated,Non-elevated}
                        Visual read of the scan (choices: Elevated, Non-elevated)
  -t T                  Path to the template directory for merging multislice images (default: templates/)
  -z Z [Z ...]          List of image slices to show along the z-axis (default: [-50, -44, -38, -32, -26, -20, -14, -8, -2, 4, 10, 16, 22, 28, 34, 40])
  --crop, --no-crop     Crop the multislice images to the brain (default: True)
  --mask_thresh MASK_THRESH
                        Threshold for cropping empty voxels outside the brain (default: 0.05)
  --crop_prop CROP_PROP
                        Proportion of empty voxels allowed when cropping (default: 0.05)
  --cmap CMAP           Colormap to use for the multislice images
  --vmin VMIN           Minimum intensity threshold (default: 0)
  --vmax VMAX           Maximum intensity threshold
  --autoscale           Autoscale vmax to a percentile of image values > 0
  --autoscale_max_pct AUTOSCALE_MAX_PCT
                        Percentile for autoscaling vmax (default: 99.5)
  -o, --overwrite       Overwrite existing files
  -q, --quiet           Run without printing output

### Example run from the command line
`python pdf_multislice.py -pet myfilename.nii -s 001_S_001 -d 2025-01-01 -m FBB -cl 24 -vr Elevated`

## Code requires python basic installation and these modules
### Python
- numpy
- matplotlib
- nibabel
- nilearn
- seaborn

### Other
- qpdf [can be installed as `brew install pdf`]