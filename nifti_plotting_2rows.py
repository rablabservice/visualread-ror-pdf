import sys
import os.path as op
import warnings
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
from nilearn import plotting
import utils.custom_colormaps as custom_colormaps
from utils.helper_funcs import *
import utils.array_operations as aop
import utils.str_methods as strm
import utils.nifti_ops as nops

mpl.rcParams["pdf.fonttype"] = 42

def create_multislice(
    petf,
    mrif=None,
    subj=None,
    tracer=None,
    image_date=None,
    suvr=None,
    centiloid=None,
    visual_read=None,
    title=None,
    display_mode="z",
    cut_coords=[-50, -44, -38, -32, -26, -20, -14, -8, -2, 4, 10, 16, 22, 28, 34, 40],
    cmap=None,
    cmap_mri=None,
    colorbar=True,
    cbar_tick_format=".1f",
    n_cbar_ticks=5,
    cbar_label=None,
    vmin=0,
    alpha=None,
    vmax=None,
    vmin_mri=0,
    vmax_mri=None,
    hide_cbar_values=False,
    autoscale=False,
    autoscale_values_gt0=True,
    autoscale_min_pct=0.5,
    autoscale_max_pct=99.5,
    crop=True,
    mask_thresh=0.05,
    crop_prop=0.05,
    annotate=False,
    draw_cross=False,
    facecolor=None,
    fontcolor=None,
    font={"tick": 12, "label": 14, "title": 16, "annot": 14},
    figsize=(13.29, 7.5),
    dpi=300,
    pad_figure=True,
    fig=None,
    ax=None,
    save_fig=True,
    savename=None,
    overwrite=False,
    verbose=True,
    **kws,
):
    """Create a multislice plot of image and return the saved file.

    Parameters
    ----------
    petf : str
        The path to the PET image file to plot.
    mrif : str, default : None
        The path to the MRI image file to plot. If None, only the PET
        image is plotted.
    subj : str, default : None
        The subject ID. Used only for setting the figure title if an
        explicit title is not provided.
    tracer : str, default : None
        The PET tracer used. Used for:
        - Setting the figure title if an explicit title is not provided
        - Setting vmin and vmax if these are not provided and autoscale
          is False
        - Setting cmap, facecolor, and fontcolor if these are not
          provided
    image_date : str, default : None
        The image acquisition date. Used only for setting the figure
        title if an explicit title is not provided.
    suvr : float, default : None
        The SUVR value to display in the figure title. If tracer is amyloid
        and centilois is available, no SUVR value is displayed.
    centiloid : float, default : None
        Centiloid value tp display in the figure title.
    title : str, optional
        The figure title. If provided no other title elements are added.
    display_mode : str, default : 'z'
        The direction of slice cuts (see nilearn.plotting.plot_img):
        - 'x': sagittal
        - 'y': coronal
        - 'z': axial
        - 'ortho': 3 cuts are performed in orthogonal directions
        - 'tiled': 3 cuts are performed and arranged in a 2x2 grid
        - 'mosaic': 3 cuts are performed along multiple rows and columns
    cut_coords : list, default : [-50, -38, -26, -14, -2, 10, 22, 34]
        The MNI coordinates of the point where the cut is performed
        (see nilearn.plotting.plot_img):
        - If display_mode is 'ortho' or 'tiled', this should be a
          3-tuple: (x, y, z)
        - For display_mode == 'x', 'y', or 'z', then these are the
          coordinates of each cut in the corresponding direction
        - If None is given, the cuts are calculated automatically
        - If display_mode is 'mosaic', and the number of cuts is the
          same for all directions, cut_coords can be specified as an
          integer. It can also be a length 3 tuple specifying the number
          of cuts for every direction if these are different
        - If display_mode is 'x', 'y', or 'z', cut_coords can be an
          integer, in which case it specifies the number of cuts to
          perform
    cmap : str, default: None
        The colormap to use. Either a string that is a name of a
        matplotlib colormap, or a matplotlib colormap object. "nih" as
        defined by mricron and "turbo" are also recognized.
    cmap_mri : str, default: None
        The colormap to use for the MRI image. Either a string that is a
        name of a matplotlib colormap, or a matplotlib colormap object.
        "gray" is recommended for MRI images.
    colorbar : bool, default : False
        If True, a colorbar is displayed below the image slices
        showing color mappings from vmin to vmax.
    cbar_tick_format : str, default : '%.2f'
        Controls how to format the tick labels of the colorbar. Ex:
        use "%i" to display as integers.
    n_cbar_ticks : int, default : 3
        The number of ticks to display on the colorbar.
    cbar_label : str, default : None
        The colorbar label. If None, the code will try to infer this
        from the tracer name.
    vmin : float, default: 0
        The minimum value of the colormap range.
    vmax : float, default: None
        The maximum value of the colormap range.
    hide_cbar_values : bool, default : False
        If True, the colorbar values are not displayed but are merely
        labeled from "min" to "max." Overrides n_cbar_ticks.
    autoscale : bool, default: False
        If True, autoscale vmin and vmax according to min and max
        percentiles of image voxel values. Does not override vmin or
        vmax if these variables are already defined (and hence, the
        default behavior is to set vmin to 0 and autoscale only the
        vmax intensity).
    autoscale_values_gt0 : bool, default: True
        If True, colormap intensities are autoscaled using only voxel
        values greater than zero to determine percentile cutoffs.
    autoscale_min_pct: float, default: 0.5
        The percentile of included voxel values to use for autoscaling
        the minimum colormap intensity (vmin).
    autoscale_max_pct: float, default: 99.5
        The percentile of included voxel values to use for autoscaling
        the maximum colormap intensity (vmax).
    crop : bool, default : True
        If True, the code attempts to crop the image to remove empty
        space around the edges.
    mask_thresh : float, default : None
        Cropping threshold for the first image; used together with
        crop_prop to determine how aggresively to remove planes of
        mostly empty space around the image.
    crop_prop : float, default : 0.05
        The cropping threshold for removing empty space around the edges
        of the image.
    annotate : bool, default : False
        If True, positions and L/R annotations are added to the plot.
    draw_cross : bool, default : False
        If True, a cross is drawn on top of the image slices to indicate
        the cut position.
    facecolor : str, default : None
        The background color of the figure.
    fontcolor : str, default : None
        The font color used for all text in the figure.
    font : dict, default : {'tick':12,'label':14,'title':16,'annot':14}
        Font sizes for the different text elements.
    figsize : tuple, default : (13.33, 7.5)
        The figure size in inches (w, h).
    dpi : int, default : 300
        The figure resolution.
    pad_figure : bool, default : True
        If True, whitespace is added at top and bottom of the figure.
    fig : matplotlib.figure.Figure, default : None
        The preexisting figure to use.
    ax : matplotlib.axes.Axes, default : None
        The preexisting axes to use.
    save_fig : bool, default : True
        If True, the figure is saved if overwrite is False or outfile
        doesn't already exist.
    savename : str, default : None
        The path to the output file. If None, the output file is created
        automatically by appending '_multislice' to the input image
        filename.
    overwrite : bool, default : False
        If True and save_fig, outfile is overwritten if it already
        exists.
    verbose : bool, default : True
        If True, print status messages.
    **kws are passed to nifti_ops.load_nii()

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    # Return the savename if it already exists.
    if savename is None:
        if petf is not None:
            if visual_read is not None:
                savename = (
                    strm.add_presuf(petf, suffix="_multislice")
                    .replace(".nii.gz", ".pdf")
                    .replace(".nii", ".pdf")
                )
            else:
                savename = (
                    strm.add_presuf(petf, suffix="_multislice_noread")
                    .replace(".nii.gz", ".pdf")
                    .replace(".nii", ".pdf")
                )
        elif mrif is not None:
            savename = (
                strm.add_presuf(mrif, suffix="_multislice")
                .replace(".nii.gz", ".pdf")
                .replace(".nii", ".pdf")
            )
    if op.isfile(savename) and not overwrite:
        if verbose:
            print(
                "  See existing multislice PDF:"
                + "\n\t{}".format(op.dirname(savename))
                + "\n\t{}".format(op.basename(savename))
            )
        return None, savename
    if petf is not None:
        # Check that the image file exists.
        nops.find_gzip(petf, raise_error=True)
    if mrif is not None:
        nops.find_gzip(mrif, raise_error=True)

    # ----------------------------------------------------------------------------------------
    # Configure plot parameters.

    # Get min and max values for the colormap using autoscale
    # percentages if autoscale is True and if vmin and vmax are not
    # already defined.
    if autoscale:
        img, dat = nops.load_nii(petf, **kws)
        if autoscale_values_gt0:
            dat = dat[dat > 0]
        if vmin is None:
            vmin = np.percentile(dat, autoscale_min_pct)
        if vmax is None:
            vmax = np.percentile(dat, autoscale_max_pct)

    # Get tracer-specific plotting parameters.
    tracer, tracer_fancy, vmin, vmax, cmap, facecolor, fontcolor = get_tracer_defaults(
        tracer,
        petf,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        facecolor=facecolor,
        fontcolor=fontcolor,
    )
    if cmap == "nih":
        cmap = custom_colormaps.nih_cmap()
    elif cmap == "avid":
        cmap = custom_colormaps.avid_cmap()
    elif cmap == "turbo":
        cmap = custom_colormaps.turbo_cmap()
    if cmap == "neon_red":
        cmap = custom_colormaps.neon_red()
    if cmap_mri == "neon_red":
        cmap_mri = custom_colormaps.neon_red()
    if cmap == "neon_green":
        cmap = custom_colormaps.neon_green()
    if cmap_mri == "neon_green":
        cmap_mri = custom_colormaps.neon_green()
    
    if mrif is not None:
        facecolor = "k"
        fontcolor = "w"

    # Crop the data array.
    if petf is not None:
        pet, pet_dat = nops.load_nii(petf, **kws)
        if crop:
            if mask_thresh is None:
                mask_thresh = vmin * 2
            mask = pet_dat > mask_thresh
            pet_dat = aop.crop_arr3d(pet_dat, mask, crop_prop)
            pet = nib.Nifti1Image(pet_dat, pet.affine)
            pet, *_ = nops.recenter_nii(pet)
            pet_crop_params = {
                "mask": mask,
                "crop_prop": crop_prop,
                "affine": pet.affine,
                "recenter": True,
            }
        else:
            pet_crop_params = None
    else:
        pet_crop_params = None

    if mrif is not None:
        mri, mri_dat = nops.load_nii(mrif, **kws)
        if crop and pet_crop_params is not None:
            # Apply the same crop and recenter parameters as PET
            mri_dat = aop.crop_arr3d(mri_dat, pet_crop_params["mask"], pet_crop_params["crop_prop"])
            mri = nib.Nifti1Image(mri_dat, pet_crop_params["affine"])
            if pet_crop_params["recenter"]:
                mri, *_ = nops.recenter_nii(mri)
        elif crop:
            if mask_thresh is None:
                mask_thresh = vmin * 2
            mask = mri_dat > mask_thresh
            mri_dat = aop.crop_arr3d(mri_dat, mask, crop_prop)
            mri = nib.Nifti1Image(mri_dat, mri.affine)
            mri, *_ = nops.recenter_nii(mri)
        if vmax_mri is None:
            vmax_mri = np.percentile(mri_dat, 90)
        if vmin_mri is None:
            vmin_mri = np.percentile(mri_dat, 10)
        if cmap_mri is None:
            cmap_mri = "gray"
        if alpha is None:
            alpha = 0.7

    # Define slice parameters before creating the figure.
    nrows = 2  
    ncols = 8  
   
    # Make the plot.
    plt.close("all")
    if fig is None or ax is None:
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,  figsize=figsize, dpi=dpi)
        plt.subplots_adjust(hspace=-0.5, wspace=0)
        if ax.ndim == 1:
            ax = np.reshape(ax, (nrows, ncols))  
            
    if pad_figure:
        iax = 1
    else:
        iax = 0
    _ax = ax[iax]

    # Format remaining plot parameters.
    if len(cut_coords) == 1:
        cut_coords = cut_coords[0]

    black_bg = True if facecolor == "k" else False
    if display_mode == "mosaic":
        cut_coords = None
        _colorbar = True
        colorbar = False
    elif display_mode in ("ortho", "tiled"):
        _colorbar = True
        colorbar = False
    else:
        _colorbar = False

    # ---------------------------------------------------------
    # Call the plotting function for each slice.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for i, coord in enumerate(cut_coords):  # Loop through all slices
        row = i // ncols
        col = i % ncols
        
        if mrif is not None:
            display = plotting.plot_img(
                mri,
                cut_coords=[coord],  # Plot each slice individually
                display_mode=display_mode, 
                annotate=False,
                draw_cross=draw_cross,
                black_bg=False,
                cmap=cmap_mri,
                colorbar=False,  # Disable individual colorbars
                vmin=vmin_mri,
                vmax=vmax_mri,
                title=None,
                figure=fig,
                axes=ax[row, col],  # Use the corresponding subplot
            )
            if petf is not None:
                display.add_overlay( # Add the PET overlay
                    pet,
                    threshold=0,
                    alpha=alpha,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
        else:
            _ = plotting.plot_img(
                pet,
                cut_coords=[coord],  # Plot each slice individually
                display_mode=display_mode,  
                annotate=annotate,
                draw_cross=draw_cross,
                black_bg=False,
                cmap=cmap,
                colorbar=False,  # Disable individual colorbars
                vmin=vmin,
                vmax=vmax,
                title=None,
                figure=fig,
                axes=ax[row, col],  # Use the corresponding subplot
            ) 
    warnings.resetwarnings()

    # Turn off empty subplots (e.g., if len(cut_coords) is less than the total subplots)
    for i in range(len(cut_coords), nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax[row, col].axis("off")
        ax[row, col].set_facecolor('none') 
        
    # Set facecolor to None for all subplots
    for row in range(nrows):
        for col in range(ncols):
            ax[row, col].set_facecolor('none')  # Set facecolor to None for all axes
            ax[row, col].axis("off")  # Optionally, turn off the axis if needed

    # ----------------------------------------------------------
    # Add the colorbar.
    if colorbar and petf is not None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=_ax,
            location="bottom",
            pad=0.05,
            shrink=0.25,
            aspect=15,
            drawedges=False,
        )
        cbar.outline.set_color(fontcolor)
        cbar.outline.set_linewidth(1)
        cbar.ax.tick_params(
            labelsize=font["tick"], labelcolor=fontcolor, color=fontcolor, width=1
        )
        if hide_cbar_values:
            cbar.ax.set_xticks([vmin, vmax])
            cbar.ax.set_xticklabels(["Low", "High"])
        else:
            cbar_ticks = np.linspace(vmin, vmax, n_cbar_ticks)
            cbar.ax.set_xticks(cbar_ticks)
            cbar.ax.set_xticklabels(
                ["{:{_}}".format(tick, _=cbar_tick_format) for tick in cbar_ticks]
            )
        if cbar_label is None:
            if (tracer is None) or (tracer==""):
                cbar_label = "Value"
            else:
                cbar_label = f"{tracer_fancy} SUVR"
        cbar.ax.set_xlabel(
            cbar_label,
            fontsize=font["label"],
            color=fontcolor,
            labelpad=8,
        )
        # Get the current position of the colorbar as [left, bottom, width, height]
        pos = cbar.ax.get_position().bounds

        # Modify the 'bottom' position to move it up
        cbar.ax.set_position([pos[0], pos[1] + 0.02, pos[2], pos[3]])

    # -----------------------------------------------------------
    # Add the title.
    if title is None:
        title = "\n"
        if subj:
            title += f"Participant: {subj}\n"
        if image_date:
            title += f"Scan date: {image_date}\n"
        if tracer:
            title += f"Tracer: {tracer_fancy}\n\n"
        if suvr and not centiloid:
            if tracer == 'fbb' or tracer == 'fbp' or tracer == 'nav' or tracer == 'pib' or tracer == 'flute':
                title += f"Amyloid cortical mask SUVR: {suvr}\n"
            elif tracer == 'ftp' or tracer == 'mk6240':
                title += f"Temporal meta-ROI SUVR: {suvr}\n"
        if visual_read:
            if tracer == 'ftp' or tracer == 'mk6240' or tracer == 'pi2620':
                if visual_read == 'Elevated':
                    visual_read = 'Elevated (AD pattern)'
                elif visual_read == 'Non-elevated':
                    visual_read = 'Non-elevated (AD pattern)'
            title += f"Expert visual read: {visual_read}\n"
        if centiloid:
            title += f"Centiloid: {centiloid}\n"  # Fixed indentation here
        if hide_cbar_values:
            title += "SUVR range: {:.1f}-{:.1f}".format(vmin, vmax)

    ax[0, 0].set_title(
        title,
        fontsize=font["title"],
        color=fontcolor,
        loc="left",
        zorder=100,
    )
    ax[0, 0].set_position([0.1, 0.63, 0.3, 0.1])
    ax[0, 0].set_facecolor("none") 

    # Add L/R text
    ax[0, 0].text(
        x=0.08, y=0.68, s="L",
        fontsize=font["title"], color=fontcolor,
        #transform=ax.transAxes,  # Set the coordinate system (default is 'data')
        #horizontalalignment="center"
    )
    ax[0, 7].text(
        x=0.9, y=0.65, s="R",
        fontsize=font["title"], color=fontcolor,
        #transform=ax.transAxes,  # Set the coordinate system (default is 'data')
        #horizontalalignment="center"
    )

    # Set the background color.
    for iax in range(len(ax)):
        _ax = ax[iax,0]
        _ax.set_facecolor(facecolor)
    fig.patch.set_facecolor(facecolor)
    
    # Get rid of lines at the top and bottom of the figure.
    #if pad_figure:
    rows_to_disable = [0, 1]  # Specify rows as a list
    cols_to_disable = [0, 1, 2]
    for iax in rows_to_disable:
        for jax in cols_to_disable:
            ax[iax, jax].axis("off")

    # -----------------------------------------------------------------------------
    # Save the figure as a pdf.
    
    if save_fig:
        fig.savefig(
            savename,
            facecolor=facecolor,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.2,
        )
        if verbose:
            print(
                "  Saved new multislice PDF:"
                + "\n\t{}".format(op.dirname(savename))
                + "\n\t{}".format(op.basename(savename))
            )
    elif not op.isfile(savename):
        savename = None

    return fig, savename

# Get the tracer defaults for colormap, vmin, vmax, etc.
# based on the tracer name.
def get_tracer_defaults(
    tracer, petf=None, vmin=None, vmax=None, cmap=None, facecolor=None, fontcolor=None
):
    """Set undefined plot parameters based on tracer."""
    tracer_labels = {
        "fbb": "[18F]Florbetaben (amyloid)",
        "fbp": "[18F]Florbetapir (amyloid)",
        "nav": "[18F]NAV4694 (amyloid)",
        "pib": "[11C]PIB (amyloid)",
        "ftp": "[18F]Flortaucipir (tau)",
        "fdg": "[18F]FDG",
        "flute": "[18F]Flutemetamol (amyloid)",
        "mk6240": "[18F]MK6240 (tau)",
        "pi2620": "[18F]PI2620 (tau)",
    }
    if tracer is None:
        tracer = ""
    _tracer = tracer
    tracer = tracer.lower()
    if (tracer not in tracer_labels) and (petf is not None):
        infile_basename = op.basename(imagef).lower()
        for key in tracer_labels:
            if key in infile_basename:
                tracer = key
    tracer_fancy = tracer_labels.get(tracer, _tracer)
    if tracer == "fbb":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.5
        if cmap is None:
            cmap = "binary_r"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    elif tracer == "fbp":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.5
        if cmap is None:
            cmap = "binary"
        if facecolor is None:
            facecolor = "w"
        if fontcolor is None:
            fontcolor = "k"
    elif tracer == "nav":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 3.0
        if cmap is None:
            cmap = "nih"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    elif tracer == "pib":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.5
        if cmap is None:
            cmap = "nih"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    elif tracer == "ftp":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 3.7
        if cmap is None:
            cmap = "avid"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    elif tracer == "mk6240":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 4.5
        if cmap is None:
            cmap = "nih"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    elif tracer == "fdg":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.2
        if cmap is None:
            cmap = "nih"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    elif tracer == "flute":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.5
        if cmap is None:
            cmap = "nih"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    elif tracer == "pi2620":
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 4.5
        if cmap is None:
            cmap = "nih"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"
    else:
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 2.5
        if cmap is None:
            cmap = "binary_r"
        if facecolor is None:
            facecolor = "k"
        if fontcolor is None:
            fontcolor = "w"

    # Import custom colormaps.
    if cmap == "avid":
        cmap = custom_colormaps.avid_cmap()
    elif cmap == "nih":
        cmap = custom_colormaps.nih_cmap()
    elif cmap == "turbo":
        cmap = custom_colormaps.turbo_cmap()

    return tracer, tracer_fancy, vmin, vmax, cmap, facecolor, fontcolor


class TextFormatter(argparse.RawTextHelpFormatter):
    """Custom formatter for argparse help text."""

    # use defined argument order to display usage
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "usage: "

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = "%(prog)s" % dict(prog=self._prog)
        elif usage is None:
            prog = "%(prog)s" % dict(prog=self._prog)
            # build full usage string
            action_usage = self._format_actions_usage(actions, groups)  # NEW
            usage = " ".join([s for s in [prog, action_usage] if s])
            # omit the long line wrapping code
        # prefix with 'usage:'
        return "%s%s\n\n" % (prefix, usage)