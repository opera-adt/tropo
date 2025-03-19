from __future__ import annotations

import functools

import click
from cmap import Colormap

__all__ = ["make_browse"]
# Always show defaults
click.option = functools.partial(click.option, show_default=True)


@click.command()
@click.option("-o", "--out-fname", help="Path to output png file")
@click.option("-i", "--in-fname", required=True, help="Path to input NetCDF file")
@click.option(
    "-m",
    "--max-img-dim",
    default=2048,
    help="Maximum dimension allowed for either length or width of browse image",
)
@click.option("--cmap", default="arctic_r", help="Colormap to use for the image")
@click.option("--vmin", default=1.9, type=float, help="Minimum value for color scaling")
@click.option("--vmax", default=2.5, type=float, help="Maximum value for color scaling")
@click.option("--height", default=800, type=float, help="Tropo height level to plot")
def make_browse(out_fname, in_fname, max_img_dim, cmap, vmin, vmax, height):
    """Create browse images for troposphere products from command line."""
    import opera_tropo.browse_image

    if out_fname is None:
        out_fname = in_fname.replace(".nc", ".png")

    cmap = Colormap(cmap).to_mpl()
    opera_tropo.browse_image.make_browse_image_from_nc(
        out_fname, in_fname, max_img_dim, cmap, vmin, vmax, height
    )
