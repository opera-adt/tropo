import click

from opera_tropo.log.loggin_setup import setup_logging
from opera_tropo.validate import compare_two_datasets


@click.command()
@click.argument("golden", type=click.Path(exists=True))
@click.argument("test", type=click.Path(exists=True))
@click.option("--debug", is_flag=True)
def validate(golden: str, test: str, debug: bool) -> None:
    """Validate an OPERA TROPO product."""
    setup_logging(logger_name="opera_tropo", debug=debug)
    compare_two_datasets(golden, test)
