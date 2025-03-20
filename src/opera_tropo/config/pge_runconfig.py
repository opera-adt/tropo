from pathlib import Path
from typing import ClassVar, Optional

from pydantic import ConfigDict, Field

from ._yaml import YamlModel
from .runconfig import InputOptions, OutputOptions, TropoWorkflow, WorkerSettings


class PrimaryExecutable(YamlModel):
    """Group describing the primary executable."""

    product_type: str = Field(
        default="OPERA_TROPO",
        description="Product type of the PGE.",
    )
    model_config = ConfigDict(extra="forbid")


class ProductPathGroup(YamlModel):
    """Group describing the product paths."""

    product_path: Path = Field(
        default=Path(),
        description="Directory where PGE will place results",
    )
    scratch_path: Path = Field(
        default=Path("./scratch"),
        description="Path to the scratch directory.",
    )
    output_path: Path = Field(
        default=Path("./output"),
        description="Path to the SAS output directory.",
        # The alias means that in the YAML file, the key will be "sas_output_path"
        # instead of "output_directory", but the python instance attribute is
        # "output_directory" (to match DisplacementWorkflow)
        alias="sas_output_path",
    )
    product_version: str = Field(
        default="0.1",
        description="Version of the product, in <major>.<minor> format.",
    )
    model_config = ConfigDict(extra="forbid")


class RunConfig(YamlModel):
    """A PGE run configuration."""

    # Used for the top-level key
    name: ClassVar[str] = "opera_tropo_workflow"

    input_file: InputOptions = Field(default_factory=InputOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)
    primary_executable: PrimaryExecutable = Field(default_factory=PrimaryExecutable)
    product_path_group: ProductPathGroup = Field(default_factory=ProductPathGroup)

    # General workflow metadata
    worker_settings: WorkerSettings = Field(default_factory=WorkerSettings)

    log_file: Optional[Path] = Field(
        default=Path("output/opera_tropo_workflow.log"),
        description="Path to the output log file in addition to logging to stderr.",
    )
    # model_config = ConfigDict(extra="forbid")

    @classmethod
    def model_construct(cls, **kwargs):
        """Recursively use model_construct without validation."""
        if "input_file" not in kwargs:
            kwargs["input_file"] = InputOptions.model_construct()
        if "product_path_group" not in kwargs:
            kwargs["product_path_group"] = ProductPathGroup.model_construct()
        return super().model_construct(
            **kwargs,
        )

    def to_workflow(self):
        """Convert to a `TropoWorkflow` object."""
        # We need to go from the PGE format to opera_tropo's TropoWorkflow:
        # Note that the top two levels of nesting can be accomplished by wrapping
        # the normal model export in a dict.
        #
        # The things from the RunConfig that are used in the
        # TropoWorkflow are the input files,
        # the output directory, and the scratch directory.

        # input_file = (self.input_file_group.input_file_group.input_file_path
        scratch_directory = self.product_path_group.scratch_path
        output_directory = self.product_path_group.output_path
        tmp_directory = Path(scratch_directory) / "tmp"
        tmp_directory.mkdir(parents=True, exist_ok=True)
        worker_settings = self.worker_settings.copy()
        worker_settings.dask_temp_dir = str(tmp_directory)
        output_options = self.output_options.copy()
        output_options.product_version = self.product_path_group.product_version

        return TropoWorkflow(
            input_options=self.input_file.__dict__,
            output_options=output_options.__dict__,
            work_directory=scratch_directory,
            output_directory=output_directory,
            # These ones directly translate
            worker_settings=worker_settings.__dict__,
            log_file=self.log_file,
        )  # type: ignore
