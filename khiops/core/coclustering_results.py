######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Classes to access Khiops Coclustering JSON reports

Class Overview
--------------
Below we describe with diagrams the relationships of the classes in this modules. They
are mostly compositions (has-a relations) and we omit native attributes (str, int,
float, etc).

The main class of this module is `CoclusteringResults` and it is largely a composition
of sub-reports objects given by the following structure::

    CoclusteringResults
    |- coclustering_report -> CoclusteringReport

    CoclusteringReport
    |- dimensions -> list of CoclusteringDimension
    |- cells      -> list of CoclusteringCell

    CoclusteringDimension
    |- parts                    -> list of CoclusteringDimensionPart
    |- variable_part_dimensions -> list of CoclusteringDimension
    |- clusters                 -> list of CoclusteringCluster
    |- root_cluster             -> CoclusteringCluster

    CoclusteringDimensionPartValueGroup
    |- values -> list of CoclusteringDimensionPartValue

    CoclusteringCluster
    |- leaf_part        -> CoclusteringDimensionPart or None
    |- parent_cluster  |
    |- child_cluster1  |-> CoclusteringCluster or None
    |- child_cluster2  |

To have a complete illustration of the access to the information of all classes in this
module look at their ``to_dict`` methods which write Python dictionaries in the
same format as the Khiops JSON reports.
"""
import io
import warnings

from khiops.core.exceptions import KhiopsJSONError
from khiops.core.internals.common import deprecation_message, type_error_message
from khiops.core.internals.io import (
    KhiopsJSONObject,
    KhiopsOutputWriter,
    flexible_json_load,
)


class CoclusteringResults(KhiopsJSONObject):
    """Main class containing the information of a Khiops Coclustering JSON file

    Parameters
    ----------
    json_data : dict, optional
        Python dictionary representing the data of a Khiops Coclustering JSON report
        file. If not specified it returns an empty instance.

        .. note::
            Prefer either the the `read_coclustering_results_file` function from the
            core API to obtain an instance of this class from a Khiops Coclustering JSON
            file.

    Attributes
    ----------
    tool : str
        Name of the Khiops tool that generated the JSON file.
    version : str
        Version of the Khiops tool that generated the JSON file.
    coclustering_report : `CoclusteringReport`
        Coclustering modeling report.
    """

    # Set coclustering report order key specification
    # pylint: disable=line-too-long
    json_key_sort_spec = {
        "tool": None,
        "version": None,
        "shortDescription": None,
        "coclusteringReport": {
            "summary": {
                "instances": None,
                "cells": None,
                "nullCost": None,
                "cost": None,
                "level": None,
                "initialDimensions": None,
                "frequencyVariable": None,
                "dictionary": None,
                "database": None,
                "samplePercentage": None,
                "samplingMode": None,
                "selectionVariable": None,
                "selectionValue": None,
            },
            "dimensionSummaries": [
                {
                    "name": None,
                    "isVarPart": None,
                    "type": None,
                    "parts": None,
                    "initialParts": None,
                    "values": None,
                    "interest": None,
                    "description": None,
                    "min": None,
                    "max": None,
                },
            ],
            "dimensionPartitions": [
                {
                    "name": None,
                    "type": None,
                    "innerVariables": {
                        "dimensionSummaries": [
                            {
                                "name": None,
                                "type": None,
                                "parts": None,
                                "initialParts": None,
                                "values": None,
                                "interest": None,
                                "description": None,
                                "min": None,
                                "max": None,
                            }
                        ],
                        "dimensionPartitions": [
                            {
                                "name": None,
                                "type": None,
                                "intervals": [
                                    {
                                        "cluster": None,
                                        "bounds": None,
                                    }
                                ],
                                "valueGroups": [
                                    {
                                        "cluster": None,
                                        "values": None,
                                        "valueFrequencies": None,
                                    }
                                ],
                                "defaultGroupIndex": None,
                            }
                        ],
                    },
                    "intervals": [
                        {
                            "cluster": None,
                            "bounds": None,
                        }
                    ],
                    "valueGroups": [
                        {
                            "cluster": None,
                            "values": None,
                            "valueFrequencies": None,
                            "valueTypicalities": None,
                        }
                    ],
                    "defaultGroupIndex": None,
                },
            ],
            "dimensionHierarchies": [
                {
                    "name": None,
                    "type": None,
                    "clusters": [
                        {
                            "cluster": None,
                            "parentCluster": None,
                            "frequency": None,
                            "interest": None,
                            "hierarchicalLevel": None,
                            "rank": None,
                            "hierarchicalRank": None,
                            "isLeaf": None,
                            "shortDescription": None,
                            "description": None,
                        }
                    ],
                }
            ],
            "cellPartIndexes": None,
            "cellFrequencies": None,
        },
        "khiops_encoding": None,
        "ansi_chars": None,
        "colliding_utf8_chars": None,
    }
    # pylint: enable=line-too-long

    def __init__(self, json_data=None):
        """See class docstring"""
        # Initialize super class
        super().__init__(json_data=json_data)

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Initialize empty report attributes
        self.short_description = json_data.get("shortDescription", "")

        # Initialize the sub-report only if available
        if "coclusteringReport" in json_data:
            self.coclustering_report = CoclusteringReport(
                json_data["coclusteringReport"]
            )
        else:
            self.coclustering_report = None

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report = super().to_dict()
        if self.short_description is not None:
            report["shortDescription"] = self.short_description
        if self.coclustering_report is not None:
            report["coclusteringReport"] = self.coclustering_report.to_dict()
        return report

    def write_report_file(self, report_file_path):  # pragma: no cover
        """Writes a TSV report file with the object's information

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        report_file_path : str
            Path of the output TSV report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_file", "12.0.0"))

        # Write report to file
        with open(report_file_path, "wb") as report_file:
            writer = self.create_output_file_writer(report_file)
            self.write_report(writer)

    def write_report(self, stream_or_writer):  # pragma: no cover
        """Writes the instance's TSV report to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        stream_or_writer : `io.IOBase` or `.KhiopsOutputWriter`
            Output stream or writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report", "12.0.0", "to_dict"))

        # Check input writer/stream type
        if isinstance(stream_or_writer, io.IOBase):
            writer = self.create_output_file_writer(stream_or_writer)
        elif isinstance(stream_or_writer, KhiopsOutputWriter):
            writer = stream_or_writer
        else:
            raise TypeError(
                type_error_message(
                    "stream_or_writer",
                    stream_or_writer,
                    io.IOBase,
                    KhiopsOutputWriter,
                )
            )

        # Write report
        writer.writeln(f"#Khiops {self.version}")
        writer.writeln(f"Short description\t{self.short_description}")
        if self.coclustering_report is not None:
            self.coclustering_report.write_report(writer)


def read_coclustering_results_file(json_file_path):
    """Reads a Khiops Coclustering JSON report

    Parameters
    ----------
    json_file_path : str
        Path of the JSON report file.

    Returns
    -------
    `.CoclusteringResults`
        An instance of CoclusteringResults containing the report's information.
    """
    return CoclusteringResults(json_data=flexible_json_load(json_file_path))


class CoclusteringReport:
    """Main coclustering report

    A coclustering is an unsupervised data grid equipped with additional structures to
    ease its exploration. In particular, it is a piecewise constant density estimator of
    the data distribution. The additional structures are the following:

    - A cluster hierarchy for each dimension
    - Indicators (such as the interest) for each variable, part and value.

    A coclustering consists of one to many variables (dimensions), where each variable
    is partitioned as:

    - Intervals in the numerical case
    - Individual values or value groups in the categorical case.

    The cross-product of the partitions forms a multivariate partition of cells and
    their frequencies allow to estimate the multivariate density.

    In case of an unsupervised data grid, the cells are described by their index on the
    variable partitions, together with their frequencies.

    Parameters
    ----------
    json_data : dict, optional
        JSON data of the ``coclusteringReport`` field of a Khiops Coclustering JSON
        report file. If not specified it returns an empty instance.

    Attributes
    ----------
    instance_number : int
        Number of individuals in the learning data table.
    cell_number : int
        Number of coclustering cells.
    null_cost : float
        Cost of the null model.
    level : float
        Measure between 0 and 1 measuring the information gain over the null model.
    initial_dimension_number : int
        Initial number of dimensions. The number of dimensions (``len(dimensions)``) may
        be less than this quantity after a simplification (see
        `~.api.simplify_coclustering`).
    frequency_variable : str
        Name of the variable to be aggregated in the cells. By default is the number of
        individuals.
    dictionary : str
        Name dictionary from which the model was learned.
    database : str
        Path of the main training data table file.
    sample_percentage : float
        Percentage of instances used in training.
    sampling_mode : "Include sample" or "Exclude samples"
        Sampling mode used to split the train and datasets.
    selection_variable : str
        Variable used to select instances for training.
    selection_value : str
        Value of ``selection_variable`` to select instances for training.
    dimensions : list of `CoclusteringDimension`
        Coclustering dimensions (variable).
    cells : list of `CoclusteringCell`
        Coclustering cells.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise raise an exception if the summary is not found
        elif "summary" not in json_data:
            raise KhiopsJSONError("'summary' key not found in coclustering report")

        # Initialize summary fields
        json_summary = json_data.get("summary", {})
        self.instance_number = json_summary.get("instances", 0)
        self.cell_number = json_summary.get("cells", 0)
        self.null_cost = json_summary.get("nullCost", 0)
        self.cost = json_summary.get("cost", 0)
        self.level = json_summary.get("level", 0)
        self.initial_dimension_number = json_summary.get("initialDimensions", 0)
        self.frequency_variable = json_summary.get("frequencyVariable", "")
        self.dictionary = json_summary.get("dictionary", "")
        self.database = json_summary.get("database", "")
        self.sample_percentage = json_summary.get("samplePercentage", 0.0)
        self.sampling_mode = json_summary.get("samplingMode", "")
        self.selection_variable = json_summary.get("selectionVariable")
        self.selection_value = json_summary.get("selectionValue")

        # Create dimensions and initialize their summaries
        self.dimensions = []
        self._dimensions_by_name = {}
        if "dimensionSummaries" in json_data:
            for json_dimension_summary in json_data["dimensionSummaries"]:
                dimension = CoclusteringDimension().init_summary(json_dimension_summary)
                self.dimensions.append(dimension)
                self._dimensions_by_name[dimension.name] = dimension

        # Initialize dimensions' partitions
        if "dimensionPartitions" in json_data:
            json_dimension_partitions = json_data["dimensionPartitions"]
            if len(self.dimensions) != len(json_dimension_partitions):
                raise KhiopsJSONError(
                    "'dimensionPartitions' list has length "
                    f"{len(json_dimension_partitions)} instead of "
                    f"{len(self.dimensions)}"
                )
            for i, json_dimension_partition in enumerate(json_dimension_partitions):
                dimension = self.dimensions[i]
                dimension.init_partition(json_dimension_partition)

        # Initialize dimensions' hierarchies
        if "dimensionHierarchies" in json_data:
            json_dimension_hierarchies = json_data["dimensionHierarchies"]
            if len(self.dimensions) != len(json_dimension_hierarchies):
                raise KhiopsJSONError(
                    "'dimensionHierarchies' list has length "
                    f"{len(json_dimension_hierarchies)} instead of "
                    f"{len(self.dimensions)}"
                )
            for i, json_dimension_hierarchy in enumerate(json_dimension_hierarchies):
                dimension = self.dimensions[i]
                dimension.init_hierarchy(json_dimension_hierarchy)

        # Initialize cells
        self.cells = []
        if "cellPartIndexes" in json_data:
            # Check minimum consistency of input data
            if "cellFrequencies" not in json_data:
                raise KhiopsJSONError(
                    "'cellFrequencies' key not found but 'cellPartIndexes' found."
                )
            json_cell_frequencies = json_data["cellFrequencies"]
            json_cell_part_indexes = json_data["cellPartIndexes"]
            if len(json_cell_part_indexes) != len(json_cell_frequencies):
                raise KhiopsJSONError(
                    "'cellPartIndexes' length is different from "
                    f"that of 'cellFrequencies': {len(json_cell_part_indexes)} != "
                    f"{len(json_cell_frequencies)}"
                )
            # Create and initialize all cells
            for i, part_indexes in enumerate(json_cell_part_indexes):
                # Initialize cell
                cell = CoclusteringCell()
                cell.frequency = json_cell_frequencies[i]

                # Initialize cell parts:
                # Retrieve part from its index in the partition, per dimension
                for j, index in enumerate(part_indexes):
                    dimension = self.dimensions[j]
                    part = dimension.parts[index]
                    cell.parts.append(part)

                # Add to the cell list
                self.cells.append(cell)

    def get_dimension_names(self):
        """Returns the names of the available dimensions

        Returns
        -------
        list of str
            The names of the available dimensions.
        """
        return [dimension.name for dimension in self.dimensions]

    def get_dimension(self, dimension_name):
        """Returns the specified dimension

        Parameters
        ----------
        dimension_name : str
            Name of the dimension (variable).

        Returns
        -------
        `CoclusteringDimension`
            The specified dimension.

        Raises
        ------
        `KeyError`
            If no dimension with the specified names exist.
        """
        return self._dimensions_by_name[dimension_name]

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        # Compute cellPartIndexes
        cell_parts_indexes = []
        for cell in self.cells:
            cell_part_indexes = []
            for cell_part in cell.parts:
                for dimension in self.dimensions:
                    for dimension_part_index, dimension_part in enumerate(
                        dimension.parts
                    ):
                        if cell_part == dimension_part:
                            cell_part_indexes.append(dimension_part_index)
                            break
            cell_parts_indexes.append(cell_part_indexes)
        report_summary = {
            "instances": self.instance_number,
            "cells": self.cell_number,
            "nullCost": self.null_cost,
            "cost": self.cost,
            "level": self.level,
            "initialDimensions": self.initial_dimension_number,
            "frequencyVariable": self.frequency_variable,
            "dictionary": self.dictionary,
            "database": self.database,
            "samplePercentage": self.sample_percentage,
            "samplingMode": self.sampling_mode,
            "selectionVariable": self.selection_variable,
            "selectionValue": self.selection_value,
        }
        report = {
            "summary": report_summary,
            "dimensionSummaries": [
                dimension.to_dict(report_type="summary")
                for dimension in self.dimensions
            ],
            "dimensionPartitions": [
                dimension.to_dict(report_type="partition")
                for dimension in self.dimensions
            ],
            "dimensionHierarchies": [
                dimension.to_dict(report_type="hierarchy")
                for dimension in self.dimensions
            ],
            "cellPartIndexes": cell_parts_indexes,
            "cellFrequencies": [cell.frequency for cell in self.cells],
        }
        return report

    def write_report(self, writer):  # pragma: no cover
        """Writes the instance's TSV report to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output stream or writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report", "12.0.0", "to_dict"))

        # Write each section
        self.write_dimensions(writer)
        self.write_coclustering_stats(writer)
        self.write_bounds(writer)
        self.write_hierarchies(writer)
        self.write_compositions(writer)
        self.write_cells(writer)
        self.write_annotations(writer)

    def write_dimensions(self, writer):  # pragma: no cover
        """Writes the "dimensions" section of the TSV report to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_dimensions", "12.0.0", "to_dict"))

        # Write dimension report
        writer.writeln(f"Dimensions\t{len(self.dimensions)}")
        for i, cc_dimension in enumerate(self.dimensions):
            if i == 0:
                cc_dimension.write_dimension_header_line(writer)
            cc_dimension.write_dimension_line(writer)

    def write_coclustering_stats(self, writer):  # pragma: no cover
        """Writes the "stats" section of the TSV report to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(
            deprecation_message("write_coclustering_stats", "12.0.0", "to_dict")
        )

        # Write report
        writer.writeln("")
        writer.writeln("Coclustering stats")
        writer.writeln(f"Instances\t{self.instance_number}")
        writer.writeln(f"Cells\t{self.cell_number}")
        writer.writeln(f"Null cost\t{self.null_cost}")
        writer.writeln(f"Cost\t{self.cost}")
        writer.writeln(f"Level\t{self.level}")
        writer.writeln(f"Initial dimensions\t{self.initial_dimension_number}")
        writer.writeln(f"Frequency variable\t{self.frequency_variable}")
        writer.writeln(f"Dictionary\t{self.dictionary}")
        writer.writeln(f"Database\t{self.database}")
        writer.writeln(f"Sample percentage\t{self.sample_percentage}")
        writer.writeln(f"Sampling mode\t{self.sampling_mode}")
        writer.writeln(f"Selection variable\t{self.selection_variable}")
        writer.writeln(f"Selection value\t{self.selection_value}")

    def write_bounds(self, writer):  # pragma: no cover
        """Writes the "bounds" section of the TSV report to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_bounds", "12.0.0", "to_dict"))

        # Compute number of numerical dimensions
        numerical_dimension_number = 0
        for cc_dimension in self.dimensions:
            if cc_dimension.type == "Numerical":
                numerical_dimension_number += 1

        # Write "Bounds" section only in case of numerical dimensions
        if numerical_dimension_number > 0:
            writer.writeln("")
            writer.writeln("Bounds")
            writer.writeln("Name\tMin\tMax")
            for cc_dimension in self.dimensions:
                if cc_dimension.type == "Numerical":
                    writer.write(f"{cc_dimension.name}\t")
                    writer.write(f"{cc_dimension.min}\t")
                    writer.writeln(str(cc_dimension.max))

    def write_hierarchies(self, writer):  # pragma: no cover
        """Writes the dimension reports' "hierarchy" sections to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_hierarchies", "12.0.0", "to_dict"))

        # Write dimension hierarchy report for each dimension
        for cc_dimension in self.dimensions:
            cc_dimension.write_hierarchy(writer)

    def write_compositions(self, writer):  # pragma: no cover
        """Writes the dimensions' "composition" sections to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_compositions", "12.0.0", "to_dict"))

        # Write dimension composition report for each dimension
        for cc_dimension in self.dimensions:
            cc_dimension.write_composition(writer)

    def write_cells(self, writer):  # pragma: no cover
        """Writes the "cells" section of the TSV report to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_cells", "12.0.0", "to_dict"))

        # Write header
        writer.writeln("")
        writer.writeln("Cells")

        # Write table header
        for cc_dimension in self.dimensions:
            writer.write(f"{cc_dimension.name}\t")
        writer.writeln("Frequency")

        # Write cell report lines
        for cc_cell in self.cells:
            cc_cell.write_line(writer)

    def write_annotations(self, writer):  # pragma: no cover
        """Writes the dimensions' "annotation" sections to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_annotations", "12.0.0", "to_dict"))

        # Decide whether annotation sections need to be reported
        need_report = False
        for cc_dimension in self.dimensions:
            need_report = need_report or cc_dimension.needs_annotation_report()

        # Write "Annotation" sections for each dimension
        if need_report:
            for cc_dimension in self.dimensions:
                cc_dimension.write_annotation(writer)
        else:
            writer.writeln("")


class CoclusteringDimension:
    """A coclustering dimension (variable)

    A coclustering dimension is a hierarchical clustering of an input variable. The
    leaves of this hierarchy are linked to an element of a partition of the input
    variable. Leaf clusters have variable parts as their children.

    It only has a no-parameter constructor.

    .. note::
        The instance information is initialized with the `init_summary`,
        `init_partition` and `init_hierarchy` methods. Its owner object (class
        `CoclusteringReport`) uses the information found in the fields
        ``dimensionSummaries``, ``dimensionPartitions`` and ``dimensionHierarchies`` to
        coherently initialize the all dimensions with these methods.

    Attributes
    ----------
    name : str
        Name of the variable associated to this dimension.
    is_variable_part : bool
        ``True`` if the dimension is a part of a variable in an instance-variable
        coclustering.
    type : "Numerical" or "Categorical"
        Dimension type.
    part_number : int
        Number of parts of the variable associated to this dimension.
    initial_part_number : int
        Number of initial parts. Note that ``part_number`` <= ``initial_part_number``
        after a coclustering simplification (see `~.api.simplify_coclustering`).
    value_number : int
        Number of values of the dimension's variable.
    interest : float
        Interest of the dimension with respect to the other coclustering dimensions.
    description : str
        Description of the dimension/variable.
    min : float
        Minimum value of a numerical dimension/variable.
    max : float
        Maximum value of a numerical dimension/variable.
    parts : list of `CoclusteringDimensionPart`
        Partition of this dimension.
    variable_part_dimensions : list of `CoclusteringDimension`
        Variable part instance-variable coclustering dimensions. ``None`` for
        variable-variable clustering.
    clusters : list of `CoclusteringCluster`
        Clusters of this dimension's hierarchy. Note that includes intermediary
        clusters.
    root_cluster : `CoclusteringCluster`
        Root cluster of the hierarchy.
    """

    def __init__(self):
        """See class docstring"""
        # Summary attributes
        self.name = ""
        self.is_variable_part = False
        self.type = ""
        self.part_number = 0
        self.initial_part_number = 0
        self.value_number = 0
        self.interest = 0
        self.description = ""

        # Min & Max attributes: Numerical dimension only
        self.min = None
        self.max = None

        # List of parts: intervals or value groups according to dimension type
        self.parts = []

        # Default group attribute only for categorical dimensions
        self.default_group = None

        # List of the hierarchy clusters, ranging from the root to the leaves
        self.root_cluster = None
        self.clusters = []

        # Parts internal dictionary
        self._parts_by_name = {}

        # Clusters internal dictionary
        self._clusters_by_name = {}

        # Variable part dimensions
        self.variable_part_dimensions = None

    def init_summary(self, json_data=None):
        """Initializes the summary attributes from a Python JSON object

        Parameters
        ----------
        json_data : dict, optional
            Dictionary representing the data of an element of the list found at the
            ``dimensionSummaries`` field of a Khiops Coclustering JSON report file. If
            not specified it leaves the object as-is.

        Returns
        -------
        self
            A reference to the caller instance.
        """
        # Do nothing if json_data is not specified
        if json_data is None:
            return self
        # Otherwise check the type of json_data
        elif not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Initialize the summary fields
        self.name = json_data.get("name", "")
        self.is_variable_part = json_data.get("isVarPart", False)
        self.type = json_data.get("type", "")
        self.part_number = json_data.get("parts", 0)
        self.initial_part_number = json_data.get("initialParts", 0)
        self.value_number = json_data.get("values", 0)
        self.interest = json_data.get("interest", 0)
        self.description = json_data.get("description", "")
        self.min = json_data.get("min")
        self.max = json_data.get("max")

        return self

    def init_partition(self, json_data=None):
        """Initializes the partition attributes from a Python JSON object

        Parameters
        ----------
        json_data : dict, optional
            Python dictionary representing the data of an element of the list found at
            the ``dimensionPartitions`` field of a Khiops Coclustering JSON report file.
            If not specified it leaves the object as-is.

        Returns
        -------
        self
            A reference to the caller instance.
        """
        # Do nothing if json_data is not specified
        if json_data is None:
            return self
        # Otherwise check the type of json_data
        elif not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Check minimum required attributes
        if self.name != json_data.get("name"):
            raise KhiopsJSONError(
                f"""'name' field value is '{json_data.get("name")}' """
                f"it should be '{self.name}'"
            )
        if self.type != json_data.get("type"):
            raise KhiopsJSONError(
                f"""'name' field value is '{json_data.get("type")}' """
                f"it should be '{self.type}'"
            )

        # Initialize interval partitions
        if self.type == "Numerical":
            # Check the "intervals" field
            if "intervals" not in json_data:
                raise KhiopsJSONError(
                    "'intervals' key not found in numerical partition"
                )
            elif not isinstance(json_data["intervals"], list):
                raise KhiopsJSONError("'intervals' key must be a list")
            elif len(json_data["intervals"]) < 1:
                raise KhiopsJSONError("'intervals' key must have length at least 1")

            # Initialize the intervals data
            for json_interval in json_data["intervals"]:
                interval = CoclusteringDimensionPartInterval(json_interval)
                self.parts.append(interval)
                self._parts_by_name[interval.cluster_name] = interval

            # Initialize open interval flags
            first_interval = self.parts[0]
            if first_interval.is_missing:
                if len(json_data["intervals"]) < 2:
                    raise KhiopsJSONError(
                        "'intervals' key must have at least 2 elements "
                        "when one element contains missing values."
                    )
                first_interval = self.parts[1]
            first_interval.is_left_open = True
            last_interval = self.parts[-1]
            last_interval.is_right_open = True

        # Initialize value groups partitions
        elif self.type == "Categorical":
            # Initialize regular value groups
            if "valueGroups" not in json_data:
                raise KhiopsJSONError("'valueGroups' key not found")
            for json_value_group in json_data["valueGroups"]:
                value_group = CoclusteringDimensionPartValueGroup(json_value_group)
                self.parts.append(value_group)
                self._parts_by_name[value_group.cluster_name] = value_group

            # Initialize default group
            # The default group contains all the values not specified in
            # any part of the partition and all the unknown values
            default_group_index = json_data.get("defaultGroupIndex")
            self.default_group = self.parts[default_group_index]
            self.default_group.is_default_part = True

            # Instance-variable coclustering: initialize inner variables
            if self.is_variable_part:

                # Create inner variables dimensions (subpartition)
                if "innerVariables" not in json_data:
                    raise KhiopsJSONError("'innerVariables' key not found")
                self.variable_part_dimensions = []
                json_inner_variables = json_data["innerVariables"]
                if "dimensionSummaries" in json_inner_variables:
                    for json_dimension_summary in json_inner_variables[
                        "dimensionSummaries"
                    ]:
                        dimension = CoclusteringDimension().init_summary(
                            json_dimension_summary
                        )
                        self.variable_part_dimensions.append(dimension)

                # Initialize inner variables dimensions' partitions
                if "dimensionPartitions" in json_inner_variables:
                    json_dimension_partitions = json_inner_variables[
                        "dimensionPartitions"
                    ]
                    if len(self.variable_part_dimensions) != len(
                        json_dimension_partitions
                    ):
                        raise KhiopsJSONError(
                            "'ineerVariables/dimensionPartitions' list has length "
                            f"{len(json_dimension_partitions)} instead of "
                            f"{len(self.variable_part_dimensions)}"
                        )
                    for i, json_dimension_partition in enumerate(
                        json_dimension_partitions
                    ):
                        dimension = self.variable_part_dimensions[i]
                        dimension.init_partition(json_dimension_partition)

        return self

    def init_hierarchy(self, json_data):
        """Initializes the hierarchy attributes from a Python JSON object

        Parameters
        ----------
        json_data : dict, optional
            Python dictionary representing the data of an element of the list found at
            the ``dimensionHierarchies`` field of a Khiops Coclustering JSON report
            file. If not specified it leaves the object as-is.

        Returns
        -------
        self
            A reference to the caller instance.
        """
        # Do nothing if json_data is not specified
        if json_data is None:
            return self
        # Otherwise check the type of json_data
        elif not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Check minimum required attributes
        if self.name != json_data.get("name"):
            raise KhiopsJSONError(
                f"""'name' field is '{json_data.get("name")}' """
                f"it should be '{self.name}'"
            )
        if self.type != json_data.get("type"):
            raise KhiopsJSONError(
                f"""'name' field is '{json_data.get("type")}' """
                f"it should be '{self.type}'"
            )

        # Initialize clusters
        if "clusters" not in json_data:
            raise KhiopsJSONError("'clusters' key not found")
        for json_cluster in json_data["clusters"]:
            cluster = CoclusteringCluster(json_cluster)
            self.clusters.append(cluster)
            self._clusters_by_name[cluster.name] = cluster

        # Link the clusters according to their hierarchy
        # and link the leaf cluster to their parts
        for cluster in self.clusters:
            # Link leaf cluster to its part
            if cluster.is_leaf:
                cluster.leaf_part = self.get_part(cluster.name)

            # Root cluster case: Set the dimension's root cluster reference
            if cluster.parent_cluster_name == "":
                self.root_cluster = cluster
            # Non root case: Link the parent to the current cluster
            else:
                # Lookup the parent
                cluster.parent_cluster = self.get_cluster(cluster.parent_cluster_name)

                # If parent's first child empty: Set current cluster to it
                if cluster.parent_cluster.child_cluster1 is None:
                    cluster.parent_cluster.child_cluster1 = cluster
                # If not: Set current cluster to the second child
                else:
                    cluster.parent_cluster.child_cluster2 = cluster
        return self

    def get_part(self, part_name):
        """Returns a part of the dimension given the part's name

        Parameters
        ----------
        part_name : str
            Name of the part.

        Returns
        -------
        `CoclusteringDimensionPart`
            The part with the specified name.

        Raises
        ------
        `KeyError`
            If there is no part with the specified name.
        """
        return self._parts_by_name[part_name]

    def get_cluster(self, cluster_name):
        """Returns the specified cluster

        Parameters
        ----------
        cluster_name : str
            Name of the cluster.

        Returns
        -------
        `CoclusteringCluster`
            The specified cluster.

        Raises
        ------
        `KeyError`
            If there is no cluster with the specified name.
        """
        return self._clusters_by_name[cluster_name]

    def to_dict(self, report_type):
        """Transforms this instance to a dict with the Khiops JSON file structure

        Parameters
        ----------
        report_type : str
            Type of the report. Can be either one of "summary", "dimension", and
            "hierarchy".
        """
        if report_type == "summary":
            report = {
                "name": self.name,
                "type": self.type,
                "parts": self.part_number,
                "initialParts": self.initial_part_number,
                "values": self.value_number,
                "interest": self.interest,
                "description": self.description,
            }
            if self.type == "Numerical":
                report.update({"min": self.min, "max": self.max})
            if self.is_variable_part:
                report["isVarPart"] = self.is_variable_part
            return report
        elif report_type == "partition":
            report = {
                "name": self.name,
                "type": self.type,
            }
            if self.type == "Numerical":
                report["intervals"] = [part.to_dict() for part in self.parts]
            elif self.type == "Categorical":
                report["valueGroups"] = [part.to_dict() for part in self.parts]

                # Get default group index
                for i, part in enumerate(self.parts):
                    if part.is_default_part:
                        default_group_index = i
                        break
                report["defaultGroupIndex"] = default_group_index

                # Inner variables dimensions for instance-variable coclustering
                if self.is_variable_part:
                    report["innerVariables"] = {
                        "dimensionSummaries": [
                            dimension.to_dict(report_type="summary")
                            for dimension in self.variable_part_dimensions
                        ],
                        "dimensionPartitions": [
                            dimension.to_dict(report_type="partition")
                            for dimension in self.variable_part_dimensions
                        ],
                    }
            return report
        elif report_type == "hierarchy":
            report = {
                "name": self.name,
                "type": self.type,
                "clusters": [cluster.to_dict() for cluster in self.clusters],
            }
            return report
        else:
            raise ValueError(f"Unknown 'report_type 'value: '{report_type}'")

    def write_dimension_header_line(self, writer):  # pragma: no cover
        """Writes the "dimensions" section header to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(
            deprecation_message("write_dimension_header_line", "12.0.0", "to_dict")
        )

        # Write dimension report header
        writer.write("Name\t")
        writer.write("Is variable part\t")
        writer.write("Type\t")
        writer.write("Parts\t")
        writer.write("Initial parts\t")
        writer.write("Values\t")
        writer.write("Interest\t")
        writer.writeln("Description")

    def write_dimension_line(self, writer):  # pragma: no cover
        """Writes the "dimensions" section line to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_dimension_line", "12.0.0", "to_dict"))

        # Write dimension report line
        writer.write(f"{self.name}\t")
        writer.write(f"{self.is_variable_part}\t")
        writer.write(f"{self.type}\t")
        writer.write(f"{self.part_number}\t")
        writer.write(f"{self.initial_part_number}\t")
        writer.write(f"{self.value_number}\t")
        writer.write(f"{self.interest}\t")
        writer.writeln(self.description)

    def write_hierarchy(self, writer):  # pragma: no cover
        """Writes the "hierarchy" section to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_hierarchy", "12.0.0", "to_dict"))

        # Write header
        writer.writeln("")
        writer.writeln(f"Hierarchy\t{self.name}")

        # Write the hierarchy for each cluster
        for i, cluster in enumerate(self.clusters):
            if i == 0:
                cluster.write_hierarchy_header_line(writer)
            cluster.write_hierarchy_line(writer)

    def write_composition(self, writer):  # pragma: no cover
        """Writes the "composition" section to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_composition", "12.0.0", "to_dict"))

        # Write only categorical dimensions
        if self.type == "Categorical":
            # Write header
            writer.writeln("")
            writer.writeln(f"Composition\t{self.name}")
            writer.write("Cluster\t")
            writer.write("Value\t")
            writer.write("Frequency\t")
            writer.writeln("Typicality")

            # Write value groups
            for value_group in self.parts:
                # Write value's information
                for value in value_group.values:
                    writer.write(f"{value_group.cluster_name}\t")
                    writer.write(f"{value.value}\t")
                    writer.write(f"{value.frequency}\t")
                    writer.writeln(str(value.typicality))

                # Write special value for the default part
                if value_group.is_default_part:
                    writer.write(f"{value_group.cluster_name}\t")
                    writer.write(" * \t")
                    writer.write(f"{0}\t")
                    writer.writeln(str(0))

    def write_annotation(self, writer):  # pragma: no cover
        """Writes the "annotation" section to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_annotation", "12.0.0", "to_dict"))

        # Write header
        writer.writeln("")
        writer.writeln(f"Annotation\t{self.name}")

        # Write annotations for each cluster
        for i, cluster in enumerate(self.clusters):
            if i == 0:
                cluster.write_annotation_header_line(writer)
            cluster.write_annotation_line(writer)

    def needs_annotation_report(self):
        """Status about the annotation report

        Returns
        -------
        bool
            True if the "annotation" section is reported
        """
        # Check if any cluster has a non empty description
        for cluster in self.clusters:
            if cluster.short_description != "" or cluster.description != "":
                return True
        return False

    def write_hierarchy_structure_report_file(
        self, report_file_path
    ):  # pragma: no cover
        """Writes the hierarchical structure of the clusters to a file

        This method is mainly a test of the encoding of the cluster hierarchy.

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        report_file_path : str
            Path of the output file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(
            deprecation_message(
                "write_hierarchy_structure_report_file", "12.0.0", "to_dict"
            )
        )

        # Write report to file
        with open(report_file_path, "wb") as report_file:
            writer = KhiopsOutputWriter(report_file)
            writer.writeln(f"Hierarchical structure\t{self.name}")
            self.root_cluster.write_hierarchy_structure_report(writer)


class CoclusteringDimensionPart:
    """An element of a partition of a dimension

    Abstract class.

    Parameters
    ----------
    json_data: dict, optional
        See child classes for specific information about this parameter.

    Attributes
    ----------
    cluster_name : str
        Name of the cluster to which this part belongs.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise check the validity of json_data
        elif "cluster" not in json_data:
            raise KhiopsJSONError("'cluster' key not found")

        # Cluster name of the part (only referenced by a leaf cluster of a hierarchy)
        self.cluster_name = json_data.get("cluster", "")


class CoclusteringDimensionPartInterval(CoclusteringDimensionPart):
    """An interval of a numerical partition

    Parameters
    ----------
    json_data : dict, optional
        Python dictionary representing an element of type "Numerical" of the list at the
        ``dimensionPartitions`` field of a Khiops Coclustering JSON report file. If not
        specified it returns an empty instance.

    Raises
    ------
    `.KhiopsJSONError`
        If ``json_data`` does not contain a "cluster" key.


    Attributes
    ----------
    cluster_name : str
        Name of the cluster containing this interval.
    lower_bound : float
        Lower bound of the interval.
    upper_bound : float
        Upper bound of the interval.
    is_missing : bool
        True if the instance's represent the missing values. In this case
        ``lower_bound`` and ``upper_bound`` are set to ``None``.
    is_left_open : bool
        True if the interval is unbounded below ``lower_bound`` may contain the minimum
        value of the training data.
    is_right_open : bool
        True if the interval is unbounded above ``upper_bound`` may contain the maximum
        value of training data.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Initialize base class
        super().__init__(json_data=json_data)

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise check that the 'bounds' key exists
        else:
            if "bounds" not in json_data:
                raise KhiopsJSONError("'bounds' key not found")
            elif not isinstance(json_data["bounds"], list):
                raise KhiopsJSONError("'bounds' key must be a list")
            elif len(json_data["bounds"]) != 2:
                raise KhiopsJSONError("'bounds' key must be a list of length 2")

        # Initialize a "true" interval
        json_bounds = json_data.get("bounds")
        if json_bounds:
            self.is_missing = False
            self.lower_bound = json_bounds[0]
            self.upper_bound = json_bounds[1]
        # Initialize a "missing part" interval
        else:
            self.is_missing = True
            self.lower_bound = None
            self.upper_bound = None

        # Initialize the 'open' flags (initialized by another class)
        self.is_left_open = False
        self.is_right_open = False

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report = {
            "cluster": self.cluster_name,
        }
        if not self.is_missing:
            report["bounds"] = [self.lower_bound, self.upper_bound]
        return report

    def __str__(self):
        """Returns a human-readable string representation"""
        if self.is_missing:
            label = "Missing"
        else:
            if self.is_left_open:
                label = "]-inf"
            else:
                label = f"]{self.lower_bound}"
            if self.is_right_open:
                label += ";+inf["
            else:
                label += f";{self.upper_bound}]"
        return label

    def part_type(self):
        """Part type of this instance

        Returns
        -------
        str
            Only possible value: "Interval".
        """
        return "Interval"


class CoclusteringDimensionPartValueGroup(CoclusteringDimensionPart):
    """A value group of a categorical partition

    Parameters
    ----------
    json_data : dict, optional
        Python dictionary representing an element of type "Categorical" of the list at
        the ``dimensionPartitions`` field of a Khiops Coclustering JSON report file. If
        None it returns an empty instance.

    Raises
    ------
    `.KhiopsJSONError`
        If ``json_data`` does not contain a "cluster" key.

    Attributes
    ----------
    cluster_name : str
        Name of the cluster containing this group.
    values : list of `CoclusteringDimensionPartValue`
        The singleton parts composing this group part.
    is_default_part : bool
        True if the instance represents the "unknown values" group.
    """

    def __init__(self, json_data=None):
        """Constructs an instance from a python JSON object"""
        # Initialize base class
        super().__init__(json_data=json_data)

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise raise an error if the relevant keys are not found
        else:
            # Value typicalities are absent for variable parts dimensions in
            # instance-variable coclustering
            mandatory_keys = ("values", "valueFrequencies")
            for key in mandatory_keys:
                if key not in json_data:
                    raise KhiopsJSONError(f"'{key}' key not found")
                elif not isinstance(json_data[key], list):
                    raise KhiopsJSONError(f"'{key}' key must be a list")
                elif len(json_data[key]) != len(json_data["values"]):
                    raise KhiopsJSONError(
                        f"'{key}' key list must have the same length as 'values'"
                    )

        # Initialize value list
        self.values = []
        json_values = json_data.get("values", [])
        json_value_frequencies = json_data.get("valueFrequencies", [])
        json_value_typicalities = json_data.get("valueTypicalities", [])
        for i, json_value in enumerate(json_values):
            value = CoclusteringDimensionPartValue()
            self.values.append(value)
            value.value = json_value
            value.frequency = json_value_frequencies[i]

            # valueTypicalities are absent for variable part dimension parts,
            # as used in instance-variable coclustering
            value.typicality = (
                json_value_typicalities[i] if i < len(json_value_typicalities) else None
            )

        # Initialize default values (set for real from another class)
        self.is_default_part = False

    def __str__(self):
        """Returns a human-readable string representation"""
        label = "{"
        for i, value in enumerate(self.values):
            if i > 0:
                label += ", "
            if i == 3:
                label += "..."
                break
            label += value.value
        label += "}"
        return label

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report = {
            "cluster": self.cluster_name,
            "values": [value.value for value in self.values],
            "valueFrequencies": [value.frequency for value in self.values],
        }
        typicalities = [
            value.typicality for value in self.values if value.typicality is not None
        ]
        if typicalities:
            report["valueTypicalities"] = typicalities
        return report

    def part_type(self):
        """Part type of this instance

        Returns
        -------
        str
            Only possible value: "Value group".
        """
        return "Value group"


class CoclusteringDimensionPartValue:
    """A specific value of a variable in a dimension value group.

    .. note::
        This class has only a no-parameter constructor initializing an instance with the
        default values.

    Attributes
    ----------
    value : str
        String representation of the value.
    frequency : int
        Number of individuals having this value.
    typicality : float
        Indicates how much the value is representative of the cluster. Ranges from 0 to
        1, 1 being completely representative.
    """

    def __init__(self):
        """See class doctstring"""
        self.value = ""
        self.frequency = 0
        self.typicality = 0


class CoclusteringCluster:
    """A cluster in a coclustering dimension hierarchy

    Parameters
    ----------
    json_data : dict, optional
        JSON data of an element of the list at the ``dimensionHierarchies`` field within
        the ``coclusteringReport`` field of a Khiops Coclustering JSON report file. If
        not specified it returns an empty instance.

    Attributes
    ----------
    name : str
        Name of the cluster.
    parent_cluster_name : str
        Name of the parent cluster.
    frequency : int
        Number of individuals in the cluster.
    interest : float
        The cluster's interest/informativeness.
    hierarchical_level : float
        A measure interpretable as the distance of the cluster to the root. Between 0
        and 1.
    rank : int
        Rank of clusters in the top-down list of clusters, with the smallest ranks at
        the top.
    hierarchical_rank : int
        Rank of clusters in the hierarchy, with the smallest ranks being the closest
        from the root of the hierarchy.
    is_leaf : bool
        ``True`` if the cluster is a leaf of the hierarchy.
    short_description : str
        Succinct cluster description.
    description : str
        Cluster description.
    leaf_part : `CoclusteringDimensionPart`
        On a leaf cluster: Its unique associated partition element. Otherwise ``None``.
    parent_cluster : `CoclusteringCluster`
        On a non-root cluster: Its unique parent cluster. Otherwise ``None``.
    child_cluster1 : `CoclusteringCluster`
        On a non-leaf cluster : The first child cluster. Otherwise ``None``.
    child_cluster2 : `CoclusteringCluster`
        On a non-leaf cluster : The second child cluster. Otherwise ``None``.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise check the "cluster" and "parentCluster" keys
        else:
            if "cluster" not in json_data:
                raise KhiopsJSONError("'cluster' key not found")
            elif "parentCluster" not in json_data:
                raise KhiopsJSONError("'parentCluster' key not found")

        # Initialize attributes
        self.name = json_data.get("cluster", "")
        self.parent_cluster_name = json_data.get("parentCluster", "")
        self.frequency = json_data.get("frequency", 0)
        self.interest = json_data.get("interest", 0.0)
        self.hierarchical_level = json_data.get("hierarchicalLevel", 0.0)
        self.rank = json_data.get("rank", 0)
        self.hierarchical_rank = json_data.get("hierarchicalRank", 0)
        self.is_leaf = json_data.get("isLeaf", False)
        self.short_description = json_data.get("shortDescription")
        self.description = json_data.get("description")

        # Link to child clusters, None for the leaves of the hierarchy
        # The user must specify the CoclusteringCluster references parent_cluster
        # and child_cluster that link this instance to the hierarchy
        self.parent_cluster = None
        self.child_cluster1 = None
        self.child_cluster2 = None
        self.leaf_part = None

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report = {
            "cluster": self.name,
            "parentCluster": self.parent_cluster_name,
            "frequency": self.frequency,
            "interest": self.interest,
            "hierarchicalLevel": self.hierarchical_level,
            "rank": self.rank,
            "hierarchicalRank": self.hierarchical_rank,
            "isLeaf": self.is_leaf,
        }
        if self.short_description is not None:
            report["shortDescription"] = self.short_description
        if self.description is not None:
            report["description"] = self.description
        return report

    def write_hierarchy_header_line(self, writer):  # pragma: no cover
        """Writes the "hierarchy" section's header to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(
            deprecation_message("write_hierarchy_header_line", "12.0.0", "to_dict")
        )

        # Write report header
        writer.write("Cluster\t")
        writer.write("ParentCluster\t")
        writer.write("Frequency\t")
        writer.write("Interest\t")
        writer.write("HierarchicalLevel\t")
        writer.write("Rank\t")
        writer.writeln("HierarchicalRank")

    def write_hierarchy_line(self, writer):  # pragma: no cover
        """Writes a line of the "hierarchy" section to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_hierarchy_line", "12.0.0", "to_dict"))

        # Write hierarchy report line
        writer.write(f"{self.name}\t")
        writer.write(f"{self.parent_cluster_name}\t")
        writer.write(f"{self.frequency}\t")
        writer.write(f"{self.interest}\t")
        writer.write(f"{self.hierarchical_level}\t")
        writer.write(f"{self.rank}\t")
        writer.writeln(str(self.hierarchical_rank))

    def write_annotation_header_line(self, writer):  # pragma: no cover
        """Writes the "annotation" section's header to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(
            deprecation_message("write_annotation_header_line", "12.0.0", "to_dict")
        )

        # Write annotation report header
        writer.write("Cluster\t")
        writer.write("Expand\t")
        writer.write("Selected\t")
        writer.write("ShortDescription\t")
        writer.writeln("Description")

    def write_annotation_line(self, writer):  # pragma: no cover
        """Writes a line of the "annotation" section to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_annotation_line", "12.0.0", "to_dict"))

        # Write annotation report line
        writer.write(f"{self.name}\t")
        # TODO: Why "Expand" and "Selected" are not available?
        writer.write("FALSE\t")
        writer.write("FALSE\t")
        writer.write(f"{self.short_description}\t")
        writer.writeln(self.description)

    def write_hierarchy_structure_report(self, writer):  # pragma: no cover
        """Writes the hierarchical structure from this instance to a writer object

        This method is mainly a test of the encoding of the cluster hierarchy.

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer for the report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(
            deprecation_message("write_hierarchy_structure_report", "12.0.0", "to_dict")
        )

        # Write first child cluster
        if self.child_cluster1 is not None:
            self.child_cluster1.write_hierarchy_structure_report(writer)

        # Write current cluster
        writer.write(" " * self.hierarchical_rank)
        writer.write(self.name)
        writer.writeln("")

        # Write second child cluster
        if self.child_cluster2 is not None:
            self.child_cluster2.write_hierarchy_structure_report(writer)


class CoclusteringCell:
    """A coclustering cell

    .. note::
        This class has only a no-parameter constructor initializing an instance with the
        default values.

    Attributes
    ----------
    parts : list of `CoclusteringDimensionPart`
        Parts for each coclustering dimension.
    frequency : int
        Frequency of this cell.
    """

    def __init__(self):
        """Constructs an instance with default attribute values"""
        self.parts = []
        self.frequency = 0

    def write_line(self, writer):  # pragma: no cover
        """Writes a line of the instance's report to a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_line", "12.0.0"))

        # Write part report line
        for part in self.parts:
            writer.write(f"{part.cluster_name}\t")
        writer.writeln(str(self.frequency))
