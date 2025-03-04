######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tests for the khiops.core module"""
import glob
import io
import os
import shutil
import textwrap
import unittest
import warnings
from pathlib import Path
from unittest import mock

import khiops
import khiops.core as kh
from khiops.core import KhiopsRuntimeError
from khiops.core.internals.common import create_unambiguous_khiops_path
from khiops.core.internals.io import KhiopsOutputWriter
from khiops.core.internals.runner import KhiopsLocalRunner, KhiopsRunner
from khiops.core.internals.scenario import ConfigurableKhiopsScenario
from khiops.core.internals.version import KhiopsVersion
from tests.test_helper import KhiopsTestHelper

# Disable warning about access to protected member: These are tests
# pylint: disable=protected-access


class KhiopsCoreIOTests(unittest.TestCase):
    """Tests the reading/writing of files for the core module classes/functions"""

    def test_analysis_results(self):
        """Tests for the analysis_results module"""
        # Set the test paths
        test_resources_dir = os.path.join(resources_dir(), "analysis_results")
        ref_reports_dir = os.path.join(test_resources_dir, "ref_reports")
        ref_json_reports_dir = os.path.join(test_resources_dir, "ref_json_reports")
        output_reports_dir = os.path.join(test_resources_dir, "output_reports")

        # Cleanup previous output files
        cleanup_dir(output_reports_dir, "*.txt")

        # Read the json reports, dump them as txt reports, and compare to the reference
        reports = [
            "Adult",
            "AdultEvaluation",
            "AdultLegacy",
            "Ansi",
            "AnsiGreek",
            "AnsiLatin",
            "AnsiLatinGreek",
            "AnyChar",
            "AnyCharLegacy",
            "BadTool",
            "Deft2017ChallengeNGrams1000",
            "EmptyDatabase",
            "Greek",
            "Iris2D",
            "Iris2DLegacy",
            "IrisC",
            "IrisG",
            "IrisMAPLegacy",
            "IrisR",
            "IrisU",
            "IrisU2D",
            "LargeSpiral",
            "Latin",
            "LatinGreek",
            "MissingDiscretization",
            "MissingMODLEqualWidth",
            "NoBivariateDetailedStats",
            "NoPredictorDetails",
            "NoVersion",
            "XORRegression",
        ]
        reports_warn = [
            "AdultLegacy",
            "AnsiLatin",
            "AnsiLatinGreek",
            "AnyCharLegacy",
            "Iris2DLegacy",
            "IrisMAPLegacy",
        ]
        reports_ko = ["BadTool", "NoVersion"]
        for report in reports:
            ref_report = os.path.join(ref_reports_dir, f"{report}.txt")
            ref_json_report = os.path.join(ref_json_reports_dir, f"{report}.khj")
            output_report = os.path.join(output_reports_dir, f"{report}.txt")
            with self.subTest(report=report):
                if report in reports_ko:
                    with self.assertRaises(kh.KhiopsJSONError):
                        results = kh.read_analysis_results_file(ref_json_report)
                elif report in reports_warn:
                    with self.assertWarns(UserWarning):
                        results = kh.read_analysis_results_file(ref_json_report)
                        results.write_report_file(output_report)
                        assert_files_equal(self, ref_report, output_report)
                else:
                    results = kh.read_analysis_results_file(ref_json_report)
                    results.write_report_file(output_report)
                    assert_files_equal(self, ref_report, output_report)

    def test_coclustering_results(self):
        """Tests for the coclustering_results module"""
        # Set the test paths
        test_resources_dir = os.path.join(resources_dir(), "coclustering_results")
        ref_reports_dir = os.path.join(test_resources_dir, "ref_reports")
        ref_json_reports_dir = os.path.join(test_resources_dir, "ref_json_reports")
        output_reports_dir = os.path.join(test_resources_dir, "output_reports")

        # Cleanup output files
        cleanup_dir(output_reports_dir, "*.txt")

        # Read then json reports, dump them as txt reports and compare to the reference
        reports = [
            "Adult",
            "AdultLegacy",
            "Iris",
            "Ansi_Coclustering",
            "AnsiGreek_Coclustering",
            "AnsiLatin_Coclustering",
            "AnsiLatinGreek_Coclustering",
            "Greek_Coclustering",
            "Latin_Coclustering",
            "LatinGreek_Coclustering",
            "MushroomAnnotated",
        ]
        reports_warn = [
            "AdultLegacy",
            "AnsiLatin_Coclustering",
            "AnsiLatinGreek_Coclustering",
        ]
        for report in reports:
            ref_report = os.path.join(ref_reports_dir, f"{report}.khc")
            ref_json_report = os.path.join(ref_json_reports_dir, f"{report}.khcj")
            output_report = os.path.join(output_reports_dir, f"{report}.khc")
            with self.subTest(report=report):
                if report in reports_warn:
                    with self.assertWarns(UserWarning):
                        results = kh.read_coclustering_results_file(ref_json_report)
                else:
                    results = kh.read_coclustering_results_file(ref_json_report)
                results.write_report_file(output_report)
                assert_files_equal(self, ref_report, output_report)
                for dimension in results.coclustering_report.dimensions:
                    ref_hierarchy_report = os.path.join(
                        ref_reports_dir, f"{report}_hierarchy_{dimension.name}.txt"
                    )
                    output_hierarchy_report = os.path.join(
                        output_reports_dir, f"{report}_hierarchy_{dimension.name}.txt"
                    )
                    dimension.write_hierarchy_structure_report_file(
                        output_hierarchy_report
                    )
                    assert_files_equal(
                        self, ref_hierarchy_report, output_hierarchy_report
                    )

    def test_binary_dictionary_domain(self):
        """Test binary dictionary write"""
        # Set the test paths
        test_resources_dir = os.path.join(resources_dir(), "dictionary")
        dictionary_name = "Bytes"
        ref_kdic = os.path.join(
            test_resources_dir, "ref_kdic", f"{dictionary_name}.kdic"
        )
        ref_kdicj = os.path.join(
            test_resources_dir, "ref_kdicj", f"{dictionary_name}.kdicj"
        )
        output_kdic_dir = os.path.join(test_resources_dir, "output_kdic")
        output_kdic = os.path.join(output_kdic_dir, f"{dictionary_name}.kdic")
        copy_output_kdic_dir = os.path.join(test_resources_dir, "copy_output_kdic")
        copy_output_kdic = os.path.join(copy_output_kdic_dir, f"{dictionary_name}.kdic")

        # Create output dirs if not existing, and delete their contents
        cleanup_dir(output_kdic_dir, "*.kdic")
        cleanup_dir(copy_output_kdic_dir, "*.kdic")

        # Build dictionary domain programmatically
        domain_from_api = kh.DictionaryDomain()
        domain_from_api.version = b"10.0.0.3i"
        dictionary = kh.Dictionary()
        dictionary.name = bytes("MyDictê", encoding="cp1252")
        metadata = kh.MetaData()
        metadata.add_value(
            bytes("aKey", encoding="cp1252"), bytes("aValué", encoding="cp1252")
        )
        variable = kh.Variable()
        variable.name = bytes("MyVarî", encoding="cp1252")
        variable.type = "Categorical"
        variable.meta_data = metadata
        dictionary.add_variable(variable)
        domain_from_api.add_dictionary(dictionary)

        # Read domain from JSON file
        domain_from_json = kh.read_dictionary_file(ref_kdicj)

        for domain in (domain_from_api, domain_from_json):
            # Dump domain object as kdic file and compare it to the reference
            domain.export_khiops_dictionary_file(output_kdic)
            assert_files_equal(self, ref_kdic, output_kdic)

            # Make a copy of the domain object, then dump it as kdic file and
            # compare it to the reference
            domain_copy = domain.copy()
            domain_copy.export_khiops_dictionary_file(copy_output_kdic)
            assert_files_equal(self, ref_kdic, copy_output_kdic)

    def test_dictionary(self):
        """Tests for the dictionary module"""
        # Set the test paths
        test_resources_dir = os.path.join(resources_dir(), "dictionary")
        ref_kdic_dir = os.path.join(test_resources_dir, "ref_kdic")
        ref_kdicj_dir = os.path.join(test_resources_dir, "ref_kdicj")
        output_kdic_dir = os.path.join(test_resources_dir, "output_kdic")
        copy_output_kdic_dir = os.path.join(test_resources_dir, "copy_output_kdic")

        # Cleanup previous output files
        cleanup_dir(output_kdic_dir, "*.kdic")
        cleanup_dir(copy_output_kdic_dir, "*.kdic")

        # Read then json reports then:
        # - dump the domain objects as kdic files and compare to the reference
        # - make a copy of the domain objects and then dump and compare to the reference
        dictionaries = [
            "AIDSBondCounts",
            "Adult",
            "AdultKey",
            "AdultLegacy",
            "AdultModeling",
            "Ansi",
            "AnsiGreek",
            "AnsiGreek_Modeling",
            "AnsiLatin",
            "AnsiLatinGreek",
            "AnsiLatinGreek_Modeling",
            "AnsiLatin_Modeling",
            "Ansi_Modeling",
            "Customer",
            "CustomerExtended",
            "D_Modeling",
            "Dorothea",
            "Greek",
            "Greek_Modeling",
            "Latin",
            "LatinGreek",
            "LatinGreek_Modeling",
            "Latin_Modeling",
            "SpliceJunction",
            "SpliceJunctionModeling",
        ]

        dictionaries_warn = [
            "AdultLegacy",
            "AnsiLatin",
            "AnsiLatinGreek",
            "AnsiLatinGreek_Modeling",
            "AnsiLatin_Modeling",
        ]

        for dictionary in dictionaries:
            ref_kdic = os.path.join(ref_kdic_dir, f"{dictionary}.kdic")
            ref_kdicj = os.path.join(ref_kdicj_dir, f"{dictionary}.kdicj")
            output_kdic = os.path.join(output_kdic_dir, f"{dictionary}.kdic")
            copy_output_kdic = os.path.join(copy_output_kdic_dir, f"{dictionary}.kdic")
            with self.subTest(dictionary=dictionary):
                if dictionary in dictionaries_warn:
                    with self.assertWarns(UserWarning):
                        domain = kh.read_dictionary_file(ref_kdicj)
                else:
                    domain = kh.read_dictionary_file(ref_kdicj)
                domain.export_khiops_dictionary_file(output_kdic)
                assert_files_equal(self, ref_kdic, output_kdic)

                domain_copy = domain.copy()
                domain_copy.export_khiops_dictionary_file(copy_output_kdic)
                assert_files_equal(self, ref_kdic, copy_output_kdic)

    def _build_mock_api_method_parameters(self):
        # Pseudo-mock data to test the creation of scenarios
        datasets = ["Adult", "SpliceJunction", "Customer"]
        additional_data_tables = {
            "Adult": None,
            "SpliceJunction": {"DNA": "SpliceJunctionDNABidon.csv"},
            "Customer": {
                "Services": "ServicesBidon.csv",
                "Services/Usages": "UsagesBidon.csv",
                "Address": "AddressBidon.csv",
                "/City": "CityBidon.csv",
                "/Country": "CountryBidon.csv",
                "/Product": "ProductBidon.csv",
            },
        }
        output_additional_data_tables = {
            "Adult": None,
            "SpliceJunction": {"DNA": "TransferSpliceJunctionDNABidon.csv"},
            "Customer": {
                "Services": "TransferServicesBidon.csv",
                "Services/Usages": "TransferUsagesBidon.csv",
                "Address": "TransferAddressBidon.csv",
                "/City": "TransferCityBidon.csv",
                "/Country": "TransferCountryBidon.csv",
                "/Product": "TransferProductBidon.csv",
            },
        }
        target_variables = {"Adult": "class", "SpliceJunction": "Class", "Customer": ""}
        construction_rules = {
            "Adult": [],
            "SpliceJunction": ["TableMode", "TableSelection"],
            "Customer": None,
        }
        coclustering_variables = {
            "Adult": ["age", "workclass", "race", "sex"],
            "SpliceJunction": ["SampleId", "NonExistentVar"],
            "Customer": ["id_customer", "Name"],
        }
        max_part_numbers = {
            "Adult": {"age": 2, "workclass": 4, "race": 8, "sex": 16},
            "SpliceJunction": {"SampleId": 32, "NonExistentVar": 64},
            "Customer": None,
        }
        sort_variables = {
            "Adult": ["Label", "age", "race"],
            "SpliceJunction": ["SampleId"],
            "Customer": None,
        }
        specific_pairs = {
            "Adult": [("age", "rage"), ("Label", ""), ("", "capital_gain")],
            "SpliceJunction": [],
            "Customer": None,
        }

        detect_data_table_format_kwargs = {
            "Adult": {
                "dictionary_file_path_or_domain": "Adult.kdic",
                "dictionary_name": "Adult",
            },
            "SpliceJunction": {
                "dictionary_file_path_or_domain": "SpliceJunctionDNA.kdic",
                "dictionary_name": "SpliceJunctionDNA",
            },
            "Customer": {
                "dictionary_file_path_or_domain": None,
                "dictionary_name": None,
            },
        }

        # Store the relation method_name -> (dataset -> mock args and kwargs)
        method_test_args = {
            "build_deployed_dictionary": {
                dataset: {
                    "args": [
                        f"{dataset}.kdic",
                        dataset,
                        f"{dataset}Deployed.kdic",
                    ],
                    "kwargs": {},
                }
                for dataset in datasets
            },
            "build_dictionary_from_data_table": {
                dataset: {
                    "args": [f"{dataset}.csv", dataset, f"{dataset}.kdic"],
                    "kwargs": {},
                }
                for dataset in datasets
            },
            "check_database": {
                dataset: {
                    "args": [f"{dataset}.kdic", dataset, f"{dataset}.csv"],
                    "kwargs": {
                        "additional_data_tables": additional_data_tables[dataset]
                    },
                }
                for dataset in datasets
            },
            "detect_data_table_format": {
                dataset: {
                    "args": [f"{dataset}.csv"],
                    "kwargs": detect_data_table_format_kwargs[dataset],
                }
                for dataset in datasets
            },
            # We profit to test byte strings in the deploy_model test
            "deploy_model": {
                dataset: {
                    "args": [
                        bytes(f"{dataset}.kdic", encoding="ascii"),
                        bytes(dataset, encoding="ascii"),
                        bytes(f"{dataset}.csv", encoding="ascii"),
                        bytes(f"{dataset}Deployed.csv", encoding="ascii"),
                    ],
                    "kwargs": {
                        "additional_data_tables": (
                            {
                                bytes(key, encoding="ascii"): bytes(
                                    value, encoding="ascii"
                                )
                                for key, value in additional_data_tables[
                                    dataset
                                ].items()
                            }
                            if additional_data_tables[dataset] is not None
                            else None
                        ),
                        "output_additional_data_tables": (
                            {
                                bytes(key, encoding="ascii"): bytes(
                                    value, encoding="ascii"
                                )
                                for key, value in output_additional_data_tables[
                                    dataset
                                ].items()
                            }
                            if output_additional_data_tables[dataset] is not None
                            else None
                        ),
                    },
                }
                for dataset in datasets
            },
            "evaluate_predictor": {
                dataset: {
                    "args": [
                        f"Modeling{dataset}.kdic",
                        dataset,
                        f"{dataset}.csv",
                        f"{dataset}Results",
                    ],
                    "kwargs": {
                        "additional_data_tables": additional_data_tables[dataset]
                    },
                }
                for dataset in datasets
            },
            "export_dictionary_as_json": {
                dataset: {
                    "args": [f"{dataset}.kdic", f"{dataset}.kdicj"],
                    "kwargs": {},
                }
                for dataset in datasets
            },
            "extract_clusters": {
                dataset: {
                    "args": [
                        f"{dataset}Coclustering.khc",
                        coclustering_variables[dataset][0],
                        f"{dataset}Clusters.txt",
                    ],
                    "kwargs": {},
                }
                for dataset in datasets
            },
            "extract_keys_from_data_table": {
                dataset: {
                    "args": [
                        f"{dataset}.kdic",
                        dataset,
                        f"{dataset}.csv",
                        f"{dataset}Keys.csv",
                    ],
                    "kwargs": {},
                }
                for dataset in datasets
            },
            "prepare_coclustering_deployment": {
                dataset: {
                    "args": [
                        f"{dataset}.kdic",
                        dataset,
                        f"{dataset}._khc",
                        coclustering_variables[dataset][0],
                        coclustering_variables[dataset][1],
                        f"{dataset}Results",
                    ],
                    "kwargs": {
                        "max_part_numbers": max_part_numbers[dataset],
                    },
                }
                for dataset in datasets
            },
            "simplify_coclustering": {
                dataset: {
                    "args": [
                        f"{dataset}._khc",
                        f"Simplified{dataset}._khc",
                        f"{dataset}Results",
                    ],
                    "kwargs": {
                        "max_part_numbers": max_part_numbers[dataset],
                    },
                }
                for dataset in datasets
            },
            "sort_data_table": {
                dataset: {
                    "args": [
                        f"{dataset}.kdic",
                        dataset,
                        f"{dataset}.csv",
                        f"{dataset}Sorted.csv",
                    ],
                    "kwargs": {
                        "sort_variables": sort_variables[dataset],
                    },
                }
                for dataset in datasets
            },
            "train_coclustering": {
                dataset: {
                    "args": [
                        f"{dataset}.kdic",
                        dataset,
                        f"{dataset}.csv",
                        coclustering_variables[dataset],
                        f"{dataset}Results",
                    ],
                    "kwargs": {
                        "additional_data_tables": additional_data_tables[dataset],
                    },
                }
                for dataset in datasets
            },
            "train_predictor": {
                dataset: {
                    "args": [
                        f"{dataset}.kdic",
                        dataset,
                        f"{dataset}.csv",
                        target_variables[dataset],
                        f"{dataset}Results",
                    ],
                    "kwargs": {
                        "additional_data_tables": additional_data_tables[dataset],
                        "construction_rules": construction_rules[dataset],
                        "specific_pairs": specific_pairs[dataset],
                    },
                }
                for dataset in datasets
            },
            "train_recoder": {
                dataset: {
                    "args": [
                        f"{dataset}.kdic",
                        dataset,
                        f"{dataset}.csv",
                        target_variables[dataset],
                        f"{dataset}Results",
                    ],
                    "kwargs": {
                        "additional_data_tables": additional_data_tables[dataset],
                        "construction_rules": construction_rules[dataset],
                        "specific_pairs": specific_pairs[dataset],
                    },
                }
                for dataset in datasets
            },
        }

        return method_test_args

    def test_api_scenario_generation(self):
        """Tests the scenarios generated by the API

        These tests are not exhaustive, executed with the minimal parameters to trigger
        the more complex scenario generation code (lists, key-value sections) when they
        are present.
        """
        # Set the root directory of these tests
        test_resources_dir = os.path.join(resources_dir(), "scenario_generation", "api")

        # Use the test runner that only compares the scenarios
        default_runner = kh.get_runner()
        test_runner = ScenarioWriterRunner(self, test_resources_dir)
        kh.set_runner(test_runner)

        # Run test for all methods and all mock datasets parameters
        method_test_args = self._build_mock_api_method_parameters()
        for method_name, method_full_args in method_test_args.items():
            # Set the runners test name
            test_runner.test_name = method_name

            # Clean the directory for this method's tests
            cleanup_dir(test_runner.output_scenario_dir, "*/output/*._kh", verbose=True)

            # Test for each dataset mock parameters
            for dataset, dataset_method_args in method_full_args.items():
                test_runner.subtest_name = dataset
                with self.subTest(dataset=dataset, method=method_name):
                    # Execute the method
                    method = getattr(kh, method_name)
                    dataset_args = dataset_method_args["args"]
                    dataset_kwargs = dataset_method_args["kwargs"]
                    method(*dataset_args, **dataset_kwargs)

                    # Compare the reference with the output
                    assert_files_equal(
                        self,
                        test_runner.ref_scenario_path,
                        test_runner.output_scenario_path,
                        line_comparator=scenario_line_comparator,
                    )

        # Restore the default runner
        kh.set_runner(default_runner)

    def test_unknown_argument_in_api_method(self):
        """Tests if core.api raises ValueError when an unknown argument is passed"""
        # Obtain mock arguments for each API call
        method_test_args = self._build_mock_api_method_parameters()

        # Test for each dataset mock parameters
        for method_name, method_full_args in method_test_args.items():
            for dataset, dataset_method_args in method_full_args.items():
                # Test only for the Adult dataset
                if dataset != "Adult":
                    continue

                with self.subTest(method=method_name):
                    # These methods do not have kwargs so they cannot have extra args
                    if method_name in [
                        "detect_data_table_format",
                        "export_dictionary_as_json",
                    ]:
                        continue

                    # Execute the method with an invalid parameter
                    method = getattr(kh, method_name)
                    dataset_args = dataset_method_args["args"]
                    dataset_kwargs = dataset_method_args["kwargs"]
                    dataset_kwargs["INVALID_PARAM"] = False

                    # Check that the call raised ValueError
                    with self.assertRaises(ValueError) as context:
                        method(*dataset_args, **dataset_kwargs)

                    # Check the message
                    expected_msg = "Unknown argument 'INVALID_PARAM'"
                    output_msg = str(context.exception)
                    self.assertEqual(output_msg, expected_msg)

    def test_system_settings(self):
        """Test that the system settings are written to the scenario file"""
        # Create the root directory of these tests
        test_resources_dir = os.path.join(resources_dir(), "scenario_generation")

        # Use the test runner that only compares the scenarios
        default_runner = kh.get_runner()
        test_runner = ScenarioWriterRunner(self, test_resources_dir)
        test_runner.test_name = "system_settings"
        test_runner.subtest_name = "default"
        cleanup_dir(test_runner.output_scenario_dir, "*/output/*._kh")
        kh.set_runner(test_runner)

        # Call check_database (could be any other method), with the common execution
        # options set
        kh.check_database(
            "a.kdic",
            "dict_name",
            "data.txt",
            max_cores=10,
            memory_limit_mb=1000,
            temp_dir="/another/tmp",
            scenario_prologue="// Scenario prologue test",
        )

        # Compare the reference with the output
        assert_files_equal(
            self,
            test_runner.ref_scenario_path,
            test_runner.output_scenario_path,
            line_comparator=scenario_line_comparator,
        )

        # Set the runner to the default one
        kh.set_runner(default_runner)

    def test_runner_version(self):
        """Test that the runner respects the _write_version internal parameter"""
        # Create the root directory of these tests
        test_resources_dir = os.path.join(resources_dir(), "scenario_generation")

        # Use the test runner that only compares the scenarios, set _write_version
        default_runner = kh.get_runner()
        test_runner = ScenarioWriterRunner(self, test_resources_dir)
        test_runner.test_name = "runner_version"
        test_runner.subtest_name = "default"
        test_runner._write_version = True
        cleanup_dir(test_runner.output_scenario_dir, "*/output/*._kh")
        kh.set_runner(test_runner)

        # Call check_database (could be any other method)
        kh.check_database("a.kdic", "dict_name", "data.txt")

        # Check that the output scenario path has the version in its first line
        with open(
            test_runner.output_scenario_path, "r", encoding="ascii"
        ) as scenario_file:
            first_line = next(scenario_file).strip()
        self.assertEqual(
            first_line, f"// Generated by khiops-python {khiops.__version__}"
        )

        kh.set_runner(default_runner)

    def test_std_streams_files(self):
        """Test that the std* streams are written correctly to the specified files"""
        # Run the tests for each stream
        fixtures = {
            "stdout": create_mocked_raw_run(True, False, 0),
            "stderr": create_mocked_raw_run(False, True, 0),
        }
        test_resources_dir = os.path.join(resources_dir(), "tmp")
        for stream_name, mocked_raw_run in fixtures.items():
            # Run the subtest with the mocked runner
            stream_file_path = os.path.join(
                test_resources_dir, f"{stream_name}_test.txt"
            )
            fun_kwargs = {f"{stream_name}_file_path": stream_file_path}
            with self.subTest(stream_name=stream_name):
                with MockedRunnerContext(mocked_raw_run):
                    kh.check_database("a.kdic", "a", "a.txt", **fun_kwargs)

            # Check that the stream file exists and that the contents match the mock
            with open(stream_file_path, encoding="ascii") as stream_file:
                stream = stream_file.read().strip()
            self.assertEqual(stream, f"{stream_name}_content")

            # Clean up the output file
            os.remove(stream_file_path)

    def test_std_stream_warnings(self):
        """Test that if Khiops OK + non-empty std streams they are shown in a warning"""
        # Run the tests for each stream
        fixtures = {
            "stdout": create_mocked_raw_run(True, False, 0),
            "stderr": create_mocked_raw_run(False, True, 0),
        }
        for stream_name, mocked_raw_run in fixtures.items():
            # Run the subtest with the mocked runner
            with self.subTest(stream_name=stream_name):
                with MockedRunnerContext(mocked_raw_run):
                    with self.assertWarns(UserWarning) as cm:
                        kh.check_database("a.kdic", "a", "a.txt")

            # Check that the warning contains the stream content
            self.assertIn(f"{stream_name}_content", str(cm.warning))

    def test_std_stream_errors(self):
        """Test that if Khiops KO + non-empty std streams they are show in the exc."""
        # Run the tests for each stream
        fixtures = {
            "stdout": create_mocked_raw_run(True, False, 1),
            "stderr": create_mocked_raw_run(False, True, 1),
        }
        for stream_name, mocked_raw_run in fixtures.items():
            # Run the subtest with the mocked runner
            with self.subTest(stream_name=stream_name):
                with MockedRunnerContext(mocked_raw_run):
                    with self.assertRaises(kh.KhiopsRuntimeError) as cm:
                        kh.check_database("a.kdic", "a", "a.txt")

            # Check that the error contains the stream content
            self.assertIn(f"{stream_name}_content", str(cm.exception))


class MockedRunnerContext:
    """A context to mock the `~.KhiopsLocalRunner.raw_run` function"""

    def __init__(self, mocked_raw_run):
        self.mocked_raw_run = mocked_raw_run

    def __enter__(self):
        # Save the initial runner
        self._initial_runner = kh.get_runner()

        # Create the mock runner, patch the `raw_run` function, enter its context
        self.mocked_runner = KhiopsLocalRunner()
        kh.set_runner(self.mocked_runner)
        self.mock_context = mock.patch.object(
            self.mocked_runner, "raw_run", new=self.mocked_raw_run
        )
        self.mock_context.__enter__()

        # The original `KhiopsLocalRunner._get_khiops_version` method needs to
        # call into the Khiops binary via `raw_run` which is mocked; hence, it
        # needs to be mocked as well
        self.mock_context = mock.patch.object(
            self.mocked_runner,
            "_get_khiops_version",
            return_value=KhiopsVersion("10.2.2"),
        )
        self.mock_context.__enter__()

        # Return the runner to be used in the context
        return self.mocked_runner

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Restore the mock context and the original runner
        self.mock_context.__exit__(exc_type, exc_value, exc_traceback)
        kh.set_runner(self._initial_runner)


def create_mocked_raw_run(stdout, stderr, return_code):
    """Creates a mock for the `.KhiopsRunner.run` method"""

    def mocked_raw_run(*_, **__):
        return (
            "stdout_content" if stdout else "",
            "stderr_content" if stderr else "",
            return_code,
        )

    return mocked_raw_run


class KhiopsCoreServicesTests(unittest.TestCase):
    """Test the services of the core module classes

    Specifically, the tests in this class are for the services not used in the *write_*
    methods, as those are already tested in KhiopsCoreIOTests.
    """

    def test_analysis_results_simple_initializations(self):
        """Tests simple initialization operations analysis_results classes"""
        results = kh.AnalysisResults()
        with open(os.devnull, "wb") as devnull_file:
            results.write_report(devnull_file)
            results.tool = "Khiops Coclustering"
            results.write_report(devnull_file)
        kh.PreparationReport()
        kh.BivariatePreparationReport()
        kh.ModelingReport()
        kh.EvaluationReport()
        var_stats = kh.VariableStatistics()
        var_stats.init_details()
        var_pair_stats = kh.VariablePairStatistics()
        var_pair_stats.init_details()
        kh.DataGrid()
        kh.DataGridDimension()
        kh.PartInterval()
        kh.PartValue()
        kh.PartValueGroup()
        predictor = kh.TrainedPredictor()
        predictor.init_details()
        kh.SelectedVariable()
        predictor_perf = kh.PredictorPerformance()
        predictor_perf.init_details()
        kh.ConfusionMatrix()
        kh.PredictorCurve()

    def test_analysis_results_simple_edge_cases(self):
        """Test simple edge cases for analysis_results classes"""
        # Test the writing to an invalid object
        with self.assertRaises(TypeError):
            results = kh.AnalysisResults()
            results.write_report("A STRING IS NOT A VALID STREAM")

        # Test errors on the preparation report creation
        with self.assertRaises(TypeError):
            kh.PreparationReport(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.PreparationReport(json_data={"summary": None})
        self.assertIn("'reportType' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.PreparationReport(json_data={"reportType": "Preparation"})
        self.assertIn("'summary' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.PreparationReport(json_data={"reportType": "WHATEVER", "summary": None})
        self.assertIn("'reportType' is not 'Preparation'", cm.exception.args[0])

        # Test errors on the bivariate preparation report creation
        with self.assertRaises(TypeError):
            kh.BivariatePreparationReport(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.BivariatePreparationReport(json_data={"summary": None})
        self.assertIn("'reportType' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.BivariatePreparationReport(
                json_data={"reportType": "BivariatePreparation"}
            )
        self.assertIn("'summary' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.BivariatePreparationReport(
                json_data={"reportType": "WHATEVER", "summary": None}
            )
        self.assertIn(
            "'reportType' is not 'BivariatePreparation'", cm.exception.args[0]
        )

        # Test errors modeling report creation
        with self.assertRaises(TypeError):
            kh.ModelingReport(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.ModelingReport(json_data={"summary": None})
        self.assertIn("'reportType' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.ModelingReport(json_data={"reportType": "Modeling"})
        self.assertIn("'summary' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError):
            kh.ModelingReport(json_data={"reportType": "WHATEVER", "summary": None})

        # Test the evaluation report creation
        with self.assertRaises(TypeError):
            kh.EvaluationReport(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.EvaluationReport(json_data={"summary": None})
        self.assertIn("'reportType' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.EvaluationReport(json_data={"reportType": "Evaluation"})
        self.assertIn("'summary' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.EvaluationReport(json_data={"reportType": "Evaluation", "summary": None})
        self.assertIn("'evaluationType' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.EvaluationReport(
                json_data={
                    "reportType": "WHATEVER",
                    "evaluationType": "Train",
                    "summary": None,
                }
            )
        self.assertIn("'reportType' is not 'Evaluation'", cm.exception.args[0])

        # Test errors in the variable stats creation
        with self.assertRaises(TypeError):
            kh.VariableStatistics(json_data="NOT A DICT")
        with self.assertRaises(TypeError):
            var_stats = kh.VariableStatistics()
            var_stats.init_details(json_data="NOT A DICT")

        # Test errors in the pair variable stats creation
        with self.assertRaises(TypeError):
            kh.VariablePairStatistics(json_data="NOT A DICT")
        with self.assertRaises(TypeError):
            var_pair_stats = kh.VariablePairStatistics()
            var_pair_stats.init_details(json_data="NOT A DICT")

        # Test errors in the data grid creation
        with self.assertRaises(TypeError):
            kh.DataGrid(json_data="NOT A DICT")

        # Test errors in the data grid dimension creation
        with self.assertRaises(TypeError):
            kh.DataGridDimension(json_data="NOT A DICT")

        # Test errors in the interval part creation
        with self.assertRaises(TypeError):
            kh.PartInterval(json_data="NOT A LIST")
        with self.assertRaises(ValueError):
            kh.PartInterval([0])
        with self.assertRaises(ValueError):
            kh.PartInterval([0, 1, 2])

        # Test errors in the value part creation
        with self.assertRaises(TypeError):
            kh.PartValue(json_data={})

        # Test errors in the value group part creation
        with self.assertRaises(TypeError):
            kh.PartValueGroup(json_data="NOT A LIST")

        # Test errors in the trained predictor creation
        with self.assertRaises(TypeError):
            kh.TrainedPredictor(json_data="NOT A DICT")

        # Test errors in the selected variable creation
        with self.assertRaises(TypeError):
            kh.SelectedVariable(json_data="NOT A DICT")
        with self.assertRaises(TypeError):
            trained_predictor = kh.TrainedPredictor()
            trained_predictor.init_details(json_data="NOT A DICT")

        # Test errors in the confusion matrix creation
        with self.assertRaises(TypeError):
            kh.ConfusionMatrix(json_data="NOT A DICT")

        # Test errors in the predictor performance creation
        with self.assertRaises(TypeError):
            kh.PredictorPerformance(json_data="NOT A DICT")
        with self.assertRaises(TypeError):
            predictor_perf = kh.PredictorPerformance()
            predictor_perf.init_details(json_data="NOT A DICT")
        with self.assertRaises(ValueError):
            predictor_perf = kh.PredictorPerformance()
            predictor_perf.get_metric_names()

        # Test the error when a predictor curve does not have the 'classifier' or
        # 'regression' field.
        with self.assertRaises(TypeError):
            kh.PredictorCurve(json_data="NOT A DICT")
        with self.assertRaises(ValueError):
            kh.PredictorCurve(json_data={"curve": [0.0]})

    def test_analysis_results_accessors(self):
        """Test the accessors of the analysis results classes"""
        # Set the test paths
        test_resources_dir = os.path.join(resources_dir(), "analysis_results")
        ref_json_reports_dir = os.path.join(test_resources_dir, "ref_json_reports")

        # Set the expected method outputs
        expected_outputs = {
            "PreparationReport": {
                "get_variable_names": {
                    "Adult": [
                        "relationship",
                        "marital_status",
                        "capital_gain",
                        "age",
                        "education_num",
                        "education",
                        "occupation",
                        "hours_per_week",
                        "capital_loss",
                        "sex",
                        "workclass",
                        "race",
                        "native_country",
                        "Label",
                        "fnlwgt",
                    ],
                    "AdultEvaluation": None,
                    "Iris2D": [
                        "SPetalLength",
                        "PetalLength",
                        "PetalWidth",
                        "Class2",
                        "LowerPetalLength",
                        "Class1",
                        "UpperPetalWidth",
                        "SepalLength",
                        "SepalWidth",
                        "Dummy1",
                        "Dummy2",
                    ],
                    "IrisC": [
                        "SPetalLength",
                        "PetalLength",
                        "PetalWidth",
                        "Class2",
                        "LowerPetalLength",
                        "Class1",
                        "UpperPetalWidth",
                        "SepalLength",
                        "SepalWidth",
                        "Dummy1",
                        "Dummy2",
                    ],
                    "IrisR": [
                        "SPetalLength",
                        "Class",
                        "PetalWidth",
                        "LowerPetalLength",
                        "Class1",
                        "SepalLength",
                        "Class2",
                        "UpperPetalWidth",
                        "SepalWidth",
                        "Dummy1",
                        "Dummy2",
                    ],
                }
            },
            "BivariatePreparationReport": {
                "get_variable_pair_names": {
                    "Iris2D": [
                        ("Class1", "Dummy2"),
                        ("Class2", "Dummy2"),
                        ("Dummy2", "LowerPetalLength"),
                        ("Dummy2", "PetalLength"),
                        ("Dummy2", "PetalWidth"),
                        ("Dummy2", "SPetalLength"),
                        ("Dummy2", "SepalLength"),
                        ("Dummy2", "SepalWidth"),
                        ("Dummy2", "UpperPetalWidth"),
                        ("SepalWidth", "UpperPetalWidth"),
                        ("Class2", "SepalLength"),
                        ("SepalLength", "SepalWidth"),
                        ("Class2", "SepalWidth"),
                        ("Class2", "UpperPetalWidth"),
                        ("Class1", "SepalWidth"),
                        ("LowerPetalLength", "SepalWidth"),
                        ("PetalWidth", "SepalWidth"),
                        ("SPetalLength", "SepalWidth"),
                        ("PetalLength", "SepalWidth"),
                        ("Class1", "UpperPetalWidth"),
                        ("LowerPetalLength", "UpperPetalWidth"),
                        ("SepalLength", "UpperPetalWidth"),
                        ("Class2", "LowerPetalLength"),
                        ("Class1", "Class2"),
                        ("Class1", "SepalLength"),
                        ("LowerPetalLength", "SepalLength"),
                        ("PetalWidth", "SepalLength"),
                        ("SPetalLength", "SepalLength"),
                        ("PetalLength", "SepalLength"),
                        ("PetalWidth", "UpperPetalWidth"),
                        ("SPetalLength", "UpperPetalWidth"),
                        ("PetalLength", "UpperPetalWidth"),
                        ("Class2", "PetalWidth"),
                        ("Class2", "PetalLength"),
                        ("Class2", "SPetalLength"),
                        ("Class1", "PetalLength"),
                        ("Class1", "LowerPetalLength"),
                        ("LowerPetalLength", "PetalLength"),
                        ("LowerPetalLength", "PetalWidth"),
                        ("LowerPetalLength", "SPetalLength"),
                        ("Class1", "SPetalLength"),
                        ("Class1", "PetalWidth"),
                        ("PetalWidth", "SPetalLength"),
                        ("PetalLength", "PetalWidth"),
                        ("PetalLength", "SPetalLength"),
                    ],
                }
            },
            "ModelingReport": {
                "get_predictor_names": {
                    "Adult": ["Selective Naive Bayes", "Univariate relationship"],
                    "Iris2D": ["Selective Naive Bayes", "Univariate SPetalLength"],
                    "IrisC": ["Selective Naive Bayes", "Univariate SPetalLength"],
                    "IrisR": ["Selective Naive Bayes", "Univariate SPetalLength"],
                }
            },
            "EvaluationReport": {
                "get_predictor_names": {
                    "Adult": ["Selective Naive Bayes", "Univariate relationship"],
                    "AdultEvaluation": [
                        "Selective Naive Bayes",
                        "Univariate relationship",
                    ],
                    "Iris2D": ["Selective Naive Bayes", "Univariate SPetalLength"],
                    "IrisC": ["Selective Naive Bayes", "Univariate SPetalLength"],
                    "IrisR": ["Selective Naive Bayes", "Univariate SPetalLength"],
                }
            },
            "PredictorPerformance": {
                "get_metric_names": {
                    "Adult": ["accuracy", "compression", "auc"],
                    "AdultEvaluation": ["accuracy", "compression", "auc"],
                    "Iris2D": ["accuracy", "compression", "auc"],
                    "IrisC": ["accuracy", "compression", "auc"],
                    "IrisR": [
                        "rmse",
                        "mae",
                        "nlpd",
                        "rank_rmse",
                        "rank_mae",
                        "rank_nlpd",
                    ],
                }
            },
        }

        # Test the accessors functions in different results files
        results_file_names = ["Adult", "AdultEvaluation", "Iris2D", "IrisC", "IrisR"]
        for result_file_name in results_file_names:
            results_file_path = os.path.join(
                ref_json_reports_dir, f"{result_file_name}.khj"
            )
            results = kh.read_analysis_results_file(results_file_path)
            for report in results.get_reports():
                if isinstance(report, kh.PreparationReport):
                    with self.subTest(
                        result_file_name=result_file_name,
                        report_class="PreparationReport",
                    ):
                        self._test_preparation_report_accessors(
                            result_file_name, report, expected_outputs
                        )
                elif isinstance(report, kh.BivariatePreparationReport):
                    with self.subTest(
                        result_file_name=result_file_name,
                        report_class="BivariatePreparationReport",
                    ):
                        self._test_bivariate_preparation_report_accessors(
                            result_file_name, report, expected_outputs
                        )
                elif isinstance(report, kh.ModelingReport):
                    with self.subTest(
                        result_file_name=result_file_name, report_class="ModelingReport"
                    ):
                        self._test_modeling_report_accessors(
                            result_file_name, report, expected_outputs
                        )
                else:
                    with self.subTest(
                        result_file_name=result_file_name,
                        report_class="EvaluationReport",
                    ):
                        self.assertIsInstance(report, kh.EvaluationReport)
                        self._test_evaluation_report_accessors(
                            result_file_name, report, expected_outputs
                        )

    def _test_preparation_report_accessors(
        self, result_file_name, report, expected_outputs
    ):
        """Tests accessors for the PreparationReport class"""
        # Test normal access
        self.assertEqual(
            report.get_variable_names(),
            expected_outputs["PreparationReport"]["get_variable_names"][
                result_file_name
            ],
        )
        for variable_index, variable_name in enumerate(report.get_variable_names()):
            variable_stats = report.get_variable_statistics(variable_name)
            self.assertIsInstance(variable_stats, kh.VariableStatistics)
            self.assertEqual(
                variable_stats, report.variables_statistics[variable_index]
            )

        # Test anomalous access
        with self.assertRaises(KeyError):
            report.get_variable_statistics("INEXISTENT VARIABLE NAME")

    def _test_bivariate_preparation_report_accessors(
        self, result_file_name, report, expected_outputs
    ):
        """Tests accessors for the BivariatePreparationReport class"""
        # Test normal access
        self.assertEqual(
            report.get_variable_pair_names(),
            expected_outputs["BivariatePreparationReport"]["get_variable_pair_names"][
                result_file_name
            ],
        )
        for var_index, (var_name1, var_name2) in enumerate(
            report.get_variable_pair_names()
        ):
            var_pair_stats = report.get_variable_pair_statistics(var_name1, var_name2)
            self.assertIsInstance(var_pair_stats, kh.VariablePairStatistics)
            self.assertEqual(
                var_pair_stats, report.variables_pairs_statistics[var_index]
            )

        # Test anomalous access
        with self.assertRaises(KeyError):
            report.get_variable_pair_statistics("INEXISTENT VARIABLE", "PAIR NAME")

    def _test_modeling_report_accessors(
        self, result_file_name, report, expected_outputs
    ):
        """Tests accessors functions for the ModelingReport class"""
        # Test normal access
        self.assertEqual(
            report.get_predictor_names(),
            expected_outputs["ModelingReport"]["get_predictor_names"][result_file_name],
        )
        for predictor_index, predictor_name in enumerate(report.get_predictor_names()):
            predictor = report.get_predictor(predictor_name)
            self.assertIsInstance(predictor, kh.TrainedPredictor)
            self.assertEqual(predictor, report.trained_predictors[predictor_index])
        self.assertEqual(
            report.get_snb_predictor(),
            report.get_predictor("Selective Naive Bayes"),
        )

        # Test anomalous access
        with self.assertRaises(KeyError):
            report.get_predictor("INEXISTENT REPORT NAME")

    def _test_evaluation_report_accessors(
        self, result_file_name, report, expected_outputs
    ):
        """Test accessors for the EvaluationReport class"""
        # Test normal access
        self.assertEqual(
            report.get_predictor_names(),
            expected_outputs["EvaluationReport"]["get_predictor_names"][
                result_file_name
            ],
        )
        for predictor_index, predictor_name in enumerate(report.get_predictor_names()):
            predictor_performance = report.get_predictor_performance(predictor_name)
            self.assertIsInstance(predictor_performance, kh.PredictorPerformance)
            self.assertEqual(
                predictor_performance, report.predictors_performance[predictor_index]
            )
        self.assertEqual(
            report.get_snb_performance(),
            report.get_predictor_performance("Selective Naive Bayes"),
        )

        # Test anomalous access
        with self.assertRaises(KeyError):
            report.get_predictor_performance("INEXISTENT REPORT NAME")

        # Test anomalous access to performance objects
        for predictor_name in report.get_predictor_names():
            self._test_performance_report_accessors(
                result_file_name,
                report.learning_task,
                report.get_predictor_performance(predictor_name),
                expected_outputs,
            )

        # Test normal and anomalous access to performance curves
        for predictor_name in report.get_predictor_names():
            # Test normal access
            if report.learning_task == "Classification analysis":
                for target_value in report.classification_target_values:
                    report.get_classifier_lift_curve(predictor_name, target_value)
                    report.get_classifier_lift_curve("Random", target_value)
            else:
                report.get_regressor_rec_curve(predictor_name)

            # Test anomalous access
            with self.assertRaises(ValueError):
                if report.learning_task == "Classification analysis":
                    report.get_regressor_rec_curve(predictor_name)
                else:
                    report.get_classifier_lift_curve(predictor_name, "INEXISTENT VALUE")
            if report.learning_task == "Classification analysis":
                with self.assertRaises(KeyError):
                    report.get_classifier_lift_curve(predictor_name, "INEXISTENT VALUE")
        with self.assertRaises(KeyError):
            if report.learning_task == "Classification analysis":
                report.get_classifier_lift_curve(
                    "INEXISTENT PREDICTOR", report.classification_target_values[0]
                )
            else:
                report.get_regressor_rec_curve("INEXISTENT PREDICTOR")

        # Test anomalous access to SNB curves
        with self.assertRaises(ValueError):
            if report.learning_task == "Classification analysis":
                report.get_snb_rec_curve()
            else:
                report.get_snb_lift_curve("INEXISTENT VALUE")
        if report.learning_task == "Classification analysis":
            with self.assertRaises(KeyError):
                report.get_snb_lift_curve("INEXISTENT VALUE")

    def _test_performance_report_accessors(
        self, result_file_name, learning_task, report, expected_outputs
    ):
        """Test accessors of the PerformanceReport class"""
        self.assertEqual(
            report.get_metric_names(),
            expected_outputs["PredictorPerformance"]["get_metric_names"][
                result_file_name
            ],
        )
        # Test normal access
        for metric_name in report.get_metric_names():
            metric = report.get_metric(metric_name)
            self.assertTrue(isinstance(metric, (float, int)))

        # Test anomalous access
        with self.assertRaises(ValueError):
            if learning_task == "Classification analysis":
                report.get_metric("rmse")
            else:
                report.get_metric("auc")

    def test_coclustering_results_simple_initializations(self):
        kh.CoclusteringResults()
        kh.CoclusteringReport()
        kh.CoclusteringDimension()
        kh.CoclusteringDimensionPart()
        kh.CoclusteringDimensionPartInterval()
        kh.CoclusteringDimensionPartValueGroup()
        kh.CoclusteringDimensionPartValue()
        kh.CoclusteringCluster()
        kh.CoclusteringCell()

    def test_coclustering_results_simple_edge_cases(self):
        """Test simple edge cases for coclustering_results classes"""
        # Test the writing to an invalid object
        with self.assertRaises(TypeError):
            results = kh.CoclusteringResults()
            results.write_report("A STRING IS NOT A VALID STREAM")

        # Test errors in the creation of coclustering classes
        # Test the evaluation report creation
        with self.assertRaises(TypeError):
            kh.CoclusteringResults(json_data="NOT A DICT")
        with self.assertRaises(TypeError):
            kh.CoclusteringReport(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.CoclusteringReport(json_data={})
        self.assertIn("'summary' key not found", cm.exception.args[0])
        dimension = kh.CoclusteringDimension()
        with self.assertRaises(TypeError):
            dimension.init_summary(json_data="NOT A DICT")
        with self.assertRaises(TypeError):
            dimension.init_partition(json_data="NOT A DICT")
        with self.assertRaises(TypeError):
            dimension.init_hierarchy(json_data="NOT A DICT")
        with self.assertRaises(TypeError):
            kh.CoclusteringDimensionPart(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.CoclusteringDimensionPart(json_data={})
        self.assertIn("'cluster' key not found", cm.exception.args[0])
        with self.assertRaises(TypeError):
            kh.CoclusteringDimensionPartInterval(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.CoclusteringDimensionPartInterval(json_data={"cluster": "MYCLUSTER"})
        self.assertIn("'bounds' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.CoclusteringDimensionPartInterval(
                json_data={"cluster": "MYCLUSTER", "bounds": []}
            )
        self.assertIn("'bounds' key must be a list of length 2", cm.exception.args[0])
        with self.assertRaises(TypeError):
            kh.CoclusteringDimensionPartValueGroup(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.CoclusteringDimensionPartValueGroup({"cluster": "MYCLUSTER"})
        self.assertIn("'values' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.CoclusteringDimensionPartValueGroup(
                {"cluster": "MYCLUSTER", "values": []}
            )
        self.assertIn("'valueFrequencies' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.CoclusteringDimensionPartValueGroup(
                {"cluster": "MYCLUSTER", "values": [], "valueFrequencies": []}
            )
        self.assertIn("'valueTypicalities' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.CoclusteringDimensionPartValueGroup(
                {
                    "cluster": "MYCLUSTER",
                    "values": [],
                    "valueFrequencies": [1],
                    "valueTypicalities": [],
                }
            )
        self.assertIn(
            "'valueFrequencies' key list must have the same length",
            cm.exception.args[0],
        )
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.CoclusteringDimensionPartValueGroup(
                {
                    "cluster": "MYCLUSTER",
                    "values": [],
                    "valueFrequencies": [],
                    "valueTypicalities": [1],
                }
            )
        self.assertIn(
            "'valueTypicalities' key list must have the same length",
            cm.exception.args[0],
        )
        with self.assertRaises(TypeError):
            kh.CoclusteringCluster(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.CoclusteringCluster({})
        self.assertIn("'cluster' key not found", cm.exception.args[0])
        with self.assertRaises(kh.KhiopsJSONError) as cm:
            kh.CoclusteringCluster({"cluster": "MYCLUSTER"})
        self.assertIn("'parentCluster' key not found", cm.exception.args[0])

    def test_coclustering_results_accessors(self):
        """Test CoclusteringResults accessors functions"""
        # Set the test paths
        test_resources_dir = os.path.join(resources_dir(), "coclustering_results")
        ref_json_reports_dir = os.path.join(test_resources_dir, "ref_json_reports")

        # Set the expected method outputs
        expected_outputs = {
            "CoclusteringReport": {
                "get_dimension_names": {
                    "Adult": [
                        "age",
                        "occupation",
                        "education_num",
                        "hours_per_week",
                        "marital_status",
                        "sex",
                    ],
                    "Iris": ["PetalLength", "PetalWidth", "Class"],
                }
            }
        }

        results_file_names = ["Adult", "Iris"]
        for results_file_name in results_file_names:
            results_file_path = os.path.join(
                ref_json_reports_dir, f"{results_file_name}.khcj"
            )
            results = kh.read_coclustering_results_file(results_file_path)
            self.assertEqual(
                results.coclustering_report.get_dimension_names(),
                expected_outputs["CoclusteringReport"]["get_dimension_names"][
                    results_file_name
                ],
            )
            for dimension_index, dimension_name in enumerate(
                results.coclustering_report.get_dimension_names()
            ):
                self.assertEqual(
                    results.coclustering_report.dimensions[dimension_index],
                    results.coclustering_report.get_dimension(dimension_name),
                )

    def test_dictionary_simple_initializations(self):
        """Test simple initialization operation of dictionary classes"""
        domain = kh.DictionaryDomain()
        domain.tool = "Khiops Dictionary"
        domain.name = "Iris"
        dictionary = kh.Dictionary()
        dictionary.name = "Test"
        dictionary.label = "Some comment"
        domain.add_dictionary(dictionary)
        with open(os.devnull, "wb") as devnull_file:
            domain.write(devnull_file)

    def test_dictionary_simple_edge_cases(self):
        """Test simple edge cases of the classes of dictionary classes"""
        # Test anomalous DictionaryDomain actions
        with self.assertRaises(kh.KhiopsJSONError):
            kh.DictionaryDomain(json_data={"tool": "INVALID TOOL", "version": "0.0"})
        with self.assertRaises(kh.KhiopsJSONError):
            kh.DictionaryDomain(json_data={"tool": "Khiops Dictionary"})
        domain = kh.DictionaryDomain()
        with self.assertRaises(TypeError):
            domain.add_dictionary("NOT A DICTIONARY OBJECT")
        with self.assertRaises(TypeError):
            domain.write("NOT A STREAM")

        # This is anomalous but ok
        kh.DictionaryDomain(json_data={"tool": "Khiops Dictionary", "version": "0.0"})

        # Test anomalous Dictionary actions
        with self.assertRaises(TypeError):
            kh.Dictionary(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError):
            kh.Dictionary(json_data={})
        with self.assertRaises(kh.KhiopsJSONError):
            kh.Dictionary(json_data={"name": "Iris", "variables": "NOT A LIST"})
        with self.assertRaises(kh.KhiopsJSONError):
            kh.Dictionary(
                json_data={"name": "Iris", "variables": [{"NotNameNorBlockName": None}]}
            )
        dictionary = kh.Dictionary(json_data={"name": "Iris"})
        with self.assertRaises(TypeError):
            dictionary.add_variable("NOT A VARIABLE OBJECT")
        with self.assertRaises(TypeError):
            dictionary.add_variable_block("NOT A VARIABLE BLOCK OBJECT")
        with self.assertRaises(TypeError):
            dictionary.write("NOT A WRITER")

        # Test anomalous Variable actions
        with self.assertRaises(TypeError):
            kh.Variable(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError):
            kh.Variable(json_data={})
        with self.assertRaises(kh.KhiopsJSONError):
            kh.Variable(json_data={"name": "SomeVar"})
        variable = kh.Variable()
        variable.type = "Categorical"
        variable.name = "SomeVar"
        with self.assertRaises(TypeError):
            variable.write("NOT A WRITER")

        # Test anomalous VariableBlock actions
        with self.assertRaises(TypeError):
            kh.VariableBlock(json_data="NOT A DICT")
        with self.assertRaises(kh.KhiopsJSONError):
            kh.VariableBlock(json_data={})
        variable_block = kh.VariableBlock()
        with self.assertRaises(TypeError):
            variable_block.add_variable("NOT A VARIABLE")
        with self.assertRaises(TypeError):
            variable_block.remove_variable("NOT A VARIABLE")
        with self.assertRaises(ValueError):
            variable_block.remove_variable(variable)
        with self.assertRaises(TypeError):
            variable_block.write("NOT A WRITER")

        # Test Anomalous MetaData actions
        with self.assertRaises(TypeError):
            kh.MetaData("NOT A DICT")
        meta_data = kh.MetaData()
        with self.assertRaises(TypeError):
            meta_data.write("NOT A WRITER")
        with self.assertRaises(TypeError):
            meta_data.add_value(42, "value")
        with self.assertRaises(TypeError):
            meta_data.add_value("key", object())
        with self.assertRaises(TypeError):
            meta_data.get_value(object())
        with self.assertRaises(TypeError):
            meta_data.remove_key(object())
        meta_data.add_value("key", "value")
        with self.assertRaises(KeyError):
            meta_data.get_value("INEXISTENT KEY")
        with self.assertRaises(ValueError):
            meta_data.add_value("key", "REPEATED KEY")
        with self.assertRaises(KeyError):
            meta_data.remove_key("INEXISTENT KEY")

    def test_dictionary_accessors(self):
        """Tests accessors functions of the dictionary classes"""
        # Set the test paths
        test_resources_dir = os.path.join(resources_dir(), "dictionary")
        ref_kdicj_dir = os.path.join(test_resources_dir, "ref_kdicj")

        # Test the accessors functions in different dictionary files
        domain_names = ["Adult", "Customer", "SpliceJunction", "SpliceJunctionModeling"]
        for domain_name in domain_names:
            kdicj_path = os.path.join(ref_kdicj_dir, f"{domain_name}.kdicj")
            domain = kh.read_dictionary_file(kdicj_path)

            # Test addition and removal a dictionary
            dictionary_copy = domain.dictionaries[0].copy()
            dictionary_copy.name = f"Copy{dictionary_copy.name}"
            domain.add_dictionary(dictionary_copy)
            self.assertEqual(
                dictionary_copy, domain.remove_dictionary(dictionary_copy.name)
            )

            # Test removal an inexistent dictionary
            with self.assertRaises(KeyError):
                domain.remove_dictionary("INEXISTENT DICTIONARY")

            for dictionary in domain.dictionaries:
                for key in dictionary.meta_data.keys:
                    self.assertIn(key, dictionary.meta_data)
                    self.assertEqual(
                        dictionary.get_value(key), dictionary.meta_data.get_value(key)
                    )

                # Test mass "Used" variable set
                dictionary_copy = dictionary.copy()
                dictionary_copy.use_all_variables(False)
                for variable in dictionary_copy.variables:
                    self.assertFalse(variable.used)
                dictionary_copy.use_all_variables(True)
                for variable in dictionary_copy.variables:
                    self.assertTrue(variable.used)

                # Test key access
                for variable in dictionary_copy.variables:
                    if variable.name in dictionary_copy.key:
                        self.assertTrue(dictionary_copy.is_key_variable(variable))
                    else:
                        self.assertFalse(dictionary_copy.is_key_variable(variable))

                # Test Dictionary variable accessors
                variable = kh.Variable()
                variable.name = kh.name = "NewVar"
                dictionary_copy.add_variable(variable)
                with self.assertRaises(ValueError):
                    dictionary_copy.add_variable(variable)
                removed_variable = dictionary_copy.remove_variable(variable.name)
                self.assertEqual(removed_variable, variable)
                with self.assertRaises(KeyError):
                    dictionary_copy.remove_variable(variable.name)
                variable.name = ""
                with self.assertRaises(ValueError):
                    dictionary_copy.add_variable(variable)

                # Test Dictionary variable block accessors
                # Create a simple block
                block = kh.VariableBlock()
                block.name = ""
                with self.assertRaises(ValueError):
                    dictionary_copy.add_variable_block(block)
                block.name = "NewBlock"
                block_variable = kh.Variable()
                block_variable.name = "VarInBlock"
                block_variable.type = "Numerical"
                block_variable.used = True
                block_variable.block = block
                block.add_variable(block_variable)

                # Add and remove the block
                dictionary_copy.add_variable_block(block)
                self.assertEqual(block, dictionary_copy.get_variable_block(block.name))
                removed_block = dictionary_copy.remove_variable_block(block.name)
                self.assertEqual(block, removed_block)
                self.assertIsNone(block_variable.variable_block)
                self.assertEqual(block.variables, [])
                with self.assertRaises(KeyError):
                    dictionary_copy.get_variable_block(block.name)

                # Add and remove the block and remove the native variables
                dictionary_copy.remove_variable(block_variable.name)
                block.add_variable(block_variable)
                dictionary_copy.add_variable_block(block)
                self.assertEqual(block, dictionary_copy.get_variable_block(block.name))
                removed_block = dictionary_copy.remove_variable_block(
                    block.name, keep_native_block_variables=False
                )
                self.assertEqual(block, removed_block)
                self.assertEqual(block.variables, [block_variable])
                self.assertEqual(block_variable.block, removed_block)
                with self.assertRaises(KeyError):
                    dictionary_copy.get_variable(block_variable.name)
                with self.assertRaises(KeyError):
                    dictionary_copy.get_variable_block(block.name)

                # Set the block as non-native add, and remove it
                block.rule = "SomeBlockCreatingRule()"
                dictionary_copy.add_variable_block(block)
                self.assertEqual(block, dictionary_copy.get_variable_block(block.name))
                removed_block = dictionary_copy.remove_variable_block(
                    block.name,
                )
                self.assertEqual(block, removed_block)
                self.assertEqual(block.variables, [block_variable])
                self.assertEqual(block_variable.block, removed_block)
                with self.assertRaises(KeyError):
                    dictionary_copy.get_variable(block_variable.name)
                with self.assertRaises(KeyError):
                    dictionary_copy.get_variable_block(block.name)

                # Test Dictionary variable and block accessors by cleaning the dict.
                for variable_name in [
                    variable.name for variable in dictionary_copy.variables
                ]:
                    dictionary_copy.remove_variable(variable_name)
                self.assertEqual(dictionary_copy.variables, [])

                # Test Variable data accessors
                for variable in dictionary.variables:
                    for key in variable.meta_data.keys:
                        self.assertEqual(
                            variable.get_value(key), variable.meta_data.get_value(key)
                        )

                    if variable.variable_block is not None:
                        self.assertEqual(
                            variable.variable_block,
                            dictionary.get_variable_block(variable.variable_block.name),
                        )

                # Test Variable block meta_data accessors
                for variable_block in dictionary.variable_blocks:
                    variable_block.meta_data.add_value("SomeKey", "SomeValue")
                    for key in variable_block.meta_data.keys:
                        self.assertEqual(
                            variable_block.get_value(key),
                            variable_block.meta_data.get_value(key),
                        )
                    removed_value = variable_block.meta_data.remove_key("SomeKey")
                    self.assertEqual(removed_value, "SomeValue")

    def test_dictionary_extract_data_paths(self):
        """Tests the extract_data_paths Dictionary method"""
        # Set the test paths
        test_resources_dir = os.path.join(resources_dir(), "dictionary")
        ref_kdicj_dir = os.path.join(test_resources_dir, "ref_kdicj")

        # Set the expected outputs
        expected_data_paths = {
            "Adult": {"Adult": []},
            "SpliceJunction": {
                "SpliceJunction": ["DNA"],
                "SpliceJunctionDNA": [],
            },
            "SpliceJunctionModeling": {
                "SNB_SpliceJunction": ["SpliceJunctionDNA"],
                "SNB_SpliceJunctionDNA": [],
            },
            "Customer": {
                "Address": [],
                "Customer": [
                    "Services",
                    "Services/Usages",
                    "Address",
                ],
                "Service": ["Usages"],
                "Usage": [],
            },
            "CustomerExtended": {
                "Address": ["/City", "/Country"],
                "City": ["/Country"],
                "Country": [],
                "Customer": [
                    "Services",
                    "Services/Usages",
                    "Address",
                    "/City",
                    "/Country",
                    "/Product",
                ],
                "Product": [],
                "Service": ["Usages", "/Product"],
                "Usage": ["/Product"],
            },
        }
        dictionaries_by_domain = {
            "Adult": ["Adult"],
            "SpliceJunction": ["SpliceJunction", "SpliceJunctionDNA"],
            "Customer": [
                "Address",
                "Customer",
                "Service",
                "Usage",
            ],
            "CustomerExtended": [
                "Address",
                "City",
                "Country",
                "Customer",
                "Product",
                "Service",
                "Usage",
            ],
        }

        # Test the method for different dictionary files
        for domain_name, dictionary_names in dictionaries_by_domain.items():
            domain = kh.read_dictionary_file(
                os.path.join(ref_kdicj_dir, f"{domain_name}.kdicj")
            )
            for dictionary_name in dictionary_names:
                with self.subTest(
                    domain_name=domain_name, dictionary_name=dictionary_name
                ):
                    current_data_paths = set(
                        expected_data_paths[domain_name][dictionary_name]
                    )
                    data_paths = set(domain.extract_data_paths(dictionary_name))
                    self.assertEqual(data_paths, current_data_paths)

    def test_dictionary_get_dictionary_at_data_path(self):
        # Set the paths
        test_resources_dir = os.path.join(resources_dir(), "dictionary")
        ref_kdicj_dir = os.path.join(test_resources_dir, "ref_kdicj")

        # Set the expected outputs
        expected_dictionary_names = {
            "SpliceJunction": {"DNA": "SpliceJunctionDNA"},
            "SpliceJunctionModeling": {"SpliceJunctionDNA": "SNB_SpliceJunctionDNA"},
            "Customer": {
                "Services": "Service",
                "Services/Usages": "Usage",
                "Address": "Address",
                "Services/Usages": "Usage",
            },
            "CustomerExtended": {
                "/City": "City",
                "/Country": "Country",
                "Services": "Service",
                "Address": "Address",
                "/Product": "Product",
                "Services/Usages": "Usage",
            },
        }

        valid_non_table_vars = {
            "SpliceJunction": "Class",
            "SpliceJunctionModeling": "Class",
            "Customer": "Name",
            "CustomerExtended": "Name",
        }

        # Test the method for various dictionary files
        for (
            domain_name,
            expected_dictionary_names_by_data_path,
        ) in expected_dictionary_names.items():
            # Test normal access
            domain = kh.read_dictionary_file(
                os.path.join(ref_kdicj_dir, f"{domain_name}.kdicj")
            )
            for (
                data_path,
                expected_dictionary_name,
            ) in expected_dictionary_names_by_data_path.items():
                with self.subTest(domain_name=domain_name, data_path=data_path):
                    self.assertEqual(
                        domain.get_dictionary_at_data_path(data_path),
                        domain.get_dictionary(expected_dictionary_name),
                    )

            # Test anomalous access
            with self.assertRaises(ValueError):
                domain.get_dictionary_at_data_path("INVALID DATA PATH")
            with self.assertRaises(ValueError):
                domain.get_dictionary_at_data_path("Some/Path")
            first_data_path = list(expected_dictionary_names_by_data_path.keys())[0]
            data_path_parts = first_data_path.split("/")
            with self.assertRaises(ValueError):
                domain.get_dictionary_at_data_path("Some/Path")
            with self.assertRaises(ValueError):
                domain.get_dictionary_at_data_path(
                    f"{valid_non_table_vars[domain_name]}/Path"
                )


class KhiopsCoreSimpleUnitTests(unittest.TestCase):
    """Test simple testable functions in the core package"""

    def test_create_unambiguous_khiops_path(self):
        """Test the create_unambiguous_khiops_path function"""
        expected_outputs = {
            "/normal/path": "/normal/path",
            "./relative/path": "./relative/path",
            "relative/path": os.path.join(".", "relative/path"),
            ".": ".",
            "./": "./",
            ".\\": ".\\",
            "C:/Normal/Path": "C:/Normal/Path",
            "C:\\Normal\\Path": "C:\\Normal\\Path",
            "s3://host/some/path": "s3://host/some/path",
        }
        for path, unambiguous_path in expected_outputs.items():
            self.assertEqual(create_unambiguous_khiops_path(path), unambiguous_path)


class ScenarioWriterRunner(KhiopsRunner):
    """A khiops runner that only generates scenarios to a specific subdirectory"""

    def __init__(self, test_case, root_dir):
        super().__init__()
        self.test_case = test_case
        self.root_dir = root_dir
        self.test_name = None
        self.subtest_name = None
        self.create_ref = False
        self._initialize_khiops_version()

        # Do not write the khiops-python version to the scenarios
        self._write_version = False

    def _initialize_khiops_version(self):
        self._khiops_version = KhiopsVersion("10.1.0")

    @property
    def ref_scenario_dir(self):
        assert self.test_name is not None
        return os.path.join(self.root_dir, self.test_name, "ref")

    @property
    def ref_scenario_path(self):
        assert self.subtest_name is not None
        return os.path.join(self.ref_scenario_dir, f"{self.subtest_name}._kh")

    @property
    def output_scenario_dir(self):
        assert self.test_name is not None
        return os.path.join(self.root_dir, self.test_name, "output")

    @property
    def output_scenario_path(self):
        assert self.subtest_name is not None
        return os.path.join(self.output_scenario_dir, f"{self.subtest_name}._kh")

    @property
    def execution_scenario_path(self):
        assert self.subtest_name is not None
        return os.path.join(self.output_scenario_dir, f"{self.subtest_name}_exec_._kh")

    def _create_scenario_file(self, task):
        return self.execution_scenario_path

    def _write_task_scenario_file(
        self, task, task_args, system_settings, force_ansi_scenario=False
    ):
        """Create the scenario and compare it to a reference"""
        # Create the execution scenario files with the parent method
        scenario_path = super()._write_task_scenario_file(
            task, task_args, system_settings
        )

        # Create the reference if does not exists
        if self.create_ref:
            os.makedirs(self.ref_scenario_dir, exist_ok=True)
            shutil.copy(self.output_scenario_path, self.ref_scenario_path)

        # Copy the execution scenario (which is erased) to compare afterwards
        shutil.copy(self.execution_scenario_path, self.output_scenario_path)

        return scenario_path

    def run(
        self,
        task,
        task_args,
        command_line_options,
        trace=False,
        system_settings=None,
        force_ansi_scenario=False,
        **kwargs,
    ):
        # Call the parent method
        super().run(
            task,
            task_args,
            command_line_options,
            trace=trace,
            system_settings=system_settings,
            force_ansi_scenario=force_ansi_scenario,
        )

        # Mocking the log file contents for detect_data_table_format function
        if (
            task.name
            in ["detect_data_table_format", "detect_data_table_format_with_dictionary"]
            and command_line_options.log_file_path is not None
        ):
            with open(
                command_line_options.log_file_path, "w", encoding="ascii"
            ) as log_file:
                log_file.write(
                    "warning : detect_data_table_format should ignore this\n"
                )
                log_file.write(
                    "File format detected: header line and field separator tabulation\n"
                )

    def _run(
        self,
        tool_name,
        scenario_path,
        command_line_options,
        trace,
    ):
        return 0, "", ""


def resources_dir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")


def cleanup_dir(dir_path, glob_pattern, verbose=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    else:
        file_paths = glob.glob(os.path.join(dir_path, glob_pattern))
        for file_path in file_paths:
            if verbose:
                print(f"Removing {file_path}")
            os.remove(file_path)


def shorten_path(file_path, size):
    return str(Path(*Path(file_path).parts[-size:]))


def default_line_comparator(
    ref_line, output_line, ref_file_path, output_file_path, line_number
):
    if len(ref_line) != len(output_line):
        raise ValueError(
            f"line {line_number} has different length\n"
            + f"Ref file            : {shorten_path(ref_file_path, 5)}\n"
            + f"Output file         : {shorten_path(output_file_path, 5)}\n"
            + f"Ref byte length     : {len(ref_line)}\n"
            + f"Output byte length  : {len(output_line)}"
        )
    else:
        if ref_line != output_line:
            (
                first_diff_pos,
                first_diff_ref_byte,
                first_diff_output_byte,
            ) = find_first_different_byte(ref_line, output_line)

            raise ValueError(
                f"line {line_number} is different\n"
                + f"Ref file            : {shorten_path(ref_file_path, 5)}\n"
                + f"Output file         : {shorten_path(output_file_path, 5)}\n"
                + f"First diff position : {first_diff_pos}\n"
                + f"Ref byte            : {first_diff_ref_byte}\n"
                + f"Output byte         : {first_diff_output_byte}"
            )


PATH_STATEMENTS = [
    "ClassFileName",
    "EvaluationFileName",
    "ImportFileName",
    "InputCoclusteringName",
    "JSONFileName",
    "PostProcessedCoclusteringFileName",
    "ResultFilesDirectory",
    "TargetDataTable.DatabaseName",
    "TargetDatabase.DatabaseFiles.DataTableName",
]


def scenario_line_comparator(
    ref_line, output_line, ref_file_path, output_file_path, line_number
):
    # Special case for paths: Check if there is path field in the line and if it is the
    # case analyze it with a special function
    for path_statement in PATH_STATEMENTS:
        bytes_path_statement = bytes(path_statement, encoding="ascii")
        if bytes_path_statement in ref_line:
            equal_path_statement(ref_line, output_line, line_number)
            return

    default_line_comparator(
        ref_line, output_line, ref_file_path, output_file_path, line_number
    )


def find_first_different_byte(ref_line, output_line):
    first_diff_pos = None
    first_diff_ref_byte = None
    first_diff_output_byte = None
    for i, byte in enumerate(ref_line):
        if i >= len(output_line):
            break
        if byte != output_line[i]:
            first_diff_pos = i
            first_diff_ref_byte = hex(byte)
            first_diff_output_byte = hex(output_line[i])
            break
    return first_diff_pos, first_diff_ref_byte, first_diff_output_byte


def assert_files_equal(
    test_suite, ref_file_path, output_file_path, line_comparator=default_line_comparator
):
    """Portably tests if two files are equal by comparing line-by-line"""
    # Read all lines from the files
    with open(ref_file_path, "rb") as ref_file:
        ref_file_lines = [line.strip() for line in ref_file.read().split(b"\n")]
    with open(output_file_path, "rb") as output_file:
        output_file_lines = [line.strip() for line in output_file.read().split(b"\n")]

    # Check the number of lines
    ref_file_len = len(ref_file_lines)
    output_file_len = len(output_file_lines)
    if ref_file_len != output_file_len:
        test_suite.fail(
            "Files have different number of lines\n"
            + f"Ref file           : {shorten_path(ref_file_path, 5)}\n"
            + f"Output file        : {shorten_path(output_file_path, 5)}\n"
            + f"Ref no. of lines   : {ref_file_len}\n"
            + f"Output no. of lines: {output_file_len}"
        )

    # Compare each line
    paired_lines = list(zip(ref_file_lines, output_file_lines))
    for line_number, (ref_line, output_line) in enumerate(paired_lines):
        line_comparator(
            ref_line, output_line, ref_file_path, output_file_path, line_number
        )


def equal_path_statement(ref_line, output_line, line_number):
    """Compares two Khiops scenario statements containing paths

    The reference in the tests is a Windows path, thus to compare it we transform it if
    necessary.
    """
    ref_tokens = ref_line.strip().split()
    output_tokens = output_line.strip().split()

    if len(ref_tokens) > 2 or len(ref_tokens) == 0:
        print(f"line {line_number} must have either 1 or 2 tokens")
        print("> " + ref_line)
        return False

    if len(output_tokens) > 2 or len(output_tokens) == 0:
        print(f"line {line_number} must have either 1 or 2 tokens")
        print("> " + output_line)
        return False

    if len(ref_tokens) != len(output_tokens):
        print(
            f"line {line_number} in output has different number of tokens: "
            f"{len(output_tokens)} instead of {len(ref_tokens)}"
        )

    if ref_tokens[0] != output_tokens[0]:
        print(f"line {line_number} has different operators")
        print(f"> {ref_tokens[0]} != {output_tokens[0]}")
        return False

    if len(ref_tokens) == 2:
        # The reference is a windows path whereas the output path depends on the system
        ref_path = ref_tokens[1].split(b"\\")
        output_path = output_tokens[1].split(bytes(os.path.sep, encoding="ascii"))

        if ref_path != output_path:
            raise ValueError(
                f"path argument in line {line_number} is different "
                + f"{ref_path} != {output_path}"
            )


class KhiopsCoreVariousTests(unittest.TestCase):
    """Test for small units of core"""

    # Disable SonarQube here because it believes that some versions are IPs
    # sonar: disable

    def test_version_comparisons(self):
        """Test version comparisons"""
        versions = [
            "8.5.0-b10",
            "9.0.1",
            "9.5.1-a1",
            "9.5.1-a2",
            "9.5.1-a.3",
            "9.5.1",
            "10.0.0",
            "10.0.1",
            "10.0.8-a57",
            "10.0.8-b1",
            "10.0.8-b2",
            "10.0.8-b10",
            "10.0.8-b12",
            "10.0.8-rc1",
            "10.0.8",
            "10.1.0",
        ]

        for i, version_str1 in enumerate(versions):
            version1 = KhiopsVersion(version_str1)
            for j, version_str2 in enumerate(versions):
                version2 = KhiopsVersion(version_str2)
                if i < j:
                    self.assertLess(version1, version2)
                    self.assertLessEqual(version1, version2)
                elif i == j:
                    self.assertLessEqual(version1, version2)
                    self.assertEqual(version1, version2)
                    self.assertGreaterEqual(version1, version2)
                else:
                    self.assertGreaterEqual(version1, version2)
                    self.assertGreater(version1, version2)

    # sonar: enable

    def test_invalid_versions(self):
        """Test invalid versions"""
        for version in [
            "a.b.c-4",
            "...",
            ".0.4",
            "ver10.0.0",
            "10",
            "10.0",
            "10.4.0-5.4," "10i.4.0",
            "10.4b.3",
            "10.4.1-b..2",
            "10.4.1.-b.",
            "10.2.@",
            "10.@.2",
            "10.1.2b",
            "10.1.2-b01",
            "10.1.0-beta",
            "10.1.1.1",
            "10.0.8-b",
            "10.01.8",
        ]:
            with self.assertRaises(ValueError):
                KhiopsVersion(version)

    @staticmethod
    def _build_multi_table_dictionary_args():
        resources_directory = KhiopsTestHelper.get_resources_dir()
        dictionaries_dir = os.path.join(resources_directory, "dictionary", "ref_kdic")
        splice_domain = kh.read_dictionary_file(
            os.path.join(dictionaries_dir, "SpliceJunction.kdic")
        )
        monotable_domain = kh.DictionaryDomain()
        monotable_domain.add_dictionary(
            splice_domain.get_dictionary("SpliceJunctionDNA")
        )
        output_directory = os.path.join(
            resources_directory, "dictionary", "output_kdic"
        )
        root_dict_name = "SpliceJunction"
        secondary_table_variable_name = "DNA"
        multi_table_dict_out_path = os.path.join(
            output_directory, "SpliceJunctionTest.kdic"
        )

        return {
            "dictionary_file_path_or_domain": monotable_domain,
            "root_dictionary_name": root_dict_name,
            "secondary_table_variable_name": secondary_table_variable_name,
            "output_dictionary_file_path": multi_table_dict_out_path,
        }

    def test_build_multi_table_dictionary_deprecation(self):
        """Test that `api.build_multi_table_dictionary` raises deprecation warning"""
        in_args = KhiopsCoreVariousTests._build_multi_table_dictionary_args()

        with warnings.catch_warnings(record=True) as warning_list:
            kh.build_multi_table_dictionary(**in_args)

        self.assertEqual(len(warning_list), 1)
        warning = warning_list[0]
        self.assertTrue(issubclass(warning.category, UserWarning))
        warning_message = warning.message
        self.assertEqual(len(warning_message.args), 1)
        message = warning_message.args[0]
        self.assertTrue(
            "'build_multi_table_dictionary'" in message and "deprecated" in message
        )

    def test_build_multi_table_dictionary_behavior(self):
        """Test that the helper function is called with the right parameters"""
        parameter_trace = KhiopsTestHelper.create_parameter_trace()

        in_args = KhiopsCoreVariousTests._build_multi_table_dictionary_args()
        helper_name = "build_multi_table_dictionary_domain"
        KhiopsTestHelper.wrap_with_parameter_trace(
            "khiops.core.api", helper_name, parameter_trace
        )
        with self.assertWarns(UserWarning):
            kh.build_multi_table_dictionary(**in_args)
        # Test that at least one trace has been created, so that the assertions can fail
        self.assertTrue(any(True for _ in parameter_trace.items()))
        for _, function_parameters in parameter_trace.items():
            # Test that at least a traced function has been called
            self.assertTrue(any(True for _ in function_parameters.items()))
            for function_name, parameters in function_parameters.items():
                # Test that the helper has been called
                self.assertEqual(function_name, helper_name)
                first_call_parameters = parameters[0]
                args = first_call_parameters["args"]

                # Test that the parameters have been passed
                self.assertEqual(args[1], in_args["root_dictionary_name"])
                self.assertEqual(args[2], in_args["secondary_table_variable_name"])

                # Test that the first argument passed is a DictionaryDomain
                domain = args[0]
                self.assertTrue(isinstance(domain, kh.DictionaryDomain))

                # Shallowly test that the domain passed to the helper reflects
                # the source dictionary
                # N.B. We do not test the function for reading a dictionary file
                # into a domain here
                self.assertEqual(len(domain.dictionaries), 1)
                dictionary = domain.dictionaries[0]
                self.assertEqual(dictionary.name, "SpliceJunctionDNA")
                self.assertEqual(dictionary.key, ["SampleId"])

    def test_scenario_generation(self):
        """Test the scenario generation from template and arguments"""
        templates = {
            "raw_lines": """
            // A scenario comment
            SomeStatement

            AnotherStatementWithAValue a_value
            """,
            "opt_argument": """
            __OPT__
            __opt_argument__
            A.Statement
            Another.StatementWithValue some_value
            __END_OPT__
            """,
            "single_type_list_argument": """
            __LIST__
            __list_argument__
            Prologue.Statement
            Another.Prologue.Statement
            ListArgument.InsertItemAfter
            ListArgument.SingleTupleValue
            __END_LIST__
            """,
            "tuple_type_list_argument": """
            __LIST__
            __list_argument__
            ListArgument.InsertItemAfter
            ListArgument.FirstTupleValue
            ListArgument.SecondTupleValue
            ListArgument.ThirdTupleValue
            __END_LIST__
            """,
            "dict_argument": """
            __DICT__
            __dict_argument__
            Prologue.Statement
            DictArgument.Key
            DictArgument.Value
            __END_DICT__
            """,
        }

        arguments = {
            ("raw_lines", "default"): {},
            ("opt_argument", "true"): {"__opt_argument__": "true"},
            ("opt_argument", "false"): {"__opt_argument__": "false"},
            ("single_type_list_argument", "default"): {
                "__list_argument__": ["Val1", "Val2"]
            },
            ("tuple_type_list_argument", "default"): {
                "__list_argument__": [
                    ("Val11", "Val12", "Val13"),
                    ("Val21", "Val22", "Val23"),
                ]
            },
            ("dict_argument", "default"): {
                "__dict_argument__": {"Key1": "Val1", "Key2": "Val2"}
            },
            ("dict_argument", "list_input"): {"__dict_argument__": ["Key1", "Key2"]},
        }

        expected_scenarios = {
            (
                "raw_lines",
                "default",
            ): """
            // A scenario comment
            SomeStatement

            AnotherStatementWithAValue a_value
            """,
            (
                "opt_argument",
                "true",
            ): """
            A.Statement
            Another.StatementWithValue some_value
            """,
            ("opt_argument", "false"): "",
            (
                "single_type_list_argument",
                "default",
            ): """
            Prologue.Statement
            Another.Prologue.Statement
            ListArgument.InsertItemAfter
            ListArgument.SingleTupleValue Val1
            ListArgument.InsertItemAfter
            ListArgument.SingleTupleValue Val2
            """,
            (
                "tuple_type_list_argument",
                "default",
            ): """
            ListArgument.InsertItemAfter
            ListArgument.FirstTupleValue Val11
            ListArgument.SecondTupleValue Val12
            ListArgument.ThirdTupleValue Val13
            ListArgument.InsertItemAfter
            ListArgument.FirstTupleValue Val21
            ListArgument.SecondTupleValue Val22
            ListArgument.ThirdTupleValue Val23
            """,
            (
                "dict_argument",
                "default",
            ): """
            Prologue.Statement
            DictArgument.Key Key1
            DictArgument.Value Val1
            DictArgument.Key Key2
            DictArgument.Value Val2
            """,
            (
                "dict_argument",
                "list_input",
            ): """
            Prologue.Statement
            DictArgument.Key Key1
            DictArgument.Value true
            DictArgument.Key Key2
            DictArgument.Value true
            """,
        }

        for (
            template_name,
            argument_set,
        ), expected_scenario in expected_scenarios.items():
            with self.subTest(template_name=template_name, argument_set=argument_set):
                # Dedent the template and the expected scenario
                template_code = textwrap.dedent(templates[template_name]).lstrip()
                expected_scenario = textwrap.dedent(expected_scenario).lstrip()

                # Detemplatize the scenario with the fixture arguments
                stream = io.BytesIO()
                writer = KhiopsOutputWriter(stream)
                scenario = ConfigurableKhiopsScenario(template_code)
                scenario.write(writer, arguments[template_name, argument_set])
                output_scenario = stream.getvalue().decode("ascii").replace("\r", "")

                # Compare the output scenario and the expected fixture
                self.assertEqual(output_scenario, expected_scenario)

    def test_invalid_templates(self):
        # Define the fixtures
        templates = {
            "invalid_keyword": """
            __SECTION__
            Statement
            __END__SECTION__
            """,
            "invalid_statement": """
            __OPT__
            __opt_argument__
            invalid_statement.NotValid
            __END_OPT__
            """,
            "opt_no_arg": """
            __OPT__
            Statement
            __END_OPT__
            """,
            "opt_no_end": """
            __OPT__
            Statement
            """,
            "list_no_arg": """
            __LIST__
            Argument.InsertItemAfter
            Argument.Value
            __END_LIST__
            """,
            "list_bad_arg": """
            __LIST__
            list_argument
            Argument.InsertItemAfter
            Argument.Value
            __END_LIST__
            """,
            "list_no_insert": """
            __LIST__
            __list_argument__
            Argument.NoInsert
            Argument.Value
            __END_LIST__
            """,
            "list_too_few_lines": """
            __LIST__
            __list_argument__
            __END_LIST__
            """,
            "list_no_end": """
            __LIST__
            __list_argument__
            """,
            "list_no_value": """
            __LIST__
            __list_argument__
            Prologue.FirstStatement
            Argument.InsertItemAfter
            __END_LIST__
            """,
            "dict_no_arg": """
            __DICT__
            Argument.Key
            Argument.Value
            __END_DICT__
            """,
            "dict_no_key": """
            __DICT__
            __dict_value__
            Argument.KeyValue
            Argument.Value
            __END_DICT__
            """,
            "dict_too_few_lines": """
            __DICT__
            __dict_argument__
            __END_DICT__
            """,
            "dict_no_end": """
            __DICT__
            __dict_argument__
            """,
        }
        # pylint: disable=line-too-long
        error_msgs = {
            "invalid_keyword": "Expected keyword __DICT__, __OPT__ or __LIST__",
            "invalid_statement": "Statement must contain only alphabetic characters and '.'",
            "opt_no_arg": "__OPT__ template parameter name does not conform to the __param__ notation",
            "opt_no_end": "__OPT__ section has no matching __END_OPT__",
            "list_no_arg": "__LIST__ template parameter name does not conform to the __param__ notation",
            "list_bad_arg": "__LIST__ template parameter name does not conform to the __param__ notation",
            "list_no_insert": "__LIST__ section does not contain list statement ending with '.InsertItemAfter'",
            "list_too_few_lines": "__LIST__ section must have at least 3 statements",
            "list_no_end": "__LIST__ section has no matching __END_LIST__",
            "list_no_value": "__LIST__ section does not contain any value statement",
            "dict_no_arg": "__DICT__ template parameter name does not conform to the __param__ notation",
            "dict_no_key": "__DICT__ key statement must end with '.Key'",
            "dict_too_few_lines": "__DICT__ section must have at least 2 statements",
            "dict_no_end": "__DICT__ section has no matching __END_DICT__",
        }
        # pylint: enable=line-too-long

        # Test that the templates fail with the proper message
        for template_name, template_code in templates.items():
            with self.subTest(template_name=template_name, template_code=template_code):
                with self.assertRaisesRegex(ValueError, error_msgs[template_name]):
                    ConfigurableKhiopsScenario(template_code)

    def test_raise_exception_on_error_case_without_a_message(self):
        with self.assertRaises(KhiopsRuntimeError) as context:
            with MockedRunnerContext(
                create_mocked_raw_run(
                    stdout=False,  # ask for an empty stdout
                    stderr=False,  # ask for an empty stderr
                    return_code=9,  # non-zero error code
                )
            ):
                kh.train_predictor(
                    "/tmp/Iris.kdic",
                    dictionary_name="Iris",
                    data_table_path="/tmp/Iris.txt",
                    target_variable="Class",
                    results_dir="/tmp",
                    trace=True,
                )
        expected_msg = (
            "khiops execution had errors (return code 9) but no message is available"
        )
        output_msg = str(context.exception)
        self.assertEqual(output_msg, expected_msg)


if __name__ == "__main__":
    unittest.main()
