######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Miscellaneous utility tools

.. warning::
    Entry point functions in this module use `sys.exit`. They are not designed to be
    called from another program or python shell.
"""
import argparse
import os
import re
import sys

import khiops.core as pk

# Note: We dont include these tools in coverage


def pk_status_entry_point():  # pragma: no cover
    """Entry point of the pk-status command"""
    try:
        pk.get_runner().print_status()
        print("\nKhiops Python installation OK")
        sys.exit(0)
    except pk.PyKhiopsEnvironmentError as error:
        print(
            f"Khiops Python backend ERROR: {error}"
            "\nCheck https://www.khiops.com to install the Khiops app in your computer"
        )
        sys.exit(1)


def convert_pk10_entry_point():  # pragma: no cover
    """Entry point of the convert-pk10 script"""
    parser = argparse.ArgumentParser(
        prog="convert-pk10",
        formatter_class=argparse.RawTextHelpFormatter,
        description="converts a khiops 9 script to a khiops 10 script",
    )
    parser.add_argument(
        "input_path", metavar="INPYFILE", help="input python script path"
    )
    parser.add_argument(
        "output_path", metavar="OUTPYFILE", help="output python script path"
    )
    args = parser.parse_args()

    # Call the main function
    convert_ok = convert_pk10(args.input_path, args.output_path)
    if convert_ok:
        sys.exit(0)
    else:
        sys.exit(1)


def convert_pk10(input_path, output_path):  # pragma: no cover
    """Main function for the convert-pk10 script"""

    # Sanity check
    if os.path.abspath(output_path) == os.path.abspath(input_path):
        print("error: input and output paths are the same")
        return False

    # Setup the translation objects
    import_line = "import khiops as "
    line_splitter = re.compile(f"({_build_keywords_regex()})")
    translator_dict = _build_translator_dict()
    import_line_changed = False
    keywords_changed = {}

    # Translate every line of the input file to the output file
    with open(input_path) as script_file, open(output_path, "w") as out_file:
        for line in script_file:
            if line.startswith(import_line):
                tokens = line.split(import_line)
                print(f"from khiops import core as {tokens[1]}\n", file=out_file)
                import_line_changed = True
            elif line_splitter.search(line):
                tokens = line_splitter.split(line)
                for token in tokens:
                    if token in translator_dict:
                        print(translator_dict[token], end="", file=out_file)
                        if token in keywords_changed:
                            keywords_changed[token] += 1
                        else:
                            keywords_changed[token] = 1
                    else:
                        print(token, end="", file=out_file)
            else:
                print(line, end="", file=out_file)
        out_file.close()

    print(f"file {output_path} written")
    if import_line_changed:
        print("import line changed")

    problematic_keywords = [
        "getKhiopsBinDir",
        "getKhiopsSampleDir",
        "getMaxCoreNumber",
        "getMemoryLimit",
        "getScriptHeaderLines",
        "getTempDir",
        "dictionaryDomain",
    ]
    if keywords_changed:
        print(f"changed {len(keywords_changed)} keywords, ocurrences:")
        for keyword, change_number in keywords_changed.items():
            print(
                f"{keyword} -> {translator_dict[keyword]}: {change_number}",
                end="",
            )
            if keyword in problematic_keywords:
                print(" *** check these changes ***")
            else:
                print("")
    return True


PYKHIOPS9_KEYWORDS = [
    "CCCell",
    "CCCluster",
    "CCDimension",
    "CCPart",
    "CCPartInterval",
    "CCPartValueGroup",
    "CCValue",
    "TestKhiopsCoclusteringResults",
    "TestKhiopsDictionaryDomain",
    "TestKhiopsResults",
    "__KhiopsBinDir",
    "__KhiopsSampleDir",
    "__MaxCoreNumber",
    "__MemoryLimit",
    "__ScriptHeaderLines",
    "__TempDir",
    "_dictionariesByName",
    "_dictionaryClusters",
    "_dictionaryDimensions",
    "_dictionaryParts",
    "_predictorsPerformanceByName",
    "_trainedPredictorsByName",
    "_variableBlocksByName",
    "_variablesByName",
    "_variablesPairsStatisticsByName",
    "_variablesStatisticsByName",
    "addDictionary",
    "addValue",
    "addVariable",
    "additionalDataTables",
    "allConstructionRules",
    "basicRunKhiopsTool",
    "bivariatePreparationReport",
    "blockName",
    "buildClusterVariable",
    "buildDictionaryFromDataTable",
    "buildDistanceVariables",
    "buildFrequencyVariables",
    "buildTransferredDictionary",
    "ccChildCluster1",
    "ccChildCluster2",
    "ccLeafPart",
    "ccParentCluster",
    "ccRootCluster",
    "cellFrequencies",
    "cellIds",
    "cellInterests",
    "cellPartIndexes",
    "cellTargetFrequencies",
    "cells",
    "checkDatabase",
    "classificationLiftCurves",
    "classificationTargetValues",
    "cluster",
    "clusters",
    "coclusteringReport",
    "confusionMatrix",
    "constructedNumber",
    "constructionCost",
    "constructionRules",
    "dataCost",
    "dataGrid",
    "defaultGroup",
    "deltaLevel",
    "derivationRule",
    "dictionaryDomain",
    "discretizationMethod",
    "evaluatePredictor",
    "evaluatedVariables",
    "evaluationReport",
    "evaluationType",
    "exportDictionaryAsJSON",
    "exportKhiopsDictionaryFile",
    "extractKeysFromDataTable",
    "fieldSeparator",
    "fillTestDatabaseSettings",
    "frequencyVariable",
    "fullType",
    "getClassifierLiftCurve",
    "getCluster",
    "getDictionary",
    "getDimension",
    "getKhiopsBinDir",
    "getKhiopsCoclusteringInfo",
    "getKhiopsInfo",
    "getKhiopsSampleDir",
    "getMaxCoreNumber",
    "getMemoryLimit",
    "getPart",
    "getPredictor",
    "getPredictorPerformance",
    "getPyKhiopsDir",
    "getRegressorRecCurve",
    "getScriptHeaderLines",
    "getTempDir",
    "getToolInfo",
    "getValue",
    "getVariable",
    "getVariableBlock",
    "getVariablePairStatistics",
    "getVariableStatistics",
    "groupTargetValue",
    "groupingMethod",
    "headerLine",
    "hierarchicalLevel",
    "hierarchicalRank",
    "informativeVariables",
    "informativeVariablesOnly",
    "initDetails",
    "initHierarchy",
    "initPartition",
    "initSummary",
    "initialDimensions",
    "initialParts",
    "inputValueFrequencies",
    "inputValues",
    "instances",
    "isDefaultPart",
    "isDetailed",
    "isEmpty",
    "isKeyVariable",
    "isLeaf",
    "isLeftOpen",
    "isMissing",
    "isNative",
    "isReferenceRule",
    "isRelational",
    "isRightOpen",
    "isSupervised",
    "keepInitialCategoricalVariables",
    "keepInitialNumericalVariables",
    "learningTask",
    "level1",
    "level2",
    "listCells",
    "listClusters",
    "listDimensions",
    "listParts",
    "listValues",
    "lowerBound",
    "mainTargetValue",
    "mapPredictor",
    "maxCellNumber",
    "maxEvaluatedVariableNumber",
    "maxGroupNumber",
    "maxIntervalNumber",
    "maxMessageNumber",
    "maxPartNumbers",
    "maxPreservedInformation",
    "maxSelectedVariableNumber",
    "maxVariableNumber",
    "metaData",
    "minGroupFrequency",
    "minIntervalFrequency",
    "minOptimizationTime",
    "missingNumber",
    "modeFrequency",
    "modelingReport",
    "name1",
    "name2",
    "nbPredictor",
    "needAnnotationReport",
    "nullCost",
    "nullModelConstructionCost",
    "nullModelDataCost",
    "nullModelPreparationCost",
    "objectType",
    "onlyPairsWith",
    "outputAdditionalDataTables",
    "outputFieldSeparator",
    "outputHeaderLine",
    "outputScript",
    "pairNumber",
    "parentCluster",
    "partInterests",
    "partTargetFrequencies",
    "partType",
    "partitionType",
    "parts",
    "parts1",
    "parts2",
    "predictorFamilies",
    "predictorTypes",
    "predictorsPerformance",
    "preparationCost",
    "preparationReport",
    "prepareCoclusteringDeployment",
    "preparedName",
    "rankMae",
    "rankNlpd",
    "rankRmse",
    "readAnalysisResultsFile",
    "readCoclusteringResultsFile",
    "readDictionaryFile",
    "readKhiopsCoclusteringJsonFile",
    "readKhiopsDictionaryJsonFile",
    "readKhiopsJsonFile",
    "recodeBivariateVariables",
    "recodeCategoricalVariables",
    "recodeNumericalVariables",
    "regressionRecCurves",
    "removeDictionary",
    "removeKey",
    "removeVariable",
    "reportType",
    "resultsPrefix",
    "runKhiops",
    "runKhiopsCoclustering",
    "runKhiopsTool",
    "samplePercentage",
    "samplingMode",
    "searchReplace",
    "selectedVariables",
    "selectionValue",
    "selectionVariable",
    "setKhiopsBinDir",
    "setKhiopsSampleDir",
    "setMaxCoreNumber",
    "setMemoryLimit",
    "setScriptHeaderLines",
    "setTempDir",
    "shortDescription",
    "simplifyCoclustering",
    "snbPredictor",
    "sortDataTable",
    "sortVariables",
    "stdDev",
    "strName",
    "strValue",
    "structureType",
    "targetParts",
    "targetStatsMax",
    "targetStatsMean",
    "targetStatsMin",
    "targetStatsMissingNumber",
    "targetStatsMode",
    "targetStatsModeFrequency",
    "targetStatsStdDev",
    "targetValueFrequencies",
    "targetValues",
    "targetVariable",
    "testEvaluationReport",
    "trainCoclustering",
    "trainEvaluationReport",
    "trainPredictor",
    "trainRecoder",
    "trainedPredictors",
    "transferDatabase",
    "treeNumber",
    "univariatePredictorNumber",
    "upperBound",
    "useAllVariables",
    "valueGrouping",
    "variableBlock",
    "variableBlocks",
    "variableNumbers",
    "variableTypes",
    "variablesPairsStatistics",
    "variablesPrefix",
    "variablesStatistics",
    "writeAnnotation",
    "writeAnnotationHeaderLine",
    "writeAnnotationLine",
    "writeAnnotations",
    "writeBounds",
    "writeCells",
    "writeCoclusteringStats",
    "writeComposition",
    "writeCompositions",
    "writeDimensionHeaderLine",
    "writeDimensionLine",
    "writeDimensions",
    "writeHierarchies",
    "writeHierarchy",
    "writeHierarchyHeaderLine",
    "writeHierarchyLine",
    "writeHierarchyStructureReport",
    "writeHierarchyStructureReportFile",
    "writeLine",
    "writeReport",
    "writeReportDetails",
    "writeReportFile",
    "writeReportHeaderLine",
    "writeReportLine",
]

snake_caser_step1 = re.compile("(.)([A-Z][a-z]+)")
snake_caser_step2 = re.compile("([a-z0-9])([A-Z])")


def _camel_to_snake(name):  # pragma: no cover
    """Transform camelCase to snake_case"""
    tmp = snake_caser_step1.sub(r"\1\2", name)
    return snake_caser_step2.sub(r"\1_\2", tmp).lower()


def _build_translator_dict():  # pragma: no cover
    """Builds the translator dictionary from Khiops 9 keywords to Khiops 10"""
    # Transform to snake_case the list of keywords
    translator_dict = {
        keyword: _camel_to_snake(keyword) for keyword in PYKHIOPS9_KEYWORDS
    }

    # Ad-hoc changes in the API
    translator_dict.update(
        {
            "CCCell": "CoclusteringCell",
            "CCCluster": "CoclusteringCluster",
            "CCDimension": "CoclusteringDimension",
            "CCPart": "CoclusteringDimensionPart",
            "CCPartInterval": "CoclusteringDimensionPartInterval",
            "CCPartValueGroup": "CoclusteringDimensionPartValueGroup",
            "CCValue": "CoclusteringDimensionPartValue",
            "__KhiopsBinDir": "runner.khiops_bin_dir",
            "__KhiopsSampleDir": "runner.samples_dir",
            "__MaxCoreNumber": "runner.max_cores",
            "__MemoryLimit": "runner.max_memory_mb",
            "__ScriptHeaderLines": "runner.scenario_prologue",
            "__TempDir": "runner.khiops_temp_dir",
            "buildTransferredDictionary": "build_deployment_dictionary",
            "ccChildCluster1": "child_cluster1",
            "ccChildCluster2": "child_cluster2",
            "ccLeafPart": "leaf_part",
            "ccParentCluster": "parent_cluster",
            "ccRootCluster": "root_cluster",
            "cells": "cell_number",
            "cluster": "name",
            "clusters": "cluster_number",
            "constructedNumber": "max_constructed_variables",
            "getKhiopsBinDir": "runner.khiops_bin_dir + str",
            "getKhiopsSampleDir": "runner.samples_dir + str",
            "getMaxCoreNumber": "runner.max_cores + int",
            "getMemoryLimit": "runner.max_memory_mb + int",
            "getPyKhiopsDir": "get_khiops_core_dir + str",
            "getScriptHeaderLines": "runner.scenario_prologue + str",
            "getTempDir": "runner.khiops_temp_dir + str",
            "initialDimensions": "initial_dimension_number",
            "initialParts": "initial_part_number",
            "instances": "instance_number",
            "listCells": "cells",
            "listClusters": "clusters",
            "listDimensions": "dimensions",
            "listParts": "parts",
            "listValues": "values",
            "maxCellNumber": "max_cells",
            "maxEvaluatedVariableNumber": "max_evaluated_variables",
            "maxGroupNumber": "max_groups",
            "maxIntervalNumber": "max_intervals",
            "maxMessageNumber": "max_messages",
            "maxSelectedVariableNumber": "max_selected_variables",
            "needAnnotationReport": "needs_annotation_report",
            "outputScript": "output_scenario_path",
            "pairNumber": "max_pairs",
            "parentCluster": "parent_cluster_name",
            "parts": "part_number",
            "parts1": "part_number1",
            "parts2": "part_number2",
            "setKhiopsBinDir": "runner.khiops_bin_dir = ",
            "setKhiopsSampleDir": "runner.samples_dir = ",
            "setMaxCoreNumber": "runner.max_cores = ",
            "setMemoryLimit": "runner.max_memory_mb = ",
            "setScriptHeaderLines": "runner.scenario_prologue = ",
            "setTempDir": "runner.khiops_temp_dir = ",
            "targetParts": "target_part_number",
            "targetStatStdDev": "target_stats_std_dev",
            "transferDatabase": "deploy_model",
            "treeNumber": "max_trees",
        }
    )

    return translator_dict


def _build_keywords_regex():  # pragma: no cover
    """Builds a regular expression to match the keywords

    Takes into account substrings: if word, wordLong and wordLongest are present in
    the keyword list then it generates a single regular expression (word[a-zA-Z]*)

    It returns all keyword regexps joined with an "or"
    """
    khiops9_keywords_regexes = []
    current_keyword_index = 0
    i = 1
    num_keywords = len(PYKHIOPS9_KEYWORDS)
    while i < num_keywords:
        current_keyword = PYKHIOPS9_KEYWORDS[current_keyword_index]
        while i < num_keywords and PYKHIOPS9_KEYWORDS[i].startswith(current_keyword):
            i += 1
        if i == current_keyword_index + 1:
            khiops9_keywords_regexes.append(current_keyword)
        else:
            khiops9_keywords_regexes.append(f"{current_keyword}[a-zA-Z]*")
        current_keyword_index = i
    return "|".join(khiops9_keywords_regexes)
