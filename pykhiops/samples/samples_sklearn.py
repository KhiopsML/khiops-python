######################################################################################
# Copyright (c) 2018 - 2022 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""pyKhiops sklearn submodule samples
The functions in this script demonstrate the basic use of the sklearn submodule of the
pyKhiops library.
"""
import argparse
import os
import pickle
from os import path

import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer

# isort: off
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# isort: on
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import pykhiops
from pykhiops import core as pk
from pykhiops.sklearn import (
    KhiopsClassifier,
    KhiopsCoclustering,
    KhiopsEncoder,
    KhiopsRegressor,
)

# Disable PEP8 variable names because of scikit-learn X,y conventions
# To capture invalid-names other than X,y run:
#   pylint --disable=all --enable=invalid-names estimators.py
# pylint: disable=invalid-name


def khiops_classifier():
    """Trains a `.KhiopsClassifier`"""
    # Load the dataset into a pandas dataframe
    adult_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    adult_df = pd.read_csv(adult_path, sep="\t")

    # Split the whole dataframe into train and test (70%-30%)
    adult_train_df, adult_test_df = train_test_split(
        adult_df, test_size=0.3, random_state=1
    )

    # Split the dataset into:
    # - the X feature table
    # - the y target vector ("class" column)
    X_train = adult_train_df.drop(["class"], axis=1)
    X_test = adult_test_df.drop(["class"], axis=1)
    y_train = adult_train_df["class"]
    y_test = adult_test_df["class"]

    # Create the classifier object
    pkc = KhiopsClassifier()

    # Train the classifier
    pkc.fit(X_train, y_train)

    # Predict the classes on the test dataset
    y_test_pred = pkc.predict(X_test)
    print("Predicted classes (first 10):")
    print(y_test_pred[0:10])
    print("---")

    # Predict the class probabilities on the test dataset
    y_test_probas = pkc.predict_proba(X_test)
    print(f"Class order: {pkc.classes_}")
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[0:10])
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_probas[:, 1])
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")


def khiops_classifier_multiclass():
    """Trains a multiclass `.KhiopsClassifier`"""
    # Load the dataset into a pandas dataframe
    iris_path = path.join(pk.get_samples_dir(), "Iris", "Iris.txt")
    iris_df = pd.read_csv(iris_path, sep="\t")

    # Split the whole dataframe into train and test (70%-30%)
    iris_train_df, iris_test_df = train_test_split(
        iris_df, test_size=0.3, random_state=1
    )

    # Split the dataset into:
    # - the X feature table
    # - the y target vector ("Class" column)
    X_train = iris_train_df.drop("Class", axis=1)
    X_test = iris_test_df.drop("Class", axis=1)
    y_train = iris_train_df["Class"]
    y_test = iris_test_df["Class"]

    # Create the classifier object
    pkc = KhiopsClassifier()

    # Train the classifier
    pkc.fit(X_train, y_train)

    # Predict the classes on the test dataset
    y_test_pred = pkc.predict(X_test)
    print("Predicted classes (first 10):")
    print(y_test_pred[:10])
    print("---")

    # Predict the class probabilities on the test datasets
    y_test_probas = pkc.predict_proba(X_test)
    print(f"Class order: {pkc.classes_}")
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[:10])
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_probas, multi_class="ovr")
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")


def khiops_regressor():
    """Trains a `.KhiopsRegressor`"""
    # Load the dataset into a pandas dataframe
    adult_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    adult_df = pd.read_csv(adult_path, sep="\t")

    # Split the whole dataframe into train and test (40%-60% for speed)
    adult_train_df, adult_test_df = train_test_split(
        adult_df, test_size=0.6, random_state=1
    )

    # Split the dataset into:
    # - the X feature table
    # - the y target vector ("age" column)
    X_train = adult_train_df.drop("age", axis=1)
    X_test = adult_test_df.drop("age", axis=1)
    y_train = adult_train_df["age"]
    y_test = adult_test_df["age"]

    # Create the regressor object
    pkr = KhiopsRegressor()

    # Train the regressor
    pkr.fit(X_train, y_train)

    # Predict the values on the test dataset
    y_test_pred = pkr.predict(X_test)
    print("Predicted values for 'age' (first 10):")
    print(y_test_pred[:10])
    print("---")

    # Evaluate R2 and MAE metrics on the test dataset
    test_r2 = metrics.r2_score(y_test, y_test_pred)
    test_mae = metrics.mean_absolute_error(y_test, y_test_pred)
    print(f"Test R2  = {test_r2}")
    print(f"Test MAE = {test_mae}")


def khiops_classifier_multitable():
    """Trains a `.KhiopsClassifier` on a multi-table dataset"""
    # Load the root table of the dataset into a pandas dataframe
    accidents_dataset_path = path.join(pk.get_samples_dir(), "AccidentsSummary")
    accidents_df = pd.read_csv(
        path.join(accidents_dataset_path, "Accidents.txt"),
        sep="\t",
        encoding="latin1",
    )

    # Split the root dataframe into train and test
    accidents_train_df, accidents_test_df = train_test_split(
        accidents_df, test_size=0.3, random_state=1
    )

    # Obtain the main X feature table and the y target vector ("Class" column)
    y_train = accidents_train_df["Gravity"]
    y_test = accidents_test_df["Gravity"]
    X_train_main = accidents_train_df.drop("Gravity", axis=1)
    X_test_main = accidents_test_df.drop("Gravity", axis=1)

    # Load the secondary table of the dataset into a pandas dataframe
    vehicles_df = pd.read_csv(
        path.join(accidents_dataset_path, "Vehicles.txt"), sep="\t"
    )

    # Split the secondary dataframe with the keys of the splitted root dataframe
    X_train_ids = X_train_main["AccidentId"].to_frame()
    X_test_ids = X_test_main["AccidentId"].to_frame()
    X_train_secondary = X_train_ids.merge(vehicles_df, on="AccidentId")
    X_test_secondary = X_test_ids.merge(vehicles_df, on="AccidentId")

    # Create the dataset multitable specification for the train/test split
    # We specify each table with a name and a tuple (dataframe, key_columns)
    X_train_dataset = {
        "main_table": "Accidents",
        "tables": {
            "Accidents": (X_train_main, "AccidentId"),
            "Vehicles": (X_train_secondary, ["AccidentId", "VehicleId"]),
        },
    }
    X_test_dataset = {
        "main_table": "Accidents",
        "tables": {
            "Accidents": (X_test_main, "AccidentId"),
            "Vehicles": (X_test_secondary, ["AccidentId", "VehicleId"]),
        },
    }

    # Train the classifier (by default it analyzes 100 multi-table features)
    pkc = KhiopsClassifier()
    pkc.fit(X_train_dataset, y_train)

    # Predict the class on the test dataset
    y_test_pred = pkc.predict(X_test_dataset)
    print("Predicted classes (first 10):")
    print(y_test_pred[:10])
    print("---")

    # Predict the class probability on the test dataset
    y_test_probas = pkc.predict_proba(X_test_dataset)
    print(f"Class order: {pkc.classes_}")
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[:10])
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_probas[:, 1])
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")


def khiops_encoder():
    """Trains a `.KhiopsEncoder`

    The Khiops encoder is a supervised feature encoder. It discretizes numerical
    features and groups categorical features in a way that the resulting interval/groups
    have the highest class-purity.

    .. note::
        For simplicity we train from the whole dataset. To assess the performance one
        usually splits the dataset into train and test subsets.
    """
    # Load the dataset into a pandas dataframe
    iris_path = path.join(pk.get_samples_dir(), "Iris", "Iris.txt")
    iris_df = pd.read_csv(iris_path, sep="\t")

    # Train the model with the whole dataset
    X = iris_df.drop(["Class"], axis=1)
    y = iris_df["Class"]

    # Create the encoder object
    pke = KhiopsEncoder()
    pke.fit(X, y)

    # Transform the training dataset
    X_transformed = pke.transform(X)

    # Print both the original and transformed features
    print("Original:")
    print(X.head(10))
    print("---")
    print("Encoded feature names:")
    print(pke.feature_names_out_)
    print("Encoded data:")
    print(X_transformed[:10])
    print("---")


def khiops_encoder_multitable():
    """Trains a `.KhiopsEncoder` on a multi-table dataset"""
    # Load the root table of the dataset into a pandas dataframe
    accidents_dataset_path = path.join(pk.get_samples_dir(), "AccidentsSummary")
    accidents_df = pd.read_csv(
        path.join(accidents_dataset_path, "Accidents.txt"),
        sep="\t",
        encoding="latin1",
    )

    # Obtain the root X feature table and the y target vector ("Class" column)
    y = accidents_df["Gravity"]
    X_main = accidents_df.drop("Gravity", axis=1)

    # Load the secondary table of the dataset into a pandas dataframe
    X_secondary = pd.read_csv(
        path.join(accidents_dataset_path, "Vehicles.txt"), sep="\t"
    )

    # Create the dataset multitable specification for the train/test split
    # We specify each table with a name and a tuple (dataframe, key_columns)
    X_dataset = {
        "main_table": "Accidents",
        "tables": {
            "Accidents": (X_main, "AccidentId"),
            "Vehicles": (X_secondary, ["AccidentId", "VehicleId"]),
        },
    }

    # Create the KhiopsEncoder with 10 additional multitable features and fit it
    pke = KhiopsEncoder(n_features=10)
    pke.fit(X_dataset, y)

    # Transform the train dataset
    print("Encoded feature names:")
    print(pke.feature_names_out_)
    print("Encoded data:")
    print(pke.transform(X_dataset)[:10])


def khiops_encoder_pipeline_with_hgbc():
    """Chains a `.KhiopsEncoder` in a pipeline with a HistGradientBoostingClassifier"""
    # Load the dataset into a pandas dataframe
    adult_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    adult_df = pd.read_csv(adult_path, sep="\t")

    # Split the whole dataframe into train and test (70%-30%)
    adult_train_df, adult_test_df = train_test_split(
        adult_df, test_size=0.3, random_state=1
    )

    # Split the dataset into:
    # - the X feature table
    # - the y target vector ("class" column)
    X_train = adult_train_df.drop("class", axis=1)
    X_test = adult_test_df.drop("class", axis=1)
    y_train = adult_train_df["class"]
    y_test = adult_test_df["class"]

    # Create the pipeline and fit it. Steps:
    # - The khiops supervised column encoder, generates a full-categorical table
    # - One hot encoder in all columns
    # - Train the HGB classifier
    pipe_steps = [
        ("khiops_enc", KhiopsEncoder()),
        ("onehot_enc", ColumnTransformer([], remainder=OneHotEncoder(sparse=False))),
        ("hgb_clf", HistGradientBoostingClassifier()),
    ]
    pipe = Pipeline(pipe_steps)
    pipe.fit(X_train, y_train)

    # Predict the classes on the test dataset
    y_test_pred = pipe.predict(X_test)
    print("Predicted classes (first 10):")
    print(y_test_pred[:10])
    print("---")

    # Predict the class probabilities on the test dataset
    y_test_probas = pipe.predict_proba(X_test)
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[:10])
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_probas[:, 1])
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")


def khiops_coclustering():
    """Trains a `.KhiopsCoclustering` estimator"""
    # Load the secondary table of the dataset into a pandas dataframe
    splice_dataset_path = path.join(pk.get_samples_dir(), "SpliceJunction")
    splice_dna_X = pd.read_csv(
        path.join(splice_dataset_path, "SpliceJunctionDNA.txt"), sep="\t"
    )

    # Train with only 70% of data (for speed in this example)
    X, _ = train_test_split(splice_dna_X, test_size=0.3, random_state=1)

    # Create the KhiopsCoclustering instance
    pkcc = KhiopsCoclustering()

    # Train the model with the whole dataset
    pkcc.fit(X, id_column="SampleId")

    # Predict the clusters in some instances
    X_clusters = pkcc.predict(X)
    print("Predicted clusters (first 10)")
    print(X_clusters[:10])
    print("---")


def khiops_classifier_pickle():
    """Shows the serialization and deserialization of a `.KhiopsClassifier`"""
    # Load the dataset into a pandas dataframe
    iris_path = path.join(pk.get_samples_dir(), "Iris", "Iris.txt")
    iris_df = pd.read_csv(iris_path, sep="\t")

    # Train the model with the whole dataset
    X = iris_df.drop(["Class"], axis=1)
    y = iris_df["Class"]
    pkc = KhiopsClassifier()
    pkc.fit(X, y)

    # Create/clean the output directory
    results_dir = path.join("pk_samples", "khiops_classifier_pickle")
    pkc_pickle_path = path.join(results_dir, "khiops_classifier.pkl")
    if path.exists(pkc_pickle_path):
        os.remove(pkc_pickle_path)
    else:
        os.makedirs(results_dir, exist_ok=True)

    # Pickle its content to a file
    with open(pkc_pickle_path, "wb") as pkc_pickle_write_file:
        pickle.dump(pkc, pkc_pickle_write_file)

    # Unpickle it
    with open(pkc_pickle_path, "rb") as pkc_pickle_file:
        new_pkc = pickle.load(pkc_pickle_file)

    # Make some predictions on the training dataset with the unpickled classifier
    new_pkc.predict(X)
    y_predicted = new_pkc.predict(X)
    print("Predicted classes (first 10):")
    print(y_predicted[:10])
    print("---")


def khiops_classifier_multitable_file():
    """Trains a `.KhiopsClassifier` with a file path based dataset

    .. warning::
        This dataset input method is **Deprecated** and will be removed in pyKhiops 11.
        If you need to handle large datasets that do not easily fit into memory then you
        may use the `~.pykhiops.core` API directly, which allows us to specify file
        paths directly.
    """
    # Create output directory
    results_dir = path.join("pk_samples", "khiops_classifier_multitable_file")
    if not path.exists("pk_samples"):
        os.mkdir("pk_samples")
        os.mkdir(results_dir)
    else:
        if not path.exists(results_dir):
            os.mkdir(results_dir)

    # Load the root table of the dataset into a pandas dataframe
    accidents_dataset_path = path.join(pk.get_samples_dir(), "AccidentsSummary")
    accidents_df = pd.read_csv(
        path.join(accidents_dataset_path, "Accidents.txt"),
        sep="\t",
        encoding="latin1",
    )

    # Split the root dataframe into train and test
    X_train_main, X_test_main = train_test_split(
        accidents_df, test_size=0.3, random_state=1
    )

    # Load the secondary table of the dataset into a pandas dataframe
    vehicles_df = pd.read_csv(
        path.join(accidents_dataset_path, "Vehicles.txt"), sep="\t"
    )

    # Split the secondary dataframe with the keys of the splitted root dataframe
    X_train_ids = X_train_main["AccidentId"].to_frame()
    X_test_ids = X_test_main["AccidentId"].to_frame()
    X_train_secondary = X_train_ids.merge(vehicles_df, on="AccidentId")
    X_test_secondary = X_test_ids.merge(vehicles_df, on="AccidentId")

    # Write the train and test dataset sets to disk
    # For the test file we remove the target column from the main table
    X_train_main_path = path.join(results_dir, "X_train_main.txt")
    X_train_main.to_csv(X_train_main_path, sep="\t", header=True, index=False)
    X_train_secondary_path = path.join(results_dir, "X_train_secondary.txt")
    X_train_secondary.to_csv(X_train_secondary_path, sep="\t", header=True, index=False)
    X_test_main_path = path.join(results_dir, "X_test_main.txt")
    y_test = X_test_main.sort_values("AccidentId")["Gravity"]
    X_test_main.drop(columns="Gravity").to_csv(
        X_test_main_path, sep="\t", header=True, index=False
    )
    X_test_secondary_path = path.join(results_dir, "X_test_secondary.txt")
    X_test_secondary.to_csv(X_test_secondary_path, sep="\t", header=True, index=False)

    # Define the dictionary of train
    X_train_dataset = {
        "main_table": "Accidents",
        "tables": {
            "Accidents": (X_train_main_path, "AccidentId"),
            "Vehicles": (X_train_secondary_path, ["AccidentId", "VehicleId"]),
        },
        "format": ("\t", True),
    }
    X_test_dataset = {
        "main_table": "Accidents",
        "tables": {
            "Accidents": (X_test_main_path, "AccidentId"),
            "Vehicles": (X_test_secondary_path, ["AccidentId", "VehicleId"]),
        },
        "format": ("\t", True),
    }

    # Create the classifier and fit it
    pkc = KhiopsClassifier(output_dir=results_dir)
    pkc.fit(X_train_dataset, y="Gravity")

    # Predict the class in addition to the class probabilities on the test dataset
    y_test_pred_path = pkc.predict(X_test_dataset)
    y_test_pred = pd.read_csv(y_test_pred_path, sep="\t")
    print("Predicted classes (first 10):")
    print(y_test_pred["PredictedGravity"].head(10))
    print("---")

    y_test_probas_path = pkc.predict_proba(X_test_dataset)
    y_test_probas = pd.read_csv(y_test_probas_path, sep="\t")
    proba_columns = [col for col in y_test_probas if col.startswith("Prob")]
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[proba_columns].head(10))
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred["PredictedGravity"])
    test_auc = metrics.roc_auc_score(y_test, y_test_probas["ProbGravityLethal"])
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")


def khiops_classifier_multitable_dataframe_list():
    """Trains a KhiopsClassifier using a list dataset specification

    .. warning::
        This dataset input method is **Deprecated** and will be removed in pyKhiops 11.
    """
    # Load the root table of the dataset into a pandas dataframe
    accidents_dataset_path = path.join(pk.get_samples_dir(), "AccidentsSummary")
    accidents_df = pd.read_csv(
        path.join(accidents_dataset_path, "Accidents.txt"),
        sep="\t",
        encoding="latin1",
    )

    # Split the root dataframe into train and test
    accidents_train_df, accidents_test_df = train_test_split(
        accidents_df, test_size=0.3, random_state=1
    )

    # Obtain the main X feature table and the y target vector ("Class" column)
    y_train = accidents_train_df["Gravity"]
    y_test = accidents_test_df["Gravity"]
    X_train_main = accidents_train_df.drop("Gravity", axis=1)
    X_test_main = accidents_test_df.drop("Gravity", axis=1)

    # Load the secondary table of the dataset into a pandas dataframe
    vehicles_df = pd.read_csv(
        path.join(accidents_dataset_path, "Vehicles.txt"), sep="\t"
    )

    # Split the secondary dataframe with the keys of the splitted root dataframe
    X_train_ids = X_train_main["AccidentId"].to_frame()
    X_test_ids = X_test_main["AccidentId"].to_frame()
    X_train_secondary = X_train_ids.merge(vehicles_df, on="AccidentId")
    X_test_secondary = X_test_ids.merge(vehicles_df, on="AccidentId")

    # Create the classifier specifying the key column name
    pkc = KhiopsClassifier(key="AccidentId")

    # Train the classifier
    pkc.fit([X_train_main, X_train_secondary], y_train)

    # Predict the class on the test dataset
    y_test_pred = pkc.predict([X_test_main, X_test_secondary])
    print("Predicted classes (first 10):")
    print(y_test_pred[:10])
    print("---")

    # Predict the class probability on the test dataset
    y_test_probas = pkc.predict_proba([X_test_main, X_test_secondary])
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[:10])
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_probas[:, 1])
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")


exported_samples = [
    khiops_classifier,
    khiops_classifier_multiclass,
    khiops_regressor,
    khiops_classifier_multitable,
    khiops_encoder,
    khiops_encoder_multitable,
    khiops_encoder_pipeline_with_hgbc,
    khiops_coclustering,
    khiops_classifier_pickle,
    khiops_classifier_multitable_file,
    khiops_classifier_multitable_dataframe_list,
]


def execute_samples(args):
    """Executes all non-interactive samples"""
    # Create the results directory if it does not exist
    if not path.isdir("./pk_samples"):
        os.mkdir("./pk_samples")

    # Set the user-defined samples dir if any
    if args.samples_dir is not None:
        pk.get_runner().samples_dir = args.samples_dir

    if args.include is not None:
        execution_samples = filter_samples(
            exported_samples, args.include, args.exact_match
        )
    else:
        execution_samples = exported_samples

    # Print the execution title
    if execution_samples:
        print(f"pyKhiops {pykhiops.__version__} running on Khiops ", end="")
        print(f"{pk.get_khiops_version()}\n")
        print(f"Sample datasets location: {pk.get_samples_dir()}")
        print(f"{len(execution_samples)} sample(s) to execute\n")

        for sample in execution_samples:
            print(">>> Executing samples_sklearn." + sample.__name__)
            sample.__call__()
            print("> Done\n")

        print("*** Samples run! ****")

    else:
        print("*** No samples to run ***")


def filter_samples(exported_samples, include, exact_match):
    """Filter the samples according to the command line options"""
    filtered_samples = []
    for sample in exported_samples:
        for sample_name in include:
            if (exact_match and sample_name == sample.__name__) or (
                not exact_match and sample_name in sample.__name__
            ):
                filtered_samples.append(sample)

    return filtered_samples


# Run the samples if executed as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python samples_sklearn.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Examples of use of the sklearn submodule of the pyKhiops library",
    )
    parser.add_argument(
        "-d",
        "--samples-dir",
        metavar="URI",
        help="Location of the Khiops 'samples' directory",
    )
    parser.add_argument(
        "-i", "--include", nargs="*", help="Executes only the tests matching this list"
    )
    parser.add_argument(
        "-e",
        "--exact-match",
        action="store_true",
        help="Matches with --include are exact",
    )
    execute_samples(parser.parse_args())
