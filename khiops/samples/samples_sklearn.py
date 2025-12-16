######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Khiops Python sklearn submodule samples

The functions in this script demonstrate the basic use of the sklearn submodule of the
Khiops Python library.
"""
import argparse
import khiops
import os
from khiops import core as kh

# Disable PEP8 variable names because of scikit-learn X,y conventions
# To capture invalid-names other than X,y run:
#   pylint --disable=all --enable=invalid-names samples_sklearn.py
# pylint: disable=invalid-name

# For ease of use the functions in this module contain (repeated) import statements
# We disable all pylint warnings related to imports
# pylint: disable=import-outside-toplevel,redefined-outer-name,reimported


def khiops_classifier():
    """Trains a `.KhiopsClassifier` on a monotable dataframe"""
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsClassifier
    from sklearn import metrics
    from sklearn.model_selection import train_test_split

    # Load the dataset into a pandas dataframe
    adult_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
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

    # Create the classifier object
    khc = KhiopsClassifier()

    # Train the classifier
    khc.fit(X_train, y_train)

    # Predict the classes on the test dataset
    y_test_pred = khc.predict(X_test)
    print("Predicted classes (first 10):")
    print(y_test_pred[0:10])
    print("---")

    # Predict the class probabilities on the test dataset
    y_test_probas = khc.predict_proba(X_test)
    print(f"Class order: {khc.classes_}")
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[0:10])
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_probas[:, 1])
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")

    # If you have Khiops Visualization installed you may open the report as follows
    # khc.export_report_file("report.khj")
    # kh.visualize_report("report.khj")


def khiops_classifier_multiclass():
    """Trains a multiclass `.KhiopsClassifier` on a monotable dataframe"""
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsClassifier
    from sklearn import metrics
    from sklearn.model_selection import train_test_split

    # Load the dataset into a pandas dataframe
    iris_path = os.path.join(kh.get_samples_dir(), "Iris", "Iris.txt")
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
    khc = KhiopsClassifier()

    # Train the classifier
    khc.fit(X_train, y_train)

    # Predict the classes on the test dataset
    y_test_pred = khc.predict(X_test)
    print("Predicted classes (first 10):")
    print(y_test_pred[:10])
    print("---")

    # Predict the class probabilities on the test datasets
    y_test_probas = khc.predict_proba(X_test)
    print(f"Class order: {khc.classes_}")
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[:10])
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_probas, multi_class="ovr")
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")


def khiops_classifier_text():
    """Train a `.KhiopsClassifier` on a monotable dataframe with textual data"""
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsClassifier
    from sklearn import metrics
    from sklearn.model_selection import train_test_split

    # Load the dataset into a pandas dataframe
    data_table_path = os.path.join(
        kh.get_samples_dir(), "NegativeAirlineTweets", "NegativeAirlineTweets.txt"
    )
    data_df = pd.read_csv(data_table_path, sep="\t")

    # Split the whole dataframe into train and test (70%-30%)
    data_train_df, data_test_df = train_test_split(
        data_df, test_size=0.3, random_state=1
    )

    # Split the dataset into:
    # - the X feature table
    # - the y target vector ("negativereason" column)
    X_train = data_train_df.drop("negativereason", axis=1)
    X_test = data_test_df.drop("negativereason", axis=1)
    y_train = data_train_df["negativereason"]
    y_test = data_test_df["negativereason"]

    # Set Pandas StringDType on the "text" column
    X_train["text"] = X_train["text"].astype("string")
    X_test["text"] = X_test["text"].astype("string")

    # Create the classifier object
    khc = KhiopsClassifier()

    # Train the classifier
    khc.fit(X_train, y_train)

    # Predict the classes on the test dataset
    y_test_pred = khc.predict(X_test)
    print("Predicted classes (first 10):")
    print(y_test_pred[0:10])
    print("---")

    # Predict the class probabilities on the test dataset
    y_test_probas = khc.predict_proba(X_test)
    print(f"Class order: {khc.classes_}")
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[0:10])
    print("---")

    # Evaluate the accuracy metric on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy = {test_accuracy}")

    # If you have Khiops Visualization installed you may open the report as follows
    # khc.export_report_file("report.khj")
    # kh.visualize_report("report.khj")


def khiops_classifier_multitable_star():
    """Trains a `.KhiopsClassifier` on a star multi-table dataset"""
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsClassifier, train_test_split_dataset
    from sklearn import metrics

    # Load the dataset into pandas dataframes
    accidents_data_dir = os.path.join(kh.get_samples_dir(), "AccidentsSummary")
    accidents_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Accidents.txt"),
        sep="\t",
    )
    vehicles_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Vehicles.txt"), sep="\t"
    )

    # Create the dataset spec and the target
    X = {
        "main_table": (accidents_df.drop("Gravity", axis=1), ["AccidentId"]),
        "additional_data_tables": {
            "Vehicles": (vehicles_df, ["AccidentId", "VehicleId"]),
        },
    }
    y = accidents_df["Gravity"]

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split_dataset(
        X, y, test_size=0.3, random_state=1
    )

    # Train the classifier (by default it analyzes 100 multi-table features)
    khc = KhiopsClassifier()
    khc.fit(X_train, y_train)

    # Predict the class on the test dataset
    y_test_pred = khc.predict(X_test)
    print("Predicted classes (first 10):")
    print(y_test_pred[:10])
    print("---")

    # Predict the class probability on the test dataset
    y_test_probas = khc.predict_proba(X_test)
    print(f"Class order: {khc.classes_}")
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[:10])
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_probas[:, 1])
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")


def khiops_classifier_multitable_snowflake():
    """Trains a `.KhiopsClassifier` on a snowflake multi-table dataset"""
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsClassifier, train_test_split_dataset
    from sklearn import metrics

    # Load the dataset tables into dataframes
    accidents_data_dir = os.path.join(kh.get_samples_dir(), "Accidents")
    accidents_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Accidents.txt"), sep="\t"
    )
    users_df = pd.read_csv(os.path.join(accidents_data_dir, "Users.txt"), sep="\t")
    vehicles_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Vehicles.txt"), sep="\t"
    )
    places_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Places.txt"), sep="\t", low_memory=False
    )

    # Build the multi-table dataset spec (drop the target column "Gravity")
    X = {
        "main_table": (accidents_df.drop("Gravity", axis=1), ["AccidentId"]),
        "additional_data_tables": {
            "Vehicles": (vehicles_df, ["AccidentId", "VehicleId"]),
            "Vehicles/Users": (users_df, ["AccidentId", "VehicleId"]),
            "Places": (places_df, ["AccidentId"], True),
        },
    }

    # Load the target variable "Gravity"
    y = accidents_df["Gravity"]

    # Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)

    # Train the classifier (by default it creates 1000 multi-table features)
    khc = KhiopsClassifier(n_trees=0)
    khc.fit(X_train, y_train)

    # Predict the class on the test dataset
    y_test_pred = khc.predict(X_test)
    print("Predicted classes (first 10):")
    print(y_test_pred[:10])
    print("---")

    # Predict the class probability on the test dataset
    y_test_probas = khc.predict_proba(X_test)
    print(f"Class order: {khc.classes_}")
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[:10])
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test_pred, y_test)
    test_auc = metrics.roc_auc_score(y_test, y_test_probas[:, 1])
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")


def khiops_classifier_sparse():
    """Trains a `.KhiopsClassifier` on a monotable sparse matrix"""
    # Imports
    from khiops.sklearn import KhiopsClassifier
    from sklearn import metrics
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import HashingVectorizer

    # Load 3 classes of the 20newsgroups dataset
    categories = ["comp.graphics", "sci.space", "misc.forsale"]
    data_train, y_train = fetch_20newsgroups(
        subset="train",
        categories=categories,
        return_X_y=True,
    )
    data_test, y_test = fetch_20newsgroups(
        subset="test",
        categories=categories,
        return_X_y=True,
    )

    # Extract features from the training data using a sparse vectorizer
    vectorizer = HashingVectorizer(n_features=2**10, stop_words="english")
    X_train = vectorizer.fit_transform(data_train)

    # Extract features from the test data using the same vectorizer
    X_test = vectorizer.transform(data_test)

    # Create the classifier object
    khc = KhiopsClassifier()

    # Train the classifier
    khc.fit(X_train, y_train)

    # Predict the classes on the test dataset
    y_test_pred = khc.predict(X_test)
    print("Predicted classes (first 10):")
    print(y_test_pred[0:10])
    print("---")

    # Predict the class probabilities on the test dataset
    y_test_probas = khc.predict_proba(X_test)
    print(f"Class order: {khc.classes_}")
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[0:10])
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_probas, multi_class="ovr")
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")


def khiops_classifier_pickle():
    """Shows the serialization and deserialization of a `.KhiopsClassifier`"""
    # Imports
    import os
    import pandas as pd
    import pickle
    from khiops import core as kh
    from khiops.sklearn import KhiopsClassifier

    # Create/clean the output directory
    results_dir = os.path.join("kh_samples", "khiops_classifier_pickle")
    khc_pickle_path = os.path.join(results_dir, "khiops_classifier.pkl")
    if os.path.exists(khc_pickle_path):
        os.remove(khc_pickle_path)
    else:
        os.makedirs(results_dir, exist_ok=True)

    # Load the "Iris" dataset
    iris_path = os.path.join(kh.get_samples_dir(), "Iris", "Iris.txt")
    iris_df = pd.read_csv(iris_path, sep="\t")
    X = iris_df.drop("Class", axis=1)
    y = iris_df["Class"]

    # Train the model with the Iris dataset
    khc = KhiopsClassifier()
    khc.fit(X, y)

    # Pickle its content to a file
    with open(khc_pickle_path, "wb") as khc_pickle_output_file:
        pickle.dump(khc, khc_pickle_output_file)

    # Unpickle it
    with open(khc_pickle_path, "rb") as khc_pickle_file:
        new_khc = pickle.load(khc_pickle_file)

    # Make some predictions on the training dataset with the unpickled classifier
    new_khc.predict(X)
    y_predicted = new_khc.predict(X)
    print("Predicted classes (first 10):")
    print(y_predicted[:10])
    print("---")


def khiops_classifier_with_hyperparameters():
    """Trains a `.KhiopsClassifier` on a star multi-table dataset
    (advanced version with more hyperparameters)
    """
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsClassifier
    from sklearn import metrics
    from sklearn.model_selection import train_test_split

    # Load the root table of the dataset into a pandas dataframe
    accidents_dataset_path = os.path.join(kh.get_samples_dir(), "AccidentsSummary")
    accidents_df = pd.read_csv(
        os.path.join(accidents_dataset_path, "Accidents.txt"),
        sep="\t",
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
        os.path.join(accidents_dataset_path, "Vehicles.txt"), sep="\t"
    )

    # Split the secondary dataframe with the keys of the split root dataframe
    X_train_ids = X_train_main["AccidentId"].to_frame()
    X_test_ids = X_test_main["AccidentId"].to_frame()
    X_train_secondary = X_train_ids.merge(vehicles_df, on="AccidentId")
    X_test_secondary = X_test_ids.merge(vehicles_df, on="AccidentId")

    # Create the dataset multitable specification for the train/test split
    # We specify each table with a name and a tuple (dataframe, key_columns)
    X_train = {
        "main_table": (X_train_main, ["AccidentId"]),
        "additional_data_tables": {
            "Vehicles": (X_train_secondary, ["AccidentId", "VehicleId"]),
        },
    }
    X_test = {
        "main_table": (X_test_main, ["AccidentId"]),
        "additional_data_tables": {
            "Vehicles": (X_test_secondary, ["AccidentId", "VehicleId"]),
        },
    }
    # Train the classifier (by default it analyzes 100 multi-table features)
    khc = KhiopsClassifier(
        n_features=20,
        n_pairs=5,
        n_trees=5,
        n_selected_features=10,
        n_evaluated_features=15,
        specific_pairs=[("Light", "Weather"), ("Light", "IntersectionType")],
        all_possible_pairs=True,
        construction_rules=["TableMode", "TableSelection"],
        group_target_value=False,
    )
    khc.fit(X_train, y_train)

    # Predict the class on the test dataset
    y_test_pred = khc.predict(X_test)
    print("Predicted classes (first 10):")
    print(y_test_pred[:10])
    print("---")

    # Predict the class probability on the test dataset
    y_test_probas = khc.predict_proba(X_test)
    print(f"Class order: {khc.classes_}")
    print("Predicted class probabilities (first 10):")
    print(y_test_probas[:10])
    print("---")

    # Evaluate accuracy and auc metrics on the test dataset
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_probas[:, 1])
    print(f"Test accuracy = {test_accuracy}")
    print(f"Test auc      = {test_auc}")


def khiops_regressor():
    """Trains a `.KhiopsRegressor` on a monotable dataframe"""
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsRegressor
    from sklearn import metrics
    from sklearn.model_selection import train_test_split

    # Load the "Adult" dataset and set the target to the "age" column
    adult_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    adult_df = pd.read_csv(adult_path, sep="\t")
    X = adult_df.drop("age", axis=1)
    y = adult_df["age"]

    # Split the whole dataframe into train and test (40%-60% for speed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )

    # Create the regressor object
    khr = KhiopsRegressor()

    # Train the regressor
    khr.fit(X_train, y_train)

    # Predict the values on the test dataset
    y_test_pred = khr.predict(X_test)
    print("Predicted values for 'age' (first 10):")
    print(y_test_pred[:10])
    print("---")

    # Evaluate R2 and MAE metrics on the test dataset
    test_r2 = metrics.r2_score(y_test, y_test_pred)
    test_mae = metrics.mean_absolute_error(y_test, y_test_pred)
    print(f"Test R2  = {test_r2}")
    print(f"Test MAE = {test_mae}")

    # If you have Khiops Visualization installed you may open the report as follows
    # khr.export_report_file("report.khj")
    # kh.visualize_report("report.khj")


def khiops_encoder():
    """Trains a `.KhiopsEncoder` on a monotable dataframe

    The Khiops encoder is a supervised feature encoder. It discretizes numerical
    features and groups categorical features in a way that the resulting interval/groups
    have the highest class-purity.

    .. note::
        For simplicity we train from the whole dataset. To assess the performance one
        usually splits the dataset into train and test subsets.
    """
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsEncoder

    # Load the dataset
    iris_path = os.path.join(kh.get_samples_dir(), "Iris", "Iris.txt")
    iris_df = pd.read_csv(iris_path, sep="\t")
    X = iris_df.drop("Class", axis=1)
    y = iris_df["Class"]

    # Create the encoder object
    khe = KhiopsEncoder(transform_type_numerical="part_label")
    khe.fit(X, y)

    # Transform the training dataset
    X_transformed = khe.transform(X)

    # Print both the original and transformed features
    print("Original:")
    print(X[:10])
    print("---")
    print("Encoded feature names:")
    print(khe.feature_names_out_)
    print("Encoded data:")
    print(X_transformed[:10])
    print("---")

    # If you have Khiops Visualization installed you may open the report as follows
    # khe.export_report_file("report.khj")
    # kh.visualize_report("report.khj")


def khiops_encoder_multitable_star():
    """Trains a `.KhiopsEncoder` on a star multi-table dataset"""
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsEncoder

    # Load the dataset tables into dataframe
    accidents_data_dir = os.path.join(kh.get_samples_dir(), "AccidentsSummary")
    accidents_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Accidents.txt"),
        sep="\t",
    )
    vehicles_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Vehicles.txt"), sep="\t"
    )

    # Build the multi-table dataset spec (drop the target column "Gravity")
    X = {
        "main_table": (accidents_df.drop("Gravity", axis=1), ["AccidentId"]),
        "additional_data_tables": {
            "Vehicles": (vehicles_df, ["AccidentId", "VehicleId"]),
        },
    }

    # Load the target variable "Gravity"
    y = accidents_df["Gravity"]

    # Create the KhiopsEncoder with 5 multitable features and fit it
    khe = KhiopsEncoder(n_features=10)
    khe.fit(X, y)

    # Transform the train dataset
    print("Encoded feature names:")
    print(khe.feature_names_out_)
    print("Encoded data:")
    print(khe.transform(X)[:10])


def khiops_encoder_multitable_snowflake():
    """Trains a `.KhiopsEncoder` on a snowflake multi-table dataset"""
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsEncoder

    # Load the tables into dataframes
    accidents_data_dir = os.path.join(kh.get_samples_dir(), "Accidents")
    accidents_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Accidents.txt"), sep="\t"
    )
    users_df = pd.read_csv(os.path.join(accidents_data_dir, "Users.txt"), sep="\t")
    vehicles_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Vehicles.txt"), sep="\t"
    )
    places_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Places.txt"), sep="\t", low_memory=False
    )

    # Build the multi-table dataset spec (drop the target column "Gravity")
    X = {
        "main_table": (accidents_df.drop("Gravity", axis=1), ["AccidentId"]),
        "additional_data_tables": {
            "Vehicles": (vehicles_df, ["AccidentId", "VehicleId"]),
            "Vehicles/Users": (users_df, ["AccidentId", "VehicleId"]),
            "Places": (places_df, ["AccidentId"], True),
        },
    }

    # Load the target variable "Gravity"
    y = accidents_df["Gravity"]

    # Create the KhiopsEncoder with 10 additional multitable features and fit it
    khe = KhiopsEncoder(n_features=10)
    khe.fit(X, y)

    # Transform the train dataset
    print("Encoded feature names:")
    print(khe.feature_names_out_)
    print("Encoded data:")
    print(khe.transform(X)[:10])


# Disable line too long just to have a title linking the sklearn documentation
# pylint: disable=line-too-long
def khiops_encoder_pipeline_with_hgbc():
    """Uses a `.KhiopsEncoder` with a `~sklearn.ensemble.HistGradientBoostingClassifier`"""
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsEncoder
    from sklearn import metrics
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    # Load the dataset into dataframes
    adult_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    adult_df = pd.read_csv(adult_path, sep="\t")
    X = adult_df.drop("class", axis=1)
    y = adult_df["class"]

    # Split the dataset into train and test (70%-30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # Create the pipeline and fit it. Steps:
    # - The khiops supervised column encoder, generates a full-categorical table
    # - One hot encoder in all columns
    # - Train the HGB classifier
    pipe_steps = [
        ("khiops_enc", KhiopsEncoder()),
        (
            "onehot_enc",
            ColumnTransformer([], remainder=OneHotEncoder(sparse_output=False)),
        ),
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


def khiops_encoder_with_hyperparameters():
    """Trains a `.KhiopsEncoder` on a star multi-table dataset
    (advanced version with more hyperparameters)
    """
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsEncoder

    # Load the tables into dataframes
    accidents_data_dir = os.path.join(kh.get_samples_dir(), "AccidentsSummary")
    accidents_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Accidents.txt"), sep="\t"
    )
    vehicles_df = pd.read_csv(
        os.path.join(accidents_data_dir, "Vehicles.txt"), sep="\t"
    )

    # Build the multi-table dataset spec (drop the target column "Gravity")
    X = {
        "main_table": (accidents_df.drop("Gravity", axis=1), ["AccidentId"]),
        "additional_data_tables": {
            "Vehicles": (vehicles_df, ["AccidentId", "VehicleId"]),
        },
    }

    # Load the target variable "Gravity"
    y = accidents_df["Gravity"]

    # Create the KhiopsEncoder with 10 additional multitable features and fit it
    khe = KhiopsEncoder(
        n_features=20,
        n_pairs=5,
        n_trees=5,
        specific_pairs=[("Light", "Weather"), ("Light", "IntersectionType")],
        all_possible_pairs=True,
        construction_rules=["TableMode", "TableSelection"],
        group_target_value=False,
        informative_features_only=True,
        keep_initial_variables=True,
        transform_type_categorical="part_id",
        transform_type_numerical="part_id",
        transform_type_pairs="part_id",
    )
    khe.fit(X, y)

    # Transform the train dataset
    print("Encoded feature names:")
    print(khe.feature_names_out_)
    print("Encoded data:")
    print(khe.transform(X)[:10])


# pylint: enable=line-too-long


def khiops_coclustering():
    """Trains a `.KhiopsCoclustering` on a dataframe"""
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsCoclustering
    from sklearn.model_selection import train_test_split

    # Load the secondary table of the dataset into a pandas dataframe
    splice_data_dir = os.path.join(kh.get_samples_dir(), "SpliceJunction")
    splice_dna_df = pd.read_csv(
        os.path.join(splice_data_dir, "SpliceJunctionDNA.txt"), sep="\t"
    )

    # Train with only 70% of data (for speed in this example)
    X, _ = train_test_split(splice_dna_df, test_size=0.3, random_state=1)

    # Create the KhiopsCoclustering instance
    khcc = KhiopsCoclustering()

    # Train the model with the whole dataset
    khcc.fit(X, id_column="SampleId")

    # Predict the clusters in some instances
    X_clusters = khcc.predict(X)
    print("Predicted clusters (first 10)")
    print(X_clusters[:10])
    print("---")

    # If you have Khiops Co-Visualization installed you may open the report as follows
    # khcc.export_report_file("report.khcj")
    # kh.visualize_report("report.khcj")


def khiops_coclustering_simplify():
    """Simplifies a `.KhiopsCoclustering` already trained on a dataframe"""
    # Imports
    import os
    import pandas as pd
    from khiops import core as kh
    from khiops.sklearn import KhiopsCoclustering
    from sklearn.model_selection import train_test_split

    # Load the secondary table of the dataset into a pandas dataframe
    splice_data_dir = os.path.join(kh.get_samples_dir(), "SpliceJunction")
    splice_dna_X = pd.read_csv(
        os.path.join(splice_data_dir, "SpliceJunctionDNA.txt"), sep="\t"
    )

    # Train with only 70% of data (for speed in this example)
    X, _ = train_test_split(splice_dna_X, test_size=0.3, random_state=1)

    # Create the KhiopsCoclustering instance
    khcc = KhiopsCoclustering()

    # Train the model with the whole dataset
    khcc.fit(X, id_column="SampleId")

    # Simplify coclustering along the individual ID dimension
    simplified_khcc = khcc.simplify(max_part_numbers={"SampleId": 3})

    # Predict the clusters using the simplified model
    X_clusters = simplified_khcc.predict(X)
    print("Predicted clusters (only three at most)")
    print(X_clusters)
    print("---")


exported_samples = [
    khiops_classifier,
    khiops_classifier_multiclass,
    khiops_classifier_text,
    khiops_classifier_multitable_star,
    khiops_classifier_multitable_snowflake,
    khiops_classifier_sparse,
    khiops_classifier_pickle,
    khiops_classifier_with_hyperparameters,
    khiops_regressor,
    khiops_encoder,
    khiops_encoder_multitable_star,
    khiops_encoder_multitable_snowflake,
    khiops_encoder_pipeline_with_hgbc,
    khiops_encoder_with_hyperparameters,
    khiops_coclustering,
    khiops_coclustering_simplify,
]


def execute_samples(args):
    """Executes all non-interactive samples"""
    # Create the results directory if it does not exist
    os.makedirs("./kh_samples", exist_ok=True)

    # Set the user-defined samples dir if any
    if args.samples_dir is not None:
        kh.get_runner().samples_dir = args.samples_dir

    if args.include is not None:
        execution_samples = filter_samples(
            exported_samples, args.include, args.exact_match
        )
    else:
        execution_samples = exported_samples

    # Print the execution title
    if execution_samples:
        print(f"Khiops Python library {khiops.__version__} running on Khiops ", end="")
        print(f"{kh.get_khiops_version()}\n")
        print(f"Sample datasets location: {kh.get_samples_dir()}")
        print(f"{len(execution_samples)} sample(s) to execute\n")

        for sample in execution_samples:
            print(f">>> Executing samples_sklearn.{sample.__name__}")
            sample.__call__()
            print("> Done\n")

        print("*** Samples run! ****")

    else:
        print("*** No samples to run ***")


def filter_samples(sample_list, include, exact_match):
    """Filter the samples according to the command line options"""
    filtered_samples = []
    for sample in sample_list:
        for sample_name in include:
            if (exact_match and sample_name == sample.__name__) or (
                not exact_match and sample_name in sample.__name__
            ):
                filtered_samples.append(sample)

    return filtered_samples


def build_argument_parser(prog, description):
    """Samples argument parser builder function

    Parameters
    ----------
    prog : str
        Name of the program, as required by the argument parser.
    description : str
        Description of the program, as required by the argument parser.

    Returns
    -------
    ArgumentParser
        Argument parser object.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.RawTextHelpFormatter,
        description=description,
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
    return parser


# Run the samples if executed as a script
if __name__ == "__main__":
    argument_parser = build_argument_parser(
        prog="python samples.py",
        description=(
            "Examples of use of the sklearn submodule of the Khiops Python library"
        ),
    )
    execute_samples(argument_parser.parse_args())
