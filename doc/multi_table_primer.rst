===========================
Multi-Table Learning Primer
===========================

Machine learning on tabular data is traditionally performed on a single table containing a record
for each statistical object of the sample. However, data is usually stored in databases with
multiple tables whose relationships are specified through a schema. Thus, before training
a predictor, a preprocessing is necessary to flatten the relational tables into a single one
containing all relevant information for the learning task.

This preprocessing (which belongs to the feature engineering process) is often very time consuming.
One of the main Khiops features is to automate this process by natively learning predictors from
multi-table datasets. Specifically, Khiops automatically:

- generates features (aggregates) from the multi-table schema
- evaluates the predictive value of the generated features
- selects a small subset of the generated features to build a predictor

Supported Multi-Table Schemas
=============================
Khiops allows to train an estimator on the two most common dataset schemas: *star* and *snowflake*.
More complex schemas are supported only by the ``core`` library at the moment.

Star Schema
-----------
In this schema there is one main table containing the identifier of the statistical object and its
basic properties. The main table points to one or more secondary ones, each having or more records
associated to a given object.

Here is a simple example of a star schema (for brevity we do not show the tables' columns):

.. code-block:: text

  Customer(id_customer)
  |
  +---1:1--- Address(id_customer)
  |
  +---1:n--- Service(id_customer, id_product)

The statistical object in this schema is a ``Customer`` that is associated to a unique ``Address``
and to one or more ``Services``. In parentheses we show the key columns of each table that act as
`foreign keys <https://en.wikipedia.org/wiki/Foreign_key>`_ to associate the records of each table.

You can find more information about the star schema in its `Wikipedia article
<https://en.wikipedia.org/wiki/Star_schema>`_.

Snowflake Schema
----------------
The snowflake schema generalizes the star schema by allowing each secondary table to be in a star
schema by itself, forming a tree whose root is the main table.

Here we extend the previous star schema to a snowflake schema

.. code-block:: text

  Customer(id_customer)
  |
  +---1:1--- Address(id_customer)
  |
  +---1:n--- Service(id_customer, id_product)
             |
             +---1:n--- Usage(id_customer, id_product)

Again, the statistical object in this schema is a ``Customer`` that is associated to a unique
``Address`` and to one or more ``Services``. But additionally, each ``Service`` is associated to one
or more ``Usages``.

You can find more information about the snowflake schema in `its Wikipedia article
<https://en.wikipedia.org/wiki/Snowflake_schema>`_.

Multi-Table Learning with Scikit-Learn Estimators
=================================================

The supervised estimators in :doc:`sklearn/index` handle multi-table datasets with a special input
feature object ``X``. Specifically, instead of a `pandas.DataFrame`, ``X`` must be a ``dict`` that
specifies the dataset schema in the following way::

   X = {
      "main_table": <name of the main table>,
      "tables" : {
          <name of the main table>: (<dataframe of the main table>, <key of the main table>),
          <name of table 1>: (<dataframe of table 1>, <key of table 1>),
          <name of table 2>: (<dataframe of table 2>, <key of table 2>),
          ...
       }
       "relations" : [
            (<name of the main table>, <name of a different table>, <entity flag>),
            (<name of another table>, <name of yet another table>, <entity flag>),
            ...
       ],
   }

The three fields of this dictionary are:

- ``main_table``: The name of the main table.
- ``tables``: A dictionary indexed by the tables' names. Each table is associated to a 2-tuple
  containing the following fields:

  - The `pandas.DataFrame` object of the table.
  - The key columns' names : Either a list of strings or a single string.

- ``relations``: An optional field containing a list of tuples describing the relations between
  tables. The first two values (Strings) of each tuple correspond to names of both the parent and the child table
  involved in the relation. A third value (Boolean) can be optionally added to the tuple to indicate if the relation is
  either ``1:n`` or ``1:1`` (entity). For example, If the tuple ``(table1, table2, True)`` is contained in this
  field, it means that:

  - ``table1`` and ``table2`` are in a ``1:1`` relationship
  - The key of ``table1`` is contained in that of ``table2`` (ie. keys are hierarchical)

  If the ``relations`` field is not present then Khiops Python assumes that the tables are in a *star*
  schema.

.. note::

    With respect to Khiops, Khiops Python sklearn estimators have some limitations. They currently do not
    support external data tables.

    This feature will be available in upcoming releases. If you need to use it, you can use the `khiops.core`
    sub-module (see below).

Examples
--------

Star Schema
~~~~~~~~~~~
For the ``AccidentsSummary`` dataset above where tables are related through the following *star*
schema:

.. code-block:: text

    Accident(AccidentId)
    |
    +---1:n--- Vehicle(AccidentId, VehicleId)

We build the input ``X`` as follows::

   accidents_df = pd.read_csv(f"{kh.get_samples_dir()}/AccidentsSummary/Accidents.txt", sep="\t", encoding="latin1")
   vehicles_df = pd.read_csv(f"{kh.get_samples_dir()}/AccidentsSummary/Vehicles.txt", sep="\t", encoding="latin1")
   X = {
      "main_table" : "Accident",
      "tables": {
          "Accident": (accidents_df.drop("Gravity", axis=1), "AccidentId"),
          "Vehicle": (vehicles_df, ["AccidentId", "VehicleId"])
      }
    }


Snowflake Schema
~~~~~~~~~~~~~~~~

For the ``Accidents`` dataset (an extension of ``AccidentsSummary``) where tables are related
through the following *snowflake* schema

.. code-block:: text

    Accident(AccidentId)
    |
    +--- 1:n --- Vehicle(AccidentId, VehicleId)
    |            |
    |            +--- 1:n --- User(AccidentId, VehicleId)
    |
    +--- 1:1 --- Place(AccidentId)

We build the input ``X`` as follows::

    # We use `Accidents.txt` table of `AccidentsSummary` as it contains the `Gravity` label pre-calculated
    accidents_df = pd.read_csv(f"{kh.get_samples_dir()}/AccidentsSummary/Accidents.txt", sep="\t", encoding="latin1")
    vehicles_df = pd.read_csv(f"{kh.get_samples_dir()}/Accidents/Vehicles.txt", sep="\t", encoding="latin1")
    users_df = pd.read_csv(f"{kh.get_samples_dir()}/Accidents/Users.txt", sep="\t", encoding="latin1")
    places_df = pd.read_csv(f"{kh.get_samples_dir()}/Accidents/Places.txt", sep="\t", encoding="latin1")

    X = {
        "main_table": "Accidents",
        "tables": {
            "Accidents": (accidents_df.drop("Gravity", axis=1), "AccidentId"),
            "Vehicles": (vehicles_df, ["AccidentId", "VehicleId"]),
            "Users": (users_df, ["AccidentId", "VehicleId"]),
            "Places": (places_df, ["AccidentId"]),

        },
        "relations": [
            ("Accidents", "Vehicles"),
            ("Vehicles", "Users"),
            ("Accidents", "Places", True),
        ],
    }

Both datasets can be found in the Khiops samples directory.

Multi-table learning with the Core API
======================================

The functions in `khiops.core` that allow using multi-table datasets have the optional parameter
``additional_data_tables``. This dictionary links the secondary tables to their data file paths and
it's indexed by their **data paths** which are specified as the regular expression::

    root_table_name(`table_variable_name)*

Specifically:

- the data path for a root table is its name
- the data path for a secondary table is composed of the name of its source root table followed by
  the chain of *table variable* names leading to it. The path parts are separated by a backtick
  `````.

Types of secondary tables include:

- ``Table`` type: sub-tables in a 0:n relationship

  - Example: A "Customers" main table with a "Services" secondary table describing the services that
    each customer has subscribed to. So a customer can have zero services (inactive customer) or one
    or many.

- ``Entity`` table: sub-tables in a 0:1 relationship

  - Example: A "Customers" main table with a "Address" secondary table describing the address of
    a customer with fields such as "Street", "StreetNumber", etc. In this setting a customer can
    have at most one address.

- External data tables: Another table set (with a ``Root`` table) that is entirely loaded in
  memory

  - Example: The "Address" sub-table in the example above can point to a table "City" containing
    information about the city where the address is located. The number of cities is much smaller
    than the number of addresses so it may make sense to load it entirely in memory for efficiency
    reasons.

Note that besides the root table names the components of a data path are **table variable names**
and not *table names*. For further details about the multi-table capabilities of Khiops refer to the
documentation at `the Khiops site <https://khiops.org/setup/KhiopsGuide.pdf>`_.

The class `.DictionaryDomain` provides the helper method `.extract_data_paths` that extracts the
data paths from a given root dictionary.

.. note::
   To execute multi-table tasks, Khiops requires the data table files **to be sorted** by their key
   columns. You may use the `~.api.sort_data_table` function to preprocess your data files before
   executing these tasks.

Examples
--------

Star Schema
~~~~~~~~~~~

Let's consider the following Khiops dictionary file for the ``AccidentsSummary`` dataset
found in Khiops samples. Note that tables in this dataset are related through a *star* schema.

.. code-block:: c

    # samples/AccidentsSummary/Accidents.kdic
    Root Dictionary Accident(AccidentId)
    {
      Categorical AccidentId;
      Categorical Gravity;
      // <more variables ...>
      Table(Vehicle) Vehicles; // This is a table variable (type Table)
    };

    Dictionary Vehicle(AccidentId, VehicleId)
    {
     Categorical AccidentId;
     Categorical VehicleId;
     Categorical Direction;
     Categorical Category;
     // <more variables ...>
    };

This dictionary represents the following relational schema:

.. code-block:: text

  Accident(AccidentId)
  |
  +---1:n--- Vehicle(AccidentId, VehicleId)


In this case the ``additional_data_tables`` argument consists of only one path: that of the
secondary table ``Vehicle``. Since it is pointed by the main table ``Accident`` via the table
variable ``Vehicle`` the ``additional_data_tables`` parameter should be set as::

    additional_data_tables = {"Accident`Vehicles": f"{kh.get_samples_dir()}/Vehicles.txt"}


Snowflake Schema
~~~~~~~~~~~~~~~~

Let's now consider the dictionary file for the ``Accidents`` dataset where tables are related
through a *snowflake* schema.

.. code-block:: c

    # samples/Accidents/Accidents.kdic
    Root Dictionary Accident(AccidentId)
    {
      Categorical AccidentId;
      // The target "Gravity" is calculated from a sub-table
      // See: https://khiops.org/setup/KhiopsGuide.pdf#page=58
      Categorical	Gravity = IfC(
          G(TableSum(Vehicles, TableCount(TableSelection(Users, EQc(Gravity, "Death")))), 0),
          "Lethal", "NonLethal");
      // <more variables..>
      Entity(Place) Place; // This is a table variable type Entity: 1-1 relation)
      Table(Vehicle) Vehicles; // This is a table variable (type Table)
    };

    Dictionary Place(AccidentId)
    {
      Categorical AccidentId;
      Categorical RoadType;
      // <more variables..>
      Categorical SchoolNear;
    };


    Dictionary Vehicle(AccidentId, VehicleId)
    {
      Categorical AccidentId;
      Categorical VehicleId;
      // <more variables..>
      Table(User) Users; // This is a table variable (type Table)
    };

    Dictionary User(AccidentId, VehicleId) {
      Categorical AccidentId;
      Categorical VehicleId;
      Categorical Seat;
      Categorical Category;
      Unused Categorical Gravity; // Must be disabled since the target is a function of it
      // <more variables..>
      Numerical BirthYear;
    };


This time, the relational schema is as follows:

.. code-block:: text

    Accident(AccidentId)
    |
    +--- 1:n --- Vehicle(AccidentId, VehicleId)
    |            |
    |            +--- 1:n --- User(AccidentId, VehicleId)
    |
    +--- 1:1 --- Place(AccidentId)


The ``additional_data_tables`` parameter must be set as::

    additional_data_tables = {
        "Accident`Place": "/path/to/Places.txt",
        "Accident`Vehicles": "/path/to/Vehicles.txt",
        "Accident`Vehicles`Users": "/path/to/Users.txt"
    }

