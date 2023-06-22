Multi-Table Tasks
=================
Khiops supervised estimators allow learning on multi-table datasets. This drastically reduces the
time spent on the feature engineering phase. The specification of these datasets differ between
`pykhiops.core` functions and `pykhiops.sklearn` estimator classes. See below how to do it in each
case.

.. note::
   In a multi-table dataset all tables must have a key. This is because they act as `foreign keys
   <https://en.wikipedia.org/wiki/Foreign_key>`_ to associate the records of each table.

``pykhiops.core`` Module
------------------------
The functions in `pykhiops.core` that allow using multi-table datasets have the optional parameter
``additional_data_tables``. This dictionary links the secondary tables to their data file paths and
it's indexed by their **data paths** which are specified as the regular expression::

    root_table_name(`table_variable_name)*

Specifically:
    - the data path for a root table is its name
    - the data path for secondary table is composed of the name of its source root table followed
      by the chain of *table variable* names leading to it. The path elements are separated by a
      backtick `````.

Secondary tables include:
    - ``Table`` type: sub-tables in a 0-n relationship
        - Example: A "Customers" main table with a "Services" secondary table describing the
          services that he has subscribed. So customer can have zero services (inactive customer) or
          one or many.

    - ``Entity`` table: sub-tables in a 0-1 relationship
        - Example: A "Customers" main table with a "Address" secondary table describing the address
          of a customer with fields such as "Street", "StreetNumber", etc. In this setting a
          customer can have at most one address.

    - External data tables: Another table set (each with a ``Root`` table) that is entirely loaded
      in memory

        - Example: The "Address" sub-table in the example above can point to a table "City"
          containing the information about the city where the address is located. The number of
          cities is much smaller than the number of addresses so it may make sense to load it
          entirely in memory for efficiency reasons.

Note that besides the root table names the components of a data path are **table variable names**
and not *table names*. For further details about the multi-table capabilities of Khiops refer to the
documentation at `the Khiops site <https://www.khiops.com/html/KhiopsGuide.htm>`_.

The class `.DictionaryDomain` provides the helper method `.extract_data_paths` that extracts the
data paths from a given root dictionary.

.. note::
   To execute multi-table tasks, Khiops requires that the data table files **must be sorted** by
   their keys. You may use the `~.api.sort_data_table` function to preprocess your files before
   executing these tasks.

Examples
~~~~~~~~

Two Tables
^^^^^^^^^^

Let's consider the following Khiops dictionary file for the ``AccidentsSummary`` dataset
found in Khiops samples:

.. code-block:: c

    # samples/AccidentsSummary/Accidents.kdic
    Root Dictionary Accident(AccidentId)
    {
      Categorical AccidentId;
      Categorical Gravity;
      <more variables ...>
      Table(Vehicle) Vehicles; // This is a table variable (type Table)
    };

    Dictionary Vehicle(AccidentId, VehicleId)
    {
     Categorical AccidentId;
     Categorical VehicleId;
     Categorical Direction;
     Categorical Category;
     <more variables ...>
    };

This dictionary represents the following relational schema:

.. code-block:: text

    Accident
    |
    +----1:n---- Vehicle

In this case the ``additional_data_tables`` argument consists of only one path: that of the
secondary table ``Vehicle``. Since it is pointed by the main table ``Accident`` via the table
variable ``Vehicle`` the ``additional_data_tables`` parameter should be set as::

    additional_data_tables = {"Accident`Vehicles": f"{pk.get_samples_dir()}/Vehicles.txt"}


Many Tables
^^^^^^^^^^^
Let's now consider the dictionary file for the ``Customer`` dataset:

.. code-block:: c

    # samples/Customer/Customer.kdic
    Root Dictionary Customer(id_customer)
    {
        Categorical id_customer;
        Categorical Name;
        Table(Service) Services; // This is a table variable (type Table)
        Entity(Address) Address; // This is a table variable (type Entity: 1-1 relation)
    };

    Dictionary Address(id_customer)
    {
        Categorical id_customer;
        Numerical StreetNumber;
        Categorical StreetName;
        Categorical id_city;
    };

    Dictionary Service(id_customer, id_product)
    {
        Categorical id_customer;
        Categorical id_product;
        Date SubscriptionDate;
        Table(Usage) Usages; // This is a table variable (type Table)
    };

    Dictionary Usage(id_customer, id_product)
    {
        Categorical id_customer;
        Categorical id_product;
        Date Date;
        Time Time;
        Numerical Duration;
    };

This time, relational schema is as follows:

.. code-block:: text

    Customer
    |
    +----1:1---- Address
    |
    +----1:n---- Service
                 |
                 +----1:n---- Usage


The ``additional_data_tables`` parameter must be set as::

    additional_data_tables = {
        "Customer`Address": "/path/to/Address.txt",
        "Customer`Services": "/path/to/Service.txt",
        "Customer`Services`Usages": "/path/to/Usage.txt"
    }


``pykhiops.sklearn`` Module
---------------------------

The supervised estimators classes in `pykhiops.sklearn` handle multi-table datasets with a special
``X`` input. Specifically, instead of `pandas.DataFrame`, ``X`` must be a ``dict`` containing the
dataset schema in the following way::

   X = {
      "main_table": <name of the main table>,
      "tables" : {
          <name of table 1>: (<dataframe of table 1>, <key of table 1>),
          <name of table 2>: (<dataframe of table 2>, <key of table 2>),
          ...
       }
   }

The keys of tables are either a single column name, or a tuple containing the columns composing the
key.

.. note::
   pyKhiops ``sklearn`` estimators support a limited number of multi-table features. In
   particular:

     - pyKhiops ``sklearn`` estimators currently handle only *star* schemas: the secondary tables
       must be directly linked to the root table.
     - ``Entity`` (``1:1`` table relations) are not currently supported.
     - External data tables are not currently supported.

   These features will be available in upcoming releases.

Example
~~~~~~~
For the ``AccidentsSummary`` dataset above the input ``X`` can be built as follows::

  accidents_df = pd.read_csv(f"{pk.get_samples_dir()}/AccidentsSummary/Accidents.txt", sep="\t", encoding="latin1")
  vehicles_df = pd.read_csv(f"{pk.get_samples_dir()}/AccidentsSummary/Vehicles.txt", sep="\t", encoding="latin1")

  X = {
      "main_table" : "Accident",
      "tables": {
          "Accident": (accidents_df.drop("Gravity", axis=1), "AccidentId"),
          "Vehicle": (vehicles_df, ["AccidentId", "VehicleId"])
      }
  }

