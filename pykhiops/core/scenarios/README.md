Khiops Scenarios Folder
=======================

The scenarios necessary to implement the API are contained in subfolders named after a
Khiops version. pyKhiops looks for the latest version that is compatible with that of
its Khiops backend.  For example if the backend version is 10.0.1 then the scenarios
used will be those of the 10.0 folder.

Note that the version is not necessarily linked to a public release. This is necessary
to be able to test pyKhiops with development versions of Khiops.

Note also that a folder may not contain all possible scenarios, only those that changed
from one version to another.
