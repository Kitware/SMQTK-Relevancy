Pending Release Notes
=====================

Updates / New Features
----------------------

CI

* Add workflow to inherit the smqtk-core publish workflow.

Dependencies

* Remove the direct package dependency on Pillow as it is not directly utilized
  by this package.
  The locked version has been updated to address a security vulnerability.

Implementations

* Added `RankRelevancyWithRandomFeedback` and `RankRelevancywithSortedFeedback`
  implementations.

Miscellaneous

* Added a wrapper script to pull the versioning/changelog update helper from
  smqtk-core to use here without duplication.

Fixes
-----
