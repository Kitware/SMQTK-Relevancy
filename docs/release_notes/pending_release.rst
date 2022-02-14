Pending Release Notes
=====================

Updates / New Features
----------------------

Documentation

* Updated CONTRIBUTING.md to reference smqtk-core's CONTRIBUTING.md file.

Fixes
-----

CI

* Modified CI unittests workflow to run for PRs targetting branches that match
  the `release*` glob.

Dependency Versions

* Updated the developer dependency and locked version of ipython to address a
  security vulnerability.

* Removed `jedi = "^0.17"` requirement and updated to `ipython = "^7.17.3"`
  since recent ipython update appropriately addresses the dependency.
