Pending Release Notes
=====================

Updates / New Features
----------------------

Documentation

* Updated CONTRIBUTING.md to reference smqtk-core's CONTRIBUTING.md file.

Fixes
-----

CI

* Also run CI unittests for PRs targetting branches that match the `release*`
  glob.

Dependency Versions

* Update the developer dependency and locked version of ipython to address a
  security vulnerability.

* Removed `jedi = "^0.17"` requirement and update to `ipython = "^7.17.3"`
  since recent ipython update appropriately addresses the dependency.
