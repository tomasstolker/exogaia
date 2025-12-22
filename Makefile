.PHONY: help pypi pypi-test docs coverage test clean

help:
	@echo "pypi - submit to PyPI server"
	@echo "pypi-check - check the distribution for PyPI"
	@echo "pypi-test - submit to TestPyPI server"
	@echo "docs - generate Sphinx documentation"
	@echo "coverage - check code coverage"
	@echo "test - run unit tests"
	@echo "clean - remove artifacts"

pypi:
	python -m build
	twine upload dist/*

pypi-check:
	python -m build
	twine check dist/*

pypi-test:
	python -m build
	twine upload -r testpypi dist/*

docs:
	rm -f docs/exogaia.rst
	sphinx-apidoc -o docs exogaia
	cd docs/
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

coverage:
	coverage run --source=exogaia -m pytest
	coverage report -m

test:
	pytest --cov=exogaia/ --cov-report=xml

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	rm -f .coverage*
	rm -f coverage.xml
	rm -f exogaia/_version.py
	rm -rf .pytest_cache/
	rm -rf docs/_build/
	rm -rf docs/.ipynb_checkpoints/
	rm -rf docs/multinest/
	rm -f docs/*.png
	rm -f docs/*.csv
	rm -f docs/exogaia.pkl
	rm -rf build/
	rm -rf dist/
	rm -rf exogaia.egg-info/
	rm -rf htmlcov/
	rm -rf .tox/
