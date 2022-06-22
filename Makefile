.PHONY: deps_table_update modified_only_fixup extra_quality_checks quality style fixup fix-copies test test-examples docs

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := examples tests src utils

modified_only_fixup:
	$(eval modified_py_files := $(shell python utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		black $(modified_py_files); \
		isort $(modified_py_files); \
		flake8 $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi

# Update src/transformers/dependency_versions_table.py

deps_table_update:
	@python setup.py deps_table_update

# autogenerating code

autogenerate_code: deps_table_update
	python utils/class_mapping_update.py

# Check that source code meets quality standards

# NOTE FOR adapter-transformers: The following check is skipped as not all copies implement adapters yet
# python utils/check_copies.py
# python utils/check_table.py
# python utils/check_dummies.py
extra_quality_checks:
	python utils/check_repo.py
	python utils/check_inits.py
	python utils/check_adapters.py

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	python utils/custom_init_isort.py --check_only
	flake8 $(check_dirs)
	${MAKE} extra_quality_checks

# Format source code automatically and check is there are any problems left that need manual fixing

extra_style_checks:
	python utils/custom_init_isort.py
	python utils/style_doc.py src/transformers docs/source --max_len 119

# this target runs checks on all files and potentially modifies some of them
style:
	black $(check_dirs)
	isort $(check_dirs)
	${MAKE} autogenerate_code
	${MAKE} extra_style_checks

# Super fast fix and check target that only works on relevant modified files since the branch was made

fixup: modified_only_fixup extra_style_checks autogenerate_code extra_quality_checks

# Make marked copies of snippets of codes conform to the original

fix-copies:
	python utils/check_copies.py --fix_and_overwrite
	python utils/check_table.py --fix_and_overwrite
	python utils/check_dummies.py --fix_and_overwrite

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run the adapter tests

test-adapter:
	python -m pytest -n auto --dist=loadfile -s -v\
		-k test_adapter\
		--ignore-glob='tests/test_tokenization*'\
		--ignore-glob='tests/test_processor*'\
		./tests/

# Run a reduced test suite in the CI pipeline of adapter-transformers
test-reduced:
	python utils/run_tests.py

# Run tests for examples

test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/

# Run tests for SageMaker DLC release

test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]
	TEST_SAGEMAKER=True python -m pytest -n auto  -s -v ./tests/sagemaker


# Check that docs can build

docs:
	cd docs && make html SPHINXOPTS="-W -j 4"

# Release stuff

pre-release:
	python utils/release.py

pre-patch:
	python utils/release.py --patch

post-release:
	python utils/release.py --post_release

post-patch:
	python utils/release.py --post_release --patch