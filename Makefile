TESTS_DIR=tests


install:
	@echo "Installing requirements"
	pip install -r requirements.txt

validate:
	@echo "Validate"
	@echo "Running flake8"
	flake8 .
	@echo "Running mypy"
	mypy .
	@echo "Running isort"
	isort . -c
	@echo "Running black"
	black --check --line-length 79 .

test:
	python -m unittest discover $(TESTS_DIR)

run:
    python main.py
