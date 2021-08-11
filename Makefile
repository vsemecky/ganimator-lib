all: clean venv

clean:
	rm -rf venv

venv:
	python3 -m venv --clear venv
	python3 -m pip install -r requirements.txt
