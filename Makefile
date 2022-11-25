
install:
	python setup.py install --user

test: install
	python setup.py test

clean:
	rm -rf dist/ build/
