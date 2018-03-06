.PHONY: test

test:
	PYTHONPATH=../:$(PYTHONPATH) \
	nosetests3 --with-coverage --cover-package=flame_utils \
		--with-id -v --logging-level=INFO
