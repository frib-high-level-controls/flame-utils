.PHONY: test

test:
	PYTHONPATH=../:$(PYTHONPATH) \
	nosetests --with-coverage --cover-package=flame_utils \
		--with-id -v --logging-level=INFO \
		--exclude-dir-file="skiptests"
