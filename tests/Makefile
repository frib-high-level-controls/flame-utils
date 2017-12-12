.PHONY: test #clean

test:
	PYTHONPATH=../:$(PYTHONPATH) \
	nosetests --with-coverage --cover-package=flame_utils \
		--with-id -v --logging-level=INFO \
		--exclude-dir-file="skiptests"

#clean:
#	[ -e /tmp/phantasy_data ] && rm -rf /tmp/phantasy_data
