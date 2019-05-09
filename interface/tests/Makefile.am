
TESTS = test_sequential_TuckerDingens.py

AM_PY_LOG_DRIVER_FLAGS=$(DEBUGFLAG)

$(TESTS:%.py=%.log): all-am

TEST_EXTENSIONS = .py
PY_LOG_COMPILER = $(PYTHON)
AM_TESTS_ENVIRONMENT = \
    export PYTHONPATH='.:..:$(top_srcdir)/TuckerDingens';

EXTRA_DIST = \
	$(TESTS)

$(TESTS:%.py=%.log): P
.PHONY: P
