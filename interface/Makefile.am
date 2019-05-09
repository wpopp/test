include $(top_srcdir)/cython.am

instdir = $(pythondir)/TuckerDingens
inst_PYTHON = __init__.py
inst_LTLIBRARIES = TuckerDingens.la
TuckerDingens_la_SOURCES = TuckerDingens.pyx cpython_TuckerDingens.cpp

PACKAGE_STRING+= python interface

CLEANFILES += TuckerDingens.cpp

AM_CXXFLAGS = -pthread -Wl,-O1 -Wl,-Bsymbolic-functions \
              -fno-strict-aliasing -g -fwrapv -O2 -Wall \
              -Wformat -Werror=format-security

AM_CPPFLAGS = @PYTHON_INCLUDES@ -I$(top_srcdir)/src
AM_LDFLAGS = -module -omit-version

EXTRA_DIST = \
	cpython_TuckerDingens.hpp \
	TuckerDingens.pxd \
	TuckerDingens.pyx

uninstall-hook:
	-rmdir ${DESTDIR}/${instdir}
	-rmdir ${DESTDIR}/${pythondir}
