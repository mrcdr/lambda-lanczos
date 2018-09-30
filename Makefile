CXX = g++

INCDIR = ./include
LIBDIR = ./lib
SRCDIR = ./src
LDFLAGS = -lm

HEADERS = $(INCDIR)/lambda_lanczos/lambda_lanczos.hpp $(INCDIR)/lambda_lanczos/lambda_lanczos_util.hpp

lambda_lanczos_test: $(SRCDIR)/lambda_lanczos_test/lambda_lanczos_test.cpp $(HEADERS)

	$(CXX) -I$(INCDIR) $(SRCDIR)/lambda_lanczos_test/lambda_lanczos_test.cpp  -o lambda_lanczos_test

.PHONY: clean

clean:
	rm $(LIBDIR)/*
