CXX = g++

INCDIR = ./include/lambda_lanczos
SRCDIR = ./src
LDFLAGS = -lm

HEADERS = $(INCDIR)/lambda_lanczos.hpp $(INCDIR)/lambda_lanczos_util.hpp

lambda_lanczos_test: $(SRCDIR)/lambda_lanczos_test/lambda_lanczos_test.cpp $(HEADERS)
	$(CXX) -I$(INCDIR) $(SRCDIR)/lambda_lanczos_test/lambda_lanczos_test.cpp -o lambda_lanczos_test

sample1: $(SRCDIR)/samples/sample1_simple.cpp $(HEADERS)
	$(CXX) -I$(INCDIR) $(SRCDIR)/samples/sample1_simple.cpp -o sample1

sample2: $(SRCDIR)/samples/sample2_sparse.cpp $(HEADERS)
	$(CXX) -I$(INCDIR) $(SRCDIR)/samples/sample2_sparse.cpp -o sample2

sample3: $(SRCDIR)/samples/sample3_dynamic.cpp $(HEADERS)
	$(CXX) -I$(INCDIR) $(SRCDIR)/samples/sample3_dynamic.cpp -o sample3

sample4: $(SRCDIR)/samples/sample4_use_Eigen_library.cpp $(HEADERS)
	$(CXX) -I$(INCDIR) $(SRCDIR)/samples/sample4_use_Eigen_library.cpp -o sample4

det_offset: $(SRCDIR)/determine_eigenvalue_offset/determine_eigenvalue_offset.cpp $(HEADERS)
	$(CXX) -I$(INCDIR) $(SRCDIR)/determine_eigenvalue_offset/determine_eigenvalue_offset.cpp -o det_offset

