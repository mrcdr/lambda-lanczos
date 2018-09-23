CXX = g++
AR = ar

CXXFLAGS = -fPIC

INCDIR = ./include
SRCDIR = ./src
OBJDIR = ./obj
LIBDIR = ./lib
LDFLAGS = -lm
LIBS =

all:	$(LIBDIR)/liblambda_lanczos.so $(LIBDIR)/liblambda_lanczos.a

$(LIBDIR)/liblambda_lanczos.so: $(OBJDIR)/lambda_lanczos.o $(OBJDIR)/lambda_lanczos_util.o
	$(CXX) -shared -fPIC $(OBJDIR)/lambda_lanczos.o $(OBJDIR)/lambda_lanczos_util.o -o $(LIBDIR)/liblambda_lanczos.so

$(LIBDIR)/liblambda_lanczos.a: $(OBJDIR)/lambda_lanczos.o $(OBJDIR)/lambda_lanczos_util.o
	$(AR) rcs $(LIBDIR)/liblambda_lanczos.a $(OBJDIR)/lambda_lanczos.o $(OBJDIR)/lambda_lanczos_util.o

lambda_lanczos_test: $(SRCDIR)/lambda_lanczos_test/lambda_lanczos_test.cpp $(LIBDIR)/liblambda_lanczos.a
	$(CXX) -I$(INCDIR) $(LIBS) $(SRCDIR)/lambda_lanczos_test/lambda_lanczos_test.cpp $(LIBDIR)/liblambda_lanczos.a -o lambda_lanczos_test

$(OBJDIR)/%.o: $(SRCDIR)/lambda_lanczos/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -o $@ -c $<

.PHONY: clean

clean:
	rm $(OBJDIR)/*
	rm $(LIBDIR)/*

$(OBJDIR)/lambda_lanczos.o: $(SRCDIR)/lambda_lanczos/lambda_lanczos.cpp $(INCDIR)/lambda_lanczos/lambda_lanczos.hpp
$(OBJDIR)/lambda_lanczos_util.o: $(SRCDIR)/lambda_lanczos/lambda_lanczos_util.cpp $(SRCDIR)/lambda_lanczos/lambda_lanczos_util.hpp
