CPP = g++
INCFLAGS =
LIBS = -lboost_program_options -larmadillo -llapack -lblas
CPPFLAGS = -g $(INCFLAGS) -Wall -O2
HEADERS=$(wildcard *.h**)

all: variational

variational: src/variational.cpp
	mkdir -p bin
	$(CPP) $(CPPFLAGS) src/variational.cpp $(LIBS) -o bin/variational
clean:
	rm -rf bin