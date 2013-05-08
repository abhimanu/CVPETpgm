CPP = clang++
FLAGS =  -O4
LIBS = -larmadillo -lgflags

HPPFILES = util.hpp approx_solver.hpp
CPPFILES = util.cpp approx_solver.cpp

all: main_binary 

#compress_data: compress_data.cpp
#	$(CPP) $^ -o $@ $(FLAGS)	

main_binary: main_binary.cpp $(CPPFILES)
	$(CPP) $^ -o $@ $(FLAGS) $(LIBS)

main_yeast: main_yeast.cpp $(CPPFILES)
	$(CPP) $^ -o $@ $(FLAGS) $(LIBS)
	
clean:
	rm -f main_binary main_yeast 
