CPP = clang++
FLAGS = -larmadillo -lgflags -O4

all: compress_data map_est

compress_data: compress_data.cpp
	$(CPP) $^ -o $@ $(FLAGS)	

map_est: map_est.cpp
	$(CPP) $^ -o $@ $(FLAGS)
	
clean:
	rm -f compress_data map_est
