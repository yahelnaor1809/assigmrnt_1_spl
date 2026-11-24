# Default target
all: bin/server

# Linking the executable
bin/server: bin/Main.o bin/Server.o bin/Model.o bin/Request.o bin/GPU.o bin/MemoryBlock.o bin/SchedulingPolicy.o
	g++ -o bin/server bin/Main.o bin/Server.o bin/Model.o bin/Request.o bin/GPU.o bin/MemoryBlock.o bin/SchedulingPolicy.o

# Compiling Main.cpp
bin/Main.o: src/Main.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/Main.cpp -o bin/Main.o

# Compiling Server.cpp
bin/Server.o: src/Server.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/Server.cpp -o bin/Server.o

# Compiling Model.cpp
bin/Model.o: src/Model.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/Model.cpp -o bin/Model.o

# Compiling Request.cpp
bin/Request.o: src/Request.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/Request.cpp -o bin/Request.o

# Compiling GPU.cpp
bin/GPU.o: src/GPU.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/GPU.cpp -o bin/GPU.o

# Compiling MemoryBlock.cpp
bin/MemoryBlock.o: src/MemoryBlock.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/MemoryBlock.cpp -o bin/MemoryBlock.o

# Compiling SchedulingPolicy.cpp
bin/SchedulingPolicy.o: src/SchedulingPolicy.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/SchedulingPolicy.cpp -o bin/SchedulingPolicy.o

# Cleaning compiled files
clean:
	rm -f bin/*.o bin/server