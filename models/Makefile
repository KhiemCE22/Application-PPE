TARGET = simple_check
CXX = g++
CXXFLAGS = -Wall -O2

all: $(TARGET)

$(TARGET): check_ncnn.cpp
	$(CXX) $(CXXFLAGS) check_ncnn.cpp -o $(TARGET)
	@echo "Build complete! No NCNN libraries were harmed."

clean:
	rm -f $(TARGET)