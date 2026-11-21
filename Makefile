# slm-pt — local build (Termux / Linux / macOS)
# Usage: make -j        (builds build/slmpt_train, build/slmpt_chat, build/slmpt_gradcheck)

CXX ?= g++
BUILD ?= build
CXXFLAGS ?= -std=c++17 -O3 -funroll-loops -march=native -funsafe-math-optimizations -ffp-contract=fast -Iinclude -pthread
LDFLAGS ?= -pthread

LIB_SRC := $(wildcard src/*.cpp)
LIB_OBJ := $(patsubst src/%.cpp,$(BUILD)/%.o,$(LIB_SRC))
APPS := train chat gradcheck
APP_BIN := $(addprefix $(BUILD)/slmpt_,$(APPS))

all: $(APP_BIN)

$(BUILD)/%.o: src/%.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/slmpt_%: apps/%.cpp $(LIB_OBJ)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(LIB_OBJ) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD)

.PHONY: all clean
