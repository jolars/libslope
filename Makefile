BUILD_DIR = build
TEST_DIR = test

all: configure build

.PHONY: configure
configure:
	cmake -B $(BUILD_DIR) -S .

.PHONY: build
build:
	cmake --build $(BUILD_DIR)

.PHONY: python
python:
	cmake -B $(BUILD_DIR) -S . -DBUILD_PYTHON_BINDINGS=ON
	cmake --build $(BUILD_DIR)

.PHONY: release
release: clean
	cmake -B $(BUILD_DIR) -S . -DCMAKE_BUILD_TYPE=Release
	$(MAKE) build

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: install
install:
	cmake --install $(BUILD_DIR)

.PHONY: test
test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure
