BUILD_DIR = build
TEST_DIR = test

all: configure build

.PHONY: configure
configure:
	cmake -B $(BUILD_DIR) -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

.PHONY: debug
configure:
	cmake -B $(BUILD_DIR) -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

.PHONY: build
build:
	cmake --build $(BUILD_DIR)

.PHONY: docs
docs:
	cmake -B $(BUILD_DIR) -S . -DBUILD_DOCS=ON -DBUILD_TESTING=OFF
	cmake --build $(BUILD_DIR)

.PHONY: coverage
coverage:
	cmake -B $(BUILD_DIR) -S . -DBUILD_DOCS=ON -DBUILD_TESTING=ON -DENABLE_COVERAGE=ON
	cmake --build $(BUILD_DIR)
	ctest --test-dir $(BUILD_DIR) --output-on-failure

.PHONY: release
release: clean
	cmake -B $(BUILD_DIR) -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
	$(MAKE) build
	$(MAKE) install

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: install
install:
	cmake --install $(BUILD_DIR)

.PHONY: test
test:
	cmake -B $(BUILD_DIR) -S . -DBUILD_DOCS=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
	cmake --build $(BUILD_DIR)
	ctest --test-dir $(BUILD_DIR) --output-on-failure
