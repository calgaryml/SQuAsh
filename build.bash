if [ ! -d "./bin" ];
then
  mkdir ./bin
fi

if (($# != 1)); then
    echo "Usage: $0 <build_type>"
    exit 1
fi

build_dir="bin/$1"
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE="$1"
cmake --build "$build_dir"