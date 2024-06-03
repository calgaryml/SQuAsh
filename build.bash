if [ ! -d "./.build" ];
then
  mkdir ./.build
fi

cmake "$@" -B ./.build/ -S . && cmake --build ./.build/ -j8