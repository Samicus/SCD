mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=~/libtorch
cmake --build . --config Release
cd ..
./mobile_application ../traced_DR_TANet_ref_1024x224.pt