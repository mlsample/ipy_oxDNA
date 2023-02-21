#!/bin/bash

wget http://membrane.urmc.rochester.edu/sites/default/files/wham/wham-release-2.0.11.tgz
tar -xf wham-release-2.0.11.tgz
rm wham-release-2.0.11.tgz
sed -i "s/#define k_B 0.001982923700 \/\/ Boltzmann's constant in kcal\/mol K/\/\/#define k_B 0.001982923700 \/\/ Boltzmann's constant in kcal\/mol K/g" ./wham/wham/wham.h
sed -i "s/\/\/#define k_B 1.0  \/\/ Boltzmann's constant in reduced units/#define k_B 1.0  \/\/ Boltzmann's constant in reduced units/g" ./wham/wham/wham.h

sed -i "s/#define k_B 0.001982923700 \/\/ Boltzmann's constant in kcal\/mol K/\/\/#define k_B 0.001982923700 \/\/ Boltzmann's constant in kcal\/mol K/g" ./wham/wham-2d/wham-2d.h
sed -i "s/\/\/#define k_B 1.0  \/\/ Boltzmann's constant in reduced units/#define k_B 1.0  \/\/ Boltzmann's constant in reduced units/g" ./wham/wham-2d/wham-2d.h

cd ./wham/wham
make clean
make

cd ../wham-2d
make clean
make
