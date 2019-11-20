#! /bin/bash

# Make sure HugePages are enabled.
# Running the following commands as root on an ubuntu system should do the trick:
# echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
# mount -t hugetlbfs nodev /dev/hugepages

# To build QATzip:
# ./configure --with-ICP-ROOT=<path-to-qat>
# make -j $(nproc)

mkdir silesia
cp silesia.zip silesia/
cd silesia/
unzip silesia.zip
rm silesia.zip
