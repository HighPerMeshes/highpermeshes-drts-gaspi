#!/bin/bash

#
# Copyright (c) Fraunhofer ITWM - <http://www.itwm.fraunhofer.de/>, 2018
#
# This file is part of HighPerMeshesDRTS, the HighPerMeshes distributed runtime
# system.
#
# The HighPerMeshesDRTS is free software; you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.
#
# HighPerMeshesDRTS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HighPerMeshesDRTS. If not, see <http://www.gnu.org/licenses/>.
#
# run_distributed_tests.sh
#

GTEST_PARAMS=--gtest_output="xml:results.xml"
GTEST_TEST=""
if [ $# -gt 0 ]; then
  GTEST_TEST=--gtest_filter=*$2*
fi

# Generate machinefile
rm -f machine_file
for I in  $(seq 1 $1)
do
  hostname >> machine_file
done

# Run tests
gaspi_cleanup -m machine_file
gaspi_run -m machine_file -n $1 tests ${GTEST_PARAMS} ${GTEST_TEST} ${TEST_CFG}
gaspi_cleanup -m machine_file