#!/bin/bash

targets=(
	"matrix_vec_product"
	"matrix_vec_product_omp"
	"surface_kernel"
	"surface_kernel_omp"  
	"volume_kernel"
	"volume_kernel_omp"
)

for target in ${targets[@]}
do
	make $target
done

for target in ${targets[@]} 
do
	for proc_num in 1 2 4 8 16
	do
		let factor=${proc_num}*10
		~/work/GPI-2/bin/gaspi_run -m machine -n ${proc_num} ${target} -dtopo topo 10
	done
done
