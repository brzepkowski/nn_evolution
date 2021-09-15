#!/bin/bash

CATALOG_NAMES=2_layers_simple_*

rm -r Best_results
mkdir Best_results

for catalog_name in $CATALOG_NAMES
do
	cd "$catalog_name"
	result=$(less Meta_information_N\=7.txt | grep "Test accuracy:")
	arr=($result)
	accuracy=${arr[2]}
	cd ..
	if (( $(echo "$accuracy > 0.99" |bc -l) )); then
		echo "Accuracy: $accuracy -> $catalog_name"
		cp -r "$catalog_name" Best_results
	fi
done
