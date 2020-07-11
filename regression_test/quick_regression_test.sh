#!/bin/bash
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
base_dir="${script_dir}/.."
prog="${base_dir}/build/test_prog"
query_input="${base_dir}/build/query_batch.fasta"
target_input="${base_dir}/build/target_batch.fasta"
output_dir="${script_dir}"
${prog} -s   -p -y local  ${query_input} ${target_input} > "${output_dir}/new_start"
${prog} -t   -p -y local  ${query_input} ${target_input} > "${output_dir}/new_tb"
${prog} -k 4 -p -y banded ${query_input} ${target_input} > "${output_dir}/new_banded"
${prog}      -p -y local  ${query_input} ${target_input} > "${output_dir}/new_standard"

echo "DIFFS"
diff "${output_dir}/new_start"    "${output_dir}/original_start"
diff "${output_dir}/new_tb"       "${output_dir}/original_tb"
diff "${output_dir}/new_standard" "${output_dir}/original_standard"
diff "${output_dir}/new_banded"   "${output_dir}/original_banded"