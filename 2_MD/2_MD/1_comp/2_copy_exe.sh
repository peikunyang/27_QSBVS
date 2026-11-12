#!/bin/bash

src_exe="./pdb/1ecq/exe_1ecq"
src_min="./pdb/1ecq/step6_minimization.inp"
root="./pdb"
run_all="3_run_all_com.sh"

echo "#!/bin/bash" > "$run_all"

for dir in "$root"/*; do
    [ -d "$dir" ] || continue

    name=$(basename "$dir")

    cp "$src_exe" "$dir/exe_$name"
    chmod +x "$dir/exe_$name"

    if [ -f "$src_min" ]; then
        cp "$src_min" "$dir/step6_minimization.inp"
    fi

    if [ -f "$dir/toppar.str" ]; then
        tmp="$dir/toppar.str.tmp"
        awk '
        {
            if ($0 ~ /^\.\.\// && $0 !~ /^\.\.\/toppar/) {
                sub(/^\./, "", $0)
            }
            print
        }' "$dir/toppar.str" > "$tmp" && mv "$tmp" "$dir/toppar.str"
    fi

    if [ -f "$dir/step4_equilibration.inp" ]; then
        sed -i 's/^r_on[[:space:]]*=.*/r_on        = 0.8/' "$dir/step4_equilibration.inp"
        sed -i 's/^r_off[[:space:]]*=.*/r_off       = 1.0/' "$dir/step4_equilibration.inp"
    fi

    if [ -f "$dir/step5_production.inp" ]; then
        sed -i 's/^r_on[[:space:]]*=.*/r_on        = 0.8/' "$dir/step5_production.inp"
        sed -i 's/^r_off[[:space:]]*=.*/r_off       = 1.0/' "$dir/step5_production.inp"
    fi

    echo "cd \"$dir\"" >> "$run_all"
    echo "./exe_$name" >> "$run_all"
    echo "cd - > /dev/null" >> "$run_all"
    echo "" >> "$run_all"

done

chmod +x "$run_all"

echo "Done"

