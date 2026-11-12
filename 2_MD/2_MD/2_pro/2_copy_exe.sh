#!/bin/bash

src_exe="./pdb/1ecq/exe_1ecq"
root="./pdb"
run_all="3_run_all.sh"

echo "#!/bin/bash" > $run_all

for dir in "$root"/*; do
    [ -d "$dir" ] || continue

    name=$(basename "$dir")

    # copy + rename
    cp "$src_exe" "$dir/exe_$name"
    chmod +x "$dir/exe_$name"

    # 寫入總執行 script
    echo "cd $dir" >> $run_all
    echo "./exe_$name" >> $run_all
    echo "cd - > /dev/null" >> $run_all
    echo "" >> $run_all

done

chmod +x $run_all

echo "Done: exe copied + run_all.sh generated"

