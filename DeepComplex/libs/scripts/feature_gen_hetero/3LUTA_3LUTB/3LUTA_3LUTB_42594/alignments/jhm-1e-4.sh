#!/bin/bash
touch jhm-1e-4.running
echo "running jackhmmer job jhm-1e-4.."
/storage/htc/bdm/tools/hmmer-3.1b2-linux-intel-x86_64/binaries/jackhmmer --cpu 19 -N 5 -E 1e-4 -A jhm-1e-4.ali 3LUTA_3LUTB.jh.fasta /storage/htc/bdm/farhan/string_db/combined_strings_db/filter_combo/all_sequences.txt > jhm-1e-4-jackhmmer.log
if [ ! -f "jhm-1e-4.ali" ]; then
   mv jhm-1e-4.running jhm-1e-4.failed
   echo "jackhmmer job jhm-1e-4 failed!"
   exit
fi
/storage/htc/bdm/farhan/DeepComplex/feature_gen_hetero_dncon2/reformat.pl -l 1500 -d 1500 sto a3m jhm-1e-4.ali jhm-1e-4.a3m
egrep -v "^>" jhm-1e-4.a3m | sed 's/[a-z]//g' > jhm-1e-4.aln
if [ -f "jhm-1e-4.aln" ]; then
   rm jhm-1e-4.running
   rm jhm-1e-4-jackhmmer.log
   rm jhm-1e-4.ali
   echo "jackhmmer job jhm-1e-4 done."
   exit
fi
echo "Something went wrong! jhm-1e-4.aln file not present!"
mv jhm-1e-4.running jhm-1e-4.failed
