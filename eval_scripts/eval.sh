experiment=$1
anserini_path=$2
qrels_file=$3

echo "Experiment: ${experiment}"

echo "1S:"
${anserini_path}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 "${anserini_path}/src/main/resources/topics-and-qrels/${qrels_file}" "runs/run.${experiment}.cv.a"

echo "2S:"
${anserini_path}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 "${anserini_path}/src/main/resources/topics-and-qrels/${qrels_file}" "runs/run.${experiment}.cv.ab"

echo "SS:"
${anserini_path}/eval/trec_eval.9.0.4/trec_eval -M1000 -m map -m P.20 "${anserini_path}/src/main/resources/topics-and-qrels/${qrels_file}" "runs/run.${experiment}.cv.abc"