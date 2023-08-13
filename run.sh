gpu_n=$1
DATASET=$2
struct=$3

seed=1
BATCH_SIZE=32
SLIDE_WIN=10
dim=128
out_layer_num=2
SLIDE_STRIDE=1
topk=4
out_layer_inter_dim=128
val_ratio=0.3
decay=0


path_pattern="${DATASET}"
COMMENT="${DATASET}"

EPOCH=50
report='best'

if [[ "$gpu_n" == "cpu" ]]; then
    python3 main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -device 'cpu'\
        -graph_structure $struct
else
    CUDA_VISIBLE_DEVICES=$gpu_n  python3 main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk
fi
