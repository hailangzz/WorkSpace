python tools/export_model.py -c configs/rec/CRNN_ctc.yml -o Global.checkpoints="./output/rec_CRNN/best_accuracy" Global.save_inference_dir="./inference/CRNN"

python tools/export_model.py -c configs/rec/CRNN_ctc_enchance.yml -o Global.checkpoints="./output/rec_CRNN_ok3/best_accuracy" Global.save_inference_dir="./inference/CRNN"



python tools/export_model.py -c configs/rec/CRNN_ctc_enchance.yml -o Global.checkpoints="./pretrain_models/ch_rec_r34_vd_crnn_enchance/best_accuracy" Global.save_inference_dir="./inference/CRNN"

