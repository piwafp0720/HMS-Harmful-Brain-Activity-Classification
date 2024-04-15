# train model A
python src/train_offset.py -c configs/config_model_A.yml -o dataset.fold=0 \
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_0-model_A>
python src/train_offset.py -c configs/config_model_A.yml -o dataset.fold=1
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_1-model_A>
python src/train_offset.py -c configs/config_model_A.yml -o dataset.fold=2
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_2-model_A>
python src/train_offset.py -c configs/config_model_A.yml -o dataset.fold=3
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_3-model_A>

# train model B
python src/train_offset.py -c configs/config_model_B.yml -o dataset.fold=0
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_0-model_B>
python src/train_offset.py -c configs/config_model_B.yml -o dataset.fold=1
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_1-model_B>
python src/train_offset.py -c configs/config_model_B.yml -o dataset.fold=2
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_2-model_B>
python src/train_offset.py -c configs/config_model_B.yml -o dataset.fold=3
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_3-model_B>

# train model C
python src/train_offset.py -c configs/config_model_C.yml -o dataset.fold=0
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_0-model_C>
python src/train_offset.py -c configs/config_model_C.yml -o dataset.fold=1
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_1-model_C>
python src/train_offset.py -c configs/config_model_C.yml -o dataset.fold=2
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_2-model_C>
python src/train_offset.py -c configs/config_model_C.yml -o dataset.fold=3
dataset.group=high_vote \
model.load_checkpoint=<write path to the 1st_stage-fold_3-model_C>
