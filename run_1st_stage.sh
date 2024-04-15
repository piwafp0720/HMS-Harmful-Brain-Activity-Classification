# train model A
python src/train_offset.py -c configs/config_model_A.yml -o dataset.fold=0
python src/train_offset.py -c configs/config_model_A.yml -o dataset.fold=1
python src/train_offset.py -c configs/config_model_A.yml -o dataset.fold=2
python src/train_offset.py -c configs/config_model_A.yml -o dataset.fold=3

# train model B
python src/train_offset.py -c configs/config_model_B.yml -o dataset.fold=0
python src/train_offset.py -c configs/config_model_B.yml -o dataset.fold=1
python src/train_offset.py -c configs/config_model_B.yml -o dataset.fold=2
python src/train_offset.py -c configs/config_model_B.yml -o dataset.fold=3

# train model C
python src/train_offset.py -c configs/config_model_C.yml -o dataset.fold=0
python src/train_offset.py -c configs/config_model_C.yml -o dataset.fold=1
python src/train_offset.py -c configs/config_model_C.yml -o dataset.fold=2
python src/train_offset.py -c configs/config_model_C.yml -o dataset.fold=3