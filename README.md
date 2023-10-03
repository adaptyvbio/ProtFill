# ProtFill

downloading the data:
```
proteinflow download --tag 20230102_stable

proteinflow download --tag 20230626_sabdab --skip_splitting
rm -r data/proteinflow_20230626_sabdab/splits_dict/
mv data/splits_dict data/proteinflow_20230626_sabdab/
proteinflow split --tag 20230626_sabdab
```