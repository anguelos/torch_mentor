### Dataset curation
In order to utilise data they must be ingested

```bash
export PYTHONPATH="./src"
./bin/ingest_dataset -inputs ./data/nonpapal/* -output_dir ./data/ingested/nonpapal
./bin/ingest_dataset -inputs ./data/papal/* -output_dir ./data/ingested/papal
```



### TODO
* make archives (zip, rar, etc) ingestable for folder datasets


#### Simple training of a classifier
```bash
git clone git@github.com:anguelos/mentor.git
cd mentor
# Download precomputed dataset version (URL works only inside ZIM)
wget http://143.50.30.95:7999/datasets/papal_v1.zip
# unzip it
unzip papal_v1.zip
# train a classifier
PYTHONPATH="./src" ./bin/mnt_train -h
```
