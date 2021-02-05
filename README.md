# Entity-Relation Extraction as Multi-stage&turn Question Answering


**input**
```
"HIV 感染@目前已发现每 4 或 8 周给药一次的cabotegravir+利匹韦林的注射治疗方案与cabotegravir+利匹韦林口服方案在维持病毒抑制达 96 周方面的有效性相同，并且被患者广泛接受且耐受性良好。"
```

**output** 
```
"spo_list": [{"Combined": false, "predicate": "药物治疗", "subject": "HIV 感染", "subject_type": "疾病", "object": {"@value": "cabotegravir+利匹韦林的注射"}, "object_type": {"@value": "药物"}}, {"Combined": false, "predicate": "药物治疗", "subject": "HIV 感染", "subject_type": "疾病", "object": {"@value": "cabotegravir+利匹韦林口服"}, "object_type": {"@value": "药物"}}]
```


将schema中的所有关系设置为问题，问题和文本拼接后输入模型抽取实体，把实体抽取任务转换成序列标注任务，

## Document description

|name|description|
|-|-|
|run_qa.py|entrance for model for training and prediction |
|data/data_utils.py|Prepare formatted data for the model|
|data/dataset.py|dataset defination for QA format|
|models/bert_mrc.py|our model framework for IE |
|layers/classifier.py|classfier layer defination|


### Getting Started

#### Environment Requirements
+ python 3.6+
+ Pytorch 1.5+



#### Step 1: Download the training data, put it to data dir



#### Step2: Data preprocessing
```
python data/data_utils.py
```

#### Step3： Model training


```

python run_qa.py \

--do_train=true \
--do_predict=false \
--cuda=true \
--data_dir=./data \
--config_path=./pretrained_model/chinese-bert_chinese_wwm_pytorch/bert_config.json  \
--bert_model=./pretrained_model/chinese-bert_chinese_wwm_pytorch \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-3 \
--num_train_epochs=50 \
--use_crf=true
--output_dir=./output/bert_batch32_learningrate2e-3
```

#### Step4: Model prediction
```
python run_qa.py \

--do_train=false \
--do_predict=true \
--cuda=true \
--data_dir=./data \
--config_path=./pretrained_model/chinese-bert_chinese_wwm_pytorch/bert_config.json  \
--bert_model=./pretrained_model/chinese-bert_chinese_wwm_pytorch \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-3 \
--num_train_epochs=50 \
--use_crf=true
--output_dir=../output/bert_batch32_learningrate2e-3
```

