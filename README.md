【写这个`NodeBook`的目的是全面梳理一下NER相关的知识】
### 1、NER在工业界挑战性问题 及 解决方式
- 实体嵌套或长文本---->`Span-based NER`
- 实体类别动态变化 / OOV---->`增量学习（CL-NER）`
    - 原有模型重新训练代价高，而且容易灾难性遗忘（catastrophic forgetting）,所以使用增量学习的方式
- 行业领域知识依赖强（如金融、医疗）---->`使用领域预训练模型（如 FinBERT、BioBERT）`
    - 某些专业领域如医疗、金融、法律有大量术语和特殊表达，普通NER模型泛化能力差
- 多语言 / 小语种---->`使用多语言模型（mBERT, XLM-R）`
### 2、解决NER问题  常用到的的模型组合（以BERT为主）

- BERT + Span
- BERT + LSTM + Span

- BERT + SoftMax
- BERT + LSTM + SoftMax

- BERT + CRF
- BERT + LSTM + CRF

注意：①有实验表明，加了LSTM不一定更好 ②CRF的作用：学习token之间的转移规则，可以利用CRF捕捉全局依赖；

### 3、数据集介绍
名称：CLUENER 细粒度命名实体识别

格式：`JSON`格式;

类别标签：数据分为10个标签类别，分别为: 
- 地址（address）
- 书名（book）
- 公司（company）
- 游戏（game）
- 政府（goverment）
- 电影（movie）
- 姓名（name）
- 组织机构（organization）
- 职位（position）
- 景点（scene）

数据详细介绍、基线模型和效果测评，见 https://github.com/CLUEbenchmark/CLUENER


测试集上SOTA效果见榜单：http://www.CLUEbenchmark.com

### 4、仅使用BERT做NER的核心代码（调用HuggingFace中的BERT）
```python
from transformers import BertForTokenClassification
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=17)#这样模型就可以拿过来直接使用
model.cuda()#将模型挪到GPU

model.train()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
Epochs = 10
for epoch in range(Epochs):
    losses = 0.0
    for data in trainloader:
        tokens_tensors, masks_tensors, label_tensors = [t.cuda() for t in data]
        outputs = model(input_ids = tokens_tensors,attention_mask = masks_tensors,labels = label_tensors)
        loss = outputs[0]#上一行代码已经传入了labels，所以这一步可以直接得到loss
```

### 5、做增量的核心代码

```python
class BertNER(nn.Module):
    def __init__():
        pass
    def forward():
        pass
    def extend_output_layer(self, new_labels):
        # 实体类别增加
        new_num_labels = self.num_labels + len(new_labels)
        old_classifier = self.classifier
        self.classifier = nn.Linear(self.bert.config.hidden_size, new_num_labels)
        #关闭参数更新
        with torch.no_grad():
            self.classifier.weight[:old_classifier.weight.size(0)] = old_classifier.weight
            self.classifier.bias[:old_classifier.bias.size(0)] = old_classifier.bias
        self.num_labels = new_num_labels
```








