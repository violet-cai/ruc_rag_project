# Evaluator：文本生成评估框架

`evaluator` 文件夹包含了一组 Python 模块，用于评估文本生成模型的性能。该框架提供了多种评估指标，包括 BLEU、ROUGE、F1 分数等，以评估生成文本与参考答案的质量。

## 目录

- [Evaluator：文本生成评估框架](#evaluator文本生成评估框架)
  - [目录](#目录)
  - [安装](#安装)
  - [使用方法](#使用方法)
  - [配置](#配置)
  - [评估指标](#评估指标)

## 安装

确保您的环境中安装了以下依赖：

- `transformers`：用于加载预训练模型和分词器。
- `rouge`：用于计算 ROUGE 指标。
- `re`：Python 正则表达式库。
- `collections`：用于数据处理。

您可以通过 `pip` 安装所需的库：

```bash
pip install transformers rouge
```
## 使用方法

以下是如何使用 `evaluator` 框架进行评估的步骤：

1. **准备数据**：
   - 您需要准备模型的预测结果和对应的参考答案数据。这些数据将被用来计算评估指标。

2. **配置评估参数**：
   - 我们已经在basic_config中配置了一些评估参数。
```python
 - # Metrics to evaluate the result
metrics: [ 'f1','em','bleu','acc','precision','recall','rouge-1','rouge-2','rouge-l','retrieval_recall','retrieval_precision']
# Specify setting for metric, will be called within certain metrics
metric_setting:
  bleu_max_order: 4
  bleu_smooth: False
  retrieval_recall_topk: 2
  #tokenizer_name: "gpt-4"
save_metric_score: True #　whether to save the metric score into txt file
save_intermediate_data: False
```
   - 通过evaluator_demo中的 `config_dict` ，您可以自行配置评估参数。您可以设置数据集名称、需要补充的计算的评估指标列表、结果保存目录以及其他指标特定的设置。

3. **运行评估**：
   - 使用 `run_evaluation` 函数执行评估。此函数将根据提供的配置和数据计算所有指定的评估指标，并返回结果。

## 配置

评估配置可以通过 `config_dict` 进行设置。以下是一个配置示例：

```python
config_dict = {
        "dataset_name": "example_dataset",
        "save_dir": "./results"
}
```

## 评估指标

`evaluator` 框架支持以下评估指标，用于全面评估文本生成模型的性能：

- **Retrieval Recall/Precision**
  - 用于评估检索系统的性能，其中召回率衡量检索到的相关文档占所有相关文档的比例，精确率衡量检索到的相关文档占检索结果的比例。

- **BLEU (Bilingual Evaluation Understudy)**
  - 用于评估机器翻译质量的经典指标，通过计算候选文本与参考文本之间的n-gram重叠来衡量翻译的准确度。

- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
  - 用于评估自动文摘和机器翻译的质量，主要关注候选文本中包含参考文本信息的程度。

- **F1 Score**
  - 结合了精确率（Precision）和召回率（Recall）的指标，用于衡量模型预测结果的质量，尤其在分类问题中。

- **ExactMatch**
  - 用于评估预测答案与标准答案是否完全一致的严格指标。


这些指标可以帮助我们从不同角度理解模型的性能，包括生成文本的准确性、相关性和完整性。通过综合这些指标的结果，我们可以更全面地评估和优化文本生成模型。