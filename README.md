# I Detect What I Don’t Know: Active Learning with SWAG for Unknown Anomaly Detection

<hr />

![main figure](media/intro_fig.png)
> **Abstract:** *Anomaly detection plays a critical role in medical imaging applications such as COVID-19 and pneumonia screening. However, most existing techniques depend on labeled anomalous samples, which are often scarce or impractical to obtain. We propose an unsupervised anomaly detection framework that integrates active learning with generative models to improve detection without requiring anomalous training data. Experimental results on the COVID CXR dataset show that active learning substantially enhances performance, raising ROC$-$AUC from 0.9489 (baseline) to 0.9982 and F1 from 0.8048 to 0.9746, while reducing false positives from 34 to 5. Similarly, on the \texttt{chest\_xray} pneumonia dataset, our method improves ROC$-$AUC from 0.6834 to 0.8820 and F1 from 0.7008 to 0.7849, with notable gains in precision ($0.7620 \rightarrow 0.9060$). These results demonstrate that active learning can substantially refine anomaly detection accuracy in medical imaging tasks, even when anomalous samples are unavailable during training.* 
<hr />





## Results

### Synapse Dataset
State-of-the-art comparison on the abdominal multi-organ Synapse dataset. We report both the segmentation performance (DSC, HD95) and model complexity (parameters and FLOPs).
Our proposed UNETR++ achieves favorable segmentation performance against existing methods, while being considerably reducing the model complexity. Best results are in bold. 
Abbreviations stand for: Spl: _spleen_, RKid: _right kidney_, LKid: _left kidney_, Gal: _gallbladder_, Liv: _liver_, Sto: _stomach_, Aor: _aorta_, Pan: _pancreas_. 
Best results are in bold.

![Synapse Results](media/synapse_results.png)

<hr />

## Qualitative Comparison

### Synapse Dataset
Qualitative comparison on multi-organ segmentation task. Here, we compare our UNETR++ with existing methods: UNETR, Swin UNETR, and nnFormer. 
The different abdominal organs are shown in the legend below the examples. Existing methods struggle to correctly segment different organs (marked in red dashed box). 
Our UNETR++ achieves promising segmentation performance by accurately segmenting the organs.
![Synapse Qual Results](media/UNETR++_results_fig_synapse.jpg)

### ACDC Dataset
Qualitative comparison on the ACDC dataset. We compare our UNETR++ with existing methods: UNETR and nnFormer. It is noticeable that the existing methods struggle to correctly segment different organs (marked in red dashed box). Our UNETR++ achieves favorable segmentation performance by accurately segmenting the organs.  Our UNETR++ achieves promising segmentation performance by accurately segmenting the organs.
![ACDC Qual Results](media/acdc_vs_unetr_suppl.jpg)


<hr />

## Installation
The code is tested with PyTorch 1.11.0 and CUDA 11.3. After cloning the repository, follow the below steps for installation,

1. Create and activate conda environment
```shell
conda create --name unetr_pp python=3.8
conda activate unetr_pp
```
2. Install PyTorch and torchvision
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Install other dependencies
```shell
pip install -r requirements.txt
```
<hr />




<hr />


## Acknowledgement
We thanks to PatchCore implementaions.


```

## Contact
Should you have any question, please create an issue on this repository or contact me at abdelrahman.youssief@mbzuai.ac.ae.
