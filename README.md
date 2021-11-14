# Awesome-Multi-label-Image-Recognition

## 0. Introducation
本项目主要涵盖**多标签图像识别**方向的资料。具体而言，所包含的方向如下所示：  
- Multi-label Image Recognition
- Multi-label Image Recognition with Noisy-label
- Multi-label Image Recognition with Partial-label
- Few/Zero-shot Multi-label Image Recognition

## 1. Papers:
<details>
  <summary> ICCV (Virtual, 2021.10.11-17) <a href="https://mp.weixin.qq.com/s/DOjElL3fg198m2gdgN1_WQ"><img src="https://img.shields.io/badge/微信- 论文速记 (Day 1)-green" alt=""></a> </summary>
  ### 2021       
|  **Pub.**  | **Title**                                                    |                          **Links**                           |
| :--------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
| **TPAMI** | **[P-GCN]** Learning Graph Convolutional Networks for Multi-Label Recognition and Applications | [Paper](https://ieeexplore.ieee.org/abstract/document/9369105) |
| **TIP** | **[MCAR]** Learning to Discover Multi-Class Attentional Regions for Multi-Label Image Recognition | [Paper](https://arxiv.org/abs/2007.01755)/[Code](https://github.com/gaobb/MCAR) |
| **CVPR** | **[C-Trans]** General Multi-label Image Classification with Transformers | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lanchantin_General_Multi-Label_Image_Classification_With_Transformers_CVPR_2021_paper.pdf)/[Code](https://github.com/QData/C-Tran) | 
| **ICCV** | **[TDRG]** Transformer-based Dual Relation Graph for Multi-label Image Recognition | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Transformer-Based_Dual_Relation_Graph_for_Multi-Label_Image_Recognition_ICCV_2021_paper.pdf)/[Code](https://github.com/iCVTEAM/TDRG) |
| **ICCV** | **[ASL]** Asymmetric Loss For Multi-Label Classification | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf)/[Code](https://github.com/Alibaba-MIIL/ASL) |
| **ICCV** | **[CSRA]** Residual Attention: A Simple but Effective Method for Multi-Label Recognition | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhu_Residual_Attention_A_Simple_but_Effective_Method_for_Multi-Label_Recognition_ICCV_2021_paper.pdf)/[Code](https://github.com/Kevinz-code/CSRA) |
| **IJCAI** | **[GM-MLIC]** GM-MLIC: Graph Matching based Multi-Label Image Classification | [Paper](https://arxiv.org/abs/2104.14762) | 
| **AAAI**  | **[DSDL]** Deep Semantic Dictionary Learning for Multi-label Image Classification | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16472)/[Code](https://github.com/ZFT-CQU/DSDL) |
| **AAAI** | **[MGTN]** Modular Graph Transformer Networks for Multi-Label Image Classification | [Paper](https://people.cs.umu.se/sonvx/files/2021_AAAI_MGTN.pdf)/[Code](https://github.com/ReML-AI/MGTN) |
| **ACM MM** | **[M3TR]** M3TR: Multi-modal Multi-label Recognition with Transformer | [Paper](https://dl.acm.org/doi/pdf/10.1145/3474085.3475191)/[Code](https://github.com/iCVTEAM/M3TR) |
| arxiv | MlTr: Multi-label Classification with Transformer | [Paper](https://arxiv.org/abs/2106.06195)/[Code](https://github.com/starmemda/MlTr/) | 
| arxiv | Query2Label: A Simple Transformer Way to Multi-Label Classification | [Paper](https://arxiv.org/pdf/2107.10834.pdf)/[Code](https://github.com/SlongLiu/query2labels) |
| arxiv | Multi-layered Semantic Representation Network for Multi-label Image Classification | [Paper](https://arxiv.org/pdf/2106.11596.pdf) | 
| arxiv | Contrast Learning Visual Attention for Multi Label Classification | [Paper](https://arxiv.org/pdf/2107.11626.pdf) |
| arxiv | Learning Discriminative Representations for Multi-Label Image Recognition | [Paper](https://arxiv.org/pdf/2107.11159.pdf) |
</details>


1. <a href="#Multi-label Image Recognition"> Multi-label Image Recognition </a>
2. <a href="#Multi-label Image Recognition with Noisy-label"> Multi-label Image Recognition with Noisy-label </a>
3. <a href="#Multi-label Image Recognition with Partial-label"> Multi-label Image Recognition with Partial-label </a>
4. <a href="#Few/Zero-shot Multi-label Image Recognition"> Few/Zero-shot Multi-label Image Recognition </a>

## Multi-label Image Recognition <a id="Multi-label Image Recognition" class="anchor" href="Multi-label Image Recognition" aria-hidden="true"><span class="octicon octicon-link"></span></a>    

### 2021       
|  **Pub.**  | **Title**                                                    |                          **Links**                           |
| :--------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
| **TPAMI** | **[P-GCN]** Learning Graph Convolutional Networks for Multi-Label Recognition and Applications | [Paper](https://ieeexplore.ieee.org/abstract/document/9369105) |
| **TIP** | **[MCAR]** Learning to Discover Multi-Class Attentional Regions for Multi-Label Image Recognition | [Paper](https://arxiv.org/abs/2007.01755)/[Code](https://github.com/gaobb/MCAR) |
| **CVPR** | **[C-Trans]** General Multi-label Image Classification with Transformers | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lanchantin_General_Multi-Label_Image_Classification_With_Transformers_CVPR_2021_paper.pdf)/[Code](https://github.com/QData/C-Tran) | 
| **ICCV** | **[TDRG]** Transformer-based Dual Relation Graph for Multi-label Image Recognition | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Transformer-Based_Dual_Relation_Graph_for_Multi-Label_Image_Recognition_ICCV_2021_paper.pdf)/[Code](https://github.com/iCVTEAM/TDRG) |
| **ICCV** | **[ASL]** Asymmetric Loss For Multi-Label Classification | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf)/[Code](https://github.com/Alibaba-MIIL/ASL) |
| **ICCV** | **[CSRA]** Residual Attention: A Simple but Effective Method for Multi-Label Recognition | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhu_Residual_Attention_A_Simple_but_Effective_Method_for_Multi-Label_Recognition_ICCV_2021_paper.pdf)/[Code](https://github.com/Kevinz-code/CSRA) |
| **IJCAI** | **[GM-MLIC]** GM-MLIC: Graph Matching based Multi-Label Image Classification | [Paper](https://arxiv.org/abs/2104.14762) | 
| **AAAI**  | **[DSDL]** Deep Semantic Dictionary Learning for Multi-label Image Classification | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16472)/[Code](https://github.com/ZFT-CQU/DSDL) |
| **AAAI** | **[MGTN]** Modular Graph Transformer Networks for Multi-Label Image Classification | [Paper](https://people.cs.umu.se/sonvx/files/2021_AAAI_MGTN.pdf)/[Code](https://github.com/ReML-AI/MGTN) |
| **ACM MM** | **[M3TR]** M3TR: Multi-modal Multi-label Recognition with Transformer | [Paper](https://dl.acm.org/doi/pdf/10.1145/3474085.3475191)/[Code](https://github.com/iCVTEAM/M3TR) |
| arxiv | MlTr: Multi-label Classification with Transformer | [Paper](https://arxiv.org/abs/2106.06195)/[Code](https://github.com/starmemda/MlTr/) | 
| arxiv | Query2Label: A Simple Transformer Way to Multi-Label Classification | [Paper](https://arxiv.org/pdf/2107.10834.pdf)/[Code](https://github.com/SlongLiu/query2labels) |
| arxiv | Multi-layered Semantic Representation Network for Multi-label Image Classification | [Paper](https://arxiv.org/pdf/2106.11596.pdf) | 
| arxiv | Contrast Learning Visual Attention for Multi Label Classification | [Paper](https://arxiv.org/pdf/2107.11626.pdf) |
| arxiv | Learning Discriminative Representations for Multi-Label Image Recognition | [Paper](https://arxiv.org/pdf/2107.11159.pdf) |

### 2020       

|  **Pub.**  | **Title**                                                    |                          **Links**                           |
| :--------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
| **TPAMI** | **[KGGR]** Knowledge-Guided Multi-Label Few-Shot Learning for General Image Recognition | [Paper](https://arxiv.org/abs/2009.09450) |
|  **TMM** | **[DER]** Disentangling, Embedding and Ranking Label Cues for Multi-Label Image Recognition | [Paper](https://ieeexplore.ieee.org/document/9122471) |
|  **TMM** | **[TS-GCN]** Joint Input and Output Space Learning for Multi-Label Image Classification | [Paper](https://ieeexplore.ieee.org/document/9115821) |
|  **CVPR** | **[PLA]** Orderless_Recurrent_Models_for_Multi-Label_Classification | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yazici_Orderless_Recurrent_Models_for_Multi-Label_Classification_CVPR_2020_paper.pdf)/[Code](https://github.com/voyazici/orderless-rnn-classification) |
|  **ECCV** | **[ADD-GCN]** Attention-Driven Dynamic Graph Convolutional Network for Multi-Label Image Recognition | [Paper](https://arxiv.org/abs/2012.02994)/[Code](https://github.com/Yejin0111/ADD-GCN) |
|  **AAAI** | **[KSSNet]** Multi-Label Classification with Label Graph Superimposing | [Paper](https://arxiv.org/abs/1911.09243)/[Code](https://github.com/mathkey/mssnet) |
|  **AAAI** | **[MS-CMA]** Cross-Modality Attention with Semantic Graph Embedding for Multi-Label Classification | [Paper](https://arxiv.org/abs/1912.07872) |
| **ACM MM** | **[SGTN]** Privacy-Preserving Visual Content Tagging using Graph Transformer Networks | [Paper](https://dl.acm.org/doi/10.1145/3394171.3414047)/[Code](https://github.com/ReML-AI/sgtn) |
| **ACM MM** | **[AdaHGNN]** AdaHGNN: Adaptive Hypergraph Neural Networks for Multi-Label Image Classification | [Paper](https://dl.acm.org/doi/10.1145/3394171.3414046) |

### 2019       

| **Pub.** | **Title**                                                    |                          **Links**                           |
| :------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
| **TSCB** | Multi-label Image Classification via Feature/Label Co-Projection | [Paper](https://ieeexplore.ieee.org/document/8985434) |
| **CVPR** | **[ML-GCN]** Multi-Label Image Recognition with Graph Convolutional Networks | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.pdf)/[Code](https://github.com/Megvii-Nanjing/ML-GCN) |
| **CVPR** | **[VAC]** Visual Attention Consistency under Image Transforms for Multi-Label Image Classification | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Visual_Attention_Consistency_Under_Image_Transforms_for_Multi-Label_Image_Classification_CVPR_2019_paper.pdf)/[Code](https://github.com/hguosc/visual_attention_consistency) |
| **ICCV** | **[SSGRL]** Learning Semantic-Specific Graph Representation for Multi-Label Image Recognition | [Paper](https://arxiv.org/abs/1908.07325)/[Code](https://github.com/HCPLab-SYSU/SSGRL) |
| **ICME** | Multi-Label Image Recognition with Joint Class-Aware Map Disentangling and Label Correlation Embedding | [Paper](http://www.weixiushen.com/publication/icme19.pdf) |

### 2015~2018       

| **Pub.**  | **Title**                                                    |                          **Links**                           |
| :-------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
| **TPAMI'15** | **[HCP]** HCP: A Flexible CNN Framework for Multi-Label Image Classification | [Paper](https://ieeexplore.ieee.org/document/7305792)  |
| **CVPR'17** | **[SRN]** Learning Spatial Regularization with Image-level Supervisions for Multi-label Image Classification | [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Learning_Spatial_Regularization_CVPR_2017_paper.pdf)/[Code](https://github.com/zhufengx/SRN_multilabel) |
| **CVPR'16** | **[CNN-RNN]** CNN-RNN: A Unified Framework for Multi-label Image Classification | [Paper](https://arxiv.org/abs/1604.04573)/[Code](https://github.com/AmrMaghraby/CNN-RNN-A-Unified-Framework-for-Multi-label-Image-Classification) |
| **ICCV'17** | **[WILDCAT]** WILDCAT: Weakly Supervised Learning of Deep ConvNets for Image Classification, Pointwise Localization and Segmentation | [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Durand_WILDCAT_Weakly_Supervised_CVPR_2017_paper.pdf)/[Code](https://github.com/durandtibo/wildcat.pytorch) |
| **ICCV'17** | **[RDAR]** Multi-label Image Recognition by Recurrently Discovering Attentional Regions | [Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_Multi-Label_Image_Recognition_ICCV_2017_paper.pdf)/[Code](https://github.com/James-Yip/AttentionImageClass) |
| **AAAI'18** | **[Order-Free RNN]** Order-Free RNN with Visual Attention for Multi-Label Classification | [Paper](https://arxiv.org/abs/1707.05495) |
| **IJCAI'18** | **[MsDPD]** Multi-scale and Discriminative Part Detectors Based Features for Multi-label Image Classification | [Paper](https://www.ijcai.org/Proceedings/2018/0090.pdf) |

## Multi-label Image Recognition with Noisy-label <a id="Multi-label Image Recognition with Noisy-label" class="anchor" href="Multi-label Image Recognition with Noisy-label" aria-hidden="true"><span class="octicon octicon-link"></span></a> 

| **Pub.** | **Title**                                                    |                          **Links**                           |
| :------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
| **CVPR'19** | Weakly Supervised Image Classification through Noise Regularization | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Weakly_Supervised_Image_Classification_Through_Noise_Regularization_CVPR_2019_paper.pdf) |
| **CVPR'17** | Learning From Noisy Large-Scale Datasets With Minimal Supervision | [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Veit_Learning_From_Noisy_CVPR_2017_paper.pdf) |
| **ACCV** | Multi-label Learning from Noisy Labels with Non-linear Feature Transformation | [Paper](https://vipl.ict.ac.cn/uploadfile/upload/2018111615590567.pdf) |
| **ICMR** | Confidence-basedWeighted Loss for Multi-label Classification with Missing Labels | [Paper](https://dl.acm.org/doi/abs/10.1145/3372278.3390728) |

## Multi-label Image Recognition with Partial-label <a id="Multi-label Image Recognition with Partial-label" class="anchor" href="Multi-label Image Recognition with Partial-label" aria-hidden="true"><span class="octicon octicon-link"></span></a> 

|  **Pub.**   | **Title**                                                    |                          **Links**                           |
| :---------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
| arxiv | **[ATAM]** Rethinking Crowdsourcing Annotation: Partial Annotation with Salient Labels for
Multi-Label Image Classification | [Paper](https://arxiv.org/pdf/2109.02688.pdf) |
|  **CVPR'20** | Interactive Multi-Label CNN Learning with Partial Labels | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huynh_Interactive_Multi-Label_CNN_Learning_With_Partial_Labels_CVPR_2020_paper.pdf) |
| **NeurIPS'20** | Exploiting weakly supervised visual patterns to learn from partial annotations | [Paper](https://proceedings.neurips.cc/paper/2020/file/066ca7bf90807fcd8e4f1eaef4e4e8f7-Paper.pdf) |
|  **CVPR'19** | Learning a Deep ConvNet for Multi-label Classification with Partial Labels | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Durand_Learning_a_Deep_ConvNet_for_Multi-Label_Classification_With_Partial_Labels_CVPR_2019_paper.pdf) |

## Few/Zero-shot Multi-label Image Recognition <a id="Few/Zero-shot Multi-label Image Recognition" class="anchor" href="Few/Zero-shot Multi-label Image Recognition" aria-hidden="true"><span class="octicon octicon-link"></span></a> 

| **Pub.** | **Title**                                                    |                          **Links**                           |
| :------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
| **ICCV'21** | **[BiAM]** Discriminative Region-based Multi-Label Zero-Shot Learning |[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Narayan_Discriminative_Region-Based_Multi-Label_Zero-Shot_Learning_ICCV_2021_paper.pdf)/[Code](https://github.com/akshitac8/BiAM) |
| **ICCV'21** | Semantic Diversity Learning for Zero-Shot Multi-label Classification | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ben-Cohen_Semantic_Diversity_Learning_for_Zero-Shot_Multi-Label_Classification_ICCV_2021_paper.pdf)/[Code](https://github.com/Alibaba-MIIL/ZS_SDL) |
| **CVPR'20** | A Shared Multi-Attention Framework for Multi-Label Zero-Shot Learning | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huynh_A_Shared_Multi-Attention_Framework_for_Multi-Label_Zero-Shot_Learning_CVPR_2020_paper.pdf)/Code |
| **CVPR'18** | Multi-Label Zero-Shot Learning with Structured Knowledge Graphs | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Lee_Multi-Label_Zero-Shot_Learning_CVPR_2018_paper.pdf)/[Code](https://github.com/Phoenix1327/ML-ZSL) |
| **CVPR'16** | Fast Zero-Shot Image Tagging | [Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Fast_Zero-Shot_Image_CVPR_2016_paper.pdf)/Code |
| arxiv | Multi-Label Learning from Single Positive Labels | [Paper](https://arxiv.org/pdf/2106.09708.pdf)|
