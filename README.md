# Brian's Deep Learning reading list:

### ImageNet:

* Deep Residual Learning (MSFT, 2015) http://arxiv.org/pdf/1512.03385v1.pdf | [Slides](http://image-net.org/challenges/talks/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)
* Deep Residual Learning for Image Recognition: http://arxiv.org/pdf/1512.03385
* PReLu/Weight Initialization (MSFT) [http://arxiv.org/pdf/1502.01852]
* Rectifiers: Surpassing Human-Level Performance on ImageNet Classification: http://arxiv.org/pdf/1502.01852
* Batch Normalization: http://arxiv.org/pdf/1502.03167
* Batch Normalization: Reducing Internal Covariate Shift http://arxiv.org/pdf/1502.03167
* GoogLeNet: http://arxiv.org/pdf/1409.4842
* VGG-Net (ICLR, 2015) [[Web]](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) | http://arxiv.org/pdf/1409.1556
* AlexNet (2012): [Paper](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-25-2012)

### Object Detection:

* PVANET (PVA faster-RCNN): https://arxiv.org/pdf/1608.08021 | [Code](https://github.com/sanghoon/pva-faster-rcnn)
* OverFeat: Integrated Recognition, Localization & Detection using CNNs, ICLR 2014: http://arxiv.org/pdf/1312.6229.pdf
* R-CNN: Rich feature hierarchies for object detection & semantic segmentation, CVPR, 2014. http://arxiv.org/pdf/1311.2524 | [Paper-CVPR14](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) 
* SPP (Spatial Pyramid Pooling in DCNNs, ECCV, 2014: http://arxiv.org/pdf/1406.4729
* Fast R-CNN: Region-Based CNNs: http://arxiv.org/pdf/1504.08083
* Faster R-CNN: Real-Time Object Detection with Region Proposal Networks: http://arxiv.org/pdf/1506.01497 
* R-CNN minus R: http://arxiv.org/pdf/1506.06981
* E2E people detection in crowded scenes: http://arxiv.org/abs/1506.04878
* You Only Look Once: Real-Time Object Detection: http://arxiv.org/abs/1506.02640 | https://arxiv.org/abs/1612.08242 | [C Code](https://github.com/pjreddie/darknet) | [Tensorflow Code](https://github.com/thtrieu/darkflow)
* Inside-Outside Net: Detecting Objects with Skip Pooling and RNNs: http://arxiv.org/abs/1512.04143
* Deep Residual Networks: http://arxiv.org/abs/1512.03385
* Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning: http://arxiv.org/pdf/1503.00949.pdf
* R-FCN: Object Detection via Region-based FCNs: https://arxiv.org/abs/1605.06409 | [Code](https://github.com/daijifeng001/R-FCN)
* SSD: Single Shot MultiBox Detector: https://arxiv.org/pdf/1512.02325v2.pdf | [Code](https://github.com/weiliu89/caffe/tree/ssd)
* Speed/accuracy trade-offs for modern convolutional object detectors: https://arxiv.org/pdf/1611.10012v1.pdf

### Video Classification
* Convolutional Networks for Learning Video Representations, ICLR 2016: http://arxiv.org/pdf/1511.06432v4.pdf
* Deep Multi Scale Video Prediction Beyond Mean Square Error, ICLR 2016: http://arxiv.org/pdf/1511.05440v6.pdf

### Object Tracking
* Online Tracking by Learning Discriminative Saliency Maps with CNNs: http://arxiv.org/pdf/1502.06796
* DeepTrack: Discriminative Feature Representations with CNNs, BMVC, 2014: http://www.bmva.org/bmvc/2014/files/paper028.pdf
* Learning a Deep Compact Image Representation for Visual Tracking, NIPS, 2013: http://winsty.net/papers/dlt.pdf
* Hierarchical Convolutional Features for Visual Tracking, ICCV 2015: [Paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ma_Hierarchical_Convolutional_Features_ICCV_2015_paper.pdf) | [Code](https://github.com/jbhuang0604/CF2)
* Visual Tracking with FCNs, ICCV 2015: [Paper](http://202.118.75.4/lu/Paper/ICCV2015/iccv15_lijun.pdf) | [Code](https://github.com/scott89/FCNT)
* Learning Multi-Domain CNNs for Visual Tracking: http://arxiv.org/pdf/1510.07945.pdf | [Code](https://github.com/HyeonseobNam/MDNet) | [Project Page](http://cvlab.postech.ac.kr/research/mdnet/)

### Low-Level Vision

#### Super-Resolution
* Learning Iterative Image Reconstruction. IJCAI, 2001. [[Paper]](http://www.ais.uni-bonn.de/behnke/papers/ijcai01.pdf)
* Super-Resolution (SRCNN) [[Web]](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) | [[Paper-ECCV14]](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf) | [[Paper-arXiv15]](http://arxiv.org/pdf/1501.00092.pdf)
* Image Super-Resolution Using Deep CNNs, arXiv:1501.00092.
* Very Deep Super-Resolution
  * Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee, Accurate Image Super-Resolution Using Very Deep Convolutional Networks, arXiv:1511.04587, 2015. [[Paper]](http://arxiv.org/abs/1511.04587)
* Deeply-Recursive Convolutional Network
  * Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee, Deeply-Recursive Convolutional Network for Image Super-Resolution, arXiv:1511.04491, 2015. [[Paper]](http://arxiv.org/abs/1511.04491)
* Casade-Sparse-Coding-Network
  * Zhaowen Wang, Ding Liu, Wei Han, Jianchao Yang and Thomas S. Huang, Deep Networks for Image Super-Resolution with Sparse Prior. ICCV, 2015. [[Paper]](http://www.ifp.illinois.edu/~dingliu2/iccv15/iccv15.pdf) [[Code]](http://www.ifp.illinois.edu/~dingliu2/iccv15/)
* Perceptual Losses for Real-Time Style Transfer and Super-Resolution: http://arxiv.org/abs/1603.08155 | [Supplement](http://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf)
* SRGAN: Photo-Realistic Single Image Super-Resolution with GANs, 2016: https://arxiv.org/pdf/1609.04802v3.pdf
* Image Super-Resolution with Fast Approximate Convolutional Sparse Coding, ICONIP, 2014: [Paper](http://brml.org/uploads/tx_sibibtex/281.pdf)

#### Other Applications
* Optical Flow (FlowNet) [[Paper]](http://arxiv.org/pdf/1504.06852)
  * Philipp Fischer, Alexey Dosovitskiy, Eddy Ilg, Philip Häusser, Caner Hazırbaş, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, Thomas Brox, FlowNet: Learning Optical Flow with Convolutional Networks, arXiv:1504.06852.
* Compression Artifacts Reduction [[Paper-arXiv15]](http://arxiv.org/pdf/1504.06993)
  * Chao Dong, Yubin Deng, Chen Change Loy, Xiaoou Tang, Compression Artifacts Reduction by a Deep Convolutional Network, arXiv:1504.06993.
* Blur Removal
  * Christian J. Schuler, Michael Hirsch, Stefan Harmeling, Bernhard Schölkopf, Learning to Deblur, arXiv:1406.7444 [[Paper]](http://arxiv.org/pdf/1406.7444.pdf)
  * Jian Sun, Wenfei Cao, Zongben Xu, Jean Ponce, Learning a Convolutional Neural Network for Non-uniform Motion Blur Removal, CVPR, 2015 [[Paper]](http://arxiv.org/pdf/1503.00593)
* Image Deconvolution [[Web]](http://lxu.me/projects/dcnn/) [[Paper]](http://lxu.me/mypapers/dcnn_nips14.pdf)
  * Li Xu, Jimmy SJ. Ren, Ce Liu, Jiaya Jia, Deep Convolutional Neural Network for Image Deconvolution, NIPS, 2014.
* Deep Edge-Aware Filter [[Paper]](http://jmlr.org/proceedings/papers/v37/xub15.pdf)
  * Li Xu, Jimmy SJ. Ren, Qiong Yan, Renjie Liao, Jiaya Jia, Deep Edge-Aware Filters, ICML, 2015.
* Computing the Stereo Matching Cost with a Convolutional Neural Network [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zbontar_Computing_the_Stereo_2015_CVPR_paper.pdf)
  * Jure Žbontar, Yann LeCun, Computing the Stereo Matching Cost with a Convolutional Neural Network, CVPR, 2015.
* Colorful Image Colorization Richard Zhang, Phillip Isola, Alexei A. Efros, ECCV, 2016 [[Paper]](http://arxiv.org/pdf/1603.08511.pdf), [[Code]](https://github.com/richzhang/colorization)
* Ryan Dahl, [[Blog]](http://tinyclouds.org/colorize/)
* Feature Learning by Inpainting[[Paper]](https://arxiv.org/pdf/1604.07379v1.pdf)[[Code]](https://github.com/pathak22/context-encoder)
  * Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros, Context Encoders: Feature Learning by Inpainting, CVPR, 2016

### Edge Detection

* Holistically-Nested Edge Detection: http://arxiv.org/pdf/1504.06375 | [Code](https://github.com/s9xie/hed)
* DeepEdge: Multi-Scale Bifurcated Deep Network for Top-Down Contour Detection, CVPR, 2015: http://arxiv.org/pdf/1412.1123
* DeepContour: A Deep Convolutional Feature Learned by Positive-Sharing Loss for Contour Detection: [CVPR15](http://mc.eistar.net/UpLoadFiles/Papers/DeepContour_cvpr15.pdf)

### Semantic Segmentation

* SEC (Seed, Expand & Constrain): Three Principles for Weakly-Supervised Image Segmentation: [ECCV, 2016](http://pub.ist.ac.at/~akolesnikov/files/ECCV2016/main.pdf) | [Code](https://github.com/kolesman/SEC)
* Adelaide: Piecewise training of deep structured models: http://arxiv.org/pdf/1504.01013
* Deep Parsing Network (DPN): http://arxiv.org/pdf/1509.02634
* CentraleSuperBoundaries: http://arxiv.org/pdf/1511.07386
* BoxSup: Exploiting Bounding Boxes to Supervise CNNs: http://arxiv.org/pdf/1503.01640
* POSTECHL: Learning Deconvolution Network for Semantic Segmentation: http://arxiv.org/pdf/1505.04366
* Decoupled DNNs for Semi-supervised Semantic Segmentation: http://arxiv.org/pdf/1506.04924
* Learning Transferrable Knowledge for Semantic Segmentation with DCNNs: http://arxiv.org/pdf/1512.07928.pdf | [Project Page](http://cvlab.postech.ac.kr/research/transfernet/)
* Conditional Random Fields as RNNs: http://arxiv.org/pdf/1502.03240
* DeepLab: Weakly-and semi-supervised learning of a DCNN for semantic image segmentation: http://arxiv.org/pdf/1502.02734
* Zoom-out: Feedforward Semantic Segmentation With Zoom-Out Features: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf)
* Joint Calibration for Semantic Segmentation: http://arxiv.org/pdf/1507.01581
* FCNs for Semantic Segmentation: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) | http://arxiv.org/pdf/1411.4038
* Hypercolumns for Object Segmentation and Fine-Grained Localization: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hariharan_Hypercolumns_for_Object_2015_CVPR_paper.pdf)
* Deep Hierarchical Parsing for Semantic Segmentation: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sharma_Deep_Hierarchical_Parsing_2015_CVPR_paper.pdf)
* Learning Hierarchical Features for Scene Labeling [ICML 2012](http://yann.lecun.com/exdb/publis/pdf/farabet-icml-12.pdf) | [PAMI 2013](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)
* SegNet: [Web](http://mi.eng.cam.ac.uk/projects/segnet/)
* SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation: http://arxiv.org/abs/1511.00561
* Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding:  http://arxiv.org/abs/1511.00561
* Multi-Scale Context Aggregation by Dilated Convolutions: http://arxiv.org/pdf/1511.07122v2.pdf
* Segment-Phrase Table for Semantic Segmentation, Visual Entailment and Paraphrasing: [ICCV 2015](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Izadinia_Segment-Phrase_Table_for_ICCV_2015_paper.pdf)
* Pusing the Boundaries of Boundary Detection Using deep Learning: http://arxiv.org/pdf/1511.07386v2.pdf
* Weakly supervised graph based semantic segmentation by learning communities of image-parts: [ICCV 2015](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Pourian_Weakly_Supervised_Graph_ICCV_2015_paper.pdf)

### [Visual Attention and Saliency](http://www.scholarpedia.org/article/Visual_salience)

* Mr-CNN: Predicting Eye Fixations using CNNs: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Predicting_Eye_Fixations_2015_CVPR_paper.pdf)
* Sequential Search for Landmarks: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Singh_Learning_a_Sequential_2015_CVPR_paper.pdf)
* Multiple Object Recognition with Visual Attention: http://arxiv.org/pdf/1412.7755.pdf
* Recurrent Models of Visual Attention: [NIPS 2014](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)

### Object Recognition
* Weakly-supervised learning with CNNs: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Oquab_Is_Object_Localization_2015_CVPR_paper.pdf)
* FV-CNN: Deep Filter Banks for Texture Recognition and Segmentation: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Cimpoi_Deep_Filter_Banks_2015_CVPR_paper.pdf)

### Human Pose Estimation
* Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields: https://arxiv.org/pdf/1611.08050.pdf
* Deepcut: Joint subset partition and labeling for multi person pose estimation: https://arxiv.org/pdf/1511.06645.pdf
* Convolutional pose machines: https://arxiv.org/pdf/1602.00134.pdf
* Stacked hourglass networks for human pose estimation: https://arxiv.org/pdf/1603.06937
* Flowing convnets for human pose estimation in videos: https://arxiv.org/pdf/1506.02897.pdf
* Joint training of a CNN & graphical model for human pose estimation: https://arxiv.org/pdf/1701.00295.pdf

### Image Understanding

* Understanding image representations by measuring equivariance and equivalence: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Lenc_Understanding_Image_Representations_2015_CVPR_paper.pdf)
* DNNs are Easily Fooled:High Confidence Predictions for Unrecognizable Images: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.pdf)
* Understanding Deep Image Representations by Inverting Them: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mahendran_Understanding_Deep_Image_2015_CVPR_paper.pdf)
* Object Detectors in Deep Scene CNNs: http://arxiv.org/abs/1412.6856
* Inverting Visual Representations with CNNs: http://arxiv.org/abs/1506.02753
* Visualizing and Understanding CNNs: [ECCV 2014](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)

### Image Captioning

* Explaining Images with Multimodal RNNs: http://arxiv.org/pdf/1410.1090, http://arxiv.org/pdf/1411.2539
* Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models: http://arxiv.org/pdf/1411.4389
* Show and Tell: A Neural Image Caption Generator: http://arxiv.org/pdf/1411.4555
* Stanford [[Web]](http://cs.stanford.edu/people/karpathy/deepimagesent/), [CVPR 2015](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
* Deep Visual-Semantic Alignments for Generating Image Descriptions: http://arxiv.org/pdf/1412.4729
* Translating Videos to Natural Language Using Deep RNNs: http://arxiv.org/pdf/1411.5654, [CVPR 2015](http://www.cs.cmu.edu/~xinleic/papers/cvpr15_rnn.pdf)
* Mind’s Eye: A Recurrent Visual Representation for Image Caption Generation: 
* CapGen: From Captions to Visual Concepts and Back: http://arxiv.org/pdf/1411.4952, [Web](http://kelvinxu.github.io/projects/capgen.html), [Paper](http://www.cs.toronto.edu/~zemel/documents/captionAttn.pdf)
* Show, Attend, and Tell: Neural Image Caption Generation with Visual Attention: http://arxiv.org/pdf/1502.03671
* Phrase-based Image Captioning: http://arxiv.org/pdf/1504.06692
* Learning like a Child: Fast Novel Visual Concept Learning from Sentence Descriptions of Images: https://arxiv.org/pdf/1504.06692.pdf
* Exploring Nearest Neighbor Approaches for Image Captioning: http://arxiv.org/pdf/1505.04467.pdf
* Language Models for Image Captioning: The Quirks and What Works: http://arxiv.org/pdf/1505.01809.pdf
* Image Captioning with an Intermediate Attributes Layer: http://arxiv.org/pdf/1506.01144.pdf
* Learning language through pictures: http://arxiv.org/pdf/1506.03694.pdf
* Describing Multimedia Content using Attention-based Encoder-Decoder Networks: http://arxiv.org/pdf/1508.02091.pdf
* Image Representations and New Domains in Neural Image Captioning: [ICCV 2015](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Learning_Query_and_ICCV_2015_paper.pdf)]

### Video Captioning

* LCRNs: Long-term Recurrent CNNs for Visual Recognition and Description: [Web](http://jeffdonahue.com/lrcn), http://arxiv.org/pdf/1411.4389.pdf, http://arxiv.org/pdf/1412.4729
  * Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, Kate Saenko, Translating Videos to Natural Language Using Deep Recurrent Neural Networks, arXiv:1412.4729.
* Microsoft [[Paper]](http://arxiv.org/pdf/1505.01861)
* Joint Modeling Embedding and Translation to Bridge Video and Language: http://arxiv.org/pdf/1505.00487
* Sequence to Sequence--Video to Text: http://arxiv.org/pdf/1502.08029.pdf
* Describing Videos by Exploiting Temporal Structure: http://arxiv.org/pdf/1506.01698.pdf
* The Long-Short Story of Movie Description: http://arxiv.org/pdf/1506.06724.pdf
* Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books: http://arxiv.org/pdf/1507.01053.pdf
* Describing Multimedia Content using Attention-based Encoder-Decoder Networks: https://arxiv.org/pdf/1612.06950.pdf
* Temporal Tessellation for Video Annotation and Summarization: https://arxiv.org/pdf/1612.06950.pdf

#### Image (IQA) and Visual (VQA) Question Answering

* Visual Question Answering: [Web](http://www.visualqa.org/), http://arxiv.org/pdf/1505.00468, [MPI / Berkeley](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/visual-turing-challenge/), http://arxiv.org/pdf/1505.01121
* Ask Your Neurons: A Neural-based Approach to IQA http://arxiv.org/pdf/1505.01121
* Models & Data for IQA: http://arxiv.org/pdf/1505.02074, [Dataset](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/)
* Are You Talking to a Machine? Multilingual Image Question Answering: http://arxiv.org/pdf/1505.05612
* DPPnet: http://arxiv.org/pdf/1511.05756.pdf, [Project](http://cvlab.postech.ac.kr/research/dppnet/)]
* Image Question Answering using CNNs with Dynamic Parameter Prediction, http://arxiv.org/pdf/1511.05765
* CMU / Microsoft Research: http://arxiv.org/pdf/1511.02274v2.pdfhttp://arxiv.org/pdf/
* Stacked Attention Networks for Image Question Answering. http://arxiv.org/pdf/1511.02274.
* MetaMind: http://arxiv.org/pdf/1603.01417v1.pdf
* Dynamic Memory Networks for Visual and Textual Question Answering: http://arxiv.org/pdf/1603.01417
* SNU + NAVER: http://arxiv.org/abs/1606.01455
* Multimodal Residual Learning for Visual QA: http://arxiv.org/pdf/1606:01455
* Multimodal Compact Bilinear Pooling for VQA & Visual Grounding: https://arxiv.org/pdf/1606.01847
* Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding, http://arxiv.org/pdf/1606.01847
* Training Recurrent Answering Units with Joint Loss Minimization for VQA: http://arxiv.org/pdf/1606.03647.pdf
* Hadamard Product for Low-rank Bilinear Pooling: http://arxiv.org/pdf/1610.04325

### Image Generation: Convolutional / Recurrent Networks

* Conditional Image Generation with PixelCNN Decoders: https://arxiv.org/pdf/1606.05328v2.pdf, [Code](https://github.com/kundan2510/pixelCNN)
* Learning to Generate Chairs with Convolutional Neural Networks: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf)
* DRAW: A Recurrent Neural Network For Image Generation: https://arxiv.org/pdf/1502.04623v2.pdf

### Image Generation: Adversarial Networks

* Generative Adversarial Networks: http://arxiv.org/abs/1406.2661
* Deep Generative Image Models using a ￼Laplacian Pyramid of Adversarial Networks: http://arxiv.org/abs/1506.05751
* Evaluation of generative models: http://arxiv.org/abs/1511.01844
* Variationally Auto-Encoded Deep Gaussian Processes: http://arxiv.org/pdf/1511.06455v2.pdf
* Generating Images from Captions with Attention: http://arxiv.org/pdf/1511.02793v2.pdf
* Unsupervised and Semi-supervised Learning with Categorical GANs: http://arxiv.org/pdf/1511.06390v1.pdf
* Censoring Representations with an Adversary: http://arxiv.org/pdf/1511.05897v3.pdf
* Distributional Smoothing with Virtual Adversarial Training: http://arxiv.org/pdf/1507.00677v8.pdf
* Generative Visual Manipulation on the Natural Image Manifold: https://arxiv.org/pdf/1609.03552v2.pdf, [Code](https://github.com/junyanz/iGAN)], [Video](https://youtu.be/9c4z6YsBGQ0)]
* Unsupervised Representation Learning with Deep Convolutional GANs: http://arxiv.org/pdf/1511.06434.pdf

### Other Topics
* Visual Analogies: [NIPS 2015](https://web.eecs.umich.edu/~honglak/nips2015-analogy.pdf)
* Designing Deep Networks for Surface Normal Estimation: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Designing_Deep_Networks_2015_CVPR_paper.pdf)
* Action Detection [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Gkioxari_Finding_Action_Tubes_2015_CVPR_paper.pdf)
* Crowd Counting with DCNNs: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Cross-Scene_Crowd_Counting_2015_CVPR_paper.pdf)
* 3D Shape Retrieval [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Sketch-Based_3D_Shape_2015_CVPR_paper.pdf)
* Auxiliary Image Regularization for Deep CNNs with Noisy Labels: http://arxiv.org/pdf/1511.07069v2.pdf
* Artistic Style: http://arxiv.org/abs/1508.06576, [Code](https://github.com/jcjohnson/neural-style)
* Human Gaze Estimation: [CVPR 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Appearance-Based_Gaze_Estimation_2015_CVPR_paper.pdf), [Web](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/)
* DeepFace: Closing the Gap to Human-Level Performance in Face Verification: [CVPR 2014](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)
* DeepID3: Face Recognition with Very Deep Neural Networks: http://arxiv.org/abs/1502.00873
* FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
* Facial Landmark Detection with Tweaked CNNs: http://arxiv.org/abs/1511.04031, [Project](http://www.openu.ac.il/home/hassner/projects/tcnn_landmarks/)

## Courses
* Deep Vision
  * [Stanford] [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
  * [CUHK] [ELEG 5040: Advanced Topics in Signal Processing(Introduction to Deep Learning)](https://piazza.com/cuhk.edu.hk/spring2015/eleg5040/home)
* More Deep Learning
  * [Stanford] [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
  * [Oxford] [Deep Learning by Prof. Nando de Freitas](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)
  * [NYU] [Deep Learning by Prof. Yann LeCun](http://cilvr.cs.nyu.edu/doku.php?id=courses:deeplearning2014:start)

### Books
* Free Online Books
  * [Deep Learning: Goodfellow, Bengio, Courville](http://www.iro.umontreal.ca/~bengioy/dlbook/)
  * [Neural Networks and Deep Learning: Nielsen](http://neuralnetworksanddeeplearning.com/)
  * [Deep Learning Tutorial: LISA lab](http://deeplearning.net/tutorial/deeplearning.pdf)

### Frameworks

* [Tensorflow](https://www.tensorflow.org/)
* [Torch7](http://torch.ch/)
* [Torchnet](https://github.com/torchnet/torchnet)
* [Caffe](http://caffe.berkeleyvision.org/)
* [Theano](http://deeplearning.net/software/theano/)
* [Pylearn2](http://deeplearning.net/software/pylearn2/)
* [Blocks](https://github.com/mila-udem/blocks)
* [Keras](http://keras.io/)
* [Lasagne](https://github.com/Lasagne/Lasagne)
* [MatConvNet: CNNs for MATLAB](http://www.vlfeat.org/matconvnet/)
* [MXNet](http://mxnet.io/)
* [Deepgaze](https://github.com/mpatacchiola/deepgaze)

### Applications
* Adversarial Training
  * Code and hyperparameters for the paper "Generative Adversarial Networks" [[Web]](https://github.com/goodfeli/adversarial)
* Understanding and Visualizing
  * Source code for "Understanding Deep Image Representations by Inverting Them," CVPR, 2015. [[Web]](https://github.com/aravindhm/deep-goggle)
* Semantic Segmentation
  * Source code for the paper "Rich feature hierarchies for accurate object detection and semantic segmentation," CVPR, 2014. [[Web]](https://github.com/rbgirshick/rcnn)
  * Source code for the paper "Fully Convolutional Networks for Semantic Segmentation," CVPR, 2015. [[Web]](https://github.com/longjon/caffe/tree/future)
* Super-Resolution
  * Image Super-Resolution for Anime-Style-Art [[Web]](https://github.com/nagadomi/waifu2x)
* Edge Detection
  * Source code for the paper "DeepContour: A Deep Convolutional Feature Learned by Positive-Sharing Loss for Contour Detection," CVPR, 2015. [[Web]](https://github.com/shenwei1231/DeepContour)
  * Source code for the paper "Holistically-Nested Edge Detection", ICCV 2015. [[Web]](https://github.com/s9xie/hed)

