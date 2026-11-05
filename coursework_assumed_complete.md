# Coursework: 02516 IDLCV (draft assuming all required work is finished)

**Status: planning draft, not a record of completed work.** Before using a line on a CV, finish that work, replace each `[TODO]` with the real result or your actual role, and delete anything you did not do. `coursework.md` remains the version based on the current files.

Number sources: "notebook output" = already in a saved notebook; "report section" = `project2/sec/3_method.tex`. `[TODO]` = no confirmed result exists, so none has been entered.

### 02516 IDLCV | Week 1 (Exercises 1.1-1.3) | Neural networks and CNNs for image classification
- **What this week covered**: Classifying MNIST with fully connected and convolutional networks, then CIFAR-10 with optimizers, batch normalization, and residual blocks.
- **What I did myself**:
  - Implemented MNIST fully connected and convolutional networks (with guidance).
  - Tested depth, width, learning rates, and sigmoid vs. ReLU (from scratch).
  - Added two types of augmentation and dropout, and inspected filters and misclassifications (with guidance).
  - Trained a CIFAR-10 CNN with SGD and Adam, and plotted both in one comparison (with guidance).
  - Ran a BatchNorm comparison on CIFAR-10 with augmentation applied correctly (with guidance).
  - Implemented residual blocks and trained a ResNet using `nn.CrossEntropyLoss` (with guidance).
- **Concepts I can explain**:
  - What goes wrong when a learning rate is too high or too low.
  - Why ReLU trains more easily than sigmoid in deeper networks.
  - How SGD and Adam differ in convergence.
  - How residual connections help training.
  - Why combining softmax and log separately can be numerically unstable.
- **Tech / methods**: Languages: Python; Libraries: PyTorch, torchvision, NumPy, Matplotlib; Methods: CNNs, ResNet blocks, BatchNorm, dropout, data augmentation, SGD, Adam; Tools: Jupyter, Apple MPS
- **Results (quantified, only if available)**:
  - MNIST fully connected, test accuracy: 96.4% -> 97.8% (notebook output); best tested setting `[TODO]`.
  - MNIST CNN, final test accuracy: 98.8% (notebook output).
  - CIFAR-10 CNN with BatchNorm and Adam: 72.2% final test (notebook output). SGD `[TODO]`; no BatchNorm `[TODO]`; augmented `[TODO]`.
  - CIFAR-10 ResNet: 53.0% final test (notebook output); after tuning `[TODO]`.
- **Difficulties & how I solved them**: Identified Softmax combined with CrossEntropyLoss as a likely cause of poor ResNet training, and removed it. Add before-and-after results from a clean rerun: `[TODO]`.
- **Possible CV bullet**: Compared SGD and Adam, BatchNorm, and augmentation for a custom CIFAR-10 CNN in PyTorch, reaching `[TODO]` test accuracy, and implemented residual blocks for a small ResNet.

### 02516 IDLCV | Week 2 (Exercise 2.1) | GPU computing on HPC
- **What this week covered**: Running PyTorch work on the DTU GPU cluster through batch jobs.
- **What I did myself**: Wrote and submitted an LSF GPU job that moved a tensor to CUDA and saved a CPU copy (with guidance).
- **Concepts I can explain**:
  - What each LSF resource request in the job script controls.
  - Why a tensor is moved to CPU before saving for portability.
  - How a virtual environment and module load set up a batch job.
- **Tech / methods**: Languages: Python, shell; Libraries: PyTorch, torchvision, Pillow; Methods: GPU batch jobs; Tools: LSF, HPC
- **Results (quantified, only if available)**: Job output `[TODO]`.
- **Difficulties & how I solved them**: `[TODO]`
- **Possible CV bullet**: Combine with other weeks

### 02516 IDLCV | Project 1 | Hotdog vs. not-hotdog classification
- **What this week covered**: Building, improving, and explaining a binary image classifier on 2,047 training and 1,862 test images, and reporting the results on a poster.
- **What I did myself**: `[TODO: my part of the team's work]`. All tasks below were completed by the team:
  - Compared a baseline CNN, optimizers, BatchNorm, resizing strategies, and augmentation.
  - Fine-tuned ResNet-18 and ConvNeXt-Tiny, and tested a higher input resolution, TTA, and an ensemble.
  - Analyzed misclassified images and generated saliency and SmoothGrad maps.
- **Concepts I can explain**:
  - Why transfer learning helps on small datasets.
  - How TTA and ensembling combine predictions.
  - How saliency maps work and why SmoothGrad averages gradients over noisy inputs.
  - Why choosing models using test data makes results optimistic.
- **Tech / methods**: Languages: Python; Libraries: PyTorch, torchvision, Matplotlib; Methods: CNNs, BatchNorm, data augmentation, transfer learning, TTA, ensembling, saliency, SmoothGrad; Tools: Jupyter, Git, Apple MPS
- **Results (quantified, only if available)**:
  - My notebook, final test accuracy: custom CNN 74.0%; ResNet-18 90.8% (notebook output).
  - Team notebook, test accuracy: baseline 71.97%; ResNet-18 91.57%; ConvNeXt-Tiny with TTA 97.21% (notebook output).
  - Rerun baseline and BatchNorm with a validation split: `[TODO]`.
- **Difficulties & how I solved them**: Add how the stuck baseline (47.5% train, 51.9% test) was resolved: `[TODO]`.
- **Possible CV bullet**: Worked in a 3-person team on hotdog image classification, `[TODO: my contribution]`; the team's best model, ConvNeXt-Tiny with TTA, reached 97.21% test accuracy.

### 02516 IDLCV | Project 2 | Video action classification
- **What this week covered**: Classifying 10 action classes from video, comparing frame-based, fusion, 3D, and two-stream models, and evaluating on data with and without leakage.
- **What I did myself**: `[TODO: confirm my part]`. Assumed scope:
  - Implemented per-frame, late fusion, early fusion, R3D-18, and dual-stream models (with guidance).
  - Trained all models on the DTU GPU cluster and evaluated them per video (with guidance).
  - Compared results with and without data leakage, and wrote up the method and results.
- **Concepts I can explain**:
  - How the models differ in where time is combined.
  - Why early fusion adapts the first ResNet layer.
  - Why RGB and flow images must share the same crop and flip.
  - How overlapping video groups cause leakage.
  - What optical flow adds over RGB frames.
- **Tech / methods**: Languages: Python, LaTeX, shell; Libraries: PyTorch, torchvision; Methods: 2D and 3D CNNs, late and early fusion, two-stream networks, optical flow (precomputed RAFT), transfer learning, AdamW; Tools: LSF, HPC
- **Results (quantified, only if available)**:
  - Leakage-free split: dual-stream best validation 80.0%, test 77.5%; RGB late fusion best validation 62.5% (report section).
  - Test accuracy for per-frame, late fusion, early fusion, and R3D-18: `[TODO]`.
  - Leaky vs. leakage-free comparison: `[TODO]`.
- **Difficulties & how I solved them**: `[TODO]`
- **Possible CV bullet**: Compared five video action classifiers on a leakage-free UCF split using a GPU cluster, including an RGB + optical flow model that reached 77.5% test accuracy (`[TODO: my role]`).

## Course summary (rewritten on every run)
### Course summary: 02516 IDLCV (assuming completion)

**Topics**
- Week 1: Fully connected networks, CNNs, optimizers, BatchNorm, and ResNet blocks
- Week 2: GPU batch jobs on the DTU cluster
- Project 1: Image classification, transfer learning, and model explainability
- Project 2: Video action classification and data leakage

**Skills and methods by depth**
- **Implemented from scratch**: MNIST hyperparameter experiments and prediction visualization. *Reasoning: these exercises provide no code.*
- **Implemented with guidance**: CNNs and residual blocks; project experiments; LSF job scripts; video models. *Reasoning: briefs or templates set the structure, and the key parts are yours.*
- **Applied with a library or ran and analyzed**: ResNet-18, ConvNeXt-Tiny, and R3D-18 fine-tuning. *Reasoning: the architectures and weights come from torchvision.*
- Check every label against what you actually wrote, particularly in the team projects.

**Consolidated CV bullets: ML / computer vision internships**
- Fine-tuned an ImageNet-pretrained ResNet-18 for binary food classification, reaching 90.8% final test accuracy on 1,862 images, compared with 74.0% for a custom CNN (one run each).
- Compared SGD and Adam, BatchNorm, and augmentation for a custom CIFAR-10 CNN in PyTorch, reaching `[TODO]` test accuracy.
- Compared five video action classifiers on a leakage-free UCF split using a GPU cluster, including an RGB + optical flow model that reached 77.5% test accuracy (`[TODO: my role]`).

**Consolidated CV bullets: research assistant roles**
- Ran controlled comparisons of resizing, BatchNorm, and augmentation on an image classifier, with results reported using a validation split (`[TODO]`).
- Analyzed classifier errors and applied saliency and SmoothGrad maps to inspect model decisions (`[TODO: my role]`).
- Quantified how data leakage changes video classification results (`[TODO]`).

**Gaps: what is still needed before using this**
- **Roles:** State your exact part in both team projects.
- **Results:** Replace every `[TODO]`; no bullet with a `[TODO]` is usable yet.
- **Reproducibility:** Rerun key experiments from clean notebooks with several seeds, and keep logs for Project 2.
- **Interview stories:** Add one or two real debugging stories.
- **Optional exercise:** t-SNE/PCA feature visualization in 1.2 is not included.
