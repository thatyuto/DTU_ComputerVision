# Coursework: 02516 Introduction to Deep Learning in Computer Vision

Source key: "notebook output" = printed in the saved notebook; "my note" = written by me in the notebook, with no saved output to check it against; "report section" = `project2/sec/3_method.tex`, with no run logs in this folder.
Week labels are inferred from exercise numbers. Teamwork is labeled as team work until my part is confirmed.
Unfinished or unconfirmed work is marked "TODO: to be filled in".

### 02516 IDLCV | Week 1 (Exercises 1.1-1.3) | Neural networks and CNNs for image classification
- **What this week covered**: Classifying MNIST with fully connected and convolutional networks, then CIFAR-10 with optimizers, batch normalization, and residual blocks.
- **What I did myself**:
  - Implemented a one-hidden-layer MNIST network in a provided template (with guidance).
  - Built a deeper MNIST network and plotted predictions with confidence and second-choice classes (from scratch).
  - Implemented a 4-layer CNN for MNIST, with padding and stride chosen to preserve feature-map size (with guidance).
  - Added rotation augmentation and dropout, plotted augmented images, and visualized first-layer filters (with guidance).
  - Explained in writing why first-epoch test accuracy can exceed training accuracy (from scratch).
  - Designed a CIFAR-10 CNN with BatchNorm and Kaiming initialization, and trained it with Adam (with guidance).
  - Implemented a residual block that passed the provided sanity check, then trained a ResNet on CIFAR-10 (with guidance).
  - Added MPS device selection so the exercises ran on Apple Silicon (with guidance).
  - TODO: to be filled in: learning-rate and sigmoid tests in 1.1, a second augmentation and t-SNE/PCA in 1.2, and a saved SGD vs. Adam plot in 1.3 are not visible.
- **Concepts I can explain**:
  - How kernel size, stride, and padding keep or change feature-map size.
  - Why training accuracy is averaged during an epoch, so first-epoch test accuracy can be higher.
  - Why a residual block needs matching input and output shapes for `F(x) + x`.
  - Why `CrossEntropyLoss` expects logits, not softmax output.
  - How `train()` and `eval()` change dropout and BatchNorm.
- **Tech / methods**: Languages: Python; Libraries: PyTorch, torchvision, NumPy, Matplotlib; Methods: fully connected networks, CNNs, ResNet blocks, BatchNorm, dropout, data augmentation, SGD, Adam, Kaiming initialization; Tools: Jupyter, Apple MPS
- **Results (quantified, only if available)** (single runs, no seeds):
  - MNIST fully connected, 100 units at 5 epochs -> 2x256 units at 7 epochs and a higher learning rate: test accuracy 96.4% -> 97.8% (notebook output).
  - MNIST CNN, 5 epochs, final test accuracy: 98.8%; with rotation 98.5%; with dropout 98.6% (notebook output). These differences are too small to show an effect from one run.
  - CIFAR-10 CNN with BatchNorm and Adam, 10 epochs: final test 72.2%, peak 74.0%, final training 89.4% (notebook output).
  - CIFAR-10 ResNet with SGD, 10 epochs: final test 53.0%, peak 58.2%, final training 95.2% (notebook output).
  - CIFAR-10 SGD: test about 68%, peak 69.5%; Adam without BatchNorm: final test 72% (my notes).
- **Difficulties & how I solved them**: Identified Softmax combined with CrossEntropyLoss as a likely cause of poor ResNet training (about 54% train and 47% test, my note). The saved code has Softmax removed, and training accuracy then reached 95.2% (notebook output), though with a large overfitting gap.
- **Possible CV bullet**: Trained and compared MNIST and CIFAR-10 classifiers in PyTorch, including a custom CNN with BatchNorm that reached 72.2% final test accuracy on CIFAR-10 (peak 74.0%) and a ResNet built from self-implemented residual blocks.

### 02516 IDLCV | Week 2 (Exercise 2.1) | GPU computing on HPC
- **What this week covered**: Running PyTorch work on the DTU GPU cluster through batch jobs.
- **What I did myself**:
  - Wrote a script that checks CUDA, moves an image tensor to GPU, and saves a CPU copy (with guidance).
  - Wrote an LSF job script requesting a GPU, CPU cores, memory, and a time limit (with guidance).
  - TODO: to be filled in: no job output is saved here, so a completed run is not confirmed.
- **Concepts I can explain**:
  - What each LSF resource request in the job script controls.
  - Why a tensor is moved to CPU before saving for portability.
  - How a virtual environment and module load set up a batch job.
- **Tech / methods**: Languages: Python, shell; Libraries: PyTorch, torchvision, Pillow; Methods: GPU batch jobs; Tools: LSF, HPC
- **Results (quantified, only if available)**: None saved.
- **Difficulties & how I solved them**: TODO: to be filled in
- **Possible CV bullet**: Combine with other weeks

### 02516 IDLCV | Project 1 | Hotdog vs. not-hotdog classification
- **What this week covered**: Building and improving a binary image classifier on 2,047 training and 1,862 test images, and reporting the results on a poster.
- **What I did myself** (from `IDLCV_Project_1_hotdog.ipynb`, run on my machine):
  - Implemented a 3-stage CNN baseline with Kaiming initialization (with guidance).
  - Compared squish resizing with aspect-ratio-preserving center crops (with guidance).
  - Ran a BatchNorm comparison on the same architecture and optimizer settings (with guidance).
  - Fine-tuned an ImageNet-pretrained ResNet-18 with ImageNet normalization and augmentation (with a library).
  - TODO: to be filled in: my part in the team notebook (`IDLCV_Project_1_hotdog_v1.ipynb`). Its saved output comes from another account, so I have not counted it as my work.
- **Concepts I can explain**:
  - The trade-off between squishing and center-cropping non-square images.
  - Why pretrained models need ImageNet normalization.
  - Why a small dataset benefits from transfer learning.
  - Why a lower learning rate is used when fine-tuning pretrained weights.
  - Why reporting the best test epoch is optimistic.
- **Tech / methods**: Languages: Python; Libraries: PyTorch, torchvision, Matplotlib; Methods: CNNs, BatchNorm, data augmentation, transfer learning, fine-tuning; Tools: Jupyter, Git, Apple MPS
- **Results (quantified, only if available)**:
  - My notebook, same 10-epoch CNN (final test): squish 74.0%; center crop 74.0%, peak 75.2% (notebook output).
  - My notebook, BatchNorm: 76.5% without vs. 74.4% with, peak 76.7% (notebook output). This baseline value conflicts with the saved baseline run, which stuck at 51.9%, so rerun it before using this comparison.
  - My notebook, ResNet-18, 5 epochs: final test 90.8%, peak 91.6% (notebook output).
  - Team notebook, test accuracy: baseline 71.97%; BatchNorm 77.12%; BatchNorm + augmentation 76.96%; ResNet-18 91.57%; ResNet-18 at 224x224 93.77%; with flip TTA 94.47%; ConvNeXt-Tiny with TTA 97.21%; ResNet + ConvNeXt ensemble 96.29% (notebook output). These models were all compared on the test set, so the best score is optimistic.
- **Difficulties & how I solved them**: TODO: to be filled in. The saved baseline stuck at 47.5% train and 51.9% test for all 10 epochs; add how this was resolved.
- **Possible CV bullet**: Fine-tuned an ImageNet-pretrained ResNet-18 for binary food classification, reaching 90.8% final test accuracy on 1,862 images, compared with 74.0% for a custom CNN (one run each).

### 02516 IDLCV | Project 2 | Video action classification
- **What this week covered**: Classifying 10 action classes from video, comparing frame-based, fusion, 3D, and two-stream models, and evaluating on data with and without leakage.
- **What I did myself**: TODO: to be filled in: confirm which parts are mine. The code and study comments suggest this work, but the folder cannot show who wrote it:
  - Per-frame, late fusion, early fusion, R3D-18, and RGB + optical flow classifiers in PyTorch.
  - Training and evaluation scripts, datasets with crop and flip shared across a clip's RGB and flow images, and LSF job scripts.
  - A report section on the dual-stream method.
- **Concepts I can explain** (confirm each one):
  - How per-frame, late fusion, early fusion, and 3D CNN models differ in where time is combined.
  - Why early fusion adapts the first ResNet layer, and how its pretrained weights are repeated and scaled.
  - Why RGB and flow images in a clip must share the same crop and flip.
  - How overlapping video groups cause leakage and inflate validation results.
- **Tech / methods**: Languages: Python, LaTeX, shell; Libraries: PyTorch, torchvision; Methods: 2D and 3D CNNs, late and early fusion, two-stream networks, optical flow (precomputed RAFT), transfer learning, AdamW; Tools: LSF, HPC
- **Results (quantified, only if available)**: Leakage-free split (500/120/120 videos): dual-stream best validation 80.0%, test 77.5%; RGB late fusion best validation 62.5% (report section). The runs used different training setups, so the gain is not due to optical flow alone. Results for the other models: TODO: to be filled in.
- **Difficulties & how I solved them**: TODO: to be filled in
- **Possible CV bullet**: TODO: confirm my role first. Draft: Built PyTorch pipelines for five video action classifiers on a leakage-free UCF split, including an RGB + optical flow model that reached 77.5% test accuracy.

## Course summary (rewritten on every run)
### Course summary: 02516 Introduction to Deep Learning in Computer Vision

**Topics so far**
- Week 1: Fully connected networks, CNNs, optimizers, BatchNorm, and ResNet blocks on MNIST and CIFAR-10
- Week 2: GPU batch jobs on the DTU cluster
- Project 1: Hotdog classification, with resizing, BatchNorm, and transfer learning comparisons
- Project 2: Video action classification, with fusion, 3D, and two-stream models

**Skills and methods by depth**
- **Implemented from scratch**: Deeper MNIST network; prediction visualization with second-choice classes. *Reasoning: the exercise asked for these without providing any code.*
- **Implemented with guidance**: MNIST CNN; CIFAR-10 CNN; residual block; augmentation and dropout; hotdog baseline and comparisons; LSF job script. *Reasoning: an exercise brief or template set the structure, and you supplied the key parts.*
- **Applied with a library or ran and analyzed**: Pretrained ResNet-18 fine-tuning. *Reasoning: the architecture and weights come from torchvision, and you adapted the head, data, and training.*
- **Not grouped until confirmed**: Project 2 models, and the team's ConvNeXt, TTA, ensemble, saliency, and SmoothGrad work.

**Consolidated CV bullets: ML / computer vision internships**
- Fine-tuned an ImageNet-pretrained ResNet-18 for binary food classification, reaching 90.8% final test accuracy on 1,862 images, compared with 74.0% for a custom CNN (one run each).
- Trained a custom CNN with BatchNorm on CIFAR-10 to 72.2% final test accuracy (peak 74.0%), and implemented residual blocks for a small ResNet.
- Configured GPU training jobs on an LSF cluster using PyTorch.
- Pending role confirmation: Built PyTorch pipelines for five video action classifiers on a leakage-free UCF split, including an RGB + optical flow model that reached 77.5% test accuracy.

**Consolidated CV bullets: research assistant roles**
- Compared image-resizing strategies on a fixed CNN and training setup, finding the same final test accuracy (74.0%) for squish and center crop.
- Visualized learned first-layer CNN filters and class probabilities, including the second-choice class on misclassified MNIST digits.
- Pending role confirmation: Reported dual-stream video results on a leakage-free split (validation 80.0%, test 77.5%), documenting that training differences prevent attributing the gain to optical flow alone.

**Gaps: what would make this material stronger**
- **Project roles:** State your task in Project 1 (the plan assigns A, B, and C) and your part in Project 2. This unlocks the team's stronger results.
- **Hotdog baseline:** Resolve the 51.9% vs. 76.5% conflict by rerunning the notebook from the top. Typos in saved code (`transet_tl`, `"cude"`) need fixing for a clean rerun.
- **Augmentation claim:** Your "No difference" note in 1.3 can't be used yet. The saved loaders use plain `ToTensor()`, not the augmented transform, so rerun it with `train_transform`.
- **ResNet note:** Your 1.3 note says the ResNet underfits, but the saved run is clearly overfitting (95.2% train, 53.0% test). Update it before discussing this.
- **Evaluation:** Your notebook selects the best epoch using test data. Use a validation split, and repeat runs across seeds to report mean and spread.
- **Project 2 results:** Save `history.json` and `eval.py` outputs for all five models, since `runs/` is empty here.
- **Interview stories:** Add how you debugged the stuck baseline, and how you found the leakage effect in Project 2.
