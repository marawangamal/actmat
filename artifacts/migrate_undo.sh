#!/bin/bash
# Auto-generated undo for migrate_artifacts.py
set -euo pipefail
mkdir -p "$(dirname 'artifacts/checkpoints/t5-base/qasc')" && mv 'artifacts/checkpoints/t5-base/experts/qasc' 'artifacts/checkpoints/t5-base/qasc'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-base/wiki_qa')" && mv 'artifacts/checkpoints/t5-base/experts/wiki_qa' 'artifacts/checkpoints/t5-base/wiki_qa'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-base/quartz')" && mv 'artifacts/checkpoints/t5-base/experts/quartz' 'artifacts/checkpoints/t5-base/quartz'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-base/paws')" && mv 'artifacts/checkpoints/t5-base/experts/paws' 'artifacts/checkpoints/t5-base/paws'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-base/story_cloze')" && mv 'artifacts/checkpoints/t5-base/experts/story_cloze' 'artifacts/checkpoints/t5-base/story_cloze'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-base/winogrande')" && mv 'artifacts/checkpoints/t5-base/experts/winogrande' 'artifacts/checkpoints/t5-base/winogrande'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-base/wsc')" && mv 'artifacts/checkpoints/t5-base/experts/wsc' 'artifacts/checkpoints/t5-base/wsc'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-large/qasc')" && mv 'artifacts/checkpoints/t5-large/experts/qasc' 'artifacts/checkpoints/t5-large/qasc'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-large/wiki_qa')" && mv 'artifacts/checkpoints/t5-large/experts/wiki_qa' 'artifacts/checkpoints/t5-large/wiki_qa'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-large/quartz')" && mv 'artifacts/checkpoints/t5-large/experts/quartz' 'artifacts/checkpoints/t5-large/quartz'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-large/paws')" && mv 'artifacts/checkpoints/t5-large/experts/paws' 'artifacts/checkpoints/t5-large/paws'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-large/story_cloze')" && mv 'artifacts/checkpoints/t5-large/experts/story_cloze' 'artifacts/checkpoints/t5-large/story_cloze'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-large/winogrande')" && mv 'artifacts/checkpoints/t5-large/experts/winogrande' 'artifacts/checkpoints/t5-large/winogrande'
mkdir -p "$(dirname 'artifacts/checkpoints/t5-large/wsc')" && mv 'artifacts/checkpoints/t5-large/experts/wsc' 'artifacts/checkpoints/t5-large/wsc'
mkdir -p "$(dirname 'artifacts/results/t5-base-actmat')" && mv 'artifacts/results/t5-base/merged/actmat' 'artifacts/results/t5-base-actmat'
mkdir -p "$(dirname 'artifacts/results/t5-base-experts')" && mv 'artifacts/results/t5-base/experts' 'artifacts/results/t5-base-experts'
mkdir -p "$(dirname 'artifacts/results/t5-base-isoc')" && mv 'artifacts/results/t5-base/merged/isoc' 'artifacts/results/t5-base-isoc'
mkdir -p "$(dirname 'artifacts/results/t5-base-mean')" && mv 'artifacts/results/t5-base/merged/mean' 'artifacts/results/t5-base-mean'
mkdir -p "$(dirname 'artifacts/results/t5-base-regmean')" && mv 'artifacts/results/t5-base/merged/regmean' 'artifacts/results/t5-base-regmean'
mkdir -p "$(dirname 'artifacts/results/t5-base-ties')" && mv 'artifacts/results/t5-base/merged/ties' 'artifacts/results/t5-base-ties'
mkdir -p "$(dirname 'artifacts/results/t5-base-tsv')" && mv 'artifacts/results/t5-base/merged/tsv' 'artifacts/results/t5-base-tsv'
mkdir -p "$(dirname 'artifacts/results/t5-base-zeroshot')" && mv 'artifacts/results/t5-base/pretrained' 'artifacts/results/t5-base-zeroshot'
mkdir -p "$(dirname 'artifacts/results/t5-large-actmat')" && mv 'artifacts/results/t5-large/merged/actmat' 'artifacts/results/t5-large-actmat'
mkdir -p "$(dirname 'artifacts/results/t5-large-experts')" && mv 'artifacts/results/t5-large/experts' 'artifacts/results/t5-large-experts'
mkdir -p "$(dirname 'artifacts/results/t5-large-isoc')" && mv 'artifacts/results/t5-large/merged/isoc' 'artifacts/results/t5-large-isoc'
mkdir -p "$(dirname 'artifacts/results/t5-large-mean')" && mv 'artifacts/results/t5-large/merged/mean' 'artifacts/results/t5-large-mean'
mkdir -p "$(dirname 'artifacts/results/t5-large-ties')" && mv 'artifacts/results/t5-large/merged/ties' 'artifacts/results/t5-large-ties'
mkdir -p "$(dirname 'artifacts/results/t5-large-tsv')" && mv 'artifacts/results/t5-large/merged/tsv' 'artifacts/results/t5-large-tsv'
mkdir -p "$(dirname 'artifacts/results/t5-large-zeroshot')" && mv 'artifacts/results/t5-large/pretrained' 'artifacts/results/t5-large-zeroshot'
#!/bin/bash
# Undo for vision checkpoint migration (experts/ grouping + head.pt co-location).
# Results were git-tracked; restore those separately with:
#   git checkout -- artifacts/results artifacts/results14 artifacts/results20
#   rm -rf artifacts/results-8tasks artifacts/results-14tasks artifacts/results-20tasks
set -euo pipefail

mv artifacts/checkpoints/ViT-B-16/experts/CIFAR100Val/head.pt artifacts/checkpoints/ViT-B-16/head_CIFAR100Val.pt
mv artifacts/checkpoints/ViT-B-16/experts/CIFAR100Val artifacts/checkpoints/ViT-B-16/CIFAR100Val
mv artifacts/checkpoints/ViT-B-16/experts/CIFAR10Val/head.pt artifacts/checkpoints/ViT-B-16/head_CIFAR10Val.pt
mv artifacts/checkpoints/ViT-B-16/experts/CIFAR10Val artifacts/checkpoints/ViT-B-16/CIFAR10Val
mv artifacts/checkpoints/ViT-B-16/experts/CarsVal/head.pt artifacts/checkpoints/ViT-B-16/head_CarsVal.pt
mv artifacts/checkpoints/ViT-B-16/experts/CarsVal artifacts/checkpoints/ViT-B-16/CarsVal
mv artifacts/checkpoints/ViT-B-16/experts/DTDVal/head.pt artifacts/checkpoints/ViT-B-16/head_DTDVal.pt
mv artifacts/checkpoints/ViT-B-16/experts/DTDVal artifacts/checkpoints/ViT-B-16/DTDVal
mv artifacts/checkpoints/ViT-B-16/experts/EMNISTVal/head.pt artifacts/checkpoints/ViT-B-16/head_EMNISTVal.pt
mv artifacts/checkpoints/ViT-B-16/experts/EMNISTVal artifacts/checkpoints/ViT-B-16/EMNISTVal
mv artifacts/checkpoints/ViT-B-16/experts/EuroSATVal/head.pt artifacts/checkpoints/ViT-B-16/head_EuroSATVal.pt
mv artifacts/checkpoints/ViT-B-16/experts/EuroSATVal artifacts/checkpoints/ViT-B-16/EuroSATVal
mv artifacts/checkpoints/ViT-B-16/experts/FER2013Val/head.pt artifacts/checkpoints/ViT-B-16/head_FER2013Val.pt
mv artifacts/checkpoints/ViT-B-16/experts/FER2013Val artifacts/checkpoints/ViT-B-16/FER2013Val
mv artifacts/checkpoints/ViT-B-16/experts/FashionMNISTVal/head.pt artifacts/checkpoints/ViT-B-16/head_FashionMNISTVal.pt
mv artifacts/checkpoints/ViT-B-16/experts/FashionMNISTVal artifacts/checkpoints/ViT-B-16/FashionMNISTVal
mv artifacts/checkpoints/ViT-B-16/experts/Flowers102Val/head.pt artifacts/checkpoints/ViT-B-16/head_Flowers102Val.pt
mv artifacts/checkpoints/ViT-B-16/experts/Flowers102Val artifacts/checkpoints/ViT-B-16/Flowers102Val
mv artifacts/checkpoints/ViT-B-16/experts/Food101Val/head.pt artifacts/checkpoints/ViT-B-16/head_Food101Val.pt
mv artifacts/checkpoints/ViT-B-16/experts/Food101Val artifacts/checkpoints/ViT-B-16/Food101Val
mv artifacts/checkpoints/ViT-B-16/experts/GTSRBVal/head.pt artifacts/checkpoints/ViT-B-16/head_GTSRBVal.pt
mv artifacts/checkpoints/ViT-B-16/experts/GTSRBVal artifacts/checkpoints/ViT-B-16/GTSRBVal
mv artifacts/checkpoints/ViT-B-16/experts/KMNISTVal/head.pt artifacts/checkpoints/ViT-B-16/head_KMNISTVal.pt
mv artifacts/checkpoints/ViT-B-16/experts/KMNISTVal artifacts/checkpoints/ViT-B-16/KMNISTVal
mv artifacts/checkpoints/ViT-B-16/experts/MNISTVal/head.pt artifacts/checkpoints/ViT-B-16/head_MNISTVal.pt
mv artifacts/checkpoints/ViT-B-16/experts/MNISTVal artifacts/checkpoints/ViT-B-16/MNISTVal
mv artifacts/checkpoints/ViT-B-16/experts/OxfordIIITPetVal/head.pt artifacts/checkpoints/ViT-B-16/head_OxfordIIITPetVal.pt
mv artifacts/checkpoints/ViT-B-16/experts/OxfordIIITPetVal artifacts/checkpoints/ViT-B-16/OxfordIIITPetVal
mv artifacts/checkpoints/ViT-B-16/experts/PCAMVal/head.pt artifacts/checkpoints/ViT-B-16/head_PCAMVal.pt
mv artifacts/checkpoints/ViT-B-16/experts/PCAMVal artifacts/checkpoints/ViT-B-16/PCAMVal
mv artifacts/checkpoints/ViT-B-16/experts/RESISC45Val/head.pt artifacts/checkpoints/ViT-B-16/head_RESISC45Val.pt
mv artifacts/checkpoints/ViT-B-16/experts/RESISC45Val artifacts/checkpoints/ViT-B-16/RESISC45Val
mv artifacts/checkpoints/ViT-B-16/experts/RenderedSST2Val/head.pt artifacts/checkpoints/ViT-B-16/head_RenderedSST2Val.pt
mv artifacts/checkpoints/ViT-B-16/experts/RenderedSST2Val artifacts/checkpoints/ViT-B-16/RenderedSST2Val
mv artifacts/checkpoints/ViT-B-16/experts/STL10Val/head.pt artifacts/checkpoints/ViT-B-16/head_STL10Val.pt
mv artifacts/checkpoints/ViT-B-16/experts/STL10Val artifacts/checkpoints/ViT-B-16/STL10Val
mv artifacts/checkpoints/ViT-B-16/experts/SUN397Val/head.pt artifacts/checkpoints/ViT-B-16/head_SUN397Val.pt
mv artifacts/checkpoints/ViT-B-16/experts/SUN397Val artifacts/checkpoints/ViT-B-16/SUN397Val
mv artifacts/checkpoints/ViT-B-16/experts/SVHNVal/head.pt artifacts/checkpoints/ViT-B-16/head_SVHNVal.pt
mv artifacts/checkpoints/ViT-B-16/experts/SVHNVal artifacts/checkpoints/ViT-B-16/SVHNVal
rmdir artifacts/checkpoints/ViT-B-16/experts 2>/dev/null || true
mv artifacts/checkpoints/ViT-B-32/experts/CIFAR100Val/head.pt artifacts/checkpoints/ViT-B-32/head_CIFAR100Val.pt
mv artifacts/checkpoints/ViT-B-32/experts/CIFAR100Val artifacts/checkpoints/ViT-B-32/CIFAR100Val
mv artifacts/checkpoints/ViT-B-32/experts/CIFAR10Val/head.pt artifacts/checkpoints/ViT-B-32/head_CIFAR10Val.pt
mv artifacts/checkpoints/ViT-B-32/experts/CIFAR10Val artifacts/checkpoints/ViT-B-32/CIFAR10Val
mv artifacts/checkpoints/ViT-B-32/experts/CarsVal/head.pt artifacts/checkpoints/ViT-B-32/head_CarsVal.pt
mv artifacts/checkpoints/ViT-B-32/experts/CarsVal artifacts/checkpoints/ViT-B-32/CarsVal
mv artifacts/checkpoints/ViT-B-32/experts/DTDVal/head.pt artifacts/checkpoints/ViT-B-32/head_DTDVal.pt
mv artifacts/checkpoints/ViT-B-32/experts/DTDVal artifacts/checkpoints/ViT-B-32/DTDVal
mv artifacts/checkpoints/ViT-B-32/experts/EMNISTVal/head.pt artifacts/checkpoints/ViT-B-32/head_EMNISTVal.pt
mv artifacts/checkpoints/ViT-B-32/experts/EMNISTVal artifacts/checkpoints/ViT-B-32/EMNISTVal
mv artifacts/checkpoints/ViT-B-32/experts/EuroSATVal/head.pt artifacts/checkpoints/ViT-B-32/head_EuroSATVal.pt
mv artifacts/checkpoints/ViT-B-32/experts/EuroSATVal artifacts/checkpoints/ViT-B-32/EuroSATVal
mv artifacts/checkpoints/ViT-B-32/experts/FER2013Val/head.pt artifacts/checkpoints/ViT-B-32/head_FER2013Val.pt
mv artifacts/checkpoints/ViT-B-32/experts/FER2013Val artifacts/checkpoints/ViT-B-32/FER2013Val
mv artifacts/checkpoints/ViT-B-32/experts/FashionMNISTVal/head.pt artifacts/checkpoints/ViT-B-32/head_FashionMNISTVal.pt
mv artifacts/checkpoints/ViT-B-32/experts/FashionMNISTVal artifacts/checkpoints/ViT-B-32/FashionMNISTVal
mv artifacts/checkpoints/ViT-B-32/experts/Flowers102Val/head.pt artifacts/checkpoints/ViT-B-32/head_Flowers102Val.pt
mv artifacts/checkpoints/ViT-B-32/experts/Flowers102Val artifacts/checkpoints/ViT-B-32/Flowers102Val
mv artifacts/checkpoints/ViT-B-32/experts/Food101Val/head.pt artifacts/checkpoints/ViT-B-32/head_Food101Val.pt
mv artifacts/checkpoints/ViT-B-32/experts/Food101Val artifacts/checkpoints/ViT-B-32/Food101Val
mv artifacts/checkpoints/ViT-B-32/experts/GTSRBVal/head.pt artifacts/checkpoints/ViT-B-32/head_GTSRBVal.pt
mv artifacts/checkpoints/ViT-B-32/experts/GTSRBVal artifacts/checkpoints/ViT-B-32/GTSRBVal
mv artifacts/checkpoints/ViT-B-32/experts/KMNISTVal/head.pt artifacts/checkpoints/ViT-B-32/head_KMNISTVal.pt
mv artifacts/checkpoints/ViT-B-32/experts/KMNISTVal artifacts/checkpoints/ViT-B-32/KMNISTVal
mv artifacts/checkpoints/ViT-B-32/experts/MNISTVal/head.pt artifacts/checkpoints/ViT-B-32/head_MNISTVal.pt
mv artifacts/checkpoints/ViT-B-32/experts/MNISTVal artifacts/checkpoints/ViT-B-32/MNISTVal
mv artifacts/checkpoints/ViT-B-32/experts/OxfordIIITPetVal/head.pt artifacts/checkpoints/ViT-B-32/head_OxfordIIITPetVal.pt
mv artifacts/checkpoints/ViT-B-32/experts/OxfordIIITPetVal artifacts/checkpoints/ViT-B-32/OxfordIIITPetVal
mv artifacts/checkpoints/ViT-B-32/experts/PCAMVal/head.pt artifacts/checkpoints/ViT-B-32/head_PCAMVal.pt
mv artifacts/checkpoints/ViT-B-32/experts/PCAMVal artifacts/checkpoints/ViT-B-32/PCAMVal
mv artifacts/checkpoints/ViT-B-32/experts/RESISC45Val/head.pt artifacts/checkpoints/ViT-B-32/head_RESISC45Val.pt
mv artifacts/checkpoints/ViT-B-32/experts/RESISC45Val artifacts/checkpoints/ViT-B-32/RESISC45Val
mv artifacts/checkpoints/ViT-B-32/experts/RenderedSST2Val/head.pt artifacts/checkpoints/ViT-B-32/head_RenderedSST2Val.pt
mv artifacts/checkpoints/ViT-B-32/experts/RenderedSST2Val artifacts/checkpoints/ViT-B-32/RenderedSST2Val
mv artifacts/checkpoints/ViT-B-32/experts/STL10Val/head.pt artifacts/checkpoints/ViT-B-32/head_STL10Val.pt
mv artifacts/checkpoints/ViT-B-32/experts/STL10Val artifacts/checkpoints/ViT-B-32/STL10Val
mv artifacts/checkpoints/ViT-B-32/experts/SUN397Val/head.pt artifacts/checkpoints/ViT-B-32/head_SUN397Val.pt
mv artifacts/checkpoints/ViT-B-32/experts/SUN397Val artifacts/checkpoints/ViT-B-32/SUN397Val
mv artifacts/checkpoints/ViT-B-32/experts/SVHNVal/head.pt artifacts/checkpoints/ViT-B-32/head_SVHNVal.pt
mv artifacts/checkpoints/ViT-B-32/experts/SVHNVal artifacts/checkpoints/ViT-B-32/SVHNVal
rmdir artifacts/checkpoints/ViT-B-32/experts 2>/dev/null || true
mv artifacts/checkpoints/ViT-L-14/experts/CIFAR100Val/head.pt artifacts/checkpoints/ViT-L-14/head_CIFAR100Val.pt
mv artifacts/checkpoints/ViT-L-14/experts/CIFAR100Val artifacts/checkpoints/ViT-L-14/CIFAR100Val
mv artifacts/checkpoints/ViT-L-14/experts/CIFAR10Val/head.pt artifacts/checkpoints/ViT-L-14/head_CIFAR10Val.pt
mv artifacts/checkpoints/ViT-L-14/experts/CIFAR10Val artifacts/checkpoints/ViT-L-14/CIFAR10Val
mv artifacts/checkpoints/ViT-L-14/experts/CarsVal/head.pt artifacts/checkpoints/ViT-L-14/head_CarsVal.pt
mv artifacts/checkpoints/ViT-L-14/experts/CarsVal artifacts/checkpoints/ViT-L-14/CarsVal
mv artifacts/checkpoints/ViT-L-14/experts/DTDVal/head.pt artifacts/checkpoints/ViT-L-14/head_DTDVal.pt
mv artifacts/checkpoints/ViT-L-14/experts/DTDVal artifacts/checkpoints/ViT-L-14/DTDVal
mv artifacts/checkpoints/ViT-L-14/experts/EMNISTVal/head.pt artifacts/checkpoints/ViT-L-14/head_EMNISTVal.pt
mv artifacts/checkpoints/ViT-L-14/experts/EMNISTVal artifacts/checkpoints/ViT-L-14/EMNISTVal
mv artifacts/checkpoints/ViT-L-14/experts/EuroSATVal/head.pt artifacts/checkpoints/ViT-L-14/head_EuroSATVal.pt
mv artifacts/checkpoints/ViT-L-14/experts/EuroSATVal artifacts/checkpoints/ViT-L-14/EuroSATVal
mv artifacts/checkpoints/ViT-L-14/experts/FER2013Val/head.pt artifacts/checkpoints/ViT-L-14/head_FER2013Val.pt
mv artifacts/checkpoints/ViT-L-14/experts/FER2013Val artifacts/checkpoints/ViT-L-14/FER2013Val
mv artifacts/checkpoints/ViT-L-14/experts/FashionMNISTVal/head.pt artifacts/checkpoints/ViT-L-14/head_FashionMNISTVal.pt
mv artifacts/checkpoints/ViT-L-14/experts/FashionMNISTVal artifacts/checkpoints/ViT-L-14/FashionMNISTVal
mv artifacts/checkpoints/ViT-L-14/experts/Flowers102Val/head.pt artifacts/checkpoints/ViT-L-14/head_Flowers102Val.pt
mv artifacts/checkpoints/ViT-L-14/experts/Flowers102Val artifacts/checkpoints/ViT-L-14/Flowers102Val
mv artifacts/checkpoints/ViT-L-14/experts/Food101Val/head.pt artifacts/checkpoints/ViT-L-14/head_Food101Val.pt
mv artifacts/checkpoints/ViT-L-14/experts/Food101Val artifacts/checkpoints/ViT-L-14/Food101Val
mv artifacts/checkpoints/ViT-L-14/experts/GTSRBVal/head.pt artifacts/checkpoints/ViT-L-14/head_GTSRBVal.pt
mv artifacts/checkpoints/ViT-L-14/experts/GTSRBVal artifacts/checkpoints/ViT-L-14/GTSRBVal
mv artifacts/checkpoints/ViT-L-14/experts/KMNISTVal/head.pt artifacts/checkpoints/ViT-L-14/head_KMNISTVal.pt
mv artifacts/checkpoints/ViT-L-14/experts/KMNISTVal artifacts/checkpoints/ViT-L-14/KMNISTVal
mv artifacts/checkpoints/ViT-L-14/experts/MNISTVal/head.pt artifacts/checkpoints/ViT-L-14/head_MNISTVal.pt
mv artifacts/checkpoints/ViT-L-14/experts/MNISTVal artifacts/checkpoints/ViT-L-14/MNISTVal
mv artifacts/checkpoints/ViT-L-14/experts/OxfordIIITPetVal/head.pt artifacts/checkpoints/ViT-L-14/head_OxfordIIITPetVal.pt
mv artifacts/checkpoints/ViT-L-14/experts/OxfordIIITPetVal artifacts/checkpoints/ViT-L-14/OxfordIIITPetVal
mv artifacts/checkpoints/ViT-L-14/experts/PCAMVal/head.pt artifacts/checkpoints/ViT-L-14/head_PCAMVal.pt
mv artifacts/checkpoints/ViT-L-14/experts/PCAMVal artifacts/checkpoints/ViT-L-14/PCAMVal
mv artifacts/checkpoints/ViT-L-14/experts/RESISC45Val/head.pt artifacts/checkpoints/ViT-L-14/head_RESISC45Val.pt
mv artifacts/checkpoints/ViT-L-14/experts/RESISC45Val artifacts/checkpoints/ViT-L-14/RESISC45Val
mv artifacts/checkpoints/ViT-L-14/experts/RenderedSST2Val/head.pt artifacts/checkpoints/ViT-L-14/head_RenderedSST2Val.pt
mv artifacts/checkpoints/ViT-L-14/experts/RenderedSST2Val artifacts/checkpoints/ViT-L-14/RenderedSST2Val
mv artifacts/checkpoints/ViT-L-14/experts/STL10Val/head.pt artifacts/checkpoints/ViT-L-14/head_STL10Val.pt
mv artifacts/checkpoints/ViT-L-14/experts/STL10Val artifacts/checkpoints/ViT-L-14/STL10Val
mv artifacts/checkpoints/ViT-L-14/experts/SUN397Val/head.pt artifacts/checkpoints/ViT-L-14/head_SUN397Val.pt
mv artifacts/checkpoints/ViT-L-14/experts/SUN397Val artifacts/checkpoints/ViT-L-14/SUN397Val
mv artifacts/checkpoints/ViT-L-14/experts/SVHNVal/head.pt artifacts/checkpoints/ViT-L-14/head_SVHNVal.pt
mv artifacts/checkpoints/ViT-L-14/experts/SVHNVal artifacts/checkpoints/ViT-L-14/SVHNVal
rmdir artifacts/checkpoints/ViT-L-14/experts 2>/dev/null || true
