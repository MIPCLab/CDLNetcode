# Fine-Grained Ship Classification of Remote Sensing Images Based on Class Discrepancy Learning Network

***Abstract:*** Remote sensing imagery provides critical support for maritime situational awareness, vessel monitoring, and coastal management. Fine-grained classification of ships in such imagery aims to distinguish visually similar vessel types (e.g., cargo ships, oilers, and fishing boats), which is challenging due to intra-class variation caused by different sensors, observation angles, and complex backgrounds. To address this issue, we propose a class-discrepancy learning network (CDLNet) specifically designed for ship categories exhibiting imbalanced intra-class variation. An intra-class variation index (IVI) is formulated to quantitatively assess category-specific diversity and guides a targeted adaptive learning strategy to enhance category-level discriminability. For classes with substantial appearance variations, a causally intervened masked attention module (CIMAM) is introduced to adaptively capture diverse intra-class representations. The proposed method is particularly effective in remote sensing contexts, where diverse acquisition conditions significantly impact recognition performance. Extensive experiments conducted on two benchmark datasets for fine-grained ship classification in remote sensing imagery demonstrate the superior performance and generalization capability of the proposed CDLNet. On the FGSC-23 dataset, CDLNet achieves 93.23% overall accuracy, surpassing the current state-of-the-art method RGCRL-Net (92.89%) and other competitive baselines. The source code is available at https://github.com/[your-repo]/CDLNet.



## Citation

If you find this work valuable or use our code in your own research, please consider citing us: 





## Dependencies

> torch=1.13.1 
> torchvision 
scipy 
numpy 
matplotlib 

If you encounter any problems, please feel free to open an issue.
