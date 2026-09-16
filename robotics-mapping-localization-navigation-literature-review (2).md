# Literature Review: Spatial Mapping, Localization, and Navigation in Robotics — From Classical Geometry to Foundation Models

*Links marked ✓ were confirmed live this session. Others are the canonical arXiv ID / DOI for that paper from standard bibliographic record — if a link ever 404s, search the exact title on Google Scholar or Semantic Scholar, which will always resolve it. Lines marked* **Note:** *are personal synthesis/annotations on that entry — how it relates to our own pipeline, contrasts with other methods, or connects to our specific bugs/design decisions — not part of the paper's own claims.*

## TL;DR
- The field has shifted decisively from hand-engineered geometric/probabilistic pipelines toward learned and foundation-model approaches: reconstruction now runs through feed-forward transformers (DUSt3R→MASt3R→VGGT), relocalization through self-supervised retrieval (DINOv2/AnyLoc) plus learned matching/pose regression, and navigation through cross-embodiment policies (GNM/ViNT/NoMaD) — yet classical backends (bundle adjustment, factor graphs, PnP/RANSAC, A*/DWA) remain in every deployed stack.
- For the LingBot-Map project (DINOv2 retrieval + learned 6DoF pose + topological graph on MyAGV/ROS LiDAR), the most directly relevant literature is the hierarchical localization paradigm (HLoc), DINOv2-based VPR robust to appearance change (AnyLoc, SALAD, EigenPlaces), scene coordinate regression (ACE/DSAC*), and topological navigation foundation models (ViNT/NoMaD; AnyLoc-based topological mapping).
- The winning architecture pattern across the literature is hybrid: learned front-ends (retrieval + matching/pose) feeding classical geometric back-ends and classical planners, with topological graphs bridging global retrieval and local metric control.

---

# SECTION 1 — RECONSTRUCTION OF SPATIAL MAPS

## 1A. Traditional (geometric/probabilistic)

### Sparse visual SLAM
1. **MonoSLAM** — Davison, Reid, Molton, Stasse, IEEE TPAMI 2007. First real-time, drift-free EKF-based SLAM using a single camera, maintaining a small online sparse landmark map.
   https://dl.acm.org/doi/10.1109/TPAMI.2007.1049 ✓
   **Note:** First SLAM — monocular.
2. **PTAM (Parallel Tracking and Mapping)** — Klein & Murray, ISMAR 2007. Splits tracking and mapping into parallel threads, using keyframe bundle adjustment to scale to thousands of landmarks.
   https://www.researchgate.net/publication/4334429_Parallel_Tracking_and_Mapping_for_Small_AR_Workspaces ✓
   **Note:** Parallelization was the key innovation.
3. **ORB-SLAM** — Mur-Artal, Montiel, Tardós, IEEE T-RO 2015. Feature-based monocular SLAM using ORB features for unified tracking, mapping, relocalization, and loop closing.
   https://www.semanticscholar.org/paper/ORB-SLAM:-A-Versatile-and-Accurate-Monocular-SLAM-Mur-Artal-Montiel/6933c70c747e6a8103f68f1a1db80185401d537b ✓
   **Note:** Basically the traditional version of our own problem. Pipeline: (1) feature extraction every frame — FAST+BRIEF; (2) tracking thread — "where am I right now?"; (3) local mapping thread — "build/refine the map" (triangulation + local bundle adjustment); (4) loop closing thread — "have I been here before?" (new keyframe → DBoW2 bag-of-words vector built on ORB → check against the database of past keyframes; if a strong match, verify geometrically and close the loop).
4. **ORB-SLAM2** — Mur-Artal & Tardós, IEEE T-RO 2017. Extends ORB-SLAM to stereo and RGB-D cameras with map reuse across sessions.
   https://arxiv.org/abs/1610.06475
   **Note:** Improved depth and no scale drift (monocular, stereo, RGB-D). Introduces the reusable map — similar to what we want to do.
5. **ORB-SLAM3** — Campos, Elvira, Gómez Rodríguez, Montiel, Tardós, IEEE T-RO 2021. Unifies visual, visual-inertial, and multi-map SLAM across pinhole and fisheye cameras with fast IMU initialization.
   https://arxiv.org/abs/2007.11898 ✓
   **Note:** Monocular, stereo, RGB-D, each with or without IMU. Expands from one map to a multi-map system (the Atlas): if tracking in one map fails (e.g. a door→hallway transition, poor texture, fast motion, occlusion), it automatically switches to a new map, constantly checking all maps in the background with one active map at a time; a verified candidate match triggers a merge event. Map is a graph structure — full map (nodes = keyframes + map points; edges = observations connecting them; IMU edges connect keyframe nodes), covisibility graph (nodes = keyframes only, defines "local neighborhoods" cheaply — an edge exists between two keyframes if they share observed map points, weighted by the number shared), and essential graph (a sparser subgraph keeping only the strongest covisibility edges plus loop-closure edges; nodes are poses only, no points). Main contributions: tight IMU integration, fisheye/wide-FOV support, and the Atlas multi-map system.
6. **LSD-SLAM** — Engel, Schöps, Cremers, ECCV 2014. Direct (feature-less) monocular SLAM building semi-dense maps via photometric image alignment on the similarity group.
   https://doi.org/10.1007/978-3-319-10605-2_54
   **Note:** Monocular; works directly on pixels, no feature detection. A different philosophical approach to SLAM: instead of extracting sparse features (like ORB), estimate camera motion by directly minimizing the photometric error.
7. **DSO (Direct Sparse Odometry)** — Engel, Koltun, Cremers, IEEE TPAMI 2018. Jointly optimizes camera motion and sparse inverse depth using full photometric calibration, without feature extraction.
   https://arxiv.org/abs/1607.02565 · project page: https://cvg.cit.tum.de/research/vslam/dso ✓
   **Note:** Successor of LSD-SLAM. Uses a sliding window of recent keyframes — when a keyframe leaves the window it's marginalized into a prior — with joint optimization over the window to predict camera poses, inverse depth, and camera intrinsics/photometric calibration parameters.

### Dense SLAM
8. **DTAM** — Newcombe, Lovegrove, Davison, ICCV 2011. Pioneered dense, per-pixel monocular scene reconstruction with direct whole-image alignment for camera tracking.
   https://www.doc.ic.ac.uk/~ajd/Publications/newcombe_etal_iccv2011.pdf
   **Note:** LSD-SLAM's idea except dense = full 3D surface reconstruction. Computationally very expensive; the first real case of dense SLAM.
9. **KinectFusion** — Newcombe et al., ISMAR 2011. Real-time dense surface reconstruction from a depth camera using volumetric TSDF fusion and ICP-based tracking.
   https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf
   **Note:** Uses a TSDF voxel grid; direct ancestor of InfiniTAM. Fuses many noisy depth measurements into one clean volume. Kinect was one of the first cheap consumer RGB-D cameras.
10. **ElasticFusion** — Whelan, Salas-Moreno, Glocker, Davison, Leutenegger, RSS 2015/IJRR 2016. Surfel-based dense SLAM maintaining global map consistency through local and global surface deformations instead of a pose graph.
    https://www.roboticsproceedings.org/rss11/p01.pdf ✓
   **Note:** Uses surfels, not voxels. Hybrid tracking — both geometric (ICP-based) and photometric (direct pixel-intensity alignment, like DSO/LSD-SLAM). Computationally heavy, still needs RGB-D. No discrete pose graph — corrects via geometric surface deformation instead.
11. **InfiniTAM** — Prisacariu, Kähler et al. Real-time voxel-hashing volumetric reconstruction system designed to run on CPUs and mobile hardware.
    https://github.com/victorprad/InfiniTAM
   **Note:** KinectFusion but scalable — solves the fixed-size memory problem and the need for a powerful GPU. Still RGB-D only, so no outdoor/sunlight use.

### Factor-graph / pose-graph optimization backends
12. **g2o** — Kümmerle, Grisetti, Strasdat, Konolige, Burgard, ICRA 2011. General-purpose, extensible C++ library for graph-based nonlinear least-squares optimization, used as the backend in ORB-SLAM and others.
    https://www.researchgate.net/publication/224252449_G2o_A_general_framework_for_graph_optimization ✓
   **Note:** The C++ library used for bundle adjustment throughout this review.
13. **GTSAM / iSAM2** — Dellaert (GTSAM); Kaess et al., IJRR 2012 (iSAM2). Factor-graph formulation of SLAM/VIO with an incremental Bayes-tree solver enabling efficient online updates.
    https://gtsam.org/ · https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf
   **Note:** iSAM2's contribution is updating the optimization incrementally rather than solving from scratch each time.
14. **Bundle Adjustment — A Modern Synthesis** — Triggs, McLauchlan, Hartley, Fitzgibbon, 1999/2000. Foundational survey formalizing nonlinear least-squares refinement of poses and structure.
    https://hal.science/hal-00548290/document
   **Note:** The original foundation: minimize — given a set of camera poses, a set of 3D points, and the observed 2D pixel locations where each point was seen in each camera — find the poses and points that minimize total reprojection error.

### Visual-inertial odometry/SLAM
15. **MSCKF (Multi-State Constraint Kalman Filter)** — Mourikis & Roumeliotis, ICRA 2007. EKF-based VIO imposing multi-view geometric constraints without adding feature positions to the state.
    https://ieeexplore.ieee.org/document/4209642
   **Note:** Sliding-window, EKF-based — builds a geometric constraint between multiple camera poses directly. Problem: an EKF only ever refines the current state going forward, which is why the field shifted toward optimization-based VIO.
16. **OKVIS** — Leutenegger, Lynen, Bosse, Siegwart, Furgale, IJRR 2015. Tightly-coupled optimization-based VIO using keyframes with marginalization.
    https://journals.sagepub.com/doi/10.1177/0278364914554813
   **Note:** Represents the shift to the optimization/bundle-adjustment framework for VIO. Keyframe-based sliding window — marginalizes older states out of the window (folds them into a prior).
17. **ROVIO** — Bloesch, Omari, Hutter, Siegwart, IROS 2015. Direct, photometric EKF fusing IMU with multilevel patch features.
    https://ieeexplore.ieee.org/document/7353389
   **Note:** Takes the direct/photometric philosophy (like DSO/LSD-SLAM — track using pixel-intensity patches rather than descriptor matching) but implements it inside an EKF framework (like MSCKF) rather than an optimization framework (like OKVIS).
18. **VINS-Mono** — Qin, Li, Shen, IEEE T-RO 2018. Robust monocular VIO with sliding-window optimization, IMU preintegration, and loop closure.
    https://arxiv.org/abs/1708.03852
   **Note:** A mature, complete optimization-based system: sliding-window tightly-coupled optimization with IMU preintegration; has a real map, so it counts as full SLAM — robust initialization + tightly-coupled VIO + loop closure + relocalization.

### LiDAR SLAM
19. **LOAM** — Zhang & Singh, RSS 2014. Low-drift, real-time LiDAR odometry and mapping split into high-frequency odometry and lower-frequency mapping.
    https://www.ri.cmu.edu/pub_files/2014/7/Ji_LidarMapping_RSS2014_v8.pdf
   **Note:** Extracts edge/plane features; fast local estimate (high-frequency odometry) plus a slower, more thorough correction (low-frequency, refined mapping).
20. **LeGO-LOAM** — Shan & Englot, IROS 2018. Lightweight, ground-optimized variant of LOAM for embedded/UGV platforms.
    https://dl.acm.org/doi/10.1109/IROS.2018.8594299 ✓
   **Note:** LOAM adapted specifically for ground vehicles — ground-plane segmentation as an explicit first step, plus a lightweight design.
21. **Cartographer** — Hess, Kohler, Rapp, Andor, ICRA 2016. Submap-based scan matching with branch-and-bound loop closure for real-time 2D LiDAR SLAM.
    https://research.google/pubs/real-time-loop-closure-in-2d-lidar-slam/
   **Note:** Submap-based scan matching + branch-and-bound loop closure + pose-graph optimization. Each submap is a small local probability grid; the pose graph's nodes are individual scan poses (with submap origins also participating); the edges are scan-matching constraints (intra-submap) and loop-closure matches found via branch-and-bound search (inter-submap). The grid is what gets matched against — the graph is what gets optimized using the results of that matching.

## 1B. Modern (learning-based / foundation models)

### NeRF-based mapping/SLAM
22. **iMAP** — Sucar, Liu, Ortiz, Davison, ICCV 2021. First real-time RGB-D SLAM representing the scene as a single implicit neural network, jointly optimized with camera poses.
    https://arxiv.org/abs/2103.12352
   **Note:** Represents the whole scene as a single MLP.
23. **NICE-SLAM** — Zhu, Peng et al., CVPR 2022. Hierarchical, multi-resolution feature grid making neural implicit SLAM scalable.
    https://arxiv.org/abs/2112.12130
24. **NICER-SLAM** — Zhu et al., 2023. RGB-only hierarchical neural implicit SLAM with SDF and monocular cues.
    https://arxiv.org/abs/2302.03594
25. **NeRF-SLAM** — Rosinol, Leonard, Carlone, IROS 2023. Monocular depth/uncertainty network with volumetric neural rendering for real-time dense SLAM.
    https://arxiv.org/abs/2210.13641
26. **Orbeez-SLAM** — Chung et al., ICRA 2023. ORB-feature visual odometry + instant NeRF for fast dense mapping without pretraining.
    https://arxiv.org/abs/2209.13274
27. **Co-SLAM** — Wang, Wang, Agapito, CVPR 2023. Hybrid joint coordinate + hash-grid encoding for fast convergence and hole-filling.
    https://arxiv.org/abs/2304.14377
28. **Point-SLAM** — Sandström, Li, Van Gool, Oswald, ICCV 2023. Dense neural point representation adaptively densified by image gradients.
    https://arxiv.org/abs/2304.04278
29. **Nerfstudio** — Tancik et al., SIGGRAPH 2023. Open, modular PyTorch framework standardizing NeRF training/rendering.
    https://arxiv.org/abs/2302.04264

### 3D Gaussian Splatting SLAM
30. **3D Gaussian Splatting for Real-Time Radiance Field Rendering** — Kerbl, Kopanas, Leimkühler, Drettakis, ACM ToG 2023. Explicit, differentiable 3D Gaussian primitives with fast rasterization for real-time novel-view synthesis.
    https://arxiv.org/abs/2308.04079
31. **SplaTAM** — Keetha, Karhade, Jatavallabhula, Yang, Scherer, Ramanan, Luiten, CVPR 2024. Dense RGB-D SLAM built directly on explicit 3D Gaussians.
    https://arxiv.org/abs/2312.02126
32. **Gaussian Splatting SLAM (MonoGS)** — Matsuki, Murai, Kelly, Davison, CVPR 2024. Monocular/stereo/RGB-D Gaussian-splat SLAM with direct photometric tracking.
    https://arxiv.org/abs/2312.06741
33. **Photo-SLAM** — Huang, Li, Cheng, Yeung, CVPR 2024. Real-time SLAM with simultaneous photorealistic Gaussian-splat mapping.
    https://arxiv.org/abs/2311.16728
34. **GS-SLAM** — Yan et al., CVPR 2024. Adaptive 3D Gaussian scene representation with coarse-to-fine tracking.
    https://arxiv.org/abs/2311.11700
35. **Gaussian-SLAM** — Yugay, Li, Gevers, Oswald, 2023. Photorealistic dense SLAM organizing Gaussians into submaps for unbounded scenes.
    https://arxiv.org/abs/2312.10070
   **Note:** Organizes Gaussians into submaps.
36. **RTG-SLAM** — Peng et al., SIGGRAPH 2024. Compact, low-redundancy real-time Gaussian mapping for large indoor scenes.
    https://arxiv.org/abs/2404.19706
37. **Splat-SLAM** — Sandström et al., 2024/CVPR 2025. Globally optimized RGB-only Gaussian-splatting SLAM with loop closure and dense bundle adjustment.
    https://arxiv.org/abs/2405.16544 ✓
38. **LoopSplat** — 2024. Loop closure by registering 3D Gaussian-splat submaps against each other.
    https://arxiv.org/abs/2408.10154

### Learned depth/pose SLAM
39. **DROID-SLAM** — Teed & Deng, NeurIPS 2021. End-to-end differentiable bundle adjustment driven by learned optical-flow updates.
    https://arxiv.org/abs/2108.10869 ✓
   **Note:** Trains a neural network to iteratively refine pose and depth estimates, using a differentiable version of bundle adjustment as one component inside the network's own iteration loop.
40. **DeepFactors** — Czarnowski et al., RA-L/ICRA 2020. Real-time probabilistic dense monocular SLAM optimizing a learned depth-code representation.
    https://arxiv.org/abs/2001.05049
41. **DPV-SLAM (Deep Patch Visual Odometry/SLAM)** — Lipson, Teed, Deng, ECCV 2024. Patch-based deep visual SLAM balancing efficiency and accuracy.
    https://arxiv.org/abs/2408.13878
42. **GO-SLAM** — Zhang et al., ICCV 2023. Globally optimized RGB-only implicit SLAM with online loop closure and full bundle adjustment.
    https://arxiv.org/abs/2309.02436

### 3D foundation models for reconstruction and pose
43. **DUSt3R** — Wang, Leroy, Cabon, Chidlovskii, Revaud, CVPR 2024. Feed-forward transformer regressing dense 3D pointmaps from an image pair without calibration or poses.
    https://arxiv.org/abs/2312.14132 ✓
44. **MASt3R** — Leroy, Cabon, Revaud, ECCV 2024. Adds a local feature head, metric pointmaps, and fast reciprocal matching to DUSt3R.
    https://arxiv.org/abs/2406.09756 ✓
45. **MASt3R-SfM** — Duisterhof et al., 3DV 2025. Full unconstrained SfM pipeline using MASt3R matches plus image retrieval.
    https://arxiv.org/abs/2409.19152 ✓
46. **VGGT (Visual Geometry Grounded Transformer)** — Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny, CVPR 2025. Jointly predicts camera parameters, depth, pointmaps, and point tracks in one forward pass.
    https://arxiv.org/abs/2503.11651 ✓
47. **Fast3R** — Yang, Sax et al., CVPR 2025. Multi-view DUSt3R generalization reconstructing 1000+ images in one forward pass.
    https://fast3r-3d.github.io/ ✓ · arXiv: https://arxiv.org/abs/2501.13928
48. **MUSt3R** — Cabon et al., CVPR 2025. Multi-view network with a latent scene state, avoiding pairwise global alignment.
    https://www.researchgate.net/publication/394513146_MUSt3R_Multi-view_Network_for_Stereo_3D_Reconstruction ✓ · arXiv: https://arxiv.org/abs/2503.01661
49. **CUT3R** — Wang et al., CVPR 2025. Persistent, continuously updated internal state for streaming 3D perception from video.
    https://arxiv.org/abs/2501.12387
50. **Spann3R** — Wang, Agapito, CVPR 2025. External spatial memory for online registration against accumulated scene memory.
    https://arxiv.org/abs/2408.13912
51. **SLAM3R** — Liu et al., CVPR 2025. Sliding-window dense reconstruction producing real-time, globally consistent point maps from monocular video.
    https://arxiv.org/abs/2412.09401
52. **StreamVGGT** — 2025. Causal/streaming variant of VGGT processing video frame-by-frame.
    https://arxiv.org/abs/2507.11539
53. **VGGT-Long** — 2025. Extends VGGT to long video via chunked processing and inter-chunk alignment.
    https://arxiv.org/abs/2507.16443
54. **MASt3R-SLAM** — Murai, Dexheimer, Davison, CVPR 2025. Real-time dense SLAM built on the MASt3R prior.
    https://arxiv.org/abs/2412.12392
55. **VGGT-SLAM** — Maggio, Lim, Carlone, NeurIPS 2025. Dense RGB SLAM aligning VGGT submaps on the SL(4) projective manifold.
    https://arxiv.org/abs/2505.12549
56. **SLAM-Former** — Yuan, Chen et al., 2025. Integrates the full frontend and a global-refinement backend of SLAM into one transformer.
    https://arxiv.org/abs/2509.16909 ✓

---

# SECTION 2 — LOCALIZATION IN A MAP (incl. relocalization across time/viewpoint/appearance)

## 2A. Traditional

### Place recognition / loop closure
57. **FAB-MAP** — Cummins & Newman, IJRR 2008. Probabilistic, appearance-only place recognition using a bag-of-visual-words vocabulary with a Chow-Liu tree.
    https://www.semanticscholar.org/paper/Highly-scalable-appearance-only-SLAM-FAB-MAP-2.0-Cummins-Newman/db39b7be3c56071b610f5557aa93066d442d529c ✓
58. **DBoW2** — Gálvez-López & Tardós, IEEE T-RO 2012. Fast binary-descriptor vocabulary-tree place recognition, used as ORB-SLAM's loop-closure module.
    https://doi.org/10.1109/TRO.2012.2197158
59. **VLAD** — Jégou, Douze, Schmid, Pérez, CVPR 2010. Compact global image descriptor via residual aggregation to a small codebook.
    https://doi.org/10.1109/CVPR.2010.5540039

### Probabilistic localization
60. **Monte Carlo Localization** — Dellaert, Fox, Burgard, Thrun, ICRA 1999. Particle-filter pose belief for global localization and tracking.
    https://www.borg.cc.gatech.edu/projects/monte-carlo-localization.html ✓
61. **Robust Monte Carlo Localization** — Thrun, Fox, Burgard, Dellaert, Artificial Intelligence 2001. Extends MCL for robustness to noise and the kidnapped-robot problem.
    https://www.sciencedirect.com/science/article/pii/S0004370200000904

### Feature pipelines (SIFT/ORB + PnP/RANSAC)
62. **SIFT** — Lowe, IJCV 2004. Scale- and rotation-invariant keypoint detector/descriptor.
    https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94 ✓
63. **ORB** — Rublee, Rabaud, Konolige, Bradski, ICCV 2011. Fast binary alternative to SIFT/SURF.
    https://doi.org/10.1109/ICCV.2011.6126544

## 2B. Modern (learning-based / foundation models)

### Learned visual place recognition (appearance-robust)
64. **NetVLAD** — Arandjelović, Gronát, Torii, Pajdla, Sivic, CVPR 2016. First differentiable VLAD aggregation layer for end-to-end place recognition.
    https://arxiv.org/abs/1511.07247
65. **CosPlace** — Berton, Masone, Caputo, CVPR 2022. Large-scale classification-based training on a very large curated Street View dataset.
    https://arxiv.org/abs/2204.02287
66. **EigenPlaces** — Berton, Trivigno, Caputo, Masone, ICCV 2023. Viewpoint-robust training across multiple viewpoints of the same place.
    https://arxiv.org/abs/2308.10832
67. **MixVPR** — Ali-bey, Chaib-draa, Giguère, WACV 2023. All-MLP feature-mixing aggregation for global descriptors.
    https://arxiv.org/abs/2303.02190
68. **AnyLoc** — Keetha et al., RA-L 2023. Frozen DINOv2 features + VLAD/GeM for universal, training-free place recognition.
    https://arxiv.org/abs/2308.00688 ✓
69. **SALAD** — Izquierdo & Civera, CVPR 2024. Reformulates soft assignment as optimal transport with a learned dustbin on fine-tuned DINOv2.
    https://arxiv.org/abs/2405.11244
70. **MegaLoc** — Berton & Masone, 2025. Retrieval model unifying VPR, SfM retrieval, and visual localization.
    https://arxiv.org/abs/2502.17237

### Foundation-model features
71. **DINOv2** — Oquab et al., 2023. Self-supervised ViT producing general-purpose dense features reused across VPR, matching, and pose estimation.
    https://arxiv.org/abs/2304.07193
72. **DINO** — Caron et al., ICCV 2021. Predecessor self-supervised ViT showing object-aware attention emerges without labels.
    https://arxiv.org/abs/2104.14294
73. **CLIP** — Radford et al., ICML 2021. Contrastive image-text pretraining producing joint embeddings for zero-shot retrieval.
    https://arxiv.org/abs/2103.00020

### Learned feature matching
74. **SuperGlue** — Sarlin, DeTone, Malisiewicz, Rabinovich, CVPR 2020. GNN + optimal transport for learned correspondence matching.
    https://arxiv.org/abs/1911.11763
75. **LoFTR** — Sun, Shen, Wang, Bao, Zhou, CVPR 2021. Detector-free, dense transformer-based matcher robust to low texture.
    https://arxiv.org/abs/2104.00680
76. **LightGlue** — Lindenberger, Sarlin, Pollefeys, ICCV 2023. Faster, adaptive-depth successor to SuperGlue.
    https://arxiv.org/abs/2306.13643 ✓
77. **SuperPoint** — DeTone, Malisiewicz, Rabinovich, CVPRW 2018. Self-supervised joint keypoint detection and description.
    https://arxiv.org/abs/1712.07629

### Hierarchical localization
78. **HLoc (From Coarse to Fine)** — Sarlin, Cadena, Siegwart, Dymczyk, CVPR 2019. Global retrieval → local matching → PnP/RANSAC pipeline.
    https://arxiv.org/abs/1812.03506
   **Note:** Draw from HLoc for our own pipeline, which currently outputs one 6DoF pose from the localization step. Alternative: instead of one joint window with all k references, run k separate LingBot-Map calls (query + reference 1 alone, query + reference 2 alone, ...) — each gives an independent relative pose, composed with that reference's known global pose into k separate global pose hypotheses for the same query frame. Check whether most of them cluster together (agree within some distance/angle threshold); if 4 out of 5 agree closely and 1 is a wild outlier, discard the bad reference and average/refine over the remaining consensus.
79. **Back to the Feature** — Sarlin et al., CVPR 2021. Learns camera localization end-to-end from pixels to pose.
    https://arxiv.org/abs/2103.09213
80. **LaMAR** — Sarlin et al., ECCV 2022. Large-scale benchmark for localization/mapping targeting AR use cases.
    https://arxiv.org/abs/2210.10770

### Absolute/relative pose regression
81. **PoseNet** — Kendall, Grimes, Cipolla, ICCV 2015. First CNN to regress 6DoF pose from a single RGB image.
    https://arxiv.org/abs/1505.07427
82. **MapNet** — Brahmbhatt et al., CVPR 2018. Geometric and visual-odometry consistency constraints to regularize pose regression.
    https://arxiv.org/abs/1712.03342
83. **AtLoc** — Wang et al., AAAI 2020. Self-attention focusing pose regression on geometrically robust regions.
    https://arxiv.org/abs/1909.03557
84. **VLocNet** — Valada, Radwan, Burgard, ICRA 2018. Multi-task CNN jointly regressing global pose and relative visual odometry.
    https://arxiv.org/abs/1808.10276
85. **Understanding the Limitations of CNN-based Absolute Camera Pose Regression** — Sattler, Zhou, Pollefeys, Leal-Taixé, CVPR 2019. Shows APR often no more accurate than image-retrieval baselines.
    https://arxiv.org/abs/1903.07504
   **Note:** Explicitly debunks the idea that you can just brute-force into localizing via direct pose regression — this is exactly why people do HLoc (hierarchical localization; e.g., our LingBot-Map → DINOv2 pipeline) instead.

### Scene coordinate regression
86. **DSAC / DSAC++ / DSAC\*** — Brachmann et al., CVPR 2017/2018; TPAMI 2021. Differentiable RANSAC + dense scene-coordinate regression.
    https://arxiv.org/abs/1611.05705 (DSAC) · https://arxiv.org/abs/1711.10228 (DSAC++)
87. **ACE (Accelerated Coordinate Encoding)** — Brachmann, Cavallari, Prisacariu, CVPR 2023. Scene-agnostic backbone + tiny scene-specific head, minutes to train.
    https://arxiv.org/abs/2305.14059
88. **GLACE** — Wang et al., CVPR 2024. Global-local accelerated coordinate encoding scaling SCR to larger environments.
    https://arxiv.org/abs/2406.04340
89. **R-SCoRe** — 2024/2025. Robust scene coordinate regression for generalization across appearance change.
    https://arxiv.org/abs/2501.01823
90. **ACE-G** — 2024. Generalizing ACE variant reducing per-scene retraining.
    https://arxiv.org/abs/2408.14740
91. **ACE0** — Brachmann et al., 2024. Learns relocalization maps without known ground-truth poses at training time.
    https://arxiv.org/abs/2404.14351

### Long-term / appearance-change localization
92. **Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions** — Sattler, Maddern, Toft et al., CVPR 2018. Introduces the Aachen Day-Night / RobotCar Seasons / CMU Seasons benchmark suite.
    https://arxiv.org/abs/1707.09092 · benchmark site: https://www.visuallocalization.net/
93. **DeepMEL** — Gridseth & Barfoot, ICRA 2020. Compiles multiple lighting/seasonal "experiences" into one network for robust teach-and-repeat.
    https://arxiv.org/abs/2001.02725
   **Note:** Contrast with Doan et al.'s filtering approach below — DeepMEL compiles a fixed set of known conditions rather than continuously updating.
94. **Keeping an Eye on Things: Deep Learned Features for Long-Term Visual Localization** — Gridseth & Barfoot, 2021.
    https://arxiv.org/abs/2109.04041 ✓
95. **Visual Localization Under Appearance Change: A Filtering Approach** — Doan, Latif, Chin, Reid. Filtering-based localization under continual environment change.
    https://arxiv.org/abs/1811.08063
   **Note:** Rather than pre-training robustness to a fixed set of conditions, treat localization as a filtering problem over time: let each new observation both localize the robot and incrementally update the model's belief about current appearance, rather than expecting one static reference map (or one static robust feature set) to remain valid forever without ever being refreshed.

---

# SECTION 3 — PHYSICAL NAVIGATION IN THE MAP

## 3A. Traditional

### Occupancy grids, costmaps, ROS stacks
96. **Nav2 ("The Marathon 2")** — Macenski, Martín, White, Ginés Clavero, 2020. Modular, pluggable ROS 2 navigation stack (global planners + costmap-based local controller).
    https://arxiv.org/abs/2003.00368
   **Note:** Almost certainly what's actually running on the MyAGV under the hood. Modular pipeline: a global planner (NavFn — Dijkstra/A*, or SMAC — Hybrid-State A*) computes a coarse path across the whole known map, and a local controller (DWB, a modernized Dynamic Window Approach) handles moment-to-moment obstacle avoidance and trajectory execution, reacting to live sensor data the global plan couldn't have anticipated.
97. **Occupancy grid mapping** — Elfes/Moravec foundational work. Probabilistic grid representation updated by fusing sensor readings.
    https://www.ri.cmu.edu/pub_files/pub3/elfes_alberto_1989_1/elfes_alberto_1989_1.pdf
   **Note:** Builds a live obstacle map the planner reasons over directly.

### Path planning
98. **A\*** — Hart, Nilsson, Raphael, IEEE SSC 1968. Classic heuristic graph-search for minimum-cost paths.
    https://doi.org/10.1109/TSSC.1968.300136
99. **D\* / D\* Lite** — Koenig & Likhachev, AAAI 2002. Incremental heuristic search for efficient replanning as the map changes.
    https://www.cs.cmu.edu/~maxim/files/dlite_tro05.pdf
100. **RRT** — LaValle, TR 98-11, Iowa State 1998. Sampling-based tree planner for high-dimensional/kinodynamic problems.
    https://lavalle.pl/rrtpubs.html ✓
101. **PRM** — Kavraki, Švestka, Latombe, Overmars, IEEE T-RA 1996. Reusable roadmap of sampled collision-free configurations.
    https://www.researchgate.net/publication/3298646_Probabilistic_Roadmaps_for_Path_Planning_in_High-Dimensional_Configuration_Spaces ✓
102. **RRT\* / PRM\*** — Karaman & Frazzoli, IJRR 2011. Adds asymptotic optimality via rewiring.
    https://arxiv.org/abs/1105.1186
103. **Dynamic Window Approach** — Fox, Burgard, Thrun, IEEE RAM 1997. Reactive local planner over an acceleration-limited velocity window.
    https://doi.org/10.1109/100.580977
   **Note:** At each control cycle, consider only the velocities achievable given the robot's current speed and acceleration limits (the "dynamic window"), simulate short trajectories forward for each candidate velocity, score them (progress toward goal, distance from obstacles, smoothness), and execute the best-scoring one. Almost certainly what Nav2's DWB controller is doing live on the MyAGV, reacting to LiDAR obstacle data in real time — the layer that would actually stop the robot from running into something the topological graph/global plan didn't know about.
104. **Social Force Model** — Helbing & Molnár, Physical Review E 1995. Attraction-repulsion forces modeling pedestrian dynamics.
    https://arxiv.org/abs/cond-mat/9805244
   **Note:** Relevant only if the deployment environment has moving people/other agents to navigate around (plausible, given it's an office-like environment with a microkitchen etc.) — these model how to smoothly avoid other moving agents without collision or awkward oscillation, treating them as either physical forces to move around (Social Force) or as reciprocally cooperating agents each taking half the avoidance responsibility (ORCA).
105. **ORCA (Optimal Reciprocal Collision Avoidance)** — van den Berg, Guy, Lin, Manocha, ISRR 2011. Multi-agent reciprocal collision avoidance.
    https://gamma.cs.unc.edu/ORCA/publications/ORCA.pdf

## 3B. Modern (learning-based)

### End-to-end learned navigation policies
106. **GNM (General Navigation Model)** — Shah, Sridhar, Bhorkar, Hirose, Levine, ICRA 2023. Shared navigation action space for zero-shot cross-embodiment transfer.
    https://www.researchgate.net/publication/372123300_GNM_A_General_Navigation_Model_to_Drive_Any_Robot ✓ · arXiv: https://arxiv.org/abs/2210.03370
107. **ViNT (Visual Navigation Transformer)** — Shah, Sridhar, Dashora, Stachowicz, Black, Hirose, Levine, CoRL 2023. Transformer foundation model for image-goal navigation with topological graph + diffusion subgoals.
    https://arxiv.org/abs/2306.14846
   **Note:** Most similar to our own work: a transformer trained for image-goal navigation ("here's a photo of where I want to go, get me there"), paired explicitly with a topological graph and diffusion-based subgoal proposals for handling long routes. Key difference: ViNT's model is trained end-to-end for action prediction, whereas our setup separates localization (DINOv2+LingBot-Map) from navigation execution (MyAGV's own stack) as genuinely distinct stages.
108. **NoMaD** — Sridhar, Shah, Glossop, Levine, ICRA 2024. Diffusion policy unifying directed navigation and exploration via goal masking.
    https://arxiv.org/abs/2310.07896
   **Note:** Extends this with goal-masking so one policy handles both directed navigation and open-ended exploration, depending on whether a goal is specified.
109. **SACSoN** — Hirose et al., RA-L 2023. Scalable autonomous social navigation via counterfactual learning.
    https://arxiv.org/abs/2306.01874
   **Note:** Adds social awareness (navigating considerately around people) via counterfactual training — learning what effect the robot's actions actually had on nearby people's behavior, not just imitating trajectories.

### Vision-language navigation
110. **R2R (Room-to-Room)** — Anderson et al., CVPR 2018. Foundational VLN benchmark in real photorealistic environments.
    https://arxiv.org/abs/1711.07280
111. **VLN-BERT** — Hong et al., CVPR 2021. Panoramic BERT selecting the next navigation node from language + panoramic vision.
    https://arxiv.org/abs/2011.13922
112. **LM-Nav** — Shah et al., CoRL 2022. Combines an LLM, a VLM, and a learned visual navigation model over a topological graph.
    https://arxiv.org/abs/2207.04429
   **Note:** Explicitly combines an LLM/VLM with a topological graph — same "language/semantic layer on top of a graph" shape as ours, just with the graph nodes tied to language-parseable landmarks rather than the room labels we've defined manually.
113. **NavGPT** — Zhou, Hong, Wu, AAAI 2024. Explicit LLM reasoning for step-by-step VLN planning.
    https://arxiv.org/abs/2305.16986
114. **NavGPT-2** — Zhou et al., ECCV 2024. Bridges VLM latents with topological VLN policies.
    https://arxiv.org/abs/2407.12366
115. **CoW (CLIP on Wheels)** — Gadre et al., 2022. Zero-shot object-goal navigation grounded by CLIP embeddings.
    https://arxiv.org/abs/2203.10421
116. **NaVid** — Zhang et al., RSS 2024. Video-based VLM predicting next navigation action from egocentric video and instructions.
    https://arxiv.org/abs/2402.15852

### Reinforcement learning for navigation
117. **SARL (Socially Attentive Reinforcement Learning)** — Chen, Liu, Kreiss, Alahi, ICRA 2019. Pairwise/group human-interaction modeling for crowd navigation.
    https://arxiv.org/abs/1809.08835
118. **Motion Planning Among Dynamic, Decision-Making Agents with Deep RL** — Everett, Chen, How, IROS 2018. LSTM-based deep RL collision avoidance among a variable number of moving agents.
    https://arxiv.org/abs/1805.01956
119. **Map-based DRL crowd navigation (PPO)** — General line of work training navigation policies on local costmaps/lidar with PPO, typically trained in ORCA/Social-Force simulators before sim-to-real transfer. (No single canonical paper — search "PPO local planner crowd navigation" on Semantic Scholar for current work.)
120. **ReViND** — Offline RL for vision-based navigation with post-hoc reward specification.
    https://arxiv.org/abs/2308.07405

### Hybrid / hierarchical topological navigation
121. **Topological Mapping and Navigation using a Monocular Camera based on AnyLoc** — 2025. AnyLoc descriptors build a topological keyframe graph for loop detection and navigation without a metric map.
    https://arxiv.org/abs/2601.01067 ✓
   **Note:** Close to a direct blueprint for what we're doing, minus the LingBot-Map precision-pose layer — builds a topological keyframe graph purely from AnyLoc (DINOv2+VLAD) descriptor similarity, used for loop detection and navigation with no metric map at all. Contrast: this method treats the topological graph itself as sufficient for navigation (move toward the next node whose descriptor looks closer to the goal), whereas we add LingBot-Map specifically to get metric precision within a node, not just topological connectivity between nodes — a deliberate choice to combine coarse+fine rather than relying on coarse alone. Also relevant here: retrieval-quality gating from the VPR literature broadly — AnyLoc/CosPlace papers report retrieval performance as "recall@k," the confidence that the true match is even among the top-k; this same metric can be used to filter out bad frames/matches upstream of localization.
122. **MG-Nav (Memory-Guided Navigation)** — 2025. Dual-scale global memory-guided planning with local geometry-enhanced control.
    https://arxiv.org/abs/2503.09300
   **Note:** Dual-scale (global memory-guided + local geometry-enhanced) planning.

---

## Synthesis / Discussion of Trends
Three cross-cutting trends emerge. **(1) Geometric → learned → foundation-model.** Reconstruction moved from SfM/MVS and filter/keyframe SLAM to end-to-end differentiable SLAM (DROID-SLAM) and now to pose-free feed-forward transformers (DUSt3R/MASt3R/VGGT), which are already being wrapped back into real-time SLAM systems (MASt3R-SLAM, VGGT-SLAM). **(2) Convergence on hybrid pipelines.** The strongest localization results come from the HLoc pattern — self-supervised global retrieval (DINOv2/AnyLoc/SALAD) → learned local matching (SuperGlue/LightGlue/LoFTR) → classical PnP/RANSAC — with scene coordinate regression (ACE/DSAC*) as a compact learned alternative; absolute pose regression remains the weakest link. **(3) Topological + local decomposition for navigation.** Both classical (Nav2: global A*/Dijkstra + local DWB) and modern (ViNT/NoMaD: topological graph + learned local policy) systems converge on a global-graph/local-metric split — precisely the LingBot-Map architecture.

## Recommendations (for the LingBot-Map project)
1. **Anchor the relocalization front-end on the HLoc paradigm** — DINOv2/AnyLoc or SALAD for coarse retrieval, then LightGlue + PnP/RANSAC or your learned 6DoF regressor for fine pose; benchmark against HLoc as a baseline.
2. **Adopt ACE as a comparison/fallback** — compact, fast to train, already built on DINOv2 features.
3. **Evaluate appearance robustness on RobotCar Seasons / Extended CMU Seasons**, preferring DINOv2-based descriptors, with sequence-based verification against perceptual aliasing.
4. **Keep MyAGV navigation on Nav2**, using the topological room/hallway graph as the global layer feeding metric goals — mirroring ViNT/NoMaD.
5. **Watch the 3D foundation-model track for mapping** — MASt3R-SLAM and VGGT-SLAM for offline map (re)construction; LiDAR SLAM for online metric localization.
6. **Retrieval-quality gating** — from the VPR literature broadly: AnyLoc/CosPlace papers report retrieval performance as "recall@k," the confidence that the true match is even among the top-k; this same metric can be used to filter out bad frames/matches before they ever reach the pose-estimation stage.
7. **Sequence-based re-ranking** — if queries come from a moving robot rather than isolated stills, don't retrieve independently per-frame; require that a *sequence* of consecutive query frames retrieves a spatially coherent sequence of references (frame t matches keyframe 40, frame t+1 should match near keyframe 41, not keyframe 5 on the other side of the building). Catches perceptual-aliasing errors that single-frame retrieval margin alone can miss, and is cheap since frame ordering is already available.
8. **EigenPlaces' explicit viewpoint-robustness training** — worth knowing even while using AnyLoc's training-free approach: if DINOv2+VLAD retrieval specifically fails under viewpoint change (approaching a room from a different angle than the walkthrough captured), EigenPlaces-style fine-tuning (deliberately training on multiple viewpoints of the same physical place) is the documented fix for that specific failure mode, distinct from the appearance/lighting robustness AnyLoc targets.
9. **LaMAR's evaluation methodology** — built specifically around evaluating localization across *sessions* (map built once, queries from a different session/day), exactly this use case; worth reviewing how they structure train/query splits and error metrics for reporting accuracy numbers later.

Decision thresholds: promote learned pose regression to primary if it achieves <5 cm / 2° indoors and generalizes across sessions; otherwise keep the HLoc/ACE hybrid. If DINOv2 retrieval recall@1 drops below ~80% under appearance change, add sequence matching or fine-tune SALAD on in-domain data.

## Caveats
- Links marked ✓ were confirmed live via search this session; the rest are the canonical arXiv ID/DOI from standard bibliographic record but were not individually re-verified in this pass — if one 404s, the title will resolve instantly on Google Scholar or Semantic Scholar.
- Several 2025/2026-dated preprints (VGGT-SLAM, StreamVGGT, VGGT-Long, and various DUSt3R-family follow-ups) are very recent arXiv postings; treat venue/year as provisional until camera-ready versions appear.
- Absolute pose regression accuracy claims are contested (Sattler et al., CVPR 2019) — do not assume learned 6DoF beats geometric methods without in-domain evaluation.
- Some pre-2015 classics (DTAM, KinectFusion, LOAM, Cartographer, MSCKF, OKVIS, ROVIO, DWA, ORCA, occupancy grids, D*/D* Lite) predate or bypass arXiv norms; links point to project/lab-hosted PDFs or DOIs rather than arXiv.

---

# SECTION 4 — 2026 SOTA ADDITIONS

123. **LingBot-Map** — Robbyant/Ant Group, 2026. Streaming 3D reconstruction via Geometric Context Attention (GCA): an *anchor context* for coordinate/scale grounding, a local *pose-reference window* for dense nearby geometry, and a *trajectory memory* compressing full history into compact per-frame tokens — near-constant memory/compute over 10,000+ frame sequences at ~20 FPS. DINO backbone → alternating Frame Attention/GCA → pose + depth heads. Same lineage as CUT3R/Spann3R (persistent-state streaming reconstruction), with a more explicit three-part memory decomposition than either. Native output is per-frame pose/depth during a single continuous streaming run — it has no built-in "relocalize a fresh disconnected query frame against an already-built map" mode, which is the localization capability we're building on top via DINOv2 retrieval + windowed relative-pose queries.
    **Note:** This is a full 3D transformer, same class as VGGT/DUSt3R — dense depth+pose per frame plus known camera geometry gives a genuinely complete streaming point-cloud reconstruction (per the official demo: multi-room traversal, aerial scenes, roaming), not a thin localization-only signal. GCA's trajectory memory was purpose-built to handle long sequences natively (not a chunk-and-stitch retrofit like VGGT-Long/StreamVGGT), so it's a real candidate for offline long-term map-building on its own merits, not just the localization utility we're using it as — worth benchmarking directly against VGGT/MASt3R on a long sequence rather than assuming either is better a priori.
    https://technology.robbyant.com/lingbot-map · tech report: https://arxiv.org/abs/2604.14141

124. **JanusVLN** — Zeng, Qi, Chang, Xiong, Xie, Wu, Liang, Xu, Wei, Guo, ICLR 2026. Vision-language navigation with **dual implicit memory**: a semantic memory (from a VLM, Qwen2.5-VL-7B) and a spatial memory built by extending the VLM with **VGGT** as a frozen 3D geometry backbone, giving 3D spatial structure from RGB video alone — no depth sensor, no explicit stored map, no textual cognitive-map bookkeeping. Only the LLM and a projection layer are fine-tuned; VGGT and the semantic encoder stay frozen. Validated on a real Unitree Go2 with remote-GPU inference. Directly relevant: it's essentially "VGGT as a frozen geometry backbone feeding a decision layer" — the same fusion this review's own architecture performs manually (DINOv2 retrieval + LingBot-Map pose feeding a topological-graph navigation decision), just realized as one jointly-trained model instead of composed modules.
    https://arxiv.org/abs/2509.22548 · code: https://github.com/MIV-XJTU/JanusVLN

125. **LightNav-0** — Light Origins, Sept 2026. Real2Sim2Real: 2,000+ internet-sourced real scenes turned into 4,000+ hours of simulated vision-language-action experience; trained via Embodied Reasoning mid-training → Embodied SFT → Online RL (GRPO). Core intermediate representation is **image-space "Point CoT"** (an object point + a free-space affordance point, both 2D pixel coordinates in the current frame) rather than metric 3D targets — explicitly avoiding needing depth, calibration, or a shared coordinate system across robot embodiments. Motion is tokenized via a 3-level residual vector quantizer over SE(2) trajectories. State-of-the-art on 10 monocular VLN/ObjectNav/tracking benchmarks; zero-shot across 4 physical robot embodiments. Their own INSIGHT-Bench disaggregates results by instruction mechanism (Base/Direction/Relation/Extremum/Ordinal) and scene type rather than reporting one aggregate success rate — a more diagnostically useful evaluation format than most VLN benchmarks.
    https://www.lightorigins.com/en/blog/lightnav-0 · tech report: https://static.lightorigins.com/website/reports/lightnav-0-technical-report_02cfc2f.pdf · code: https://github.com/lightorigins/LightNav-0 · benchmark: https://lightorigins.github.io/Light-INSIGHT-Bench/

126. **DyGeoVLN** — Liu, Zheng, Jeong, Yoon, Zhao, Zhong, Li, Yoon, 2026. "Infusing a *dynamic* geometry foundation model into vision-language navigation" — near-identical philosophy to JanusVLN (fuse a geometry foundation model with a VLM for navigation) but specifically targeting scenes that change over time, closer to the appearance-change/map-staleness problem than JanusVLN's static-scene framing.
    https://arxiv.org/abs/2603.21269

127. **FutureNav** — 2026. Unified world-action modeling for VLN — predicts future scene state jointly with action, rather than decoupling perception/memory from action the way JanusVLN does. Explicitly benchmarks against and reports outperforming JanusVLN on R2R/RxR (FutureNav-4B improves 4.4% SR on R2R and 6.0% SR on RxR over JanusVLN under matched training-data conditions) — the immediate next-step-past-JanusVLN reference if pursuing the end-to-end direction.
    https://arxiv.org/abs/2606.30367

128. **StreamVLN, Uni-NaVid, NaVILA** — 2024-2025. The SOTA tier immediately preceding JanusVLN/LightNav-0/FutureNav; useful as baselines for gauging how much the newest 2026 releases are actually improving over, rather than taking their self-reported comparisons at face value.

129. **CorrectNav** — 2026. Strong monocular VLN baseline appearing in LightNav-0's own comparison tables (65.1 SR on R2R, just behind LightNav-0 and the Qwen-RobotNav family) — worth including in any current monocular-VLN frontier survey.

---

## How People Typically Approach Localization (Summary)
1. **Coarse retrieval** — narrow a potentially huge map down to a handful of plausible candidates cheaply (DBoW2 classically, DINOv2/AnyLoc/SALAD now). Never trusted as a final answer — a fast filter only.
2. **Fine geometric estimation against the candidates** — local feature matching + PnP (HLoc classic), dense scene coordinate regression + PnP (ACE), or a learned geometric predictor (LingBot-Map).
3. **Robustness/consensus check** — RANSAC over pooled correspondences (HLoc), inlier-count thresholds, or multi-hypothesis consensus across independently-computed pose estimates, since step 2 can be fooled by any single bad candidate.
4. **Confidence gating and fallback** — fall back to the last known good pose plus dead-reckoning until a confident relocalization arrives, or explicitly flag "lost" and trigger a wider re-search (ORB-SLAM3's Atlas philosophy).
5. **(Increasingly common) Long-term adaptation** — treat the map as something periodically refreshed from new observations (Doan et al.'s filtering philosophy) rather than a fixed, one-shot artifact.

## Patterns Observed Across This Review
1. **Coarse-then-fine, everywhere.** LOAM's fast-odometry+slow-mapping, Cartographer's submap+branch-and-bound, HLoc's retrieval+matching, ViNT's topological-graph+diffusion-subgoal, DINOv2+LingBot-Map — every working system splits "roughly where am I" from "precisely where am I," using different tools at different rates.
2. **A persistent map beats a chain of relative estimates whenever drift matters.** DSO (no persistent map) drifts unrecoverably; ORB-SLAM/KinectFusion (check against accumulated model) drift less; ElasticFusion/loop-closure systems (explicit revisit-and-correct) drift least. Naive sequential frame-appending is a live instance of the worst end of this spectrum.
3. **Single-hypothesis pipelines are fragile to one bad input; consensus/verification is what production systems add on top.** RANSAC in PnP, essential-graph optimization in ORB-SLAM, geometric verification in DBoW2/HLoc, k-hypothesis consensus for reference fusion — mature pipelines never trust one estimate blindly.
4. **Metric precision and topological/semantic structure are different needs, solved by different tools, composed hierarchically.** Nav2's global/local split, ViNT's graph+policy split, AnyLoc-topological's "graph alone is enough" vs. combining a graph with metric pose within a node, LightNav-0's rejection of metric pose entirely in favor of image-space points — this axis (shared coordinate frame vs. purely relative/pointing information) is the single biggest design fork in the field.
5. **The field's trajectory is "replace hand-engineered optimization with a trained model, then re-add classical structure (windows, graphs, consensus, loop closure) once the learned version hits scale/robustness limits."** DROID-SLAM → DUSt3R → VGGT → CUT3R/streaming-wrappers for reconstruction; PoseNet's failure → HLoc/ACE's hybrid dominance for localization; end-to-end VLA (LightNav-0) vs. modular (LM-Nav) is the same tension currently playing out for navigation.

## A Fresh Design (Independent of Current Implementation)

### Mapping (long-term)
Do not treat the map as one static artifact captured once. The recurring failure mode across this review — motion blur, appearance drift, a bad reference silently corrupting everything downstream — traces to treating the map as fixed.
- **Map = a living store of (keyframe, pose, quality-score, last-verified-timestamp) records**, not a frozen image folder. New observations from live deployment should be eligible to replace/supplement stale or low-quality entries automatically, gated by confidence.
- **Reconstruction**: LingBot-Map itself is a real candidate for the offline map-building pass, not just a localization utility — it's a full 3D transformer (same class as VGGT/DUSt3R) producing dense point-cloud reconstructions, not a sparse skeleton, and its Geometric Context Attention was purpose-built to handle long sequences without chunking (trajectory memory compressing full history into compact per-frame tokens, near-constant memory/compute over 10,000+ frames) — a native design for long sequences, not a chunk-and-stitch retrofit like VGGT-Long/StreamVGGT, so there's no clear reason to prefer VGGT/MASt3R over it for this pass on drift grounds. The one open question worth testing empirically: whether GCA's *compression* into trajectory-memory tokens introduces its own subtle long-range drift from information loss, as distinct from the chunk-misalignment drift that VGGT-Long/VGGT-SLAM's submap alignment exists to manage. Worth benchmarking LingBot-Map's own reconstruction against VGGT/MASt3R on a long test sequence before assuming either is better.
- **Blur/quality gating at ingestion**, unconditionally — cheap, no architectural cost, and directly prevents the exact failure class already observed.

### Localization
Build the HLoc pattern properly, with two independent geometric estimators rather than one:
- Retrieval: DINOv2/AnyLoc (frozen, no training needed at this stage); EigenPlaces-style fine-tuning as a scoped upgrade path if viewpoint robustness specifically becomes the bottleneck.
- Precise pose: run **two structurally different estimators** — a windowed relative-pose model (e.g., LingBot-Map) and ACE (fast to train per-room, dense-regression+RANSAC, fails differently than attention-based fusion) — and treat *disagreement between them* as the actual confidence signal, rather than trusting either one's internal confidence.
- Default to **k-independent-hypotheses-plus-consensus** rather than joint-window fusion; only fall back to joint-window fusion once consensus across independent estimates is already high.

### Getting the robot to follow the path
Keep navigation execution classical (Nav2: global A*/SMAC + local DWB) rather than adopting end-to-end learned policies (ViNT/NoMaD/LightNav-0) for actual motion at this stage — a classical planner is debuggable and its failures are attributable; a learned policy's are not, and the reliability payoff for switching hasn't been earned yet. The one idea worth borrowing from the learned side: **image-space pointing (LightNav-0-style) as a cheap sanity-check overlay** — "does the rough pointing direction roughly agree with where the topological graph says to go?" — not as the control loop itself, since it catches localization failures Nav2 has no way to notice, without replacing Nav2's predictable control loop.

## Framework for Evaluating Any New Paper
1. **Which of the three sub-problems does it touch — reconstruction, localization, or navigation execution — and is it improving the coarse stage or the fine stage?** Tells you whether it competes with the retrieval stage, the pose/matching stage, or the execution stage, or is a different layer entirely.
2. **Does it report failure modes, or only aggregate success-rate averages?** Aggregate SR/SPL tells you little about robustness to a specific failure class (blur, aliasing, stale maps); disaggregated evaluation (by instruction type, by scene type, by condition) is far more actionable.
3. **Is the claimed improvement about accuracy on clean data, or about robustness/generalization?** Given where actual bugs originate, a paper improving clean-condition SOTA by a couple points is less relevant than one specifically targeting appearance change, corrupted input, or cross-session localization.
4. **Can it be tested against existing test data before being adopted?** Anything droppable into an existing evaluation pipeline gives a real answer quickly; anything requiring retraining/re-architecting before it can even be evaluated should queue behind cheaper options.
