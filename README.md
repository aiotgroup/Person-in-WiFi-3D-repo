# Person-in-WiFi 3D: End-to-End Multi-Person 3D Pose Estimation with Wi-Fi

Wi-Fi signals, in contrast to cameras, offer privacy protection and occlusion resilience for some practical scenarios such as smart homes, elderly care, and virtual reality.
Recent years have seen remarkable progress in the estimation of single-person 2D pose, single-person 3D pose, and multi-person 2D pose. This paper takes a step forward by introducing Person-in-WiFi 3D, a pioneering Wi-Fi
system that accomplishes multi-person 3D pose estimation. Person-in-WiFi 3D has two main updates. Firstly, it has a greater number of Wi-Fi devices to enhance the capability for capturing spatial reflections from multiple individuals.
Secondly, it leverages the Transformer for end-to-end estimation. Compared to its predecessor, Person-in-WiFi 3D is storage-efficient and fast. We deployed a proof-of-concept system in 4m × 3.5m areas and collected a dataset of over
97K frames with seven volunteers. Person-in-WiFi 3D attains 3D joint localization errors of 91.7mm (1-person), 108.1mm (2-person), and 125.3mm (3-person), comparable to cameras and millimeter-wave radars.


Links to our paper: [Person-in-WiFi 3D: End-to-End Multi-Person 3D Pose Estimation with Wi-Fi](https://github.com/aiotgroup/Person-in-WiFi3D)

## Prerequisites

- Linux
- Python 3.7+
- PyTorch 1.8+
- CUDA 10.1+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/#installation)

## Getting Started

### Installation

Please see [get_started.md](docs/get_started.md) for the basic usage of Opera.

## Acknowledgement

Opera is an open source project built upon [OpenMMLab](https://github.com/open-mmlab/). We appreciate all the contributors who implement this flexible and efficient toolkits.



### File Structrue
```bash
.
│  LICENSE
│  README.md
│  requirements.txt
│  setup.cfg
│  setup.py
├─data
│  ├─wifipose
│  │  ├─train_data
│  │  │     ├─ csi
│  │  │     ├─ keypoint
│  │  │     ├─ train_data_list.txt
│  │  ├─test_data
│  │  │     ├─ csi
│  │  │     ├─ keypoint
│  │  │     ├─ test_data_list.txt
├─config
│  ├─base
│  ├─wifi
│  │  ├─petr_wifi.py
├─docs
│  ├─get_started.md
├─opera
│  ├─apis
│  ├─core
│  ├─datasets
│  ├─models
│  ├─__init__.py
│  ├─version.py
├─requirements
│  ├─build.txt
│  ├─docs.txt
│  ├─mminstall.txt
│  ├─optional.txt
│  ├─readthedocs.txt
│  ├─runtime.txt
│  ├─tests.txt
├─result
├─third_party
│  ├─mmcv
│  ├─mmdet
├─tools
│  ├─dataset_converters
│  ├─dist_test.sh
│  ├─dist_train.sh
│  ├─eval_metric.py
│  ├─test_all.sh
│  ├─test.py
│  ├─train.py


```


