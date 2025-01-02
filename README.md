# Code for Event-Enhanced Snapshot Mosaic Hyperspectral Frame Deblurring, TPAMI 2024

## Usage

### Step 1: Download the Dataset
Download the real-captured Event-SMHC Frame Deblurring dataset from BaiduNetDisk:  
[Dataset Link](https://pan.baidu.com/s/1YCQjS6ucHLvaJdYHiD9NzQ?pwd=7dvn)  
Password: `7dvn`

The dataset contains blur/gt sharp frames and corresponding event voxels.

Besides event voxels, we also provide raw events (and blur/gt frames) for the train/test/validation sets:  
[Raw Data Link](https://pan.baidu.com/s/1Nu4W4PW6YDQBgpwayZg2NA?pwd=i5ek)
Password: `i5ek`


### Step 2: Extract the Dataset
Extract the `train` and `test` tar files into the same directory.

### Step 3: Configure Data Paths
Modify the data paths in `config/train.json` and `config/test.json` to point to the directory containing the extracted files.

### Step 4: Start Training
Run the training script:  

bash run_train.sh

Ensure all required dependencies are installed. Install any missing packages until no errors occur.

### Step 5: Update Model Path for Testing
Once training is complete, update the `"resume_state"` field in `config/test.json` with the path to the saved model checkpoint.  

### Step 6: Start Testing
Run the testing script:  

bash run_test.sh

### Acknowledgements
This code is built upon the [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) project.
We sincerely thank the authors for their excellent implementation.


