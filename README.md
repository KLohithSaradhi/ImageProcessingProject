# IVP-Project

## Project Title

**NTire Dense and Non-Homogeneous Dehazing Challenge**

## Team Members

Kandukuri Lohith Saradhi

Achintya Lakshmanan

## Requirements
- Numpy 1.24.3
- OpenCV-Python 4.9.0.80
- Pillow (PIL) 10.2.0
- Pytorch 2.2.1
- Pytorch-cuda 12.1 (if running on GPU)
- Scikit-Image 0.21.0
- Scikit-Lean 1.3.0
- TQDM 4.65.0

## Model Weights and Results
[Google Drive](https://drive.google.com/drive/folders/1WE39pOm_FTal7C25mY4AKwt8Il9TkUXb?usp=sharing)

## Data Folder structure

>Data
>>clear<br>
>>hazy





## To Augment the data

- Create a folder with the following structure

>AugmentedData
>>clear<br>
>>hazy

- Run augmentation.ipynb with the appropriate paths for src and target paths.

## To train

- Create a <i>weights</i> folder

- Change the GPU index in <i>train.py</i> as per the hardware.

- Run <i>train.py</i>

## For Inference and processed Output

#### Inference

- Create a folder with the following structure

>Submission
>>\<folder name><br>
>> .<br>
>> .<br>
>> .<br>

- Change the Submission folder path in <i>submission.ipynb</i> file.

- Run the <i>submission.ipynb</i> file.

#### Post processing

- Create an output folder in <i>Submission</i> as:

>Submission
>>\<folder name><br>
>> .<br>
>> .<br>
>> .<br>

- In <i>hist_eq.ipynb</i>, change the <i>input_path</i> to the path with the model inference and the <i>savepath</i> to the target folder.


