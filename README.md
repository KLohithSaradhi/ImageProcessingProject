# IVP-Project

## Project Title

**NTire Dense and Non-Homogeneous Dehazing Challenge**

## Team Members

Kandukuri Lohith Saradhi

Achintya Lakshmanan

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

## Output

### Inference

- Create a folder with the following structure

>Submission
>>\<folder name><br>
>> .<br>
>> .<br>
>> .<br>

- Change the Submission folder path in <i>submission.ipynb</i> file.

- Run the <i>submission.ipynb</i> file.

### Post processing

- Create 2 output folder in <i>Submission</i> as:

>Submission
>>\<folder name><br>
>> .<br>
>> .<br>
>> .<br>

- In <i>hist_eq.ipynb</i>, change the <i>input_path</i> to the path with the model inference and the <i>savepath</i> to the target folder.


