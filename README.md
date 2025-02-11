# DLCV-Fall-2021-HW4


Please refer to [hw4_intro.pdf](https://docs.google.com/presentation/d/1nUbj_qLHLNm9GZppw-msVuxuxwib5Z6L/edit?usp=sharing&ouid=102321765388877382261&rtpof=true&sd=true) for HW4 details and rules. **Note that all of hw4 videos and introduction pdf files can be accessed in your NTU COOL.**

## Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.
```bash
    git clone https://github.com/DLCV-Fall-2021/HW4-<username>.git
```
Note that you should replace `<username>` with your own GitHub username.

## Install Packages
To install the packages automatically, we have provided a "requirements.txt" file for this homework. Please use the following script to set up the environment.
```bash
pip install -r requirements.txt
```

## Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.
```bash
bash ./get_dataset.sh
```
The shell script will automatically download the dataset and store the data in a folder called `hw4_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://docs.google.com/presentation/d/1nUbj_qLHLNm9GZppw-msVuxuxwib5Z6L/edit?usp=sharing&ouid=102321765388877382261&rtpof=true&sd=true) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `hw4_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

## Submission Rules
### Deadline
111/01/04 (Tue.) 02:00 AM (GMT+8)

## Q&A
If you have any problems related to HW4, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under hw4 Discussion section in NTU COOL
