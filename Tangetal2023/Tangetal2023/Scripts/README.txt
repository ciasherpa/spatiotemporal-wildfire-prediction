README 

Original pipeline (Tang et al., 2023, TSECfire): classification_demo.py and regression_demo.py.
Our extensions: tests whether temporal context, spatial context, fire history, season, and other models improve next-month fire prediction. Scripts: rf_baseline.py, convlstm_v2.py, experiments.py, new_ignitions.py, robustness.py, compare_results.py.


All scripts must be run from inside the Scripts folder.

### Prerequisites

install the software and packages listed in the requirements.txt


### Installing

Python 3.10, 3.11, or 3.12 (64-bit).
Git.
Internet connection (for pip install; TensorFlow is a large download, allow 5 to 10 minutes).
No GPU needed.

### Setup

# Step 1: Download the code

git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git
cd <YOUR-REPO>

or

Download as a ZIP file:
Open the GitHub repository.
Click the green Code button.
Select Download ZIP.
Extract the downloaded ZIP file.
Open the extracted spatiotemporal-wildfire-prediction folder in VS Code.

# Step 2: Create a clean virtual environment

Windows (Command Prompt or PowerShell):

python -m venv venv
venv\Scripts\activate

Mac / Linux:

python3 -m venv venv
source venv/bin/activate

You should see (venv) at the start of your prompt.

# Step 3: Install every package

python -m pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Check the install worked

python -c "import numpy, pandas, sklearn, scipy, imblearn, lightgbm, xgboost, catboost, tensorflow, openpyxl, xlsxwriter, tqdm; print('All packages OK')"

It must print All packages OK. If it errors, see Troubleshooting (section 6).

# Step 5: Move into the Scripts folder

cd Scripts


### Running scripts

command: python classification_demo.py	
Standardizes data, splits train/test, applies SMOTE, trains LogReg / Random Forest / Voting classifiers (fire vs no fire), saves metrics and predictions	

command: python regression_demo.py	
Predicts burned area for locations the classifier flagged as fire, with 15 regression models plus error-correction	

command: python common.py	

command: python rf_baseline.py	
Random Forest ablation: last month only, + temporal, + spatial, + both

command: python convlstm_v2.py	
ConvLSTM on the full grid, trained at several fire-weights. Slowest script.

command: python experiments.py	
Reference baselines (persistence, climatology) + 6 models x 4 feature sets, with leakage-safe threshold choice and month-level bootstrap

command: python new_ignitions.py	
Repeats the experiments but scores ONLY rows with no fire last month, to test whether gains hold for brand-new fires

command: python robustness.py	
Robustness checks on the results (see the header of the file for details)

command: python compare_results.py	
Merges all result CSVs into one table
