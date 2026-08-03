# Flyer Recast Height Analysis Script

## Overview
The ```Flyer_Recast_Anylsis.py``` script parses through .xyz files generated from profilometer scans of individual flyers and detects the inner and outer ridge created as a result of laser cutting.
It input is a an .xyz file, and the output is a .json file with the median ridge height, as well as the median ridge full width half max estimates for each file (optional output: a diagnostic plot with visualizations of what
was detected as the inner and outer edge).

The general pipeline of this script (also listed as a comment at the top of ```Flyer_Recast_Analysis.py```) is as follows:
- Step 1: Use the points with the highest z-values to estimate a rough center for the cut
- Step 2: Slice the data into cross sections(either XZ or YZ depending on the location of the slice) and detect the peaks from the recast
- Step 3: Group the peaks using radial clustering to form distinct inner and outer edges
- Fit an ellipse to each group of edge points and remove outliers from this fit
- Step 4: Use everything outside the inner and outer edge to fit a plane and use that to flatten the entire dataset (including the recast peaks)
- Step 5: Report statistical metrics on ridge heights, FWHM, and ellipse fit (although only the median height and the median FWHM are included in the final output for simplicity)

The ```Flyer_Recast_Analysis_preFlatten.py``` script follows the same pipeline except after Step 1 it uses the baseline points to fit a plane to the data and flattens before slicing into cross sections
and detecting the peaks. I haven't seen any major difference in performance from the two scripts except that sometimes the resulting ellipse fit from the pre-flatten script seems to be off-center.


## Repository Structure

```
project/
├── README.md
├── Flyer_Recast_Analysis.py
├── Flyer_Recast_Analysis_preFlatten.py
├── Run_Analysis_Notebook.ipynb
└── requirements.txt
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Analysis

1. Clone the repository

Open a terminal and navigate to the folder where you would like to store the project. Then run:

```bash
git clone <repository-url>
pip install -r requirements.txt
```

This will download the repository and create a new project folder.

2. Install dependencies

Install the required Python packages using:
```bash
pip install -r requirements.txt
```

3. Running the Analysis
Option 1: Using Jupyter Notebook

Launch Jupyter Notebook from the project directory:

```bash
jupyter notebook
```

A browser window will open. Open:

```
Run_Analysis_Notebook.ipynb
```

and run the notebook cells.

Option 2: Using VS Code

Open the repository folder in VS Code.
Open Run_Analysis_Notebook.ipynb and run the notebook cells.

## Input

- xyz files named "Flyer*.xyz" (where * represents any number (ex 1005_01, 1005_02)
- Currently, I have the notebook taking input files that are just saved directly in the same folder as the notebook. Feel free to add a separate data folder for organization and change the paths accordingly

## Output

- Prints information about the analysis (mostly used in debugging while I was developing)
- Results dictionary (which you can print separately if you want a more thorough breakdown of the stats it came up with)
- .json file containing just the median height and median FWHM value (separate file generated for each file it analyzes)
- Optional diagnostic plot that visualized the edges it detected

## Notes/Areas for improvement

- Making the signal detection more robust than just using scipy.signal.find_peaks (could look into using cumsum) - though currently the median value seems to be fairly accurate
- Verifying the FWHM measurements (sometimes scipy's inbuilt function to detect widths of the peaks isn't totally accurate, and also, because of the noise, it has a hard time distinguishing the real end of a peak or a drop in signal just due to noise). The median value seems to be a good overall metric for now, but it could very possibly be getting skewed by false measurements. Could consider some more aggressive smoothing before detecting the peaks
- Tested with Python 3.13.5
