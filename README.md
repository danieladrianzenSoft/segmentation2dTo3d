# segmentation2dTo3d

## About
This repo contains a series of utility functions used to visualize, analyze, meshify and process granular
material scaffolds in a variety of formats. Its original purpose was to segment 2d slices thorugh scaffolds
into 3d geometries, but since then has evolved to include a series of 'workflows' ranging from visualizing
voxelized scaffold data (from voxelized json files) to generating 2d slices and corresponding 3d geometries
for ml training purposes of segmentation models.

## Running Instructions
run.py is the main file. 

python run.py --help provides instructions on how to run the code and all the workflows available
python run.py --list lists all available workflows

## Architecture
The code consists of 3 main folders:

	- workflows/ 			: contains config variables for all available workflows, returned via get_config()
	- workflow_runner/ 		: main logic to run a specific workflow, executed via run()
	- core/					: all business logic and reusable functions used in various workflows