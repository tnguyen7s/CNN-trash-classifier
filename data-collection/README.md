### Resources to set up datasets
https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataloader

### The directories of collected datasets after being processed(labeling and cropping) looks as follows: 
./collected-datasets/[subset]/[category_name]/[annotation_id].[ext] 

	./collected-datasets/Extended-Taco/bio/1_0001.jpg

	./collected-datasets/DrinkingWaste/glass/3_1447.jpg


### Taco Dataset:
1. crop and labels images defined in Taco-annotations-test.json
2. crop and labels images defined in Taco-annotations-train.json
3. all cropped images goes to ./collected-datasets/Extended-Taco/[category_name]
4. all labels goes to ./collected-datasets/Extended-Taco/labels.csv

Notes: 
	
	Taco_annotations_test:  1784 trash instances and Skipped 997 instances

	Cropped and labeled: 7190 trash instances and Skipped 4315 instances
### DrinkingWaste dataset
1. crop images in Yolo_imgs folder using Yolo annotation defined in the corresponding .txt file
2. all images goes to ./collected-datasets/DrinkingWaste/[category_name]
3. all labels goes to ./collected-datasets/DrinkingWaste/labels.csv

Notes:
	
	4811 trash instances
	
	
### Train Test Split: 80-10-10
1. data collected from different sources are saved in `collected-dataset` folder
2. all data moved into a folder `collected-dataset/all`
3. resize image to 224 and save them in `collected-dataset/all-224`
image_size<224: add padding=0
image_size>224: scale down the image
4. images after being resized are split to train-test-val and save in `collected-dataset/train`, `collected-dataset/test`, `collected-dataset/val` respectively






