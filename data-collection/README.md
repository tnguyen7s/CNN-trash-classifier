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






