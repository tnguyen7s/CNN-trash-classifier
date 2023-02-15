### Resources to set up datasets
https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataloader

### The directories of collected datasets after being processed(labeling and cropping) looks as follows: 
./datasets/[subset]/[category_name]/[annotation_id].[ext] 

	./datasets/Taco/bio/1_0001.jpg

	./datasets/DrinkingWaste/glass/3_1447.jpg


### Taco Dataset:
1. crop and labels images defined in Taco-annotations-test.json
2. crop and labels images defined in Taco-annotations-train.json
3. all cropped images goes to ./datasets/Taco/[category_name]
4. all labels goes to ./datasets/Taco/labels.csv

Notes: 
	
	Taco_annotations_test: 957 trash instances and skip 1824 instances

	Taco_annotations_train: 3827 trash instances and skipped 7678 instances

### DrinkingWaste dataset
1. cropped images in Yolo_imgs folder using Yolo annotation defined in the corresponding .txt file
2. all images goes to ./datasets/DrinkingWaste/[category_name]
3. all labels goes to ./datasets/DrinkingWaste/labels.csv

Notes:
	4811 trash instances






