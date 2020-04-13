import os

# all video folders in D drive
# this file also in D drive
# create a folder called exercise
path = '10_Minute_HIIT_AB_WORKOUT'
class_path = ""

def make_new_folder(class_folder):
	class_ = class_folder.split('\\')[0]
	num_folders = len(os.listdir(class_)) + 1
	os.system(f"mkdir {os.path.join(class_, str(num_folders))}")
	copied_folder = os.path.join(class_, str(num_folders))
	print(copied_folder)
	return copied_folder

folder_class = {
	"u": "upper",
	"l": "lower",
	"c": "core",
	"h": "cardio"
	}
	
if __name__ == '__main__':

	# navigate to start point
	image_folders = sorted(os.listdir(path), key=int)
	start = input("Start point:")
	image_folders = image_folders[image_folders.index(start):]

	for image_folder in image_folders:
		folder_path = os.path.join(path, image_folder)
		print(folder_path)
		#getting all images per folder
		image_list = sorted(os.listdir(folder_path), key = lambda x: int(x.split('.')[0]))

		#getting start and end frame
		while True:
			start_frame = input("Enter start frame: ")
			end_frame = input("Enter end frame: ")
			class_name = str(input("Enter class: "))
			
			if start_frame == '' or end_frame == '' or class_name == '':
				break
			
			total_frames = int(end_frame) - int(start_frame) + 1
			
			#move into correct folder
			correct_folder = os.path.join(class_path, folder_class[class_name])
			print(correct_folder)
			
			#making new folder to move copied images
			num_folders = len(os.listdir(folder_class[class_name])) + 1
			os.system(f"mkdir {os.path.join(correct_folder, str(num_folders))}")
			copied_folder = os.path.join(correct_folder, str(num_folders))
			print(copied_folder)
			#iterating over img list
			for i in range(0, total_frames):
				if i % 80 == 0 and i != 0:
					#creates new folder since limit is 50 frames per folder
					copied_folder = make_new_folder(correct_folder)
				image_to_copy = image_list[int(start_frame) + i]
				image_path = os.path.join(folder_path, image_to_copy)
				print(image_path, copied_folder)
				#copying into new folder
				os.system(f"copy {image_path} {copied_folder} >NUL")
			
			
			continue
				
				
			
			
			
	