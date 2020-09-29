import cv2 
import os
import glob

def get_infos(file_path, image, confidence):	
	name = os.path.basename(file_path)
	formated_line = ''.join([name ,', ' , str(image.shape), ', ', confidence])
	return formated_line

def calculate_infos(shapes):
	resolution = [i[0]*i[1] for i in shapes]
	size = len(shapes)

	min_res, max_res, median_res = 0, 0, 0
	if size > 0:
		median_0 = sum([i[0] for i in shapes]) / size
		median_1 = sum([i[1] for i in shapes]) / size

		min_res, max_res = resolution.index(min(resolution)), resolution.index(max(resolution))
		median_res = (int(median_0), int(median_1))

	return size, min_res, max_res, median_res

def create_txt (txt_name_path, info_list, title, shapes):
	size, min_index, max_index, median_res = calculate_infos(shapes)

	with open(txt_name_path, "w") as file:
		if size > 0:
			file.write(str(
				title.upper() + 
				"\n\n(NAME, SHAPE, CONFIDENCE)\n\n" + 
				"TOTAL FILES: " + str(size) + 
				"\nMIN RESOLUTION: " + info_list[min_index].split(', ')[0] + ", " + str(shapes[min_index]) + 
				"\nMAX RESOLUTION: " + info_list[max_index].split(', ')[0] + ", " + str(shapes[max_index]) + 
				"\nMEDIAN RESOLUTION: " + str(median_res) + "\n\n"
					))
			file.write(str('\n'.join(info_list)))

		else:
			file.write("NONE")
			file.write(str('\n'.join(info_list)))