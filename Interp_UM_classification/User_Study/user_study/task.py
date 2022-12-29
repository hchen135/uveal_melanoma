from user_study import app
from flask import render_template, request, jsonify
from glob import glob
import sys
from user_study.util import zoom_fullres_name_locate,pie_chart_display_prep
from openslide import OpenSlide
import os
import json


zoom_fullres_name = "/static/blank_image.png"

cell_images = ["/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",]

SLIDE_INPUT_DIR = "./user_study/static/slides"
EMBEDDING_RESULT_DIR = "./user_study/static/pie_chart/pie_chart_json"

SLIDE_13 = OpenSlide(os.path.join(SLIDE_INPUT_DIR,'Slide 13.svs'))
SLIDE_24 = OpenSlide(os.path.join(SLIDE_INPUT_DIR,'Slide 24.svs'))
SLIDE_29 = OpenSlide(os.path.join(SLIDE_INPUT_DIR,'Slide 29.svs'))
SLIDE_51 = OpenSlide(os.path.join(SLIDE_INPUT_DIR,'Slide 51.svs'))
SLIDE_59 = OpenSlide(os.path.join(SLIDE_INPUT_DIR,'Slide 59.svs'))
SLIDE_65 = OpenSlide(os.path.join(SLIDE_INPUT_DIR,'Slide 65.svs'))

with open(os.path.join(EMBEDDING_RESULT_DIR,'umap_proj_'+'13'+'_info.json')) as a:
	pie_chart_dict_13 = json.load(a)

with open(os.path.join(EMBEDDING_RESULT_DIR,'umap_proj_'+'24'+'_info.json')) as a:
	pie_chart_dict_24 = json.load(a)

with open(os.path.join(EMBEDDING_RESULT_DIR,'umap_proj_'+'29'+'_info.json')) as a:
	pie_chart_dict_29 = json.load(a)

with open(os.path.join(EMBEDDING_RESULT_DIR,'umap_proj_'+'51'+'_info.json')) as a:
	pie_chart_dict_51 = json.load(a)

with open(os.path.join(EMBEDDING_RESULT_DIR,'umap_proj_'+'59'+'_info.json')) as a:
	pie_chart_dict_59 = json.load(a)

with open(os.path.join(EMBEDDING_RESULT_DIR,'umap_proj_'+'65'+'_info.json')) as a:
	pie_chart_dict_65 = json.load(a)

@app.route('/task1-introduction')
def task1_introduction():
	return render_template('task1-introduction.html')


@app.route('/task1-GUI_description', methods=['GET', 'POST'])
def task1_GUI_description():
	global zoom_fullres_name 
	
	# zoom_fullres_name = "/static/blank_image.png"
	if request.method == 'POST':
		request_json = request.get_json()
		if request_json.get('object') == 'zoom':
			zoom_fullres_name_tmp= zoom_fullres_name_locate(request_json,globals()['SLIDE_59'])
			if zoom_fullres_name_tmp != "":
				zoom_fullres_name = zoom_fullres_name_tmp
				print(zoom_fullres_name,file=sys.stdout)
			else:
				zoom_fullres_name = "/static/blank_image.png"
			# return jsonify(zoom_fullres_name = zoom_fullres_name)
	if request.method == 'GET':
		zoom_fullres_name = "/static/blank_image.png"
		print(request,file=sys.stdout)
		print(dir(request),file=sys.stdout)
	# print(zoom_fullres_name,file=sys.stdout)
	# email = request_json.get('email')
	# password = request_json.get('password')
	# print(email,passward,file=sys.stdout)
	return render_template('task1-GUI_description.html',zoom_fullres_name=zoom_fullres_name)

@app.route('/task1-slide-<slide_id>', methods=['GET', 'POST'])
def task1_main(slide_id):
	global zoom_fullres_name 
	# zoom_fullres_name = "/static/blank_image.png"
	if request.method == 'POST':
		request_json = request.get_json()
		if request_json.get('object') == 'zoom':
			zoom_fullres_name_tmp= zoom_fullres_name_locate(request_json,globals()['SLIDE_'+slide_id])
			if zoom_fullres_name_tmp != "":
				zoom_fullres_name = zoom_fullres_name_tmp
				print(zoom_fullres_name,file=sys.stdout)
			else:
				zoom_fullres_name = "/static/blank_image.png"
			# return jsonify(zoom_fullres_name = zoom_fullres_name)
	if request.method == 'GET':
		zoom_fullres_name = "/static/blank_image.png"
	# print(zoom_fullres_name,file=sys.stdout)
	return render_template('task1-main.html',slide_id=slide_id,zoom_fullres_name=zoom_fullres_name)

@app.route('/task1-per_case_survey_<slide_id>', methods=['GET'])
def task1_per_case_survey(slide_id):
	global zoom_fullres_name 
	zoom_fullres_name = "/static/blank_image.png"
	return render_template('task1-per_case_survey.html',slide_id=slide_id)

@app.route('/task1-post_survey', methods=['GET'])
def task1_post_survey():
	return render_template('task1-post_survey.html')

@app.route('/task1-fullres_loading', methods=['GET'])
def task1_fullres_loading():
	return jsonify(zoom_fullres_name = zoom_fullres_name)


@app.route('/task2-introduction')
def task2_introduction():
	return render_template('task2-introduction.html')


@app.route('/task2-GUI_description', methods=['GET', 'POST'])
def task2_GUI_description():
	global zoom_fullres_name 
	
	# zoom_fullres_name = "/static/blank_image.png"
	if request.method == 'POST':
		request_json = request.get_json()
		if request_json.get('object') == 'zoom':
			zoom_fullres_name_tmp= zoom_fullres_name_locate(request_json,globals()['SLIDE_59'])
			if zoom_fullres_name_tmp != "":
				zoom_fullres_name = zoom_fullres_name_tmp
				print(zoom_fullres_name,file=sys.stdout)
			else:
				zoom_fullres_name = "/static/blank_image.png"
	if request.method == 'GET':
		zoom_fullres_name = "/static/blank_image.png"

	return render_template('task2-GUI_description.html',zoom_fullres_name=zoom_fullres_name)

@app.route('/task2-slide-<slide_id>', methods=['GET', 'POST'])
def task2_main(slide_id):
	global zoom_fullres_name 
	# zoom_fullres_name = "/static/blank_image.png"
	if request.method == 'POST':
		request_json = request.get_json()
		if request_json.get('object') == 'zoom':
			zoom_fullres_name_tmp= zoom_fullres_name_locate(request_json,globals()['SLIDE_'+slide_id],)
			if zoom_fullres_name_tmp != "":
				zoom_fullres_name = zoom_fullres_name_tmp
				print(zoom_fullres_name,file=sys.stdout)
			else:
				zoom_fullres_name = "/static/blank_image.png"
	if request.method == 'GET':
		zoom_fullres_name = "/static/blank_image.png"

	return render_template('task2-main.html',slide_id=slide_id,zoom_fullres_name=zoom_fullres_name)

@app.route('/task2-per_case_survey_<slide_id>', methods=['GET'])
def task2_per_case_survey(slide_id):
	global zoom_fullres_name 
	zoom_fullres_name = "/static/blank_image.png"
	return render_template('task2-per_case_survey.html',slide_id=slide_id)

@app.route('/task2-post_survey', methods=['GET'])
def task2_post_survey():
	return render_template('task2-post_survey.html')

@app.route('/task2-fullres_loading', methods=['GET'])
def task2_fullres_loading():
	return jsonify(zoom_fullres_name = zoom_fullres_name)

@app.route('/task3-introduction')
def task3_introduction():
	return render_template('task3-introduction.html')


@app.route('/task3-GUI_description1', methods=['GET', 'POST'])
def task3_GUI_description1():
	global zoom_fullres_name 
	global cell_images
	if request.method == 'POST':
		request_json = request.get_json()
		if request_json.get('object') == 'zoom':
			zoom_fullres_name_tmp= zoom_fullres_name_locate(request_json,globals()['SLIDE_59'])
			if zoom_fullres_name_tmp != "":
				zoom_fullres_name = zoom_fullres_name_tmp
				print(zoom_fullres_name,file=sys.stdout)
			else:
				zoom_fullres_name = "/static/blank_image.png"
		if request_json.get('object') == 'pie_chart':
			cell_images_tmp = pie_chart_display_prep(request_json,globals()['pie_chart_dict_59'])
			if cell_images_tmp != "":
				cell_images = cell_images_tmp
	if request.method == 'GET':
		zoom_fullres_name = "/static/blank_image.png"
		cell_images = ["/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",]

	return render_template('task3-GUI_description1.html',zoom_fullres_name=zoom_fullres_name)

@app.route('/task3-GUI_description2', methods=['GET', 'POST'])
def task3_GUI_description2():
	global zoom_fullres_name 
	global cell_images
	if request.method == 'POST':
		request_json = request.get_json()
		if request_json.get('object') == 'zoom':
			zoom_fullres_name_tmp= zoom_fullres_name_locate(request_json,globals()['SLIDE_59'])
			if zoom_fullres_name_tmp != "":
				zoom_fullres_name = zoom_fullres_name_tmp
				print(zoom_fullres_name,file=sys.stdout)
			else:
				zoom_fullres_name = "/static/blank_image.png"
		if request_json.get('object') == 'pie_chart':
			cell_images_tmp = pie_chart_display_prep(request_json,globals()['pie_chart_dict_59'])
			if cell_images_tmp != "":
				cell_images = cell_images_tmp
	if request.method == 'GET':
		zoom_fullres_name = "/static/blank_image.png"
		cell_images = ["/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",]

	return render_template('task3-GUI_description2.html',zoom_fullres_name=zoom_fullres_name)

@app.route('/task3-slide-<slide_id>', methods=['GET', 'POST'])
def task3_main(slide_id):
	global zoom_fullres_name 
	global cell_images
	if request.method == 'POST':
		request_json = request.get_json()
		if request_json.get('object') == 'zoom':
			zoom_fullres_name_tmp= zoom_fullres_name_locate(request_json,globals()['SLIDE_'+slide_id])
			if zoom_fullres_name_tmp != "":
				zoom_fullres_name = zoom_fullres_name_tmp
				print(zoom_fullres_name,file=sys.stdout)
			else:
				zoom_fullres_name = "/static/blank_image.png"
		if request_json.get('object') == 'pie_chart':
			print("slide id:",file=sys.stdout)
			cell_images_tmp = pie_chart_display_prep(request_json,globals()['pie_chart_dict_'+slide_id],status="description")
			if cell_images_tmp != "":
				cell_images = cell_images_tmp
	if request.method == 'GET':
		zoom_fullres_name = "/static/blank_image.png"
		cell_images = ["/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",]


	return render_template('task3-main.html',slide_id=slide_id,zoom_fullres_name=zoom_fullres_name)

@app.route('/task3-per_case_survey_<slide_id>', methods=['GET'])
def task3_per_case_survey(slide_id):
	global zoom_fullres_name 
	global cell_images
	zoom_fullres_name = "/static/blank_image.png"
	cell_images = ["/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",
			   "/static/blank_image.png",]

	return render_template('task3-per_case_survey.html',slide_id=slide_id)

@app.route('/task3-post_survey', methods=['GET'])
def task3_post_survey():
	return render_template('task3-post_survey.html')

@app.route('/task3-fullres_loading', methods=['GET'])
def task3_fullres_loading():
	return jsonify(zoom_fullres_name = zoom_fullres_name)

@app.route('/task3-pie_chart_loading', methods=['GET'])
def task3_pie_chart_loading():
	return jsonify(cell1 = cell_images[0],
		           cell2 = cell_images[1],
		           cell3 = cell_images[2],
		           cell4 = cell_images[3],
		           cell5 = cell_images[4],
		           cell6 = cell_images[5])

@app.route('/final_survey', methods=['GET'])
def final_survey():
	return render_template('final_survey.html')
