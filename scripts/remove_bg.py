# Author: graemeniedermayer

import modules.scripts as scripts
import gradio as gr

from modules import processing, images, shared, sd_samplers, devices
from modules.processing import create_infotext, process_images, Processed
from modules.shared import opts, cmd_opts, state, Options
from PIL import Image
from pathlib import Path

import sys
import torch, gc
import torch.nn as nn
import cv2
import os.path
import numpy as np

from rembg import remove, new_session

scriptname = "remove_bg v0.0.1"

class Script(scripts.Script):
	def title(self):
		return scriptname

	def show(self, is_img2img):
		return True

	def ui(self, is_img2img):

		with gr.Row():
			model_type = gr.Dropdown(label="Model", choices=['u2net','u2netp','u2net_cloth_seg','u2net_human_seg', 'silueta'], value='u2net_cloth_seg', type="value", elem_id="model_type")
		with gr.Row():	
			convert_to_mask = gr.Checkbox(label="convert to mask",value=True)
			output_top = gr.Checkbox(label="output top",value=True)
			output_bottom = gr.Checkbox(label="output bottom",value=True)
			output_combined = gr.Checkbox(label="output combined",value=True)
		
		#would passing a dictionary be more readable?
		return [model_type, convert_to_mask, output_top, output_bottom, output_combined]

	def run(self, p, model_type, convert_to_mask, output_top, output_bottom, output_combined):

		# sd process 
		processed = processing.process_images(p)

		# unload sd model
		shared.sd_model.cond_stage_model.to(devices.cpu)
		shared.sd_model.first_stage_model.to(devices.cpu)

		print('\n%s' % scriptname)
		# init torch device

		# model path and name
		model_dir = Path.joinpath(Path().resolve(), "models/rem_bg")
		os.makedirs(model_dir, exist_ok=True)
		os.environ["U2NET_HOME"] = str(model_dir)

		print("Starting session ", end=" ")

		session = new_session(model_type)

		try:

			devices.torch_gc()

			# iterate over input (generated) images
			numimages = len(processed.images)
			for count in range(0, numimages):
				# skip first (grid) image if count > 1
				if count == 0 and numimages > 1:
					continue

				if numimages > 1:
					print("Processed image ", count, '/', numimages-1)

				# input image
				# img = cv2.cvtColor(np.asarray(processed.images[count]), cv2.COLOR_BGR2RGB)
				
				remove_img = remove(processed.images[count], session=session)

				# get generation parameters
				if hasattr(p, 'all_prompts') and opts.enable_pnginfo:
					info = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, "", 0, count-1)
				else:
					info = None

				# output
				if model_type == 'u2net_cloth_seg':
					split = np.array(remove_img)
					hsplit = int(split.shape[0]/3)
					top = split[:hsplit, :] 
					bot = split[hsplit:2*hsplit,:]
					comb =  split[2*hsplit:,:]
					# example of boolean masking and casting
					if output_bottom:
						if convert_to_mask:
							bot[(bot[:,:,0]>1) | (bot[:,:,1]>1) | (bot[:,:,2]>1)] = 255
						bot_img = Image.fromarray(bot)
						processed.images.append(bot_img)
						images.save_image(bot_img, p.outpath_samples, "", processed.all_prompts[count-1], p.prompt, opts.samples_format, info=info, p=p, suffix="_bot")
					if output_top:
						if convert_to_mask:
							top[(top[:,:,0]>1) | (top[:,:,1]>1) | (top[:,:,2]>1)] = 255
						top_img = Image.fromarray(top)
						processed.images.append(top_img)
						images.save_image(top_img, p.outpath_samples, "", processed.all_prompts[count-1], p.prompt, opts.samples_format, info=info, p=p, suffix="_top")
					if output_combined:
						if convert_to_mask:
							comb[(comb[:,:,0]>1) | (comb[:,:,1]>1) | (comb[:,:,2]>1)] = 255
						combined_img = Image.fromarray(comb)
						processed.images.append(combined_img)
						images.save_image(combined_img, p.outpath_samples, "", processed.all_prompts[count-1], p.prompt, opts.samples_format, info=info, p=p, suffix="_combined")
				else:
					remove_arr = np.array(remove_img)
					if convert_to_mask:
						remove_arr[(remove_arr[:,:,0]>1) | (remove_arr[:,:,1]>1) | (remove_arr[:,:,2]>1)] = 255
					rem_img = Image.fromarray(remove_arr)
					processed.images.append(rem_img)
					images.save_image(rem_img, p.outpath_samples, "", processed.all_prompts[count-1], p.prompt, opts.samples_format, info=info, p=p, suffix="_rem_bg")

		except RuntimeError as e:
			if 'out of memory' in str(e):
				print("ERROR: out of memory, could not run rem_bg!")
			else:
				print(e)

		finally:
			gc.collect()
			devices.torch_gc()

			# reload sd model
			shared.sd_model.cond_stage_model.to(devices.device)
			shared.sd_model.first_stage_model.to(devices.device)

		return processed
