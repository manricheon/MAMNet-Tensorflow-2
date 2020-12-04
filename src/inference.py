import os
from importlib import import_module
import tensorflow as tf

from trainer import MAMNetTrainer
from utils import load_image, plot_sample, save_image
from model import resolve_single, resolve_chop_single
from options import args


# GPU number and memory growth setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[args.gpu_id],'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu_id],True)
    except RuntimeError as e:
        print(e)



def inference():


    # model setting
    model_module = import_module("model." + args.model_name)
    model = model_module.create_generator(args)
    model.summary(line_length=120)
    print('Model created')

    # trainer setting
    trainer = MAMNetTrainer(model=model, ckpt_path=args.ckpt_path, args=args)
	
    # input path
    input_dir = args.test_input
    input_list = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    input_path = [os.path.join(input_dir, img_name) for img_name in input_list]

    # output path
    output_dir = args.test_output
    os.makedirs(output_dir, exist_ok=True)
    output_list = [f for f in input_list]
    output_path = [os.path.join(output_dir, img_name) for img_name in output_list]

    inout_path = zip(input_path, output_path)

    # inference and save
    for in_name, out_name in inout_path:
        img = load_image(in_name)
        img_out = resolve_chop_single(trainer.model, img)
        # img_out = resolve_single(trainer.model, img)
        save_image(img_out, out_name)
    
    print('inference done')

if __name__ == '__main__':
    inference()
