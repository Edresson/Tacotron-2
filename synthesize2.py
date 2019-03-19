import argparse
import os
from warnings import warn

import tensorflow as tf

from hparams import hparams
from infolog import log

import os
import wave
from datetime import datetime
import numpy as np
import sounddevice as sd
import tensorflow as tf
from datasets import audio
from infolog import log
from librosa import effects
from tacotron.models import create_model
from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence
from datasets import audio

class Synthesizer:
	def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
                log('Constructing model: %s' % model_name)
                inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
                input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
                with tf.variable_scope('model') as scope:
                        self.model = create_model(model_name, hparams)
                        #self.model.initialize(inputs, input_lengths)
                        self.model.initialize(inputs, input_lengths, is_training=False, is_evaluating=True)
                        self.alignments = self.model.alignments
                        self.lf0_outputs = self.model.lf0_outputs
                        self.mgc_outputs = self.model.mgc_outputs
                        self.bap_outputs = self.model.bap_outputs

                self.gta = gta
                self._hparams = hparams
                #pad input sequences with the <pad_token> 0 ( _ )
                self._pad = 0

                log('Loading checkpoint: %s' % checkpoint_path)
                #Memory allocation on the GPU as needed
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                self.session = tf.Session(config=config)
                self.session.run(tf.global_variables_initializer())

                saver = tf.train.Saver()
                saver.restore(self.session, checkpoint_path)


	def synthesize(self, texts, basenames, out_dir, log_dir):
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
		input_lengths = [len(seq) for seq in seqs]
		seqs = self._prepare_inputs(seqs)
		feed_dict = {
			self.model.inputs: seqs,
			self.model.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}
		                        
		lf0s, mgcs, baps, alignments = self.session.run([self.lf0_outputs, self.mgc_outputs, self.bap_outputs, self.alignments], feed_dict=feed_dict)

		for i, _ in enumerate(lf0s):
			# Write the predicted features to disk
			# Note: outputs files and target ones have same names, just different folders
			np.save(os.path.join(out_dir, 'lf0-{:03d}.npy'.format(basenames[i])), lf0s[i], allow_pickle=False)
			np.save(os.path.join(out_dir, 'mgc-{:03d}.npy'.format(basenames[i])), mgcs[i], allow_pickle=False)
			np.save(os.path.join(out_dir, 'bap-{:03d}.npy'.format(basenames[i])), baps[i], allow_pickle=False)

			if log_dir is not None:
				#save alignments
				plot.plot_alignment(alignments[i], os.path.join(log_dir, 'plots/alignment-{:03d}.png'.format(basenames[i])),
					info='{}'.format(texts[i]), split_title=True)

				#save wav
				wav = audio.synthesize(lf0s[i], mgcs[i], baps[i], hparams)
				audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{:03d}.wav'.format(basenames[i])), hparams)


	def eval(self, text):
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seqs = [np.asarray(text_to_sequence(text, cleaner_names))]
		input_lengths = [len(seq) for seq in seqs]
		feed_dict = {
			self.model.inputs: seqs,
			self.model.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}
		lf0s, mgcs, baps = self.session.run([self.model.lf0_outputs, self.model.mgc_outputs, self.bap_outputs], feed_dict=feed_dict)

		wavs = []
		for i, _ in enumerate(lf0s):
			wavs.append(audio.synthesize(lf0s[i], mgcs[i], baps[i], hparams))
		return np.concatenate(wavs)


	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs])

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)



def generate_fast(model, text):
	model.synthesize(text, None, None, None, None)


def run_live(args, checkpoint_path, hparams):
	#Log to Terminal without keeping any records in files
	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	#Generate fast greeting message
	greetings = 'Hello, Welcome to the Live testing tool. Please type a message and I will try to read it!'
	log(greetings)
	generate_fast(synth, greetings)

	#Interaction loop
	while True:
		try:
			text = input()
			generate_fast(synth, text)

		except KeyboardInterrupt:
			leave = 'Thank you for testing our features. see you soon.'
			log(leave)
			generate_fast(synth, leave)
			sleep(2)
			break

def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
	eval_dir = os.path.join(output_dir, 'eval')
	log_dir = os.path.join(output_dir, 'logs-eval')

	#Create output path if it doesn't exist
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
		for i, text in enumerate(tqdm(sentences)):
			start = time.time()
			#wav= synth.eval(text)
			#a=audio.save_wav(wav, os.path.join(log_dir, 'wavs/eval-test.wav'), hparams)
			synth.synthesize([text], [i+1], eval_dir, log_dir)
	return eval_dir


def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
	output_dir = 'tacotron_' + args.output_dir

	try:
		checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
		log('loaded model at {}'.format(checkpoint_path))
	except:
		raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

	if args.mode == 'eval':
		return run_eval(args, checkpoint_path, output_dir, hparams, sentences)
	else:
		run_live(args, checkpoint_path, hparams)


def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	run_name = args.name or args.tacotron_name or args.model
	taco_checkpoint = os.path.join('logs-' + run_name, 'taco_' + args.checkpoint)

	run_name = args.name or args.wavenet_name or args.model
	wave_checkpoint = os.path.join('logs-' + run_name, 'wave_' + args.checkpoint)
	return taco_checkpoint, wave_checkpoint, modified_hp

def get_sentences(args):
	if args.text_list != '':
		with open(args.text_list, 'rb') as f:
			sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
	else:
		sentences = hparams.sentences
	return sentences


def main():
	accepted_modes = ['eval', 'synthesis', 'live']
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
	parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
	parser.add_argument('--wavenet_name', help='Name of logging directory of WaveNet. If trained separately')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
	parser.add_argument('--mels_dir', default='tacotron_output/eval/', help='folder to contain mels to synthesize audio from using the Wavenet')
	parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--mode', default='eval', help='mode of run: can be one of {}'.format(accepted_modes))
	parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
	parser.add_argument('--text_list', default='', help='Text file contains list of texts to be synthesized. Valid if mode=eval')
	parser.add_argument('--speaker_id', default=None, help='Defines the speakers ids to use when running standalone Wavenet on a folder of mels. this variable must be a comma-separated list of ids')
	args = parser.parse_args()

	if args.mode not in accepted_modes:
		raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))

	if args.GTA not in ('True', 'False'):
		raise ValueError('GTA option must be either True or False')

	taco_checkpoint, wave_checkpoint, hparams = prepare_run(args)
	sentences = get_sentences(args)

	tacotron_synthesize(args, hparams, taco_checkpoint, sentences)


if __name__ == '__main__':
	main()
