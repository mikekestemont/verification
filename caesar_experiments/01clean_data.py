import glob
import codecs
import os
import shutil

data_dir = '../data/'
dirty_dirs = ["caesar_dev_dirty", "../data/caesar_test_dirty"]

sample_size = 1000

for dirty_dir in dirty_dirs:
	# get a dir for the clean data:
	clean_dir_name = dirty_dir.replace("_dirty", "")
	if os.path.isdir(data_dir+"/"+clean_dir_name):
		shutil.rmtree(data_dir+"/"+clean_dir_name)
	os.mkdir(data_dir+"/"+clean_dir_name)
	# iterate over the original files:
	for filename in glob.glob(data_dir+"/"+dirty_dir+"/*.txt"):
		print(filename)
		dirty_text = codecs.open(filename, 'r', 'utf-8').read()
		# only alphabetical chars and whitespace:
		dirty_text = "".join([char for char in dirty_text
			if char.isalpha() or char.strip()==""])
		# replace v by u, j by i:
		dirty_text = dirty_text.replace("v", "u")
		dirty_text = dirty_text.replace("j", "i")
		words = []
		for w in dirty_text.lower().split():
			w = w.strip()
			if w:
				words.append(w)
		# to new file:
		filename = os.path.splitext(os.path.basename(filename))[0]
		author, title = filename.lower().split("_")
		cnt, start_idx, end_idx = 1, 0, sample_size
		while end_idx <= len(words):
			with codecs.open(data_dir+"/"+clean_dir_name+"/"+
				             author+"_"+title+"_"+str(cnt)+".txt",
				             'wt', 'utf-8') as F:
				F.write(" ".join(words[start_idx:end_idx]))
			cnt+=1
			start_idx += sample_size
			end_idx += sample_size
		print len(dirty_text)



