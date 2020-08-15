
# coding: utf-8



import tensorflow_datasets as tfds
import tensorflow as tf





file = open("./en_cn/cmn.txt",'r', encoding='UTF-8') 
all_lines = file.readlines()
enlish = []
chines = []
for line in all_lines:
    enlish.append(line.strip().split()[:-11])
    chines.append(line.strip().split()[-11])




enl = []
for i in enlish:
    enl.append(' '.join(i).replace('.','').replace('!','').replace('?',''))
chi = []
for i in chines:
    chi.append(''.join(i).replace('。','').replace('！','').replace('？',''))



enlish =  enl
chines = chi




#tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(( e for e in enlish), target_vocab_size=2**13)


#tokenizer_cn = tfds.features.text.SubwordTextEncoder.build_from_corpus((' '.join(e) for e in chines), target_vocab_size=2**13)






tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file('./en_word')
tokenizer_cn = tfds.features.text.SubwordTextEncoder.load_from_file('./cn_word')







cn = [tokenizer_cn.encode(line)for line in chines]



en= [tokenizer_en.encode(line)for line in enlish]





en_ = []
for i in en :
    en_.append([tokenizer_en.vocab_size]+list(i)+[tokenizer_en.vocab_size+1])
cn_ = []
for i in cn :
    cn_.append([tokenizer_cn.vocab_size]+list(i)+[tokenizer_cn.vocab_size+1])




en_text=tf.keras.preprocessing.sequence.pad_sequences(
    en_, maxlen=40, dtype='int32', padding='post',
    value=0.0)
cn_text=tf.keras.preprocessing.sequence.pad_sequences(
    cn_, maxlen=40, dtype='int32', padding='post',
    value=0.0)









train_dataset= tf.data.Dataset.from_tensor_slices((en_text,cn_text))



train_dataset = train_dataset.shuffle(buffer_size=200).batch(64)




de_batch, en_batch = next(iter(train_dataset))

print(de_batch[56])
print(en_batch[56])








