from config import Config
from xlrd import open_workbook  # xlrd用于读取xld
import  tensorflow as tf
import  numpy as np
def preprocess_file(Config):
    poems = []
    poem = ''
    # 读取文件
    poems_examples = read_and_clean_zh_file(Config.poetry_file)
    for poem in poems_examples:
        if 28 < len(poem) < 33:
            poems.append(poem)
    for p in poems:
        word = seperate_line(p)
        word = word.strip().split(' ')
        for w in word:
            poem += w
        poem = poem +']'

    words = sorted(poem)
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1
    # 去掉低频的字
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])
    print(wordPairs)
    words, _ = zip(*wordPairs)
    print(type(words))
    print(_)
    words += (" ",)
    # word到id的映射
    word2num = dict((c, i) for i, c in enumerate(words))
    num2word = dict((i, c) for i, c in enumerate(words))
    word2numF = lambda x: word2num.get(x, len(words) - 1)
    print(word2numF)


    x_ve = []
    y_ve = []
    for i  in  range(len(poem)-8):
        x = poem[i: i + Config.max_len]
        y = poem[i + Config.max_len]

        if ']' in x or ']' in y:
            i += 1
            continue

        y_vec = np.zeros(
            shape=(1, len(words)),

        )
        y_vec[0, word2numF(y)] = 1.0

        x_vec = np.zeros(
            shape=(1, Config.max_len, len(words)),

        )

        for t, char in enumerate(x):
            x_vec[0, t, word2numF(char)] = 1.0
        x_ve.append(x_vec)
        y_ve.append(y_vec)
    print(type(x_ve))
    print(x_ve[1][0][0])
    x_ve = np.array(x_ve)
    print(type(x_ve))
    y_ve = np.array(y_ve)
    x_ve = tf.reshape(x_ve,[8936,8,804])
    print(type(x_ve))
    y_ve = tf.reshape(y_ve,[8936,804])


    return  x_ve,y_ve





def seperate_line(line):
    return ''.join([word + ' ' for word in line])

def read_and_clean_zh_file(input_file, output_cleaned_file=None):
    # lines = list(open(input_file, "rb").readlines())
    workbook = open_workbook(input_file)  # 打开xls文件\
    sheet = workbook.sheet_by_index(0)
    content = sheet.row_values(1)  # 第4列内容
    lines = content[1:]
    return lines






if __name__ == '__main__':
    import  numpy as np
    x,y= preprocess_file(Config)
