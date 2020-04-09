import tensorflow as tf
from sklearn.externals import joblib
import jieba
from config.lr_config import LrConfig
from lr_model import LrModel


def pre_data(data, config):
    """分词去停用词"""
    stopwords = list()
    text_list = list()
    with open(config.stopwords_path, 'r', encoding='utf-8') as f:
        for word in f.readlines():
            stopwords.append(word[:-1])
    seg_text = jieba.cut(data)
    text = [word for word in seg_text if word not in stopwords]
    text_list.append(' '.join(text))
    return text_list


def read_categories():
    """读取类别"""
    with open(config.categories_save_path, 'r', encoding='utf-8') as f:
        categories = f.readlines()
    return categories[0].split('|')


def predict_line(data, categories):
    """预测结果"""
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=config.lr_save_path)
    y_pred_cls = session.run(model.y_pred_cls, feed_dict={model.x: data})
    return categories[y_pred_cls[0]]


if __name__ == "__main__":
    data = "三星ST550以全新的拍摄方式超越了以往任何一款数码相机"
    config = LrConfig()
    line = pre_data(data, config)
    tfidf_model = joblib.load(config.tfidf_model_save_path)
    X_test = tfidf_model.transform(line).toarray()
    model = LrModel(config, len(X_test[0]))
    categories = read_categories()
    print(predict_line(X_test, categories))
