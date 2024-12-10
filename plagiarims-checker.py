from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from io import StringIO
import re
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
import nltk.stem.snowball
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from Levenshtein import distance
import functools

# load the trained model and count the size of the word vector
model = Word2Vec.load('the_model')
word_model = Word2Vec.load('the_word_model')
word_keys = list(word_model.wv.vocab.keys())
vector_size = model.wv.vector_size

# remove most common headers and footers of the scientific papers
header_footer = ['international', 'journal', 'of', 'computer', 'science', 'trend', 's', 'and', 'technology', 'ijcst', 'volume', 'issue', '2', 'marapr', '2015', 'issn:', '23478578', 'wwwijcstjournalorg', 'page', '83', 'i', 'ii', 'iii', 'iv', 'v', '11', 'fig', 'fig', '12', '21', '22', '23', '24', '25', '26', '31', '32', '33', '34', '35', '36', '84', '85', '86', '87', 'scientific', 'research', 'publi', 'cations', 'june', '2013', 'issn', '2250', '', '3153', 'wwwijsrporg', 'jayashri', 'khairnar', '*', 'mayura', 'kinikar', '**', '*', 'department', 'mit', 'academy',  '**', 'd', 'epartment', 'university', '', 'mit', 'academy', 'pune', '&', 'engineering', 'eissn:', '23948299', 'special', 'issue:', 'technoxtreme', '16', 'pissn:', '23948280', 'ijrise|', 'wwwijriseorg|editor@ijriseorg488495', 'profs', 'minzalkar1', 'jai', 'sharma2', '1sminzalkar', 'cse', 'jdiet', 'yavatmal', 'sachininzlkar@gmailcom', '2jai', 'sharma', 'jdiet', 'yavatmal', 'sharmajr49@gmailcom','27th', 'annual', 'acm', 'sigir', 'conference', 'researchand', 'development', 'information', 'retrieval', 'computational', 'linguistics26', 'proceedings']

TOP_WORD_NUMBER = 10

TOP_SIMILAR_WORDS = 3

app = Flask(__name__)

UPLOAD_DIR = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

if not os.path.isdir(UPLOAD_DIR):
    os.mkdir(UPLOAD_DIR)

# Allowed file types for file upload (PDF only)
ALLOWED_EXTENSIONS = set(['pdf'])


def allowed_file(filename):
    """Does filename have the right extension?"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# split all text to words and perform stemming!
# stemming will significantly reduce lexicon size
def textToWords(text):
    stemmer = nltk.stem.snowball.EnglishStemmer()
    return [stemmer.stem(word).lower() for word in re.split('\W+', text) if 2 < len(word) < 25]


def sortSecond(val):
    return val[1]


# get top N most words from file with highest tf-idf score
def getTopNWords(feature_names, tf_idf_array, n):
    top_n_pairs = []
    for i in range(tf_idf_array.shape[0]):
        arr = list(zip(feature_names, tf_idf_array.toarray()[i, :]))
        arr.sort(key=sortSecond, reverse=True)
        top_n_pairs.append(arr[:10])
    return top_n_pairs


def pdf_to_text(pdfname):

    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    fp = open(os.path.join(app.config['UPLOAD_FOLDER'], pdfname), 'rb')
    parser = PDFParser(fp)
    doc = PDFDocument(parser)
    try:
        title = doc.info[0]['Title']
        if not isinstance(title, str):
            title = title.decode('ascii')
    except:
        title = None
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()

    # Cleanup
    device.close()
    sio.close()

    return text, title


def parse_pdf(filename):
    # provide regular expressions for the removing punctuation
    p = re.compile('\s{2,}')
    p2 = re.compile('[^\w\s]')
    p3 = re.compile('\d')

    # provide list of English stop words
    nltk_stopwords = stopwords.words('english')

    # read the pdf
    # text to lowercase
    # get rid of punctuation
    text, title = pdf_to_text(filename)
    text = text.lower()
    text = p3.sub('', p2.sub('', p.sub(' ', text.replace('\n', ''))))

    # get rid of stopwords
    filtered_text = word_tokenize(text)
    filtered_text = [w for w in filtered_text if w not in nltk_stopwords]
    text = " ".join(filtered_text)

    # get rid of most common headers and footers of the pdf page
    filtered_text = []
    for word in text.split():
        if word not in header_footer and len(word) < 20:
            filtered_text.append(word)
    text = " ".join(filtered_text)

    # perform stemming
    stemmed_text = textToWords(text)

    return title, text, stemmed_text


def count_similarity(stemmed_texts):
    summary_vectors = []

    # for each text summary_vector is counted as sum of the all included stems vectors in it
    for stemmed_text in stemmed_texts:

        summary_vector = np.array([0 for _ in range(vector_size)], dtype='float32')
        for stemmed_word in stemmed_text:
            try:
                summary_vector += model.wv.get_vector(stemmed_word)
            except KeyError:
                pass
        summary_vectors.append(summary_vector / len(stemmed_text))

    # for each pair of texts counted cosine similarity, the closer to 1, the more similar texts are
    cosine_similarity = [[round(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)), 2) for a in summary_vectors] for b
                         in summary_vectors]
    return cosine_similarity


# find 3 most similar words for each word in a list of top words
def find_most_similar_words(top_tf_idf_words, texts, n):
    all_triples = []
    for idx, top_words in enumerate(top_tf_idf_words):
        triples = {}
        for word, score in top_words:
            word_ = None
            if word not in word_keys:
                leven_distance = list(map(functools.partial(distance, word), word_model.wv.vocab.keys()))
                word_ = word_keys[leven_distance.index(min(leven_distance))]
            similar_words = []
            for jdx, text in enumerate(texts):
                if idx != jdx:
                    if not word_:
                        for wrd in text.split():
                            try:
                                similar_words += [(wrd, round(word_model.wv.similarity(word, wrd), 2))]
                            except KeyError:
                                pass
                    else:
                        for wrd in text.split():
                            try:
                                similar_words += [(wrd, round(word_model.wv.similarity(word_, wrd), 2))]
                            except KeyError:
                                pass
            similar_words.sort(key=sortSecond, reverse=True)
            triples[word] = similar_words[:n]
        all_triples.append(triples)
    return all_triples


def print_result(titles, top_tf_idf_words, similarity, all_triples):
    result = '''<html>
<head>
<style>
table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
}
th, td {
    padding: 5px;
    text-align: left;    
}
</style>
</head>
<body>
<p>
<table>
<tr>'''

    for title in titles:
        result += "<th colspan=\"4\">" + title + "</th>"

    result += '</tr>\n'

    result += '<tr>' + len(titles) * '<th>Word</th><th>Tf-idf Score</th><th>Similar Words from Other Texts</th><th>Similarity</th>' + '</tr>'

    for num in range(TOP_WORD_NUMBER):
        for idx in range(TOP_SIMILAR_WORDS):
            result += '<tr>'
            for text_num in range(len(top_tf_idf_words)):
                the_word = top_tf_idf_words[text_num][num][0]
                top_n = all_triples[text_num][the_word]
                if idx == 0:
                    result += "<td rowspan = \"" + str(TOP_SIMILAR_WORDS) + "\">" + str(the_word) + \
                              "</td><td rowspan = \"" + str(TOP_SIMILAR_WORDS) \
                              + "\">" + str(round(top_tf_idf_words[text_num][num][1], 2)) + "</td>"
                result += "<td>" + str(top_n[idx][0]) + "</td><td>" + str(top_n[idx][1]) + "</td>"
            result += '</tr>\n'

    result += '<tr>' + len(titles) * '<th colspan = \"3\">Text</th><th>Similarity</th>' + '</tr>'

    for i in range(len(similarity) - 1):
        result += '<tr>'
        for j in range(len(similarity)):
            if i < j:
                result += '<td colspan = \"3\">' + str(titles[i]) + '</td><td>' + str(similarity[i][j]) + '</td>'
            else:
                result += '<td colspan = \"3\">' + str(titles[i + 1]) + '</td><td>' + str(similarity[i + 1][j]) + '</td>'
        result += '</tr>\n'

    result += '''</table>
    </p>
    </body>
    </html>'''
    return result


def work_with_pdfs(filenames):
    titles = []
    texts = []
    stemmed_texts = []

    # get title, text without stopwords and stemmed text for each pdf file
    for fname in filenames:
        title, text, stemmed_text = parse_pdf(fname)
        if not title:
            title = fname.replace('_', ' ').rsplit('.', 1)[0]
        titles.append(title)
        texts.append(text)
        stemmed_texts.append(stemmed_text)

    # count tf-idf vector for all texts using text without stopwords
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf = tf_idf_vectorizer.fit_transform(texts)
    top_tf_idf_words = getTopNWords(tf_idf_vectorizer.get_feature_names(), tf_idf, TOP_WORD_NUMBER)

    # count similarities of the texts
    similarity = count_similarity(stemmed_texts)

    all_triples = find_most_similar_words(top_tf_idf_words, texts, TOP_SIMILAR_WORDS)

    result = print_result(titles, top_tf_idf_words, similarity, all_triples)

    return result


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/results', methods=['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        files = []
        filenames = []
        for i in range(1, 6):
            try:
                files.append(request.files['file'+str(i)])
            except:
                pass
        if len(files) > 1:
            for f in files:
                if f and allowed_file(f.filename):
                    filename = secure_filename(f.filename)
                    filenames.append(filename)
                    f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                                        filename))
        else:
            return "At least two .pdf files should be chosen!"

        return work_with_pdfs(filenames)


if __name__ == '__main__':
    app.run(debug=True)
