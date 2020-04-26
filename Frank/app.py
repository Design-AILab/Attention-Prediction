import flask
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = flask.Flask(__name__, template_folder='templates')

@app.route('/',methods=['GET', 'POST'])

def home():
    if flask.request.method == 'POST':
        file = flask.request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return flask.redirect(flask.url_for('prediction', filename=filename))
    return(flask.render_template('home.html'))


@app.route('/prediction/<filename>')
def prediction(filename):

    image = plt.imread(os.path.join('uploads', filename))

    ### code for prediction
    
    return flask.render_template('predict.html')


app.run()
