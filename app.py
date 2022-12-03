from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('desenho.html')


@app.errorhandler(404)
def pageNotFound(error):
    return render_template('notfound.html')


if __name__ == '__main__':
    app.run()
