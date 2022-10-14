import main
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/extract_features')
def extract_features():
    return main.main('extract_features')

@app.route('/train_models')
def train_models():
    return main.main('train_models')


if __name__ == '__main__':
    app.run(
                use_reloader=False,
                use_debugger=True,
                # use_evalex=True,
                # reloader_interval=1,
                # reloader_type='auto',
                # threaded=False,
                # processes=1,
                # request_handler=None,
                # static_files=None,
                passthrough_errors=False,
            )
