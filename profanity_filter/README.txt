Installation instruction for text profanity moderation service.

Install Anaconda : https://docs.anaconda.com/anaconda/install/

Create new environment with python 3.6 version and use requirements.txt to install all required packages.
Run following commands:
"conda create -n textProfanityEnv python=3.6"
"conda activate textProfanityEnv"
"pip install -r requirements.txt"
"python -m spacy download en"

create kafka topics if not already present
Topic names:
    1. flagged_ai
    2. moderated_ai

modify config.json file with kafka endpoint address "kafka_bootstrap_servers" or add as environment variable.


Finally run the following to start the service:
"gunicorn --workers=4 --bind 0.0.0.0:4001 server:app"

Troubleshooting:

1. If you get an error similar to : OSError: [E050] Can't find model 'en'. Run following command the download the model:
"python -m spacy download en"
(windows users may need to run above command in administrator mode for proper linking)

2. If you get any error related to shape of model, delete the pre-trained models from "model" directory. After this step, code will retrain the model and this step may take some time for the first time.

