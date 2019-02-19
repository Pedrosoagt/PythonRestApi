### Designing RESTful API with Python-Flask and MongoDB

##### Create your local environment

```bash
conda create -n restfulapi python=3.7 anaconda # Create the environment
source activate restfulapi # Activate the environment
```

##### Install dependencies

```python
pip install -r requirements.txt
```

##### Start MongoDB Server

If you're using MacOS, you could use `brew` to start the server.

```bash
brew services start mongodb
```

#### Config the application

Change the `DBNAME` in the config file according to the database name you are using.

##### Start the application

```bash
python run-app.py
```

Once the application is started, go to [localhost](http://localhost:5000/)
on Postman and explore the APIs.
