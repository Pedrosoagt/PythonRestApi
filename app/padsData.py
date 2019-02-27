"""This module will serve the api request."""

from config import client
from app import app
from bson.json_util import dumps
from flask import request, jsonify
import json
import ast
import imp

# Import the helpers module
helper_module = imp.load_source('*', './app/helpers.py')

# Select the database
db = client['DEV']
# Select the collection
collection = db['SPIA_new']

@app.route("/pads", methods=['GET'])
def fetch_pads():
    """
       Function to fetch the pads.
    """
    try:
        db = client['DEV']
        collection = db['SPIA_new']
        # Call the function to get the query params
        query_params = helper_module.parse_query_params(request.query_string)
        # Check if dictionary is not empty
        if query_params:

            # Try to convert the value to int
            query = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in query_params.items()}

            # Fetch all the record(s)
            records_fetched = collection.find(query)

            # Check if the records are found
            if records_fetched.count() > 0:
                # Prepare the response
                return dumps(records_fetched)
            else:
                # No records are found
                return "Records not found", 404

        # If dictionary is empty
        else:
            # Return all the records as query string parameters are not available
            if collection.find().count > 0:
                # Prepare response if the pads are found
                return dumps(collection.find())
            else:
                # Return empty array if no pads are found
                return jsonify([])
    except:
        # Error while trying to fetch the resource
        # Add message for debugging purpose
        return "", 500

# @app.route("/pads/<xml_path>", methods=['GET'])
# def fetch_pads_by_xml_path(xml_path):
#     """
#         Function to fetch the pads from a given xml
#     """
#     collection = [collection for col in collection if collection['name'] == xml_path]
#     if collection.find().count > 0:
#         return dumps(collection.find())
#     else:
#         abort(404)

# @app.route("/pads/<sort>", methods=['GET'])
# def fetch_sorted_pads(sort):
#     """
#         Function to fetch the sorted pads
#     """
#     padsList =  fetch_pads()
#     return padsList.sort(reverse = True)

@app.route("/pads/ze", methods=['GET'])
def fetch_top_five_pads():
    try:
        return jsonify(top_five)
    except:
        return "", 404

@app.errorhandler(404)
def page_not_found(e):
    """Send message to the user with notFound 404 status."""
    # Message to the user
    message = {
        "err":
            {
                "msg": "This route is currently not supported."
            }
    }
    # Making the message looks good
    resp = jsonify(message)
    # Sending OK response
    resp.status_code = 404
    # Returning the object
    return resp
